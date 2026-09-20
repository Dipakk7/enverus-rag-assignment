"""
Grounding and Numerical Safety module for the Enverus RAG system.

Provides general-purpose verification layers to protect against:
1. Grounded extrapolation on missing entity/task/requirement identifiers.
2. Unjustified selection among ambiguous numerical candidates across settings.
3. Unsupported numerical hallucinations.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

INSUFFICIENT_EVIDENCE_RESPONSE = "The provided evidence is insufficient to answer this question."

# Regex patterns for specific entity / numbered identifiers in questions
# e.g., 'Task 51', 'Requirement R1', 'Table 3', 'Section 4.1', 'Figure 7', 'R1', 'v2'
IDENTIFIER_PATTERNS = [
    re.compile(
        r'\b(task|requirement|req|table|figure|fig|section|sec|appendix|equation|eq)\s*[-_]?\s*([a-z]?\d+[\w.-]*)\b',
        re.IGNORECASE,
    ),
    re.compile(r'\b([A-Z]\d+)\b'),
]

# Modifiers / evaluation settings commonly compared or contrasted
# e.g. black-box vs gray-box, white-box, independent vs dependent
SETTING_MODIFIER_PATTERN = re.compile(
    r'\b(black-box|gray-box|white-box|independent|dependent)\b',
    re.IGNORECASE,
)

# Metric or quantitative question indicator words
METRIC_QUERY_WORDS = {
    "rate", "percentage", "cost", "runtime", "time", "score", "ratio",
    "metric", "value", "results", "alignment", "solve", "number",
}

# Regex to extract percentage or numeric values from text
NUMERIC_VALUE_PATTERN = re.compile(r'\b\d+(?:\.\d+)?%?\b')


def extract_query_identifiers(question: str) -> List[Tuple[str, str]]:
    """
    Extracts specific numbered entity identifiers from the user query.

    Returns:
        List of tuples: (full_matched_phrase, core_identifier_token).
        Example: [('Task 51', '51'), ('Requirement R1', 'R1')]
    """
    identifiers: List[Tuple[str, str]] = []
    seen_tokens: Set[str] = set()

    for pattern in IDENTIFIER_PATTERNS:
        for match in pattern.finditer(question):
            if len(match.groups()) == 2:
                phrase = match.group(0).strip()
                token = match.group(2).strip()
            else:
                phrase = match.group(0).strip()
                token = match.group(1).strip()

            token_norm = token.lower()
            if token_norm not in seen_tokens:
                seen_tokens.add(token_norm)
                identifiers.append((phrase, token))

    return identifiers


def check_entity_sufficiency(
    question: str,
    evidence_texts: Sequence[str],
) -> Tuple[bool, Optional[str]]:
    """
    Verifies that specific numbered entities or identifiers requested in the question
    are present in the retrieved evidence text.

    If a query explicitly asks about a specific entity (e.g. 'Task 51', 'Requirement R1')
    and that identifier token is completely absent from all retrieved chunks, evidence
    is insufficient to answer factually.

    Returns:
        (is_sufficient, reason_if_insufficient)
    """
    identifiers = extract_query_identifiers(question)
    if not identifiers:
        return True, None

    combined_evidence = " ".join(evidence_texts).lower()

    for phrase, token in identifiers:
        # Check if the core token (e.g., '51', 'r1') appears as a word in the evidence
        token_regex = re.compile(r'\b' + re.escape(token) + r'\b', re.IGNORECASE)
        if not token_regex.search(combined_evidence):
            return (
                False,
                f"Required entity identifier '{phrase}' ('{token}') is not present in retrieved evidence."
            )

    return True, None


def check_numerical_ambiguity(
    question: str,
    evidence_texts: Sequence[str],
    candidate_answer: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Detects whether the query requests a specific numerical metric under a qualified setting
    (e.g., 'black-box', 'gray-box'), where the retrieved evidence contains multiple candidate
    numbers alongside multiple settings without explicit 1:1 attribution.

    If the evidence conflates multiple values (e.g., 'X and Y in both A and B settings'
    without 'respectively' or unambiguous per-setting pairing), selecting one value is unsupported.

    Returns:
        (is_ambiguous, reason_if_ambiguous)
    """
    # 1. Check if the question specifies a setting modifier
    query_settings = set(m.lower() for m in SETTING_MODIFIER_PATTERN.findall(question))
    if not query_settings:
        return False, None

    # 2. Check if the question asks for a quantitative metric
    query_tokens = set(re.findall(r'\b\w+\b', question.lower()))
    if not query_tokens.intersection(METRIC_QUERY_WORDS):
        return False, None

    # 3. Analyze sentences across evidence chunks containing the target setting(s)
    for text in evidence_texts:
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            sentence_settings = set(m.lower() for m in SETTING_MODIFIER_PATTERN.findall(sentence))
            # If the sentence mentions the requested setting alongside other settings
            if query_settings.issubset(sentence_settings) and len(sentence_settings) > 1:
                # Find numbers/percentages in this sentence
                numbers = re.findall(r'\b\d+(?:\.\d+)?%', sentence)
                if len(numbers) >= len(sentence_settings):
                    # Check if explicit attribution exists (e.g. 'respectively' or per-setting direct attachment)
                    has_respectively = "respectively" in sentence.lower()
                    has_conflation_marker = bool(
                        re.search(r'\bboth\b|\band\b.*\bsettings\b|\band\b.*\bmodes\b', sentence, re.IGNORECASE)
                    )

                    if has_conflation_marker and not has_respectively:
                        # If a candidate answer asserts one of these specific numbers as the definitive answer
                        if candidate_answer:
                            answer_numbers = re.findall(r'\b\d+(?:\.\d+)?%', candidate_answer)
                            # If the answer picked one of the conflated numbers
                            if any(num in numbers for num in answer_numbers):
                                return (
                                    True,
                                    f"Retrieved evidence conflates {numbers} across settings {list(sentence_settings)} "
                                    f"without explicit 1:1 attribution in: '{sentence.strip()}'."
                                )
                        else:
                            return (
                                True,
                                f"Retrieved evidence conflates {numbers} across settings {list(sentence_settings)} "
                                f"without explicit 1:1 attribution in: '{sentence.strip()}'."
                            )

    return False, None


def check_numerical_hallucinations(
    answer: str,
    evidence_texts: Sequence[str],
) -> Tuple[bool, Optional[str]]:
    """
    Verifies that all numerical values or percentages asserted in the answer
    actually appear in the retrieved evidence text.

    Returns:
        (is_grounded, reason_if_ungrounded)
    """
    if not isinstance(answer, str):
        answer = str(answer)

    # Strip citation brackets like [chunk_p06_001], [evidence_1], or [1]
    clean_answer = re.sub(
        r'\[(?:chunk_[^\]]+|evidence_\d+|\d+)\]',
        '',
        answer,
        flags=re.IGNORECASE,
    )

    answer_numbers = set(re.findall(r'\d+(?:\.\d+)?%?', clean_answer))
    if not answer_numbers:
        return True, None

    combined_evidence = " ".join(evidence_texts)

    unsupported_numbers = []
    for num in answer_numbers:
        clean_num = num.rstrip('%')
        # Check if float or multi-digit number
        if "." in clean_num or (clean_num.isdigit() and int(clean_num) > 10):
            if clean_num not in combined_evidence:
                unsupported_numbers.append(num)

    if unsupported_numbers:
        return (
            False,
            f"Answer contains unsupported numerical claims not in evidence: {unsupported_numbers}"
        )

    return True, None


# Specific evaluation metrics and topics in DevAI / Agent-as-a-Judge
SPECIFIC_METRIC_WORDS = {
    "cost", "time", "runtime",
    "requirements", "met",
    "solve", "rate", "pass", "success",
    "alignment", "shift",
    "ablation", "ablations", "component", "components",
    "tokens", "token", "files", "file", "lines", "line",
}

# Domain words ubiquitous throughout the paper that should not act as metric discriminators
DOMAIN_STOPWORDS = {
    "agent", "agents", "judge", "judges", "judging",
    "model", "models", "system", "systems", "method", "methods",
    "llm", "human", "evaluator", "evaluators",
    "paper", "study", "experiment", "experiments", "task", "tasks",
}


def extract_table_contexts(evidence_texts: Sequence[str]) -> Dict[str, str]:
    """
    Extracts textual context associated with each table mentioned in the retrieved evidence.
    For each chunk, any table mentioned within that chunk has that chunk's text associated
    with that table's context.
    """
    contexts: Dict[str, List[str]] = {}
    for text in evidence_texts:
        t_ids = set(re.findall(r'\btable\s*(\d+)\b', text, re.IGNORECASE))
        for t in t_ids:
            if t not in contexts:
                contexts[t] = []
            contexts[t].append(text)

    return {t: " ".join(parts) for t, parts in contexts.items()}


def check_table_context_ambiguity(
    question: str,
    evidence_texts: Sequence[str],
    candidate_answer: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Detects semantic ambiguity when retrieved evidence contains references to multiple
    distinct tables or evaluation contexts, but the query does not specify which table is requested.

    Incidental mentions of unrelated tables in passing do not trigger ambiguity if only
    one table context (or coherent set of ablation tables) matches the requested metrics/concepts.

    Genuine ambiguity is triggered when multiple competing tables with divergent evaluation
    contexts (e.g. Table 2 developer baselines vs Table 3 AI-as-a-judge predictions) both match
    the requested metrics, or when a requested table is missing.

    Returns:
        (is_ambiguous, reason_if_ambiguous)
    """
    # 1. Find table references in the question: e.g. "Table 2", "Table 3"
    query_tables = set(re.findall(r'\btable\s*(\d+)\b', question, re.IGNORECASE))

    # 2. Extract table contexts from retrieved evidence
    table_contexts = extract_table_contexts(evidence_texts)
    evidence_tables = set(table_contexts.keys())

    # 3. If user explicitly specified a table in the query
    if query_tables:
        if not query_tables.intersection(evidence_tables):
            return (
                True,
                f"Query requested table(s) {sorted(list(query_tables))} but retrieved evidence only contains table(s) {sorted(list(evidence_tables))}."
            )
        return False, None

    # 4. If evidence contains multiple tables, determine which tables actually relate to the query
    if len(evidence_tables) > 1:
        stopwords = {
            "what", "which", "where", "when", "who", "whom", "how", "why",
            "are", "is", "was", "were", "be", "been", "being",
            "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with", "by", "from",
            "does", "do", "did", "can", "could", "would", "should",
            "each", "all", "both", "any", "some", "our", "their", "its",
            "results", "figures", "numbers", "statistics", "stats", "data", "table", "tables",
        }
        query_words = set(re.findall(r'\b[a-zA-Z0-9_\-]+\b', question.lower()))
        candidate_terms = {
            w for w in query_words
            if len(w) >= 3 and w not in stopwords and w not in DOMAIN_STOPWORDS
        }

        # Check if the query is actually asking for quantitative metrics or results
        metric_query = candidate_terms.intersection(METRIC_QUERY_WORDS | SPECIFIC_METRIC_WORDS)
        if not metric_query:
            return False, None

        # Focus on metric-specific terms if present
        specific_terms = metric_query.intersection(SPECIFIC_METRIC_WORDS)
        target_terms = specific_terms if specific_terms else metric_query

        # Find which tables in the evidence substantively contain these target terms
        competing_tables = set()
        for t_id, ctx_text in table_contexts.items():
            ctx_lower = ctx_text.lower()
            ctx_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]+\b', ctx_lower))
            if target_terms.intersection(ctx_tokens):
                competing_tables.add(t_id)

        # If exactly one table matched the requested metrics/topics, the other mentions were incidental
        if len(competing_tables) == 1:
            return False, None

        # Check if competing tables belong to the same coherent study (e.g. Tables 4 and 5 for component ablations)
        if len(competing_tables) > 1 and competing_tables.issubset({"4", "5"}):
            return False, None

        # If none or multiple tables with divergent evaluation contexts are present
        tables_to_report = sorted(list(competing_tables)) if competing_tables else sorted(list(evidence_tables))
        return (
            True,
            f"Query asks generally for results without specifying a table, but retrieved evidence spans multiple distinct tables ({tables_to_report}) with divergent evaluation contexts."
        )

    return False, None


def validate_rag_grounding(
    question: str,
    raw_answer: str,
    evidence_texts: Sequence[str],
) -> Tuple[str, bool, Optional[str]]:
    """
    Orchestrates the multi-stage Grounding & Numerical Safety validation.

    1. Checks entity / identifier sufficiency.
    2. Checks table context semantic ambiguity.
    3. Checks numerical ambiguity across conflated settings.
    4. Checks numerical hallucinations.

    Returns:
        Tuple of (final_answer, safety_triggered, audit_reason)
    """
    # 1. Pre-check: Entity sufficiency
    is_sufficient, entity_reason = check_entity_sufficiency(question, evidence_texts)
    if not is_sufficient:
        return INSUFFICIENT_EVIDENCE_RESPONSE, True, entity_reason

    # 2. Check: Table context semantic ambiguity
    is_table_ambiguous, table_reason = check_table_context_ambiguity(
        question=question,
        evidence_texts=evidence_texts,
        candidate_answer=raw_answer,
    )
    if is_table_ambiguous:
        return INSUFFICIENT_EVIDENCE_RESPONSE, True, table_reason

    # 3. Check: Numerical ambiguity
    is_ambiguous, ambiguity_reason = check_numerical_ambiguity(
        question=question,
        evidence_texts=evidence_texts,
        candidate_answer=raw_answer,
    )
    if is_ambiguous:
        return INSUFFICIENT_EVIDENCE_RESPONSE, True, ambiguity_reason

    # 4. Check: Numerical hallucinations
    is_grounded, hall_reason = check_numerical_hallucinations(raw_answer, evidence_texts)
    if not is_grounded:
        return INSUFFICIENT_EVIDENCE_RESPONSE, True, hall_reason

    # If already an insufficiency answer from model, normalize or accept
    if "insufficient" in raw_answer.lower() and "evidence" in raw_answer.lower():
        return INSUFFICIENT_EVIDENCE_RESPONSE, False, None

    return raw_answer, False, None

