"""
Document chunking and metadata preservation module.

Converts page-level records from src.ingest into semantically coherent,
retrieval-ready chunks while preserving page numbers and section hierarchy.
"""

from dataclasses import asdict, dataclass
import re
from typing import List, Optional

from src.ingest import PageRecord


@dataclass(frozen=True)
class Chunk:
    """
    Structured representation of a document chunk.

    Attributes:
        chunk_id: Deterministic identifier formatted as 'chunk_p{page}_{index}'.
        page_number: 1-indexed page number of origin.
        text: Extracted, semantically coherent text passage.
        section: Identified section, subsection, or appendix heading (or None).
    """
    chunk_id: str
    page_number: int
    text: str
    section: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _is_valid_heading_title(title: str) -> bool:
    """Verifies that a candidate heading string is a genuine section title."""
    title = title.strip()
    if len(title) > 70 or len(title) < 3:
        return False
    if title.endswith((".", ",", ";", ":", "!", "?")):
        return False
    bad_tokens = {"USD", "http", "https", "Table", "Figure", "Eq", "Equation"}
    if any(t in title for t in bad_tokens):
        return False
    return True


def identify_heading(lines: List[str], idx: int) -> Optional[str]:
    """
    Detects section, subsection, or appendix headings at a line index.

    Returns:
        The formatted heading string if detected, otherwise None.
    """
    line = lines[idx].strip()

    # 1. Numbered section on single line (e.g., '1 Introduction', '2.1 Motivation')
    m_single = re.match(r"^([1-9]\d{0,1}(?:\.\d{1,2}){0,2})\s+([A-Z].+)$", line)
    if m_single:
        sec_num, title = m_single.group(1), m_single.group(2).strip()
        if _is_valid_heading_title(title):
            return f"{sec_num} {title}"

    # 2. Numbered section across two lines ('2' followed by 'DevAI: A Dataset...')
    m_num = re.match(r"^([1-9]\d{0,1}(?:\.\d{1,2}){0,2})$", line)
    if m_num and idx + 1 < len(lines):
        next_line = lines[idx + 1].strip()
        if re.match(r"^[A-Z]", next_line) and _is_valid_heading_title(next_line):
            return f"{m_num.group(1)} {next_line}"

    # 3. Appendix letter across two lines ('A' followed by 'Outline of this Paper')
    if re.match(r"^[A-M]$", line) and idx + 1 < len(lines):
        next_line = lines[idx + 1].strip()
        if re.match(r"^[A-Z]", next_line) and _is_valid_heading_title(next_line):
            return f"Appendix {line}: {next_line}"

    # 4. Appendix with explicit prefix
    if re.match(r"^Appendix\s+[A-M](?::\s*.+)?$", line) and _is_valid_heading_title(line):
        return line

    # 5. Standard top-level non-numbered sections
    if line in {"References", "Abstract"}:
        return line

    return None


def _split_text_into_blocks(text: str) -> List[str]:
    """
    Groups raw page lines into natural semantic blocks (paragraphs, captions, lists).
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return []

    blocks: List[str] = []
    current_lines = [lines[0]]

    for i in range(1, len(lines)):
        prev_line = lines[i - 1]
        line = lines[i]
        is_break = False

        # Table and Figure captions start new blocks
        if re.match(r"^(?:Table|Figure)\s+\d+", line):
            is_break = True
        # Bullet items and list markers start new blocks
        elif line.startswith(("•", "\ufffd", "- ", "* ", "(1)", "(2)", "(3)")):
            is_break = True
        # Section headings start new blocks
        elif identify_heading(lines, i) is not None:
            is_break = True
        # Sentence termination followed by capitalized start
        elif prev_line.endswith((".", "!", "?", ":")) and re.match(r"^[A-Z]", line):
            is_break = True

        if is_break:
            blocks.append("\n".join(current_lines))
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        blocks.append("\n".join(current_lines))

    return blocks


def _split_by_sentences(text: str, max_size: int) -> List[str]:
    """
    Splits text recursively by sentence boundaries, then words if required.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    cur: List[str] = []

    for s in sentences:
        s_len = len(s)
        cur_len = sum(len(x) + 1 for x in cur)
        if cur and cur_len + s_len > max_size:
            chunks.append(" ".join(cur))
            cur = [s]
        else:
            cur.append(s)

    if cur:
        chunks.append(" ".join(cur))

    # If any individual sentence exceeds max_size, split by words
    final_chunks: List[str] = []
    for c in chunks:
        if len(c) <= max_size:
            final_chunks.append(c)
        else:
            words = c.split()
            w_cur: List[str] = []
            for w in words:
                if w_cur and sum(len(x) + 1 for x in w_cur) + len(w) > max_size:
                    final_chunks.append(" ".join(w_cur))
                    w_cur = [w]
                else:
                    w_cur.append(w)
            if w_cur:
                final_chunks.append(" ".join(w_cur))

    return final_chunks


def _recursive_split_block(block: str, max_size: int) -> List[str]:
    """
    Splits a block recursively if it exceeds the maximum size limit.
    """
    if len(block) <= max_size:
        return [block]

    lines = block.split("\n")
    if len(lines) > 1:
        sub_blocks: List[str] = []
        cur: List[str] = []
        for line in lines:
            line_len = len(line)
            cur_len = sum(len(x) + 1 for x in cur)
            if cur and cur_len + line_len > max_size:
                sub_blocks.append("\n".join(cur))
                cur = [line]
            else:
                cur.append(line)
        if cur:
            sub_blocks.append("\n".join(cur))

        results: List[str] = []
        for sb in sub_blocks:
            if len(sb) <= max_size:
                results.append(sb)
            else:
                results.extend(_split_by_sentences(sb, max_size))
        return results
    else:
        return _split_by_sentences(block, max_size)


def chunk_page_records(
    records: List[PageRecord],
    target_chunk_size: int = 1000,
    max_chunk_size: int = 1200,
    overlap_size: int = 150,
    min_chunk_size: int = 200,
) -> List[Chunk]:
    """
    Converts page-level records into structured, retrieval-ready chunks.

    Args:
        records: List of PageRecord objects from src.ingest.
        target_chunk_size: Preferred chunk length in characters (~200 words).
        max_chunk_size: Primary block ceiling for chunk length to fit embedding windows.
        overlap_size: Context overlap carried forward across chunk splits.
        min_chunk_size: Minimum threshold before merging trailing fragments.

    Returns:
        List of Chunk objects preserving chunk_id, page_number, text, and section.
    """
    chunks: List[Chunk] = []
    current_section: Optional[str] = None
    consolidation_ceiling = int(max_chunk_size * 1.35)  # 1,620 chars max for consolidated fragments

    for record in records:
        blocks = _split_text_into_blocks(record.text)
        current_chunk_blocks: List[str] = []
        chunk_section: Optional[str] = current_section
        page_chunks: List[Chunk] = []
        chunk_idx = 1
        pending_table_prefix: Optional[str] = None

        for block in blocks:
            b_lines = [l.strip() for l in block.splitlines() if l.strip()]

            # Detect if this block introduces a new section heading
            new_heading: Optional[str] = None
            for bi in range(len(b_lines)):
                heading = identify_heading(b_lines, bi)
                if heading:
                    new_heading = heading
                    break

            # If a new section starts and we already have accumulated text from prior section
            if new_heading and new_heading != current_section:
                cur_len = sum(len(x) + 1 for x in current_chunk_blocks)
                if cur_len >= min_chunk_size:
                    chunk_text = "\n".join(current_chunk_blocks).strip()
                    if chunk_text:
                        chunk_id = f"chunk_p{record.page_number:02d}_{chunk_idx:03d}"
                        page_chunks.append(
                            Chunk(
                                chunk_id=chunk_id,
                                page_number=record.page_number,
                                text=chunk_text,
                                section=chunk_section,
                            )
                        )
                        chunk_idx += 1
                        current_chunk_blocks = []

                current_section = new_heading
                chunk_section = new_heading
                pending_table_prefix = None

            # Detect table caption to supply continuation context if the table splits
            tbl_match = re.match(r"^(Table\s+\d+)\s+([^.\n]+)", block)
            if tbl_match:
                table_prefix_to_set = f"{tbl_match.group(1)} (continued): {tbl_match.group(2).strip()}\n"
            else:
                table_prefix_to_set = None

            # Recursively split block if it exceeds maximum chunk size
            sub_blocks = _recursive_split_block(block, max_size=max_chunk_size)

            for sb in sub_blocks:
                cur_len = sum(len(x) + 1 for x in current_chunk_blocks)
                if current_chunk_blocks and (cur_len + len(sb) > max_chunk_size):
                    chunk_text = "\n".join(current_chunk_blocks).strip()
                    if chunk_text:
                        chunk_id = f"chunk_p{record.page_number:02d}_{chunk_idx:03d}"
                        page_chunks.append(
                            Chunk(
                                chunk_id=chunk_id,
                                page_number=record.page_number,
                                text=chunk_text,
                                section=chunk_section,
                            )
                        )
                        chunk_idx += 1

                    # Compute overlap from trailing blocks
                    overlap_blocks: List[str] = []
                    overlap_len = 0
                    for b in reversed(current_chunk_blocks):
                        if overlap_len + len(b) <= overlap_size:
                            overlap_blocks.insert(0, b)
                            overlap_len += len(b)
                        else:
                            break

                    # If this sub-block now starts a new chunk separated from table caption
                    if pending_table_prefix and "Table " not in sb and not any("Table " in b for b in overlap_blocks):
                        sb = pending_table_prefix + sb

                    current_chunk_blocks = overlap_blocks + [sb]
                    chunk_section = current_section
                else:
                    # If pending table prefix is active and current buffer does not contain table
                    if pending_table_prefix and "Table " not in sb and not any("Table " in b for b in current_chunk_blocks):
                        sb = pending_table_prefix + sb

                    current_chunk_blocks.append(sb)
                    if chunk_section is None:
                        chunk_section = current_section

            # Store or clear pending table prefix
            if table_prefix_to_set:
                pending_table_prefix = table_prefix_to_set
            else:
                pending_table_prefix = None

        # Flush remaining blocks on this page
        if current_chunk_blocks:
            chunk_text = "\n".join(current_chunk_blocks).strip()
            if chunk_text:
                # Merge small trailing fragments if the page already has prior chunks
                if (
                    page_chunks
                    and len(chunk_text) < min_chunk_size
                    and (len(page_chunks[-1].text) + len(chunk_text) < consolidation_ceiling)
                ):
                    prev = page_chunks[-1]
                    page_chunks[-1] = Chunk(
                        chunk_id=prev.chunk_id,
                        page_number=prev.page_number,
                        text=prev.text + "\n" + chunk_text,
                        section=prev.section,
                    )
                else:
                    chunk_id = f"chunk_p{record.page_number:02d}_{chunk_idx:03d}"
                    page_chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            page_number=record.page_number,
                            text=chunk_text,
                            section=chunk_section or current_section,
                        )
                    )
                    chunk_idx += 1

        chunks.extend(page_chunks)

    return chunks
