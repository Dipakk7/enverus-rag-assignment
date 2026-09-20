from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Union
import fitz  # PyMuPDF


DEFAULT_PDF_PATH = Path("data/agent_as_a_judge.pdf")


@dataclass(frozen=True)
class PageRecord:
    page_number: int  # 1-indexed
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def extract_pages_from_pdf(pdf_path: Union[str, Path] = DEFAULT_PDF_PATH) -> List[PageRecord]:
    """
    Extracts text page-by-page from a PDF document using PyMuPDF.

    Args:
        pdf_path: Path to the target PDF file.

    Returns:
        List of PageRecord items with 1-indexed page_number and extracted text.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        ValueError: If path is not a file or the PDF has 0 pages.
        RuntimeError: If the document cannot be opened or page counts mismatch.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise RuntimeError(f"Unable to open PDF at {path}: {exc}") from exc

    try:
        total_pages = len(doc)
        if total_pages == 0:
            raise ValueError(f"PDF contains no pages: {path}")

        records: List[PageRecord] = []
        for index in range(total_pages):
            page = doc[index]
            text = page.get_text() or ""
            records.append(
                PageRecord(
                    page_number=index + 1,
                    text=text,
                )
            )

        if len(records) != total_pages:
            raise RuntimeError(
                f"Page count mismatch: expected {total_pages}, extracted {len(records)}"
            )

        return records
    finally:
        doc.close()
