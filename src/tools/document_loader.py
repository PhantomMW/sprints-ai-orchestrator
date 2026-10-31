from pathlib import Path

from pptx import Presentation
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = PROJECT_ROOT / "data" / "curriculum_docs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def extract_from_pptx(file_path: Path) -> str:
    """
    Extract complete text from PowerPoint slides.

    Supports text frames, tables, and grouped shapes.
    Preserves slide numbers and source structure.
    """

    presentation = Presentation(file_path)
    extracted_slides = []

    def extract_shape_text(shape):
        texts = []

        if shape.has_text_frame:
            text = shape.text.strip()
            if text:
                texts.append(text)

        if shape.has_table:
            for row in shape.table.rows:
                cells = [
                    cell.text.strip()
                    for cell in row.cells
                ]
                texts.append(" | ".join(cells))

        if shape.shape_type == 6:
            for child_shape in shape.shapes:
                texts.extend(extract_shape_text(child_shape))

        return texts

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1
    ):
        slide_content = []

        for shape in slide.shapes:
            slide_content.extend(extract_shape_text(shape))

        if slide_content:
            extracted_slides.append(
                f"## Slide {slide_number}\n\n"
                + "\n\n".join(slide_content)
            )

    return "\n\n".join(extracted_slides)


def extract_from_pdf(file_path: Path) -> str:
    """
    Extract text from all pages of a PDF document.
    """

    reader = PdfReader(str(file_path))
    extracted_pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        page_text = page.extract_text() or ""

        if page_text.strip():
            extracted_pages.append(
                f"## Page {page_number}\n\n{page_text}"
            )

    return "\n\n".join(extracted_pages)


def extract_document(file_path: Path) -> str:
    """
    Extract text according to the document format.
    """

    extension = file_path.suffix.lower()

    if extension == ".pptx":
        return extract_from_pptx(file_path)

    if extension == ".pdf":
        return extract_from_pdf(file_path)

    if extension in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8")

    raise ValueError(
        f"Unsupported document format: {extension}"
    )


def save_extracted_text(
    source_file: Path,
    extracted_text: str
) -> Path:
    """
    Save extracted text as UTF-8 Markdown with source metadata.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / (
        source_file.stem + ".md"
    )

    content = (
        f"# Source: {source_file.name}\n\n"
        f"Source Type: Client Project Context\n\n"
        f"Original Format: {source_file.suffix}\n\n"
        f"---\n\n"
        f"{extracted_text}"
    )

    output_file.write_text(
        content,
        encoding="utf-8"
    )

    return output_file


def main():
    """
    Extract all supported documents from the source directory.
    """

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    supported_extensions = {
        ".pptx",
        ".pdf",
        ".txt",
        ".md"
    }

    source_files = [
        file_path
        for file_path in SOURCE_DIR.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in supported_extensions
    ]

    if not source_files:
        print("No supported documents found.")
        return

    for source_file in source_files:
        try:
            print(f"\nProcessing: {source_file.name}")

            extracted_text = extract_document(source_file)

            if not extracted_text.strip():
                print("Warning: No text extracted.")
                continue

            output_file = save_extracted_text(
                source_file,
                extracted_text
            )

            print(f"Saved: {output_file}")
            print(
                f"Extracted characters: "
                f"{len(extracted_text)}"
            )

        except Exception as error:
            print(
                f"Failed to process {source_file.name}: "
                f"{error}"
            )


if __name__ == "__main__":
    main()