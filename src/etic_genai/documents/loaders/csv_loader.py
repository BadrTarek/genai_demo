import csv
from pathlib import Path

from etic_genai.documents.loaders.base import DocumentLoader
from etic_genai.documents.processing.metadata import fallback_title


class CsvDocumentLoader(DocumentLoader):
    """Turns each row into a self-describing line ("role: x; permission: y"), so a row
    keeps its meaning even when the chunker separates it from the header."""

    supported_extensions = {".csv"}

    def _read(self, path: Path) -> tuple[str, str]:
        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        lines = [
            "; ".join(
                f"{key.strip()}: {(value or '').strip()}" for key, value in row.items() if key
            )
            for row in rows
        ]
        return "\n".join(lines), fallback_title(path)
