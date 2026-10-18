import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path


def fallback_title(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").title()


def extract_markdown_title(content: str, path: Path) -> str:
    for line in content.splitlines():
        match = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()

    return fallback_title(path)


def calculate_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def get_modified_at(path: Path) -> datetime:
    return datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=UTC,
    )
