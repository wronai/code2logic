from __future__ import annotations

from pathlib import Path
import shutil


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content)
    tmp_path.replace(path)


def cleanup_generated_root(generated_root: Path, allowed_dirs: set[str]) -> None:
    if not generated_root.exists():
        return

    for child in generated_root.iterdir():
        if child.is_dir() and child.name not in allowed_dirs:
            shutil.rmtree(child, ignore_errors=True)
