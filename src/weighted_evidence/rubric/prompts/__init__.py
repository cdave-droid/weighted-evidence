"""Jinja prompt templates for LLM-graded rubric calls."""

from importlib.resources import files
from pathlib import Path


def prompt_path(name: str) -> Path:
    """Return the on-disk path to a packaged prompt file (e.g., `grade_modifiers.j2`)."""

    resource = files(__package__).joinpath(name)
    return Path(str(resource))
