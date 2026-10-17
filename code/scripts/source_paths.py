"""Locate the original BNPL archives without changing their contents or audits."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_NAMES = tuple(f"project-2-bnpl-tables-part{part}.zip" for part in range(1, 5))


def source_directory(source_dir=None):
    """Use an explicit path, BNPL_SOURCE_DIR, legacy archives, or repository landing."""
    if source_dir is not None:
        return Path(source_dir).expanduser()
    configured = os.environ.get("BNPL_SOURCE_DIR")
    if configured:
        return Path(configured).expanduser()
    legacy = ROOT.parent / "project 2 materials"
    if all((legacy / name).is_file() for name in ARCHIVE_NAMES):
        return legacy
    return ROOT / "tables/landing/archives"
