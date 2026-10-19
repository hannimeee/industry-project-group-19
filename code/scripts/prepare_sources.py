"""Extract the original BNPL archives into the landing layout used by notebooks.

Existing matching files are reused. Conflicting files are never overwritten.
The original ZIP files remain unchanged for the research scripts and audits.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZipFile

from source_paths import ARCHIVE_NAMES, ROOT, source_directory


def _destination(landing: Path, part: str, member) -> Path | None:
    """Accept only ordinary files and directories inside the archive's tables/."""
    name = member.filename
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name
            or not path.parts or path.parts[0] != "tables"
            or stat.S_ISLNK(member.external_attr >> 16)):
        raise ValueError(f"Unsafe or unexpected archive member: {name!r}")
    if member.is_dir():
        return None
    if len(path.parts) < 2:
        raise ValueError(f"Expected a file beneath tables/: {name!r}")
    target = landing / part / Path(*path.parts[1:])
    for candidate in (target, *target.parents):
        if candidate == landing:
            break
        if candidate.is_symlink():
            raise ValueError(f"Refusing to extract through a symbolic link: {candidate}")
        if candidate != target and candidate.exists() and not candidate.is_dir():
            raise ValueError(f"Expected a directory but found a file: {candidate}")
    return target


def _same_contents(archive: ZipFile, member, target: Path) -> bool:
    if not target.is_file() or target.stat().st_size != member.file_size:
        return False
    with archive.open(member) as original, target.open("rb") as existing:
        while chunk := original.read(1024 * 1024):
            if chunk != existing.read(len(chunk)):
                return False
    return True


def prepare_sources(source_dir=None, landing_dir=None) -> list[dict]:
    source = source_directory(source_dir).resolve()
    landing = (Path(landing_dir).expanduser() if landing_dir is not None
               else ROOT / "tables/landing").resolve()
    missing = [name for name in ARCHIVE_NAMES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing BNPL archives in {source}: {', '.join(missing)}. "
            "Set BNPL_SOURCE_DIR or pass --source-dir to the folder containing all four ZIPs. "
            "Unset BNPL_SOURCE_DIR to use the default repository/legacy source location.")

    # Inspect every archive and existing destination before writing any files.
    with ExitStack() as stack:
        plans = []
        for name in ARCHIVE_NAMES:
            archive = stack.enter_context(ZipFile(source / name))
            part = Path(name).stem
            files, seen = [], set()
            for member in archive.infolist():
                target = _destination(landing, part, member)
                if target is None:
                    continue
                if target in seen:
                    raise ValueError(f"Duplicate destination in {name}: {member.filename}")
                seen.add(target)
                exists = target.exists()
                if exists and not _same_contents(archive, member, target):
                    raise FileExistsError(
                        f"Existing source file differs from {name}: {target}. "
                        "Move the conflicting file aside before retrying; it was not overwritten.")
                files.append((member, target, exists))
            if not files:
                raise ValueError(f"No source files found under tables/ in {source / name}")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise BadZipFile(f"Corrupt member in {source / name}: {bad_member}")
            plans.append((archive, part, files))

        results = []
        for archive, part, files in plans:
            extracted = skipped = 0
            for member, target, exists in files:
                if exists:
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # Stage complete content beside its destination. Linking publishes
                # atomically and fails if another process created the target.
                with NamedTemporaryFile(prefix=".bnpl-source-", dir=target.parent,
                                        delete=False) as staged:
                    temporary = Path(staged.name)
                    try:
                        with archive.open(member) as original:
                            shutil.copyfileobj(original, staged)
                        staged.close()
                        os.link(temporary, target)
                    finally:
                        staged.close()
                        temporary.unlink(missing_ok=True)
                extracted += 1
            results.append({"part": part, "extracted": extracted, "reused": skipped})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Folder containing all four original ZIP files.")
    parser.add_argument("--landing-dir", type=Path, help="Extraction destination; defaults to repository tables/landing.")
    args = parser.parse_args()
    try:
        results = prepare_sources(args.source_dir, args.landing_dir)
    except (OSError, ValueError, BadZipFile) as exc:
        parser.exit(1, f"Source preparation failed: {exc}\n")
    for result in results:
        print(f"{result['part']}: {result['extracted']} files extracted, {result['reused']} reused")


if __name__ == "__main__":
    main()
