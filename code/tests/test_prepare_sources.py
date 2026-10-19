"""Source preparation preserves originals and refuses unsafe/conflicting output."""
import hashlib
from pathlib import Path
import sys
from zipfile import BadZipFile, ZipFile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_sources import prepare_sources
from source_paths import ARCHIVE_NAMES


@pytest.fixture
def archives(tmp_path):
    source = tmp_path / "archives"
    source.mkdir()
    for part, name in enumerate(ARCHIVE_NAMES, 1):
        with ZipFile(source / name, "w") as archive:
            archive.writestr("tables/", b"")
            archive.writestr(f"tables/snapshot/order_datetime=2021-03-0{part}/data.parquet",
                             f"source part {part}".encode())
    return source


def test_extracts_partition_tree_and_reuses_matching_files(archives, tmp_path):
    originals = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in archives.iterdir()}
    landing = tmp_path / "landing"
    first = prepare_sources(archives, landing)
    assert sum(r["extracted"] for r in first) == 4
    paths = sorted(landing.rglob("*.parquet"))
    assert len(paths) == 4
    assert all(p.parts[-3] == "snapshot" for p in paths)
    assert not any(p.name == "tables" for p in landing.rglob("*"))
    modified = {p: p.stat().st_mtime_ns for p in paths}
    second = prepare_sources(archives, landing)
    assert sum(r["reused"] for r in second) == 4
    assert sum(r["extracted"] for r in second) == 0
    assert modified == {p: p.stat().st_mtime_ns for p in paths}
    assert originals == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in archives.iterdir()}


def test_invalid_environment_path_is_not_silently_replaced(archives, tmp_path, monkeypatch):
    monkeypatch.setenv("BNPL_SOURCE_DIR", str(tmp_path / "wrong"))
    landing = tmp_path / "landing"
    with pytest.raises(FileNotFoundError, match="BNPL_SOURCE_DIR"):
        prepare_sources(landing_dir=landing)
    assert not landing.exists()
    assert len(prepare_sources(archives, landing)) == 4


def test_missing_archive_fails_before_any_extraction(archives, tmp_path):
    (archives / ARCHIVE_NAMES[-1]).unlink()
    landing = tmp_path / "landing"
    with pytest.raises(FileNotFoundError, match=ARCHIVE_NAMES[-1]):
        prepare_sources(archives, landing)
    assert not landing.exists()


def test_conflict_in_last_archive_preserves_file_and_prevents_other_writes(archives, tmp_path):
    landing = tmp_path / "landing"
    conflict = landing / "project-2-bnpl-tables-part4/snapshot/order_datetime=2021-03-04/data.parquet"
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"changed value")
    with pytest.raises(FileExistsError, match="not overwritten"):
        prepare_sources(archives, landing)
    assert conflict.read_bytes() == b"changed value"
    assert not (landing / "project-2-bnpl-tables-part1").exists()


@pytest.mark.parametrize("member", ["tables/../../escape.txt", "/tables/escape.txt",
                                   "unexpected/escape.txt", "tables/..\\escape.txt"])
def test_unsafe_member_prevents_extraction(archives, tmp_path, member):
    with ZipFile(archives / ARCHIVE_NAMES[-1], "a") as archive:
        archive.writestr(member, "unexpected")
    landing = tmp_path / "landing"
    with pytest.raises(ValueError, match="Unsafe or unexpected"):
        prepare_sources(archives, landing)
    assert not landing.exists()


def test_corrupt_archive_prevents_extraction(archives, tmp_path):
    (archives / ARCHIVE_NAMES[-1]).write_bytes(b"invalid zip")
    landing = tmp_path / "landing"
    with pytest.raises(BadZipFile):
        prepare_sources(archives, landing)
    assert not landing.exists()


def test_destination_symlink_cannot_write_outside_landing(archives, tmp_path):
    landing = tmp_path / "landing"
    landing.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (landing / "project-2-bnpl-tables-part1").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        prepare_sources(archives, landing)
    assert not list(outside.iterdir())
