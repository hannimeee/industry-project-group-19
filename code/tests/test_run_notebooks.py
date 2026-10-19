"""Notebook failures stop dependent runs when requested and preserve evidence."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import nbformat
import pytest
from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_notebooks as runner


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    notebooks = tmp_path / "code/notebooks"
    notebooks.mkdir(parents=True)
    output = tmp_path / "tables/execution"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "NOTEBOOKS", notebooks)
    monkeypatch.setattr(runner, "OUTPUT", output)
    for name in ["00_first.ipynb", "01_second.ipynb"]:
        nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('example')")]), notebooks / name)
    return notebooks, output


@pytest.mark.parametrize("fail_fast, expected_count", [(True, 1), (False, 2)])
def test_failed_notebook_controls_sequence_and_prints_log(workspace, monkeypatch, capsys, fail_fast, expected_count):
    notebooks, output = workspace
    monkeypatch.setattr(KernelSpecManager, "get_kernel_spec", lambda self, name: object())
    launched = []

    def worker(command, stdout, **kwargs):
        path = Path(command[command.index("--worker") + 1])
        launched.append(path.name)
        result = {"notebook": path.name, "status": "failed", "seconds": 0.01,
                  "error": "Traceback\n\x1b[31mFileNotFoundError: missing source data\x1b[0m\n"}
        (output / (path.stem + ".json")).write_text(json.dumps(result))
        stdout.write("Full failure details\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(runner.subprocess, "run", worker)
    monkeypatch.setattr(sys, "argv", ["run_notebooks.py", *sorted(p.name for p in notebooks.iterdir()),
                                      *(["--fail-fast"] if fail_fast else [])])
    assert runner.main() == 1
    assert len(launched) == expected_count
    assert len(json.loads((output / "latest_run.json").read_text())) == expected_count
    terminal = capsys.readouterr().out
    assert "FileNotFoundError: missing source data" in terminal
    assert f"Log: {output / '00_first.log'}" in terminal
    assert "\x1b" not in terminal
    if fail_fast:
        assert not (output / "01_second.json").exists()


def test_unknown_kernel_preserves_existing_execution_records(workspace, monkeypatch, capsys):
    _, output = workspace
    output.mkdir(parents=True)
    previous = {"00_first.json": "previous record", "00_first.log": "previous log",
                "latest_run.json": "previous run"}
    for name, contents in previous.items():
        (output / name).write_text(contents)

    def missing_kernel(self, name):
        raise NoSuchKernel(name)

    monkeypatch.setattr(KernelSpecManager, "get_kernel_spec", missing_kernel)
    monkeypatch.setattr(KernelSpecManager, "find_kernel_specs", lambda self: {"bnpl-project": "available"})
    monkeypatch.setattr(sys, "argv", ["run_notebooks.py", "00_first.ipynb", "--kernel", "bnpl-proj"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2
    assert "No such kernel named bnpl-proj. Available kernels: bnpl-project." in capsys.readouterr().err
    assert {p.name: p.read_text() for p in output.iterdir()} == previous


def test_real_kernel_preserves_failed_notebook_and_records_success(workspace, tmp_path, monkeypatch):
    notebooks, output = workspace
    output.mkdir(parents=True)
    kernel_dir = tmp_path / "jupyter/kernels/runner-test"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Runner test", "language": "python",
    }))
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path / "jupyter"))
    path = notebooks / "00_first.ipynb"
    source = "raise ValueError('synthetic missing prerequisite')"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)]), path)
    original = path.read_bytes()
    failed = runner.execute_one(path, "runner-test", 30)
    assert failed["status"] == "failed"
    assert failed["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert path.read_bytes() == original
    assert nbformat.read(output / "00_first.failed.ipynb", as_version=4).cells[0].outputs[0].ename == "ValueError"

    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('ready')")]), path)
    passed = runner.execute_one(path, "runner-test", 30)
    assert passed["status"] == "passed"
    assert passed["executed_code_cells"] == 1
    assert nbformat.read(path, as_version=4).cells[0].outputs[0].text == "ready\n"
    assert json.loads((output / "00_first.json").read_text()) == passed
