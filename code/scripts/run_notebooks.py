"""Execute each project notebook in a fresh kernel and record actual outcomes.

Successful runs replace notebook outputs; unsuccessful attempts are retained in
tables/execution without overwriting the last complete output. Execution records
sit outside analysis because historical studies fingerprint that whole layer.
This runner does not supply missing research results or change model settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "code/notebooks"
OUTPUT = ROOT / "tables/execution"


def execute_one(path: Path, kernel: str, timeout: int) -> dict:
    import nbformat
    from nbclient import NotebookClient

    started = time.time()
    nb = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in nb.cells)
    result = {"notebook": path.name, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "kernel": kernel, "started_unix": started}
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Cell {cell_index + 1}/{len(nb.cells)}: {cell.source.splitlines()[0] if cell.source else ''}", flush=True)

    client = NotebookClient(nb, timeout=timeout, kernel_name=kernel,
                            resources={"metadata": {"path": str(path.parent)}},
                            on_cell_start=progress)
    try:
        client.execute()
        nbformat.write(nb, path)
        result["status"] = "passed"
    except Exception as exc:
        failed_path = OUTPUT / (path.stem + ".failed.ipynb")
        nbformat.write(nb, failed_path)
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      attempted_notebook=str(failed_path.relative_to(ROOT)))
        print(str(exc), flush=True)
    result["seconds"] = round(time.time() - started, 2)
    result["executed_code_cells"] = sum(cell.cell_type == "code" and cell.execution_count is not None for cell in nb.cells)
    (OUTPUT / (path.stem + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Notebook filenames; omitted means every notebook.")
    parser.add_argument("--kernel", default="bnpl-project")
    parser.add_argument("--timeout", type=int, default=1800, help="Maximum seconds per cell.")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return 0 if execute_one(args.worker, args.kernel, args.timeout)["status"] == "passed" else 1
    paths = [NOTEBOOKS / name for name in args.names] if args.names else sorted(NOTEBOOKS.glob("*.ipynb"), key=lambda p: (int(p.name.split("_")[0]), p.name != "02_analysis.ipynb", p.name))
    if not paths or any(not p.is_file() or p.parent != NOTEBOOKS for p in paths):
        parser.error("Provide existing notebook filenames from code/notebooks/.")
    results = []
    for path in paths:
        print(f"Running {path.name}", flush=True)
        record = OUTPUT / (path.stem + ".json")
        record.unlink(missing_ok=True)
        with (OUTPUT / (path.stem + ".log")).open("w") as log:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(path),
                                      "--kernel", args.kernel, "--timeout", str(args.timeout)], stdout=log, stderr=subprocess.STDOUT)
        result = json.loads(record.read_text()) if record.exists() else {"notebook": path.name, "status": "failed", "error": f"Worker exited {process.returncode}; see its log."}
        results.append(result)
        print(f"{path.name}: {result['status']} ({result.get('seconds', '?')} seconds)", flush=True)
        (OUTPUT / "latest_run.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(r["status"] == "passed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
