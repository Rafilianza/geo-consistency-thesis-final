"""Check the source package without loading models or launching jobs."""

import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


repository_root = Path(__file__).resolve().parents[1]
manifest_path = repository_root / "SOURCE_MANIFEST.json"
manifest = json.loads(manifest_path.read_text())
failures = []

# Check that every copied file still matches the recorded original.
for record in manifest["copied_files"]:
    relative_path = record["path"]
    source_path = repository_root / relative_path
    if not source_path.is_file():
        failures.append("Missing copied file: " + relative_path)
        continue
    actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual_hash != record["sha256"]:
        failures.append("Changed copied file: " + relative_path)

# Parse source files without importing them or creating bytecode.
python_paths = sorted(repository_root.rglob("*.py"))
json_paths = sorted(repository_root.rglob("*.json"))
shell_paths = sorted(repository_root.rglob("*.sh"))
for source_path in python_paths:
    try:
        ast.parse(source_path.read_text(), filename=str(source_path))
    except (SyntaxError, UnicodeError) as error:
        failures.append(str(error))
for source_path in json_paths:
    try:
        json.loads(source_path.read_text())
    except (ValueError, UnicodeError) as error:
        failures.append(str(source_path.relative_to(repository_root)) + ": " + str(error))

# Bash -n reads shell syntax without executing the scripts.
bash_path = shutil.which("bash")
if bash_path is None:
    failures.append("Bash is required for shell syntax checks.")
else:
    for source_path in shell_paths:
        result = subprocess.run(
            [bash_path, "-n", str(source_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failures.append(str(source_path.relative_to(repository_root)) + ": " + result.stderr.strip())

if failures:
    for failure in failures:
        print("FAIL:", failure)
    sys.exit(1)

print("Copied-file hashes verified:", len(manifest["copied_files"]))
print("Python files parsed:", len(python_paths))
print("JSON files parsed:", len(json_paths))
print("Shell scripts checked:", len(shell_paths))
print("No models loaded, experiments run or jobs submitted.")
