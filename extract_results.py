import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

NUMERIC_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def flatten_json(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten_json(v, f"{prefix}{k}." if prefix else f"{k}.")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from flatten_json(item, f"{prefix}{i}.")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix[:-1], obj


def extract_from_json_file(path: Path):
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(f"[WARN] Impossible de lire JSON {path}: {exc}")
        return []
    return list(flatten_json(data))


def extract_from_python_file(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    metrics = []
    for i, line in enumerate(lines, start=1):
        if any(keyword in line.lower() for keyword in ["precision", "recall", "f1", "iou", "fps", "inference", "latence", "latency", "bench", "acc", "accuracy"]):
            for match in NUMERIC_PATTERN.finditer(line):
                metrics.append((i, line.strip(), match.group(0)))
    return metrics


def print_results_block(title, results):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for item in results:
        print(item)


def main():
    benchmark_file = ROOT / "benchmark_results.json"
    summary_file = ROOT / "summary.json"
    results = []

    if benchmark_file.exists():
        metrics = extract_from_json_file(benchmark_file)
        if metrics:
            results.append(f"Benchmark metrics from {benchmark_file}:")
            for key, value in metrics:
                results.append(f"  - {key}: {value}")

    if summary_file.exists():
        metrics = extract_from_json_file(summary_file)
        if metrics:
            results.append(f"Summary metrics from {summary_file}:")
            for key, value in metrics:
                results.append(f"  - {key}: {value}")

    results_json_dir = ROOT / "results_json"
    if results_json_dir.exists():
        for json_path in sorted(results_json_dir.rglob("*.json")):
            metrics = extract_from_json_file(json_path)
            if metrics:
                results.append(f"Metrics from {json_path.relative_to(ROOT)}:")
                for key, value in metrics:
                    results.append(f"  - {key}: {value}")

    # Search Python sources for metric-like lines
    py_metrics = []
    for py_path in sorted(ROOT.rglob("*.py")):
        if py_path.name == os.path.basename(__file__):
            continue
        file_metrics = extract_from_python_file(py_path)
        if file_metrics:
            py_metrics.append(f"{py_path.relative_to(ROOT)}:")
            for lineno, line, value in file_metrics:
                py_metrics.append(f"  - line {lineno}: {line} --> {value}")

    if results:
        print_results_block("EXTRAITS NUMÉRIQUES DIRECTS", results)
    else:
        print("Aucun résultat numérique direct trouvé dans benchmark_results.json, summary.json ou results_json.")

    if py_metrics:
        print_results_block("LIGNES PYTHON CONTAINANT DES MÉTRIQUES POTENTIELLES", py_metrics)
    else:
        print("Aucune métrique potentielle trouvée dans les fichiers Python via mots-clés.")


if __name__ == "__main__":
    main()
