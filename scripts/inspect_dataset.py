"""Inspect downloaded datasets without making format assumptions."""
from pathlib import Path
import json
from collections import Counter
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
def main() -> None:
    root = Path("datasets/raw")
    report = {}
    for source in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        files = list(source.rglob("*"))
        labels = Counter(p.parent.name for p in files if p.suffix.lower() in IMAGE_EXTS)
        annotations = Counter(p.suffix.lower() for p in files if p.suffix.lower() in {".txt", ".xml", ".json"})
        report[source.name] = {"images": sum(labels.values()), "folder_labels": dict(labels), "annotation_extensions": dict(annotations), "files": len(files)}
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/dataset_inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
if __name__ == "__main__": main()
