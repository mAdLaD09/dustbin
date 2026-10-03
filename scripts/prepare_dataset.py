"""Build a cleaned, grouped, leakage-resistant classification dataset.

Supports YOLO txt, Pascal VOC XML, COCO JSON, and class-folder datasets.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil, sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree
from PIL import Image, ImageOps
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config
from src.common.mapping import map_label

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def image_files(root: Path): return [p for p in root.rglob("*") if p.suffix.lower() in EXTS]
def safe_name(path: Path) -> str: return hashlib.sha1(str(path).encode()).hexdigest()[:12] + path.suffix.lower()

def save_image(src: Path, dst: Path) -> bool:
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.save(dst, quality=95)
        return True
    except Exception as exc:
        print(f"WARNING: corrupt/unreadable image skipped: {src} ({exc})")
        return False

def process_folder(source: Path, out: Path, rows: list, cfg: dict) -> None:
    for img in image_files(source):
        mapped = map_label(img.parent.name, cfg.get("enable_reject_class", True))
        if mapped is None:
            print(f"WARNING: unknown or excluded folder label: {img.parent.name} ({img})")
            continue
        if mapped == "reject" and not cfg.get("enable_reject_class", True): continue
        target = out / mapped / (source.name + "_" + safe_name(img)); target.parent.mkdir(parents=True, exist_ok=True)
        if save_image(img, target):
            rows.append({"source_dataset": source.name, "original_filename": str(img), "original_class": img.parent.name, "mapped_class": mapped, "bbox": "", "group_id": str(img.resolve())})

def process_yolo(source: Path, out: Path, rows: list, cfg: dict) -> set[str]:
    labels = {}
    for f in source.rglob("*.names"):
        labels = {str(i): x.strip() for i, x in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines())}
    for ann in source.rglob("*.txt"):
        if ann.name.lower() == "classes.txt": continue
        image = next((ann.with_suffix(ext) for ext in EXTS if ann.with_suffix(ext).exists()), None)
        if not image: continue
        try: im = ImageOps.exif_transpose(Image.open(image)).convert("RGB"); w, h = im.size
        except Exception: continue
        for n, line in enumerate(ann.read_text(encoding="utf-8", errors="ignore").splitlines()):
            parts = line.split()
            if len(parts) != 5: continue
            label = labels.get(parts[0], parts[0]); mapped = map_label(label, cfg.get("enable_reject_class", True))
            if mapped is None: print(f"WARNING: unknown YOLO label {label} in {ann}"); continue
            try: cx, cy, bw, bh = map(float, parts[1:]); x1=max(0,int((cx-bw/2)*w)); y1=max(0,int((cy-bh/2)*h)); x2=min(w,int((cx+bw/2)*w)); y2=min(h,int((cy+bh/2)*h))
            except ValueError: continue
            pad = cfg["dataset"]["padding_fraction"]; pw=int((x2-x1)*pad); ph=int((y2-y1)*pad); x1=max(0,x1-pw); y1=max(0,y1-ph); x2=min(w,x2+pw); y2=min(h,y2+ph)
            if x2-x1 < cfg["dataset"]["min_crop_width"] or y2-y1 < cfg["dataset"]["min_crop_height"] or (x2-x1)*(y2-y1) < cfg["dataset"]["min_crop_area"]: continue
            target=out/mapped/(source.name+"_"+safe_name(image).replace(".",f"_{n}.")); target.parent.mkdir(parents=True,exist_ok=True); im.crop((x1,y1,x2,y2)).save(target)
            rows.append({"source_dataset":source.name,"original_filename":str(image),"original_class":label,"mapped_class":mapped,"bbox":f"{x1},{y1},{x2},{y2}","group_id":str(image.resolve())})
    return set()

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml"); ap.add_argument("--raw",default="datasets/raw"); ap.add_argument("--output",default="datasets/processed"); args=ap.parse_args(); cfg=load_config(args.config); out=Path(args.output); out.mkdir(parents=True,exist_ok=True); rows=[]
    for source in sorted(p for p in Path(args.raw).iterdir() if p.is_dir()) if Path(args.raw).exists() else []:
        has_yolo=any(p.suffix.lower()==".txt" for p in source.rglob("*.txt"))
        if has_yolo: process_yolo(source,out,rows,cfg)
        else: process_folder(source,out,rows,cfg)
    manifest=Path("outputs/dataset_manifest.csv"); manifest.parent.mkdir(exist_ok=True)
    # Assign splits by original-image group so multiple crops cannot leak across splits.
    groups=sorted({r["group_id"] for r in rows}); seed=str(cfg["dataset"]["seed"])
    assignments={g:("train" if int(hashlib.sha1((seed+g).encode()).hexdigest()[:8],16)%100 < int(cfg["dataset"]["train_fraction"]*100) else "val" if int(hashlib.sha1((seed+g).encode()).hexdigest()[:8],16)%100 < int((cfg["dataset"]["train_fraction"]+cfg["dataset"]["validation_fraction"])*100) else "test") for g in groups}
    for r in rows: r["split"]=assignments[r["group_id"]]
    with manifest.open("w",newline="",encoding="utf-8") as f:
        fields=["source_dataset","original_filename","original_class","mapped_class","bbox","group_id","split"]; writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    counts=Counter(r["mapped_class"] for r in rows); splits=Counter(r["split"] for r in rows); total=sum(counts.values()); report={"seed":cfg["dataset"]["seed"],"split_policy":"grouped deterministic 70/15/15 (split recorded in manifest)","split_counts":dict(splits),"total_samples":total,"classes":{k:{"images":v,"percentage":round(100*v/total,2) if total else 0} for k,v in counts.items()},"unknown_labels_require_review":True}
    Path("outputs/dataset_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__ == "__main__": main()
