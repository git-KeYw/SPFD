
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def _num(stem: str) -> Optional[int]:
    import re
    m = re.search(r"\d+", stem)
    return int(m.group()) if m else None


def _read_labelme(json_path: Path) -> Tuple[Optional[int], int, int, list, list]:
    with open(json_path, "r") as f:
        d = json.load(f)

    W = int(d.get("imageWidth", 346))
    H = int(d.get("imageHeight", 260))

    img_idx = None
    ip = d.get("imagePath", "")
    if ip:
        img_idx = _num(Path(ip).stem)
    if img_idx is None:
        img_idx = _num(json_path.stem)

    clses, boxes = [], []
    for shp in d.get("shapes", []):

        try:
            c = int(shp.get("label", shp.get("lable", "0")))
        except Exception:
            continue

        pts = shp.get("points", [])
        if len(pts) < 2:
            continue
        (x1, y1), (x2, y2) = pts[0], pts[1]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1 = max(0.0, min(x1, W))
        x2 = max(0.0, min(x2, W))
        y1 = max(0.0, min(y1, H))
        y2 = max(0.0, min(y2, H))
        if x2 > x1 and y2 > y1:
            clses.append(int(c))
            boxes.append([x1, y1, x2, y2])

    return img_idx, W, H, clses, boxes


def register_pku_davis_sod(
    name: str,
    root_dir: str,
    split: str,
    subsets: Optional[List[str]] = None,
    frames_dirname: str = "aps_frames",
    events_dirname: str = "events_npys",
    ann_dirname: str = "annotations",
    class_names: Tuple[str, ...] = ("car", "pedestrian", "cyclist"),
    event_repre: str = "voxel",
    no_frame: bool = False,
    no_event: bool = False,
    anno_driven: bool = True,
    anno_offset: int = 1,
):

    root = Path(root_dir)
    froot, eroot, aroot = root / frames_dirname, root / events_dirname, root / ann_dirname
    K = len(class_names)

    def _list_subsets() -> List[str]:
        if subsets is not None:
            return subsets
        s = set()
        for base in (froot, eroot, aroot):
            d = base / split
            if d.exists():
                for p in d.iterdir():
                    if p.is_dir():
                        s.add(p.name)
        return sorted(s)

    def _to_zero_based(c: int) -> Optional[int]:

        if 0 <= c < K:
            return c
        if 1 <= c <= K:
            return c - 1
        return None

    def _safe_img_size(img_path: str, fallback_w: int = 346, fallback_h: int = 260) -> Tuple[int, int]:

        if img_path:
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
                return int(W), int(H)
            except Exception:
                pass
        return fallback_w, fallback_h

    def _loader() -> List[Dict[str, Any]]:
        dataset: List[Dict[str, Any]] = []
        int_img_id = 1

        for sub in _list_subsets():
            f_sub, e_sub, a_sub = froot / split / sub, eroot / split / sub, aroot / split / sub

            seqs = set()
            for d in (f_sub, e_sub, a_sub):
                if d.exists():
                    for p in d.iterdir():
                        if p.is_dir():
                            seqs.add(p.name)

            for seq in sorted(seqs):
                f_seq, e_seq, a_seq = f_sub / seq, e_sub / seq, a_sub / seq
                rel_prefix = f"{split}/{sub}/{seq}"

                if anno_driven and a_seq.is_dir():
                    jlist = sorted(a_seq.glob("*.json"), key=lambda p: _num(p.stem))
                    for jp in jlist:
                        img_idx, W, H, clses, boxes = _read_labelme(jp)
                        if img_idx is None:
                            continue

                        img_p = (
                            str(f_seq / f"{img_idx}.png")
                            if (not no_frame and (f_seq / f"{img_idx}.png").exists())
                            else ""
                        )
                        evt_p = (
                            str(e_seq / f"{img_idx}.npy")
                            if (not no_event and (e_seq / f"{img_idx}.npy").exists())
                            else ""
                        )
                        if (not img_p) and (not evt_p):
                            continue

                        annos = []
                        for c, (x1, y1, x2, y2) in zip(clses, boxes):
                            cz = _to_zero_based(int(c))
                            if cz is None:
                                continue
                            annos.append(
                                {
                                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "category_id": cz,
                                    "iscrowd": 0,
                                }
                            )

                        dataset.append(
                            {
                                "file_name": img_p,
                                "event_npy": evt_p,
                                "height": int(H),
                                "width": int(W),
                                "image_id": int_img_id,
                                "image_key": f"{rel_prefix}_{img_idx}",
                                "annotations": annos,
                                "event_repre": event_repre,
                                "no_frame": no_frame,
                                "no_event": no_event,
                            }
                        )
                        int_img_id += 1

                else:
                    def _count(dirpath: Path, suffix: str) -> int:
                        if not dirpath.exists():
                            return 0
                        return sum(1 for p in dirpath.iterdir() if p.suffix == suffix)

                    n_png = _count(f_seq, ".png") if (f_seq.exists() and not no_frame) else 0
                    n_npy = _count(e_seq, ".npy") if (e_seq.exists() and not no_event) else 0
                    n = min(x for x in [n_png, n_npy] if x > 0) if (n_png > 0 or n_npy > 0) else 0

                    for idx in range(n):
                        img_p = str(f_seq / f"{idx}.png") if (not no_frame and (f_seq / f"{idx}.png").exists()) else ""
                        evt_p = str(e_seq / f"{idx}.npy") if (not no_event and (e_seq / f"{idx}.npy").exists()) else ""

                        clses, boxes = [], []
                        W = H = None
                        ann_p = a_seq / f"{idx + anno_offset}.json"
                        if ann_p.exists():
                            _, W, H, clses, boxes = _read_labelme(ann_p)

                        if (not img_p) and (not evt_p):
                            continue

                        if W is None or H is None:
                            W, H = _safe_img_size(img_p)

                        annos = []
                        for c, (x1, y1, x2, y2) in zip(clses, boxes):
                            cz = _to_zero_based(int(c))
                            if cz is None:
                                continue
                            annos.append(
                                {
                                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "category_id": cz,
                                    "iscrowd": 0,
                                }
                            )

                        dataset.append(
                            {
                                "file_name": img_p,
                                "event_npy": evt_p,
                                "height": int(H),
                                "width": int(W),
                                "image_id": int_img_id,
                                "image_key": f"{rel_prefix}_{idx}",
                                "annotations": annos,
                                "event_repre": event_repre,
                                "no_frame": no_frame,
                                "no_event": no_event,
                            }
                        )
                        int_img_id += 1

        return dataset

    DatasetCatalog.register(name, _loader)
    meta = MetadataCatalog.get(name)
    meta.set(thing_classes=list(class_names))
    meta.set(evaluator_type="coco")
    meta.set(event_repre=event_repre, no_frame=no_frame, no_event=no_event)



def register_pku_davis_sod_trainval(
    name: str,
    root_dir: str,
    subsets: Optional[List[str]] = None,
    frames_dirname: str = "aps_frames",
    events_dirname: str = "events_npys",
    ann_dirname: str = "annotations",
    class_names: Tuple[str, ...] = ("car", "pedestrian", "two-wheeler"),
    event_repre: str = "voxel",
    no_frame: bool = False,
    no_event: bool = False,
    anno_driven: bool = True,
    anno_offset: int = 1,
):

    root = Path(root_dir)
    froot, eroot, aroot = root / frames_dirname, root / events_dirname, root / ann_dirname
    K = len(class_names)

    def _list_subsets(split: str) -> List[str]:
        if subsets is not None:
            return subsets
        s = set()
        for base in (froot, eroot, aroot):
            d = base / split
            if d.exists():
                for p in d.iterdir():
                    if p.is_dir():
                        s.add(p.name)
        return sorted(s)

    def _to_zero_based(c: int) -> Optional[int]:
        if 0 <= c < K:  # already 0-based
            return c
        if 1 <= c <= K: # 1..K -> 0..K-1
            return c - 1
        return None

    def _safe_img_size(img_path: str, fallback_w: int = 346, fallback_h: int = 260) -> Tuple[int, int]:
        if img_path:
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
                return int(W), int(H)
            except Exception:
                pass
        return fallback_w, fallback_h

    def _build_for_split(split: str, int_start: int) -> Tuple[List[Dict[str, Any]], int]:
        dataset: List[Dict[str, Any]] = []
        int_img_id = int_start

        for sub in _list_subsets(split):
            f_sub, e_sub, a_sub = froot / split / sub, eroot / split / sub, aroot / split / sub

            seqs = set()
            for d in (f_sub, e_sub, a_sub):
                if d.exists():
                    for p in d.iterdir():
                        if p.is_dir():
                            seqs.add(p.name)

            for seq in sorted(seqs):
                f_seq, e_seq, a_seq = f_sub / seq, e_sub / seq, a_sub / seq
                rel_prefix = f"{split}/{sub}/{seq}"

                if anno_driven and a_seq.is_dir():
                    jlist = sorted(a_seq.glob("*.json"), key=lambda p: _num(p.stem))
                    for jp in jlist:
                        img_idx, W, H, clses, boxes = _read_labelme(jp)
                        if img_idx is None:
                            continue
                        img_p = str(f_seq / f"{img_idx}.png") if (not no_frame and (f_seq / f"{img_idx}.png").exists()) else ""
                        evt_p = str(e_seq / f"{img_idx}.npy") if (not no_event and (e_seq / f"{img_idx}.npy").exists()) else ""
                        if (not img_p) and (not evt_p):
                            continue

                        annos = []
                        for c, (x1, y1, x2, y2) in zip(clses, boxes):
                            cz = _to_zero_based(int(c))
                            if cz is None:
                                continue
                            annos.append({
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "category_id": cz,
                                "iscrowd": 0,
                            })

                        dataset.append({
                            "file_name": img_p,
                            "event_npy": evt_p,
                            "height": int(H),
                            "width": int(W),
                            "image_id": int_img_id,
                            "image_key": f"{rel_prefix}_{img_idx}",
                            "annotations": annos,
                            "event_repre": event_repre,
                            "no_frame": no_frame,
                            "no_event": no_event,
                        })
                        int_img_id += 1
                else:
                    def _count(dirpath: Path, suffix: str) -> int:
                        if not dirpath.exists():
                            return 0
                        return sum(1 for p in dirpath.iterdir() if p.suffix == suffix)

                    n_png = _count(f_seq, ".png") if (f_seq.exists() and not no_frame) else 0
                    n_npy = _count(e_seq, ".npy") if (e_seq.exists() and not no_event) else 0
                    n = min(x for x in [n_png, n_npy] if x > 0) if (n_png > 0 or n_npy > 0) else 0

                    for idx in range(n):
                        img_p = str(f_seq / f"{idx}.png") if (not no_frame and (f_seq / f"{idx}.png").exists()) else ""
                        evt_p = str(e_seq / f"{idx}.npy") if (not no_event and (e_seq / f"{idx}.npy").exists()) else ""

                        clses, boxes = [], []
                        W = H = None
                        ann_p = a_seq / f"{idx + anno_offset}.json"
                        if ann_p.exists():
                            _, W, H, clses, boxes = _read_labelme(ann_p)

                        if (not img_p) and (not evt_p):
                            continue
                        if W is None or H is None:
                            W, H = _safe_img_size(img_p)

                        annos = []
                        for c, (x1, y1, x2, y2) in zip(clses, boxes):
                            cz = _to_zero_based(int(c))
                            if cz is None:
                                continue
                            annos.append({
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "category_id": cz,
                                "iscrowd": 0,
                            })

                        dataset.append({
                            "file_name": img_p,
                            "event_npy": evt_p,
                            "height": int(H),
                            "width": int(W),
                            "image_id": int_img_id,
                            "image_key": f"{rel_prefix}_{idx}",
                            "annotations": annos,
                            "event_repre": event_repre,
                            "no_frame": no_frame,
                            "no_event": no_event,
                        })
                        int_img_id += 1

        return dataset, int_img_id

    def _loader_trainval() -> List[Dict[str, Any]]:
        ds_train, next_id = _build_for_split("train", 1)
        ds_val, _ = _build_for_split("val", next_id)
        return ds_train + ds_val

    DatasetCatalog.register(name, _loader_trainval)
    meta = MetadataCatalog.get(name)
    meta.set(thing_classes=list(class_names))
    meta.set(evaluator_type="coco")
    meta.set(event_repre=event_repre, no_frame=no_frame, no_event=no_event)

def register_all(root):
    register_pku_davis_sod(
        name="pku_davis_sod_train",
        root_dir=root,
        split="train",
        subsets=None,
        event_repre="voxel",
        no_frame=False,
        no_event=False,
        anno_driven=True,
    )

    register_pku_davis_sod(
        name="pku_davis_sod_val",
        root_dir=root,
        split="val",
        subsets=None,
        event_repre="voxel",
        no_frame=False,
        no_event=False,
        anno_driven=True,
    )

    register_pku_davis_sod(
        name="pku_davis_sod_test",
        root_dir=root,
        split="test",
        subsets=None,
        event_repre="voxel",
        no_frame=False,
        no_event=False,
        anno_driven=True,
    )

    register_pku_davis_sod_trainval(
        name="pku_davis_sod_trainval",
        root_dir=root,
        subsets=None,
        event_repre="voxel",
        no_frame=False,
        no_event=False,
        anno_driven=True,
    )
root = ""
register_all(root)