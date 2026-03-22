# register_dsec_det_with_events.py
import os, os.path as osp, re
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

def _infer_seq_and_index(img_path):
    parts = img_path.split("/")
    try:
        i_seqs = parts.index("seqs")
        assert parts[i_seqs+1] == "images"
        seq_name = parts[i_seqs+2]
    except Exception:
        m = re.search(r"/seqs/images/([^/]+)/", img_path)
        if not m:
            raise RuntimeError(f"Cannot infer seq name from: {img_path}")
        seq_name = m.group(1)
    stem = osp.splitext(osp.basename(img_path))[0]
    frame_idx = int(stem)
    return seq_name, frame_idx

def _add_event_fields(records, root, sync="back"):
    for r in records:
        img = r["file_name"]
        seq, i = _infer_seq_and_index(img)
        ts_txt = osp.join(root, "seqs", "images", seq, "images", "timestamps.txt")
        ts = np.loadtxt(ts_txt, dtype=np.int64)
        if i+1 >= len(ts):
            i = max(0, len(ts)-2)
        t0, t1 = int(ts[i]), int(ts[i+1])
        r["timestamps"] = [t0, t1]
        event_npz = osp.join(root, "seqs", "events", seq, "SIF", f"{i+1:06d}.npz")
        if osp.isfile(event_npz):
            r["event_file"] = event_npz
            r["event_format"] = "npz"
        else:
            r["event_file"] = None
            r["event_format"] = None
    return records

def register_dsec_det(root: str, use_sub: bool = False, sync="back"):
    split_tag = "sub" if use_sub else "all"
    ann_dir   = osp.join(root, "annotations", split_tag)
    img_root  = osp.join(root, "seqs", "images")

    for split in ["train", "test"]:
        ann_file = osp.join(ann_dir, f"{split}.json")
        name = f"dsec_det_{split}"

        def _loader(ann_file=ann_file, img_root=img_root, name=name):
            recs = load_coco_json(ann_file, img_root, name)
            recs = _add_event_fields(recs, root, sync=sync)
            return recs

        DatasetCatalog.register(name, _loader)
        MetadataCatalog.get(name).set(
            evaluator_type="coco",
            json_file=ann_file,
            image_root=img_root,
        )
root = ""
register_dsec_det(root, use_sub=False)