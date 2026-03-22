from pathlib import Path
import numpy as np
import cv2
import torch
from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np
import cv2
def compute_img_idx_to_track_idx(track_ts: np.ndarray, img_ts: np.ndarray):

    starts = np.searchsorted(track_ts, img_ts, side="left")
    next_img_ts = np.empty_like(img_ts)
    next_img_ts[:-1] = img_ts[1:]
    next_img_ts[-1] = track_ts[-1] + 1  # 保证能取到最后
    ends = np.searchsorted(track_ts, next_img_ts, side="left")

    return np.stack([starts, ends], axis=1)

def _load_seq_list_from_coco(ann_json: Path):
    with open(ann_json, "r") as f:
        coco = json.load(f)
    seqs = set()
    for img in coco["images"]:
        p = Path(img["file_name"])
        seqs.add(p.parts[-2])
    return sorted(seqs)
class DSECDet:
    def __init__(self, root: Path, split: str="train",
                 sync: str="back", split_kind: str="all",
                 ann_root: Path=None, split_config=None):

        root = Path(root)
        assert (root / "seqs").exists(), f"{root}/seqs 不存在"
        assert split in ["train", "val", "test"]
        assert sync in ["front", "back"]
        self.sync = sync
        self.height, self.width = 480, 640

        self.images_root = root / "seqs" / "images"
        self.events_root = root / "seqs" / "events"

        if split_config is None:
            if ann_root is None:
                ann_root = root / "annotations"
            ann_json = ann_root / split_kind / f"{split}.json"
            seq_list = _load_seq_list_from_coco(ann_json)
        else:
            seq_list = split_config[split]

        self.directories = {}
        self.img_idx_track_idxs = {}
        for name in sorted(seq_list):
            images_dir = self.images_root / name                 # .../seqs/images/<seq>
            events_dir = self.events_root / name                 # .../seqs/events/<seq>
            assert (images_dir / "images" / "timestamps.txt").exists(), images_dir
            assert (events_dir / "SIF").exists(), events_dir

            directory = SimpleNamespace()
            directory.images = SimpleNamespace()
            directory.images.timestamps = np.loadtxt(
                images_dir / "images" / "timestamps.txt", dtype=np.int64
            )
            directory.images.image_files_distorted = sorted(
                (images_dir / "images" / "left" / "transformed").glob("*.png")
            )

            directory.events = SimpleNamespace()
            directory.events.sif_dir = events_dir / "SIF"
            directory.events.sif_files = sorted(directory.events.sif_dir.glob("*.npz"))

            directory.tracks = SimpleNamespace()
            directory.tracks.tracks = np.zeros(len(directory.images.timestamps),
                                               dtype=[("t", "int64")])
            directory.tracks.tracks["t"] = directory.images.timestamps

            self.directories[name] = directory

            self.img_idx_track_idxs[name] = compute_img_idx_to_track_idx(
                directory.tracks.tracks["t"], directory.images.timestamps
            )

        self.seq_names = list(self.directories.keys())

    def __len__(self):
        return sum(len(v) - 1 for v in self.img_idx_track_idxs.values())

    def get_index_window(self, index, num_idx, sync="back"):
        if sync == "front":
            assert 0 < index < num_idx
            return index - 1, index
        else:
            assert 0 <= index < num_idx - 1
            return index, index + 1

    def rel_index(self, index, directory_name=None):
        if directory_name is not None:
            return index, self.img_idx_track_idxs[directory_name], self.directories[directory_name]

        for name in self.seq_names:
            idx_map = self.img_idx_track_idxs[name]
            if len(idx_map) - 1 <= index:
                index -= (len(idx_map) - 1)
                continue
            return index, idx_map, self.directories[name]
        raise ValueError("global index 越界")

    def get_image(self, index, directory_name=None):
        index, idx_map, directory = self.rel_index(index, directory_name)
        return cv2.imread(str(directory.images.image_files_distorted[index]))

    def get_events(self, index, directory_name=None):
        index, idx_map, directory = self.rel_index(index, directory_name)
        i0, i1 = self.get_index_window(index, len(idx_map), sync=self.sync)
        t0, t1 = directory.images.timestamps[[i0, i1]]

        events = {"x": [], "y": [], "t": [], "p": []}
        for f in directory.events.sif_files:
            with np.load(f) as z:
                t = z["t"]
                m = (t >= t0) & (t < t1)
                if not m.any():
                    continue
                events["x"].append(z["x"][m])
                events["y"].append(z["y"][m])
                events["t"].append(t[m])
                events["p"].append(z["p"][m])

        for k in events:
            if len(events[k]) == 0:
                events[k] = np.empty((0,), dtype=np.int32 if k in ["x", "y", "p"] else np.int64)
            else:
                events[k] = np.concatenate(events[k], axis=0)
        return events

    def get_tracks(self, index, mask=None, directory_name=None):
        index, idx_map, directory = self.rel_index(index, directory_name)
        i0, i1 = self.get_index_window(index, len(idx_map), sync=self.sync)
        idx0, idx1 = idx_map[i1]
        tracks = directory.tracks.tracks[idx0:idx1]
        if mask is not None:
            tracks = tracks[mask[idx0:idx1]]
        return tracks

    @staticmethod
    def create_3_channel_tensor_from_events(ev, image_size):
        H, W = image_size
        pos = np.zeros((H, W), dtype=np.int32)
        neg = np.zeros((H, W), dtype=np.int32)
        if len(ev["x"]) > 0:
            np.add.at(pos, (ev["y"][ev["p"] == 1], ev["x"][ev["p"] == 1]), 1)
            np.add.at(neg, (ev["y"][ev["p"] == 0], ev["x"][ev["p"] == 0]), 1)
        s = pos + neg
        arr = np.stack([pos, neg, s], axis=-1).astype(np.float32)
        return torch.from_numpy(arr)