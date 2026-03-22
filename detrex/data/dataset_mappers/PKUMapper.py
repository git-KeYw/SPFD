# datasets/pku_davis_sod_mapper_d2.py
import os, numpy as np
from typing import Dict, Any, Optional, List
from PIL import Image

import torch
from torch import Tensor
from detectron2.data import transforms as T
from detectron2.structures import Boxes, Instances

def events_to_voxel_grid(events, num_bins=3, height=260, width=346):
    if events.size == 0:
        return np.zeros((height, width, num_bins), np.uint8)
    vg = np.zeros((num_bins, height, width), np.float32).ravel()
    last, first = events[-1]['timestamp'], events[0]['timestamp']
    dT = max(float(last - first), 1.0)
    events['timestamp'] = (num_bins - 1) * (events['timestamp'] - first) / dT
    ts = events['timestamp']
    xs = events['x'].astype(np.int64); ys = events['y'].astype(np.int64)
    pol = events['polarity'].copy(); pol[pol==0] = -1
    tis = ts.astype(np.int64); dts = ts - tis
    vl, vr = pol*(1.0-dts), pol*dts
    valid = tis < num_bins
    np.add.at(vg, xs[valid]+ys[valid]*width + tis[valid]*width*height, vl[valid])
    valid = (tis+1) < num_bins
    np.add.at(vg, xs[valid]+ys[valid]*width + (tis[valid]+1)*width*height, vr[valid])
    vg = vg.reshape(num_bins, height, width).transpose(1,2,0)  # HWC
    return vg

def events_to_gray_image(events, width=346, height=260):
    if events.size == 0:
        return np.zeros((height, width, 3), np.uint8)
    g = np.zeros((height, width), np.float32).ravel()
    xs = events['x'].astype(np.int64); ys = events['y'].astype(np.int64)
    np.add.at(g, xs + ys*width, 1.0)
    g = 255*(1/(1+np.exp(-0.5*g)))
    g = g.reshape(1,height,width).transpose(1,2,0).astype(np.uint8)
    return np.repeat(g, 3, axis=2)

def make_color_histo(events, width=346, height=260):
    img = 255*np.ones((height,width,3), np.uint8)
    if events.size:
        ON = events['polarity'] == 1
        OFF = events['polarity'] == 0
        img[events['y'][ON], events['x'][ON], :] = [30,30,220]
        img[events['y'][OFF], events['x'][OFF], :] = [200,30,30]
    return img

def _load_event_as_hwc_uint8(path: str, repre: str, H: int, W: int):
    ev = np.load(path)
    if repre == "voxel":
        arr = events_to_voxel_grid(ev, 3, H, W)
    elif repre == "gray":
        arr = events_to_gray_image(ev, W, H)
    elif repre == "image":
        arr = make_color_histo(ev, W, H)
    else:
        raise ValueError(repre)
    return arr  # HWC uint8

def build_augs(is_train: bool):
    if is_train:
        return [
            T.ResizeShortestEdge(short_edge_length=(256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576),
                                 max_size=600, sample_style="choice"),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        ]
    else:
        return [T.ResizeShortestEdge(short_edge_length=352, max_size=600)]

from detectron2.data import transforms as T

def build_geom_augs(is_train: bool):
    if is_train:
        return [
            T.RandomCrop_CategoryAreaConstraint(
                crop_type="relative_range",
                crop_size=(0.8, 0.8),
                single_category_max_area=1.0
            ),
            T.RandomRotation(angle=[-7.5, 7.5], expand=False, center=None, sample_style="range"),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.ResizeShortestEdge(short_edge_length=(352, 384, 416, 448, 480, 512, 544, 576),
                                 max_size=800, sample_style="choice"),
        ]
    else:
        return [
            T.ResizeShortestEdge(short_edge_length=576, max_size=800),
        ]

def build_color_augs(is_train: bool):
    if is_train:
        return [
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomSaturation(0.8, 1.2),
            T.RandomLighting(0.15),   # PCA lighting
        ]
    else:
        return []
class PKU_DAVIS_SOD_Mapper:
    def __init__(self, is_train: bool = True, event_repre: Optional[str] = None, do_augmentation: bool = True):
        self.is_train = is_train
        self.event_repre_force = event_repre
        self.do_augmentation = do_augmentation
        self.geom_augs = T.AugmentationList(build_geom_augs(is_train)) if do_augmentation else None
        self.color_augs = T.AugmentationList(build_color_augs(is_train)) if do_augmentation else None

    def __call__(self, d: Dict[str, Any]) -> Dict[str, Any]:
        d = d.copy()
        H, W = int(d["height"]), int(d["width"])
        no_frame = d.get("no_frame", False)
        no_event = d.get("no_event", False)
        event_repre = self.event_repre_force or d.get("event_repre", "voxel")

        if (not no_frame) and d.get("file_name", "") and os.path.exists(d["file_name"]):
            img = np.asarray(Image.open(d["file_name"]).convert("RGB"))
        else:
            img = np.zeros((H, W, 3), np.uint8)

        if (not no_event) and d.get("event_npy","") and os.path.exists(d["event_npy"]):
            evt = _load_event_as_hwc_uint8(d["event_npy"], event_repre, H, W)
        else:
            evt = np.zeros((H, W, 3), np.uint8)

        boxes, classes = [], []
        for a in d.get("annotations", []):
            x1,y1,x2,y2 = a["bbox"]
            if x2 > x1 and y2 > y1:
                boxes.append([x1,y1,x2,y2]); classes.append(int(a["category_id"]))
        boxes = np.array(boxes, dtype=np.float32)
        classes = np.array(classes, dtype=np.int64)

        if self.do_augmentation and self.geom_augs is not None:
            aug_input = T.AugInput(img, boxes=boxes)
            tfm_geom = self.geom_augs(aug_input)
            img   = aug_input.image
            boxes = aug_input.boxes
            evt   = tfm_geom.apply_image(evt)

        if self.do_augmentation and self.color_augs is not None:
            aug_color_in = T.AugInput(img)
            _ = self.color_augs(aug_color_in)
            img = aug_color_in.image

        h, w = img.shape[:2]
        if boxes.size > 0:
            boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, w)
            boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, h)
            keep = (boxes[:,2] > boxes[:,0]) & (boxes[:,3] > boxes[:,1])
            boxes, classes = boxes[keep], classes[keep]
            gt_boxes = Boxes(torch.from_numpy(boxes))
            gt_classes = torch.from_numpy(classes)
        else:
            gt_boxes = Boxes(torch.zeros((0,4), dtype=torch.float32))
            gt_classes = torch.zeros((0,), dtype=torch.int64)

        image = torch.from_numpy(img.astype(np.float32)).permute(2,0,1)
        event = torch.from_numpy(evt.astype(np.float32)).permute(2,0,1)

        inst = Instances((h, w))
        inst.gt_boxes = gt_boxes
        inst.gt_classes = gt_classes

        d["image"] = image
        d["event"] = event
        d["instances"] = inst
        return d