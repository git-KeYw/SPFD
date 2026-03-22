# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyexpat import features
import cv2
from detectron2.utils.visualizer import Visualizer
from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers.nms import batched_nms

class MYDINO(nn.Module):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.03605>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: nn.Module,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        aux_loss: bool = True,
        select_box_nums_for_evaluation: int = 300,
        nms_thresh: float = 0.8,
        device="cuda",
        dn_number: int = 100,
        label_noise_ratio: float = 0.2,
        box_noise_scale: float = 1.0,
        input_format: Optional[str] = "RGB",
        vis_period: int = 0,
        init_checkpoint=None
    ):
        super().__init__()
        # define backbone and position embedding module
        self.backbone_rgb = backbone
        self.backbone_evt = backbone
        if init_checkpoint is not None:
            # 只加载到 RGB backbone
            checkpointer = DetectionCheckpointer(self.backbone_rgb)
            checkpointer.load(init_checkpoint)

            # 手动复制到事件分支
            self.backbone_evt.load_state_dict(
                self.backbone_rgb.state_dict(), strict=True
            )
            print(f"[MYDINO] Loaded pretrained weights from {init_checkpoint} into both RGB and EVT backbones")
        self.position_embedding = position_embedding
        self.neck = neck

        # number of dynamic anchor boxes and embedding dimension
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        # denoising
        self.label_enc = nn.Embedding(num_classes, embed_dim)
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # initialize weights
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        # two-stage
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # hack implementation for two-stage
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        # nms_thresh
        self.nms_thresh = nms_thresh

        # the period for visualizing training samples
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"


    def forward(self, batched_inputs):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images,events = self.preprocess_image(batched_inputs)

        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)

        # original features
        features_rgb = self.backbone_rgb(images.tensor)  # output feature dict
        features_evet = self.backbone_evt(events.tensor)
        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in DINO
        # neck 前
        fused_feats = {}
        private_feats_r, private_feats_e = {}, {}
        for k in features_rgb.keys():
            f_rgb, f_evt = features_rgb[k], features_evet[k]
            f_c, f_r, f_e = self.freq_coherence_split(f_rgb, f_evt)
            fused_feats[k] = f_c
            private_feats_r[k] = f_r
            private_feats_e[k] = f_e

        # ✅ 送到 neck，保持对齐
        multi_level_feats = self.neck(fused_feats)  # [B,256,H,W]
        multi_level_feats_r = self.neck(private_feats_r)  # [B,256,H,W]
        multi_level_feats_e = self.neck(private_feats_e)  # [B,256,H,W]
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))
        # denoising preprocessing
        # prepare label query embedding
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets, repeat_new_targets = self.prepare_targets(gt_instances)
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.embed_dim,
                label_enc=self.label_enc,
            )
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # feed into transformer
        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,  # [0..1]
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
            private_feats_r=multi_level_feats_r,  # 新增
            private_feats_e=multi_level_feats_e,  # 新增
        )
        # hack implementation for distributed training
        inter_states[0] += self.label_enc.weight[0, 0] * 0.0

        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # denoising postprocessing
        if dn_meta is not None:
            outputs_class, outputs_coord = self.dn_post_process(
                outputs_class, outputs_coord, dn_meta
            )

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        interm_coord = enc_reference
        interm_class = self.transformer.decoder.class_embed[-1](enc_state)
        output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}

        if self.training:
            # visualize training samples
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    box_cls = output["pred_logits"]
                    box_pred = output["pred_boxes"]
                    results = self.inference(box_cls, box_pred, images.image_sizes)
                    self.visualize_training(batched_inputs, results)
                    # 为每个尺度准备与之对齐的有效区域 mask（False=有效，True=padding）
                    masks = {}
                    for k, feat in fused_feats.items():  # fused_feats 跟 rgb/evt 尺度键一致
                        masks[k] = F.interpolate(img_masks[None], size=feat.shape[-2:], mode="nearest") \
                            .to(torch.bool).squeeze(0)  # [B,H,W]
                    self.visualize_freq_feats(features_rgb, features_evet , fused_feats, private_feats_r, private_feats_e,
                                              masks)
            # compute loss
            loss_dict = self.criterion(output, targets, dn_meta)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def visualize_training(self, batched_inputs, results):
        """
        训练可视化函数：
        左：GT（含类别名）
        中：预测结果（类别名）
        右：原图（可叠加事件图）
        """
        import numpy as np, cv2, torch
        from detectron2.utils.events import get_event_storage
        from detectron2.utils.visualizer import Visualizer, ColorMode
        from detectron2.data import MetadataCatalog
        from detectron2.data.detection_utils import convert_image_to_rgb

        storage = get_event_storage()
        score_thresh = 0.5
        max_vis_box = 20

        # 获取 metadata（包含类别名）
        try:
            metadata = MetadataCatalog.get(self.metadata.name) if hasattr(self, "metadata") else None
        except Exception:
            metadata = None

        for inp, res in zip(batched_inputs, results):
            # ---- 原图 ----
            img = inp["image"].permute(1, 2, 0).cpu().numpy()
            img = convert_image_to_rgb(img, self.input_format).astype("uint8")
            org = img.copy()

            # ---- 单通道事件叠加（白色） ----
            if "event" in inp:
                evt = inp["event"].permute(1, 2, 0).cpu().numpy()  # HWC
                if evt.ndim == 3 and evt.shape[2] > 1:
                    evt = evt[..., 0:1]
                elif evt.ndim == 2:
                    evt = evt[..., None]
                m = evt.max()
                evt_u8 = (evt / m * 255).astype("uint8") if m > 0 else np.zeros_like(evt, dtype="uint8")
                overlay = np.zeros_like(img, dtype="uint8")
                overlay[evt_u8[..., 0] > 0] = (255, 255, 255)
                img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

            # ---- GT 可视化（含类别名） ----
            if "instances" in inp:
                gt_instances = inp["instances"].to("cpu")
                gt_boxes = gt_instances.gt_boxes.tensor.numpy()
                gt_classes = gt_instances.gt_classes.numpy()

                if metadata is not None and hasattr(metadata, "thing_classes"):
                    class_names = metadata.thing_classes
                    gt_labels = [
                        class_names[c] if 0 <= int(c) < len(class_names) else str(int(c))
                        for c in gt_classes
                    ]
                else:
                    gt_labels = [str(int(c)) for c in gt_classes]

                v_gt = Visualizer(img, metadata=metadata, scale=1.0)
                v_gt = v_gt.overlay_instances(boxes=gt_boxes, labels=gt_labels)
                anno_img = v_gt.get_image()
            else:
                anno_img = img.copy()

            # ---- 预测结果 ----
            pred = res
            pred_boxes = pred.pred_boxes.tensor.detach().cpu().numpy() if pred.has("pred_boxes") else np.zeros((0, 4))
            pred_scores = pred.scores.detach().cpu().numpy() if pred.has("scores") else np.zeros((0,))
            pred_classes = pred.pred_classes.detach().cpu().numpy() if pred.has("pred_classes") else np.zeros((0,),
                                                                                                              dtype=int)

            # 阈值筛选 + TopK
            if pred_scores.size > 0:
                keep = pred_scores >= score_thresh
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_classes = pred_classes[keep]

                if pred_scores.size > 0:
                    order = np.argsort(-pred_scores)[:max_vis_box]
                    pred_boxes = pred_boxes[order]
                    pred_scores = pred_scores[order]
                    pred_classes = pred_classes[order]

            # 类名标签（不带分数）
            if metadata is not None and hasattr(metadata, "thing_classes") and len(metadata.thing_classes) > 0:
                class_names = metadata.thing_classes
                labels = [
                    class_names[c] if 0 <= int(c) < len(class_names) else str(int(c))
                    for c in pred_classes
                ]
            else:
                labels = [str(int(c)) for c in pred_classes]

            # ---- 预测结果可视化 ----
            v_pred = Visualizer(img, metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE_BW)
            v_pred = v_pred.overlay_instances(boxes=pred_boxes, labels=labels if len(labels) else None)
            pred_img = v_pred.get_image()

            # ---- 拼接输出 ----
            vis_img = np.concatenate((anno_img, pred_img, org), axis=1).astype("uint8")
            vis_img = vis_img.transpose(2, 0, 1)
            storage.put_image("GT | Pred | Original", vis_img)
            break  # 只显示一个 batch

    def visualize_freq_feats(self, f_rgb, f_evt, f_c, f_r, f_e, masks=None, tag="FeatVisGray"):
        def pick_largest(feats):
            if isinstance(feats, dict):
                feats = list(feats.values())
            if isinstance(feats, (list, tuple)):
                return max(feats, key=lambda t: (t.shape[-2] * t.shape[-1]))
            return feats

        def crop_by_mask(feat, mask_bhw=None):
            # feat: [B,C,H,W] or [C,H,W]; mask_bhw: [B,H,W] (True=padding)
            if mask_bhw is None:
                return feat
            if feat.ndim == 3:  # [C,H,W] -> [1,C,H,W] 对齐 batch 维
                feat = feat.unsqueeze(0);
                squeeze = True
            else:
                squeeze = False
            valid = ~mask_bhw  # False=有效 -> True=有效
            # 按 batch 0 取有效区域的外接矩形
            v0 = valid[0]
            rows = torch.where(v0.any(dim=1))[0]
            cols = torch.where(v0.any(dim=0))[0]
            if len(rows) and len(cols):
                r0, r1 = rows[0].item(), rows[-1].item() + 1
                c0, c1 = cols[0].item(), cols[-1].item() + 1
                feat = feat[:, :, r0:r1, c0:c1]
            if squeeze:
                feat = feat.squeeze(0)
            return feat

        def to_gray_uint8(feat):
            if isinstance(feat, torch.Tensor):
                feat = feat.detach().cpu()
            if feat.ndim == 4:  # [B,C,H,W] -> 第一张
                feat = feat[0]
            if feat.ndim == 3:  # [C,H,W] -> [H,W]
                feat = feat.mean(0)
            arr = feat.numpy() if isinstance(feat, torch.Tensor) else feat
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
            return (arr * 255).astype(np.uint8)

        # 取“最大分辨率”那一层，并按 mask 裁剪
        def pick_and_crop(feats, masks_dict):
            if isinstance(feats, dict):
                # 找面积最大的 key
                k = max(feats.keys(), key=lambda kk: feats[kk].shape[-2] * feats[kk].shape[-1])
                feat = feats[k]
                m = None if (masks_dict is None or k not in masks_dict) else masks_dict[k]
                return crop_by_mask(feat, m)
            return feats

        feats_group = {
            "RGB": f_rgb, "Event": f_evt, "Common": f_c, "RGB_private": f_r, "Event_private": f_e
        }

        vis_imgs = []
        for name, feats in feats_group.items():
            if feats is None:
                continue
            feat = pick_and_crop(feats, masks)
            vis_imgs.append(to_gray_uint8(feat))

        if vis_imgs:
            H = max(img.shape[0] for img in vis_imgs)
            vis_imgs = [np.pad(img, ((0, H - img.shape[0]), (0, 0)), mode="edge") for img in vis_imgs]
            concat_img = np.concatenate(vis_imgs, axis=1)  # [H, sumW]
            concat_img = np.expand_dims(concat_img, axis=0)  # [1,H,W]
            storage = get_event_storage()
            storage.put_image(tag, concat_img)


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def prepare_for_cdn(
        self,
        targets,
        dn_number,
        label_noise_ratio,
        box_noise_scale,
        num_queries,
        num_classes,
        hidden_dim,
        label_enc,
    ):
        """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding
            in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
        if dn_number <= 0:
            return None, None, None, None
            # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["labels"])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None

        dn_number = dn_number // (int(max(known_num) * 2))

        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(
                -1
            )  # half of bbox prob
            new_label = torch.randint_like(
                chosen_indice, 0, num_classes
            )  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_padding = int(max(known_num))

        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = (
            torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        )
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            )
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to("cuda")
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )  # [1,2, 1,2,3]
            map_known_indice = torch.cat(
                [map_known_indice + single_padding * i for i in range(2 * dn_number)]
            ).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * i * 2
                ] = True
            else:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * 2 * i
                ] = True

        dn_meta = {
            "single_padding": single_padding * 2,
            "dn_num": dn_number,
        }

        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas):
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]

            out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1]}
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord)
            dn_metas["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        events = [self.normalizer(x["event"].to(self.device)) for x in batched_inputs]
        events = ImageList.from_tensors(events)
        return images,events

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)

        # # nms for the results
        # if self.nms_thresh > 0:
        #     results = self.nms(results)

        return results

    def nms(self, results):
        new_results = []
        for i, result in enumerate(results):
            keep = batched_nms(result.pred_boxes.tensor, result.scores, result.pred_classes, self.nms_thresh)
            new_result = Instances(result.image_size)
            new_result.pred_boxes = Boxes(result.pred_boxes.tensor[keep])
            new_result.scores = result.scores[keep]
            new_result.pred_classes = result.pred_classes[keep]
            new_results.append(new_result)
        return new_results

    def prepare_targets(self, targets):
        new_targets = []
        repeat_new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            num_inst = len(gt_classes)
            repeat_gt_classes = gt_classes.repeat(3)
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            repeat_gt_boxes = gt_boxes.repeat(3, 1)
            # # create repeat label. Note 0 for repeated while 1 for not repeated
            # repeat_sign = torch.zeros(num_inst * 3, dtype=torch.int64, device=gt_classes.device)
            # repeat_sign[:num_inst] = 1

            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            repeat_new_targets.append({"labels": repeat_gt_classes, "boxes": repeat_gt_boxes})
        return new_targets, repeat_new_targets

    import torch

    def freq_coherence_split(
            self, x, y, eps=1e-6,
            tau=0.25, temp=0.2,
            share_scale=0.7,  # 控制共有特征包含程度
            private_boost=1.5  # 提升私有特征强度
    ):
        """
        改进版频域特征分离（AMP安全版）：
          - 共有特征包含两模态的能量与相干信息
          - 私有特征突出差异与模态独有响应
        """
        # === Step 1: FFT (AMP安全执行) ===
        with torch.cuda.amp.autocast(enabled=False):
            X = torch.fft.rfft2(x.float(), norm="ortho")
            Y = torch.fft.rfft2(y.float(), norm="ortho")

            # === Step 2: 功率谱与互谱 ===
            Sxx = (X.real ** 2 + X.imag ** 2) + eps
            Syy = (Y.real ** 2 + Y.imag ** 2) + eps
            Sxy = X * torch.conj(Y)

            # === Step 3: 相干度 γ² + 强度平衡 ===
            coh = (Sxy.real ** 2 + Sxy.imag ** 2) / (Sxx * Syy)
            strength_balance = torch.sqrt(Sxx * Syy) / (Sxx + Syy + eps)

            # === Step 4: 构造共有掩膜 ===
            Mc = torch.sigmoid(((coh * strength_balance) - tau) / temp)
            Mc = Mc * share_scale + (1 - share_scale) * 0.5  # 保证两模态都参与
            Mc = Mc.clamp(0, 1)

            # === Step 5: 差异掩膜（私有区域） ===
            diff_mag = torch.abs(Sxx - Syy) / (Sxx + Syy + eps)
            Mr = (1.0 - Mc) * diff_mag * (Sxx / (Sxx + Syy))
            Me = (1.0 - Mc) * diff_mag * (Syy / (Sxx + Syy))

            # 私有特征额外增强
            Mr = Mr * private_boost
            Me = Me * private_boost

            # === Step 6: 归一化 ===
            denom = Mc + Mr + Me + eps
            Mc, Mr, Me = Mc / denom, Mr / denom, Me / denom

            # === Step 7: 频域加权 ===
            Zc_hat = Mc * (X + Y) * 0.5
            Zr_hat = Mr * X
            Ze_hat = Me * Y

            # === Step 8: 逆FFT ===
            Zc = torch.fft.irfft2(Zc_hat, s=x.shape[-2:], norm="ortho")
            Zr = torch.fft.irfft2(Zr_hat, s=x.shape[-2:], norm="ortho")
            Ze = torch.fft.irfft2(Ze_hat, s=x.shape[-2:], norm="ortho")

        # 转回混合精度类型，继续后续训练
        return Zc.to(x.dtype), Zr.to(x.dtype), Ze.to(x.dtype)