from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

dataloader = OmegaConf.create()
from detrex.data  import (DsecDetMapper)

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="dsec_det_train"),
    mapper=L(DsecDetMapper)(
        augmentation=[
            T.RandomFlip(),
            T.ResizeShortestEdge(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704,736,),
                max_size=1222,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704,736,),
                max_size=1222,
                sample_style="choice",
            ),
        ],
        is_train=True,
        img_format="RGB",
    ),
    total_batch_size=4,
    num_workers=2,
)
dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="dsec_det_test"),
    mapper=L(DsecDetMapper)(
        augmentation=[
            T.ResizeShortestEdge(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,  # 测试集这里最好改成 False
        img_format="RGB",
    ),
    num_workers=2,
    batch_size=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)