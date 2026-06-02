from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

from detrex.data import PKU_DAVIS_SOD_Mapper

dataloader = OmegaConf.create()


dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names=("pku_davis_sod_trainval",)),
    mapper=L(PKU_DAVIS_SOD_Mapper)(       
        is_train=True,
    ),
    total_batch_size=4,
    num_workers=2,
)



dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names=("pku_davis_sod_test",)),
    mapper=L(PKU_DAVIS_SOD_Mapper)(
        is_train=False,
    ),
    num_workers=2,
    batch_size=4,
)


dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names[0]}",
)
