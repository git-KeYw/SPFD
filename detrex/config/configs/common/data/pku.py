from omegaconf import OmegaConf

import detectron2.data.transforms as T  # 保留无妨，后面不用它
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

from detrex.data import PKU_DAVIS_SOD_Mapper  # 确保这个 import 路径正确

dataloader = OmegaConf.create()

# 训练 loader（你现在就是想用 val 看可视化就先指向 val，正常训练改成 train）
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names=("pku_davis_sod_trainval",)),  # 用元组/列表更稳
    mapper=L(PKU_DAVIS_SOD_Mapper)(          # 注意这里不要写成 L(cls(args))(…)
        is_train=True,
    ),
    total_batch_size=4,
    num_workers=2,
)


# 测试 loader
dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names=("pku_davis_sod_test",)),
    mapper=L(PKU_DAVIS_SOD_Mapper)(
        is_train=False,
    ),
    num_workers=2,
    batch_size=4,
)


# COCOEvaluator 这里 dataset_name 期望是字符串；从 test.dataset.names 里取第一个
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names[0]}",
)