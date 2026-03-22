from detrex.config import get_config
from ..models.midetr_r50_pku import model

dataloader = get_config("common/data/pku.py").dataloader
train = get_config("common/train.py").train



train.init_checkpoint = "checkpoint/pku.pth"

train.output_dir = "output/pku/trans_eval"

train.device = "cuda"
model.device = train.device

dataloader.test.batch_size = 16

dataloader.evaluator.output_dir = train.output_dir