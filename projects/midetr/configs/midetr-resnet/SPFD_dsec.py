from detrex.config import get_config
from ..models.midetr_r50_dsec import model

dataloader = get_config("common/data/dsec-det.py").dataloader
train = get_config("common/train.py").train



train.init_checkpoint = "checkpoint/dsec.pth"

train.output_dir = "output/dsec/trans_eval"

train.device = "cuda"
model.device = train.device

dataloader.test.batch_size = 16

dataloader.evaluator.output_dir = train.output_dir