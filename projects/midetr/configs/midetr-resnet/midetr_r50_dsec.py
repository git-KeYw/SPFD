from detrex.config import get_config
from ..models.midetr_r50_dsec import model

# get default config
dataloader = get_config("common/data/dsec-det.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_120k
train = get_config("common/train.py").train

# modify training config
#train.init_checkpoint = "/home/liushuai/wky/DFFT/MI-DETR/output/TriAdapter_mydecoder/model_0089999.pth"
train.output_dir = "output/sfnet_lable_simplefusion2"

#resume train
#train.init_checkpoint = "output/train/model_0013999.pth"
train.resume = True

# max training iterations
train.max_iter = 120000
train.eval_period = 4000
train.log_period = 20
train.checkpointer.period = 2000

# set random seed
train.seed = 42

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 2
dataloader.test.num_workers = 2
# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 4

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
# enable AMP (Automatic Mixed Precision)
train.amp.enabled = True