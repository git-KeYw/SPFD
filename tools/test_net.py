#!/usr/bin/env python
"""
Evaluation script using LazyConfig.

This script loads a python config file, builds the model and dataloader,
loads a checkpoint, and runs evaluation. It does NOT support training.
"""

import logging
import os
import sys
import torch
import torch.nn as nn
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.checkpoint import DetectionCheckpointer

from detrex.modeling import ema

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def do_test(cfg, model):

    logger = logging.getLogger("detectron2")

    if "evaluator" not in cfg.dataloader:
        logger.warning("No evaluator found in cfg.dataloader. Skipping evaluation.")
        return {}

    logger.info("Start evaluation on test set.")
    ret = inference_on_dataset(
        model,
        instantiate(cfg.dataloader.test),
        instantiate(cfg.dataloader.evaluator),
    )
    print_csv_format(ret)

    if cfg.train.model_ema.enabled and not cfg.train.model_ema.use_ema_weights_for_eval_only:
        logger.info("Run additional evaluation with EMA weights.")
        with ema.apply_model_ema_and_restore(model):
            ema_ret = inference_on_dataset(
                model,
                instantiate(cfg.dataloader.test),
                instantiate(cfg.dataloader.evaluator),
            )
            print_csv_format(ema_ret)
            ret.update(ema_ret)

    return ret


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)

    ema.may_build_model_ema(cfg, model)

    ckpt_path = cfg.train.init_checkpoint
    logger = logging.getLogger("detectron2")
    logger.info(f"Loading checkpoint (possibly FP16) from: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "model" in ckpt:
        state = ckpt["model"]
    elif "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
    state_fp32 = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            state_fp32[k] = v.float()
        else:
            state_fp32[k] = v

    state_fp32 = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            state_fp32[k] = v.float()
        else:
            state_fp32[k] = v

    if isinstance(model, nn.parallel.DistributedDataParallel):
        real_model = model.module
    else:
        real_model = model

    missing, unexpected = real_model.load_state_dict(state_fp32, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading state_dict: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading state_dict: {unexpected}")

    results = do_test(cfg, model)
    print(results)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )