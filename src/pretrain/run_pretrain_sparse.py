import os

import pandas as pd
import torch
import time
import random
import pprint
import math
import json
from transformers import BertConfig, BertTokenizerFast, logging, BertTokenizer

from src.datasets.dataset_pretrain_sparse import McgPretrainSparseDataset, PretrainImageTextDataset, PretrainCollator
from src.datasets.dataloader import MetaLoader, PrefetchLoader
from src.datasets.data_utils import ImageNorm, mk_input_group
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from src.configs.config import shared_configs
from src.utils.misc import set_random_seed, NoOp, zero_none_grad
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter
from src.utils.basic_utils import load_jsonl, load_json, read_dataframe
from src.utils.load_save import (ModelSaver,
                                 save_training_meta,
                                 load_state_dict_with_pos_embed_resizing)
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.optimization.sched import get_lr_sched
from src.optimization.utils import setup_e2e_optimizer
from collections import defaultdict
from tqdm import tqdm
from os.path import join
from apex import amp
from torch.utils.data.distributed import DistributedSampler
import horovod.torch as hvd
from src.utils.distributed import all_gather_list

from src.modeling.mcg_models import McgForPretrain


def mk_captions_pretrain_dataloader(dataset_name, anno_path, video_dir, txt_dir, cfg, tokenizer, 
                                    is_train=True, max_txt_len=80):
    if dataset_name == "webvid2m":
        datalist = read_dataframe(anno_path)
        datalist = datalist[datalist['txt_len'] < max_txt_len]
        LOGGER.info('Found {} entries for webvid2m'.format(len(datalist)))
    elif dataset_name == "cc3m":
        datalist = json.load(open(anno_path))
        LOGGER.info('Found {} entries for cc3m'.format(len(datalist)))
    else:
        raise ValueError("Invalid dataset_name")

    if dataset_name in ["webvid2m"]:
        frm_sampling_strategy = cfg.frm_sampling_strategy
        if not is_train and frm_sampling_strategy == "rand":
            frm_sampling_strategy = "uniform"
        dataset = McgPretrainSparseDataset(
            datalist=datalist,
            tokenizer=tokenizer,
            img_lmdb_dir=video_dir,
            img_db_type='rawvideo',
            txt_dir=txt_dir,
            crop_size=cfg.crop_img_size,
            resize_size=cfg.resize_size,
            max_txt_len=cfg.max_txt_len,
            fps=cfg.fps,
            num_frm=cfg.num_frm,
            frm_sampling_strategy=frm_sampling_strategy,
            is_train=is_train
            # vis_format=vis_format
        )
    elif dataset_name in ["cc3m"]:
        dataset = PretrainImageTextDataset(datalist=datalist, 
                                           tokenizer=tokenizer,
                                           crop_size=cfg.crop_img_size,
                                           resize_size=cfg.resize_size,
                                           max_txt_len=cfg.max_txt_len,
                                           num_frm=cfg.num_frm)
        LOGGER.info(f"[{dataset_name}] is_train {is_train} "f"dataset size {len(dataset)}, ")

    batch_size = cfg.train_batch_size if is_train else cfg.val_batch_size
    sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=is_train)
    data_collator = PretrainCollator(tokenizer=tokenizer,
                                     mlm=cfg.use_mlm,
                                     mlm_probability=0.15,
                                     max_length=cfg.max_txt_len,
                                     is_train=is_train)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=data_collator.collate_batch)

    return dataloader


def setup_dataloaders(cfg, tokenizer):
    LOGGER.info("Init. train_loader and val_loader...")

    train_loaders = {}
    for db in cfg.train_datasets:
        train_loaders[db.name] = mk_captions_pretrain_dataloader(
            dataset_name=db.name,
            anno_path=db.ann, video_dir=db.img, txt_dir=db.txt,
            cfg=cfg, tokenizer=tokenizer, is_train=True
        )

    val_loaders = {}
    for db in cfg.val_datasets:
        val_loaders[db.name] = mk_captions_pretrain_dataloader(
            dataset_name=db.name,
            anno_path=db.ann, video_dir=db.img, txt_dir=db.txt,
            cfg=cfg, tokenizer=tokenizer, is_train=False
        )
    return train_loaders, val_loaders


def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")
    # has to be a BertConfig instance
    model_cfg = load_json(cfg.model_config)
    model_cfg = BertConfig(**model_cfg)
    # add model-specific config
    add_attr_list = ["max_n_example_per_group"]
    for k in add_attr_list:
        setattr(model_cfg, k, cfg[k])
    LOGGER.info(f"model_cfg {pprint.pformat(model_cfg.to_dict())}")

    LOGGER.info("setup e2e model")

    if cfg.model_type == 'pretrain':
        # initialize cnn config
        video_enc_cfg = load_json(cfg.visual_model_cfg)

        video_enc_cfg['num_frm'] = cfg.num_frm
        video_enc_cfg['img_size'] = cfg.crop_img_size

        model = McgForPretrain(
            model_cfg, 
            input_format=cfg.img_input_format,
            video_enc_cfg=video_enc_cfg
            )
        if cfg.e2e_weights_path:
            LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
            num_patches = (cfg.crop_img_size // video_enc_cfg['patch_size']) ** 2
            load_state_dict_with_pos_embed_resizing(model,
                                                    cfg.e2e_weights_path, 
                                                    num_patches=num_patches, 
                                                    num_frames=cfg.num_frm, 
                                                    strict=False
                                                    )
        else:
            LOGGER.info(f"Loading visual weights from {cfg.visual_weights_path}")
            model.load_separate_ckpt(visual_weights_path=cfg.visual_weights_path)
    else:
        raise NotImplementedError(f"cfg.model_type not found {cfg.model_type}.")
    
    LOGGER.info("Moving model to device") 
    model.to(device)
    LOGGER.info("Completed moving model to device.") 

    LOGGER.info("Setup model done!")
    return model


def forward_step(cfg, model, batch):
    """shared for training and validation"""
    # used to make visual feature copies
    outputs = model(batch)  # dict
    return outputs


@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()

    mlm_loss = 0
    n_mlm_tokens = 0
    n_mlm_corrects = 0
    itc_loss = 0
    mcl_loss = 0
    st = time.time()
    val_log = {'valid/mlm_loss': 0, 'valid/mlm_acc': 0,
               'valid/itc_loss': 0, 'valid/mcl_loss': 0}
    debug_step = 5
    val_loaders = val_loader if isinstance(val_loader, dict) else {
        "unnamed_val_loader": val_loader}
    
    total_val_iters = 0 

    LOGGER.info(f"In total {len(val_loaders)} val loaders")
    for loader_name, val_loader in val_loaders.items():
        LOGGER.info(f"Loop val_loader {loader_name}.")

        total_val_iters += len(val_loader)
        for val_step, batch in enumerate(val_loader):
            # use iter to reset MetaLoader
            # forward pass
            outputs = forward_step(cfg, model, batch)

            # mlm
            mlm_labels = outputs["mlm_labels"]
            if cfg.use_mlm:
                mlm_loss += outputs["mlm_loss"].sum().item()
                mlm_mask = mlm_labels != -100  # (B, Lt)  -100 is the ignored label for cross entropy
                n_mlm_tokens += mlm_mask.sum().item()
                if n_mlm_tokens > 0:
                    n_mlm_corrects += (
                            outputs["mlm_scores"][mlm_mask].max(
                                dim=-1)[1] == mlm_labels[mlm_mask]).sum().item()
                else:
                    n_mlm_corrects = 0

            if cfg.use_itc:
                itc_loss += outputs["itc_loss"].sum().item()

            if cfg.use_mcl:
                mcl_loss += outputs["mcl_loss"].sum().item()

            if cfg.debug and val_step >= debug_step:
                break

    # Gather across all processes
    # mlm_loss = sum(all_gather_list(mlm_loss))
    all_gather_mlm_loss = all_gather_list(mlm_loss)
    mlm_loss = sum(all_gather_mlm_loss)
    n_mlm_corrects = sum(all_gather_list(n_mlm_corrects))
    n_mlm_tokens = sum(all_gather_list(n_mlm_tokens))

    all_gather_itc_loss = all_gather_list(itc_loss)
    itc_loss = sum(all_gather_itc_loss)

    all_gather_mcl_loss = all_gather_list(mcl_loss)
    mcl_loss = sum(all_gather_mcl_loss)

    if n_mlm_tokens != 0:
        val_log.update({
            'valid/mlm_loss': float(mlm_loss),
            'valid/mlm_acc': float(n_mlm_corrects / n_mlm_tokens)
        })

    # FIXME check this whether take mean?
    if cfg.use_itc:
        val_log.update({
            'valid/itc_loss': float(itc_loss),
        })

    if cfg.use_mcl:
        val_log.update({
            'valid/mcl_loss': float(mcl_loss),
        })


    TB_LOGGER.log_scalar_dict(val_log)
    LOGGER.info(f"validation finished in {int(time.time() - st)} seconds, "
                f"[mlm_acc (per token)]: {val_log['valid/mlm_acc'] * 100:.2f} ")

    LOGGER.info("[mlm_loss]: {} ".format(mlm_loss))
    LOGGER.info("[itc_loss]: {} ".format(itc_loss))
    LOGGER.info("[mcl_loss]: {} ".format(mcl_loss))
    LOGGER.info("In total, {} validation iters.".format(total_val_iters))

    model.train()
    return val_log


def start_training():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cfg = shared_configs.get_sparse_pretraining_args()
    set_random_seed(cfg.seed)

    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())

    os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())

    if hvd.rank() != 0:
        LOGGER.disabled = True
    LOGGER.info(f"device: {device} n_gpu: {n_gpu}, "f"rank: {hvd.rank()}, 16-bits training: {cfg.fp16}")

    model = setup_model(cfg, device=device)
    model.train()

    optimizer = setup_e2e_optimizer(model, cfg)

    # Horovod: (optional) compression algorithm.compressin
    compression = hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression)

    #  Horovod: broadcast parameters & optimizer state.
    compression = hvd.Compression.none
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    model, optimizer = amp.initialize(
        model, optimizer, enabled=cfg.fp16, opt_level='O1')
        # keep_batchnorm_fp32=True)

    # prepare data
    tokenizer = init_tokenizer(cfg.tokenizer_dir)
    train_loaders, val_loaders = setup_dataloaders(cfg, tokenizer)
    train_loader = MetaLoader(train_loaders, accum_steps=cfg.gradient_accumulation_steps, distributed=n_gpu > 1)
    img_norm = ImageNorm(mean=cfg.img_pixel_mean, std=cfg.img_pixel_std)
    train_loader = PrefetchLoader(train_loader, img_norm)
    val_loaders = {k: PrefetchLoader(v, img_norm) for k, v in val_loaders.items()}

    # compute the number of steps and update cfg
    total_train_batch_size = int(
        n_gpu * cfg.train_batch_size *
        cfg.gradient_accumulation_steps * cfg.max_n_example_per_group)
    total_n_epochs = cfg.num_train_epochs
    cfg.num_train_steps = int(math.ceil(
        1. * train_loader.n_batches_in_epoch * total_n_epochs /
        (n_gpu * cfg.gradient_accumulation_steps)))
    cfg.valid_steps = int(math.ceil(
        1. * cfg.num_train_steps / cfg.num_valid /
        cfg.min_valid_steps)) * cfg.min_valid_steps
    actual_num_valid = int(math.floor(
        1. * cfg.num_train_steps / cfg.valid_steps)) + 1
    
    save_steps = int(cfg.save_steps_ratio * cfg.num_train_steps)

    # restore
    restorer = TrainingRestorer(cfg, model, optimizer)
    global_step = restorer.global_step
    TB_LOGGER.global_step = global_step
    if hvd.rank() == 0:
        LOGGER.info("Saving training meta...")
        save_training_meta(cfg)
        LOGGER.info("Saving training done...")
        TB_LOGGER.create(join(cfg.output_dir, 'log'))
        pbar = tqdm(total=cfg.num_train_steps)
        model_saver = ModelSaver(join(cfg.output_dir, "ckpt"))
        add_log_to_file(join(cfg.output_dir, "log", "log.txt"))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()
        restorer = NoOp()

    if global_step > 0:
        pbar.update(global_step)

    LOGGER.info(cfg)
    LOGGER.info("Starting training...")
    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info(f"  Single-GPU Non-Accumulated batch size = {cfg.train_batch_size}")
    LOGGER.info(f"  max_n_example_per_group = {cfg.max_n_example_per_group}")
    LOGGER.info(f"  Accumulate steps = {cfg.gradient_accumulation_steps}")
    LOGGER.info(f"  Total batch size = #GPUs * Single-GPU batch size * "
                f"max_n_example_per_group * Accumulate steps [Image] = {total_train_batch_size}")
    LOGGER.info(f"  Total #batches - single epoch = {train_loader.n_batches_in_epoch}.")
    LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
    LOGGER.info(f"  Total #epochs = {total_n_epochs}.")
    LOGGER.info(f"  Validate every {cfg.valid_steps} steps, in total {actual_num_valid} times")
    LOGGER.info(train_loader)

    # quick hack for amp delay_unscale bug
    with optimizer.skip_synchronize():
        optimizer.zero_grad()
        if global_step == 0:
            optimizer.step()
    debug_step = 20

    tasks = []
    for name, flag in zip(["mlm", "itc", "mcl"], [cfg.use_mlm, cfg.use_itc, cfg.use_mcl]):
        if flag:
            tasks.append(name)
    task2loss = {t: RunningMeter(f'train_loss/{t}') for t in tasks}
    task2loss["loss"] = RunningMeter('train_loss/loss')

    train_log = {'train/mlm_acc': 0}

    for step, (task, batch) in enumerate(train_loader):
        # forward pass
        outputs = forward_step(cfg, model, batch)
        mlm_loss, itc_loss, mcl_loss = 0, 0, 0
        if cfg.use_mlm:
            mlm_loss = outputs["mlm_loss"]
            mlm_mask = outputs["mlm_labels"] != -100
            n_mlm_tokens = mlm_mask.sum().item()
            task2loss["mlm"](mlm_loss.item())
        
        if cfg.use_itc:
            itc_loss = outputs["itc_loss"]
            task2loss["itc"](itc_loss.item())

        if cfg.use_mcl:
            mcl_loss = outputs["mcl_loss"]
            task2loss["mcl"](mcl_loss.item())

        loss = mlm_loss + itc_loss + mcl_loss
        task2loss["loss"](loss.item())

        if step % cfg.log_interval == 0:
            # training mlm token acc
            if n_mlm_tokens > 0:
                n_mlm_corrects = (
                        outputs["mlm_scores"][mlm_mask].max(
                            dim=-1)[1] == outputs['mlm_labels'][mlm_mask]).sum().item()
            else:
                n_mlm_corrects = 0

            train_log.update({
                'train/mlm_acc': float(n_mlm_corrects / n_mlm_tokens),
            })

            TB_LOGGER.log_scalar_dict(train_log)

        delay_unscale = (step + 1) % cfg.gradient_accumulation_steps != 0
        with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale) as scaled_loss:
            scaled_loss.backward()
            zero_none_grad(model)
            optimizer.synchronize()

        # optimizer
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            global_step += 1
            if (step + 1) % cfg.log_interval == 0:
                TB_LOGGER.log_scalar_dict({l.name: l.val
                                        for l in task2loss.values()
                                        if l.val is not None})
            n_epoch = int(1. * n_gpu * cfg.gradient_accumulation_steps *
                          global_step / train_loader.n_batches_in_epoch)

            # learning rate scheduling for the whole model
            lr_this_step = get_lr_sched(
                global_step, cfg.decay, cfg.learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.step_decay_epochs,
                multi_step_epoch=n_epoch)

            # Hardcoded param group length
            # assert len(optimizer.param_groups) == 8
            for pg_n, param_group in enumerate(
                    optimizer.param_groups):
                    param_group['lr'] = lr_this_step

            if (step + 1) % cfg.log_interval == 0:
                TB_LOGGER.add_scalar(
                    "train/lr", lr_this_step, global_step)

            # update model params
            if cfg.grad_norm != -1:
                # import pdb; pdb.set_trace()
                grad_norm = clip_grad_norm_(
                    amp.master_params(optimizer), cfg.grad_norm)
                if (step + 1) % cfg.log_interval == 0:
                    TB_LOGGER.add_scalar("train/grad_norm", grad_norm, global_step)
            TB_LOGGER.step()

            # Check if there is None grad
            none_grads = [
                p[0] for p in model.named_parameters()
                if p[1].requires_grad and p[1].grad is None]

            assert len(none_grads) == 0, f"{none_grads}"

            with optimizer.skip_synchronize():
                optimizer.step()
                optimizer.zero_grad()
            restorer.step()
            pbar.update(1)

            if global_step % cfg.valid_steps == 0:
                LOGGER.info(f'Step {global_step}: start validation')
                validate(model, val_loaders, cfg)
                model_saver.save(step=global_step, model=model)

            if global_step % save_steps == 0:
                LOGGER.info(f'Step {global_step}: saving model checkpoints.')
                model_saver.save(step=global_step, model=model)

        if global_step >= cfg.num_train_steps:
            break

        if cfg.debug and global_step >= debug_step:
            break

    if global_step % cfg.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        validate(model, val_loaders, cfg)
        model_saver.save(step=global_step, model=model)


def init_tokenizer(path):
    tokenizer = BertTokenizer.from_pretrained(path)
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


if __name__ == '__main__':
    # Initialize Horovod
    hvd.init()
    start_training()
