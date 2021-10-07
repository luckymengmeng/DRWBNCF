import os
import sys
import time
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import move_data_to_device
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.io as scio
from pprint import pformat

from src import metric_fn
from src.utils import init_logger, logger
from src.dataloader import CVDataset, DRDataset
from src.model import WBNCF

@torch.no_grad()
def train_test_fn(model, train_loader, val_loader, save_file_format=None):
    device = model.device
    state = model.training
    model.eval()
    scores, labels, edges = [], [], []
    for batch in train_loader:
        model.train_step(batch)
    for batch in val_loader:
        batch = move_data_to_device(batch, device)
        output = model.test_step(batch)
        label, score = output["label"], output["predict"]
        edge = batch.interaction_pair[:, batch.valid_mask.reshape(-1)]
        scores.append(score.detach().cpu())
        labels.append(label.cpu())
        edges.append(edge.cpu())
    model.train(state)
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    edges = torch.cat(edges, dim=1).numpy()
    eval_star_time_stamp = time.time()
    metric = metric_fn.evaluate(predict=scores, label=labels)
    eval_end_time_stamp = time.time()
    logger.info(f"eval time cost: {eval_end_time_stamp - eval_star_time_stamp}")
    if save_file_format is not None:
        save_file = save_file_format.format(aupr=metric["aupr"], auroc=metric["auroc"])
        scio.savemat(save_file, {"row": edges[0],
                                 "col": edges[1],
                                 "score": scores,
                                 "label": labels,
                                 })
        logger.info(f"save time cost: {time.time() - eval_end_time_stamp}")
    return scores, labels, edges, metric

@torch.no_grad()
def test_fn(model, val_loader, save_file_format=None):
    device = model.device
    state = model.training
    model.eval()
    scores, labels, edges = [], [], []
    for batch in val_loader:
        batch = move_data_to_device(batch, device)
        output = model.step(batch)
        label, score = output["label"], output["predict"]
        edge = batch.interaction_pair[:, batch.valid_mask.reshape(-1)]
        scores.append(score.detach().cpu())
        labels.append(label.cpu())
        edges.append(edge.cpu())
    model.train(state)
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    edges = torch.cat(edges, dim=1).numpy()
    eval_star_time_stamp = time.time()
    metric = metric_fn.evaluate(predict=scores, label=labels)
    eval_end_time_stamp = time.time()
    logger.info(f"eval time cost: {eval_end_time_stamp-eval_star_time_stamp}")
    if save_file_format is not None:
        save_file = save_file_format.format(aupr=metric["aupr"], auroc=metric["auroc"])
        scio.savemat(save_file, {"row": edges[0],
                      "col": edges[1],
                      "score": scores,
                      "label": labels,
                      })
        logger.info(f"save time cost: {time.time()-eval_end_time_stamp}")
    return scores, labels, edges, metric


def train_fn(config, model, train_loader, val_loader):
    checkpoint_callback = ModelCheckpoint(monitor="val/auroc",
                                          mode="max",
                                          save_top_k=1,
                                          verbose=False,
                                          save_last=True)
    lr_callback = pl.callbacks.LearningRateMonitor("epoch")
    trainer = Trainer(max_epochs=config.epochs,
                      default_root_dir=config.log_dir,
                      profiler=config.profiler,
                      fast_dev_run=False,
                      checkpoint_callback=checkpoint_callback,
                      callbacks=[lr_callback],
                      gpus=config.gpus,
                      check_val_every_n_epoch=1
                      )
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    if not hasattr(config, "dirpath"):
        config.dirpath = trainer.checkpoint_callback.dirpath
    # checkpoint and add path
    # checkpoint = torch.load("lightning_logs/version_7/checkpoints/epoch=85.ckpt")
    # trainer.on_load_checkpoint(checkpoint)
    print(model.device)


def train(config, model_cls=WBNCF):
    time_stamp = time.asctime()
    datasets = DRDataset(dataset_name=config.dataset_name, drug_neighbor_num=config.drug_neighbor_num,
                         disease_neighbor_num=config.disease_neighbor_num)
    log_dir = os.path.join(f"{config.comment}", f"{config.split_mode}-{config.n_splits}-fold", f"{config.dataset_name}",
                           f"{model_cls.__name__}", f"{time_stamp}")
    config.log_dir = log_dir
    config.n_drug = datasets.drug_num
    config.n_disease = datasets.disease_num

    config.size_u = datasets.drug_num
    config.size_v = datasets.disease_num

    config.gpus = 1 if torch.cuda.is_available() else 0
    config.pos_weight = datasets.pos_weight
    config.time_stamp = time_stamp
    logger = init_logger(log_dir)
    logger.info(pformat(vars(config)))
    config.dataset_type = config.dataset_dype if config.dataset_type is not None else model_cls.DATASET_TYPE
    cv_spliter = CVDataset(datasets, split_mode=config.split_mode, n_splits=config.n_splits,
                           drug_idx=config.drug_idx, disease_idx=config.disease_idx,
                           train_fill_unknown=config.train_fill_unknown,
                           global_test_all_zero=config.global_test_all_zero, seed=config.seed,
                           dataset_type=config.dataset_type)
    pl.seed_everything(config.seed)
    scores, labels, edges, split_idxs = [], [], [], []
    metrics = {}
    start_time_stamp = time.time()
    for split_id, datamodule in enumerate(cv_spliter):
        # if split_id not in [4, 5]:
        #     continue
        config.split_id = split_id
        split_start_time_stamp = time.time()

        datamodule.prepare_data()
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        config.pos_weight = train_loader.dataset.pos_weight
        model = model_cls(**vars(config))
        model = model.cuda() if config.gpus else model

        if split_id==0:
            logger.info(model)
        logger.info(f"begin train fold {split_id}/{len(cv_spliter)}")
        train_fn(config, model, train_loader=train_loader, val_loader=val_loader)
        logger.info(f"end train fold {split_id}/{len(cv_spliter)}")
        save_file_format = os.path.join(config.log_dir,
                                        f"{config.dataset_name}-{config.split_id} fold-{{auroc}}-{{aupr}}.mat")
        score, label, edge, metric = test_fn(model, val_loader, save_file_format)
        # score, label, edge, metric = train_test_fn(model, train_loader, val_loader, save_file_format)
        metrics[f"split_id_{split_id}"] = metric
        scores.append(score)
        labels.append(label)
        edges.append(edge)
        split_idxs.append(np.ones(len(score), dtype=int)*split_id)
        logger.info(f"{split_id}/{len(cv_spliter)} folds: {metric}")
        logger.info(f"{split_id}/{len(cv_spliter)} folds time cost: {time.time()-split_start_time_stamp}")

        if config.debug:
            break
    end_time_stamp = time.time()
    logger.info(f"total time cost:{end_time_stamp-start_time_stamp}")
    with pd.ExcelWriter(os.path.join(log_dir, f"tmp.xlsx")) as f:
        pd.DataFrame(metrics).T.to_excel(f, sheet_name="metrics")
        params = pd.DataFrame({key:str(value) for key, value in vars(config).items()}, index=[str(time.time())])
        params.to_excel(f, sheet_name="params")

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    edges = np.concatenate(edges, axis=1)
    split_idxs = np.concatenate(split_idxs, axis=0)
    final_metric = metric_fn.evaluate(predict=scores, label=labels, is_final=True)
    metrics["final"] = final_metric
    metrics = pd.DataFrame(metrics).T
    metrics.index.name = "split_id"
    metrics["seed"] = config.seed
    logger.info(f"final {config.dataset_name}-{config.split_mode}-{config.n_splits}-fold-auroc:{final_metric['auroc']}-aupr:{final_metric['aupr']}")
    output_file_name = f"final-{config.dataset_name}-{config.split_mode}-{config.n_splits}-auroc:{final_metric['auroc']}-aupr:{final_metric['aupr']}-fold"
    scio.savemat(os.path.join(log_dir, f"{output_file_name}.mat"),
                 {"row": edges[0],
                  "col": edges[1],
                  "score": scores,
                  "label": labels,
                  "split_idx":split_idxs}
                 )
    with pd.ExcelWriter(os.path.join(log_dir, f"{output_file_name}.xlsx")) as f:
        metrics.to_excel(f, sheet_name="metrics")
        params = pd.DataFrame({key:str(value) for key, value in vars(config).items()}, index=[str(time.time())])
        for key, value in final_metric.items():
            params[key] = value
        params["file"] = output_file_name
        params.to_excel(f, sheet_name="params")

    logger.info(f"save final results to r'{os.path.join(log_dir, output_file_name)}.mat'")
    logger.info(f"final results: {final_metric}")



def parse(print_help=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NNCFDR", type=str)
    parser.add_argument("--epochs", default=64, type=int)
    parser.add_argument("--drug_feature_topk", default=20, type=int)  # add prior knowledge
    parser.add_argument("--disease_feature_topk", default=20, type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--profiler", default=False, type=str)
    parser.add_argument("--comment", default="runs", type=str, help="experiment name")
    parser = DRDataset.add_argparse_args(parser)
    parser = CVDataset.add_argparse_args(parser)
    parser = WBNCF.add_model_specific_args(parser)
    args = parser.parse_args()
    if print_help:
        parser.print_help()
    return args


if __name__=="__main__":
    args = parse(print_help=True)
    train(args)
