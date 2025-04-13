# so this is the main training script for my class project
# we are training the JoinABLe model architecture using PyTorch Lightning
# a lot of the data loading and utility code is reused from the original repo

import os
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

from utils import util, metrics  # these are helper functions I didn't touch
from datasets.joint_graph_dataset import JointGraphDataset  # same for this
from args import args_train  # this parses args from CLI
from models.joinable import JoinABLe  # this is the model I rewrote for class


class JointPrediction(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = JoinABLe(
            args.hidden,
            args.input_features,
            dropout=args.dropout,
            mpn=args.mpn,
            batch_norm=args.batch_norm,
            reduction=args.reduction,
            post_net=args.post_net,
            pre_net=args.pre_net
        )
        self.save_hyperparameters()  # this is useful to keep track of config

        # metrics to evaluate performance
        self.test_iou = torchmetrics.IoU(threshold=args.threshold, num_classes=2)
        self.test_accuracy = torchmetrics.Accuracy(threshold=args.threshold, num_classes=2, multiclass=True)

    def forward(self, batch):
        g1, g2, jg = batch
        jg.edge_attr = jg.edge_attr.long()
        return self.model(g1, g2, jg)

    def training_step(self, batch, batch_idx):
        g1, g2, jg = batch
        jg.edge_attr = jg.edge_attr.long()
        out = self(g1, g2, jg)
        loss = self.model.compute_loss(self.args, out, jg)
        self.log("train_loss", loss, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        g1, g2, jg = batch
        jg.edge_attr = jg.edge_attr.long()
        out = self(g1, g2, jg)
        loss = self.model.compute_loss(self.args, out, jg)
        top1 = self.model.precision_at_top_k(out, jg.edge_attr, g1.num_nodes, g2.num_nodes, k=1)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_top_1", top1, prog_bar=True, logger=True)
        return {"loss": loss, "top_1": top1}

    def test_step(self, batch, batch_idx):
        g1, g2, jg = batch
        jg.edge_attr = jg.edge_attr.long()
        out = self(g1, g2, jg)
        loss = self.model.compute_loss(self.args, out, jg)
        probs = F.softmax(out, dim=0)
        self.test_iou.update(probs, jg.edge_attr)
        self.test_accuracy.update(probs, jg.edge_attr)
        top_k = self.model.precision_at_top_k(out, jg.edge_attr, g1.num_nodes, g2.num_nodes)
        return {"loss": loss, "top_k": top_k}

    def test_epoch_end(self, outputs):
        iou_score = self.test_iou.compute()
        acc_score = self.test_accuracy.compute()
        self.log("eval_test_iou", iou_score)
        self.log("eval_test_accuracy", acc_score)

    def configure_optimizers(self):
        opt = Adam(self.model.parameters(), lr=self.args.lr)
        sched = ReduceLROnPlateau(opt, mode="min")
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}


def load_data(args, split, rotate=False):
    return JointGraphDataset(
        root_dir=args.dataset,
        split=split,
        center_and_scale=True,
        random_rotate=rotate,
        delete_cache=args.delete_cache,
        limit=args.limit,
        threads=args.threads,
        label_scheme=(args.train_label_scheme if split == "train" else args.test_label_scheme),
        max_node_count=args.max_node_count,
        input_features=args.input_features
    )


def run_training(args, model_dir, loggers):
    pl.seed_everything(args.seed)
    model = JointPrediction(args)
    ckpt_cb = ModelCheckpoint(monitor="val_loss", mode="min", dirpath=model_dir, filename="best", save_last=True)
    trainer = Trainer(gpus=args.gpus, max_epochs=args.epochs, logger=loggers, callbacks=[ckpt_cb])

    train_ds = load_data(args, split="train", rotate=args.random_rotate)
    val_ds = load_data(args, split="val")
    trainer.fit(model, train_ds.get_train_dataloader(batch_size=args.batch_size), val_ds.get_test_dataloader(batch_size=1))


def run_test(args, model_dir, loggers):
    pl.seed_everything(args.seed)
    ckpt = model_dir / f"{args.checkpoint}.ckpt"
    model = JointPrediction.load_from_checkpoint(ckpt, map_location=torch.device("cpu"))
    trainer = Trainer(gpus=0, logger=loggers)
    test_ds = load_data(args, split=args.test_split)
    trainer.test(model, test_ds.get_test_dataloader(batch_size=1))


def main():
    args = args_train.get_args()
    exp_dir = Path(args.exp_dir)
    model_dir = exp_dir / args.exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    loggers = util.get_loggers(model_dir)

    if args.traintest in ["train", "traintest"]:
        run_training(args, model_dir, loggers)

    if args.traintest in ["test", "traintest"]:
        run_test(args, model_dir, loggers)


if __name__ == "__main__":
    main()
