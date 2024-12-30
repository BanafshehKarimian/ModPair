from Data.PCAMTDataLoaderConch import PCAMTDataModule
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.utils import shuffle
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import auroc
from Models.LateFuser_ import FuserModel
import numpy as np
import random
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
import argparse
from utils.utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=str, default="val_loss")
    parser.add_argument("--ds", type=str, default="pcam")
    parser.add_argument("--learner", type=str, default="late_fusion")
    parser.add_argument("--dir", type=str, default='output')
    parser.add_argument("--output", type=str, default='train_with_text')
    parser.add_argument("--sd", type=int, default=0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--val-int", type=float, default=0.1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--log", type=int, default=1)
    parser.add_argument("--early", action="store_true")
    args = parser.parse_args()


    output_base_dir = args.dir
    output_name = args.output
    output_dir = os.path.join(output_base_dir,output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    seed_function(args.sd)
    datasets_conf = {"pcam":{"num_classes": 2, "ds_class": PCAMTDataModule}}
    learners = {"late_fusion":FuserModel}

    epochs = args.ep
    device = "gpu" if torch.cuda.is_available() else "cpu"
    
    num_classes = datasets_conf[args.ds]["num_classes"]
    batch_size, num_workers = args.batch, args.worker
    Net = FuserModel
    '''Loading Data'''
    data = datasets_conf[args.ds]["ds_class"](batch_size, num_workers)
    '''Creating the model'''
    model = Net(num_classes, batch_size)

    print('=============================================================')
    print('Training...')
    print(device)

    checkpoint_callback = ModelCheckpoint(monitor=args.monitor, mode='min')
    early_stop_callback = EarlyStopping(
            monitor=args.monitor,
            min_delta=args.delta,
            patience=args.patience,  # NOTE no. val epochs, not train epochs
            verbose=False,
            mode="min",
        )
    callbacks=[checkpoint_callback]
    if args.early:
        callbacks.append(early_stop_callback)
    trainer = pl.Trainer(
            callbacks = callbacks,
            log_every_n_steps=args.log,
            max_epochs=epochs,
            accelerator=device,
            devices=1,
            val_check_interval = args.val_int,        
            logger=TensorBoardLogger(output_base_dir, name=output_name),
            gradient_clip_val=args.clip, 
        )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)
    model = Net.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    print(trainer.checkpoint_callback.best_model_path)
    print(trainer.test(model=model, datamodule=data))
    save_predictions(model, os.path.join(output_dir, 'predictions.csv'), num_classes)