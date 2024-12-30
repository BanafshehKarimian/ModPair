import os
import numpy as np
import pandas as pd
import torch
from torchmetrics.functional import auroc
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
import random
import pytorch_lightning as pl

def seed_function(seed, extra = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)

    # If you're using GPUs:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if extra:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(1)

def save_predictions(model, output_fname, num_classes):
    prds = torch.cat(model.predictions, dim=0)
    trgs = torch.cat(model.targets, dim=0)

    auc = auroc(prds, trgs, num_classes=num_classes, average='macro', task='multiclass')

    print('AUROC (test)')
    print(auc)

    cols_names = ['class_' + str(i) for i in range(0, num_classes)]

    df = pd.DataFrame(data=prds.cpu().numpy(), columns=cols_names)    
    df['target'] = trgs.cpu().numpy()
    df.to_csv(output_fname, index=False)
    l = []
    for i in range(num_classes):
        l.append(df['class_'+ str(i)])
    preds = np.stack(l).transpose()
    targets = np.array(df['target'])
    print("balanced accuracy, F1 score:")
    print(balanced_accuracy_score(targets, preds.argmax(1)), f1_score(targets, preds.argmax(1), average='micro'))



class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


import torch.nn as nn

def replace_linear_with_lora(model, rank, alpha):
    """
    Replace all nn.Linear layers in the model with LinearWithLoRA layers.
    
    Args:
        model (nn.Module): The PyTorch model.
        rank (int): Rank parameter for the LoRA layer.
        alpha (float): Alpha parameter for the LoRA layer.

    Returns:
        nn.Module: The model with LinearWithLoRA layers replacing Linear layers.
    """
    for name, module in model.named_children():
        # Recursively apply to child modules
        if isinstance(module, nn.Linear):
            # Replace Linear with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)
    
    return model