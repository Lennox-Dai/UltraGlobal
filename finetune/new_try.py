import os
import cv2
import math
import pdb
import json
import torch
import warnings
import torchvision
import core.checkpoint as checkpoint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import core.transforms as transforms

from finetune.util import setup_model, GeneralizedMeanPooling, GeneralizedMeanPoolingP, GlobalHead, sgem, rgem, NonLocalBlock, gemp
from finetune.PAnet import PANet
from torch.utils.data import Dataset
from transformers import get_cosine_schedule_with_warmup
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from model.CVNet_Rerank_model import CVNet_Rerank
from torchmetrics.functional.classification import auroc
import core.checkpoint as checkpoint
from other.GLAM import GLAM
from torchvision import models
from torch.optim import AdamW

_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataSet(torch.utils.data.Dataset):

    def __init__(self, data_path, fn, mode):
        assert os.path.exists(
            data_path), "Data path '{}' not found".format(data_path)
        self._data_path, self._fn = data_path, fn
        self._label_path = os.path.join(data_path, "data")
        self.mode = mode
        self._construct_db()

    def _construct_db(self):
        # Compile the split data path
        self._db = []
        self._dbl = []

        with open(os.path.join(self._data_path, self._fn), 'rb') as fin:
            gnd = json.load(fin)

        if self.mode == 'train':
            start_pic = 0
            end_pic = int(len(gnd) * 0.7)
        else:
            start_pic = int(len(gnd) * 0.7)
            end_pic = len(gnd)

        for i in range(start_pic, end_pic):
            im_fn = gnd[f'{i}']["query"]
            im_path = os.path.join(
                self._data_path, "images", im_fn)
            self._db.append({"im_path": im_path})
            rel_img_paths = []
            for j in range(len(gnd[f'{i}']["similar"])):
                rel_img_path = os.path.join(
                    self._data_path, "images", gnd[f'{i}']["similar"][j])
                rel_img_paths.append(rel_img_path)
            self._dbl.append({"rel_img_paths": rel_img_paths})


    def _prepare_im(self, im):
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # Color normalization
        im = transforms.color_norm(im, _MEAN, _SD)
        return im

    def __getitem__(self, index):
        im_list = []
        rel = []
        try:
            im = cv2.imread(self._db[index]["im_path"])
            if im is None:
                raise ValueError("Image not found or unable to load.")
            im_np = im.astype(np.float32, copy=False)
            im_list.append(im_np)
            for i in range(len(self._dbl[index]["rel_img_paths"])):
                im_rel = cv2.imread(self._dbl[index]["rel_img_paths"][i])
                if im_rel is None:
                    raise ValueError("Image not found or unable to load.")
                im_rel = im_rel.astype(np.float32, copy=False)
                rel.append(im_rel)

        except Exception as e:
            print('error:', self._db[index]["im_path"], e)

        for idx in range(len(im_list)):
            im_list[idx] = self._prepare_im(im_list[idx])
        for idx in range(len(rel)):
            rel[idx] = self._prepare_im(rel[idx])

        return {"img": im_list, "rel": rel}

    def __len__(self):
        return len(self._db)

class UCC_Data_Module(pl.LightningDataModule):

  def __init__(self, train_path, gnd_fn, batch_size: int=1, shuffle: bool=False):
    super().__init__()
    self.train_path = train_path
    self.gnd_fn = gnd_fn
    self.batch_size = batch_size
    self.shuffle = shuffle

  def train_dataloader(self):
    return self._construct_loader(self.train_path, self.gnd_fn, self.batch_size, self.shuffle, mode="train")

  def val_dataloader(self):
    return self._construct_loader(self.train_path, self.gnd_fn, self.batch_size, shuffle=False, mode="eval")

  # def train_dataloader(self):
  #   return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

  # def val_dataloader(self):
  #   return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

  def predict_dataloader(self):
    return self._construct_loader(self.train_path, self.gnd_fn, self.batch_size, shuffle=False)
  
  def _construct_loader(self, _DATA_DIR, fn, batch_size, shuffle, mode):
    dataset = DataSet(_DATA_DIR, fn, mode)
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
    )
    return loader

class UCC_Classifier(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.val_losses = []
        self.train_losses = [] * config['n_epochs']
        # self.train_losses.unsqueeze(0)
        self.validation_step_outputs = []
        self.best_val_loss = float('inf')
        self.config = config
        self.reduction_dim = config["reduction_dim"]
        self.pan = PANet()
        self.load = config["load"]
        self.resnet = setup_model(self.config['model_depth'], self.reduction_dim, self.config['SupG'])
        checkpoint.load_checkpoint(self.config['pre_trained_weight'], self.resnet)
        
        # FIXME 如果之后想要写resnet18可以再扩展一下
        
        # if self.config["model"]=='resnet18':
        #     self.gemp = gemp(m=self.reduction_dim, channel=512)
        #     self.head = GlobalHead(512, nc=self.reduction_dim)
        # else:
        #     self.gemp = gemp(m=self.reduction_dim)
        #     self.head = GlobalHead(2048, nc=self.reduction_dim)
        # self.rgem = rgem()
        # self.sgem = sgem()
        # self.head = GlobalHead(2048, nc=self.reduction_dim)

        # FIXME loss fun的选择
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')
        self.dropout = nn.Dropout()

        # 用来快速调试用的

        # FIXME 先试试holiday，不做分类而做相似度比较
        # self.hidden = nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        # self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])

    def _forward_singlescale(self, x):
        if self.config['withpan']:
            x = self.pan(x)
        output = self.resnet.extract_global_descriptor(x, self.config['SupG']['gemp'], self.config['SupG']['rgem'], self.config['SupG']['sgem'], self.config['SupG']['scale_list'])
        output = F.normalize(output, p=2, dim=-1)
    
        return output



    def forward_in(self, img):

        x = img[0].unsqueeze(0)
        x = torchvision.transforms.functional.resize(x, [int(x.shape[-2]),int(x.shape[-1])])
        x = self._forward_singlescale(x)

        return x
    
    def forward(self, img, rel=None):
        x_out = self.forward_in(img[0])
        y_outs = []

        for i in range(len(rel)):
            y_out = self.forward_in(rel[i])
            y_outs.append(y_out)
        
        loss = 0
        for y_out in y_outs:
            loss += F.mse_loss(x_out, y_out)
            # loss += F.l1_loss(x_out, y_out)
        loss /= len(y_outs)

        loss *= 1000000

        return loss

    def training_step(self, batch, batch_index):
        loss = self(**batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.train_losses[self.current_epoch].append(loss.item())
        return {"loss": loss}

    def validation_step(self, batch, batch_index):
        loss = self(**batch)
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"val_loss": loss}

    # def predict_step(self, batch, batch_index):
    #     _, logits = self(**batch)
    #     return logits
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('avg_val_loss', avg_loss, prog_bar=True, logger=True)
        self.val_losses.append(avg_loss.item())
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            output_dir = './finetune/model'
            model_state_dict = model.state_dict()

            if self.config['model_depth'] == 101 and self.config['withpan']:
                torch.save(model_state_dict, os.path.join(output_dir, "withpan101.pyth"))

            elif self.config['model_depth'] == 101 and not self.config['withpan']:
                torch.save(model_state_dict, os.path.join(output_dir, "notwithpan101.pyth"))

            elif self.config['model_depth'] == 50 and self.config['withpan']:
                torch.save(model_state_dict, os.path.join(output_dir, "withpan50.pyth"))

            else:
                model_state_dict = model.state_dict()
                torch.save(model_state_dict, os.path.join(output_dir, "notwithpan50.pyth"))
            
        output_dir = './finetune/fig'
        self.validation_step_outputs = []
        pdb.set_trace()
        if len(self.train_losses[self.current_epoch]) > 0:
            average_loss = sum(self.train_losses[self.current_epoch]) / len(self.train_losses[self.current_epoch])
            self.train_losses[self.current_epoch] = average_loss
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        loss_plot_filename = os.path.join(output_dir, f'nopan-loss_plot_epoch_{self.current_epoch}.png')
        plt.savefig(loss_plot_filename)
        plt.close()


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
        total_steps = self.config['train_size'] / self.config['bs']
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]
    
if __name__ == '__main__':

    torch.set_float32_matmul_precision('medium')
    warnings.filterwarnings("ignore", category=FutureWarning)
    train_path = './holidays'

    gnd_fn = 'groundtruth.json'

    ucc_data_module = UCC_Data_Module(train_path, gnd_fn)
    dl = ucc_data_module.train_dataloader()

    config = {
        'bs': 1,
        'lr': 1.5e-10,
        'warmup': 0.2,
        'train_size': len(ucc_data_module.train_dataloader()),
        'w_decay': 0.001,
        'n_epochs': 10,
        'reduction_dim': 2048,
        'model_depth': 50,
        'withpan': False,
        'load': False,
        'pre_trained_weight': './weights/CVPR2022_CVNet_R50.pyth',
        # 'Dataset_list': ['roxford5k'],
        'SupG': {'gemp': True, 'sgem': True, 'rgem': True, 'relup': True, 'rerank': True, 'onemeval': False, 'scale_list': 3}
    }

    model = UCC_Classifier(config)

    trainer = pl.Trainer(max_epochs=config['n_epochs'], devices=[0], num_sanity_val_steps=0)

    trainer.fit(model, ucc_data_module)