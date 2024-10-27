import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import warnings
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F
from safetensors.torch import save_file
import pdb
import pickle as pkl
import torchvision
from transformers import get_cosine_schedule_with_warmup

import cv2
import numpy as np
import core.transforms as transforms
import torch.utils.data
import core.checkpoint as checkpoint
from model.CVNet_Rerank_model import CVNet_Rerank
from other.GLAM import GLAM
from torchvision import models
from torch.optim import AdamW

_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

class GeneralizedMeanPooling(nn.Module):

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool1d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'

class GeneralizedMeanPoolingP(GeneralizedMeanPooling):

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class GlobalHead(nn.Module):
    def __init__(self, w_in, nc, pp=3):
        super(GlobalHead, self).__init__()
        self.fc = nn.Linear(w_in, nc, bias=True)
        self.pool = GeneralizedMeanPoolingP(norm=pp)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class sgem(nn.Module):

    def __init__(self, ps=10., infinity = True):
        super(sgem, self).__init__()
        self.ps = ps
        self.infinity = infinity
    def forward(self, x):

        x = torch.stack(x,0)

        if self.infinity:
            x = F.normalize(x, p=2, dim=-1) # 3 C
            x = torch.max(x, 0)[0] 
        else:
            gamma = x.min()
            x = (x - gamma).pow(self.ps).mean(0).pow(1./self.ps) + gamma

        return x
    
class rgem(nn.Module):

    def __init__(self, pr=2.5, size = 5):
        super(rgem, self).__init__()
        self.pr = pr
        self.size = size
        self.lppool = nn.LPPool2d(self.pr, int(self.size), stride=1)
        self.pad = nn.ReflectionPad2d(int((self.size-1)//2.))
    def forward(self, x):
        nominater = (self.size**2) **(1./self.pr)
        x = 0.5*self.lppool(self.pad(x/nominater)) + 0.5*x
        return x

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        self.g = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv1d(self.inter_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        batch_size, C, T = x.size()

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = self.bn(W_y + x)

        return z
    
# class gemp(nn.Module):

#     def __init__(self, p=4.6, eps=1e-8, m=2048):
#         super(gemp, self).__init__()
#         self.m = m
#         self.p = [p] * m
#         self.eps = [eps] * m

#     def forward(self, x):
#         pooled_features = []
#         for i in range(self.m):
#             x_clamped = x.clamp(self.eps[i]).pow(self.p[i])
#             pooled = torch.nn.functional.adaptive_avg_pool1d(x_clamped, 1).pow(1. / self.p[i])
#             pooled_features.append(pooled)

#         concatenated_features = torch.cat(pooled_features, dim=-1)

#         return concatenated_features

class gemp(nn.Module):
    def __init__(self, p=4.6, eps=1e-8, channel=2048, m=2048):
        super(gemp, self).__init__()
        self.m = m
        self.p = [p] * m
        self.eps = [eps] * m
        self.nonlocal_block = NonLocalBlock(channel)
        # self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        pooled_features = []
        for i in range(self.m):
            x1 = self.nonlocal_block(x)
            x_clamped = x1.clamp(self.eps[i]).pow(self.p[i])
            pooled = torch.nn.functional.adaptive_avg_pool1d(x_clamped, 1).pow(1. / self.p[i])
            pooled_features.append(pooled)

        concatenated_features = torch.cat(pooled_features, dim=-1)

        return concatenated_features



def setup_model():

    print("=> creating CVNet_Rerank model")
    # pdb.set_trace()
    model = CVNet_Rerank(RESNET_DEPTH=50, REDUCTION_DIM=2048, relup=True)
    print(model)
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)

    return model

class DataSet(torch.utils.data.Dataset):

    def __init__(self, data_path, dataset, fn):
        assert os.path.exists(
            data_path), "Data path '{}' not found".format(data_path)
        self._data_path, self._dataset, self._fn = data_path, dataset, fn
        self._label_path = os.path.join(data_path, "data")
        self._construct_db()

    def _construct_db(self):
        # Compile the split data path
        self._db = []
        self._dbl = []
        if self._dataset in ['oxford5k', 'roxford5k', 'paris6k', 'rparis6k']:
            with open(os.path.join(self._data_path, self._dataset, self._fn), 'rb') as fin:
                gnd = pkl.load(fin)
            with open(os.path.join(self._label_path, self._dataset, "relabel_data.pkl"), 'rb') as f:
                loaded_label = pkl.load(f)

            for i in range(len(gnd["qimlist"])):
                im_fn = gnd["qimlist"][i]
                im_path = os.path.join(
                    self._data_path, self._dataset, "jpg", im_fn+".jpg")
                self._db.append({"im_path": im_path})
            for i in range(len(gnd["imlist"])):
                im_fn = gnd["imlist"][i]
                im_path = os.path.join(
                    self._data_path, self._dataset, "jpg", im_fn+".jpg")
                self._db.append({"im_path": im_path})

            for i in range(len(loaded_label["qimlist"])):
                # pdb.set_trace()
                self._dbl.append({"label": loaded_label["qimlist"][i]})
            for i in range(len(loaded_label["imlist"])):
                self._dbl.append({"label": loaded_label["imlist"][i]})

            # pdb.set_trace()

        else:
            assert() # Unsupported dataset

    def _prepare_im(self, im):
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # Color normalization
        im = transforms.color_norm(im, _MEAN, _SD)
        return im

    # def __getitem__(self, index):
    #     # Load the image
    #     try:
    #         im = cv2.imread(self._db[index]["im_path"])
    #         im_np = im.astype(np.float32, copy=False)
    #         im_list.append(im_np)
    #         im_list = []
    #         label_list = []
      
    #     except:
    #         print('error: ', self._db[index]["im_path"])

    #     for idx in range(index):
    #         im_list.append(self._prepare_im(im_list[idx]))
    #     # return {"img": im_list, "label": }
    
    def __getitem__(self, index):
        im_list = []
        label_list = []
        try:
            im = cv2.imread(self._db[index]["im_path"])
            if im is None:
                raise ValueError("Image not found or unable to load.")
            im_np = im.astype(np.float32, copy=False)
            im_list.append(im_np)
            label_list.append(self._dbl[index]["label"])
        except Exception as e:
            print('error:', self._db[index]["im_path"], e)

        for idx in range(len(im_list)):
            im_list[idx] = self._prepare_im(im_list[idx])
        # return 
        return {"img": im_list, "label": label_list}
        

    def __len__(self):
        return len(self._db)

class UCC_Data_Module(pl.LightningDataModule):

  def __init__(self, train_path, dataset, gnd_fn, attributes, batch_size: int=1, shuffle: bool=False):
    super().__init__()
    self.train_path = train_path
    self.dataset = dataset
    self.gnd_fn = gnd_fn
    self.attributes = attributes
    self.batch_size = batch_size
    self.shuffle = shuffle

  def train_dataloader(self):
    return self._construct_loader(self.train_path, self.dataset, self.gnd_fn, self.batch_size, self.shuffle)

  # def train_dataloader(self):
  #   return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

  # def val_dataloader(self):
  #   return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

  # def predict_dataloader(self):
  #   return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
  
  def _construct_loader(self, _DATA_DIR, dataset_name, fn, batch_size, shuffle):
    dataset = DataSet(_DATA_DIR, dataset_name, fn)
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
        self.validation_step_outputs = []
        self.best_val_loss = float('inf')
        self.config = config
        self.n_labels = config["n_labels"]
        self.reduction_dim = config["reduction_dim"]
        self.pan = GLAM()
        resnet101 = models.resnet101(pretrained=True)
        resnet50 = models.resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(*list(resnet101.children())[:-2])
        self.resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-2])
        self.resnet.eval()
        self.resnet50.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet50.parameters():
            param.requires_grad = False
        self.gemp = gemp(m=self.reduction_dim)
        self.rgem = rgem()
        self.sgem = sgem()
        self.head = GlobalHead(2048, nc=self.reduction_dim)
        self.fc = nn.Linear(256, self.n_labels)
        # self.hidden = nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        # self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])
        
        # FIXME loss fun的选择
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')
        self.dropout = nn.Dropout()

    def _forward_singlescale(self, x, gemp=True, rgem=True):
        # pdb.set_trace()
        output = self.pan(x)
        # pdb.set_trace()
        output = self.resnet(output)
        # output = output.view(output.shape[0], output.shape[1], -1)
        if rgem and output.shape[2]>2 and output.shape[3]>2:
            output = self.rgem(output)
        output = output.view(output.shape[0], output.shape[1], -1)
        if gemp:
            output = self.gemp(output)
        else:
            output = self.head.pool(output)
        
        output = torch.transpose(output, 1, 2)
        output = F.normalize(output, p=2, dim=-1)
        
        # print(output.device)
        output = self.head.fc(output)
        
        return output

    def forward(self, img, scale=1, label=None, gemp=True, rgem=True):
        assert scale in [1, 3, 5], "scale must be in [1, 3, 5]"
        feature_list = []
        if scale == 1:
            scale_list = [1.]
        elif scale == 3:
            scale_list = [0.7071, 1., 1.4142]
        elif scale == 5:
            scale_list = [0.5, 0.7071, 1., 1.4142, 2.]
        else:
            raise 
        for _, scl in zip(range(scale), scale_list):
            # pdb.set_trace()
            x = torchvision.transforms.functional.resize(img[0], [int(img[0].shape[-2]*scl),int(img[0].shape[-1]*scl)])
            x = self._forward_singlescale(x, gemp, rgem)
            feature_list.append(x)
        if sgem:
            x_out = self.sgem(feature_list)
        else:
            x_out = torch.stack(feature_list, 0)
            x_out = torch.mean(x_out, 0)

        # TODO 到这里为止，保存！

        x_pooled = torch.nn.functional.adaptive_avg_pool1d(x_out, output_size=1)
        x_pooled = x_pooled.squeeze(dim=-1)
        # pdb.set_trace()
        x_cl = self.fc(x_pooled)

        # logits = self.classifier(pooled_output)
        loss = 0
        # pdb.set_trace()
        # TODO 硬编码了，bs只有1
        if label is not None:
            loss = self.loss_func(x_cl, label[0])
        return loss

    def training_step(self, batch, batch_index):
        loss = self(**batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        # pdb.set_trace()
        return {"loss": loss, "labels": batch['label']}

    def validation_step(self, batch, batch_index):
        loss = self(**batch)
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"val_loss": loss, "labels": batch['label']}

    # def predict_step(self, batch, batch_index):
    #     _, logits = self(**batch)
    #     return logits
    
    def on_validation_epoch_end(self):
        # TODO 这里写保存
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('avg_val_loss', avg_loss, prog_bar=True, logger=True)
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            
            model_state_dict = {
                'pan_state_dict': self.pan.state_dict(),
                'resnet_state_dict': self.resnet.state_dict(),
                'gemp_state_dict': self.gemp.state_dict(),
                'rgem_state_dict': self.rgem.state_dict(),
                'sgem_state_dict': self.sgem.state_dict(),
                'head_state_dict': self.head.state_dict(),
                'fc_state_dict': self.fc.state_dict(),
            }
            output_dir = './finetune/model'
            torch.save(model_state_dict, os.path.join(output_dir, "model_state.pth"))

        self.validation_step_outputs = []

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
        total_steps = self.config['train_size'] / self.config['bs']
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]
    

if __name__ == '__main__':

    # with open(r"D:\上海cv\SuperGlobal\revisitop\data\roxford5k\relabel_data.pkl", 'rb') as f:
    #     loaded_label = pkl.load(f)
    # print(loaded_label)
    # pdb.set_trace()
    # loaded_label.cuda()
    # print(loaded_label)

    warnings.filterwarnings("ignore", category=FutureWarning)
    train_path = './revisitop'
    dataset = "roxford5k"

    attributes = ['all_souls', 'oxford', 'ashmolean', 'balliol', 'bodleian', 'christ_church', 'cornmarket', 'hertford', 'keble', 'magdalen', 'pitt_rivers', 'radcliffe_camera', 'jesus', 'new', 'oriel', 'trinity', 'worcester']

    if dataset == 'roxford5k':
        gnd_fn = 'gnd_roxford5k.pkl'
    elif dataset == 'rparis6k':
        gnd_fn = 'gnd_rparis6k.pkl'
    else:
        assert dataset

    ucc_data_module = UCC_Data_Module(train_path, dataset, gnd_fn, attributes)
    dl = ucc_data_module.train_dataloader()

    config = {
        'n_labels': len(attributes),
        'bs': 1,
        'lr': 1.5e-6,
        'warmup': 0.2,
        'train_size': len(ucc_data_module.train_dataloader()),
        'w_decay': 0.001,
        'n_epochs': 100,
        'reduction_dim': 256 
    }


    model = UCC_Classifier(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # model.to("cpu")

    trainer = pl.Trainer(max_epochs=config['n_epochs'], devices=[0], num_sanity_val_steps=50)
    # trainer = pl.Trainer(max_epochs=config['n_epochs'], devices=None, num_sanity_val_steps=50)
    

    trainer.fit(model, ucc_data_module)