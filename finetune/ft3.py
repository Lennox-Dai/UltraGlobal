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
import json
import pickle as pkl
import torchvision
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt

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

        # x = torch.stack(x,0)
        x = x.unsqueeze(0)

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
    def __init__(self, in_channels, noise_std=0.01):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        self.g = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv1d(self.inter_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(in_channels)
        self.noise_std = noise_std

    def forward(self, x, epoch):
        batch_size, C, T = x.size()

        noise = torch.randn_like(x) * (self.noise_std / (2**epoch))
        x_noisy = x + noise
        x_noisy = x
        g_x = self.g(x_noisy).view(batch_size, self.inter_channels, -1)
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
        z = self.bn(W_y + x_noisy)

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
        self.nonlocal_block = NonLocalBlock(channel, noise_std=0.001)

    def forward(self, x, epoch):
        pooled_features = []
        for i in range(self.m):
            x1 = self.nonlocal_block(x, epoch=epoch)
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

        for i in range(len(gnd)):
            # pdb.set_trace()
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
        # pdb.set_trace()
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
        # return 
        # pdb.set_trace()
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
        self.train_losses = []
        self.validation_step_outputs = []
        self.best_val_loss = float('inf')
        self.config = config
        self.reduction_dim = config["reduction_dim"]
        self.pan = GLAM()
        self.load = config["load"]
        resnet101 = models.resnet101(pretrained=True)
        resnet50 = models.resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(*list(resnet101.children())[:-2])
        self.resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-2])
        if self.config["model"]=='resnet18':
            self.gemp = gemp(m=self.reduction_dim, channel=512)
            self.head = GlobalHead(512, nc=self.reduction_dim)
        else:
            self.gemp = gemp(m=self.reduction_dim)
            self.head = GlobalHead(2048, nc=self.reduction_dim)
        self.rgem = rgem()
        self.sgem = sgem()
        # self.head = GlobalHead(2048, nc=self.reduction_dim)
        # FIXME loss fun的选择
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')
        self.dropout = nn.Dropout()

        if self.load:

            model_path = '/root/autodl-tmp/SuperGlobal/finetune/model/withpan50.pth'
            model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            self.pan.load_state_dict(model_state_dict['pan_state_dict'])
            self.resnet50.load_state_dict(model_state_dict['resnet50_state_dict'])
            self.gemp.load_state_dict(model_state_dict['gemp_state_dict'])
            self.rgem.load_state_dict(model_state_dict['rgem_state_dict'])
            self.sgem.load_state_dict(model_state_dict['sgem_state_dict'])
            self.head.load_state_dict(model_state_dict['head_state_dict'])
            self.fc.load_state_dict(model_state_dict['fc_state_dict'])

        else:

            self.resnet.eval()
            self.resnet50.eval()
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.resnet50.parameters():
                param.requires_grad = False

        # self.hidden = nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        # self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])

        
        # model_state_dict = {
        #         'pan_state_dict': self.pan.state_dict(),
        #         # 'resnet101_state_dict': self.resnet.state_dict(),
        #         'resnet50_state_dict': self.resnet50.state_dict(),
        #         'gemp_state_dict': self.gemp.state_dict(),
        #         'rgem_state_dict': self.rgem.state_dict(),
        #         'sgem_state_dict': self.sgem.state_dict(),
        #         'head_state_dict': self.head.state_dict(),
        #         'fc_state_dict': self.fc.state_dict(),
        # }
        # output_dir = './finetune/model'
        # torch.save(model_state_dict, os.path.join(output_dir, "model_state.pth"))

    def _forward_singlescale(self, x, gemp=True, rgem=True):
        # pdb.set_trace()
        # output = self.pan(x)
        # output = self.resnet(output)
        output = self.resnet(x)
        # pdb.set_trace()
        # output = output.view(output.shape[0], output.shape[1], -1)
        if rgem and output.shape[2]>2 and output.shape[3]>2:
            output = self.rgem(output)
        output = output.view(output.shape[0], output.shape[1], -1)
        if gemp:
            # pdb.set_trace()
            output = self.gemp(output, self.current_epoch)
            
            # output = output[:, :, :256]
            # output = self.pad_to_size(output, 256)
        else:
            # pdb.set_trace()
            # output = self.head.pool(output)
            stride = output.size(2) // self.reduction_dim
            output = F.avg_pool1d(output, kernel_size=stride, stride=stride)
        
        output = torch.transpose(output, 1, 2)
        output = F.normalize(output, p=2, dim=-1)
        
        # print(output.device)
        output = self.head.fc(output)
        
        return output

    # def pad_to_size(self, input_tensor, target_size):
    #     _, channels, width = input_tensor.shape
    #     # 计算需要填充的大小
    #     padding_size = target_size - width
    #     if padding_size > 0:
    #         # 计算左右两侧的填充大小
    #         pad_left = padding_size // 2
    #         pad_right = padding_size - pad_left
    #         # 填充张量
    #         padded_tensor = F.pad(input_tensor, (pad_left, pad_right), mode='constant', value=0)
    #         return padded_tensor[:, :, :target_size]  # 如果填充后尺寸超过目标大小，进行切片
    #     return input_tensor[:, :, :target_size]  # 如果尺寸已经满足要求，进行切片


    def forward_in(self, img, sgem, gemp, rgem):

        x = img[0].unsqueeze(0)
        x = torchvision.transforms.functional.resize(x, [int(x.shape[-2]),int(x.shape[-1])])
        # pdb.set_trace()
        x = self._forward_singlescale(x, gemp, rgem)
        if sgem:
            x_out = self.sgem(x)
        else:
            x_out = x.unsqueeze(0)
            x_out = torch.mean(x_out, 0)
        
        x_pooled = torch.nn.functional.adaptive_avg_pool1d(x_out, output_size=1)
        x_pooled = x_pooled.squeeze(dim=-1)

        return x_pooled
    
    def forward(self, img, rel=None, gemp=False, rgem=False, sgem=False):
        # pdb.set_trace()
        x_out = self.forward_in(img[0], sgem, gemp, rgem)
        y_outs = []

        for i in range(len(rel)):
            y_out = self.forward_in(rel[i], sgem, gemp, rgem)
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
        if self.current_epoch % 100 == 0:
            self.train_losses.append(loss.item())
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
            if self.config['model'] == 'resnet101':
                model_state_dict = {
                    'pan_state_dict': self.pan.state_dict(),
                    'resnet101_state_dict': self.resnet.state_dict(),
                    'gemp_state_dict': self.gemp.state_dict(),
                    'rgem_state_dict': self.rgem.state_dict(),
                    'sgem_state_dict': self.sgem.state_dict(),
                    'head_state_dict': self.head.state_dict(),
                    # 'fc_state_dict': self.fc.state_dict(),
                }
            else:
                model_state_dict = {
                    'pan_state_dict': self.pan.state_dict(),
                    'resnet50_state_dict': self.resnet50.state_dict(),
                    'gemp_state_dict': self.gemp.state_dict(),
                    'rgem_state_dict': self.rgem.state_dict(),
                    'sgem_state_dict': self.sgem.state_dict(),
                    'head_state_dict': self.head.state_dict(),
                    # 'fc_state_dict': self.fc.state_dict(),
                }
            output_dir = './finetune/model'
            torch.save(model_state_dict, os.path.join(output_dir, "nopan50.pth"))

        output_dir = './finetune/model'
        self.validation_step_outputs = []
        # import pdb
        # pdb.set_trace()
        # 绘制损失图
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
        # self.train_losses = []
        # self.val_losses = []

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
        'reduction_dim': 8,
        'model': 'resnet50',
        'load': False,
        'pretrain_model': './weight/CVPR2022_CVNet_R50.pyth'
    }


    model = UCC_Classifier(config)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # model.to("cpu")

    trainer = pl.Trainer(max_epochs=config['n_epochs'], devices=[0], num_sanity_val_steps=50)
    # trainer = pl.Trainer(max_epochs=config['n_epochs'], devices=None, num_sanity_val_steps=50)
    

    trainer.fit(model, ucc_data_module)