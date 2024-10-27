#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse  # 导入argparse模块，用于解析命令行参数
import os  # 导入os模块，提供与操作系统交互的功能
import sys  # 导入sys模块，提供对解释器相关的功能
from yacs.config import CfgNode as CfgNode  # 从yacs.config导入CfgNode并命名为CfgNode

# Global config object
_C = CfgNode()  # 创建全局配置对象_C

# Example usage:
#   from core.config import cfg
cfg = _C  # 将全局配置对象_C赋值给cfg

_C.MODEL_NAME = ""  # 设置模型名称为空字符串
# ------------------------------------------------------------------------------------ #
# Model options
# ------------------------------------------------------------------------------------ #
_C.MODEL = CfgNode()  # 创建模型配置节点

# Model type
_C.MODEL.TYPE = "RESNET"  # 设置模型类型为RESNET

# Number of weight layers
_C.MODEL.DEPTH = 50  # 设置模型深度为50层

# Loss function (see pycls/models/loss.py for options)
_C.MODEL.LOSSES = CfgNode()  # 创建损失函数配置节点
_C.MODEL.LOSSES.NAME = "cross_entropy"  # 设置损失函数名称为cross_entropy

# ------------------------------------------------------------------------------------ #
# Heads options
# ------------------------------------------------------------------------------------
_C.MODEL.HEADS = CfgNode()  # 创建模型头部配置节点
_C.MODEL.HEADS.NAME = "LinearHead"  # 设置头部名称为LinearHead

# Input feature dimension
_C.MODEL.HEADS.IN_FEAT = 2048  # 设置输入特征维度为2048
# Reduction dimension in head
_C.MODEL.HEADS.REDUCTION_DIM = 2048  # 设置头部的缩减维度为2048

# ------------------------------------------------------------------------------------ #
# Testing options
# ------------------------------------------------------------------------------------ #
_C.TEST = CfgNode()  # 创建测试配置节点

_C.TEST.WEIGHTS = ""  # 设置测试权重文件路径为空字符串
_C.TEST.DATA_DIR = ""  # 设置测试数据目录为空字符串
_C.TEST.DATASET_LIST = ["roxford5k"]  # 设置测试数据集列表，默认包含roxford5k
_C.TEST.SCALE_LIST = 3  # 设置测试时使用的尺度列表
_C.TEST.TOPK_LIST = [400]  # 设置测试时的TopK列表

# ------------------------------------------------------------------------------------ #
# Common train/test data loader options
# ------------------------------------------------------------------------------------ #
_C.DATA_LOADER = CfgNode()  # 创建数据加载器配置节点
# Number of data loader workers per process
_C.DATA_LOADER.NUM_WORKERS = 4  # 设置每个进程的数据加载器工作数为4
# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True  # 设置将数据加载到固定内存中

# ------------------------------------------------------------------------------------ #
# Batch norm options
# ------------------------------------------------------------------------------------ #
_C.BN = CfgNode()  # 创建批量归一化配置节点

# BN epsilon
_C.BN.EPS = 1e-5  # 设置批量归一化的epsilon值

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1  # 设置批量归一化的动量值

# Precise BN stats
_C.BN.USE_PRECISE_STATS = False  # 设置是否使用精确的批量归一化统计
_C.BN.NUM_SAMPLES_PRECISE = 1024  # 设置用于精确统计的样本数

# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = False  # 设置是否将每个块的最终批量归一化的gamma初始化为零

# Use a different weight decay for BN layers
_C.BN.USE_CUSTOM_WEIGHT_DECAY = False  # 设置是否为批量归一化层使用不同的权重衰减
_C.BN.CUSTOM_WEIGHT_DECAY = 0.0  # 设置批量归一化层的权重衰减值

# ------------------------------------------------------------------------------------ #
# CUDNN options
# ------------------------------------------------------------------------------------ #
_C.CUDNN = CfgNode()  # 创建CUDNN配置节点
# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN.BENCHMARK = True  # 设置是否执行基准测试以选择最快的CUDNN算法

# ------------------------------------------------------------------------------------ #
# SuperGlobal options
# ------------------------------------------------------------------------------------ #
_C.SupG = CfgNode()  # 创建SuperGlobal配置节点

_C.SupG.gemp = True  # 设置是否使用gemp
_C.SupG.sgem = True  # 设置是否使用sgem
_C.SupG.rgem = True  # 设置是否使用rgem
_C.SupG.relup = True  # 设置是否使用relup
_C.SupG.rerank = True  # 设置是否使用重新排序
_C.SupG.onemeval = True  # 设置是否使用单次评估

# ------------------------------------------------------------------------------------ #
# Deprecated keys
# ------------------------------------------------------------------------------------ #

_C.register_deprecated_key("PREC_TIME.BATCH_SIZE")  # 注册已废弃的配置键
_C.register_deprecated_key("PREC_TIME.ENABLED")  # 注册已废弃的配置键

def dump_cfg():  # 定义函数，用于将配置保存到输出目录
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)  # 构建配置文件路径
    with open(cfg_file, "w") as f:  # 打开配置文件
        _C.dump(stream=f)  # 将配置写入文件

def load_cfg(out_dir, cfg_dest="config.yaml"):  # 定义函数，从指定的输出目录加载配置
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)  # 构建配置文件路径
    _C.merge_from_file(cfg_file)  # 从文件合并配置

def load_cfg_fom_args(description="Config file options."):  # 定义函数，从命令行参数加载配置
    """Load config from command line arguments and set any specified options."""
    parser = argparse.ArgumentParser(description=description)  # 创建命令行参数解析器
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)  # 添加参数
    if len(sys.argv) == 1:  # 如果没有提供参数
        parser.print_help()  # 打印帮助信息
        sys.exit(1)  # 退出程序
    args = parser.parse_args()  # 解析参数
    _C.merge_from_list(args.opts)  # 从参数列表合并配置
