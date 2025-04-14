import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion
from models.decom import CTDN ,CTDNTrainer
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
from models.decom import State1Net ,State1Net_T
def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Latent-Retinex Diffusion Models')
    parser.add_argument("--config", default='unsupervised.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--mode', type=str, default='training', help='training or evaluation')
    
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--image_folder", default='results_uie/', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    parser.add_argument('--state_option', default='', type=str,
                        help='option training state1 state2 or encoder-decoder')
    
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    
    
    # 设置 PyTorch 随机种子
    torch.manual_seed(42)

    # 如果使用了GPU，也要为所有的GPU设置种子
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # 如果有多个GPU

    args, config = parse_args_and_config()
    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.train_dataset))
    DATASET = datasets.__dict__[config.data.type](config)
    print(DATASET)
    # create model
    # print("=> creating denoising-diffusion model...")
    # diffusion = DenoisingDiffusion(args, config)
    # diffusion.train(DATASET)
    # print(type(DATASET))
    if args.state_option == '' : 
        args.state_option = None 
    
    

    # only_training_ED : True 训练state1 
    # only_training_ED : False 是训练编码器 解码器
    state1 = State1Net_T(config,args.state_option,channels=64, epochs=3000, learning_rate=1e-5,only_training_ED=True,
                        ed_path=r'ckpt/encoder-decoder/model_checkpoint_1227-UIEBD-ED.pth.tar')
    
 
    # state1.train_model(DATASET,save_path=r"ckpt/stage1/model_checkpoint_0104-LSUI-APDN")  # 训练模型
    state1.train_model(DATASET,save_path=r"ckpt/stage1/model_checkpoint_25_0324-UIEBD-APDN-fft")  # 训练模型
    
    

# 初始化 CTDN 模型
    # ctdn = CTDN(channels=64).to('cuda')

    # # # 初始化训练器
    # trainer = CTDNTrainer(ctdn, device='cuda', epochs=50, lr=1e-4, lambda_1=1.0, lambda_2=0.1, lambda_3=0.1)
    
    # # # 加载数据
    # train_loader, val_loader = DATASET.get_loaders()
    
# train_loader, val_loader = DATASET.get_loaders()
    # 开始训练
    # trainer.train(train_loader, val_loader)
if __name__ == "__main__":
    main()
