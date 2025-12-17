import os
from datetime import datetime
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from torchvision import models
from models.KM import KM
from models.KM_c import KMC
from trainer.KMC_trainer import KMCTrainer
from trainer.dataLoaders import ImageTextDataLoader
from trainer.KM_trainer import KMTrainer, SampleTrainer
from utils.loss import compute_lm_loss, compute_recon_loss, compute_cb_loss
from utils.metric.metrics import compute_scores
from utils.optimizers import build_optimizer, build_lr_scheduler
from utils.tokenizers_m import Tokenizer, MixTokenizer
from utils.tool import load_json_args, setup_seed
from torchvision.models import ResNet50_Weights

tokenizers_fn = {'ori': Tokenizer, 'mix': MixTokenizer}
loss_fn = {'lm': compute_lm_loss, 'recon': compute_recon_loss,'cb':compute_cb_loss}
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="scipy")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module='multiprocessing')
    args = load_json_args("config/mimic_cxr.json")
    torch.cuda.set_device(int(args["cuda"]))
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    

    # -------------------------------
    # 设置种子，确保实验可复现
    # -------------------------------
    if args["seed"] == -1:
        args["seed"] = np.random.randint(0, 23333)

    # print(args)
    print(f"seed:{args["seed"]}")
    setup_seed(args["seed"])

    # -------------------------------
    # 创建分词器，处理文本,可以使用单数据集或混合数据集新建词表
    # -------------------------------
    tokenizer = tokenizers_fn[args['tokenizer']](args)
    print('count of tokens', len(tokenizer.token2idx))

    # torch.autograd.set_detect_anomaly(True)
    # -------------------------------
    # 创建DataLoader
    # -------------------------------
    train_dataloader = ImageTextDataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = ImageTextDataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = ImageTextDataLoader(args, tokenizer, split='test', shuffle=False)
    model = KM(args, tokenizer).cuda()
    # -------------------------------
    # get function handles of loss and metrics
    # -------------------------------
    criterion = loss_fn[args["loss_fn"]]
    metrics = compute_scores
    # -------------------------------
    # build optimizer, learning rate scheduler
    # -------------------------------
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer, len(train_dataloader))

    # -------------------------------
    # 训练模型，首先定义参数
    # -------------------------------
    kwarg = {"model": model, "criterion": criterion, "metric_ftns": metrics, "optimizer": optimizer, "args": args,
             "lr_scheduler": lr_scheduler, "train_dataloader": train_dataloader, "val_dataloader": val_dataloader,
             "test_dataloader": test_dataloader,"tokenizer":tokenizer}

    # -------------------------------
    # 构建存放结果的文件夹，文件夹按时间命名
    # -------------------------------
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args["result_dir"] = os.path.join(args["result_dir"], current_time)
    args["record_dir"] = os.path.join(args["record_dir"], current_time)

    # -------------------------------
    # 选择任务
    # -------------------------------
    if args["task"] == 'train':
        trainer = KMTrainer(**kwarg)
        trainer.train()
    elif args["task"] == 'sample':
        trainer = SampleTrainer(**kwarg)
        trainer.train()
    else:
        pass