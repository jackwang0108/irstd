import os
import os.path as osp
import time
import datetime
from argparse import ArgumentParser

import numpy as np
import torch
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import all_models, get_model
from utils.data import *
from utils.loss import SoftLoULoss
from utils.lr_scheduler import *
from utils.evaluation.my_pd_fa import my_PD_FA
from utils.evaluation.TPFNFP import SegmentationMetricTPFNFP
from utils.logger import setup_logger


def parse_args():
    ## ---Setting parameters---
    parser = ArgumentParser(description="Implement of RPCANet")
    parser.add_argument(
        "--base-size", type=int, default=256, help="base size of images"
    )
    parser.add_argument(
        "--crop-size", type=int, default=256, help="crop size of images"
    )
    parser.add_argument(
        "--dataset", type=str, default="MSISTD", help="choose datasets"
    )  #'sirstaug'  'irstd1k' 'nudt' 'MSISTD'
    parser.add_argument(
        "--batch-size", type=int, default=8, help="batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=400, help="number of epochs"
    )  # 50
    parser.add_argument("--warm-up-epochs", type=int, default=0, help="warm up epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gpu", type=str, default="2", help="GPU number")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument(
        "--lr-scheduler", type=str, default="poly", help="learning rate scheduler"
    )
    parser.add_argument(
        "--net-name",
        type=str,
        default="rpcanet",
        choices=all_models,
        help="net name: fcn",
    )  ## Net parameters
    # parser.add_argument('--rank', type=int, default=8,help='rank number')  ## Rank parameters
    parser.add_argument(
        "--save-iter-step", type=int, default=5, help="save model per step iters"
    )  ## Save parameters
    parser.add_argument(
        "--log-per-iter", type=int, default=20, help="interval of logging"
    )
    parser.add_argument("--base-dir", type=str, default="./result/", help="saving dir")
    args = parser.parse_args()

    ## ---Save folders---
    args.time_name = time.strftime("%Y%m%dT%H-%M-%S", time.localtime(time.time()))
    args.folder_name = "{}_{}_{}".format(args.time_name, args.net_name, args.dataset)
    args.save_folder = osp.join(args.base_dir, args.folder_name)

    # seed
    if args.seed != 0:
        set_seeds(args.seed)

    # logger
    args.logger = setup_logger(
        "Robust PCA Network", args.save_folder, 0, filename="log.txt"
    )
    return args


def set_seeds(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.iter_num = 0

        ## dataset
        if args.dataset == "sirstaug":
            trainset = SirstAugDataset(
                base_dir=r"./datasets/sirst_aug", mode="train", base_size=args.base_size
            )  # base_dir=r'E:\ztf\datasets\sirst_aug'
            valset = SirstAugDataset(
                base_dir=r"./datasets/sirst_aug", mode="test", base_size=args.base_size
            )  # base_dir=r'E:\ztf\datasets\sirst_aug'
        elif args.dataset == "MSISTD":
            trainset = MDataset(
                base_dir=r"./datasets/MSISTD", mode="train", base_size=args.base_size
            )
            valset = MDataset(
                base_dir=r"./datasets/MSISTD", mode="test", base_size=args.base_size
            )
        elif args.dataset == "irstd1k":
            trainset = IRSTD1kDataset(
                base_dir=r"./datasets/IRSTD-1k", mode="train", base_size=args.base_size
            )  # base_dir=r'E:\ztf\datasets\IRSTD-1k'
            valset = IRSTD1kDataset(
                base_dir=r"./datasets/IRSTD-1k", mode="test", base_size=args.base_size
            )  # base_dir=r'E:\ztf\datasets\IRSTD-1k'
        elif args.dataset == "nudt":
            trainset = NUDTDataset(
                base_dir=r"./datasets/NUDT-SIRST",
                mode="train",
                base_size=args.base_size,
            )  # base_dir=r'E:\ztf\datasets\IRSTD-1k'
            valset = NUDTDataset(
                base_dir=r"./datasets/NUDT-SIRST", mode="test", base_size=args.base_size
            )  # base_dir=r'E:\ztf\datasets\IRSTD-1k'
        else:
            raise NotImplementedError

        self.train_data_loader = Data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True
        )
        self.val_data_loader = Data.DataLoader(
            valset, batch_size=args.batch_size, shuffle=True
        )
        self.iter_per_epoch = len(self.train_data_loader)
        self.max_iter = args.epochs * self.iter_per_epoch

        ## GPU
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device(
            "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
        )

        ## model
        self.net = get_model(args.net_name)
        """
        for m in self.net.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        """

        # self.net.apply(self.weight_init)
        self.net = self.net.to(self.device)

        ## criterion
        self.softiou = SoftLoULoss()
        self.mse = torch.nn.MSELoss()

        ## lr scheduler
        self.scheduler = LR_Scheduler_Head(
            args.lr_scheduler,
            args.lr,
            args.epochs,
            len(self.train_data_loader),
            lr_step=10,
        )

        ## optimizer
        # self.optimizer = torch.optim.Adagrad(self.net.parameters(), lr=args.lr, weight_decay=1e-4)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)

        ## evaluation metrics
        self.metric = SegmentationMetricTPFNFP(nclass=1)
        self.best_miou = 0
        self.best_fmeasure = 0
        self.eval_loss = 0  # tmp values
        self.miou = 0
        self.fmeasure = 0
        self.eval_my_PD_FA = my_PD_FA()

        ## SummaryWriter
        self.writer = SummaryWriter(log_dir=args.save_folder)
        self.writer.add_text(args.folder_name, "Args:%s, " % args)

        ## log info
        self.logger = args.logger
        self.logger.info(args)
        self.logger.info("Using device: {}".format(self.device))

    def training(self):
        # training step
        start_time = time.time()
        base_log = (
            "Epoch-Iter: [{:d}/{:d}]-[{:d}/{:d}] || Lr: {:.6f} || Loss: {:.4f}={:.4f}+{:.4f} || "
            "Cost Time: {} || Estimated Time: {}"
        )

        train_loss = []
        train_lr = []
        eval_mIoU = []
        eval_fmeasure = []

        for epoch in range(args.epochs):
            # pbar = tqdm(self.train_data_loader)

            losses = []  # debug by LPei
            for i, (data, labels) in enumerate(self.train_data_loader):
                self.net.train()

                self.scheduler(self.optimizer, i, epoch, self.best_miou)

                data = data.to(self.device)

                labels = labels.to(self.device)
                out_D, out_T = self.net(data)

                loss_softiou = self.softiou(out_T, labels)
                loss_mse = self.mse(out_D, data)
                gamma = torch.Tensor([0.1]).to(self.device)
                loss_all = loss_softiou + torch.mul(gamma, loss_mse)

                self.optimizer.zero_grad()
                loss_all.backward()
                self.optimizer.step()

                losses.append(loss_all.item())  # debug by LPei

                self.iter_num += 1

                cost_string = str(
                    datetime.timedelta(seconds=int(time.time() - start_time))
                )
                eta_seconds = ((time.time() - start_time) / self.iter_num) * (
                    self.max_iter - self.iter_num
                )
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                self.writer.add_scalar(
                    "Train Loss/Loss All", np.mean(loss_all.item()), self.iter_num
                )
                self.writer.add_scalar(
                    "Train Loss/Loss SoftIoU",
                    np.mean(loss_softiou.item()),
                    self.iter_num,
                )
                self.writer.add_scalar(
                    "Train Loss/Loss MSE", np.mean(loss_mse.item()), self.iter_num
                )
                self.writer.add_scalar(
                    "Learning rate/",
                    trainer.optimizer.param_groups[0]["lr"],
                    self.iter_num,
                )

                if self.iter_num % self.args.log_per_iter == 0:
                    self.logger.info(
                        base_log.format(
                            epoch + 1,
                            args.epochs,
                            self.iter_num % self.iter_per_epoch,
                            self.iter_per_epoch,
                            self.optimizer.param_groups[0]["lr"],
                            loss_all.item(),
                            loss_softiou.item(),
                            loss_mse.item(),
                            cost_string,
                            eta_string,
                        )
                    )

                if (
                    self.iter_num % args.save_iter_step
                ) == 0 or self.iter_num % self.iter_per_epoch == 0:
                    miou, fmeasure = self.validation(epoch)

                    train_loss.append(np.mean(losses))
                    plt.figure()
                    plt.plot(train_loss)
                    plt.title("train_loss")
                    plt.xlabel("iters")
                    plt.ylabel("loss")
                    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
                    # xTicks = np.arange(0, self.iter_num, 1)
                    # plt.xticks(xTicks)
                    plt.savefig(os.path.join(args.save_folder, f"train_loss.png"))
                    # plt.imshow()
                    plt.close()

                    train_lr.append(self.optimizer.param_groups[0]["lr"])
                    plt.figure()
                    plt.plot(train_lr)
                    plt.title("learning rate")
                    plt.xlabel("iters")
                    plt.ylabel("lr")
                    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
                    # xTicks = np.arange(0, self.iter_num, 1)
                    # plt.xticks(xTicks)
                    plt.savefig(os.path.join(args.save_folder, f"train_lr.png"))
                    # plt.imshow()
                    plt.close()

                    eval_mIoU.append(miou)
                    plt.figure()
                    plt.plot(eval_mIoU)
                    plt.title("eval_mIoU")
                    plt.xlabel("iters")
                    plt.ylabel("mIoU")
                    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
                    # xTicks = np.arange(0, self.iter_num, 1)
                    # plt.xticks(xTicks)
                    plt.savefig(os.path.join(args.save_folder, f"eval_mIoU.png"))
                    # plt.imshow()
                    plt.close()

                    eval_fmeasure.append(fmeasure)
                    plt.figure()
                    plt.plot(eval_fmeasure)
                    plt.title("eval_fmeasure")
                    plt.xlabel("iters")
                    plt.ylabel("fmeasure")
                    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
                    # xTicks = np.arange(0, self.iter_num, 1)
                    # plt.xticks(xTicks)
                    plt.savefig(os.path.join(args.save_folder, f"eval_fmeasure.png"))
                    # plt.imshow()
                    plt.close()

    def validation(self, epoch):
        self.metric.reset()
        # self.eval_my_PD_FA.reset()
        self.net.eval()
        base_log = "Epoch: {:d}, Data: {:s}, mIoU: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f} "
        # base_log = "Data: {:s}, mIoU: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f}, Pd:{:.4f}, Fa:{:.8f} "

        for i, (data, labels) in enumerate(self.val_data_loader):
            with torch.no_grad():
                out_D, out_T = self.net(data.to(self.device))
            out_D, out_T = out_D.cpu(), out_T.cpu()

            loss_softiou = self.softiou(out_T, labels)
            loss_mse = self.mse(out_D, data)
            gamma = torch.Tensor([0.1]).to(self.device)
            loss_all = loss_softiou + torch.mul(gamma, loss_mse)

            self.metric.update(labels, out_T)

        miou, prec, recall, fmeasure = self.metric.get()
        torch.save(self.net.state_dict(), osp.join(self.args.save_folder, "latest.pkl"))
        model_name = "Epoch-%3d_mIoU-%.4f_fmeasure-%.4f_model.pkl" % (
            epoch + 1,
            miou,
            fmeasure,
        )  # .pkl
        if miou > self.best_miou:
            self.best_miou = miou
            torch.save(
                self.net.state_dict(), osp.join(self.args.save_folder, "best.pkl")
            )
            torch.save(
                self.net.state_dict(), osp.join(self.args.save_folder, model_name)
            )
        if fmeasure > self.best_fmeasure:
            self.best_fmeasure = fmeasure

        self.writer.add_scalar("Test/mIoU", miou, self.iter_num)
        self.writer.add_scalar("Test/F1", fmeasure, self.iter_num)
        self.writer.add_scalar("Best/mIoU", self.best_miou, self.iter_num)
        self.writer.add_scalar("Best/Fmeasure", self.best_fmeasure, self.iter_num)

        self.logger.info(
            base_log.format(
                epoch + 1,
                self.args.dataset,
                miou,
                self.best_miou,
                fmeasure,
                self.best_fmeasure,
            )
        )

        return miou, fmeasure


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)
    trainer.training()

    print(
        "Best mIoU: %.5f, Best Fmeasure: %.5f\n\n"
        % (trainer.best_miou, trainer.best_fmeasure)
    )
