import os
import time
from datetime import datetime
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from loguru import logger
from torch.utils import tensorboard
from tqdm import tqdm
from utils.visualization import visualize_gt, visualize_original_img, visualize_dis_out, visualize_prediction_init, visualize_prediction_var
from utils.helpers import dir_exists, get_instance, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component
import ttach as tta
import torch.nn as nn


class Trainer:
    def __init__(self, model, CFG=None, loss=None, train_loader=None, val_loader=None, dataset_name = None, exp_id = None):
        # added var_model here
        
        self.CFG = CFG
        if self.CFG.amp is True:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.loss = loss
        self.model = model
        
        # self.model = nn.DataParallel(model.cuda())
        self.mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = get_instance(
            torch.optim, "optimizer", CFG, self.model.parameters())
        self.dataset_name = dataset_name
        # added var_model optimizer
        
        self.exp_id = exp_id
        self.lr_scheduler = get_instance(
            torch.optim.lr_scheduler, "lr_scheduler", CFG, self.optimizer)
        start_time = datetime.now().strftime('%y%m%d%H%M%S')
        self.checkpoint_dir = os.path.join(
            CFG.save_dir, self.CFG['model']['type'], start_time, dataset_name, exp_id)
        self.writer = tensorboard.SummaryWriter(self.checkpoint_dir)
        dir_exists(self.checkpoint_dir)
        cudnn.benchmark = True
        self.best_auc = 0
        self.sampling_iter = 10

    def train(self):
        best_val_auc = 0
        for epoch in range(1, self.CFG.epochs + 1):
            print(epoch,self.CFG.val_per_epochs)
            print(self.val_loader)
            print(self.val_loader is not None and epoch % self.CFG.val_per_epochs == 0)
            self._train_epoch(epoch)

            if self.val_loader is not None and epoch % self.CFG.val_per_epochs == 0:
                results = self._valid_epoch(epoch, self.dataset_name)
                if best_val_auc <= results['AUC']:
                    print("new best model!")
                    best_val_auc = results['AUC']
                    # save weight
                    self._save_best_checkpoint(epoch)
                logger.info(f'## Info for epoch {epoch} ## ')
                for k, v in results.items():
                    logger.info(f'{str(k):15s}: {v}')
            if epoch % self.CFG.save_period == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        wrt_mode = 'train'
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=160)
        tic = time.time()
        for img, gt in tbar:
            self.data_time.update(time.time() - tic)
            img = img.cuda(non_blocking=True)
            # print("train img shape: ", img.shape)
            gt = gt.cuda(non_blocking=True)
            self.optimizer.zero_grad()
            
            if self.CFG.amp is True:
                with torch.cuda.amp.autocast(enabled=True):
                    pre, latent_loss = self.model(img)
                    loss = self.loss(pre, gt) + latent_loss
                    # print("latent loss: ",latent_loss)
                self.scaler.scale(loss.mean()).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pre, latent_loss = self.model(img)
                loss = self.loss(pre, gt) + latent_loss
                loss.mean().backward()
                self.optimizer.step()
            
            
            self.total_loss.update(loss.mean().item())
            self.batch_time.update(time.time() - tic)
            
            pred_sigmoid = torch.sigmoid(pre).clone()
            
            """ Sampling """
            with torch.no_grad():
                for kk in range(self.sampling_iter):
                    output , _ = self.model(img)
                    pred_sigmoid = torch.cat((pred_sigmoid, torch.sigmoid(output)), 1)
            
            
            
            var_map = torch.var(pred_sigmoid, 1, keepdim=True)
            var_map = (var_map - var_map.min()) / (var_map.max() - var_map.min() + 1e-8)
            mean_map = torch.mean(pred_sigmoid, 1, keepdim=True)
            
            

            
            

            """Train var approx net"""
            
           
                
                
            # var_map_approx=self.var_model(img, mean_map1)
            # var_map_approx=F.upsample(var_map_approx, size=(var_map_real.shape[2], var_map_real.shape[3]), mode='bilinear', align_corners=True)
            # consist_loss = self.mse_loss(torch.sigmoid(var_map_approx), var_map_real)
            # consist_loss.backward()
            # self.var_model_optimizer.step()
            
            folder_path = os.path.join('visual_map_final_retinal', str(self.exp_id), 'vis_e', str(epoch))
            # folder_path = 'visual_map_final/vis_e' + str(epoch) + '/'
            
            if epoch % 5 == 0:  # only save the first batch for every 5 epoch 
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                # visualize_prediction_sample(torch.sigmoid(init_pred),folder_path)
                visualize_prediction_var(torch.sigmoid(var_map), folder_path)
                visualize_prediction_init(torch.sigmoid(mean_map), folder_path)
                # visualize_dis_out(torch.sigmoid(var_map_approx), folder_path)
                visualize_gt(gt, folder_path)
                visualize_original_img(img, folder_path)
                    
            


            self._metrics_update(
                *get_metrics(pre, gt, threshold=self.CFG.threshold).values())
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average,
                    self.data_time.average))
            tic = time.time()
        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], epoch)
        self.lr_scheduler.step()

# for validation

    def _valid_epoch(self, epoch, dataset_name):
        logger.info('\n###### EVALUATION ######')
        
        self.model.eval()
        wrt_mode = 'val'
        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            for img, gt in tbar:
                img = img.cuda(non_blocking=True)
                # print(img.shape)
                gt = gt.cuda(non_blocking=True)
                if self.CFG.amp is True:
                    with torch.cuda.amp.autocast(enabled=True):
                        predict, latent_loss = self.model(img)
                        loss = self.loss(predict, gt) + latent_loss.mean()
                else:
                    predict, latent_loss = self.model(img)
                    loss = self.loss(predict, gt) + latent_loss.mean()
                    
                if dataset_name == "DRIVE":
                    H, W = 584, 565
                elif dataset_name == "CHASEDB1":
                    H, W = 960, 999
                elif dataset_name == "DCA1":
                    H, W = 300, 300

                if not dataset_name == "CHUAC":
                    img = TF.crop(img, 0, 0, H, W)
                    gt = TF.crop(gt, 0, 0, H, W)
                    predict = TF.crop(predict, 0, 0, H, W)
                
                print("the size of image for validation is ", H, W)
                self.total_loss.update(loss.item())
                self._metrics_update(
                    *get_metrics(predict, gt, threshold=self.CFG.threshold).values())
                tbar.set_description(
                    'EVAL ({})  | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f} Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |'.format(
                        epoch, self.total_loss.average, *self._metrics_ave().values()))
                self.writer.add_scalar(
                    f'{wrt_mode}/loss', self.total_loss.average, epoch)
                
            
        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        log = {
            'val_loss': self.total_loss.average,
            **self._metrics_ave()
        }
        return log

    def _save_checkpoint(self, epoch):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.CFG
        }
        filename = os.path.join(self.checkpoint_dir,
                                f'checkpoint-epoch{epoch}.pth')
        
        ## For predictive model
        
        
        logger.info(f'Saving a checkpoint: {filename} ...')
        torch.save(state, filename)
        return filename
    
    def _save_best_checkpoint(self, epoch):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.CFG
        }
        filename = os.path.join(self.checkpoint_dir,
                                f'checkpoint-best_val.pth')
        
        ## For predictive model
       
        logger.info(f'Saving a checkpoint: {filename} ...')
        torch.save(state, filename)
        
        return filename

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.auc = AverageMeter()
        self.f1 = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.pre = AverageMeter()
        self.iou = AverageMeter()
        self.CCC = AverageMeter()

    def _metrics_update(self, auc, f1, acc, sen, spe, pre, iou):
        self.auc.update(auc)
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.pre.update(pre)
        self.iou.update(iou)

    def _metrics_ave(self):

        return {
            "AUC": self.auc.average,
            "F1": self.f1.average,
            "Acc": self.acc.average,
            "Sen": self.sen.average,
            "Spe": self.spe.average,
            "pre": self.pre.average,
            "IOU": self.iou.average
        }
