import time
import cv2
import os
from PIL import ImageFile
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from loguru import logger
from tqdm import tqdm
from trainer import Trainer
from utils.helpers import dir_exists, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics_nosig, count_connect_component
import ttach as tta

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.p = 0.1
            m.train()
           
            

class Tester(Trainer):
    def __init__(self, model, loss, CFG, checkpoint, test_loader, dataset_path, show=False):
        # super(Trainer, self).__init__()
        self.loss = loss
        self.CFG = CFG
        self.test_loader = test_loader
        self.model = nn.DataParallel(model.cuda())
        
        self.dataset_path = dataset_path
        self.show = show
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.show:
            dir_exists("save_picture")
            remove_files("save_picture")
        cudnn.benchmark = True


    def test(self):
        save_path = os.path.join('uncer', 'output')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        if self.CFG.tta:
            self.model = tta.SegmentationTTAWrapper(
                self.model, tta.aliases.d4_transform(), merge_mode='mean')
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=150)
        tic = time.time()
        with torch.no_grad():
            for i, (img, gt) in enumerate(tbar):
                
                save_path_vessel = os.path.join(save_path, 'vessel') 
                if not os.path.exists(save_path_vessel):
                        os.makedirs(save_path_vessel)

                save_path_mean = os.path.join(save_path, 'vessel_mean') 
                if not os.path.exists(save_path_mean):
                    os.makedirs(save_path_mean)
                # sample here to get uncertainty
                pre_list = []
                
                """get one prediction"""
                
                self.data_time.update(time.time() - tic)
                
                img = img.cuda(non_blocking=True)
                original_img = img
                gt = gt.cuda(non_blocking=True)
                original_gt = gt
                pre, latent_loss = self.model(img)
                original_pre = pre
                loss = self.loss(pre, gt) + latent_loss 
                self.total_loss.update(loss.mean().item())
                self.batch_time.update(time.time() - tic)

                if self.dataset_path.endswith("DRIVE"):
                    H, W = 584, 565
                    # H, W = 592, 592
                elif self.dataset_path.endswith("CHASEDB1"):
                    H, W = 960, 999
                elif self.dataset_path.endswith("DCA1"):
                    H, W = 300, 300

                if not self.dataset_path.endswith("CHUAC"):
                    img = TF.crop(img, 0, 0, H, W)
                    gt = TF.crop(gt, 0, 0, H, W)
                    pre = TF.crop(pre, 0, 0, H, W)
                img = img[0,0,...]
                gt = gt[0,0,...]
                pre = pre[0,0,...]
                
                """MC-Sampling: 50 times"""
                loss_op = 100000
                aleatoric_op = 0
                sample_iter = 50
                enable_dropout(self.model)
                for k in range(sample_iter):
                    res, _ = self.model(original_img)
                    pre_list.append(res.detach())
                    
                    # temperature scaling111
                    res = res.sigmoid()
                    aleatoric_temp=-(res*torch.log(res+1e-8))
                    
                    check_loss=F.binary_cross_entropy_with_logits(aleatoric_temp, original_gt.sigmoid() , reduce='none')
                    # aleatoric_value=aleatoric_temp.sum(dim=1).mean()
                    if check_loss<loss_op:
                        loss_op = check_loss
                        aleatoric_op=aleatoric_temp
                        res_op=res
                
                
                res_op = F.upsample(res_op, size=[H, W], mode='bilinear', align_corners=False)
                res_op = res_op.sigmoid().data.cpu().numpy().squeeze()
                res_op = 255 * (res_op - res_op.min()) / (res_op.max() - res_op.min() + 1e-8)
                
                cv2.imwrite(save_path_vessel + f"res_op{i}.png", res_op)
                #aleatoric_op is the aleatoric uncertainty.
                aleatoric_map=aleatoric_op
                aleatoric_map = F.upsample(aleatoric_map, size=[H, W], mode='bilinear', align_corners=False)
                aleatoric_map = aleatoric_map.sigmoid().data.cpu().numpy().squeeze()
                aleatoric_map= (aleatoric_map - aleatoric_map.min()) / (aleatoric_map.max() - aleatoric_map.min() + 1e-8)
                
                
                
                
                ##### PATH #####
                save_path_var = os.path.join(save_path, 'var_maps_aleatoric')
                if not os.path.exists(save_path_var):
                    os.makedirs(save_path_var)
                
                fig = plt.figure()
                heatmap = plt.imshow(aleatoric_map, cmap='viridis')
                fig.colorbar(heatmap)
                fig.savefig(save_path_var + f"/aleatoric_map{i}.png")
                plt.close()
                vessel_preds = torch.sigmoid(pre_list[0]).clone()
                for iter in range(1, sample_iter):
                    vessel_preds = torch.cat((vessel_preds, torch.sigmoid(pre_list[iter])), 1)
                mean_map = torch.mean(vessel_preds, 1, keepdim=True)
                save_map=mean_map
                
                ##### PATH #####
                # save_map = F.upsample(save_map, size=[H, W], mode='bilinear', align_corners=False)
                save_map = TF.crop(save_map, 0, 0, H, W)
                save_map_eval = torch.where(save_map > 0.5, 1, 0)
                pre_sigmoid = pre.sigmoid()
                cnt1 = (save_map > 0.5).sum().item()
                cnt2 = (pre_sigmoid > 0.5).sum().item()
                print("numbers: ", cnt1, cnt2)
                
                # 이미 sigmoid 적용됨...
                save_map_tensor = save_map
                save_map = save_map.sigmoid().data.cpu().numpy().squeeze()
                # save_map_tensor = save_map_tensor.sigmoid()
                
                save_map = 255 * (save_map - save_map.min()) / (save_map.max() - save_map.min() + 1e-8)
                # save_map_tensor = 255 * (save_map_tensor - save_map_tensor.min()) / (save_map_tensor.max() - save_map_tensor.min() + 1e-8)
                cv2.imwrite(save_path_mean + '/mean_map' + f'{i}.png', save_map)
                save_map_eval = save_map_eval.data.cpu().numpy().squeeze()
                save_map_eval = np.uint8(save_map_eval * 255)
                cv2.imwrite(save_path_mean + f'/mean_map_b{i}.png', save_map_eval)
                """total ucnertainty"""
                total_uncertainty_temp=-(mean_map*torch.log(mean_map+ 1e-8))
                total_uncertainty=total_uncertainty_temp
                total_uncertainty = F.upsample(total_uncertainty, size=[H, W], mode='bilinear', align_corners=False)
                total_uncertainty = total_uncertainty.sigmoid().data.cpu().numpy().squeeze()
                total_uncertainty = (total_uncertainty - total_uncertainty.min()) / (
                            total_uncertainty.max() - total_uncertainty.min() + 1e-8)
                
                ##### PATH #####
                save_path_var = os.path.join(save_path, 'var_maps_total')
                if not os.path.exists(save_path_var):
                    os.makedirs(save_path_var)

                fig = plt.figure()
                heatmap = plt.imshow(total_uncertainty, cmap='viridis')
                fig.colorbar(heatmap)
                fig.savefig(save_path_var + f"/total_uncertainty{i}.png")
                plt.close()
                """epistemic uncertainty"""
                temp=total_uncertainty-aleatoric_map
                episilon=abs(np.min(temp[temp<0]))
                temp[temp<0]=0

                epistemic_uncertainty = temp
                ##### PATH #####
                save_path_var = os.path.join(save_path, 'var_maps_epistemic')
                if not os.path.exists(save_path_var):
                    os.makedirs(save_path_var)

                fig = plt.figure()
                heatmap = plt.imshow(epistemic_uncertainty, cmap='viridis')
                fig.colorbar(heatmap)
                
                fig.savefig(save_path_var + f"/epistemic_uncertainty{i}.png")
                plt.close()
                
                
                
                
                
                if self.show:
                    
                    predict = torch.sigmoid(pre).cpu().detach().numpy()
                    predict_b = np.where(predict >= self.CFG.threshold, 1, 0)
                    cv2.imwrite(
                        f"save_picture/img{i}.png", np.uint8(img.cpu().numpy()*255))
                    cv2.imwrite(
                        f"save_picture/gt{i}.png", np.uint8(gt.cpu().numpy()*255))
                    cv2.imwrite(
                        f"save_picture/pre{i}.png", np.uint8(predict*255))
                    cv2.imwrite(
                        f"save_picture/pre_b{i}.png", np.uint8(predict_b*255))

                if self.CFG.DTI:
                    pre_DTI = double_threshold_iteration(
                        i, pre, self.CFG.threshold, self.CFG.threshold_low, True)
                    self._metrics_update(
                        *get_metrics(pre, gt, predict_b=pre_DTI).values())
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(pre_DTI, gt))
                # TODO: check the size of modified pre and gt 
                else:
                    self._metrics_update(
                        *get_metrics_nosig(save_map_tensor, gt, 0.427).values())
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(
                            pre, gt, threshold=self.CFG.threshold))
                    # self._metrics_update(
                    #     *get_metrics(save_map_tensor, gt, self.CFG.threshold, predict_b=save_map_eval).values())
                    # if self.CFG.CCC:
                    #     self.CCC.update(count_connect_component(
                    #         save_map_tensor, gt, threshold=self.CFG.threshold))
                tbar.set_description(
                    'TEST ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                        i, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
                tic = time.time()
        logger.info(f"###### TEST EVALUATION ######")
        logger.info(f'test time:  {self.batch_time.average}')
        logger.info(f'     loss:  {self.total_loss.average}')
        if self.CFG.CCC:
            logger.info(f'     CCC:  {self.CCC.average}')
        for k, v in self._metrics_ave().items():
            logger.info(f'{str(k):5s}: {v}')
        