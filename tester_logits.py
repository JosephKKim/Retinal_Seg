import time
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from tqdm import tqdm
from trainer import Trainer
from utils.helpers import dir_exists, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics_notLogit, count_connect_component
import ttach as tta


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
        if self.CFG.tta:
            self.model = tta.SegmentationTTAWrapper(
                self.model, tta.aliases.d4_transform(), merge_mode='mean')
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=150)
        tic = time.time()
        with torch.no_grad():
            for i, (img, gt) in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                pre = self.model(img)
                loss = self.loss(pre, gt)
                self.total_loss.update(loss.item())
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
                    
                pre_orig = pre
                gt_orig = gt
                img = img[0,0,...]
                # gt
                gt = gt[0,1,...]
                # pre_s = pre[0]
                ### pre -> vessel
                pre = pre[0,1,...]
                
                anomaly_score, prediction = torch.max(pre_orig, dim=1)
                
                
                
                class_mean = np.load(f'stats/DRIVE_mean.npy', allow_pickle=True).item()
                class_var = np.load(f'stats/DRIVE_var.npy', allow_pickle=True).item()
                
                # pre_list = []
                for c in range(2):
                    anomaly_score = torch.where(prediction == c,
                                            (anomaly_score - class_mean[c]) / np.sqrt(class_var[c]),
                                            anomaly_score)
                
                
                
                
                
                mean = torch.mean(anomaly_score)
                print('Mean:', mean.item())

                # compute the standard deviation
                std = torch.std(anomaly_score)
                print('Standard deviation:', std.item())

                # compute the minimum value
                min_value = torch.min(anomaly_score)
                print('Minimum value:', min_value.item())

                # compute the maximum value
                max_value = torch.max(anomaly_score)
                print('Maximum value:', max_value.item())
                
                predict = nn.Softmax(dim = 1)(pre_orig)
                
                
                if self.show:
                    
                    # sigmoid
            
                    
                    
                    predict = torch.sigmoid(pre).cpu().detach().numpy()
                    predict_b = np.where(predict >= self.CFG.threshold, 1, 0)
                    # heatmap = cv2.applyColorMap(np.uint8(255 * predict), cv2.COLORMAP_JET)
                    # print(1)
                    cv2.imwrite(
                        f"save_picture_2class/img{i}_2class.png", np.uint8(img.cpu().numpy()*255))
                    # cv2.imwrite(
                        # f"save_picture_2class/msp{i}.png", np.uint8(MSP[0]*255))
                    cv2.imwrite(
                        f"save_picture_2class/gt{i}.png", np.uint8(gt.cpu().numpy()*255))
                    cv2.imwrite(
                        f"save_picture_2class/pre{i}.png", np.uint8(predict*255))
                    
                    
                    # visualize anomaly score
                    plt.imshow(-anomaly_score.cpu().numpy()[0], cmap = 'jet')
                    plt.colorbar()
                    plt.savefig(f"save_picture_2class/anomaly{i}.png", dpi=300, bbox_inches='tight')
                    
                    plt.close()
                    
                    
                    # save heatmap
                    plt.imshow(predict, cmap='jet')
                    plt.colorbar()

                    plt.savefig(f"save_picture_2class/heatmap{i}.png", dpi=300, bbox_inches='tight')
                    cv2.imwrite(
                        f"save_picture_2class/pre_b{i}.png", np.uint8(predict_b*255))
                    plt.close()
                    # cv2.imwrite(
                    #     f"save_picture_2class/heatmap{i}.png", heatmap)
                    # cv2.imwrite(
                    #     f"save_picture/pre_b{i}.png", np.uint8(predict_b*255))
                if self.CFG.DTI:
                    pre_DTI = double_threshold_iteration(
                        i, pre, self.CFG.threshold, self.CFG.threshold_low, True)
                    self._metrics_update(
                        *get_metrics(predict, gt, predict_b=pre_DTI).values())
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(pre_DTI, gt))
                else:
                    self._metrics_update(
                        *get_metrics(pre, gt, self.CFG.threshold).values())
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(
                            pre, gt, threshold=self.CFG.threshold))
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
        