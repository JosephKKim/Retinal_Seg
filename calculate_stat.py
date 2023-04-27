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
import argparse
from bunch import Bunch
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import models
from dataset_new import vessel_dataset
from tester_logits import Tester
from utils import losses
from utils.helpers import get_instance


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
        pred_list = None
        max_class_mean = {}
        print("Calculating statistics...")
        with torch.no_grad():
            for i, (img, gt) in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                
                
                
                
                # 나온 logit가 pre임
                pre = self.model(img)
                # [1,2,48,48]
                
                
                if pred_list is None:
                    pred_list =pre.data.cpu()
                else:
                    pred_list = torch.cat((pred_list, pre.cpu()), 0)
                # memory를 위해 지워주고
                del pre
                
                
                if i == len(tbar)-1:
                    pred_list = pred_list.transpose(1,3)
                    pred_list, prediction = pred_list.max(3)
                    
                    
                    class_max_logits = []
                    mean_dict, var_dict = {}, {}
                    
                    for c in range(2):
                        max_mask = pred_list[prediction==c]
                        
                        class_max_logits.append(max_mask)
                        
                        mean = class_max_logits[c].mean(dim=0)
                        var = class_max_logits[c].var(dim=0)
                        
                        mean_dict[c] = mean.item()
                        var_dict[c] = var.item()
                    
                    print(f"class mean: {mean_dict}")
                    print(f"class var: {var_dict}")
                    np.save('stats/DRIVE_mean.npy', mean_dict)
                    np.save('stats/DRIVE_var.npy', var_dict)
                # loss = self.loss(pre, gt)
                
                
                # self.total_loss.update(loss.item())
                # self.batch_time.update(time.time() - tic)

                # if self.dataset_path.endswith("DRIVE"):
                #     H, W = 584, 565
                #     # H, W = 592, 592
                # elif self.dataset_path.endswith("CHASEDB1"):
                #     H, W = 960, 999
                # elif self.dataset_path.endswith("DCA1"):
                #     H, W = 300, 300

                # if not self.dataset_path.endswith("CHUAC"):
                #     img = TF.crop(img, 0, 0, H, W)
                #     gt = TF.crop(gt, 0, 0, H, W)
                #     pre = TF.crop(pre, 0, 0, H, W)
                    
                # pre_orig = pre
                # gt_orig = gt
                # img = img[0,0,...]
                # # gt
                # gt = gt[0,1,...]
                # # pre_s = pre[0]
                # pre = pre[0,1,...]
                # maxlogit, argmax = torch.max(pre_orig, dim =1)
                # maxlogit = maxlogit.float()
                # argmax = argmax.float()
                
                
                
                ######## show라는 argument를 주었을 떄에만 작동하는 부분 #######
                # if self.show:
                    
                #     # softmax
                #     # predict = torch.softmax(pre_s, dim = 0).cpu().detach().numpy()
                    
                #     # vessel after softmax 
                #     # predict = predict[1]
                    
                #     # sigmoid
                    
                #     # maxlogit
                #     # maxlogit, MSP = torch.max(pre_orig, dim =1)
                    
                #     # MSP = MSP.cpu().detach().numpy()
                #     predict = torch.softmax(pre_orig).cpu().detach.numpy()
                #     # predict = torch.sigmoid(pre).cpu().detach().numpy()
                #     predict_b = np.where(predict >= self.CFG.threshold, 1, 0)
                #     # heatmap = cv2.applyColorMap(np.uint8(255 * predict), cv2.COLORMAP_JET)
                #     # print(1)
                #     cv2.imwrite(
                #         f"save_picture_2class/img{i}_2class.png", np.uint8(img.cpu().numpy()*255))
                #     # cv2.imwrite(
                #         # f"save_picture_2class/msp{i}.png", np.uint8(MSP[0]*255))
                #     cv2.imwrite(
                #         f"save_picture_2class/gt{i}.png", np.uint8(gt.cpu().numpy()*255))
                #     cv2.imwrite(
                #         f"save_picture_2class/pre{i}.png", np.uint8(predict*255))
                    
                #     # save heatmap
                #     plt.imshow(predict, cmap='jet')
                #     plt.colorbar()

                #     plt.savefig(f"save_picture_2class/heatmap{i}.png", dpi=300, bbox_inches='tight')
                #     cv2.imwrite(
                #         f"save_picture_2class/pre_b{i}.png", np.uint8(predict_b*255))
                #     plt.close()
                #     # cv2.imwrite(
                #     #     f"save_picture_2class/heatmap{i}.png", heatmap)
                #     # cv2.imwrite(
                #     #     f"save_picture/pre_b{i}.png", np.uint8(predict_b*255))
                # if self.CFG.DTI:
                #     pre_DTI = double_threshold_iteration(
                #         i, pre, self.CFG.threshold, self.CFG.threshold_low, True)
                #     self._metrics_update(
                #         *get_metrics(predict, gt, predict_b=pre_DTI).values())
                #     if self.CFG.CCC:
                #         self.CCC.update(count_connect_component(pre_DTI, gt))
                # else:
                #     self._metrics_update(
                #         *get_metrics(pre, gt, self.CFG.threshold).values())
                #     if self.CFG.CCC:
                #         self.CCC.update(count_connect_component(
                #             pre, gt, threshold=self.CFG.threshold))
                # tbar.set_description(
                #     'TEST ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                #         i, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
                # tic = time.time()
                ######## show라는 argument를 주었을 떄에만 작동하는 부분 #######
        # logger.info(f"###### TEST EVALUATION ######")
        # logger.info(f'test time:  {self.batch_time.average}')
        # logger.info(f'     loss:  {self.total_loss.average}')
        # if self.CFG.CCC:
        #     logger.info(f'     CCC:  {self.CCC.average}')
        # for k, v in self._metrics_ave().items():
        #     logger.info(f'{str(k):5s}: {v}')
            
            
            
def main(data_path, weight_path, CFG, show):
    checkpoint = torch.load(weight_path)
    CFG_ck = checkpoint['config']
    sml_dataset = vessel_dataset(data_path, mode="training")
    sml_loader = DataLoader(sml_dataset, 512,
                             shuffle=False,  num_workers=16, pin_memory=True)
    model = get_instance(models, 'model', CFG)
    loss = get_instance(losses, 'loss', CFG_ck)
    test = Tester(model, loss, CFG, checkpoint, sml_loader, data_path, show)
    test.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset_path", default="/home/kkh/research/seg/FR-UNet/dataset/DRIVE", type=str,
                        help="the path of dataset")
    parser.add_argument("-wp", "--wetght_path", default="/home/kkh/research/seg/FR-UNet/saved/FR_UNet_2class/230406222009/DRIVE/2class_BCE/checkpoint-best_val.pth", type=str,
                        help='the path of wetght.pt')
    parser.add_argument("--show", help="save predict image",
                        required=False, default=False, action="store_true")
    parser.add_argument('-cfg', '--cfg_path', type = str, required = True, default='config.yaml')
    args = parser.parse_args()
    with open(args.cfg_path, encoding="utf-8") as file:
        CFG = Bunch(safe_load(file))
    # print(args.wetght_path)
    # print(CFG)
    main(args.dataset_path, args.wetght_path, CFG, args.show)