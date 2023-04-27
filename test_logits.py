import argparse
import torch
from bunch import Bunch
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import models
from dataset_new import vessel_dataset
from tester_logits import Tester
from utils import losses
from utils.helpers import get_instance


def main(data_path, weight_path, CFG, show):
    checkpoint = torch.load(weight_path)
    CFG_ck = checkpoint['config']
    test_dataset = vessel_dataset(data_path, mode="test")
    test_loader = DataLoader(test_dataset, 1,
                             shuffle=False,  num_workers=16, pin_memory=True)
    model = get_instance(models, 'model', CFG)
    loss = get_instance(losses, 'loss', CFG_ck)
    test = Tester(model, loss, CFG, checkpoint, test_loader, data_path, show)
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
