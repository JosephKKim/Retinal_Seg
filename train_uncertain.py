import argparse
from bunch import Bunch
from loguru import logger
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset
from trainer_uncertain import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch
import torch
import torch.nn as nn

def main(CFG, data_path, batch_size, with_val=True, dataset_name=None, exp_id = None):
    # print("dataset_name: ", dataset_name)
    seed_torch()
    # print(data_path)
    if 1:
        train_dataset = vessel_dataset(data_path, mode="training")
        # val_dataset = vessel_dataset(
        #     data_path, mode="training", split=0.9, is_val=True)
        # val_loader = DataLoader(
        #     val_dataset, batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
        test_dataset = vessel_dataset(data_path, mode="test")
        val_loader = DataLoader(test_dataset, 1,
                                 shuffle=False, num_workers=16, pin_memory=True)
    else:
        train_dataset = vessel_dataset(data_path, mode="training")

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=36, pin_memory=True, drop_last=False,
                         prefetch_factor=32)

    logger.info('The patch number of train is %d' % len(train_dataset))
    model = get_instance(models, 'model', CFG)
    # var mdoel 
    var_model = get_instance(models, 'var_model', CFG)
    logger.info(f'\n{model}\n')

    # checkpoint = torch.load('/home/kkh/research/seg/FR-UNet/saved/FR_UNet/230318151027/checkpoint-epoch10.pth')
    model = nn.DataParallel(model.cuda())
    var_model = nn.DataParallel(var_model.cuda())
    # model.load_state_dict(checkpoint['state_dict'])
    loss = get_instance(losses, 'loss', CFG)
    trainer = Trainer(
        model=model,
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
        val_loader=val_loader if with_val else None,
        dataset_name = dataset_name,
        exp_id = exp_id
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="dataset/DRIVE", type=str,
                        help='the path of dataset')
    parser.add_argument('-bs', '--batch_size', default=512,
                        help='batch_size for trianing and validation')
    parser.add_argument("--val", help="split training data for validation",
                        required=False, default=True, action="store_true")
    parser.add_argument('-id', '--exp_id', type = str, required = False, default=None)
    parser.add_argument('-cfg', '--cfg_path', type = str, required = True, default='config.yaml')
    args = parser.parse_args()
    if args.dataset_path.endswith('/'):
        args.dataset_path = args.dataset_path[:-1]
    # print("dataset_name: ", args.dataset_path.split('/')[-1].strip())
    with open(args.cfg_path, encoding='utf-8') as file:
        CFG = Bunch(safe_load(file))
    main(CFG, args.dataset_path, args.batch_size, args.val, args.dataset_path.split('/')[-1], args.exp_id)
