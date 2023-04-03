import argparse
from bunch import Bunch
from loguru import logger
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch


def main(CFG, data_path, batch_size, with_val=True):
    seed_torch()
    print(data_path)
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
        train_dataset, batch_size, shuffle=True, num_workers=36, pin_memory=True, drop_last=True,
                         prefetch_factor=32)

    logger.info('The patch number of train is %d' % len(train_dataset))
    model = get_instance(models, 'model', CFG)
    logger.info(f'\n{model}\n')
    loss = get_instance(losses, 'loss', CFG)
    trainer = Trainer(
        model=model,
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
        val_loader=val_loader if with_val else None
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="/home/lwt/data_pro/vessel/DRIVE", type=str,
                        help='the path of dataset')
    parser.add_argument('-bs', '--batch_size', default=128,
                        help='batch_size for trianing and validation')
    parser.add_argument("--val", help="split training data for validation",
                        required=False, default=False, action="store_true")
    args = parser.parse_args()

    with open('config.yaml', encoding='utf-8') as file:
        CFG = Bunch(safe_load(file))
    main(CFG, args.dataset_path, args.batch_size, args.val)
