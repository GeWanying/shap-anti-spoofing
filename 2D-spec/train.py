import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import models
from test import cal_roc_eer, asv_cal_socres
import os
import sys
import time
import numpy as np
from pathlib import Path
from ASVRawDataset import ASVRawDataset
import argparse
import utils
import glob
import logging
from get_dataloader import get_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser('TSSDNet-2D model')
    parser.add_argument('--batch_size', type=int, default=8, help='using variable length takes more GPU memory than fixed')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--comment', type=str, default='EXP', help='Comment to describe the saved mdoel')
    parser.add_argument('--seed', type=int, default=789, help='random seed')
    parser.add_argument('--rand', dest='is_rand', action='store_true', help='whether use rand start of input audio')
    parser.add_argument('--fix', dest='is_fixed_length', action='store_true', help='whether use fixed audio length')
    parser.add_argument('--handler', type=int, default='0', help='1 for speech and 2 for non-speech')
    
    # by default, always fix the start point of the audio as 0, using random start point if set True
    parser.set_defaults(is_rand=False)
    # use variable length input, using fixed 6 seconds if set True
    parser.set_defaults(is_fixed_length=False)

    args = parser.parse_args()

    if args.seed:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.comment = 'train-{}-{}'.format(args.comment, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.comment, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.comment, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    model_save_path = os.path.join(args.comment, 'models')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # args.data = '/path/to/your/LA'
    args.data = '/medias/speech/projects/ge/Data/ASV2019/LA'

    protocols = {'train_protocol': args.data + '/ASVspoof2019.LA.cm.train.trn_A04.txt',
                'dev_protocol': args.data + '/ASVspoof2019.LA.cm.dev.trl_A04.txt',
                'eval_protocol': args.data + '/ASVspoof2019.LA.cm.eval.trl.txt',
    }

    train_loader, dev_loader, eval_loader = get_dataloader(args, protocols)

    # weights for CCE loss, the values are approximately the ratio of #bonafide : #spoof
    weights = torch.FloatTensor([0.10165484633, 0.89834515366]).to(device)  # weight used for WCE

    Net = models.SSDNet2D()
    Net = Net.to(device)
    num_total_learnable_params = sum(i.numel() for i in Net.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))
    optimizer = optim.Adam(Net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    print('Training started...')

    num_epoch = 100
    loss_per_epoch = torch.zeros(num_epoch,)
    best_d_eer = [.9, 0]

    time_name = time.ctime()
    time_name = time_name.replace(' ', '_')
    time_name = time_name.replace(':', '_')

    for epoch in range(num_epoch):
        Net.train()
        t = time.time()
        total_loss = 0
        counter = 0
        for batch in train_loader:
            counter += 1
            samples, _, _, labels = batch
            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            preds = Net(samples)
            loss = F.cross_entropy(preds, labels.detach(), weight=weights)

            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()

        loss_per_epoch[epoch] = total_loss/counter

        eval_accuracy, e_probs = asv_cal_socres(eval_loader, Net, device, args.comment, epoch)
        e_eer = cal_roc_eer(e_probs, show_plot=False)

        torch.save({'epoch': epoch, 'model_state_dict': Net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_per_epoch}, os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))

        elapsed = time.time() - t

        print_str = 'Epoch: {}, Elapsed: {:.2f} mins, lr: {:.3f}e-3, Loss: {:.4f}, ' \
                    'eEER: {:.2f}%.'.\
                    format(epoch, elapsed/60, optimizer.param_groups[0]['lr']*1000, total_loss / counter, 
                           e_eer * 100)
        logging.info(print_str)

        scheduler.step()


    logging.info('End of Program.')
