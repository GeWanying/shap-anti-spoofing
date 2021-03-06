import torch
import argparse
import os
from torch.utils.data.dataloader import DataLoader
import models
import numpy as np
from tqdm import tqdm
import numpy as np
from ASVRawDataset import ASVRawDataset
from pathlib import Path
from scipy.io import savemat
from captum.attr import GradientShap, IntegratedGradients


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TSSDNet-1D model')
    parser.add_argument('--fix', dest='is_fixed_length', action='store_true', help='whether use fixed audio length')      
    parser.add_argument('--model', type=str)
    parser.add_argument('--comment', type=str, default='EXP', help='the folder name of the saved files')
    parser.add_argument('--handler', type=int, default='0', help='1 for only speech and 2 for only non-speech')
    parser.add_argument('--seed', type=int, default=None, help='to get exactly same result for each run')

    # by default, use variable length in evaluation
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

    # path to your data
    args.data = '/path/to/your/LA'

    protocols = {'train_protocol': args.data + '/ASVspoof2019.LA.cm.train.trn_A04.txt',
                'eval_protocol': args.data + '/ASVspoof2019.LA.cm.eval.trl.txt',
    }

    # all obtained SHAP values for will be save under /<args.comment>-mats folder, 
    # each with its original name, like LA_E_3566209.mat
    mats_save_path = os.path.join(args.comment + '-mats')
    if not os.path.exists(mats_save_path):
        os.mkdir(mats_save_path)

    # load pre-trained model
    model = models.SSDNet1D()
    model = model.cuda()
    check_point = torch.load(args.model)
    model.load_state_dict(check_point['model_state_dict'])

    # load audio files provided in the eval_protocol
    eval_dataset = ASVRawDataset(Path(args.data), 'eval', protocols['eval_protocol'], is_fixed_length=args.is_fixed_length, handler=args.handler)
    
    # load audio files provided in the eval_protocol
    # eval_dataset = ASVRawDataset(Path(args.data), 'train', protocols['train_protocol'], is_fixed_length=args.is_fixed_length, handler=args.handler)
    
    # set shuffle=False to have the same utterance when using the same protocol
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    # use SHAP for analysis
    gs = GradientShap(model)

    # use Integrated Gradients method for analysis
    # ig = IntegratedGradients(model)

    for step, (input, file_name, attack_id, target) in tqdm(enumerate(eval_loader)):

        feature = torch.tensor(input).cuda(non_blocking=True)
        # to generate baseline/expection to the input feature
        bsline = torch.zeros(input.shape).cuda(non_blocking=True)

        # spoofed ground truth label, as defined 0 in the dataloader
        t_0 = torch.tensor(0).cuda(non_blocking=True)
        # bona fide ground truth label
        t_1 = torch.tensor(1).cuda(non_blocking=True)

        # calculate SHAP attributions for spoofed class (0) and bona fide class (1)
        attribution_0 = gs.attribute(feature, baselines=bsline, n_samples=20, target=t_0)
        attribution_1 = gs.attribute(feature, baselines=bsline, n_samples=20, target=t_1)

        # calculate Integrated gradients attributions 
        # attribution_0 = ig.attribute(feature, target=t_0)
        # attribution_1 = ig.attribute(feature, target=t_1)


        mdic = {"shap_0": attribution_0.detach().cpu().numpy(), 
                "shap_1": attribution_1.detach().cpu().numpy(), 
                "feature":feature.cpu().numpy(),
            }

        # save the result as MATLAB .mat format
        name = os.path.join(mats_save_path, file_name[0] + ".mat")
        savemat(name, mdic)

        # only save part of the database for analysis,
        # comment below if you want to save SHAP values for all the audio files in the protocol
        if step > 100:
            break
