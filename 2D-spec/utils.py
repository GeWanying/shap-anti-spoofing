import os
import numpy as np
import torch
import shutil
from sklearn.metrics import roc_curve


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def compute_eer(distances, labels):
    # Calculate evaluation metrics
    fprs, tprs, _ = roc_curve(labels, distances)
    eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]
    return eer
