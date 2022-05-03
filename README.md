# shap-anti-spoofing
This repository includes the code to reproduce our paper [Explainable deepfake and spoofing detection: an attack analysis using SHapley Additive exPlanations](https://arxiv.org/pdf/2202.13693.pdf) accepted in The Speaker and Language Recognition Workshop (Speaker Odyssey 2022).

It is also related to our previous work [Explaining deep learning models for spoofing and deepfake detection with SHapley Additive exPlanations](https://arxiv.org/pdf/2110.03309.pdf) accepeted in ICASSP 2022.

### Dependencies
Codes were tested using a GeForce RTX 3090 GPU with CUDA Version==11.2:
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

```
Then:
```
pip install -r requirements.txt
```
We also used MATLAB R2020a for plotting.

### Dataset
The ASVspoof 2019 database can be downloaded from [here](https://datashare.ed.ac.uk/handle/10283/3336)

The extracted data should be orginased as:
* LA/
   * ASVspoof2019_LA_dev/flac/...
   * ASVspoof2019_LA_eval/flac/...
   * ASVspoof2019_LA_train/flac/...
   * ASVspoof2019.LA.cm.dev.trl.txt
   * ASVspoof2019.LA.cm.eval.trl.txt
   * ASVspoof2019.LA.cm.train.trn.txt
   * ASVspoof2019.LA.cm.train.trn_A01.txt (uploaded in protocols/)
   * ASVspoof2019.LA.cm.train.trn_A02.txt (uploaded in protocols/)
   * ASVspoof2019.LA.cm.eval.trl.txt (provided in the database)
   * ...

Please change the `args.data` defined in train/save.py to `'/path/to/your/LA'`.

### Usage
#### Plot the examples

To have a look at the examples shown in the paper, go `'Plot/'` and run plot_shap_1D.m and plot_shap_2D.m with MATLAB.

#### Calculate SHAP values using the pre-trained models

You can either calculate SHAP values based on 1D-waveform input or 2D-spec input by:
```
python save.py --model=pre-trained-models/A04.pth --comment=A04
```
The obtained SHAP values will be saved in `<AUDIO_ID>.mat` format under `'A04-mats/'`. If you wish to calculate SHAP values for A05 attack, first change the corresponding protocol in `train_protocol` in the code, then run the command with the replaced attack type.

#### Train your own models
For both models, run:
```
python train.py
```
Variable length input is used by default, use `--fix` to use fixed 6s length input. Also, please notice the `train_protocol` defined in the code if you want to train the model based on any particular attack. The corresponding protocols are uploaded in `'protocols/'`.

#### Citation
If you find this repository useful, please consider citing:
```
@inproceedings{ge22_odyssey,
  author={Wanying Ge and Massimiliano Todisco and Nicholas Evans},
  title={{Explainable deepfake and spoofing detection: an attack analysis using SHapley Additive exPlanations}},
  year=2022,
  booktitle={The Speaker and Language Recognition Workshop (To appear)},
}
```
and
```
@inproceedings{ge22_icassp,
  author={Wanying Ge and Jose Patino and Massimiliano Todisco and Nicholas Evans},
  title={{Explaining deep learning models for spoofing and deepfake detection with SHapley Additive exPlanations}},
  year=2022,
  booktitle={ICASSP 2022 (To appear)},
}
```
#### Acknowledgement
This work is supported by the ExTENSoR project funded by the French Agence Nationale de la Recherche (ANR).

Codes are based on the implementations of [end-to-end-synthetic-speech-detection](https://github.com/ghuawhu/end-to-end-synthetic-speech-detection), [project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts) and [Captum](https://captum.ai/).
