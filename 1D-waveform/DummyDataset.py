import numpy as np
import torch.utils.data as data
import librosa


class DummyDataset(data.Dataset):
    def __init__(self, root, partition, protocol_name):
        super(DummyDataset, self).__init__()
        self.root = root
        self.partition = partition

        
        protocol_dir = root.joinpath(protocol_name)
        print('Dummy reading ', protocol_dir)

        protocol_lines = open(protocol_dir).readlines()

        self.features = []
        if self.partition == 'train':
            feature_address = 'ASVspoof2019_LA_train'
        elif self.partition == 'dev':
            feature_address = 'ASVspoof2019_LA_dev'
        elif self.partition == 'eval':
            feature_address = 'ASVspoof2019_LA_eval'   


        for protocol_line in protocol_lines:
            tokens = protocol_line.strip().split(' ')
            # The protocols look like this: 
            #  [0]      [1]       [2]     [3]    [4]
            # LA_0070 LA_D_7622198 -      -     bonafide 
            #    -      file_name  -  attack_id  sys_id

            file_name = tokens[1]
            feature_path = self.root.joinpath(feature_address, 'flac', tokens[1] + '.flac')
            self.features.append((feature_path, file_name))

    def __getitem__(self, index):
        feature_path, file_name  = self.features[index]
        feature, sr = librosa.load(feature_path, sr=16000)
        audio_length = len(feature)
        return file_name, audio_length

    def __len__(self):
        return len(self.features)
    