import numpy as np
import torch.utils.data as data
import librosa
from Speech_silence_vad import silence_handler


class ASVRawDataset(data.Dataset):
    def __init__(self, root, partition, protocol_name, is_rand=False, is_fixed_length=True, handler=0):
        super(ASVRawDataset, self).__init__()
        self.root = root
        self.partition = partition
        self.is_rand = is_rand
        self.handler = handler
        self.max_len = 6
        self.is_fixed_length = is_fixed_length

        self.sysid_dict = {
            'bonafide': 1,  # bonafide speech
            'spoof': 0, # Spoofed signal
        }
        
        protocol_dir = protocol_name
        print('Reading ', protocol_dir)
        if self.is_rand:
            print('Using randomly select sequence')
        if self.is_fixed_length:
            print('... with max length ', self.max_len)
        else:
            print('... with variable length')
        protocol_lines = open(protocol_dir).readlines()

        self.features = []
        if self.partition == 'train':
            feature_address = 'ASVspoof2019_LA_train/flac'
        elif self.partition == 'dev':
            feature_address = 'ASVspoof2019_LA_dev/flac'
        elif self.partition == 'eval':
            feature_address = 'ASVspoof2019_LA_eval/flac' 

        for protocol_line in protocol_lines:
            tokens = protocol_line.strip().split(' ')
            # The protocols look like this: 
            #  [0]      [1]       [2][3]  [4]
            # LA_0070 LA_D_7622198 - - bonafide 

            file_name = tokens[1]
            attack_id = tokens[3]
            feature_path = self.root.joinpath(feature_address, tokens[1] + '.flac')
            sys_id = self.sysid_dict[tokens[4]]
            self.features.append((feature_path, file_name, attack_id, sys_id))

    def load_feature(self, feature_path):
        feature, sr = librosa.load(feature_path, sr=16000)
        if self.handler == 1:
                # speech
            feature = silence_handler(feature, sr, flag_output=self.handler)
        elif self.handler == 2:
                # non-speech
            feature = silence_handler(feature, sr, flag_output=self.handler)
        if self.is_fixed_length:
            fix_len = self.max_len * sr
            feature = np.tile(feature, int((fix_len) // len(feature)) + 1)
            if self.is_rand:
                total_length = feature.shape[0]
                start = np.random.randint(0, total_length - fix_len + 1)
                feature = feature[start:start+fix_len]
            else:
                feature = feature[:fix_len]
            return feature
        else:
            return feature

    def __getitem__(self, index):
        feature_path, file_name, attack_id, sys_id = self.features[index]
        feature = self.load_feature(feature_path)
        return feature, file_name, attack_id, sys_id

    def __len__(self):
        return len(self.features)
    
