import os
import pandas as pd
from sklearn.model_selection import train_test_split

import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
torchaudio.set_audio_backend("sox_io")


class DonateACryDataset(Dataset):
    def __init__(self, wav_dirs: list, train=False, clip=None, drop_hungry=None, random_state=0):
        super(DonateACryDataset, self).__init__()
        columns = [
            'instance_id',
            'timestamp',
            'version',
            'gender',
            'age',
            'target',
            'waveform',
            'sample_rate'
        ]
        self.data = pd.DataFrame(columns=columns)
        
        for dir in wav_dirs:
            self.add_files_from_dir(dir)

        if drop_hungry:
            random_hungry_samples = self.data[self.data.target=='hu'].sample(frac=drop_hungry, random_state=random_state)
            self.data.drop(random_hungry_samples.index, inplace=True)
        
        self.normalize_features()
        
        self.target_encoding = { tag: i for i, tag in enumerate(self.data.target.unique()) }
        self.target_decoding = { i: tag for tag, i in self.target_encoding.items() }
        self.encode_target()

        self.split_data(train, random_state=random_state)

        if clip:
            self.data = self.data.iloc[:clip]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x_labels = self.data.columns.drop('target')
        return self.data.iloc[index]['waveform'], self.data.iloc[index]['target']

    @staticmethod
    def parse_filename(filename):
        return (filename[:36], *filename[37:-4].split('-'))

    def normalize_features(self, merge_coldhot=True):
        # make age readable

        # make target readable
        readable_targets = {
            'hu': 'hungry',
            'bu': 'needs burping',
            'bp': 'belly pain',
            'dc': 'discomfort',
            'ti': 'tired',
        }
        if merge_coldhot:
            readable_targets['ch'] = 'discomfort'
        self.data.target.replace(readable_targets, inplace=True)
        
        # make timestamp readable
        
        # normalize waveform tensors to be the same length
        length = lambda x: x.shape[1]
        max_length = self.data.waveform.apply(length).max()
        padding = lambda x: (0, max_length-length(x))
        add_padding = lambda x: F.pad(x, padding(x), 'constant', 0)
        self.data.loc[:, 'waveform'] = self.data.waveform.apply(add_padding)

    def split_data(self, train, random_state):
        split = train_test_split(self.data, train_size=0.8, stratify=self.data.target, random_state=random_state)
        if train:
            self.data = split[0]
        else:
            self.data = split[1]

    def encode_target(self):
        self.data.target.replace(self.target_encoding, inplace=True)

    def decode_target(self):
        self.data.target.replace(self.target_decoding, inplace=True)
    
    def value_counts(self):
        return self.data.target.replace(self.target_decoding).value_counts()

    def add_files_from_dir(self, containing_dir: str):
        for dirpath, _, filenames in os.walk(containing_dir):
            for filename in filenames:
                try:
                    if filename.endswith('.wav'):
                        self.data.loc[len(self.data)] = [
                            *self.parse_filename(filename),
                            *torchaudio.load(os.path.join(dirpath, filename))
                        ]
                except ValueError:
                    print(f'skipping {filename}')
