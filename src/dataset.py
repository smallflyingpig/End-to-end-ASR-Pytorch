import os
import pickle
import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from src.preprocess import zero_padding,target_padding, compute_deltas
import pandas as pd

# TODO : Move this to config
HALF_BATCHSIZE_TIME=800
HALF_BATCHSIZE_LABEL=150



def bucket_data(lens, data_list, bucket_size):
    data = []
    tmp_data = []
    tmp_len = []
    # print("data list len:{}, lens len:{}, bucket_size:{}".format(len(data_list), len(lens), bucket_size))
    for idx, (l,d) in enumerate(zip(lens, data_list)):
        tmp_data.append(d)
        tmp_len.append(l)
        # Half  the batch size if seq too long
        if len(tmp_data)== bucket_size:
            if (bucket_size>=2) and ((max(tmp_len)> HALF_BATCHSIZE_TIME) or (max([len(_d[1]) for _d in tmp_data])>HALF_BATCHSIZE_LABEL)):
                data.append(tmp_data[:bucket_size//2])
                data.append(tmp_data[bucket_size//2:])
            else:
                data.append(tmp_data)
            tmp_data,tmp_len = [],[]
    if len(tmp_data)>0:
        data.append(tmp_data)
    # print("data len:", len(data))
    return data


# Datasets (all datasets work in bucketing style)
# Parameters
#     - file_path    : str, file path to dataset
#     - split        : str, data split (train / dev / test)
#     - max_timestep : int, max len for input (set to 0 for no restriction)
#     - max_label_len: int, max len for output (set to 0 for no restriction)
#     - bucket_size  : int, batch size for each bucket

class TimitDataset(Dataset):
    def __init__(self, file_path, raw_file_path, sets, bucket_size, max_timestep=0, max_label_len=0, raw_wav_data=False):
        self.raw_root = raw_file_path
        self.raw_wav_data = raw_wav_data
        # Open dataset
        x = []
        y = []
        tables = []
        for s in sets:
            with open(os.path.join(file_path,s+'_x.pkl'),'rb') as fp:
                x += pickle.load(fp)
            with open(os.path.join(file_path,s+'_y.pkl'),'rb') as fp:
                y += pickle.load(fp)
            # load data path
            tables.append(pd.read_csv(os.path.join(file_path,s+'.csv')))
        assert len(x)==len(y)
        
        # Sort data w.r.t. length
        self.X = []
        self.Y = []
        sortd_len = [len(t) for t in x]
        sorted_x = [x[idx] for idx in reversed(np.argsort(sortd_len))]
        sorted_y = [y[idx] for idx in reversed(np.argsort(sortd_len))]
        self.table = pd.concat(tables,ignore_index=True).sort_values(by=['length'],ascending=False)

        # Bucketing
        for b in range(int(np.ceil(len(sorted_x)/bucket_size))):
            offset = b*bucket_size
            bound = min((b+1)*bucket_size,len(sorted_x))
            bucket_max_timestep = min(max_timestep,len(sorted_x[offset]))
            self.X.append(zero_padding(sorted_x[offset:bound], bucket_max_timestep))
            bucket_max_label_len = min(max_label_len,max([len(v) for v in sorted_y[offset:bound]]))
            self.Y.append(target_padding(sorted_y[offset:bound], bucket_max_label_len))
        self.iterator = self.prepare_feature if not raw_wav_data else self.prepara_raw_data

        # bucketing data
        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()
        Y = [list(map(int, label.split('_'))) for label in self.table['label'].tolist()]

        # Bucketing, X & X_len is dummy when text_only==True
        self.data = bucket_data(X_lens, list(zip(X,Y)), bucket_size)
        self.transorm = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40, dct_type=2, 
                                                    melkwargs={'hop_length':int(16*10), 'n_fft':int(16*25)})

    
    def prepare_feature(self, index):
        return self.X[index],self.Y[index]

    def transform_raw_data(self, raw_data):
        # (1, T)
        feature = self.transorm(raw_data)
        delta = compute_deltas(feature)
        return torch.cat([feature, delta], dim=1).transpose(1,2).squeeze()

    def post_process(self, data):
        # transform
        x, y = data
        x = [self.transform_raw_data(_x) for _x in x] # (T,D)
        x = pad_sequence(x, batch_first=True).unsqueeze(0) # (1,B,T,D)
        return x, y


    def prepare_raw_data(self, index):
        # Load label
        d = self.data[index]
        x, y = zip(*d)
        y = target_padding(y, max([len(v) for v in y]))
        # Load acoustic feature and pad
        x = [torchaudio.load(os.path.join(self.raw_root,f)).squeeze(0) for f in x]
        return x,y

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.X)


class LibriDataset(Dataset):
    def __init__(self, file_path, raw_file_path, sets, bucket_size, max_timestep=0, max_label_len=0,
        drop=False,text_only=False, raw_wav_data=False):
        # Read file
        self.root = file_path
        self.raw_root = raw_file_path
        self.raw_wav_data = raw_wav_data
        tables = [pd.read_csv(os.path.join(file_path,s+'.csv')) for s in sets]
        self.table = pd.concat(tables,ignore_index=True).sort_values(by=['length'],ascending=False)
        self.text_only = text_only

        # Crop seqs that are too long
        if drop and max_timestep >0 and not text_only:
            self.table = self.table[self.table.length < max_timestep]
        if drop and max_label_len >0:
            self.table = self.table[self.table.label.str.count('_')+1 < max_label_len]

        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()
            
        Y = [list(map(int, label.split('_'))) for label in self.table['label'].tolist()]
        if text_only:
            Y.sort(key=len,reverse=True)

        # Bucketing, X & X_len is dummy when text_only==True
        self.data = bucket_data(X_lens, list(zip(X,Y)), bucket_size)

        self.iterator = self.prepare_data if not raw_wav_data else self.prepare_raw_data
        self.transorm = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40, dct_type=2, 
                                                    melkwargs={'hop_length':int(16*10), 'n_fft':int(16*25)})

    def prepare_data(self, index):
        # Load label
        d = self.data[index]
        x, y = zip(*d)
        y = target_padding(y, max([len(v) for v in y]))
        if self.text_only:
            return y
        
        # Load acoustic feature and pad
        x = [torch.FloatTensor(np.load(os.path.join(self.root,f))) for f in x]
        x = pad_sequence(x, batch_first=True)
        return x,y
    
    def transform_raw_data(self, raw_data):
        # (1, T)
        feature = self.transorm(raw_data)
        delta = compute_deltas(feature)
        return torch.cat([feature, delta], dim=1).transpose(1,2).squeeze()

    @staticmethod
    def get_raw_data_folder(path):
        temp = path
        folder, filename = temp.split('/')
        temp_list = filename.split('-')
        path = '/'.join([folder, temp_list[0], temp_list[1], filename])
        return path

    def post_process(self, data):
        # transform
        x, y = data
        x = [self.transform_raw_data(_x) for _x in x] # (T,D)
        x = pad_sequence(x, batch_first=True).unsqueeze(0) # (1,B,T,D)
        return x, y

    def prepare_raw_data(self, index):
        # Load label
        d = self.data[index]
        x, y = zip(*d)
        y = target_padding(y, max([len(v) for v in y]))
        if self.text_only:
            return y
        
        # Load acoustic feature and pad
        x = [self.get_raw_data_folder(_x) for _x in x]
        x = [torchaudio.load(os.path.join(self.raw_root,f[:-4]+'.flac'))[0].squeeze(0) for f in x]
        return x,y

    def __getitem__(self, index):
        return self.iterator(index)
            
    
    def __len__(self):
        return len(self.data)


def LoadDataset(split, text_only, data_path, raw_data_path, batch_size, max_timestep, max_label_len, use_gpu, n_jobs,
                dataset, train_set, dev_set, test_set, dev_batch_size, decode_beam_size,**kwargs):
    if split=='train':
        bs = batch_size
        shuffle = True
        sets = train_set
        drop_too_long = True
    elif split=='dev':
        bs = dev_batch_size
        shuffle = False
        sets = dev_set
        drop_too_long = True
    elif split=='test':
        bs = 1 if decode_beam_size>1 else dev_batch_size
        n_jobs = 1
        shuffle = False
        sets = test_set
        drop_too_long = False
    elif split=='text':
        bs = batch_size
        shuffle = True
        sets = train_set
        drop_too_long = True
    else:
        raise NotImplementedError
        
    if dataset.upper() == "TIMIT":
        assert not text_only,'TIMIT does not support text only.'
        ds = TimitDataset(file_path=data_path, raw_file_path=raw_data_path, sets=sets, max_timestep=max_timestep, 
                           max_label_len=max_label_len, bucket_size=bs, raw_wav_data=kwargs.get('raw_wav_data',False))
    elif dataset.upper() =="LIBRISPEECH":
        ds = LibriDataset(file_path=data_path, raw_file_path=raw_data_path, sets=sets, max_timestep=max_timestep,text_only=text_only,
                           max_label_len=max_label_len, bucket_size=bs,drop=drop_too_long, raw_wav_data=kwargs.get('raw_wav_data',False))
    else:
        raise ValueError('Unsupported Dataset: '+dataset)

    return  DataLoader(ds, batch_size=1,shuffle=shuffle,drop_last=False,num_workers=n_jobs,pin_memory=use_gpu)


