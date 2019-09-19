import torch
import torchvision
from torch.utils.data import Dataset
import os
import os.path as osp
import numpy as np 
import soundfile as sf
import scipy 

def read_list(filename):
    with open(filename, "r") as fp:
        data = fp.readlines()
        data = [_l.strip() for _l in data]
    return data


class TIMIT(Dataset):
    def __init__(self, data_root, datalist_root, train=True, 
        fs=16000, cw_len=200, cw_shift=10, rand_amp_fact=0.2, oversampling=800):
        super(TIMIT, self).__init__()
        split = "train" if train else "test"
        self.data_root = data_root
        self.data_list = read_list(osp.join(datalist_root, "TIMIT_"+split+".scp"))
        self.label_dict = np.load(osp.join(datalist_root, "TIMIT_labels.npy")).item()
        self.iterator = self.prepara_train_data if train else self.prepare_test_data
        self.wlen, self.wshift = int(fs*cw_len/1000.00), int(fs*cw_shift/1000.00)
        self.rand_amp_fact = rand_amp_fact
        self.oversampling = oversampling if train else 1

    @staticmethod
    def preprocess(wav_data, wrd_data):
        wav_data=wav_data.astype(np.float32)
        # signal noormallization
        wav_data=wav_data/np.max(np.abs(wav_data))
        # remove silences
        beg_sig=int(wrd_data[0].split(' ')[0])
        end_sig=int(wrd_data[-1].split(' ')[1])
        wav_data=wav_data[beg_sig:end_sig]
        return wav_data, wrd_data

    def __len__(self):
        return len(self.data_list)*self.oversampling

    def prepara_train_data(self, index):
        index = index % len(self.data_list)
        filename = self.data_list[index]
        [wav_data, fs] = sf.read(osp.join(self.data_root, filename))
        wrd_data = read_list(osp.join(self.data_root, osp.splitext(filename)[0]+".wrd"))
        wav_data, wrd_data = self.preprocess(wav_data, wrd_data)
        # random chunk
        rand_amp_arr = np.random.uniform(1.0-self.rand_amp_fact, 1.0+self.rand_amp_fact, 1)[0]
        wav_len = wav_data.shape[0]
        wav_beg = np.random.randint(int(wav_len-self.wlen-1))
        # data augmentation
        wav_data = wav_data[wav_beg:wav_beg+self.wlen]*rand_amp_arr
        label = self.label_dict[filename]
        return wav_data, label

    def prepare_test_data(self, index):
        filename = self.data_list[index]
        [wav_data, fs] = sf.read(osp.join(self.data_root, filename))
        wrd_data = read_list(osp.join(self.data_root, osp.splitext(filename)[0]+".wrd"))
        wav_data, wrd_data = self.preprocess(wav_data, wrd_data)
        wav_len = wav_data.shape[0]
        chunk_num = (wav_len-self.wlen)//self.wshift+1
        wav_data_chunk = np.stack([wav_data[beg*self.wshift:beg*self.wshift+self.wlen] for beg in range(chunk_num)])
        # data augmentation
        label = np.array([self.label_dict[filename]]*chunk_num)
        
        return wav_data_chunk, label

    def __getitem__(self, index):
        return self.iterator(index)





