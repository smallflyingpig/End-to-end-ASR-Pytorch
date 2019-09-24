import sys
sys.path.insert(0, '..')
from src.preprocess import extract_feature,encode_target
from joblib import Parallel, delayed
import argparse
import os 
from pathlib import Path
from tqdm import tqdm
import pickle
import pandas as pd

def boolean_string(s):
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Preprocess program for TIMIT dataset.')
parser.add_argument('--data_path', type=str, default='/media/ubuntu/Elements/dataset/TIMIT_lower', help='Path to raw TIMIT dataset')
parser.add_argument('--feature_type', default='mfcc_torch', type=str, help='Feature type ( mfcc / fbank )', required=False)
parser.add_argument('--feature_dim', default=13, type=int, help='Dimension of feature', required=False)
parser.add_argument('--apply_delta', default=True, type=boolean_string, help='Append Delta', required=False)
parser.add_argument('--apply_delta_delta', default=True, type=boolean_string, help='Append Delta Delta', required=False)
parser.add_argument('--apply_cmvn', default=True, type=boolean_string, help='Apply CMVN on feature', required=False)
parser.add_argument('--output_path', default='/media/ubuntu/Elements/dataset/TIMIT_feature', type=str, help='Path to store output', required=False)
parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)
parser.add_argument('--target', default='subword', type=str, help='Learning target ( phoneme / char / subword / word )', required=False)
parser.add_argument('--n_tokens', default=1000, type=int, help='Vocabulary size of target', required=False)
paras = parser.parse_args()

def read_text(file,target):
    labels = []
    if target == 'phoneme':
        with open(file.replace('.wav','.phn'),'r') as f:
            for line in f:
                labels.append(line.replace('\n','').split(' ')[-1])
    elif target in ['char','subword','word']:
        with open(file.replace('.wav','.wrd'),'r') as f:
            for line in f:
                labels.append(line.replace('\n','').split(' ')[-1])
        if target =='char':
            labels = [c for c in ' '.join(labels)]
        elif target == 'subword':
            labels = ' '.join(labels)
        else:
            pass
    else:
        raise ValueError('Unsupported target: '+target)
    return labels

sets = ['train', 'test']
encode_table = None
output_dir = None
dim = paras.feature_dim*(1+paras.apply_delta+paras.apply_delta_delta)
# BPE training
if paras.target == 'subword':
    # Setup path
    output_dir = os.path.join(paras.output_path,'_'.join(['timit',str(paras.feature_type)+str(dim),str(paras.target)+str(paras.n_tokens)]))
    if not os.path.exists(output_dir):os.makedirs(output_dir)
    bpe_dir = os.path.join(output_dir,'bpe')
    if not os.path.exists(bpe_dir):os.makedirs(bpe_dir)

    print('')
    print('Pretrain BPE for subword unit.')
    print('Data sets :')
    for idx,s in enumerate(sets):
        print('\t',idx,':',s)
    bpe_tr = input('Please enter the index for training sets for BPE (seperate w/ space): ')
    bpe_tr = [sets[int(t)] for t in bpe_tr.split(' ')]
    print(bpe_tr)
    tr_txt = []
    for s in bpe_tr:
        todo = list(Path(os.path.join(paras.data_path,s)).rglob(r"*.[wW][aA][vV]"))
        todo = [_t for _t in todo if len(str(_t).split('.'))<3] # filter out the file with ext .wav.wav)
        tr_txt+=Parallel(n_jobs=paras.n_jobs)(delayed(read_text)(str(file),target=paras.target) for file in todo)
    # print(len(tr_txt), tr_txt[0])
    with open(os.path.join(bpe_dir,'train.txt'),'w') as f:
        for s in tr_txt:f.write(s+'\n')
    # Train BPE
    from subprocess import call
    call(['spm_train',
          '--input='+os.path.join(bpe_dir,'train.txt'),
          '--model_prefix='+os.path.join(bpe_dir,'bpe'),
          '--vocab_size='+str(paras.n_tokens),
          '--character_coverage=1.0'
        ])
    # Encode data
    if not os.path.exists(os.path.join(bpe_dir,'raw')):os.makedirs(os.path.join(bpe_dir,'raw'))
    if not os.path.exists(os.path.join(bpe_dir,'encode')):os.makedirs(os.path.join(bpe_dir,'encode'))
    for s in sets:
        todo = list(Path(os.path.join(paras.data_path,s)).rglob("*.[wW][aA][vV]"))
        todo = [_t for _t in todo if len(str(_t).split('.'))<3] # filter out the file with ext .wav.wav)
        txts = Parallel(n_jobs=paras.n_jobs)(delayed(read_text)(str(file),target=paras.target) for file in todo)
        with open(os.path.join(bpe_dir,'raw',s+'.txt'),'w') as f:
            for sent in txts:f.write(sent+'\n')
        call(['spm_encode',
              '--model='+os.path.join(bpe_dir,'bpe.model'),
              '--output_format=piece'
            ],stdin=open(os.path.join(bpe_dir,'raw',s+'.txt'),'r'),
              stdout=open(os.path.join(bpe_dir,'encode',s+'.txt'),'w'))

    # Make Dict
    encode_table = {'<sos>':0,'<eos>':1}
    with open(os.path.join(bpe_dir,'bpe.vocab'),'r', encoding="utf-8") as f:
        for line in f:
            tok = line.split('\t')[0]
            if tok not in ['<s>','</s>']:
                encode_table[tok] = len(encode_table)


print('')
print('Data sets :')
for idx,s in enumerate(sets):
    print('\t',idx,':',s)
tr_set = input('Please enter the index of splits you wish to use preprocess. (seperate with space): ')
tr_set = [sets[int(t)] for t in tr_set.split(' ')]

file_num = {'train':4620, 'test':1680}
for s in tr_set:
    # Process training data
    print('')
    print('Preprocessing {}ing data...'.format(s),end='')
    todo = list(Path(os.path.join(paras.data_path,s)).rglob(r"*.[wW][aA][vV]"))
    todo = [_t for _t in todo if len(str(_t).split('.'))<3] # filter out the file with ext .wav.wav
    print(len(todo),'audio files found in {}ing set (should be {})'.format(s, file_num[s]))
    
    
    print('Extracting acoustic feature...',flush=True)
    tr_x = Parallel(n_jobs=paras.n_jobs)(delayed(extract_feature)(str(file),feature=paras.feature_type,dim=paras.feature_dim,cmvn=paras.apply_cmvn,delta=paras.apply_delta,delta_delta=paras.apply_delta_delta) for file in tqdm(todo))
    print('Encoding {}ing target...'.format(s),flush=True)
    tr_y = Parallel(n_jobs=paras.n_jobs)(delayed(read_text)(str(file),target=paras.target) for file in tqdm(todo))
    tr_y, encode_table = encode_target(tr_y,table=encode_table,mode=paras.target,max_idx=paras.n_tokens)
    
    dim = paras.feature_dim*(1+paras.apply_delta+paras.apply_delta_delta)
    output_dir = os.path.join(paras.output_path,'_'.join(['timit',str(paras.feature_type)+str(dim),str(paras.target)+str(len(encode_table))]))
    print('Saving {}ing data to'.format(s),output_dir)
    if not os.path.exists(output_dir):os.makedirs(output_dir)
    tr_y_str = ['_'.join([str(i) for i in _tr_y]) for _tr_y in tr_y]
    df = pd.DataFrame(data={'file_path':todo, 'length':[len(_x) for _x in tr_x], 'label':tr_y_str})
    df.to_csv(os.path.join(output_dir, s+'.csv'))
    with open(os.path.join(output_dir,"{}_x.pkl".format(s)), "wb") as fp:
        pickle.dump(tr_x, fp)
    del tr_x
    with open(os.path.join(output_dir,"{}_y.pkl".format(s)), "wb") as fp:
        pickle.dump(tr_y, fp)
    del tr_y
    with open(os.path.join(output_dir,"mapping.pkl"), "wb") as fp:
        pickle.dump(encode_table, fp)
    
print('All done, exit.')
