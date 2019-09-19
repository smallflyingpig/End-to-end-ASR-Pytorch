import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np 
import logging
from tensorboardX import SummaryWriter
from optparse import OptionParser
from dataset import TIMIT
from model import SpeakerIDNet
from utils import read_conf
from trainer import ClassifierTrainer, save_checkpoint, load_checkpoint


def evaluate(model, test_dataset, cost):
    pass

def batch_process(model:SpeakerIDNet, data, train_mode=True, **kwargs)->dict:
    wav_data, label = data
    wav_data, label = wav_data.float().cuda(), label.long().cuda()
    if train_mode:
        model.train()
        optimizer, loss_func = kwargs.get('optimizer'), kwargs.get('loss_func')
        pout = model.forward(wav_data)
        pred = torch.max(pout, dim=1)[1]
        err = torch.mean((pred != label).float())
        loss = loss_func(pout, label)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        loss, err = loss.detach().item(), err.detach().item()
        rtn = {
            "output":"loss:{}, err:{}".format(loss, err),
            "vars":{"loss":loss, "err":err},
            "count":{"loss":len(label), "err":len(label)}
        }
    else: #eval
        model.eval()
        wav_data, label = wav_data.squeeze(0), label.squeeze(0)
        loss_func = kwargs.get("loss_func")
        # print(wav_data.shape, label.shape)
        pout = model.forward(wav_data)
        pred = torch.max(pout, dim=1)[1]
        err = torch.mean((pred != label).float())

        _, pred_snet = torch.max(torch.sum(pout, dim=0), 0)
        err_sent = (pred_snet != label[0]).float()

        loss = loss_func(pout, label)
        loss, err, err_sent = loss.detach().item(), err.detach().item(), err_sent.detach().item()
        rtn = {
            "output":"loss:{}, err:{}, err_sent:{}".format(loss, err, err_sent),
            "vars":{"loss":loss, "err":err, "err_sent":err_sent},
            "count":{"loss":len(label), "err":len(label), "err_sent":len(label)}
        }
    return rtn


class EvalHook(object):
    def __init__(self):
        self.best_accu = 0
    
    def __call__(self, model:nn.Module, epoch_idx, output_dir, 
        eval_rtn:dict, test_rtn:dict, logger:logging.Logger, writer:SummaryWriter):
        # save model
        is_best = 1-test_rtn.get('err_sent', 0) > self.best_accu
        self.best_accu = 1-test_rtn.get('err_sent', 0) if is_best else self.best_accu
        model_filename = "epoch_{}.pth".format(epoch_idx)
        save_checkpoint(model, os.path.join(output_dir, model_filename), 
            meta={'epoch':epoch_idx})
        os.system(
            "ln -sf {} {}".format(os.path.abspath(os.path.join(output_dir, model_filename)), 
            os.path.join(output_dir, "latest.pth"))
            )
        if is_best:
            os.system(
            "ln -sf {} {}".format(os.path.abspath(os.path.join(output_dir, model_filename)), 
            os.path.join(output_dir, "best.pth"))
            )

        if logger is not None:
            logger.info("EvalHook: best accu: {:.3f}, is_best: {}".format(self.best_accu, is_best))


def get_option():
    parser=OptionParser()
    parser.add_option("--cfg", type=str, default="./sincnet/cfg/SincNet_TIMIT.cfg") # Mandatory
    parser.add_option("--eval", action='store_true', default=False, help="eval the model")
    parser.add_option("--pt_file", type=str, default='', help="path for pretrained file")
    parser.add_option("--data_root", type=str, default='', help="path for data")
    parser.add_option("--datalist_root", type=str, default="./sincnet/data/TIMIT")
    parser.add_option("--output_dir", type=str, default="./output/sincnet")
    parser.add_option("--dataset", choices=['timit', 'libri'], default='timit', help="the dataset name")

    (args,_)=parser.parse_args()
    return args


def main(args):
    args = read_conf(args.cfg, args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.dataset == 'timit':
        train_dataset = TIMIT(data_root=args.data_root, datalist_root=args.datalist_root, train=True, oversampling=args.oversampling)
        test_dataset = TIMIT(data_root=args.data_root, datalist_root=args.datalist_root, train=False)
    elif args.dataset == 'libri':
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    cost = nn.NLLLoss()

    CNN_arch = {'input_dim': train_dataset.wlen,
              'fs': args.fs,
              'cnn_N_filt': args.cnn_N_filt,
              'cnn_len_filt': args.cnn_len_filt,
              'cnn_max_pool_len':args.cnn_max_pool_len,
              'cnn_use_laynorm_inp': args.cnn_use_laynorm_inp,
              'cnn_use_batchnorm_inp': args.cnn_use_batchnorm_inp,
              'cnn_use_laynorm':args.cnn_use_laynorm,
              'cnn_use_batchnorm':args.cnn_use_batchnorm,
              'cnn_act': args.cnn_act,
              'cnn_drop':args.cnn_drop,          
              }

    DNN1_arch = {'fc_lay': args.fc_lay,
              'fc_drop': args.fc_drop, 
              'fc_use_batchnorm': args.fc_use_batchnorm,
              'fc_use_laynorm': args.fc_use_laynorm,
              'fc_use_laynorm_inp': args.fc_use_laynorm_inp,
              'fc_use_batchnorm_inp':args.fc_use_batchnorm_inp,
              'fc_act': args.fc_act,
              }

    DNN2_arch = {'input_dim':args.fc_lay[-1] ,
              'fc_lay': args.class_lay,
              'fc_drop': args.class_drop, 
              'fc_use_batchnorm': args.class_use_batchnorm,
              'fc_use_laynorm': args.class_use_laynorm,
              'fc_use_laynorm_inp': args.class_use_laynorm_inp,
              'fc_use_batchnorm_inp':args.class_use_batchnorm_inp,
              'fc_act': args.class_act,
              }

    model = SpeakerIDNet(CNN_arch, DNN1_arch, DNN2_arch)
    if args.pt_file!='':
        print("load model from:", args.pt_file)
        checkpoint_load = torch.load(args.pt_file)
        ext = os.path.splitext(args.pt_file)[1]
        if ext == '.pkl':
            model.load_raw_state_dict(checkpoint_load)
        elif ext == '.pickle':
            model.load_state_dict(checkpoint_load)
        elif ext == '.pth':
            load_checkpoint(model, args.pt_file)
        else:
            raise NotImplementedError
    model = model.cuda()
    if args.eval:
        print('only eval the model')
        evaluate(model, test_dataset, cost)
        return
    else:
        print("train the model")
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr,alpha=0.95, eps=1e-8) 
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False, num_workers=8, pin_memory=True)
    trainer = ClassifierTrainer(model, train_dataloader, optimizer, cost, batch_process, args.output_dir, 0, test_dataloader, eval_every=args.N_eval_epoch, print_every=args.print_every)
    trainer.run(args.N_epochs)


if __name__=="__main__":
    args = get_option()
    print(args)
    main(args)



