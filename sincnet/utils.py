import configparser as ConfigParser
from optparse import OptionParser


def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError 

def read_conf(cfg_path, options):
    
    cfg_file=options.cfg
    Config = ConfigParser.ConfigParser()
    Config.read(cfg_file)

    #[windowing]
    options.fs = int(Config.get('windowing', 'fs'))
    options.cw_len = int(Config.get('windowing', 'cw_len'))
    options.cw_shift = int(Config.get('windowing', 'cw_shift'))

    #[cnn]
    options.cnn_N_filt = list(map(int, Config.get('cnn', 'cnn_N_filt').split(',')))
    options.cnn_len_filt = list(map(int, Config.get('cnn', 'cnn_len_filt').split(',')))
    options.cnn_max_pool_len = list(map(int, Config.get('cnn', 'cnn_max_pool_len').split(',')))
    options.cnn_use_laynorm_inp = str_to_bool(Config.get('cnn', 'cnn_use_laynorm_inp'))
    options.cnn_use_batchnorm_inp = str_to_bool(Config.get('cnn', 'cnn_use_batchnorm_inp'))
    options.cnn_use_laynorm = list(map(str_to_bool, Config.get('cnn', 'cnn_use_laynorm').split(',')))
    options.cnn_use_batchnorm = list(map(str_to_bool, Config.get('cnn', 'cnn_use_batchnorm').split(',')))
    options.cnn_act = list(map(str, Config.get('cnn', 'cnn_act').split(',')))
    options.cnn_drop = list(map(float, Config.get('cnn', 'cnn_drop').split(',')))

    #[dnn]
    options.fc_lay = list(map(int, Config.get('dnn', 'fc_lay').split(',')))
    options.fc_drop = list(map(float, Config.get('dnn', 'fc_drop').split(',')))
    options.fc_use_laynorm_inp = str_to_bool(Config.get('dnn', 'fc_use_laynorm_inp'))
    options.fc_use_batchnorm_inp = str_to_bool(Config.get('dnn', 'fc_use_batchnorm_inp'))
    options.fc_use_batchnorm = list(map(str_to_bool, Config.get('dnn', 'fc_use_batchnorm').split(',')))
    options.fc_use_laynorm = list(map(str_to_bool, Config.get('dnn', 'fc_use_laynorm').split(',')))
    options.fc_act = list(map(str, Config.get('dnn', 'fc_act').split(',')))

    #[class]
    options.class_lay = list(map(int, Config.get('class', 'class_lay').split(',')))
    options.class_drop = list(map(float, Config.get('class', 'class_drop').split(',')))
    options.class_use_laynorm_inp = str_to_bool(Config.get('class', 'class_use_laynorm_inp'))
    options.class_use_batchnorm_inp = str_to_bool(Config.get('class', 'class_use_batchnorm_inp'))
    options.class_use_batchnorm = list(map(str_to_bool, Config.get('class', 'class_use_batchnorm').split(',')))
    options.class_use_laynorm = list(map(str_to_bool, Config.get('class', 'class_use_laynorm').split(',')))
    options.class_act = list(map(str, Config.get('class', 'class_act').split(',')))

    #[optimization]
    options.lr=float(Config.get('optimization', 'lr'))
    options.batch_size=int(Config.get('optimization', 'batch_size'))
    options.N_epochs=int(Config.get('optimization', 'N_epochs'))
    options.N_eval_epoch=int(Config.get('optimization', 'N_eval_epoch'))
    options.print_every=int(Config.get('optimization', 'print_every'))
    options.oversampling=int(Config.get('optimization', 'oversampling'))
    options.seed=int(Config.get('optimization', 'seed'))
    
    return options