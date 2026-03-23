##########################################################################################
# import

import os
import sys
import logging

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "../..")  # for problem_def
sys.path.insert(0, "../../..")  # for vrp_utils
sys.path.append('')
from my_utils import create_logger, copy_all_src
from LRP_Evaluator import LRPEvaluator as evaluator_syn

##########################################################################################
# Machine Environment Config
debug_mode = False
use_cuda = True
cuda_device_num = 0
copy_source = False
##########################################################################################
# Path Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "../..")  # for problem_def
sys.path.insert(0, "../../..")  # for utils
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
##########################################################################################
# parameters
logger_params = {
    'log_file': {
        'desc': '',
        'filename': 'run_log'
    }
}

env_params = {
    'sample_size': 10, # Equal to customer size
    'load_path': './data/5_10_1000.pkl',
}

model_params = {
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 16,
    'clip': 10,
    'ff_hidden_dim': 512,
    'mlp_hidden_size': [128, 64, 32],   # half of embedding_dim
    'sample':  False,  # argmax, softmax
}

eval_params = {
    'use_cuda': use_cuda,
    'cuda_device_num': cuda_device_num,
    'model_load': {
        'path': './result/10',  # directory path of model and log files saved.
        'epoch': 'best',  # epoch version of pre-trained model to load.
    },
    'episodes': 2000,
    'eval_batch_size': 10,
    'augmentation': '8',    # None or 'int', times of augmentation
    'sgbs_beta': 4,            
    'sgbs_gamma_minus1': (4-1),
    'mode': 'origin'    # 'origin' or 'bs'
}


##########################################################################################
# main
def main():
    if debug_mode:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()


    evaluator = evaluator_syn(env_params=env_params,
                            model_params=model_params,
                            eval_params=eval_params)


    if copy_source:
        copy_all_src(evaluator.result_folder)

    evaluator.run()


def _set_debug_mode():
    global eval_params
    eval_params['episodes'] = 20


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(debug_mode))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(use_cuda, cuda_device_num))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
