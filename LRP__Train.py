##########################################################################################
# import
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

# Path Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "../..")  # for problem_def
sys.path.insert(0, "../../..")  # for utils
sys.path.append('')

from my_utils import create_logger
from LRP_Trainer import LRPTrainer

##########################################################################################
# Machine Environment Config
debug_mode = False
use_cuda = True
cuda_device_num = 0


##########################################################################################
# Scale 10
env_params = {
    'customer_size': 10,
    'mt_size': 10,
    'depot_size': 5,
    'depot_capacity_min': 100,
    'depot_capacity_max': 300,
    'depot_cost_min': 1,
    'depot_cost_max': 5,
    'vehicle_capacity': 70,
    'vehicle_cost': 1,
    'demand_min': 10,
    'demand_max': 20,
}

# Scale 20
# env_params = {
#     'customer_size': 20, 
#     'mt_size': 20, 
#     'depot_size': 5, 
#     'depot_capacity_min': 100, 
#     'depot_capacity_max': 300, 
#     'depot_cost_min': 1, 
#     'depot_cost_max': 5, 
#     'vehicle_capacity': 70, 
#     'vehicle_cost': 1, 
#     'demand_min': 10, 
#     'demand_max': 20
# }

# Scale 50
# env_params = {
#     'customer_size': 50,
#     'mt_size': 50,
#     'depot_size': 10,
#     'depot_capacity_min': 130,
#     'depot_capacity_max': 450,
#     'depot_cost_min': 1,
#     'depot_cost_max': 5,
#     'vehicle_capacity': 70,
#     'vehicle_cost': 1,
#     'demand_min': 10,
#     'demand_max': 20,
# }

# Scale 100
# env_params = {
#      'customer_size': 100,
#      'mt_size':100,
#      'depot_size': 20,
#      'depot_capacity_min': 500,
#      'depot_capacity_max': 900,
#      'depot_cost_min': 5,
#      'depot_cost_max': 15,
#      'vehicle_capacity': 70,
#      'vehicle_cost': 1,
#      'demand_min': 10,
#      'demand_max': 20,
#  }

model_params = {
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 16,
    'clip': 10,
    'ff_hidden_dim': 512,
    'mlp_hidden_size': [128, 64, 32],   # half of embedding_dim
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [701],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': use_cuda,
    'cuda_device_num': cuda_device_num,
    'epochs': 1000,
    'episodes': 10000,
    'batch_size': 128,
    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 200,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'unlimited.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': './result/',  # directory path of model and log files saved.
        'epoch': 1,  # epoch version of pre-trained model to load.

    }
}

logger_params = {
    'log_file': {
        'desc': '',
        'filename': 'run_log'
    }
}


##########################################################################################
# main

def main():
    if debug_mode:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()


    trainer = LRPTrainer(env_params=env_params,
                           model_params=model_params,
                           optimizer_params=optimizer_params,
                           trainer_params=trainer_params)

    
    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 5
    trainer_params['episodes'] = 5
    trainer_params['batch_size'] = 1
    global model_params


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(debug_mode))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(use_cuda, cuda_device_num))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
