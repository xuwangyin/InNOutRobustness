import matplotlib as mpl
mpl.use('Agg')

import os
import torch
import torch.nn as nn

from utils.model_normalization import RestrictedImageNetWrapper
import utils.datasets as dl
import utils.models.model_factory_224 as factory
import utils.run_file_helpers as rh
from distutils.util import strtobool

import argparse

parser = argparse.ArgumentParser(description='Define hyperparameters.', prefix_chars='-')
parser.add_argument('--net', type=str, default='resnet50_timm', help='ResNet18, 34, 50 or 101')
parser.add_argument('--model_params', nargs='+', default=[])
parser.add_argument('--dataset', type=str, default='restrictedimagenet', help='restrictedimagenet')
parser.add_argument('--od_dataset', type=str, default='restrictedimagenetOD',
                    help=('restrictedimagenetOD, imagenet or openImages'))
parser.add_argument('--balanced_sampling', type=lambda x: bool(strtobool(x)), default=True,
                    help='Whether to use balanced sampling for dataset')
parser.add_argument('--task', type=str, default='RestrictedImageNet',
                    help='Task name used for model and log directory naming')

rh.parser_add_commons(parser)
rh.parser_add_adversarial_commons(parser)
rh.parser_add_adversarial_norms(parser, 'restrictedimagenet')

hps = parser.parse_args()
#
device_ids = None
if len(hps.gpu)==0:
    device = torch.device('cpu')
    print('Warning! Computing on CPU')
elif len(hps.gpu)==1:
    device = torch.device('cuda:' + str(hps.gpu[0]))
else:
    device_ids = [int(i) for i in hps.gpu]
    device = torch.device('cuda:' + str(min(device_ids)))

#Load model
model_root_dir = f'{hps.task}Models'
logs_root_dir = f'{hps.task}Logs'
num_classes = 9  # RestrictedImageNet typically has 9 classes

if len(hps.model_params) == 0:
    model_params = None
else:
    model_params = hps.model_params
print(model_params)
model, model_name, model_config, img_size = factory.build_model(hps.net, num_classes, model_params=model_params)
model_dir = os.path.join(model_root_dir, model_name)
log_dir = os.path.join(logs_root_dir, model_name)

start_epoch, optim_state_dict = rh.load_model_checkpoint(model, model_dir, device, hps)
model = RestrictedImageNetWrapper(model).to(device)

msda_config = rh.create_msda_config(hps)

#load dataset
od_bs = int(hps.od_bs_factor * hps.bs)

id_config = {}
if hps.dataset == 'restrictedImagenet':
    train_loader = dl.get_restrictedImageNet(train=True, batch_size=hps.bs, augm_type=hps.augm, size=img_size,
                                  config_dict=id_config, balanced=hps.balanced_sampling)
else:
    raise ValueError(f'Dataset {hps.dataset} not supported')

if hps.train_type.lower() in ['ceda', 'acet', 'advacet', 'tradesacet', 'tradesceda']:
    od_config = {}
    loader_config = {'ID config': id_config, 'OD config': od_config}

    if hps.od_dataset == 'restrictedimagenetOD':
        tiny_train = dl.get_restrictedImageNetOD(batch_size=od_bs, augm_type=hps.augm, size=img_size,
                                                config_dict=od_config)
    elif hps.od_dataset == 'imagenet':
        tiny_train = dl.get_ImageNet(train=True, batch_size=od_bs, augm_type=hps.augm, size=img_size,
                                    exclude_restricted=True, config_dict=od_config)
    elif hps.od_dataset == 'openImages':
        tiny_train = dl.get_openImages('train', batch_size=od_bs, shuffle=True, augm_type=hps.augm, 
                                     size=img_size, exclude_dataset='restrictedimagenet', config_dict=od_config)
    else:
        raise ValueError(f'OD dataset {hps.od_dataset} not supported')
else:
    loader_config = {'ID config': id_config}

if hps.dataset == 'restrictedImagenet':
    test_loader = dl.get_restrictedImageNet(train=False, batch_size=hps.bs, augm_type='test', size=img_size, balanced=hps.balanced_sampling)
else:
    assert False

scheduler_config, optimizer_config = rh.create_optim_scheduler_swa_configs(hps)
id_attack_config, od_attack_config = rh.create_attack_config(hps, 'restrictedImagenet')
trainer = rh.create_trainer(hps, model, optimizer_config, scheduler_config, device, num_classes,
                            model_dir, log_dir, msda_config=msda_config, model_config=model_config,
                            id_attack_config=id_attack_config, od_attack_config=od_attack_config)
# import ipdb; ipdb.set_trace()
##DEBUG:
# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

# run training
if trainer.requires_out_distribution():
    train_loaders, test_loaders = trainer.create_loaders_dict(train_loader, test_loader=test_loader,
                                                              out_distribution_loader=tiny_train)
    trainer.train(train_loaders, test_loaders, loader_config=loader_config, start_epoch=start_epoch,
                  optim_state_dict=optim_state_dict, device_ids=device_ids)
else:
    train_loaders, test_loaders = trainer.create_loaders_dict(train_loader, test_loader=test_loader)
    trainer.train(train_loaders, test_loaders, loader_config=loader_config, start_epoch=start_epoch,
                  optim_state_dict=optim_state_dict, device_ids=device_ids)
