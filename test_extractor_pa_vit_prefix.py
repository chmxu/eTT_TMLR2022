"""
This code allows you to evaluate performance of a single feature extractor + pa with NCC
on the test splits of all datasets (ilsvrc_2012, omniglot, aircraft, cu_birds, dtd, quickdraw, fungi, 
vgg_flower, traffic_sign, mscoco, mnist, cifar10, cifar100). 

To test the url model on the test splits of all datasets, run:
python test_extractor_pa_vit_prefix.py --data.test ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower traffic_sign mscoco
"""

import os
import torch
import tensorflow as tf
import numpy as np
from tqdm import tqdm, trange
from tabulate import tabulate
from utils import check_dir
import torch.nn as nn

from models.losses import compute_prototypes, prototype_loss, knn_loss, lr_loss, scm_loss, svm_loss
from models.model_utils import CheckPointer
from models.model_helpers import get_model
from models.pa_prefix import apply_selection, pa
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, MetaDatasetBatchReader, TRAIN_METADATASET_NAMES, ALL_METADATASET_NAMES)
from config import args
from models.vit_dino import vit_small, vit_small_adapter


def get_init_prefix(model, support_img, context_labels):
    n_way = torch.unique(context_labels).shape[0]
    with torch.no_grad():
        patch_embed = model.module.get_patch_embed(support_img)
        proto = compute_prototypes(patch_embed, context_labels, n_way)
    return proto

def main():
    TEST_SIZE = 600
    num_gpu = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    tf.compat.v1.disable_eager_execution()

    # Setting up datasets
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
    #testsets = ALL_METADATASET_NAMES # comment this line to test the model on args['data.test']
    trainsets = TRAIN_METADATASET_NAMES
    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['test.type'], gin_file_name='meta_dataset_config_vit.gin')
    model = vit_small_adapter(global_pool=False)
    
    ckpt = torch.load(args['data.ckpt'])['student']
    new_dict = {}
    for k, v in ckpt.items():
        if 'backbone' in k:
            new_dict[k.replace('module.backbone.', '')] = v
    msg = model.load_state_dict(new_dict, strict=False)

    model = model.to("cuda:0")
    print(msg)
    model.eval()

    accs_names = ['NCC']
    var_accs = dict()

    for name, param in model.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False

    model = nn.DataParallel(model)

    old_dict = {}
    for k, v in model.state_dict().items():
        old_dict[k] = v.clone()

    default_graph = tf.compat.v1.get_default_graph()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            if dataset in ['ilsvrc_2012', 'cu_birds', 'fungi', 'mscoco']:
                lr = 5e-4
            elif dataset in ['traffic_sign']:
                lr = 5e-3
            elif dataset in ['omniglot']:
                lr = 5e-4
            else:
                lr = 1e-3
            print(dataset)
            var_accs[dataset] = {name: [] for name in accs_names}

            tqdm_bar = trange(TEST_SIZE)
            for i in tqdm_bar:
                sample = test_loader.get_test_task(session, dataset)
                support_img = sample['context_images']
                query_img = sample['target_images']
                context_labels = sample['context_labels']
                target_labels = sample['target_labels']

                init_prefix = get_init_prefix(model, support_img, context_labels)

                # optimize selection parameters and perform feature selection
                results = pa(model, support_img, query_img, 
                    context_labels, target_labels, max_iter=40, lr=lr, distance=args['test.distance'], input_dim=384, init_prefix = init_prefix, num_gpu=num_gpu, episode_idx=i)

                selected_context, selected_target = results

                _, stats_dict, _ = prototype_loss(
                    selected_context, context_labels,
                    selected_target, target_labels, distance=args['test.distance'])

                var_accs[dataset]['NCC'].append(stats_dict['acc'])
                tqdm_bar.set_description('Acc {:.2f}'.format(100 * np.array(var_accs[dataset]['NCC']).mean()))

                model.load_state_dict(old_dict, strict=False)
            dataset_acc = np.array(var_accs[dataset]['NCC']) * 100
            print(f"{dataset}: test_acc {dataset_acc.mean():.2f}%")
            torch.cuda.empty_cache()
    # Print nice results table
    print('results of {}'.format(args['model.name']))
    rows = []
    for dataset_name in testsets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(var_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)
    out_path = os.path.join(args['out.dir'], 'weights')
    out_path = check_dir(out_path, True)
    out_path = os.path.join(out_path, '{}-{}-{}-{}-test-results.npy'.format(args['model.name'], args['test.type'], 'pa', args['test.distance']))
    np.save(out_path, {'rows': rows})

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")


if __name__ == '__main__':
    main()



