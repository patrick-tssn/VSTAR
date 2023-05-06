# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import sys
import math
import time
import random
import copy
import json
import logging
import datetime
import numpy as np
import pickle as pkl
from pprint import pformat
from argparse import ArgumentParser
from requests import session
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine, Checkpoint, DiskSaver
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from ignite.handlers.stores import EpochOutputStore
from ignite.metrics import RunningAverage, Average
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

from transformers import AdamW, BertTokenizer
from transformers.file_utils import (CONFIG_NAME, WEIGHTS_NAME)
from transformers.models.bert.configuration_bert import BertConfig
from transformers.utils.dummy_pt_objects import Adafactor

from model.VSBert import BertForSegClassification, BertForVisSegClassification
from utils.eval import evaluate
from utils.metrics import F1ScoreMetric, MovieNetMetric, SklearnAPMetric, session_eval
from data.seg_dataset import DataSet, collate_fn, get_dataset

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def padding(seq, pad_token, limit=1020, ft=False):
    max_len = max([i.size(0) for i in seq])
    max_len = min(max_len, limit)
    if ft:
        result = torch.ones((len(seq), max_len)).float() * pad_token
    else:
        result = torch.ones((len(seq), max_len)).long() * pad_token
    # if len(seq[0].size()) == 1:
    #     result = torch.ones((len(seq), max_len)).long() * pad_token
    # else:
    #     result = torch.ones((len(seq), max_len, seq[0].size(-1))).float()
    for i in range(len(seq)):
        result[i, :seq[i].size(0)] = seq[i][-min(limit, seq[i].size(0)):]
    return result


# main
if __name__ =="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="bert", help="Pretrained Model type")
    parser.add_argument("--max_history", type=int, default=0, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--test_path", type=str, default="inputs/full/dialog_topic/pro_seg_clip_test.json", help="Path of the testset")
    parser.add_argument("--feature_path", type=str, default="inputs/feature/vit", help="Path of the feature")
    parser.add_argument("--output", type=str, default="result.json")
    parser.add_argument("--ckptid", type=str, help='ckpt selected for test')
    parser.add_argument("--gpuid", type=str, default='0', help='gpu id')
    parser.add_argument("--log", type=bool, default=False, help='if logging info')
    parser.add_argument('--exp_set', type=str, default='test')
    parser.add_argument('--video', type=int, default=0)
    parser.add_argument('--eval', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    set_seed()
    st = time.time()

    exp_set = args.exp_set
    model_checkpoint = 'ckpts/' + args.model + args.exp_set + '/'
    output_dir = 'results/' + args.model + exp_set
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.output = output_dir + '/res_{}.json'.format(args.ckptid)
    # args.output = output_dir + '/result_{}.json'.format(args.gen_type)
    if args.device == 'cuda':
        args.device = 'cuda:' + args.gpuid
    # args.device = 'cuda:2'
    # setproctitle.setproctitle("task_{}_ckpt_{}_beam_{}_min_{}_pen_{:.1f}".format(args.task, args.ckptid, args.beam_size, args.min_length, args.penalty))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))

    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s')
 
    logging.info('Loading model params from ' + model_checkpoint)
    
    
    tokenizer_class = BertTokenizer
    if args.video:
        model_class = BertForVisSegClassification
    else:
        model_class = BertForSegClassification
    model_config = BertConfig.from_pretrained(model_checkpoint)

    tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
    model = model_class.from_pretrained(model_checkpoint+"checkpoint_mymodel_" + args.ckptid + ".pt", config=model_config)
    model.to(args.device)
    model.eval()
    args.pad = model.config.pad_token_id

    if args.max_history:
        test_dataset_path = 'inputs/full/pkls/test_20_data.pkl'
    else:
        test_dataset_path = 'inputs/full/pkls/seg_test_vit_data.pkl'
    if not os.path.exists(test_dataset_path):
        test_dataset = get_dataset(tokenizer, args.test_set, test=True, n_history=args.max_history)
        with open(test_dataset_path, 'wb') as f:
            pkl.dump(test_dataset, f)
    else:
        with open(test_dataset_path, 'rb') as f:
            test_dataset = pkl.load(f)
    
    if args.video:
        with open('inputs/full/clipid2frames.json') as jh:
            feature = json.load(jh)
        test_ds = DataSet(test_dataset[0], tokenizer, feature)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
    else:
            test_ds = DataSet(test_dataset[0], tokenizer, None, model=args.model)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=None))
    
    # f1_metric = F1ScoreMetric(num_classes=1)
    if args.video:
        f1_metric = F1ScoreMetric(average='micro', num_classes=1, multiclass=False, threshold=0.4)
        ap_metric = SklearnAPMetric()
        miou_metric = MovieNetMetric()
        f1_metric = f1_metric.to(args.device)
        ap_metric = ap_metric.to(args.device)
        miou_metric = miou_metric.to(args.device)

    model.eval()
    pred_res = []
    ref_res = []
    pk = wd = f1 = cnt = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='test'):
            dialog_ids, dialog_type_ids, dialog_mask, session_label_ids, session_indexs,\
            feature_ids, feature_type_ids, feature_mask, scene_label_ids, scene_indexs, vid_lst = batch
            if args.video == 0:   
                dialog_ids = dialog_ids.to(args.device)
                dialog_type_ids = dialog_type_ids.to(args.device)
                dialog_mask = dialog_mask.to(args.device)
                label_ids = session_label_ids.to(args.device)
                session_indexs = [sess.to(args.device) for sess in session_indexs]
                logits = model(dialog_ids, dialog_mask, dialog_type_ids, seg_indexs=session_indexs)[0]
            else:
                feature_ids = feature_ids.to(args.device)
                feature_type_ids = feature_type_ids.to(args.device)
                feature_mask = feature_mask.to(args.device)
                scene_label_ids = scene_label_ids.to(args.device)
                label_ids = scene_label_ids.to(args.device)
                scene_indexs = [scene.to(args.device) for scene in scene_indexs]
                logits = model(feature_ids, feature_mask, feature_type_ids, seg_indexs=scene_indexs)[0]

            logit_lst = []
            label_lst = []
            start = 0
            for i in range(label_ids.size(0)):
                label = label_ids[i][label_ids[i]!=-1].squeeze(dim=-1)
                if session_indexs is not None or scene_indexs is not None:
                    logit = logits[start:start+label.size(0)]
                    start += label.size(0)
                else:
                    logit = logits[i][label_ids[i]!=-1].squeeze(dim=-1)
                
                prob = F.softmax(logit, dim=1)
                pred = torch.argmax(prob, dim=1)
                prob_list = prob.cpu().numpy().tolist()
                label_list = label.cpu().numpy().tolist()
                pred_list = pred.cpu().numpy().tolist()
                pred_res.append(prob_list)
                ref_res.append(label_list)
                if args.video:
                    label = label.to(args.device)
                    logit_lst.append(logit)
                    label_lst.append(label)
                    vids = []
                    sids = []
                    for k in range(label.size(0)):
                        vids.append(vid_lst[i])
                        sids.append(k)
                    for vid, sid, pre, gt in zip(vids, sids, pred, label):
                        miou_metric.update(vid, sid, pre, gt)
                else:
                    t_wd, t_pk, t_f1 = session_eval(pred_list, label_list)
                    wd += t_wd
                    pk += t_pk
                    f1 += t_f1
                    cnt += 1

            if args.video:
                logit_pt = torch.cat(logit_lst, dim=0)
                label_pt = torch.cat(label_lst, dim=0)
                probs = F.softmax(logit_pt, dim=1)
                f1_metric.update(probs[:,1], label_pt)
                ap_metric.update(probs[:,1], label_pt)

            # break

    if args.video:
        f1 = f1_metric.compute()
        ap = ap_metric.compute()
        miou = miou_metric.compute()
        print(f1.item(), ap.item(), miou.item())
        with open(os.path.join(output_dir, 'res.log'), 'a') as fh:
            fh.write('f1:{} ap:{} miou:{}\n'.format(f1.item(), ap.item(), miou.item()))
    else:
        with open(os.path.join(output_dir, 'res.log'), 'a') as fh:
            fh.write('wd:{} pk:{} f1:{}\n'.format(wd/cnt, pk/cnt, f1/cnt))
        print(wd/cnt, pk/cnt, f1/cnt)
    if args.video:
        with open(os.path.join(output_dir, 'vis_res_{}.json'.format(args.ckptid)), 'w') as jh:
            json.dump([pred_res, ref_res], jh)
    else:
        with open(os.path.join(output_dir, 'res_{}.json'.format(args.ckptid                                                                               )), 'w') as jh:
            json.dump([pred_res, ref_res], jh)