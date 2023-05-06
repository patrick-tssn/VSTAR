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
from xmlrpc.client import MultiCall
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

from transformers import AdamW, BertTokenizer, RobertaTokenizer
from transformers.file_utils import (CONFIG_NAME, WEIGHTS_NAME)
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.utils.dummy_pt_objects import Adafactor

from model.VSBert import BertForSegClassification, BertForVisSegClassification
from utils.eval import evaluate
from utils.metrics import F1ScoreMetric, MovieNetMetric, SklearnAPMetric, DTSMetric, session_eval
from data.seg_dataset import DataSet, collate_fn, get_dataset

logger = logging.getLogger(__file__)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def get_data_loaders_new(args, tokenizer):
    
    train_data_path = 'inputs/full/pkls/seg_train_resnet_data.pkl'
    valid_data_path = 'inputs/full/pkls/seg_valid_resnet_data.pkl'
    test_data_path = 'inputs/full/pkls/seg_test_resnet_data.pkl'
    # train
    if not os.path.exists(train_data_path):
        print('no preprocessed train pkl exist, start process data.')
        train_data = get_dataset(tokenizer, args.train_path)
        with open(train_data_path, 'wb') as f:
            pkl.dump(train_data, f)
    else:
        print('load train data from pkl')
        with open(train_data_path, 'rb') as f:
            train_data = pkl.load(f)
    # valid
    if not os.path.exists(valid_data_path):
        print('no preprocessed valid pkl exist, start process data.')
        valid_data = get_dataset(tokenizer, args.valid_path)
        with open(valid_data_path, 'wb') as f:
            pkl.dump(valid_data, f)
    else:
        print('load valid data from pkl')
        with open(valid_data_path, 'rb') as f:
            valid_data = pkl.load(f)
    # test
    if not os.path.exists(test_data_path):
        print('no preprocessed test pkl exist, start process data.')
        test_data = get_dataset(tokenizer, args.test_path)
        with open(test_data_path, 'wb') as f:
            pkl.dump(test_data, f)
    else:
        print('load test data from pkl')
        with open(test_data_path, 'rb') as f:
            test_data = pkl.load(f)

    if args.video: 
        with open(args.feature_path) as jh:
            feature = json.load(jh)
        train_dataset = DataSet(train_data, tokenizer, feature, vis=True)
        valid_dataset = DataSet(valid_data, tokenizer, feature, vis=True)
        test_dataset = DataSet(test_data, tokenizer, feature, vis=True)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=2, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=4, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=4, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
    else:
        with open(args.feature_path) as jh:
            feature = json.load(jh)
        train_dataset = DataSet(train_data, tokenizer, feature, vis=False)
        valid_dataset = DataSet(valid_data, tokenizer, feature, vis=False)
        test_dataset = DataSet(test_data, tokenizer, feature, vis=False)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=2, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=False))
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=4, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=False))
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=4, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=False))
    return train_loader, valid_loader, test_loader

def get_gpu_id(gpu_need, card_id=None):
        # gpu_need = 3000
    gpuid = ''
    avail = 0
    if card_id is not None:
        gpu_cards = [int(card_id)]
    else:
        gpu_cards = [0, 1, 2, 3, 4, 5, 6, 7]
        gpu_cards = [2]
        # gpu_cards = [0]
        # gpu_cards = [1]
    # 自动查找一张满足显存需求的卡，仅根据当前时刻，并不会预判 Orz；找不到就会一直等
    while 1:
        for gid in gpu_cards:
            gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')[gid].split('|')
            gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
            gpu_total = int(gpu_status[2].split('/')[1].split('M')[0].strip())
            # gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
            if gpu_memory < gpu_total - gpu_need:
                if gpuid == '' or avail < gpu_total - gpu_memory:
                    gpuid = str(gid)
                    avail = gpu_total - gpu_memory
            if gid == '3' and gpuid != '':
                break
        if gpuid == '':
            sys.stdout.write('\rWaiting for GPU' + str(datetime.datetime.now()))
            sys.stdout.flush()
            time.sleep(1)
        else:
            gid = '/'; gpu_need = '/'; avail = '/'; gpu_memory = '/'; gpu_total = '/' # don't log
            break
    return gpuid

def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    # Change Multi GPU to single GPU
    original_keys = list(state_dict.keys())
    for key in original_keys:
        if key.startswith("module."):
            new_key = key[len("module."):]
            state_dict[new_key] = state_dict.pop(key)    
    return state_dict

def train():
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, default="inputs/full/dialog_topic/pro_seg_clip_train.json", help="Path of the trainset")
    parser.add_argument("--valid_path", type=str, default="inputs/full/dialog_topic/pro_seg_clip_valid.json", help="Path of the validset")
    parser.add_argument("--test_path", type=str, default="inputs/full/dialog_topic/pro_seg_clip_test.json", help="Path of the testset")
    parser.add_argument("--feature_path", type=str, default="inputs/full/clipid2frames.json", help="Path of the feature")
    parser.add_argument("--frame_count", type=str, default="inputs/full/clipid2frames.json", help="Path of the frameCNT")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")

    parser.add_argument("--gpuid", type=str, default='', help='select proper gpu id')
    parser.add_argument("--model", type=str, default='bert', help='Pretrained Model name')
    parser.add_argument('--video', type=int, default=1, help='if use video: 1 use 0 not')
    parser.add_argument('--exp_set', type=str, default='_test')
    parser.add_argument('--wait', type=str, default='', help='waiting gpu id')
    parser.add_argument('--test_every_epoch', type=int, default=1, choices=[0, 1])
    parser.add_argument('--test', type=int, default=0, choices=[0, 1])
    parser.add_argument('--ft', type=int, default=1, choices=[0, 1])
    parser.add_argument('--warmup_init', type=float, default=1e-07)
    parser.add_argument('--warmup_duration', type=float, default=5000)
    args = parser.parse_args()
    
    args.test_batch_size = args.train_batch_size
    args.valid_batch_size = args.train_batch_size
    # args.gradient_accumulation_steps = 32 // args.train_batch_size
    # args.model = 'bert'
    # exp_set = '_lm'
    exp_set = args.exp_set
    args.exp = args.model + exp_set
    args.log_path = 'ckpts/' + args.exp + '/'
    args.tb_path = 'tb_logs/' + args.exp + '/'
    # args.device = 'cpu'
    if args.device == 'cuda':
        if args.wait != '':
            args.gpuid = get_gpu_id(10000, args.wait)
        else:
            if args.gpuid == '':
                args.gpuid = get_gpu_id(10000)
        if args.gpuid != '':
            args.device = 'cuda:' + args.gpuid

    # select model
    if args.model == 'bert':
        args.model_checkpoint = "prev_trained_model/bert-base-uncased"
    elif args.model == 'roberta':
        args.model_checkpoint = "prev_trained_model/roberta-base"
    else:
        raise ValueError('NO IMPLEMENTED MODEL!')


    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.tb_path):
        os.makedirs(args.tb_path)
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))
    
    if args.model == 'bert':
        tokenizer_class = BertTokenizer
    elif args.model == 'roberta':
        tokenizer_class = RobertaTokenizer
    if args.video:
        model_class = BertForVisSegClassification
        if not args.ft:
            # bert_config = BertConfig(num_hidden_layers=2)
            bert_config = BertConfig()
            model = model_class(bert_config)
        else:
            model = model_class.from_pretrained(args.model_checkpoint)
    else:
        model_class = BertForSegClassification
        if not args.ft:
            bert_config = BertConfig()
            model = model_class(bert_config)
        else:
            if args.model == 'bert':
                model = model_class.from_pretrained(args.model_checkpoint)
            elif args.model == 'roberta':
                state_dict = load_state_dict(args.model_checkpoint+'/pytorch_model.bin')
                original_keys = list(state_dict.keys())
                for key in original_keys:
                    if key.startswith('roberta'):
                        new_key = key[len("bert"):]
                        state_dict[new_key] = state_dict.pop(key)
                robert_config = RobertaConfig.from_pretrained(args.model_checkpoint)
                robert_config.num_labels = 2
                model = BertForSegClassification(robert_config)
                model.load_state_dict(state_dict, strict=False)
            # model = model_class.from_pretrained(args.model_checkpoint)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    args.pad = model.config.pad_token_id

    logger.info("Prepare datasets for Bert")
    train_loader, valid_loader, test_loader = get_data_loaders_new(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        dialog_ids, dialog_type_ids, dialog_mask, session_label_ids, session_indexs, \
        feature_ids, feature_type_ids, feature_mask, scene_label_ids, scene_indexs, utter_lst, vid_lst = batch
        if args.video == 0:   
            feature_ids = feature_ids.to(args.device)
            feature_type_ids = feature_type_ids.to(args.device)
            feature_mask = feature_mask.to(args.device)
            dialog_ids = dialog_ids.to(args.device)
            dialog_type_ids = dialog_type_ids.to(args.device)
            dialog_type_ids = torch.cat([feature_type_ids, dialog_type_ids], dim=1)
            dialog_mask = dialog_mask.to(args.device)
            dialog_mask = torch.cat([feature_mask, dialog_mask], dim=1)
            session_label_ids = session_label_ids.to(args.device)
            session_indexs = [sess.to(args.device) for sess in session_indexs]
        else:
            dialog_ids = dialog_ids.to(args.device)
            dialog_type_ids = dialog_type_ids.to(args.device)
            dialog_mask = dialog_mask.to(args.device)
            feature_ids = feature_ids.to(args.device)
            feature_type_ids = feature_type_ids.to(args.device)
            feature_type_ids = torch.cat([feature_type_ids, dialog_type_ids], dim=1)
            feature_mask = feature_mask.to(args.device)
            feature_mask = torch.cat([feature_mask, dialog_mask], dim=1)
            scene_indexs = [scene.to(args.device) for scene in scene_indexs]
            scene_label_ids = scene_label_ids.to(args.device)

        # optimize Bert
        model.train(True)
        if args.video == 0:
            loss = model(feature_ids, dialog_ids, dialog_mask, dialog_type_ids, labels=session_label_ids, seg_indexs=session_indexs)[0]
        else:
            loss = model(feature_ids, dialog_ids, feature_mask, feature_type_ids, labels=scene_label_ids, seg_indexs=scene_indexs)[0]
        loss = loss / args.gradient_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    def valid(engine, batch):
        model.train(False)
        f1_metric = F1ScoreMetric(average='micro', num_classes=1, multiclass=False, threshold=0.5)
        f1_metric = f1_metric.to(args.device)
        with torch.no_grad():
            dialog_ids, dialog_type_ids, dialog_mask, session_label_ids, session_indexs, \
        feature_ids, feature_type_ids, feature_mask, scene_label_ids, scene_indexs, utter_lst, vid_lst = batch
        if args.video == 0:   
            feature_ids = feature_ids.to(args.device)
            feature_type_ids = feature_type_ids.to(args.device)
            feature_mask = feature_mask.to(args.device)
            dialog_ids = dialog_ids.to(args.device)
            dialog_type_ids = dialog_type_ids.to(args.device)
            dialog_type_ids = torch.cat([feature_type_ids, dialog_type_ids], dim=1)
            dialog_mask = dialog_mask.to(args.device)
            dialog_mask = torch.cat([feature_mask, dialog_mask], dim=1)
            session_indexs = [sess.to(args.device) for sess in session_indexs]
            label = session_label_ids.to(args.device)
            logits = model(feature_ids, dialog_ids, dialog_mask, dialog_type_ids, seg_indexs=session_indexs)[0]
        else:
            dialog_ids = dialog_ids.to(args.device)
            dialog_type_ids = dialog_type_ids.to(args.device)
            dialog_mask = dialog_mask.to(args.device)
            feature_ids = feature_ids.to(args.device)
            feature_type_ids = feature_type_ids.to(args.device)
            feature_type_ids = torch.cat([feature_type_ids, dialog_type_ids], dim=1)
            feature_mask = feature_mask.to(args.device)
            feature_mask = torch.cat([feature_mask, dialog_mask], dim=1)
            scene_indexs = [scene.to(args.device) for scene in scene_indexs]
            label = scene_label_ids.to(args.device)
            logits = model(feature_ids, dialog_ids, feature_mask, feature_type_ids, seg_indexs=scene_indexs)[0]

        prob = F.softmax(logits, dim=1)
        f1_metric.update(prob[:, 1], label)
        f1 = f1_metric.compute()
        return f1.item()
        
    trainer = Engine(update)
    validator = Engine(valid)

    # trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: validator.run(valid_loader))

    if args.ft:
        scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    else:
        torch_lr_scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader) - args.warmup_duration, 0.0)])
        scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler, warmup_start_value=args.warmup_init, warmup_duration=args.warmup_duration)
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # # Prepare metrics - note how we compute distributed metrics 
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    # RunningAverage(output_transform=lambda x: x).attach(validator, "f1")

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=["loss"])
    # eval_pbar = ProgressBar(persist=True)
    # eval_pbar.attach(validator, metric_names=['f1'])

    tb_logger = TensorboardLogger(log_dir=args.tb_path)
    tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
    # tb_logger.attach(validator, log_handler=OutputHandler(tag='validation', metric_names=["f1"], global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

    checkpoint_handler = ModelCheckpoint(args.log_path, 'checkpoint', n_saved=args.n_epochs ,require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation
    
    torch.save(args, args.log_path + 'model_training_args.bin')
    getattr(model, 'module', model).config.to_json_file(os.path.join(args.log_path, CONFIG_NAME))
    tokenizer.save_vocabulary(args.log_path)

    # best_score = Checkpoint.get_default_score_fn('f1')
    # best_model_handler = Checkpoint(
    #     {'mymodel': getattr(model, 'module', model)},
    #     filename_prefix='best',
    #     save_handler=DiskSaver(args.log_path, create_dir=True, require_empty=False),
    #     score_name='f1',
    #     score_function=best_score,
    #     global_step_transform=global_step_from_engine(trainer, Events.ITERATION_COMPLETED),
    #     filename_pattern='{filename_prefix}_{global_step}_{score_name}={score}.{ext}'
    # )
    # validator.add_event_handler(Events.COMPLETED, best_model_handler)
    
    if args.test_every_epoch:
        @trainer.on(Events.EPOCH_COMPLETED)
        # @trainer.on(Events.EPOCH_STARTED)
        def test():
            model.train(False)  
            if args.video:
                f1_metric = F1ScoreMetric(average='micro', num_classes=1, multiclass=False, threshold=0.5)
                ap_metric = SklearnAPMetric()
                miou_metric = MovieNetMetric()
                f1_metric = f1_metric.to(args.device)
                ap_metric = ap_metric.to(args.device)
                miou_metric = miou_metric.to(args.device)
            else:
                f1_metric = F1ScoreMetric(average='micro', num_classes=1, multiclass=False, threshold=0.5)
                dts_metric = DTSMetric()
                f1_metric = f1_metric.to(args.device)
                dts_metric = dts_metric.to(args.device)
                # ap_metric = SklearnAPMetric()
                # miou_metric = MovieNetMetric()
                # ap_metric = ap_metric.to(args.device)
                # miou_metric = miou_metric.to(args.device)
            pred_res = []
            ref_res = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='test'):
                    dialog_ids, dialog_type_ids, dialog_mask, session_label_ids, session_indexs,\
                    feature_ids, feature_type_ids, feature_mask, scene_label_ids, scene_indexs, utter_lst, vid_lst = batch
                    if args.video == 0:   
                        feature_ids = feature_ids.to(args.device)
                        feature_type_ids = feature_type_ids.to(args.device)
                        feature_mask = feature_mask.to(args.device)
                        dialog_ids = dialog_ids.to(args.device)
                        # # analysis
                        # dialog_type_ids = dialog_type_ids.to(args.device)
                        # dialog_mask = dialog_mask.to(args.device)
                        dialog_type_ids = torch.cat([feature_type_ids, dialog_type_ids], dim=1)
                        dialog_mask = torch.cat([feature_mask, dialog_mask], dim=1)
                        session_indexs = [sess.to(args.device) for sess in session_indexs]
                        labels = session_label_ids.to(args.device)
                        logits = model(feature_ids, dialog_ids, dialog_mask, dialog_type_ids, seg_indexs=session_indexs)[0]
                        probs = F.softmax(logits, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        for vid, sid, pre, gt in zip(vid_lst, utter_lst, preds, labels):
                            dts_metric.update(vid, sid, pre, gt)
                            # miou_metric.update(vid, sid, pre, gt)
                        f1_metric.update(probs[:,1], labels)
                        # ap_metric.update(probs[:,1], labels)
                    else:
                        dialog_ids = dialog_ids.to(args.device)
                        dialog_type_ids = dialog_type_ids.to(args.device)
                        dialog_mask = dialog_mask.to(args.device)
                        feature_ids = feature_ids.to(args.device)
                        feature_type_ids = feature_type_ids.to(args.device)
                        feature_type_ids = torch.cat([feature_type_ids, dialog_type_ids], dim=1)
                        feature_mask = feature_mask.to(args.device)
                        feature_mask = torch.cat([feature_mask, dialog_mask], dim=1)
                        scene_indexs = [scene.to(args.device) for scene in scene_indexs]
                        labels = scene_label_ids.to(args.device)
                        logits = model(feature_ids, dialog_ids, feature_mask, feature_type_ids, seg_indexs=scene_indexs)[0]
                        probs = F.softmax(logits, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        for vid, sid, pre, gt in zip(vid_lst, utter_lst, preds, labels):
                            miou_metric.update(vid, sid, pre, gt)
                        f1_metric.update(probs[:,1], labels)
                        ap_metric.update(probs[:,1], labels)
                        

                    prob_list = probs.cpu().numpy().tolist()
                    label_list = labels.cpu().numpy().tolist()
                    pred_res.append(prob_list)
                    ref_res.append(label_list)

            output_dir = 'results/{}/'.format(args.exp)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)        
            if args.video:
                f1 = f1_metric.compute()
                ap = ap_metric.compute()
                miou = miou_metric.compute()
                print(f1.item(), ap.item(), miou.item())
                with open(os.path.join(output_dir, 'vis_res.log'), 'a') as fh:
                    fh.write('f1:{} ap:{} miou:{}\n'.format(f1.item(), ap.item(), miou.item()))
                with open(os.path.join(output_dir, 'vis_res_{}.json'.format(trainer.state.epoch)), 'w') as jh:
                    json.dump([pred_res, ref_res], jh)
            else:
                # ap = ap_metric.compute()
                # miou = miou_metric.compute()
                f1 = f1_metric.compute()
                mwd, mpk, mf1 = dts_metric.compute()
                print(f1.item(), mwd.item(), mpk.item(), mf1.item())
                with open(os.path.join(output_dir, 'res.log'), 'a') as fh:
                    fh.write('f1: {} mwd:{} mpk:{} mf1:{}\n'.format(f1.item(), mwd.item(), mpk.item(), mf1.item()))
                with open(os.path.join(output_dir, 'res_{}.json'.format(trainer.state.epoch)), 'w') as jh:
                    json.dump([pred_res, ref_res], jh)
                    
    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    tb_logger.close()
    
    if args.test:
        for ckpt in os.listdir(args.log_path):
            if 'best' in ckpt:
                best_ckpt = ckpt
        model_config = BertConfig.from_pretrained(args.log_path)                
        model = model_class.from_pretrained(args.log_path + best_ckpt, config=model_config)
        result_segs = []
        model.eval()
        if args.video:
            f1_metric = F1ScoreMetric(average='micro', num_classes=1, multiclass=False, threshold=0.5)
            ap_metric = SklearnAPMetric()
            miou_metric = MovieNetMetric()
            f1_metric = f1_metric.to(args.device)
            ap_metric = ap_metric.to(args.device)
            miou_metric = miou_metric.to(args.device)
        else:
            f1_metric = F1ScoreMetric(average='micro', num_classes=1, multiclass=False, threshold=0.5)
            dts_metric = DTSMetric()
            f1_metric = f1_metric.to(args.device)
            dts_metric = dts_metric.to(args.device)
        def eval(engine, batch):
            with torch.no_grad():
                dialog_ids, dialog_type_ids, dialog_mask, session_label_ids, session_indexs,\
                feature_ids, feature_type_ids, feature_mask, scene_label_ids, scene_indexs, utter_lst, vid_lst = batch
                if args.video == 0:   
                    feature_ids = feature_ids.to(args.device)
                    feature_type_ids = feature_type_ids.to(args.device)
                    feature_mask = feature_mask.to(args.device)
                    dialog_ids = dialog_ids.to(args.device)
                    dialog_type_ids = dialog_type_ids.to(args.device)
                    dialog_type_ids = torch.cat([feature_type_ids, dialog_type_ids], dim=1)
                    dialog_mask = dialog_mask.to(args.device)
                    dialog_mask = torch.cat([feature_mask, dialog_mask], dim=1)
                    session_indexs = [sess.to(args.device) for sess in session_indexs]
                    labels = session_label_ids.to(args.device)
                    logits = model(feature_ids, dialog_ids, dialog_mask, dialog_type_ids, seg_indexs=session_indexs)[0]
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    for vid, sid, pre, gt in zip(vid_lst, utter_lst, preds, labels):
                        dts_metric.update(vid, sid, pre, gt)
                    f1_metric.update(probs[:,1], labels)
                    
                else:
                    dialog_ids = dialog_ids.to(args.device)
                    dialog_type_ids = dialog_type_ids.to(args.device)
                    dialog_mask = dialog_mask.to(args.device)
                    feature_ids = feature_ids.to(args.device)
                    feature_type_ids = feature_type_ids.to(args.device)
                    feature_type_ids = torch.cat([feature_type_ids, dialog_type_ids], dim=1)
                    feature_mask = feature_mask.to(args.device)
                    feature_mask = torch.cat([feature_mask, dialog_mask], dim=1)
                    scene_indexs = [scene.to(args.device) for scene in scene_indexs]
                    labels = scene_label_ids.to(args.device)
                    logits = model(feature_ids, dialog_ids, feature_mask, feature_type_ids, seg_indexs=scene_indexs)[0]
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    for vid, sid, pre, gt in zip(vid_lst, utter_lst, preds, labels):
                        miou_metric.update(vid, sid, pre, gt)
                    f1_metric.update(probs[:,1], labels)
                    ap_metric.update(probs[:,1], labels)
                    
                # save result
                prob_list = probs.cpu().numpy().tolist()
                label_list = labels.cpu().numpy().tolist()
                for vid, sid, pb, gt in zip(vid_lst, utter_lst, prob_list, label_list):
                    if vid not in result_segs:
                        result_segs[vid] = {'probs':[], 'labels':[]}
                    result_segs[vid]['probs'].append(pb)
                    result_segs[vid]['labels'].append(gt)
                
        tester = Engine(eval)
        test_bar = ProgressBar(persist=True)
        test_bar.attach(tester)
        tester.run(test_loader, max_epochs=1)
        output_dir = 'results/{}/'.format(args.exp)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output = 'results/{}/result_{}_best_ckpt{}.json'.format(args.exp, 'beam_search', best_ckpt.split('_')[1])
        with open(output) as jh:
            json.dump(result_segs, jh, indent=4)
        if args.video:
            f1 = f1_metric.compute()
            ap = ap_metric.compute()
            miou = miou_metric.compute()
            print('RESULT: F1:{:.4f}, AP:{:.4f}, mIoU:{:.4f}'.format(f1.item(), ap.item(), miou.item()))
            with open(os.path.join(output_dir, 'vis_res.log'), 'a') as fh:
                fh.write('RESULT: F1:{:.4f}, AP:{:.4f}, mIoU:{:.4f}'.format(f1.item(), ap.item(), miou.item()))
        else:
            f1 = f1_metric.compute()
            mwd, mpk, mf1 = dts_metric.compute()
            print('RESULT: F1:{:.4f}, mWinDiff:{:.4f}, mPk:{:.4f}, mF1:{:.4f}'.format(f1.item(), mwd.item(), mpk.item(), mf1.item()))
            with open(os.path.join(output_dir, 'res.log'), 'a') as fh:
                fh.write('RESULT: F1:{:.4f}, mWinDiff:{:.4f}, mPk:{:.4f}, mF1:{:.4f}'.format(f1.item(), mwd.item(), mpk.item(), mf1.item()))

if __name__ == "__main__":
    set_seed()
    train()
