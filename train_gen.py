# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
from secrets import choice
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
from tqdm import tqdm

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine, Checkpoint, DiskSaver
from ignite.metrics import RunningAverage, Average
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

from transformers import AdamW, BartTokenizer
from transformers.file_utils import (CONFIG_NAME, WEIGHTS_NAME)
from transformers.models.bart.configuration_bart import BartConfig
from transformers.utils.dummy_pt_objects import Adafactor

from model.VSBart import VSTARBARTGenerationModel
from utils.eval import evaluate
from utils.utils import bleu_n, rouge_n
from generate import batch_beam_search, batch_greedy_search
from data.gen_dataset import DataSet, collate_fn, get_dataset

SPECIAL_TOKENS = ["<s>", "</s>", "<text>", "<sep>", "<video>", "<pad>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<s>", 'eos_token': "</s>", 'additional_special_tokens': ["<text>","<sep>","<video>"], 'pad_token': "<pad>"}
MODEL_INPUTS = ["input_ids", "token_type_ids","lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids","lm_labels"]

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

def padding(seq, pad_token, limit=1020):
    max_len = max([i.size(0) for i in seq])
    max_len = min(max_len, limit)
    if len(seq[0].size()) == 1:
        result = torch.ones((len(seq), max_len)).long() * pad_token
    else:
        result = torch.ones((len(seq), max_len, seq[0].size(-1))).float()
    for i in range(len(seq)):
        result[i, :seq[i].size(0)] = seq[i][-min(limit, seq[i].size(0)):]
    return result

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def get_data_loaders_new(args, tokenizer):
    
    if args.max_history:
        train_data_path = 'inputs/full/pkls/train_resnet_rcnn_vit_{}_data.pkl'.format(args.max_history)
        valid_data_path = 'inputs/full/pkls/valid_resnet_rcnn_vit_{}_data.pkl'.format(args.max_history)
        test_data_path = 'inputs/full/pkls/test_resnet_rcnn_vit_{}_data.pkl'.format(args.max_history)
    else:
        
        train_data_path = 'inputs/full/pkls/train_resnet_rcnn_vit_data_query.pkl'
        valid_data_path = 'inputs/full/pkls/valid_resnet_rcnn_vit_data_query.pkl'
        test_data_path = 'inputs/full/pkls/test_resnet_rcnn_vit_data_query.pkl'

    if not os.path.exists(train_data_path):
        print('no preprocessed train pkl exist, start process data.')
        train_data = get_dataset(tokenizer, args.train_path, args.feature_path, args.frame_count, n_history=args.max_history)
        """
        train_data[0] dialog_list: num of dialogs * num of turns [{'vid':'','history':max 3 turns [[q],[a],...],'answer':[a],'caption':[[caption list], [summary list]]}]
        train_data[1] all_feature dict
        """
        with open(train_data_path, 'wb') as f:
            pkl.dump(train_data, f)
    else:
        print('load train data from pkl')
        with open(train_data_path, 'rb') as f:
            train_data = pkl.load(f)
    if not os.path.exists(valid_data_path):
        print('no preprocessed valid pkl exist, start process data.')
        valid_data = get_dataset(tokenizer, args.valid_path, args.feature_path, args.frame_count, test=True, n_history=args.max_history)
        """
        valid_data[0] dialog_list: num of dialogs * num of turns [{'vid':'','history':max 3 turns [[q],[a],...],'answer':[a],'caption':[[caption list], [summary list]]}]
        valid_data[1] all_feature dict
        """
        with open(valid_data_path, 'wb') as f:
            pkl.dump(valid_data, f)
    else:
        print('load valid data from pkl')
        with open(valid_data_path, 'rb') as f:
            valid_data = pkl.load(f)
    if not os.path.exists(test_data_path):
        print('no preprocessed test pkl exist, start process data.')
        test_data = get_dataset(tokenizer, args.test_path, args.feature_path, args.frame_count, test=True, n_history=args.max_history)
        """
        test_data[0] dialog_list: num of dialogs * num of turns [{'vid':'','history':max 3 turns [[q],[a],...],'answer':[a],'caption':[[caption list], [summary list]]}]
        test_data[1] all_feature dict
        """
        with open(test_data_path, 'wb') as f:
            pkl.dump(test_data, f)
    else:
        print('load test data from pkl')
        with open(test_data_path, 'rb') as f:
            test_data = pkl.load(f)

    if args.video: 
        train_dataset = DataSet(train_data[0], tokenizer, train_data[1], model=args.model, fea_type=args.fea_type)
        valid_dataset = DataSet(valid_data[0], tokenizer, valid_data[1], model=args.model, fea_type=args.fea_type)
        test_dataset = DataSet(test_data[0], tokenizer, test_data[1], train=False, model=args.model, fea_type=args.fea_type)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=4, shuffle=(not args.distributed), collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True, fea_type=args.fea_type))
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=4, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True, fea_type=args.fea_type))
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=4, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True, fea_type=args.fea_type))
    else:
        train_dataset = DataSet(train_data[0], tokenizer, None, model=args.model)
        valid_dataset = DataSet(valid_data[0], tokenizer, None, train=False, model=args.model)
        test_dataset = DataSet(test_data[0], tokenizer, None, train=False, model=args.model)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=4, shuffle=(not args.distributed), collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=None))
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=4, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=None))
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=4, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=None))
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

def train():
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, default="inputs/full/train.json", help="Path of the trainset")
    parser.add_argument("--valid_path", type=str, default="inputs/full/valid.json", help="Path of the validset")
    parser.add_argument("--test_path", type=str, default="inputs/full/test.json", help="Path of the testset")
    parser.add_argument("--feature_path", type=str, default="inputs/feature/", help="Path of the feature")
    parser.add_argument("--frame_count", type=str, default="inputs/full/clipid2frames.json", help="Path of the frameCNT")
    parser.add_argument("--max_history", type=int, default=0, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--gpuid", type=str, default='', help='select proper gpu id')
    parser.add_argument("--model", type=str, default='bart', help='Pretrained Model name')
    parser.add_argument('--video', type=int, default=0, help='if use video: 1 use 0 not')
    parser.add_argument('--exp_set', type=str, default='_test')
    parser.add_argument('--wait', type=str, default='', help='waiting gpu id')
    parser.add_argument('--seg_embed', type=int, default=0)
    parser.add_argument('--seg_loss', type=int, default=0)
    parser.add_argument('--seg_only', type=int, default=0) # 1: bart encoder as segment model
    parser.add_argument('--test', type=int, default=1, choices=[0, 1])
    parser.add_argument('--test_every_epoch', type=int, default=1, choices=[0, 1])
    parser.add_argument('--fea_type', type=str, default='resnet')
    # generage setting
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size")
    parser.add_argument("--max_length", type=int, default=18, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=6, help="Minimum length of the output utterances")
    parser.add_argument("--penalty", type=float, default=1.0, help="elngth penalty")
    args = parser.parse_args()
    
    args.valid_batch_size = args.train_batch_size
    args.test_batch_size = args.train_batch_size
    args.gradient_accumulation_steps = 32 // args.train_batch_size
    args.model = 'bart'
    # exp_set = '_lm'
    exp_set = args.exp_set
    args.exp = args.model + exp_set
    args.log_path = 'ckpts/' + args.exp + '/'
    args.tb_path = 'tb_logs/' + args.exp + '/'
    # args.device = 'cpu'
    if args.device == 'cuda':
        if args.wait != '':
            args.gpuid = get_gpu_id(12000, args.wait)
        else:
            if args.gpuid == '':
                args.gpuid = get_gpu_id(12000)
        
        # args.train_batch_size = 2
        # args.valid_batch_size = 2
        # args.gpuid = '5'
        # args.gpuid = ''
        if args.gpuid != '':
            args.device = 'cuda:' + args.gpuid

    # select model
    if args.model == 'bart':
        args.model_checkpoint = "prev_trained_model/bart"
    elif args.model == 'bart-medium':
        args.model_checkoint = 'prev_trained_model/bart-medium'
    elif args.model == 'bart-large':
        args.model_checkpoint = "prev_trained_model/bart-large"
    else:
        raise ValueError('NO IMPLEMENTED MODEL!')


    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.tb_path):
        os.makedirs(args.tb_path)
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    
    tokenizer_class = BartTokenizer
    if args.fea_type == 'resnet':
        d_feature = 1000
    elif args.fea_type == 'rcnn':
        d_feature = 2048
    elif args.fea_type == 'vit':
        d_feature = 768
    
    model_class = VSTARBARTGenerationModel
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint, d_feature=d_feature)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    # optimizer = Adafactor(model.parameters(), lr=args.lr, scale_parameter=True, relative_step=True, warmup_init=True,)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    args.eos = model.config.decoder_start_token_id
    args.pad = model.config.pad_token_id
    # dict for validation
    with open('inputs/full/valid_ref.json') as jh:
        valid_ref = json.load(jh)
    with open('inputs/full/test_ref.json') as jh:
        test_ref = json.load(jh)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets for Bart")
    train_loader, valid_loader, test_loader = get_data_loaders_new(args, tokenizer)

    
    # Training function and trainer
    def update(engine, batch):

        # batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        encoder_input_ids, encoder_token_type_ids, encoder_input_mask, decoder_input_ids, decoder_token_type_ids, decoder_input_mask, labels, feature, type_labels, \
            vid_list, scene, session, seg_label = batch
        encoder_input_ids = encoder_input_ids.to(args.device)
        encoder_token_type_ids = encoder_token_type_ids.to(args.device)
        encoder_input_mask = encoder_input_mask.to(args.device)
        decoder_input_ids = decoder_input_ids.to(args.device)
        decoder_token_type_ids = decoder_token_type_ids.to(args.device)
        decoder_input_mask = decoder_input_mask.to(args.device)
        labels = labels.to(args.device)
        
        if args.video:
            if args.fea_type == 'vit':
                feature = [fea.to(args.device) for fea in feature]
            else:
                feature = feature.to(args.device)
            scene = scene.to(args.device)
        else:
            feature = None

        type_labels = type_labels.to(args.device)
        session = session.to(args.device)
        seg_label = seg_label.to(args.device)
        
        # optimize Bart
        model.train()
        if args.seg_loss:
            if args.seg_only:
                reply_loss = model(encoder_input_ids, video_ids=feature, token_type_ids=None, labels=(labels, feature), attention_mask=encoder_input_mask, \
                    scene=None, session=None, seg_label=seg_label, decoder_attention_mask=decoder_input_mask, type_labels=None, mode='seg')[0]
            else:
                reply_loss = model(encoder_input_ids, video_ids=feature, token_type_ids=None, labels=(labels, feature), attention_mask=encoder_input_mask, \
                    scene=None, session=None, seg_label=seg_label, decoder_attention_mask=decoder_input_mask, type_labels=None, mode='reply')[0]
        else:
            if args.seg_embed:
                reply_loss = model(encoder_input_ids, video_ids=feature, token_type_ids=None, labels=(labels, feature), attention_mask=encoder_input_mask, \
                    scene=scene, session=session, decoder_attention_mask=decoder_input_mask, type_labels=None, mode='reply')[0]
            else:
                reply_loss = model(encoder_input_ids, video_ids=feature, token_type_ids=None, labels=(labels, feature), attention_mask=encoder_input_mask, \
                    scene=None, session=None, decoder_attention_mask=decoder_input_mask, type_labels=None, mode='reply')[0]

        loss = reply_loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    def valid(engine, batch):
        model.eval()
        with torch.no_grad():
            encoder_input_ids, encoder_token_type_ids, encoder_input_mask, decoder_input_ids, decoder_token_type_ids, decoder_input_mask, labels, feature, type_labels, \
                vid_list, scene, session, seg_label = batch
            encoder_input_ids = encoder_input_ids.to(args.device)
            encoder_token_type_ids = encoder_token_type_ids.to(args.device)
            encoder_input_mask = encoder_input_mask.to(args.device)
            decoder_input_ids = decoder_input_ids.to(args.device)
            decoder_token_type_ids = decoder_token_type_ids.to(args.device)
            decoder_input_mask = decoder_input_mask.to(args.device)
            labels = labels.to(args.device)
            if args.video:
                if args.fea_type == 'vit':
                    feature = [fea.to(args.device) for fea in feature]
                else:
                    feature = feature.to(args.device)
                scene = scene.to(args.device)
            type_labels = type_labels.to(args.device)
            session = session.to(args.device)
            bsz = encoder_input_ids.size(0)
            decoder_input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(['</s>', '<sep>'])).expand(bsz, 2).to(args.device)
            if args.seg_embed:
                hyp_lst = batch_greedy_search(encoder_input_ids, encoder_input_mask, decoder_input_ids, tokenizer, model, args, scene=scene, session=session, video=feature)
            else:
                hyp_lst = batch_greedy_search(encoder_input_ids, encoder_input_mask, decoder_input_ids, tokenizer, model, args, video=feature)
            hyp_lst = tokenizer.batch_decode(hyp_lst, skip_special_tokens=True)
            rouge = 0
            # bleu = 0
            for k, vid in enumerate(vid_list):
                rouge += rouge_n(valid_ref[vid][-1].split(), hyp_lst[k].split())
                # bleu += bleu_n(valid_ref[vid][-1].split(), hyp_lst[k].split())
            rouge /= bsz
            # bleu /= bsz
        return rouge
        # return rouge, bleu
        

    trainer = Engine(update)
    validator = Engine(valid)
    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: validator.run(valid_loader))

    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics 
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x).attach(validator, "rouge")


    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        eval_pbar = ProgressBar(persist=True)
        eval_pbar.attach(validator, metric_names=['rouge'])
        # evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=args.tb_path)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        # tb_logger.attach(evaluator, log_handler=OutputHandler(tag='validation', metric_names=["rouge", "tb_rouge", "bleu", "tb_bleu"], global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)
        tb_logger.attach(validator, log_handler=OutputHandler(tag='validation', metric_names=["rouge"], global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(args.log_path, 'checkpoint', n_saved=args.n_epochs ,require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation
        
        torch.save(args, args.log_path + 'model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(args.log_path, CONFIG_NAME))
        tokenizer.save_vocabulary(args.log_path)

        best_score = Checkpoint.get_default_score_fn('rouge')
        best_model_handler = Checkpoint(
            {'mymodel': getattr(model, 'module', model)},
            filename_prefix='best',
            save_handler=DiskSaver(args.log_path, create_dir=True, require_empty=False),
            score_name='rouge',
            score_function=best_score,
            global_step_transform=global_step_from_engine(trainer, Events.ITERATION_COMPLETED),
            filename_pattern='{filename_prefix}_{global_step}_{score_name}={score}.{ext}'
        )
        validator.add_event_handler(Events.COMPLETED, best_model_handler)
    
    if args.test_every_epoch:
        # @trainer.on(Events.ITERATION_STARTED)
        @trainer.on(Events.EPOCH_COMPLETED)
        def test(engine):
            model.eval()
            result_dialogs = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='test'):
                    encoder_input_ids, encoder_token_type_ids, encoder_input_mask, decoder_input_ids, decoder_token_type_ids, decoder_input_mask, labels, feature, type_labels, \
                        vid_list, scene, session, seg_label = batch
                    encoder_input_ids = encoder_input_ids.to(args.device)
                    encoder_token_type_ids = encoder_token_type_ids.to(args.device)
                    encoder_input_mask = encoder_input_mask.to(args.device)
                    decoder_input_ids = decoder_input_ids.to(args.device)
                    decoder_token_type_ids = decoder_token_type_ids.to(args.device)
                    decoder_input_mask = decoder_input_mask.to(args.device)
                    labels = labels.to(args.device)
                    if args.video:
                        if args.fea_type == 'vit':
                            feature = [fea.to(args.device) for fea in feature]
                        else:
                            feature = feature.to(args.device)
                        scene = scene.to(args.device)
                    type_labels = type_labels.to(args.device)
                    session = session.to(args.device)
                    # reply_loss = model(text_encoder_input_ids, video_ids=feature, token_type_ids=text_encoder_token_type_ids, labels=(labels, feature), attention_mask=text_encoder_input_mask, decoder_attention_mask=decoder_input_mask, type_labels=type_labels, mode='reply', video=0)[0]
                    bsz = encoder_input_ids.size(0)
                    decoder_input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(['</s>', '<sep>'])).expand(bsz, 2).to(args.device)
                    if args.seg_embed:
                        hyp_lst = batch_beam_search(encoder_input_ids, encoder_input_mask, decoder_input_ids, tokenizer, model, args, scene=scene, session=session, video=feature)
                    else:
                        hyp_lst = batch_beam_search(encoder_input_ids, encoder_input_mask, decoder_input_ids, tokenizer, model, args, video=feature)
                    hyp_lst = tokenizer.batch_decode(hyp_lst, skip_special_tokens=True)
                    for i, vid in enumerate(vid_list):
                        gen_dia = copy.copy(test_ref[vid])
                        gen_dia.append(hyp_lst[i])
                        result_dialogs.append({'clip_id':vid, 'dialog':gen_dia})
                result = {'dialogs':result_dialogs}
                output_dir = 'results/{}/'.format(args.exp)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output = 'results/{}/result_{}_best_ckpt{}_beam{}_minl{}_maxl{}_pen{}.json'.format(args.exp, 'beam_search', engine.state.iteration, args.min_length, args.max_length, args.beam_size, args.penalty)
                with open(output, 'w') as jh:
                    json.dump(result, jh, indent=4)
                evaluate(result, output)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        # os.rename(checkpoint_handler._saved[-1][1], os.path.join(args.log_path, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()
    
    if args.test:
        for ckpt in os.listdir(args.log_path):
            if 'best' in ckpt:
                best_ckpt = ckpt
        model_config = BartConfig.from_pretrained(args.log_path, d_feature=d_feature)                
        model = model_class.from_pretrained(args.log_path + best_ckpt, config=model_config, d_feature=d_feature)
        with open('inputs/full/test_ref.json') as jh:
            test_ref = json.load(jh)
        model = model.to(args.device)
        result_dialogs = []
        model.eval()
        def eval(engine, batch):
            with torch.no_grad():
                encoder_input_ids, encoder_token_type_ids, encoder_input_mask, decoder_input_ids, decoder_token_type_ids, decoder_input_mask, labels, feature, type_labels, \
                    vid_list, scene, session, seg_label = batch
                encoder_input_ids = encoder_input_ids.to(args.device)
                encoder_token_type_ids = encoder_token_type_ids.to(args.device)
                encoder_input_mask = encoder_input_mask.to(args.device)
                decoder_input_ids = decoder_input_ids.to(args.device)
                decoder_token_type_ids = decoder_token_type_ids.to(args.device)
                decoder_input_mask = decoder_input_mask.to(args.device)
                labels = labels.to(args.device)
                if args.video:
                    if args.fea_type == 'vit':
                        feature = [fea.to(args.device) for fea in feature]
                    else:
                        feature = feature.to(args.device)
                    scene = scene.to(args.device)
                type_labels = type_labels.to(args.device)
                session = session.to(args.device)
                # reply_loss = model(text_encoder_input_ids, video_ids=feature, token_type_ids=text_encoder_token_type_ids, labels=(labels, feature), attention_mask=text_encoder_input_mask, decoder_attention_mask=decoder_input_mask, type_labels=type_labels, mode='reply', video=0)[0]
                bsz = encoder_input_ids.size(0)
                decoder_input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(['</s>', '<sep>'])).expand(bsz, 2).to(args.device)
                if args.seg_embed:
                    hyp_lst = batch_beam_search(encoder_input_ids, encoder_input_mask, decoder_input_ids, tokenizer, model, args, scene=scene, session=session, video=feature)
                else:
                    hyp_lst = batch_beam_search(encoder_input_ids, encoder_input_mask, decoder_input_ids, tokenizer, model, args, video=feature)
                hyp_lst = tokenizer.batch_decode(hyp_lst, skip_special_tokens=True)
                for i, vid in enumerate(vid_list):
                    gen_dia = copy.copy(test_ref[vid])
                    gen_dia.append(hyp_lst[i])
                    result_dialogs.append({'clip_id':vid, 'dialog':gen_dia})
        tester = Engine(eval)
        test_bar = ProgressBar(persist=True)
        test_bar.attach(tester)
        tester.run(test_loader, max_epochs=1)
        result = {'dialogs':result_dialogs}
        output_dir = 'results/{}/'.format(args.exp)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output = 'results/{}/result_{}_best_ckpt{}_beam{}_minl{}_maxl{}_pen{}.json'.format(args.exp, 'beam_search', best_ckpt.split('_')[1], args.min_length, args.max_length, args.beam_size, args.penalty)
        with open(output, 'w') as jh:
            json.dump(result, jh, indent=4)
        evaluate(result, output)
        
            

if __name__ == "__main__":
    set_seed()
    train()
