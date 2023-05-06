from curses import noecho
import os
import json
import logging
import random
import time
import copy
from tqdm import tqdm
import pickle as pkl
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    BartTokenizer, 
    BartConfig, 
    LogitsProcessorList, 
    MinLengthLogitsProcessor, 
    BeamSearchScorer, 
    TopKLogitsWarper, 
    TemperatureLogitsWarper,
    NoRepeatNGramLogitsProcessor)
from transformers.modeling_outputs import BaseModelOutput

from model.VSBart import VSTARBARTGenerationModel
from data.dataset import DataSet, collate_fn, get_dataset, build_input_from_segments
from utils.eval import evaluate
from utils.utils import rouge_n

SPECIAL_TOKENS = ["<s>", "</s>", "<text>", "<sep>", "<video>", "<pad>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<s>", 'eos_token': "</s>", 'additional_special_tokens': ["<text>","<sep>","<video>"], 'pad_token': "<pad>"}
MODEL_INPUTS = ["input_ids", "token_type_ids","lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids","lm_labels"]

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

def batch_sample(encoder_input_ids, encoder_input_mask, dec_ids, tokenizer, model, args, scene=None, session=None, video=None):

    # add encoder_outputs to model keyword arguments
    model_kwargs = {
        "encoder_outputs": model.get_encoder()(
            encoder_input_ids, 
            video_ids=video,
            attention_mask=encoder_input_mask, 
            session=session,
            scene=scene,
            return_dict=True
        ),
        "attention_mask": encoder_input_mask
        # "decoder"
    }
    

    # instantiate logits processors
    logits_processor = LogitsProcessorList(
        [
            MinLengthLogitsProcessor(1, eos_token_id=model.config.eos_token_id),
        ]
    )
    logit_warper = LogitsProcessorList(
        [
            TopKLogitsWarper(50),
            TemperatureLogitsWarper(0.7)
        ]
    )

    outputs = model.sample(dec_ids, logits_processor=logits_processor, logit_warper=logit_warper, pad_token_id=args.pad, max_length=args.max_length, **model_kwargs)
    # hyp = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs
    
    

def batch_greedy_search(encoder_input_ids, encoder_input_mask, dec_ids, tokenizer, model, args, scene=None, session=None, video=None):
    # add encoder_outputs to model keyword arguments
    model_kwargs = {
        "encoder_outputs": model.get_encoder()(
            encoder_input_ids, 
            video_ids=video,
            attention_mask=encoder_input_mask, 
            session=session,
            scene=scene,
            return_dict=True
        ),
        "attention_mask": encoder_input_mask
        # "decoder"
    }
    

    # instantiate logits processors
    logits_processor = LogitsProcessorList(
        [
            MinLengthLogitsProcessor(args.min_length, eos_token_id=model.config.eos_token_id),
        ]
    )

    outputs = model.greedy_search(dec_ids, logits_processor=logits_processor, pad_token_id=args.pad, max_length=args.max_length, **model_kwargs)
    # hyp = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs
    

def batch_beam_search(encoder_input_ids, encoder_input_mask, dec_ids, tokenizer, model, args, scene=None, session=None, video=None):
    # add encoder_outputs to model keyword arguments
    if video is not None:
        if type(video) is list:
            video_ids = []
            for v in video:
                video_ids += [v] * args.beam_size
            # video_ids = [v.repeat_interleave(args.beam_size, dim=0) for v in video]
        else:
            video_ids = video.repeat_interleave(args.beam_size, dim=0)
    else:
        video_ids = None
    if scene is not None:
        scene_ids = scene.repeat_interleave(args.beam_size, dim=0)
    else:
        scene_ids = None
    if session is not None:
        session_ids = session.repeat_interleave(args.beam_size, dim=0)
    else:
        session_ids = None
    model_kwargs = {
        "encoder_outputs": model.get_encoder()(
            encoder_input_ids.repeat_interleave(args.beam_size, dim=0), 
            video_ids=video_ids,
            attention_mask=encoder_input_mask.repeat_interleave(args.beam_size, dim=0), 
            scene=scene_ids,
            session=session_ids,
            return_dict=True
        ),
        "attention_mask": encoder_input_mask.repeat_interleave(args.beam_size, dim=0)
        # "decoder"
    }
    
    # instantiate beam scorer
    beam_scorer = BeamSearchScorer(
        batch_size=encoder_input_ids.size(0),
        max_length=args.max_length,
        length_penalty=args.penalty,
        num_beams=args.beam_size,
        # do_early_stopping=True,
        device=model.device,
    )

    # instantiate logits processors
    logits_processor = LogitsProcessorList(
        [
            MinLengthLogitsProcessor(args.min_length, eos_token_id=model.config.eos_token_id),
            # NoRepeatNGramLogitsProcessor(2)
        ]
    )
    outputs = model.beam_search(dec_ids.repeat_interleave(args.beam_size, dim=0), beam_scorer, logits_processor=logits_processor, pad_token_id=args.pad, max_length=args.max_length, **model_kwargs)
    # hyp = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs

# main
if __name__ =="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="bart", help="Pretrained Model type")
    parser.add_argument("--max_history", type=int, default=0, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size")
    parser.add_argument("--max_length", type=int, default=18, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=6, help="Minimum length of the output utterances")
    parser.add_argument("--penalty", type=float, default=1.0, help="elngth penalty")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--test_set", type=str, default="inputs/full/test.json")
    parser.add_argument("--output", type=str, default="result.json")
    parser.add_argument("--ckptid", type=str, help='ckpt selected for test')
    parser.add_argument("--gpuid", type=str, default='0', help='gpu id')
    parser.add_argument("--log", type=bool, default=False, help='if logging info')
    parser.add_argument('--exp_set', type=str, default='test')
    parser.add_argument('--video', type=int, default=0)
    parser.add_argument('--sess', type=int, default=0)
    parser.add_argument('--eval', type=int, default=1)
    parser.add_argument('--gen_type', type=str, default='beam_search', choices=['sample', 'greedy_search', 'beam_search'])
    args = parser.parse_args()

    set_seed()
    st = time.time()

    exp_set = args.exp_set
    model_checkpoint = 'ckpts/' + args.model + args.exp_set + '/'
    output_dir = 'results/' + args.model + exp_set
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.output = output_dir + '/result_' + args.gen_type + '_' + args.ckptid  + '_' + str(args.beam_size) + '_' + str(args.min_length) + '_' + str(args.penalty) + '.json'
    # args.output = output_dir + '/result_{}.json'.format(args.gen_type)
    if args.device == 'cuda':
        args.device = 'cuda:' + args.gpuid
    
    # setproctitle.setproctitle("task_{}_ckpt_{}_beam_{}_min_{}_pen_{:.1f}".format(args.task, args.ckptid, args.beam_size, args.min_length, args.penalty))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))

    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s')
 
    logging.info('Loading model params from ' + model_checkpoint)
    
    
    if 'bart' in args.model:
        tokenizer_class = BartTokenizer
        model_class = VSTARBARTGenerationModel
        model_config = BartConfig.from_pretrained(model_checkpoint)
    else:
        print('No pre-trained model: {}!'.format(args.model))
        raise ValueError

    tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model = model_class.from_pretrained(model_checkpoint+"checkpoint_mymodel_" + args.ckptid + ".pt", config=model_config)
    model.to(args.device)
    model.eval()
    args.pad = model.config.pad_token_id

    with open('inputs/full/test_ref.json') as jh:
        test_ref = json.load(jh)
    if args.log:
        logging.info('Loading test data from ' + args.test_set)
    test_data = json.load(open(args.test_set,'r'))
    if args.max_history:
        test_dataset_path = 'inputs/full/pkls/test_20_data.pkl'
    else:
        test_dataset_path = 'inputs/full/pkls/test_resnet_data.pkl'
    if not os.path.exists(test_dataset_path):
        test_dataset = get_dataset(tokenizer, args.test_set, test=True, n_history=args.max_history)
        with open(test_dataset_path, 'wb') as f:
            pkl.dump(test_dataset, f)
    else:
        with open(test_dataset_path, 'rb') as f:
            test_dataset = pkl.load(f)
    if args.video:
        test_ds = DataSet(test_dataset[0], tokenizer, test_dataset[1], train=False, model=args.model)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
    else:
        test_ds = DataSet(test_dataset[0], tokenizer, None, train=False, model=args.model)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=None))
    model.eval()
    with torch.no_grad():
        result_dialogs = []
        for batch in tqdm(test_loader, desc='generate'):
            encoder_input_ids, encoder_token_type_ids, encoder_input_mask, decoder_input_ids, decoder_token_type_ids, decoder_input_mask, labels, i3d, type_labels, \
                    vid_list, scene, session, seg_label = batch
            encoder_input_ids = encoder_input_ids.to(args.device)
            encoder_token_type_ids = encoder_token_type_ids.to(args.device)
            encoder_input_mask = encoder_input_mask.to(args.device)
            decoder_input_ids = decoder_input_ids.to(args.device)
            decoder_token_type_ids = decoder_token_type_ids.to(args.device)
            decoder_input_mask = decoder_input_mask.to(args.device)
            labels = labels.to(args.device)
            if args.video:
                i3d = i3d.to(args.device)
                scene = scene.to(args.device)
            type_labels = type_labels.to(args.device)
            session = session.to(args.device)
            bsz = encoder_input_ids.size(0)
            decoder_input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(['</s>', '<sep>'])).expand(bsz, 2).to(args.device)
            if args.gen_type == 'sample':
                hyp_lst = batch_sample(encoder_input_ids, encoder_input_mask, decoder_input_ids, tokenizer, model, args, scene=None, session=None, video=i3d)
            elif args.gen_type == 'greedy_search':
                hyp_lst = batch_greedy_search(encoder_input_ids, encoder_input_mask, decoder_input_ids, tokenizer, model, args, scene=None, session=None, video=i3d)
            elif args.gen_type == 'beam_search':
                if args.sess:
                    hyp_lst = batch_beam_search(encoder_input_ids, encoder_input_mask, decoder_input_ids, tokenizer, model, args, scene=scene, session=session, video=i3d)
                else:
                    hyp_lst = batch_beam_search(encoder_input_ids, encoder_input_mask, decoder_input_ids, tokenizer, model, args, scene=None, session=None, video=i3d)
            hyp_lst = tokenizer.batch_decode(hyp_lst, skip_special_tokens=True)

            for i, vid in enumerate(vid_list):
                gen_dia = test_ref[vid]
                gen_dia.append(hyp_lst[i])
                # rouge = rouge_n(test_ref[vid][-1], hyp_lst[i])
                result_dialogs.append({'clip_id':vid, 'dialog':gen_dia})

    result = {'dialogs': result_dialogs}
    if args.output:
        if args.log:
            logging.info('writing results to ' + args.output)
        json.dump(result, open(args.output, 'w'), indent=4)
    if args.eval:
        evaluate(result, args.output)
    if args.log:
        logging.info('done')
    print('generate take {} mins'.format((time.time()-st)//60))
