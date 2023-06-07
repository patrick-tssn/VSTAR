import os
import json
import pickle
import logging
import copy
import random
from itertools import chain
from pandas import IndexSlice
from tqdm import tqdm

from nltk import text

import numpy as np

import torch
import torch.utils.data
from torch.utils.data import Dataset


# from train import SPECIAL_TOKENS, MODEL_INPUTS, PADDED_INPUTS
SPECIAL_TOKENS = ["<s>", "</s>", "<text>", "<sep>", "<video>", "<pad>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<s>", 'eos_token': "</s>", 'additional_special_tokens': ["<text>","<sep>","<video>"], 'pad_token': "<pad>"}
MODEL_INPUTS = ["input_ids", "token_type_ids","lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids","lm_labels"]

def tokenize(obj,tokenizer):
    if isinstance(obj, str): # 对 string 格式的文本 tokenize
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict): # 对字典格式的文本 tokenize -> key:tokenized value
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj) # 其他情况

def get_dataset(tokenizer, data_file, feature_path=None, frame_count_path=None, test=False, n_history=None):
    """
    input data format: read datafile: {"image_id": "", "summary": "", "dialogs""[{"answer":"","question":""}], "caption":""}
    output data format: dialog_list: num of dialogs * num of turns [{'vid':'','history':max 3 turns [[q],[a],...[cur q]],'answer':[a],'caption':[[caption list], [summary list]]}]
                        all_feature: {'vggish':{'vid':(filepath,filepath)},'i3d_flow':{},'i3d_rgb':{}}
    """
    dialog_data = json.load(open(data_file, 'r'))
    dialog_list = []
    vid_set = set()
    for dialog in tqdm(dialog_data['dialogs'], desc='Process Data'): # dict {}
        qa = [tokenize(d, tokenizer) for d in dialog['dialog']] # [[dialog i id list], [], ...]
        qa_sen_lst = [d for d in dialog['dialog']]
        vid = dialog["clip_id"] # vid
        vid_set.add(vid) # vid set
        scene_lst = dialog['scene']
        session_lst = dialog['session']
        maxs = session_lst[-1] + 1

        if n_history:
            qalist = []
            history = []
            session = []
            sess = []
            if test:
                it = range(len(qa)-1, len(qa))
                for n in range((len(qa)-1)):
                    qalist.append(qa[n])
                    sess.append([min(30, maxs-session_lst[n])] * len(qa[n]))
                history = qalist[max(-len(qalist), -n_history):]
                session = sess[max(-len(qalist), -n_history):]
            else:
                it = range(min(5, len(qa)-1), len(qa))
                qalist = qa[0:min(5, len(qa)-1)]
                history = qa[0:min(5, len(qa)-1)]
                sess = [[min(30, maxs-session_lst[k])]*len(qa[k]) for k in range(min(5, len(qa)-1))]
                session = [[min(30, maxs-session_lst[k])]*len(qa[k]) for k in range(min(5, len(qa)-1))]
            for n in it:
                item = {'vid':vid, 'history':history, 'answer':qa[n], 'scene':scene_lst, 'session':session, 'session_label':session_lst, 'gold_ans':qa_sen_lst[n]}
                dialog_list.append(item)
                sess.append([min(30, maxs-session_lst[n])]*len(qa[n]))
                qalist.append(qa[n])
                history = qalist[max(-len(qalist), -n_history):]
                session = sess[max(-len(qalist), -n_history):]
                assert len(history) == len(session)
        else:
            # session = []
            # for n in range(len(qa)-1):
            #     session.append([min(30, maxs-session_lst[n])] * len(qa[n]))
            # item = {'vid':vid, 'history':qa[:-1], 'answer':qa[-1], 'scene':scene_lst, 'session':session, 'session_label':session_lst, 'gold_ans':qa_sen_lst[-1]}
            # dialog_list.append(item)

            session = []
            session.append([min(30, maxs-session_lst[-2])] * len(qa[-2]))
            item = {'vid':vid, 'history':[qa[-2]], 'answer':qa[-1], 'scene':scene_lst, 'session':session, 'session_label':session_lst, 'gold_ans':qa_sen_lst[-1]}
            dialog_list.append(item)

    if feature_path is not None:
        feature_dct = {}
        with open(frame_count_path) as jh:
            frame_count = json.load(jh)
        for vid in vid_set:
            resnet_filename = os.path.join(feature_path, 'resnet', vid.split('_clip')[0], vid+'.npy') # fdir/epi/clip
            rcnn_filename = os.path.join(feature_path, 'rcnn', vid.split('_clip')[0], vid+'.npy') # fdir/epi/clip
            vit_filename = os.path.join(feature_path, 'vit', vid.split('_clip')[0], vid+'.npz') # fdir/epi/clip
            feature_dct[vid] = {'resnet':resnet_filename, 'rcnn':rcnn_filename, 'vit':vit_filename, 'frame':frame_count[vid]}
        # # test feature_path
        # np.load(list(feature_dct.values())[-1]['feature'])
        return dialog_list, feature_dct
        """
        dialog_list: [{'vid':'','history':'','answer':'','scene':'', 'session:'', 'gold_ans':''}]
        all_feature: {'feature':'', 'frame':[]}
        """
    return dialog_list


class DataSet(Dataset):
    def __init__(self, dialogs, tokenizer, features=None, train=True, model='bart', fea_type='resnet'):
        self.dialogs = dialogs # dialog_list
        self.features = features # all_feature
        self.tokenizer = tokenizer
        self.train = train
        self.model = model
        self.eos = [tokenizer.eos_token_id]
        self.fea_type = fea_type

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index):
        dialog = self.dialogs[index]
        vid = dialog['vid'] # 'vid'
        his = self.dialogs[index]['history'] # [[u_1],[u_2],...[u_n-1]] 
        ans = self.dialogs[index]['answer'] # [[u_n]]
        scene_lst = self.dialogs[index]['scene'] # [1, 0, ...]
        session_lst = self.dialogs[index]['session'] # [[...], ..., [1, 1, 1, ...]]
        sess_label = self.dialogs[index]['session_label'] # [1, 0, ...]
        assert len(session_lst) == len(his)

        # label
        session_label = []
        prev = 1
        for s in sess_label:
            session_label.append(int(s!=prev))
            prev = s
        scene_label = []
        prev = 1
        for s in scene_lst:
            scene_label.append(int(s!=prev))
            prev = s
        
        if self.train:
            encoder_instance, decoder_instance, _, _ = build_input_from_segments(his, ans, self.tokenizer, session=session_lst, session_label=session_label, video=False, train=self.train, model=self.model)
        else:
            encoder_instance, decoder_instance, _, _ = build_input_from_segments(his, [], self.tokenizer, session=session_lst, session_label=session_label, video=False, train=self.train, model=self.model)
        encoder_input_ids = torch.Tensor(encoder_instance["input_ids"]).long() 
        decoder_input_ids = torch.Tensor(decoder_instance["input_ids"]).long()
        encoder_token_type_ids = torch.Tensor(encoder_instance["token_type_ids"]).long() #
        decoder_token_type_ids = torch.Tensor(decoder_instance["token_type_ids"]).long() #
        lm_labels = torch.Tensor(decoder_instance["lm_labels"]).long()
        type_labels = torch.Tensor(decoder_instance["type_labels"]).long()

        # encoder_his_ids = torch.Tensor(encoder_instance["his_ids"]).long()
        # encoder_his_type_ids = torch.Tensor(encoder_instance["his_type_ids"]).long()
        # encoder_query_ids = torch.Tensor(encoder_instance["query_ids"]).long()
        # encoder_query_type_ids = torch.Tensor(encoder_instance["query_type_ids"]).long()

        session_lst = torch.Tensor(encoder_instance['session']).long()
        session_label = torch.Tensor(encoder_instance['session_label']).long()

        if self.features is not None:
            if self.fea_type == 'resnet':
                """
                resnet: 1000 dim
                n * 2048
                """
                feature = np.load(self.features[vid]['resnet']) 
                frame_count = self.features[vid]['frame']
                # 1 frame/utter.
                sample_count = 1
                fcnt = 0
                sample_feature = []
                scene_label_lst = []
                scene = []
                max_pos = scene_lst[-1]
                for i in range(len(frame_count)):
                    if frame_count[i] < sample_count:
                        selected_feature = np.repeat(np.expand_dims(feature[fcnt, :], axis=0), sample_count, axis=0)
                    else:
                        index_lst = [fcnt+x for x in range(0, frame_count[i], frame_count[i]//sample_count)][:sample_count]
                        selected_feature = feature[index_lst, :]
                    # selected_feature = feature[fcnt+random.randint(0, frame_count[i]-1),:]
                    sample_feature.append(selected_feature)
                    fcnt += frame_count[i]
                    scene += [min(30, max_pos-scene_lst[i])]*sample_count # add scene embedding
                    scene_label_lst += [scene_label[i]]
                assert len(scene_label_lst) == len(sample_feature)
                assert len(scene_label_lst)*sample_count == len(scene)
                # sample_feature = np.stack(sample_feature, axis=0)
                sample_feature = np.concatenate(sample_feature, axis=0)
                sample_feature = torch.from_numpy(sample_feature).float() # (utter_num, feature_dim)
                scene_label = torch.Tensor(scene_label_lst).long()
                scene_lst = torch.Tensor(scene).long()
            elif self.fea_type == 'rcnn':
                """
                rcnn: 2048 dim
                {
                    'feature': (9*n, 2048)
                    'size': (9*n) [pix size]
                    'box': (9*n, 4)
                    'obj_id': (9*n
                    'obj_conf': (9*n
                    'obj_num': (9*n
                }
                """
                feature = torch.from_numpy(np.load(self.features[vid]['rcnn'], allow_pickle=True).item()['feature'])
                sample_feature = []
                scene_label_lst = []
                scene = []
                max_pos = scene_lst[-1]
                for i in range(feature.size(0)//9):
                    sample_feature.append(feature[i*9:(i+1)*9])
                    sample_feature.append(feature[i*9:(i+1)*9].mean(dim=0).unsqueeze(0))
                    scene += [min(30, max_pos-scene_lst[i])] * 10
                    scene_label_lst += [-1] * 9 + [scene_label[i]]
                assert len(scene_label_lst) == len(scene)
                sample_feature = torch.cat(sample_feature, dim=0)
                scene_label = torch.Tensor(scene_label_lst).long()
                scene_lst = torch.Tensor(scene).long()

            elif self.fea_type == 'vit':  
                """
                vit: (n, 3, 244, 244)
                """
                patches = 9
                try:
                    feature = torch.from_numpy(np.load(self.features[vid]['vit'], allow_pickle=True)['feature'])
                except:
                    print(self.feature[vid]['vit'])
                    raise ValueError
                scene_label_lst = []
                scene = []
                max_pos = scene_lst[-1]
                for i in range(feature.size(0)):
                    scene += [min(30, max_pos-scene_lst[i])] * (patches+1)
                    scene_label_lst += [-1] * patches + [scene_label[i]]
                assert len(scene_label_lst) == len(scene)
                sample_feature = feature
                scene_label = torch.Tensor(scene_label_lst).long()
                scene_lst = torch.Tensor(scene).long()

            else:
                raise ValueError('NO feature type implemented')
        else:
            sample_feature = None
            scene_label = None

        # return encoder_input_ids, encoder_token_type_ids, decoder_input_ids, decoder_token_type_ids, lm_labels, i3d, type_labels, vid, \
        #     scene_lst, session_lst, encoder_his_ids, encoder_his_type_ids, encoder_query_ids, encoder_query_type_ids
        return encoder_input_ids, encoder_token_type_ids, decoder_input_ids, decoder_token_type_ids, lm_labels, sample_feature, type_labels, vid, \
            scene_lst, session_lst, scene_label, session_label


def collate_fn(batch, pad_token, features=None, fea_type='resnet'):

    def padding(seq, pad_token, limit=1020, vis=False):
        max_len = max([i.size(0) for i in seq])
        max_len = min(max_len, limit)
        if vis:
            result = torch.ones((len(seq), max_len, seq[0].size(-1))).float() * pad_token
        else:
            result = torch.ones((len(seq), max_len)).long() * pad_token
        for i in range(len(seq)):
            result[i, :seq[i].size(0)] = seq[i][-min(limit, seq[i].size(0)):]
        return result

    # encoder_input_ids_list, encoder_token_type_ids_list, decoder_input_ids_list, decoder_token_type_ids_list,lm_labels_list, type_labels_list, i3d_list, \
    #     vid_list, scene_lists, session_lists, encoder_his_lst, encoder_his_type_lst, encoder_query_lst, encoder_query_type_lst = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    encoder_input_ids_list, encoder_token_type_ids_list, decoder_input_ids_list, decoder_token_type_ids_list,lm_labels_list, type_labels_list, feature_list, \
        vid_list, scene_lists, session_lists, scene_label_lists, session_label_lists = [], [], [], [], [], [], [], [], [], [], [], []

    for i in batch:
        encoder_input_ids_list.append(i[0])
        encoder_token_type_ids_list.append(i[1])
        decoder_input_ids_list.append(i[2])
        decoder_token_type_ids_list.append(i[3])
        lm_labels_list.append(i[4])
        feature_list.append(i[5])
        type_labels_list.append(i[6])
        vid_list.append(i[7]) # ['vid', ...]
        scene_lists.append(i[8]) # [[trip], ...]
        session_lists.append(i[9])
        scene_label_lists.append(i[10])
        session_label_lists.append(i[11])
        # encoder_his_lst.append(i[10])
        # encoder_his_type_lst.append(i[11])
        # encoder_query_lst.append(i[12])
        # encoder_query_type_lst.append(i[13])
        
    if features is not None:
        if fea_type != 'vit':
            limit = 1024 - max([min(x.size(0), 500) for x in feature_list]) - 1
        else:
            limit = 1024 - max([min(x.size(0)*10, 500) for x in feature_list]) - 1
    else:
        limit = 1020
    encoder_input_ids = padding(encoder_input_ids_list, pad_token, limit)
    encoder_token_type_ids = padding(encoder_token_type_ids_list, pad_token, limit)
    decoder_input_ids = padding(decoder_input_ids_list, pad_token, limit)
    decoder_token_type_ids = padding(decoder_token_type_ids_list, pad_token, limit)
    lm_labels = padding(lm_labels_list, -1, limit)
    type_labels = padding(type_labels_list, -1, limit)
    session = padding(session_lists, 0, limit)
    session_label = padding(session_label_lists, -1, limit)
    encoder_input_mask = encoder_input_ids != pad_token
    decoder_input_mask = decoder_input_ids != pad_token
    seg_label = session_label
    tmp_len_lst = []
    if features is not None:
        scene = padding(scene_lists, 0, limit=500)
        scene_mask = padding(scene_lists, -1, limit=500)
        scene_label = padding(scene_label_lists, -1, limit=500)
        seg_label = torch.cat([scene_label, seg_label], dim=1)
        if fea_type != 'vit':
            feature = padding(feature_list, 0, vis=True, limit=500)
            feature_mask = torch.sum(feature, dim=2) != 0
            encoder_input_mask = torch.cat([feature_mask, encoder_input_mask], dim=1) # bz, seq_len, feature_dim
        else:
            feature = feature_list
            feature_mask = scene_mask != -1
            encoder_input_mask = torch.cat([feature_mask, encoder_input_mask], dim=1) # bz, seq_len, feature_dim
    else:
        scene = None
        feature = None
    # return encoder_input_ids, encoder_token_type_ids, encoder_input_mask, decoder_input_ids, decoder_token_type_ids, decoder_input_mask, lm_labels, i3d, type_labels, \
    #     vid_list, scene_lists, session_lists, encoder_his_lst, encoder_his_type_lst, encoder_query_lst, encoder_query_type_lst
    return encoder_input_ids, encoder_token_type_ids, encoder_input_mask, decoder_input_ids, decoder_token_type_ids, decoder_input_mask, lm_labels, feature, type_labels, \
        vid_list, scene, session, seg_label

def build_input_from_segments(history, reply, tokenizer, session_label=None, session=None, with_eos=True, video=False, train=True, model='bart'):
    """
    caption: [[caption], [summary]] history: [[q], [a], ..., [q]], reply: [a]  other: default if train dataset
    """
    """ Build a sequence of input from 3 segments: caption(caption+summary) history and last reply """
    bos, eos, text, sep = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-2])
        
    instance = {}
    encoder_instance = {}
    decoder_instance = {}
    
    # 1.sequence
    sequence = history + [reply + ([eos] if with_eos else [])]
    # [[d1], [d2], ..., [q]] + [[a, eos]] -> [[d1], [d2], ..., [q], [a, eos]]
    sequence = [[bos]] + [[sep] + d for d in sequence]
    # [[bos], [sep, d1], ..., [sep, a, eos]]
    instance["input_ids"] = list(chain(*sequence))
    # [bos, sep, d1, sep, d2, ..., sep, q, sep, a, eos]
    instance["history_type_ids"] = [text for i, s in enumerate(history) for _ in s]
    instance["token_type_ids"] = [text for i, s in enumerate(sequence) for _ in s]
    # [text, ..., text, ...]
    instance["lm_labels"] = ([-1]*sum(len(s) for s in sequence[:-1])) + sequence[-1]
    # [-1, ..., speaker2, a, eos]
    # instance['type_labels'] = ([-1]*sum(len(s) for s in sequence[:-1])) + decoder_instance["token_type_ids"]
    
    # 2. encoder
    encoder_sequence = history
    # [[d1], [d2], ..., [q]]
    encoder_sequence = [[bos]] + [[sep] + d for d in encoder_sequence]
    # [[bos], [sep, d1], ..., [sep, q]]
    encoder_instance["input_ids"] = list(chain(*encoder_sequence))
    # [bos, sep, d1, sep, d2, ..., sep, q]
    encoder_instance["token_type_ids"] = [text for i, s in enumerate(encoder_sequence) for _ in s]
    # 2.1 encoder session
    if session is not None:
        assert len(history) == len(session)
        sess = [[0]] + [[0] + s for s in session]
        sess = list(chain(*sess))
        sess_label = [[-1]] + [[session_label[k]] + [-1]*len(session[k]) for k in range(len(session))]
        sess_label = list(chain(*sess_label))
        sess_label[1] = -1
        assert len(encoder_instance['input_ids']) == len(sess)
        assert len(sess) == len(sess_label)
    else:
        sess = []
        sess_label = []
    encoder_instance['session'] = sess
    encoder_instance['session_label'] = sess_label
    
    # 3. decoder
    decoder_sequence = reply + ([eos] if with_eos else [])
    # [a, eos]
    decoder_sequence = [[sep] + decoder_sequence]
    # [[sep, a, eos]]
    decoder_instance["input_ids"] = list(chain(*decoder_sequence))
    # [sep, a, eos]
    decoder_instance["token_type_ids"] = [text for s in decoder_sequence for _ in s]
    # [text, ...]
    decoder_instance["lm_labels"] = sequence[-1]
    decoder_instance["type_labels"] = decoder_instance["token_type_ids"]
    
    # # his query 
    # encoder_instance["his_ids"] = list(chain(*encoder_sequence[:-1]))
    # encoder_instance["his_type_ids"] = [text for i, s in enumerate(encoder_sequence[:-1]) for _ in s]
    # encoder_instance["query_ids"] = list(chain(*[encoder_sequence[-1]]))
    # encoder_instance["query_type_ids"] = [text] * len(encoder_sequence[-1])
    # encoder_instance['session'] = sess
    # assert len(encoder_instance['query_ids']) == len(encoder_instance['query_type_ids'])    

    return encoder_instance, decoder_instance, instance, sequence


