import os, re, json, sys, time, copy, datetime
sys.path.append('utils/coco-caption')

from argparse import ArgumentParser
import subprocess
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

class StopwordFilter(object):

    def __init__(self, filename):
        self.pats = []
        if os.path.exists(filename):
            for ln in open(filename, 'r').readlines():
                ww = ln.split()
                if len(ww)==1:
                    self.pats.append((re.compile(r'^' + ww[0] + r'$'), ''))
                elif len(ww)==2:
                    self.pats.append((re.compile(r'^' + ww[0] + r'$'), ww[1]))

    def _filter(self, input_words):
        output_words = []
        for w in input_words:
            target = w
            for p in self.pats:
                v = p[0].sub(p[1],w)
                if v != w:
                    target = v
                    break
            if target != '':
                output_words.append(target)
        return output_words

    def __call__(self, input_words):
        if isinstance(input_words, str):
            return ' '.join(self._filter(input_words.split()))
        elif isinstance(input_words, list):
            return self._filter(input_words)
        else:
            return None

def evaluate(ref, exp_id, ref_path='inputs/full/ref.json'):
    swfilter = StopwordFilter('utils/stopwords.txt')
    annos = []
    img_id = 1
    for hyp in ref['dialogs']:
        # annos.append({'image_id':img_id, 'caption':swfilter(hyp['dialog'][-1])})
        annos.append({'image_id':img_id, 'caption':hyp['dialog'][-1]})
        img_id += 1
    hyp_fn = exp_id.replace('.json', '_tmp.json')
    with open(hyp_fn, 'w') as jh:
        json.dump(annos, jh, indent=4)
    coco = COCO(ref_path)
    cocoRes = coco.loadRes(hyp_fn)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    log_file = exp_id.split('/')[0] + '/' + exp_id.split('/')[1] + '/' + 'res.log' 
    with open(log_file, 'a') as fh:
        fh.write(exp_id + '\n')
    for metric, score in cocoEval.eval.items():
        print('%s: %.3f'%(metric, score))
        with open(log_file, 'a') as fh:
            fh.write('%s: %.3f'%(metric, score))
            fh.write('\n')
    os.remove(hyp_fn)

def eval(ref, exp_id, ref_path='inputs/full/ref.json'):
    swfilter = StopwordFilter('utils/stopwords.txt')
    annos = []
    img_id = 1
    for hyp in ref['dialogs']:
        annos.append({'image_id':img_id, 'caption':swfilter(hyp['dialog'][-1]['answer'])})
        img_id += 1
    hyp_fn = exp_id.replace('.json', '_tmp.json')
    with open(hyp_fn, 'w') as jh:
        json.dump(annos, jh, indent=4)
    coco = COCO(ref_path)
    cocoRes = coco.loadRes(hyp_fn)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    log_file = exp_id.split('/')[0] + '/' + exp_id.split('/')[1] + '/' + 'res.log' 
    with open(log_file, 'a') as fh:
        fh.write(exp_id + '\n')
    for metric, score in cocoEval.eval.items():
        print('%s: %.3f'%(metric, score))
        with open(log_file, 'a') as fh:
            fh.write('%s: %.3f'%(metric, score))
            fh.write('\n')
    os.remove(hyp_fn)


def eval_batch(ckpt_lst, log_set):
    for ckpt in ckpt_lst:
        result_file = os.path.join('results/{}/'.format(log_set), 'result_{}_5_1_1.json'.format(ckpt))
        with open(result_file) as jh:
            ref_dct = json.load(jh)
        evaluate(ref_dct, result_file)

if __name__ == '__main__':
    with open('results/sample/baseline_i3d_rgb-i3d_flow.json') as jh:
        test = json.load(jh)
    eval(test, 'results/sample/baseline_i3d_rgb-i3d_flow.json', 'results/sample/test_set4DSTC7-AVSD_multiref.json')


