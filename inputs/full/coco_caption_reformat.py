import json, jsonlines

"""
ref: {'info':{}, "licenses":[], "images":[{"name":"", "id":}, ...], "type":"caption", "annotations":[{"image_id":, "id":, "caption":""}, ...]}
hyp: {"dialogs":[{"image_id":"", "dialog":{"answer":"", "question":""}}, {}, ...]}
"""

def get_ref():
    with open('test.json') as jh:
        test_dct = json.load(jh)
    image_lst = []
    anno_lst = []
    id_cnt = 1
    for clip_lst in test_dct.values():
        for clip in clip_lst:
            img_tmp_dct = {}
            img_tmp_dct['name'] = clip['clip_id']
            img_tmp_dct['id'] = id_cnt
            image_lst.append(img_tmp_dct)
            anno_tmp_dct = {}
            anno_tmp_dct['image_id'] = id_cnt
            anno_tmp_dct['id'] = id_cnt
            anno_tmp_dct['caption'] = clip['dialog'][-1]
            anno_lst.append(anno_tmp_dct)
            id_cnt += 1

    ref_dct = {'info':{}, "licenses":[], "images":image_lst, "type":"caption", "annotations":anno_lst}
    with open('ref.json', 'w') as jh:
        json.dump(ref_dct, jh)
    
    
def reformat_test4ref(type_id):
    with open('{}.json'.format(type_id)) as jh:
        test_data = json.load(jh)
    res_dct = {}
    for clip in test_data['dialogs']:
        res_dct[clip['clip_id']] = clip['dialog']
    with open('{}_ref.json'.format(type_id), 'w') as jh:
        json.dump(res_dct, jh)
    
if __name__ == '__main__':    
    get_ref()
    reformat_test4ref('test')
    # reformat_test4ref('valid')


