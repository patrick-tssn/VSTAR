import json

def seg_clip_data(identifier):
    with open('../{}.json'.format(identifier)) as jh:
        dataset = json.load(jh)
    seg_clip = {}
    for clip in dataset['dialogs']:
        session_tmp = []
        scene_tmp = []
        session_prev = 1
        scene_prev = 1
        for c in clip['session']:
            if c != session_prev:
                session_tmp.append(1)
                session_prev = c
            else:
                session_tmp.append(0)
        assert len(session_tmp) == len(clip['session'])
        for c in clip['scene']:
            if c != scene_prev:
                scene_tmp.append(1)
                scene_prev = c
            else:
                scene_tmp.append(0)
        assert len(scene_tmp) == len(clip['scene'])
        seg_clip[clip['clip_id']] = {'dialog':clip['dialog'], 'session':session_tmp, 'scene':scene_tmp}
    with open('seg_clip_{}.json'.format(identifier), 'w') as jh:
        json.dump(seg_clip, jh)

def reformat(split):
    with open("seg_clip_{}.json".format(split)) as jh:
        ori_dataset = json.load(jh)
    clipid_lst = list(ori_dataset.keys())
    new_dct = {}
    for i in range(len(clipid_lst)):
        clip_id = clipid_lst[i]
        clip_dct = ori_dataset[clip_id]
        new_dct[clip_id] = {"dialog":clip_dct["dialog"],"session":[],"scene":[]}
        new_session = clip_dct['session'][1:]
        new_scene = clip_dct["scene"][1:]
        if i < len(clipid_lst)-1:
            new_session.append(ori_dataset[clipid_lst[i+1]]['session'][0])
            new_scene.append(ori_dataset[clipid_lst[i+1]]['scene'][0])
        else:
            new_session.append(1)
            new_scene.append(1)
        ori_session = clip_dct['session']
        new_dct[clip_id]['session'] = new_session
        new_dct[clip_id]['scene'] = new_scene
    with open("pro_seg_clip_{}_ori.json".format(split), 'w') as jh:
        json.dump(new_dct, jh)

def revise():
    with open("pro_seg_clip_train_ori.json") as jh:
        ori_dataset = json.load(jh)
    new_dataset = {}
    remove_lst = ['Reverie_S01E04', 'Light.as.a.Feather_S01E08', 'Minority.Report_S01E08']
    for clipid, clipdct in ori_dataset.items():
        if remove_lst[0] in clipid or remove_lst[1] in clipid or remove_lst[2] in clipid:
            continue
        new_dataset[clipid] = clipdct
    with open("pro_seg_clip_train.json", 'w') as jh:
        json.dump(new_dataset, jh)


if __name__ == '__main__':
    for split in ['train', 'valid', 'test']:
        seg_clip_data(split)
    for split in ['train', 'valid', 'test']:
        reformat(split)
    revise()
