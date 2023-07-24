# VSTAR

This is the official implementation of the ACL 2023 paper "VSTAR: A Video-grounded Dialogue Dataset for Situated Semantic Understanding with Scene and Topic Transitions"

## Dataset

### Schedule

- [X] release dialogues
- [X] release feature (resnet, rcnn)
- [ ] release meta data (genres, keywords, storyline, characters: name, avatar)
- [ ] release frames

### Dialogues

> support language: English, 简体中文

- Downloads

Storage: Train (196M); Valid(11.6M); test (*Note: We will reserve the test split for further study*)

Links: [BaiduNetDisk](https://pan.baidu.com/s/1sV1rLadxNnhQbAsr_r75Ow?pwd=b2m9) or [GoogleDrive](https://drive.google.com/drive/folders/16vB8bqDtkrYPGKBV_Og3Fk77yDrUi187?usp=sharing)

- Statistics

|       | clips   | dialogues | scene/clip | topic/clip |
| ----- | ------- | --------- | ---------- | ---------- |
| Train | 172,041 | 4,319,381 | 2.42       | 3.68       |
| Val   | 9753    | 250,311   | 2.64       | 4.29       |

- Format

```
{
	"dialogs":[
		{
			"clip_id": "Friends_S01E01_clip_000",
			"dialog": ["hi", ...],
			"scene": [1, 1, 1, 1, 1, 1, 2, 2, ...],
			"session": [1, 1, 1, 2, 2, 2, 3, 3, ...]
		},
		...
]
}
```

### Feature

- Downloads

Storage: RCNN(246.2G), RESNET(109G)

Links: [BaiduNetDisk](https://pan.baidu.com/s/1eBWreYWDDFQd1cj19QxdRg?pwd=xszm)

- Format

File Structure:

```
# [name of TV show]_S[season]_E[episode]_clip_[clip id].npy
├── Friends_S01E01
   └── Friends_S01E01_clip_000.npy
   └── Friends_S01E01_clip_001.npy
   └── ...
├── ...
```

ResNet:

```
# numpy.load("Friends_S01E01_clip_000.npy")
(num_of_frames * 1000)
```

RCNN:

```
# numpy.load("Friends_S01E01_clip_000.npy", allow_pickle=True).item()
{
	"feature": (9 * num_of_frames * 2048) # array(float32), feature top 9 objects
	"size": (num_of_frames * 2) # list(int), size of original frame
	"box": (9 * num_of_frames * 4) # array(float32), bbox
	"obj_id": (9 * num_of_frames) # list(int), object id
	"obj_conf": (9 * num_of_frames) # array(float32), object conference 
	"obj_num": (num_of_frames) # list(int), number of objects/frame
}
```

- Feature Extraction Tools

Please Refer to [OpenViDial_extract_features](https://github.com/ShannonAI/OpenViDial/blob/main/video_dialogue_model/extract_features/extract_features.md)

## Installation

```
pip install -r requirements.txt
```

## Scene Segmentation

- Train

```
python train_seg.py \
	--video 1 \
	--exp_set EXP_LOG \
	--train_batch_size 4 \
```

- Infer

```
python generate_seg.py \
	--ckptid SAVED_CKPT_ID \
	--gpuid 0 \
	--exp_set EXP_LOG \
	--video 1 \
```

## Topic Segmentation

- Train

```
python train_seg.py \
	--video 0 \
	--exp_set EXP_LOG \
	--train_batch_size 4 \
```

- Infer

```
python generate_seg.py \
	--ckptid SAVED_CKPT_ID \
	--gpuid 0 \
	--exp_set EXP_LOG \
	--video 0 \
```

## Dialogue Generation

- Train

```
python train_gen.py \
	--train_batch_size 4 \
	--model bart \
	--exp_set EXP_LOG \
	--video 1 \
	--fea_type resnet \

```

- Infer

```
python generate.py \
	--ckptid SAVED_CKPT_ID \
	--gpuid 0 \
	--exp_set EXP_LOG \
	--video 1 \
	--sess 1 \
	--batch_size 4
```

## Citation

```
@misc{wang2023vstar,
    title={VSTAR: A Video-grounded Dialogue Dataset for Situated Semantic Understanding with Scene and Topic Transitions},
    author={Yuxuan Wang and Zilong Zheng and Xueliang Zhao and Jinpeng Li and Yueqian Wang and Dongyan Zhao},
    year={2023},
    eprint={2305.18756},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
