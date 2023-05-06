# VSTAR

This is the official implementation of the paper "VSTAR: A Video-grounded Dialogue Dataset for Situated Semantic Understanding with Scene and Topic Transitions"

## Dataset

- [ ] release feature (resnet, rcnn)
- [ ] release text
- [ ] release frames

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