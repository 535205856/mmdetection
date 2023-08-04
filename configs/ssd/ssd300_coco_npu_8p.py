
_base_ = './ssd300_coco.py'
dist_params = dict(backend='hccl')
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8
)