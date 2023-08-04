
_base_ = './ssd300_coco.py'
optimizer = dict(type='SGD', lr=2e-3/8, momentum=0.9, weight_decay=5e-4)
dist_params = dict(backend='hccl')
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8
)