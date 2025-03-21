# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
PWD=$(cd `dirname $0` && pwd)
cd $PWD/../
EXP_DIR=/data/users/wangying01/lth/hsmot/workdir/motip_99/motip_r50_train_hsmot_8ch_4gpu

mkdir -p ${EXP_DIR}
touch ${EXP_DIR}/output.log
cp $0 ${EXP_DIR}/


# LD_LIBRARY_PATH="" CUDA_VISIBLE_DEVICES=0,1, python3 -m torch.distributed.run --nproc_per_node=2 \
#      main.py \
#     --mode train \
#     --use-distributed True \
#     --use-wandb False\
#     --config-path /data/users/wangying01/lth/hsmot/MOTIP/configs/r50_deformable_detr_motip_hsmot8ch_99.yaml \
#     --outputs-dir ${EXP_DIR} \
#     | tee ${EXP_DIR}/output.log

LD_LIBRARY_PATH="" CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=1 \
     --master_port 20011 \
     --use_env main.py \
     --mode train \
     --use-distributed True \
     --use-wandb False\
     --config-path /data/users/wangying01/lth/hsmot/MOTIP/configs/r50_deformable_detr_motip_hsmot8ch_99.yaml \
     --outputs-dir ${EXP_DIR} \
     | tee ${EXP_DIR}/output.log