# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
PWD=$(cd `dirname $0` && pwd)
cd $PWD/../
EXP_DIR=/data/users/litianhao/hsmot_code/workdir/motip/motip_r50_train_hsmot_8ch_4gpu

mkdir -p ${EXP_DIR}
touch ${EXP_DIR}/output.log
cp $0 ${EXP_DIR}/


CUDA_VISIBLE_DEVICES=6,7 python3 -m torch.distributed.run --nproc_per_node=2 \
     main.py \
    --mode train \
    --use-distributed True \
    --use-wandb False\
    --config-path /data/users/litianhao/hsmot_code/MOTIP/configs/r50_deformable_detr_motip_hsmot8ch.yaml \
    --outputs-dir ${EXP_DIR} \
    | tee ${EXP_DIR}/output.log
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 \
#     --use_env main.py \
#     --mode train \
#     --use_distributed True \
#     --use-wandb False\
#     --config-path /data/users/litianhao/hsmot_code/MOTIP/configs/r50_deformable_detr_motip_hsmot8ch.yaml \
#     --outputs-dir ${EXP_DIR} \
#     | tee ${EXP_DIR}/output.log