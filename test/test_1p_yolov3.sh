#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="YoloV3_ID1790_for_PyTorch"

#训练batch_size,,需要模型审视修改
batch_size=512


#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"


# 指定测试所使用的npu device卡id
device_id=0

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi



###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi


#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/yolov3/eval/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/yolov3/eval/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/yolov3/eval/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/yolov3/eval/$ASCEND_DEVICE_ID
fi


#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

sed -i "s|data/coco/|$data_path/|g" configs/yolo/yolov3_d53_mstrain-608_273e_coco.py

#python3 ./tools/test.py ./configs/yolo/yolov3_d53_320_273e_coco.py ./work_dirs/yolov3_d53_320_273e_coco/latest.pth --eval bbox

#python3 ./tools/test.py --images-folder ${data_path}/ch4_test_images --output-folder ./res/ --checkpoint ./runs/best_checkpoint.pt && zip -jmq ./runs/u.zip ./res/* && python3 ./script.py -g=${data_path}/gt.zip -s=./runs/u.zip

python3 ./tools/test_npu.py ./configs/yolo/yolov3_d53_320_273e_coco.py \
  ./work_dirs/yolov3_d53_320_273e_coco/latest.pth \
  --cfg-options optimizer.lr=0.001 data.test.samples_per_gpu=${batch_size} \
  --local_rank 0 \
  --out ${test_path_dir}/output/yolov3/eval/$ASCEND_DEVICE_ID/test_out.pkl   \
  --eval bbox  > ${test_path_dir}/output/yolov3/eval/${ASCEND_DEVICE_ID}/test_${ASCEND_DEVICE_ID}.log 2>&1 &
wait
##################获取训练数据################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
Accuracy=`grep -a 'Accuracy'  ${test_path_dir}/output/yolov3/eval/${ASCEND_DEVICE_ID}/test_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $NF}'`
# 打印，不需要修改
echo "Final Train Accuracy : $Accuracy"
echo "E2E Training Duration sec : $e2e_time"