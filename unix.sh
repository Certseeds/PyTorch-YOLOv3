#!/bin/bash
set -e
set -ux
set -o pipefail
# python3 detect.py --source data/images/Original --weights ./runs/train/exp12/weights/best.pt --conf 0.25
function detect() {
  python3 ./detect.py \
    --imglist_file ./data/barcode_All/train.txt \
    --weights_path ./runs/train/exp13/weights/last.weights \
    --class_path ./data/barcode_All/classes.names \
    --model_def ./config/yolov3-barcode.cfg
  # --image_folder ../barcode_detection_dataset/pictures/images/20210317_2 \
}
function detectCorrect() {
  python3 ./detect.py \
    --imglist_file ./data/CorrectDetect.txt \
    --weights_path ./runs/train/exp14/weights/last.weights \
    --class_path ./data/barcode_All/classes.names \
    --model_def ./config/yolov3-barcode.cfg
  # --image_folder ..\barcode_detection_dataset\pictures\images\20210317_2 \
}
function train() {
  python3 ./train.py \
    --n_cpu 12 \
    --epochs 300 \
    --batch_size 8 \
    --checkpoint_interval 25 \
    --evaluation_interval 25 \
    --data_config config/barcode.data \
    --model_def config/yolov3-barcode.cfg \
    --verbose
  # --pretrained_weights checkpoints/yolov3_ckpt_87.pth

}
function train_all() {
  # one batch for about 450MB GraphCardMemory
  # 8 epoch is almost same with 10, accu than 12
  python3 ./train.py \
    --n_cpu 12 \
    --epochs 300 \
    --batch_size 8 \
    --checkpoint_interval 25 \
    --evaluation_interval 25 \
    --data_config ./config/barcode_All.data \
    --model_def ./config/yolov3-barcode.cfg \
    --verbose
  # --pretrained_weights checkpoints/yolov3_ckpt_87.pth
}
echo "$@"
for var in "$@" ; do
  echo "${var}"
  eval "${var}"
  exit
done
# example:
# run `./unix.sh detectCorrect`
