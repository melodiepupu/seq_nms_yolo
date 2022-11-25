#!/bin/bash

if test -f "yolo.weights"; then
  echo "yolo.weights already downloaded"
else
  wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights
fi

if [ -f "tiny-yolo.weights" ] || [ -f "yolov2-tiny-voc.weights" ]; then
  echo "tiny-yolo.weights already downloaded"
else
  wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights
  mv yolov2-tiny-voc.weights ./tiny-yolo.weights
fi

# Extract frames
cd video || exit
python video2img.py -i "$1"
#python video2img.py -i v_ApplyEyeMakeup_g19_c03.avi
python get_pkllist.py

# Generate detection frames in video/output
cd ..
python yolo_seqnms.py --seq_nms 0

# Reconstruct video from detection
cd video || exit
python img2video.py -i output