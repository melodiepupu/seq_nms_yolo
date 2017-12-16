# Seq_nms_YOLO

#### Membre: Yutong YAN, Sixiang XU, Heng ZHANG

---

## Introduction

![](img/index.jpg) 

This project combines **YOLOv2**([reference](https://arxiv.org/abs/1506.02640)) and **seq-nms**([reference](https://arxiv.org/abs/1602.08465)) to realise **real time video detection**.

## Steps
1. `make` the project
1. copy a video file to video folder
1. run `video.py`
1. run `get_pkllist.py`
1. run `yolo_seqnms.py`

And you will sse detection results in `video/output`

## Reference
This project copies lots of code from [darknet](https://github.com/pjreddie/darknet) , [Seq-NMS](https://github.com/lrghust/Seq-NMS) and [models](https://github.com/tensorflow/models).