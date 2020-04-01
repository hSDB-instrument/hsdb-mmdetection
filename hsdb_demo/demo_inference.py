import os
import sys
import argparse
from tqdm import tqdm
from time import gmtime, strftime

import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict

import cv2
import matplotlib
from bounding_box import bounding_box as bb

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result, init_detector
from mmdet.datasets import build_dataloader, build_dataset
from define_class_names import CHOLEC_CLASSES, GASTREC_CLASSES

matplotlib.use("Agg")


def inference(frame, model):

    result = inference_detector(model, frame)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_resutl = result, None

    bboxes = np.vstack(bbox_result)
    labels = [np.full(bbox.shape[0], i, dtype=np.int32)
              for i, bbox in enumerate(bbox_result)]
    labels = np.concatenate(labels)

    return bboxes, labels


def get_class_name(lbl, class_names):

    if lbl in list(range(len(class_names))):
        class_name = class_names[lbl]
    else:
        class_name = 'cls {}'.format(lbl)

    return class_name


def save_image(title, frame, frame_save_path):

    try:
        print("Save frame : {}".format(frame_save_path))
        cv2.imwrite(frame_save_path, frame)
    except OSError as err:
        print("Cannot save frame : {}".format(err))


def get_result(frame, labels, bboxes, class_names, dic_cls_color, frame_title,
               frame_save_path, score_thr):
    def _append_results(dic_result, frame_path, class_name, bbox):
        dic_result["frame_path"].append(frame_path)
        dic_result["class_name"].append(class_name)
        dic_result["score"].append(bbox[-1])
        dic_result["x1"].append(bbox[0])
        dic_result["y1"].append(bbox[1])
        dic_result["x2"].append(bbox[2])
        dic_result["y2"].append(bbox[3])

        return dic_result

    dic_result = {"frame_path":[],
                   "class_name":[],
                   "score":[],
                   "x1":[],
                   "y1":[],
                   "x2":[],
                   "y2":[]}

    over_score_cnt = 0
    if len(labels) != 0:
        for lbl, bbox in zip(labels, bboxes):
            class_name = get_class_name(lbl, class_names)
            score = bbox[-1]
            if score >= score_thr:
                bb.add(frame, bbox[0], bbox[1], bbox[2], bbox[3], class_name,
                       dic_cls_color[class_name])
                dic_result = _append_results(dic_result, frame_save_path,
                                             class_name, bbox)
                over_score_cnt +=1
            else:
                continue

    if over_score_cnt >= 1:
        save_image("", frame, frame_save_path)

    return dic_result


def inference_iteration(config_path, checkpoint_path,
                        frame_paths, frame_save_dir,
                        dic_cls_color, score_thr,
                        class_names):

    model = init_detector(config_path, checkpoint_path, device='cuda:0')
    model.CLASSES = class_names
    dic_results = {}
    for frame_path in tqdm(frame_paths):
        print("frame_path : {}".format(frame_path))
        frame_save_path = os.path.join(frame_save_dir,
                                       os.path.basename(frame_path))

        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        bboxes, labels = inference(frame, model)

        frame_title = os.path.basename(frame_path)
        dic_result = get_result(frame, labels, bboxes, class_names,
                                dic_cls_color, frame_title, frame_save_path,
                                score_thr)
        dic_results.update(dic_result)

    return dic_results


def parse_args():
    parser = argparse.ArgumentParser(description="hSDB-instrument \
                                                  Model Inference")
    parser.add_argument("config", help="model config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("frame_in_dir", help="directory for input frames")
    parser.add_argument("frame_out_dir", help="directory for output \
                                                 result frames")
    parser.add_argument('--cholec', action='store_true', help='model for cholec')
    parser.add_argument('--score_thr', default=0.5, help='score threshold')
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    if args.cholec:
        class_names = CHOLEC_CLASSES
    else:
        class_names = GASTREC_CLASSES

    bbox_colors = ["navy", "blue", "aqua", "teal", "olive", "green", "lime",
                  "yellow", "orange", "red", "maroon", "fuchsia", "purple",
                  "black", "gray", "silver"]

    dic_cls_color = {}
    for c_idx, cls_name in enumerate(class_names):
        dic_cls_color[cls_name] = bbox_colors[c_idx%len(bbox_colors)]

    frame_paths = [os.path.join(args.frame_in_dir, p)
                 for p in os.listdir(args.frame_in_dir)]

    
    now_dtime = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
    config_name = os.path.basename(args.config).split(".")[0]
    config_name = "".join([config_name, "_{}".format(now_dtime)])
    frame_out_dir = os.path.join(args.frame_out_dir, config_name)
    if not os.path.exists(frame_out_dir):
        os.makedirs(frame_out_dir)

    dic_results = inference_iteration(args.config, args.checkpoint,
                                      frame_paths, frame_out_dir,
                                      dic_cls_color, float(args.score_thr),
                                      class_names)

    csv_name = ".".join([config_name, "csv"])
    csv_out_path = os.path.join(frame_out_dir, csv_name)

    pd_results = pd.DataFrame(dic_results)
    pd_results.to_csv(csv_out_path, index=False)
    print("Save result csv : {}".format(csv_out_path))


if __name__ == "__main__":
    main()


