import argparse

import torch

import os,sys
sys.path = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))] + sys.path
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path 
sys.path = [os.path.dirname(os.path.abspath(__file__))] + sys.path

from vedacore.misc import Config, DictAction, ProgressBar, load_weights
from vedacore.parallel import MMDataParallel
from vedatad.datasets import build_dataloader, build_dataset
from vedatad.engines import build_engine

from Eval_Thumos14.eval_detection import ANETdetection
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in json format', default='./outputs.json')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function')

    args = parser.parse_args()
    return args


def prepare(cfg, checkpoint):
    engine = build_engine(cfg.val_engine)
    load_weights(engine.model, checkpoint, map_location='cpu')

    device = torch.cuda.current_device()
    engine = MMDataParallel(
        engine.to(device), device_ids=[torch.cuda.current_device()])

    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    dataloader = build_dataloader(dataset, 1, 1, dist=False, shuffle=False)

    return engine, dataloader


def test(engine, data_loader):
    engine.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))

    det_dict = dict()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = engine(data)
        det_dict.update(result)
        batch_size = len(data['video_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return det_dict


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    engine, data_loader = prepare(cfg, args.checkpoint)

    results = test(engine, data_loader)

    if args.out:
        print(f'\nwriting results to {args.out}')
        output_dict = {"version": "THUMOS14", "results": results, "external_data": {}}

        with open(args.out, "w") as out:
            json.dump(output_dict, out)
    tious = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    anet_detection = ANETdetection(
        ground_truth_filename="tools/Eval_Thumos14/thumos_gt.json",
        prediction_filename=args.out,
        subset='test', tiou_thresholds=tious)
    mAPs, average_mAP = anet_detection.evaluate()
    sum=0
    for (tiou, mAP) in zip(tious, mAPs):
        print("mAP at tIoU {} is {}".format(tiou, mAP))
        if tiou==0.3 or tiou==0.4 or tiou==0.5 or tiou==0.6 or tiou==0.7:
            sum+=mAP
    print("sum average:",sum*1.0/5)

if __name__ == '__main__':
    main()
