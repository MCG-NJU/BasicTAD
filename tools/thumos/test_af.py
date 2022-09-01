import argparse

import torch
import os
import sys

sys.path = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))] + sys.path
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path 
sys.path = [os.path.dirname(os.path.abspath(__file__))] + sys.path

from vedacore.fileio import dump
from vedacore.misc import Config, DictAction, ProgressBar, load_weights
from vedacore.parallel import MMDataParallel
from vedatad.datasets import build_dataloader, build_dataset
from vedatad.engines import build_engine
import pickle

import numpy as np

import sys

def nms(dets, thresh=0.4):
    """Pure Python NMS baseline."""
    if len(dets) == 0:
        return []
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = dets[:, 2]
    lengths = x2 - x1
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def nms_id(dets,id,thresh=0.4):
    """Pure Python NMS baseline."""
    id=id[0]
    if len(dets) == 0:
        return []
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = dets[:, 2]
    lengths = x2 - x1
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        real_id=id[i]
        keep.append(real_id)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def generate_classes(meta_dir, split, use_ambiguous=False):
    class_id = {0: 'Background'}
    with open(os.path.join(meta_dir, 'detclasslist.txt'), 'r') as f:
        lines = f.readlines()
        for _line in lines:
            cname = _line.strip().split()[-1]
            cid = int(_line.strip().split()[0])
            class_id[cid] = cname
        if use_ambiguous:
            class_id[21] = 'Ambiguous'

    return class_id

def get_segments(data, thresh, framerate):
    # import pdb; pdb.set_trace()
    res = {}
    vid_names = data.keys()
    for _vid in vid_names:
        dets_list = data[_vid]
        segments = []
        for dets in dets_list:
            # dets[1] ~ dets[20]
            for c in dets.keys():
                num_dets = dets[c].shape[0]
                for i in range(num_dets):
                    # import pdb; pdb.set_trace()
                    single_det = dets[c][i]
                    tmp = {}
                    tmp['label'] = c
                    tmp['score'] = single_det[2]
                    tmp['segment'] = [
                        single_det[0] / float(framerate),
                        single_det[1] / float(framerate)
                    ]
                    if tmp['score'] > thresh:
                        segments.append(tmp)
        res[_vid] = segments
    return res

def select_top(segmentations, nms_thresh=0.99999, num_cls=0, topk=0):
    # import pdb; pdb.set_trace()
    res = {}
    for vid, vinfo in segmentations.items():
        # select most likely classes
        if num_cls > 0:
            ave_scores = np.zeros(21)
            for i in range(1, 21):
                ave_scores[i] = np.sum([d['score'] for d in vinfo if d['label'] == i])
            labels = list(ave_scores.argsort()[::-1][:num_cls])
        else:
            labels = list(set([d['label'] for d in vinfo]))

        # NMS
        res_nms = []
        for lab in labels:
            nms_in = [d['segment'] + [d['score']] for d in vinfo if d['label'] == lab]
            keep = nms(np.array(nms_in), nms_thresh)
            for i in keep:
                tmp = {'label': lab, 'score': nms_in[i][2], 'segment': nms_in[i][0:2]}
                res_nms.append(tmp)

        # select topk
        scores = [d['score'] for d in res_nms]
        sortid = np.argsort(scores)[-topk:]
        res[vid] = [res_nms[id] for id in sortid]
    return res

def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--framerate', default=6,type=int,help='framerate')
    parser.add_argument('--thresh', default=0.005,type=float,help='thresh')
    parser.add_argument('--nms_thresh', default=0.6,type=float,help='nms_thresh')
    parser.add_argument('--topk', default=200,type=int,help='topk')
    parser.add_argument('--num_cls', default=0,type=int,help='num_cls')
    parser.add_argument('--cls_num', default=20,type=int,help='cls_num')
    parser.add_argument('--video_num', default=213,type=int,help='video_num')
    parser.add_argument('--proposals_per_video', default=300,type=int,help='proposals_per_video')
    parser.add_argument('--out', help='output result file in pickle format')
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


def test(engine, data_loader,args):
    engine.eval()
    results = {}
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    sum=0
    for i, data in enumerate(data_loader):

        with torch.no_grad():
            result= engine(data)[0]
            num=engine(data)[1]

        step=len(data['video_metas'][0].data)
        batch_size = len(result)
        sum=sum+num

        for j in range(step):
          for i in range(batch_size):
            video_name = data['video_metas'][0].data[0][0]['video_name'].split('/')[-1]
            single_ret = result[i]['results'][0]
            if video_name not in results:
                results[video_name] = [single_ret]
            else:
                results[video_name].append(single_ret)

        for _ in range(step): 
            prog_bar.update()

    results_dict_path=args.checkpoint.split('/')[-1].split('.')[0]+'merge.txt'
    framerate=args.framerate
    thresh=args.thresh
    nms_thresh=args.nms_thresh
    topk=args.topk
    num_cls=args.num_cls

    segmentations = get_segments(results, thresh=thresh, framerate=framerate)
    segmentations = select_top(segmentations, nms_thresh=nms_thresh, num_cls=num_cls, topk=topk)
    
    res = {
    'version': 'VERSION 1.3',
    'external_data': {'used': True,
                      'details': 'C3D pre-trained on activity-1.3 training set'},
    'results': {}
    }
    for vid, vinfo in segmentations.items():
        res['results'][vid] = vinfo
    
    with open(results_dict_path, 'w') as outfile:
        for vid, vinfo in segmentations.items():
            for seg in vinfo:
                outfile.write(
                "{} {} {} {} {}\n".format(
                    vid, seg['segment'][0], seg['segment'][1], int(seg['label']), seg['score']
                    )
                )
    results_dict_path_5and8=results_dict_path.split('.')[0]+'_5and8.txt'
    with open(results_dict_path_5and8,'w') as wf:
      with open(results_dict_path, "r") as f:
        data_alls=f.readlines()
        for data_all in data_alls:
            data=data_all.split(' ')
            video_name=data[0]
            video_start=data[1]
            video_end=data[2]
            video_label=data[3]
            video_score=data[4]
            wf.write(video_name)
            wf.write(' ')
            wf.write(video_start)
            wf.write(' ')
            wf.write(video_end)
            wf.write(' ')
            wf.write(video_label)
            wf.write(' ')
            wf.write(video_score)
            if video_label=='5':
                wf.write(video_name)
                wf.write(' ')
                wf.write(video_start)
                wf.write(' ')
                wf.write(video_end)
                wf.write(' ')
                wf.write('8')
                wf.write(' ')
                wf.write(video_score)
    subset = 'test'
    detfilename=results_dict_path_5and8
    cls_num = args.cls_num
    test_thresholds=[0.3]
    nms_thresholds =  [0.3]
    video_num=args.video_num
    proposals_per_video = args.proposals_per_video
    videonames=[]
    t1=[]
    t2=[]
    clsid=[]
    conf=[]
    for index in range(len(test_thresholds)):
        threshold=test_thresholds[index]
        with open(detfilename, "r") as f:
            data_alls=f.readlines()
            for data_all in data_alls:
                data_all=data_all.strip('\n')
                data=data_all.split(' ')
                video_name=data[0]
                video_start=data[1]
                video_end=data[2]
                video_label=data[3]
                video_score=data[4]
                videonames.append(video_name)
                t1.append(float(video_start))
                t2.append(float(video_end))
                clsid.append(int(video_label))
                conf.append(float(video_score))
        conf=np.array(conf)
        confid=conf>args.thresh
        videonames=np.array(videonames)
        t1=np.array(t1)
        t2=np.array(t2)
        clsid=np.array(clsid)
        videonames=videonames[confid]
        t1=t1[confid]
        t2=t2[confid]
        clsid = clsid[confid]
        clsid=clsid.tolist()
        conf = conf[confid]
        overlap_nms=nms_thresholds[index]
        videonames=videonames.tolist()
        videoid=list(set(videonames))
        pick_nms = []
        for id in range(len(videoid)):
            vid=videoid[id]
            for cls in range(1,args.cls_num+1):
                inputpick=[videonames[i]==vid and clsid[i]==cls for i in range(len(videonames))]
                inputpick=np.array(inputpick)
                inputpick=np.nonzero(inputpick == True)
                if len(inputpick[0]) ==0:
                    continue
                t1_temp=t1[inputpick]
                t1_temp=t1_temp[:,np.newaxis]
                t2_temp=t2[inputpick]
                t2_temp=t2_temp[:,np.newaxis]
                conf_temp=conf[inputpick]
                conf_temp=conf_temp[:,np.newaxis]
                boxes=np.concatenate((t1_temp,t2_temp,conf_temp),axis=1)
                boxes_id=nms_id(boxes,inputpick,threshold)
                for j in range(len(boxes_id)):
                    pick_nms.append(boxes_id[j])
        pick_nms=pick_nms[0:min(len(pick_nms),args.video_num*args.proposals_per_video)]
        videonames=np.array(videonames)
        clsid=np.array(clsid)
        videonames=videonames[pick_nms]
        t1=t1[pick_nms]
        t2=t2[pick_nms]
        clsid = clsid[pick_nms]
        conf = conf[pick_nms]
        with open('tmp_run.txt', "w") as wf:
            for number in range(len(clsid)):
                wf.write(videonames[number])
                wf.write(' ')
                wf.write(str(t1[number]))
                wf.write(' ')
                wf.write(str(t2[number]))
                wf.write(' ')
                wf.write(str(clsid[number]))
                wf.write(' ')
                wf.write(str(conf[number]))
                wf.write('\n')
    CLASSES = ('BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
           'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
           'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
           'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
           'SoccerPenalty', 'TennisSwing', 'ThrowDiscus',
           'VolleyballSpiking')

    f = open("./tmp_run.txt")
    import json
    from Eval_Thumos14.eval_detection import ANETdetection

    results = {}

    for line in f.readlines():
        line = line.strip().split(" ")
        video_name = line[0]
        if video_name not in results.keys():
            results[video_name] = list()
        start = float(line[1])
        end = float(line[2])
        label = CLASSES[int(line[3])-1]

        score = float(line[4])
        results[video_name].append({"segment": [start, end], "label": label, "score": score})

    output_dict = {"version": "THUMOS14", "results": results, "external_data": {}}
    final_file=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/output_'+args.checkpoint.split('/')[-1].split('.')[0]+".json"
    print("final_file:",final_file)
    with open(final_file, "w") as out:
        json.dump(output_dict, out)
    tious = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    sum=0
    anet_detection = ANETdetection(
        ground_truth_filename="./tools/Eval_Thumos14/thumos_gt.json",
        prediction_filename=final_file,
        subset='test', tiou_thresholds=tious)
    mAPs, average_mAP = anet_detection.evaluate()
    for (tiou, mAP) in zip(tious, mAPs):
        print("mAP at tIoU {} is {}".format(tiou, mAP))
        if tiou==0.3 or tiou==0.4 or tiou==0.5 or tiou==0.6 or tiou==0.7:
            sum+=mAP
    print("sum average:",sum*1.0/5)
def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    engine, data_loader = prepare(cfg, args.checkpoint)

    results = test(engine, data_loader,args)

    if args.out:
        print(f'\nwriting results to {args.out}')
        dump(results, args.out)


if __name__ == '__main__':
    main()
