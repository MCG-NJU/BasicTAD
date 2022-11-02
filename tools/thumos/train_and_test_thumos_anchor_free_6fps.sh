bash tools/dist_trainval.sh configs/trainval/basictad/thumos14/basictad_slowonly_e700_thumos14_rgb_192win_anchor_free.py "0,1,2,3,4,5,6,7" 

for i in {1..12..1}
do 
    a="$i"00_weights
    echo $a
    CUDA_VISIBLE_DEVICES=4 python tools/thumos/test_af.py --framerate 6 configs/trainval/basictad/thumos14/basictad_slowonly_e700_thumos14_rgb_192win_anchor_free.py workdir/basictad_slowonly_e700_thumos14_rgb_192win_anchor_free/epoch_$a.pth
done
