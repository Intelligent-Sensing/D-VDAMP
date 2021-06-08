python train/main.py train \
    --datadir PATH_TO_YOUR_DATA \
    --logdir train/cdncnn_20_50 \
    --modeldir train/cdncnn_20_50/cdncnn_20_50.pth \
    --logimage 0 \
    --stdrange 20 50 \
    --wavetype haar \
    --level 4 \
    --batchsize 50 \
    --learnrate 1e-4 \
    --epoch 1
    
