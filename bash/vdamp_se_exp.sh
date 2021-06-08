python experiment/vdamp_se_exp.py \
    --datadir data/mri \
    --savedir result/vdamp-se \
    --modeldir models \
    --dentype cdncnn \
    --savemode plot \
    --windows 24 32 40 48 \
    --stride 1 \
    --numnoises 1 2 \
    --errorlog \
    --verbose