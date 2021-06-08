python experiment/sure_exp.py den \
    --datadir data/natural \
    --savedir result/den \
    --modeldir models/b-20-40.pth \
    --std 25 \
    --savemode plot \
    --windows 24 32 40 48 \
    --stride 1 \
    --numnoises 1 2 \
    --errorlog