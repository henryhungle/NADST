
    
python train.py  -path=save/nadst \
    -d=256 -h_attn=16 \
    -bsz=32 -wu=20000 -dr=0.2 \
    -dv='2.1' \
    -d2s_nn_N=3 -s2s_nn_N=3
    
python test.py  -path=save_temp/nadst \
    -d=256 -h_attn=16 \
    -bsz=32 -wu=20000 -dr=0.2 \
    -dv='2.1' \
    -d2s_nn_N=3 -s2s_nn_N=3
