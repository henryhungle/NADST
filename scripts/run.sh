
    
python train.py  -path=save/nadst \
    -d=256 -h_attn=16 \
    -bsz=32 -wu=20000 -dr=0.2 \
    -dv='2.1' \
    -fert_dec_N=3 -state_dec_N=3
    
python test.py  -path=save/nadst \
    -d=256 -h_attn=16 \
    -bsz=32 -wu=20000 -dr=0.2 \
    -dv='2.1' \
    -fert_dec_N=3 -state_dec_N=3
