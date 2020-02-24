import os
import logging 
import argparse
from tqdm import tqdm
import pdb 

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0 
GATES = {"gen":0, "dontcare":1, "none":2}
REVERSE_GATES = {v: k for k, v in GATES.items()}

USE_CUDA = True
parser = argparse.ArgumentParser(description='Non-autoregressive DST')

# Data Setting
parser.add_argument('-dv', '--data_version', help='', required=False, type=str, default='2.1')
parser.add_argument('-dlhis', '--delex_his', help='', required=False, type=int, default=1) 
parser.add_argument('-sys_act', '--sys_act', help='', required=False, type=int, default=1)


# Training Setting
parser.add_argument('-ds','--dataset', help='dataset', required=False, default="multiwoz")
parser.add_argument('-path','--path', help='path of the file to load', required=False, default='temp')
parser.add_argument('-patience','--patience', help='', required=False, default=6, type=int)
parser.add_argument('-bsz','--batch', help='Batch_size', required=False, type=int, default=32)
parser.add_argument('-wu', '--warmup', help='', required=False, type=int, default=12880)
parser.add_argument('-smooth', '--label_smoothing', help='', required=False, type=int, default=1)

# Testing Setting
parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, default=1)
parser.add_argument('-reportp', '--reportp', help='report period during training', required=False, default=100)
parser.add_argument('-eb','--eval_batch', help='Evaluation Batch_size', required=False, type=int, default=16)
parser.add_argument('-ep', '--eval_epoch', help='', required=False, type=int, default=-1)
parser.add_argument('-tsplit', '--test_split', help='', required=False, type=str, default='test')
parser.add_argument('-em', '--eval_metric', help='', required=False, type=str, default='acc') #loss, acc, slot_acc
parser.add_argument('-ptest', '--p_test', help='', required=False, type=float, default=1)
parser.add_argument('-ptest_ft', '--p_test_fertility', help='', required=False, type=float, default=1)
parser.add_argument('-test_domain', '--test_domain', help='', required=False, type=str, default=None) 
parser.add_argument('-test_output', '--test_output', help='', required=False, type=str, default=None)

# Model architecture
parser.add_argument('-out2in_atrg', '--out2in_atrg', help='', required=False, default=0, type=int)
parser.add_argument('-atrg', '--auto_regressive', help='', required=False, type=int, default=0)
parser.add_argument('-sepiemb', '--sep_input_embedding', help='seperate domain/slot and context embedding', required=False, default=1, type=int)
parser.add_argument('-sepoemb', '--sep_output_embedding', help='separate output state and context embedding', required=False, default=0, type=int)
parser.add_argument('-gate', '--slot_gating', help='', required=False, default=0, type=int)
parser.add_argument('-lenval', '--slot_lenval', help='', required=False, default=1, type=int)
parser.add_argument('-ptr', '--pointer_decoder', help='', required=False, default=1, type=int)
parser.add_argument('-nope1', '--no_pe_ds_emb1', help='', required=False, default=0, type=int)
parser.add_argument('-nope2', '--no_pe_ds_emb2', help='', required=False, default=0, type=int)

# Model Hyper-Parameters
parser.add_argument('-fert_dec_N', '--fert_dec_N', help='', required=False, type=int, default=3)
parser.add_argument('-state_dec_N', '--state_dec_N', help='', required=False, type=int, default=3)
parser.add_argument('-d', '--d_model', help='', required=False, type=int, default=256)
parser.add_argument('-d_emb', '--d_embed', help='', required=False, type=int, default=-1)
parser.add_argument('-h_attn', '--h_attn', help='', required=False, type=int, default=16)
parser.add_argument('-dr','--drop', help='Drop Out', required=False, type=float, default=0.1)

args = vars(parser.parse_args())
for k,v in args.items():
    print('{}={}'.format(k,v))

