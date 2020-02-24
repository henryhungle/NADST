from tqdm import tqdm
import torch.nn as nn
import pdb 
from utils.config import *
from utils.utils_multiWOZ_DST import *
from model.nadst import * 
from model.evaluator import * 
from label_smoothing import * 
import os
import os.path 
import pickle as pkl 
import json 

def run_test(data, model):
    pbar = tqdm(enumerate(data),total=len(data), desc="evaluating", ncols=0)
    predictions = {}
    latencies = []
    src_lens = []
    tgt_lens = []
    oracle_predictions = {}
    joint_gate_matches = 0
    joint_lenval_matches = 0
    total_samples = 0
    if args['pointer_decoder']:
        predict_lang = saved['src_lang']
    else:
        predict_lang = saved['tgt_lang']
    for i, data in pbar:
        predictions, latencies, src_lens, tgt_lens = predict(None, data, model, 
            predict_lang, saved['domain_lang'], saved['slot_lang'], 
            predictions, False, saved['src_lang'], args, saved['SLOTS_LIST']['all'],
            latency=latencies, src_lens=src_lens, tgt_lens=tgt_lens)
        out = model.forward(data)
        matches, oracle_predictions = predict(out, data, model, 
            predict_lang, saved['domain_lang'], saved['slot_lang'], 
            oracle_predictions, True, saved['src_lang'], args)
        joint_lenval_matches += matches['joint_lenval']
        joint_gate_matches += matches['joint_gate']
        total_samples += len(data['turn_id'])
    
    avg_latencies = sum(latencies)/len(latencies)
    print("Average latency: {}".format(avg_latencies))
    with open(args['path'] + '/latency_eval.csv', 'w') as f:
        f.write(str(avg_latencies))
    pkl.dump(zip(latencies, src_lens, tgt_lens), open(args['path'] + '/latency_out.pkl', 'wb'))
    joint_acc_score, F1_score, turn_acc_score = -1, -1, -1
    oracle_joint_acc, oracle_f1, oracle_acc = -1, -1, -1
    joint_acc_score, F1_score, turn_acc_score = evaluator.evaluate_metrics(predictions, 'test')
    oracle_joint_acc, oracle_f1, oracle_acc = evaluator.evaluate_metrics(oracle_predictions, 'test')
    joint_lenval_acc = 1.0 * joint_lenval_matches/total_samples
    joint_gate_acc = 1.0 * joint_gate_matches/total_samples 
    with open(args['path'] + '/eval_{}_epoch{}_ptest{}-{}.csv'.format(args['test_split'], args['eval_epoch'], args['p_test'], args['p_test_fertility']), 'a') as f:
        f.write("{},{},{},{},{},{},{},{}".
                format(joint_gate_acc, joint_lenval_acc,
                       joint_acc_score,turn_acc_score, F1_score,
                      oracle_joint_acc,oracle_acc,oracle_f1))
    print("Joint Gate Acc {}".format(joint_gate_acc))
    print("Joint Lenval Acc {}".format(joint_lenval_acc))
    print("Joint Acc {} Slot Acc {} F1 {}".format(joint_acc_score,turn_acc_score, F1_score))
    print("Oracle Joint Acc {} Slot Acc {} F1 {}".format(oracle_joint_acc, oracle_f1, oracle_acc))
    json.dump(predictions, open(args['path'] + '/predictions_{}_epoch{}_ptest{}-{}.json'.format(args['test_split'], args['eval_epoch'], args['p_test'], args['p_test_fertility']), 'w'), indent=4)
    json.dump(oracle_predictions, open(args['path'] + '/oracle_predictions_{}_epoch{}_ptest{}-{}.json'.format(args['test_split'], args['eval_epoch'], args['p_test'], args['p_test_fertility']), 'w'), indent=4)

test_tagged_uttr = False 
saved = pkl.load(open(args['path'] + '/data.pkl', 'rb'))
if args['eval_epoch'] > -1: 
    model = torch.load(args['path'] + '/model_epoch{}.pth.tar'.format(args['eval_epoch']))
else:
    model = torch.load(args['path'] + '/model_best.pth.tar') 
model.cuda()
evaluator = Evaluator(saved['SLOTS_LIST'])
with open(args['path'] + '/eval_{}_epoch{}_ptest{}-{}.csv'.format(args['test_split'], args['eval_epoch'], args['p_test'], args['p_test_fertility']), 'w') as f:
    f.write('joint_lenval_acc,joint_acc,slot_acc,f1,oracle_joint_acc,oracle_slot_acc,oracle_f1\n')
model.eval()
run_test(saved[args['test_split']], model)     
