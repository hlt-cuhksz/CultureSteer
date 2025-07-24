# 设置candidate的prompt并将asso word作为候选词, 得到候选词id并计算平均概率
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # 设置使用的GPU设备
from tqdm import tqdm
import json
from scipy.stats import spearmanr
tqdm.pandas()

import math
import argparse
import torch
import pandas as pd
import numpy as np
# from candidate_template import candidate_template
from tqdm import tqdm
try:
    from .utils import _cal_R, cal_candidate, get_ranking, asso_word_set_main
    from .config import Config
except:
    from utils import _cal_R, cal_candidate, get_ranking, asso_word_set_main
    from config import Config
TOP_K = [3,5,10,20]


# 存入result的json, 按sum和chap
def cal_overall(data,args):
    result_total = {}
    for k in TOP_K:

        result = {}
        result['total'] = []
        for d in tqdm(data):
            chap = d['chap']
            if chap not in result.keys():
                result[chap] = []
            result[chap].append(d[f'pwk_{k}'])
            result['total'].append(d[f'pwk_{k}'])
        for key in result.keys():
            result[key] = [np.mean(result[key]), np.std(result[key],ddof=1),len(result[key])] # 保留t检验的必要统计量
        # 对key排序
        result_sort = {str(i):result[str(i)] for i in range(1,23) if str(i) in result.keys()}
        result_sort = {**{f'total_{k}':result['total']},
                          **result_sort}
        result_total[str(k)] = result_sort # 把当前top选定结果作为一个value
    with open(os.path.join(args.pwk_dir, f'{args.lang}_{args.model_name}.json'), 'w') as f:
        json.dump(result_total, f, indent=4)
    print(f'finish written {args.lang}_{args.model_name}.json')
    return result_total


def main(args,config):
    with open(os.path.join(args.data_dir, f'{args.lang}_steer.json'), 'r') as f:
        data = json.load(f)
    # data = data[:2] # try
    save_dict = cal_candidate(data, args.lang, config, args=args)

    if not hasattr(args,'steer_type'):
        lang = 'EN' if args.lang != 'CN' else 'CN'
    else:
        lang = args.lang
    torch.save(save_dict, os.path.join(args.score_dir, f'{lang}_{args.model_name}.pt'))
    print(f'the probability has been saved to the file {lang}_{args.model_name}.pt')

def var_main(args,config):
    with open(os.path.join(args.data_dir, f'{args.lang}_steer.json'), 'r') as f:
        data = json.load(f)
    # data = data[:20]
    def max_RK(token_space):
        ans_prob = get_ranking(asso_word_set['word set'], score_pt[cue].to(config.device), args.batch_size, lang, config, token_space=token_space)
        data[i]['candidate_p'] = ans_prob

        prob_index = [i for i, prob in sorted(enumerate(ans_prob), key=lambda x: x[1], reverse=True) if prob > 0.0] # 获取非零且排序的原始index
        asso_index = [asso_word_set['word set'].index(word) for word in asso] # asso word的index -- 不会和cue重复
        return asso_index, prob_index
    
    asso_word_set_path = os.path.join(args.data_dir, 'EN_word_set.json') if args.lang != 'CN' else os.path.join(args.data_dir, 'CN_word_set.json')

    with open(asso_word_set_path, 'r') as f:
        asso_word_set = json.load(f)
    asso_word_set['word set'] = list(set(asso_word_set['word set'])) # 再次确保去重
    # data = data[:2] # try
    if not hasattr(args,'steer_type'): # 如果不是lora_steer 则只有两个语言的score(英语的都一样)
        lang = 'EN' if args.lang != 'CN' else 'CN'
    else:
        lang = args.lang
    if config.device.type == 'cuda':

        score_pt = torch.load(os.path.join(args.score_dir, f'{lang}_{args.model_name}.pt'))
    else:
        score_pt = torch.load(os.path.join(args.score_dir, f'{lang}_{args.model_name}.pt'), map_location = torch.device('cpu'))
    for i in tqdm(range(len(data))):
        cue = data[i]['EN cue'] if args.lang != 'CN' else data[i]['CN cue']
        asso = [word for word in data[i]['asso words'] if word != cue] # 去重

        asso_index_t, prob_index_t = max_RK(True)
        asso_index_f, prob_index_f = max_RK(False)
        for top_k in TOP_K:
            data[i][f'pwk_{top_k}'] = max(_cal_R(asso_index_t, prob_index_t, top_k), _cal_R(asso_index_f, prob_index_f, top_k))
    cal_overall(data,args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',default='Llama',type=str)
    parser.add_argument('--data_dir',default='../dataset',type=str)
    parser.add_argument('--lang',default='USA',type = str)
    parser.add_argument('--score_dir',default='../results/scores',type=str)
    parser.add_argument('--pwk_dir',default='../results/jsons',type=str)
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--baseline',default=False)
    parser.add_argument('--runner', default='cal_p', type=str,help = 'calculate the scores or calculate the pwk based on scores')
    args = parser.parse_args()
    config = Config(args.model_name)

    if args.runner == 'cal_p':
        model, tokenizer, device = config.select_model()
        main(args,config)
    elif args.runner == 'cal_s':
        config.device = torch.device('cpu') # no required GPU
        model, tokenizer, device = config.select_model()
        asso_word_set_main(args) # get word set first
        var_main(args,config)


