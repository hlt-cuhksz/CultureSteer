import torch
try:
    from .candidate_template import candidate_template
except:
    from candidate_template import candidate_template
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import itertools
import json
import random
import itertools
def filter_special(word_ids,config):
    tokenizer = config.tokenizer
    word_ids = word_ids[word_ids != tokenizer.pad_token_id]  # Removing padding token IDs
    word_ids = word_ids[word_ids != tokenizer.bos_token_id] 
    return word_ids


def get_ranking(asso_words, score, batch_size, lang, config, token_space=True):
    tokenizer = config.tokenizer
    device = config.device
    # model = config.model
    # GPT优化：复杂度 + batch计算prob
    score = score.squeeze(dim=1)  # Removing extra dimension
    ans_probs_total = []
    score_prob = torch.softmax(score,dim = -1)
    max_prob_text = tokenizer.decode(torch.argmax(score_prob,dim = -1))
    # Process in batches
    for i in range(0, len(asso_words), batch_size):
        if token_space:
            batch_asso_word = [' ' + word.replace(',','') if lang != 'CN' else word.replace(',','') for word in asso_words[i:i + batch_size]] 
        else:
            batch_asso_word = [word.replace(',','') if lang != 'CN' else word.replace(',','') for word in asso_words[i:i + batch_size]] 
        tokenized_words = tokenizer.batch_encode_plus(batch_asso_word, return_tensors='pt', padding=True, truncation=True).to(device)
        input_ids = tokenized_words['input_ids']
        ans_probs = []

        for word_idx in range(len(asso_words[i:i+batch_size])):
            word_ids = input_ids[word_idx]  
            word_ids = filter_special(word_ids, config)
            n = word_ids.size(0)  
            ans_subprobs = []  
            # 滑动取最大概率
            for j in range(score.shape[0] - n + 1):
                probs = score_prob[j:j + n ,:]
                token_probs = probs[range(n), word_ids]  # Shape: (n,)
                
                if 0 in token_probs:
                    ans_subprobs.append(torch.tensor(0, device=device))
                else:    
                    ans_subprobs.append(token_probs.mean()) # 把这些只有一个token出现的词放到最后
            
            # If there are valid subprobs, use the maximum as the word score
            if ans_subprobs: # 处理CN的单一个词语的情况
                ans_probs.append(torch.max(torch.tensor(ans_subprobs, device=device)))  # Max of subprobs
            else:
                ans_probs.append(torch.tensor(0.0, device=device))  # If no valid subprobs, set to 0.0
        ans_probs_total.extend(ans_probs)  # Store the results for this batch
    return [prob.cpu().item() if prob != 0.0 else 0 for prob in ans_probs_total]


def cal_candidate(data,lang,config,args=None):
    # 一条一条的计算，暂时不要batch计算
    # 用p存放candidata的概率
    # 如果是few-shot则添加前缀
    save_dict = {}

    for i in tqdm(range(len(data))):
        cue = data[i]['EN cue'] if lang != 'CN' else data[i]['CN cue']
        if 'csp' in args.model_name or 'cct' in args.model_name:
            template = 'csp' if 'csp' in args.model_name else 'cct'
        else:
            template = None
        prompt = candidate_template(cue,lang,template,args) #prompts 
        if args and 'shot' in args.model_name:
            few_shot = args.model_name[args.model_name.find('shot')-1]
            few_shot = int(few_shot)       
            randindex = random.sample(range(len(data)),few_shot)
            while randindex.count(i) > 0: # 确保randindex中不包含当前的i
                randindex = random.sample(range(len(data)),few_shot)
            selected_examples = [data[j] for j in randindex]
            prefix_parts = []
            cue_key = 'CN cue' if lang == 'CN' else 'EN cue'
            for example in selected_examples:
                example_cue = example[cue_key]
                # randindex = random.sample(range(len(example['asso words'])),1)

                example_asso_word = ','.join(example['asso words'])
                
                
                # 根据语言选择对应的模板
                if lang == 'CN':
                    example_text = f'当提起"{example_cue}",人们往往会想到的词是:{example_asso_word}'
                else:
                    example_text = f'When "{example_cue}" is mentioned, people often think of the words: {example_asso_word}'
                
                prefix_parts.append(example_text)
            
            # 组合前缀，用换行符分隔
            few_shot_prefix = '\n'.join(prefix_parts) + '\n'
            prompt = few_shot_prefix + prompt
        P_cand = __cal_candidate(prompt,lang,config,args=args)
        if lang != 'CN':
            save_dict[data[i]['EN cue']] = P_cand
        else:
            save_dict[data[i]['CN cue']] = P_cand
    return save_dict


def __cal_candidate(prompt, lang, config, args=None):
    tokenizer = config.tokenizer
    device = config.device
    model = config.model
    if hasattr(args,'steer_type') and args.steer_type == 'lora_steer':
        Culture = ['USA','UK','OC','CN']
        if args.cross_steer_lang: # 如果是cross steer验证的情况
            if args.cross_steer_lang in Culture:
                steer_values = torch.tensor([Culture.index(args.cross_steer_lang)], device=device)
            else:
                steer_values = torch.tensor([Culture.index('USA')], device=device) # 默认使用USA -- 但是steer层的epsilon值已经置0
        else:
            steer_values = torch.tensor([Culture.index(lang)], device=device)
        scores = steer_eazy_generate(prompt, steer_values, tokenizer, model)
        return scores
    else:
        inputs = tokenizer.encode_plus(
            prompt,return_tensors="pt", 
            padding=True, 
            truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        outputs = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                max_new_tokens=args.max_token if args else 5, 
                output_scores=True,
                temperature = args.temperature if args else 1.0,
                return_dict_in_generate=True
            )
        generated_ids = outputs['sequences']
        scores = outputs['scores']
        scores = torch.stack(scores, dim=0) # 把每一步解码合并成tensor
        return scores


# evaluate
def _cal_TOPK(asso_index, prob_index, top_k, args):
    if args.topk_type == 'pwk':
        top_index = prob_index[:top_k] if len(prob_index) >= top_k else prob_index[:]
        R = 0.0
        for i, idx in enumerate(top_index):
            if idx in asso_index:
                R += 1 / (asso_index.index(idx) + 1) # 检索出来的index在原index的位置的索引 + 1 的倒数
        return R / np.sum([1 / (i+1) for i in range(len(asso_index))])
    elif args.topk_type == 'dcg':
        top_index = prob_index[:top_k] if len(prob_index) >= top_k else prob_index[:]
        dcg = 0.0
        for i, idx in enumerate(top_index):
            rel = 1 if idx in asso_index else 0
            rank = i + 1
            dcg += rel / np.log2(rank + 1)
        return dcg

def asso_word_set_main(args):
    EN_asso_word_set, CN_asso_word_set = [], []
    for lang in ['USA','UK','OC']:

        EN_asso_word_set += _asso_word_set(lang,args)
    CN_asso_word_set += _asso_word_set('CN',args)
    EN_asso_word_set = list(set(EN_asso_word_set))
    CN_asso_word_set = list(set(CN_asso_word_set))
    # 写入原始数据文件夹
    with open(os.path.join(args.data_dir,f'EN_word_set.json'),'w') as f:
        json.dump({'word set':EN_asso_word_set},f,indent=4,ensure_ascii=False)
    with open(os.path.join(args.data_dir,f'CN_word_set.json'),'w') as f:
        json.dump({'word set':CN_asso_word_set},f,indent=4,ensure_ascii=False)
    print('word set has been written')

# wordset
def _asso_word_set(lang,args):
    # 根据runner选择构建的词表
    with open(os.path.join(args.data_dir, f'{lang}_steer.json'), 'r') as f:
        data = json.load(f)
    lang = 'EN' if lang != 'CN' else 'CN'
    word_set = list(set(list(itertools.chain(*[d['asso words'] for d in data]))))
    return word_set

def steer_generate(prompt_data, steer_values, tokenizer, model,
            temperature=0.9):
    for i,_prompt in enumerate(prompt_data):
        tokenized = tokenizer(_prompt,return_tensors="pt")
        text, score = model.generate(
            input_ids=tokenized['input_ids'].to(model.model.device), 
            attention_mask=tokenized['attention_mask'].to(model.model.device), 
            steer_values = torch.where(steer_values[i,:] == 1), # 这是额外添加的参数
            seed = 0, # 设置种子 -- 在model_base中 
            max_new_tokens=5, 
            output_scores=True,
            temperature = temperature,
            return_dict_in_generate=True
            )
        print(text,flush=True)
    return score

def steer_eazy_generate(prompt_data, steer_values, tokenizer, model,
            temperature=1.0):
    tokenized = tokenizer(prompt_data,return_tensors="pt")
    text, score = model.generate(
        input_ids=tokenized['input_ids'].to(model.model.device), 
        attention_mask=tokenized['attention_mask'].to(model.model.device), 
        steer_values = steer_values, # 这是额外添加的参数
        seed = 0, # 设置种子 -- 在model_base中 
        max_new_tokens=5, 
        output_scores=True,
        temperature = temperature,
        return_dict_in_generate=True
        )
    # print(text,flush=True)
    return score


