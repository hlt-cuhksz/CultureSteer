# 读取steer文件夹下的json 每个model转换为一个excel
import os
import json

import pandas as pd
import numpy as np


# models = ['Llama','Qwen','Mis']
# models = ['LLM','Qwen']
# models = ['Llama','Qwen','CultureLLM','CultureLLMQwen','SimLLM','SimQwen','CultureMerge','CultureSPA']
# models = "'Llama' 'Qwen' 'CultureLLMLlama' 'CultureLLMQwen' 'SimLLMLlama' 'SimLLMQwen' 'CultureMergeLlama' 'CultureSPALlama' 'Llamacsp' 'Llamacct' 'Qwencsp' 'Qwencct' 'Llama1shot' 'Llama3shot' 'Llama5shot' 'Qwen1shot' 'Qwen3shot' 'Qwen5shot' 'Llama_lora_steer' 'Qwen_lora_steer'"
models = "''Llama' 'Qwen' 'CultureLLMLlama' 'CultureLLMQwen' 'SimLLMLlama' 'SimLLMQwen' 'CultureMergeLlama' 'CultureSPALlama' 'Llamacsp' 'Llamacct' 'Qwencsp' 'Qwencct' 'Llama_lora_steer' 'Qwen_lora_steer'"
models = models.split(' ')
models = [model.replace("'", '') for model in models]
# models = ['Llama','Qwen','Llama3shot', 'Llama5shot', 'Qwen3shot', 'Qwen5shot', 'CultureLLM','CultureLLMQwen','SimLLM','SimQwen','Llama_lora_steer','Qwen_lora_steer']

TopK = [3,5,10,20]
langs = ['USA','UK','OC','CN']
data_dir = '../../results/jsons'
tabel_result_dir = '../../results/tables'

model_df = pd.DataFrame()
for model in models:
    method_df = pd.DataFrame()
    for lang in langs:
        json_file = os.path.join(data_dir, f'{lang}_{model}.json')
        with open(json_file,'r') as f:
            data = json.load(f)
        model_lang = []
        for k in TopK:
            model_lang_k = data[f'{k}'][f'total_{k}'][0]
            model_lang.append(model_lang_k)
        
        model_method = pd.DataFrame(data=[model_lang], index=[model], columns=[lang + f'_{k}' for k in TopK])
        model_method = model_method.applymap(lambda x: np.round(x * 100, 2))
        method_df = pd.concat([method_df,model_method],axis=1)
    
    model_df = pd.concat([model_df,method_df],axis=0)
    method_df.to_excel(os.path.join(tabel_result_dir, f'{model}_result_main.xlsx'))

model_df.to_excel(os.path.join(tabel_result_dir, f'main_result.xlsx'))
