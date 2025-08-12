import json
import pandas as pd
cross_dir = '../../results/dcg_jsons'
write_dir = '../../results/tables'

LANG = ['USA', 'UK', 'OC','CN']

# 计算other lang
language_scores = []

TOPK = ['3','5','10','20']

def cross2excel(model_name):
    total_score = []
    model_df = pd.DataFrame()
    for k in TOPK:
        lang_scores = []
        for lang in LANG:
            topK_score = [0] * 4
            with open(f'{cross_dir}/{lang}_{model_name}_lora_steer.json', 'r') as f:
                data = json.load(f)
            # 提取TopK的得分
            topK_score[LANG.index(lang)] = round(data[f'{k}'][f'total_{k}'][0], 4)
            # lang_scores.append(topK_score)
            other_lang = [l for l in LANG if l != lang]
            for other in other_lang:
                with open(f'{cross_dir}/{lang}_{other}_{model_name}_lora_steer.json', 'r') as f:
                    data = json.load(f)
                topK_score[LANG.index(other)] = round(data[f'{k}'][f'total_{k}'][0], 4)
            
            lang_scores.append(topK_score)
        lang_df = pd.DataFrame(lang_scores, columns=LANG, index=[model_name + str(k) + lang for lang in LANG])
        lang_df = lang_df.applymap(lambda x: round(x, 2))
        model_df = pd.concat([model_df,lang_df],axis=0)
        lang_df.to_excel(f'{write_dir}/cross_{model_name}_lora_steer{k}.xlsx', index=True)
        total_score.append(lang_scores)
    model_df.to_excel(f'{write_dir}/cross_{model_name}_cross.xlsx', index=True)

# def main2excel(model_name):
#     methods = [']

if __name__ == '__main__':
    model_names = ['Llama', 'Qwen']
    for name in model_names:
        cross2excel(name)
    

    



