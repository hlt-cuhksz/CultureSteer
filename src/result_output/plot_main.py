import pandas as pd
import json
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.lines as mlines
import seaborn as sns
import itertools
TOP_K = [3,5,10,20]

# CHAPTER_NAME = [
#     'Physical world',
#     'Kinship',
#     'Animals',
#     'The body',
#     'Food and drink',
#     'Clothing and grooming',
#     'The house',
#     'Agricultural and vegetation',
#     'Action and technology',
#     'Motion',
#     'Possession',
#     'Spatial relations',
#     'Quantity',
#     'Time',
#     'Sense perception',
#     'Emotion and values',
#     'Cognition',
#     'Speech and Language',
#     'Society and politics',
#     'Warfare and hunting',
#     'Law',
#     'Religion and belief'
# ]

# 简化版本
CHAPTER_NAME = [
    'World', # Physical world
    'Kinship',
    'Animals',
    'Body',
    'Food', # Food and drink
    'Cloth', # Clothing and grooming
    'House',
    'Agriculture', # Agricultural and vegetation
    'Action', # Action and technology
    'Motion',
    'Possession',
    'Space', # Spatial relations
    'Quantity',
    'Time',
    'Sense', # Sense perception
    'Values', # Emotion and values
    'Cognit', # Cognition
    'Language', # Speech and Language
    'Society', # Society and politics
    'Warfare', # Warfare and hunting
    'Law',
    'Religion' # Religion and belief
]

ROUGH_RANGE = [[i for i in range(1,9)], [i for i in range(9,16)], [i for i in range(16,23)]]
ROUGH_CHAPTER = ['WORLD', 'SENSE & ACTION', 'VALUES']


LANGUAGE_COLOR = ['blue','green','orange','red']
LANGUAGE_COLOR_2 = ['#5F9FFF', '#98FB98', '#FFD700', '#FF6347']  # 浅色调
LANGUAGE_COLOR_3 = ['#5F9FFF', '#98FB98', '#FFD700', '#FF6347']  # 更深的颜
def get_xlabel_colors(chapter_names):
    # 为不同索引范围分配颜色
    colors = []
    for idx, chapter in enumerate(chapter_names):
        if 0 <= idx <= 7:
            colors.append('#990000')  # 深红色
        elif 8 <= idx <= 13:
            colors.append('#006400')  # 深绿色
        else:
            colors.append('#00008B')  # 深蓝色
    return colors

def json_to_dataframe(model,lang,method='',result_dir=''):
    if method == 'baseline':
        with open(os.path.join(result_dir,f'{lang}_{model}.json'),'r') as f:
            data = json.load(f)
    else:
        
        with open(os.path.join(result_dir,f'{lang}_{model}{method}.json'),'r') as f:
            data = json.load(f)


    # 提取需要的行：3, 5, 10, 20
    rows = []
    for key in ['3', '5', '10', '20']:
        row = data[key]
        row_data = {
            'total': row[f'total_{key}'][0], 
        }
        # 将1到22的值添加到row_data
        for i in range(1, 23):
            row_data[str(i)] = row.get(str(i), None)[0] # 除以平均
        rows.append(row_data)
        # 计算三大类的均值 -- 均值*数量/总数
        # for cata in range(3):
        #     row_data[f'cata_{cata}'] = np.sum([row[str(i)][0]*row[str(i)][2] for i in ROUGH_RANGE[cata]])/np.sum([row[str(i)][2] for i in ROUGH_RANGE[cata]])
    if method:
        df = pd.DataFrame(rows, index=[f'{model}_{lang}{method}_{k}' for k in TOP_K]) # 添加method信息
    else:
        df = pd.DataFrame(rows, index=[f'{model}_{lang}_{k}' for k in TOP_K])
    df = df.applymap(lambda x: np.round(x, 2)) # 乘以100并保留两位小数
    return df

def turn(vec1,vec2):
    vec1_turn = np.array([v1 if v1 > v2 else v2 for v1,v2 in zip(vec1,vec2)])
    vec2_turn = np.array([v2 if v1 > v2 else v1 for v1,v2 in zip(vec1,vec2)])
    return vec1_turn,vec2_turn

def plot_radar(ax, data, categories, title, color, xlabel_colors=None):
    xlabel_colors = get_xlabel_colors(CHAPTER_NAME) # 类别标签颜色赋值
    # 角度计算
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values = data.flatten().tolist()

    # 确保雷达图闭合
    values += values[:1]
    angles += angles[:1]
    ax.set_facecolor('#F7F7F7')  # 设置图形背景为灰色

    # 绘制雷达图
    ax.fill(angles, values, color=color, alpha=0.05)
    ax.plot(angles, values, color=color, linewidth=0.08, label=title, marker='o', markersize=0.1, alpha=1, markeredgewidth=0)  # 添加label用于图例
    ax.set_yticklabels([])  # 不显示径向的刻度标签
    
    # 设置角度并添加颜色
    ax.set_xticks(angles[:-1])  # 设置节点角度
    ax.set_xticklabels(categories, rotation=45, rotation_mode='default', fontsize=10)  # 默认颜色
    # 为每个标签单独设置颜色
    if xlabel_colors is not None:
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_color(xlabel_colors[i])  # 为每个标签设置颜色

def plot_subradar(df, models, langs, top_k, save_path, args):
    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(15, 5), subplot_kw={'polar': True})
    
    if len(models) == 1:
        axes = [axes]  # 保证axes是可迭代对象
    # 1x4布局时axes已经是一维的，不需要chain
    
    # 定义颜色
    colors = ['red', 'green', 'blue', 'orange']
    
    # 对每个model绘制子图
    for i, (ax, model) in enumerate(zip(axes, models)):
        # 筛选出该model的所有结果
        if 'LoRAMAE' not in model:
            model_data = df[(df.index.str.startswith(model) & (df.index.str.endswith(str(top_k))) & ~(df.index.str.contains('LoRAMAE')))]
            # ax.set_yticks(np.linspace(0,0.5,5))
            # ax.set_yticks(np.linspace(0,100,5)) # 统一范围 
        else:
            model_data = df[(df.index.str.startswith(model) & (df.index.str.endswith(str(top_k))) & (df.index.str.contains('LoRAMAE')))]
            # ax.set_yticks(np.linspace(0,100,5))
        model_data = model_data.drop(columns='total')
        categories = CHAPTER_NAME
        for lang_idx, lang in enumerate(langs): 
            lang_values = model_data.loc[f'{model}_{lang}_{top_k}'].values
            color = LANGUAGE_COLOR[lang_idx]
            title = f'{lang}'  # 设置每个语言的标题

            # 你可以根据语言的不同设置不同颜色
            xlabel_colors = [LANGUAGE_COLOR[lang_idx]] * len(categories)  # 为每个xlabel分配相同的颜色

            # 调用雷达图绘制函数
            plot_radar(ax, lang_values, categories, title, color, xlabel_colors)
        
        # 设置模型的标题
        if 'LLM' in model:
            ax.set_title(f'{model}'.replace('LLM', 'Llama'), fontsize=20, y=1.1)
        # elif 'Mis' in model:
        #     ax.set_title(f'{model}'.replace('Mis', 'Mistral'), fontsize=20, y=1.1)
        elif 'lora_steer' in model:
            ax.set_title(f'{model}'.replace('lora_steer', 'CultureSteer'), fontsize=20, y=1.1)
        else:
            ax.set_title(f'{model}', fontsize=20, y=1.1)

        # if 'lora_steer' not in model:
        #     # y_ticks = np.linspace(0, 0.5, 6)  # 生成 5 个自适应刻度
        #     y_ticks = np.linspace(0, 100, 6)  # 生成 5 个自适应刻度
        # else:
        y_ticks = np.linspace(0, 4, 6)  # 生成 5 个自适应刻度
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{tick:.2f}' for tick in y_ticks], fontsize=10)
        # 添加图例 -- 只添加到第一个子图即可
        if i == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type',default='inf',type=str)
    parser.add_argument('--data_dir',default='/home/daixunlian/word_asso/dataset')
    parser.add_argument('--plot_type',default='stick')
    parser.add_argument('--model_type',default='default')
    parser.add_argument('--plot_dir',default='../../results/plots')
    parser.add_argument('--result_dir',default='../../results/jsons')
    args = parser.parse_args()


    langs = ['USA','UK','OC','CN']

    dfs = pd.DataFrame()


    models = ['Llama','Qwen','Llama_lora_steer','Qwen_lora_steer']
    dfs = pd.DataFrame()
    for model in models:
        for lang in langs:
            df = json_to_dataframe(f'{model}',lang,result_dir=args.result_dir)
            dfs = pd.concat([dfs, df], axis=0)
    for k in TOP_K:
        plot_subradar(dfs, models, langs, k, os.path.join(args.plot_dir, f'main_radar_{k}.pdf'), args)






