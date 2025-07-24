import json
# 考虑四个语言 首先用align筛选 然后转成四个json
import pandas as pd

def query_template_EN(cue):
    return f'"{cue}" is associated with the word:',f'"{cue}" is often linked to:',f'People often associate "{cue}" with:',f'"{cue}" is commonly connected to:',f'The "{cue}" brings to mind the word:'

def query_template_CN(cue):
    return f'"{cue}"能够联想到:',f'能够与"{cue}"产生联想的词是:',f'看到"{cue}"通常会想到的词是:',f'一提到"{cue}"就会想到:',f'"{cue}"让人联想到:'

def merge(lang):
    df = pd.read_excel(f'/home/daixunlian/word_asso/dataset/participant/{lang}_participant.xlsx')
    # return pd.merge(df, align, left_on = 'cue', right_on = lang)
    return df

def to_clean_participant(df, lang):
    df.to_excel(f'/home/daixunlian/word_asso/dataset/participant/{lang}_participant_clean.xlsx')
    # return df


def cal_WSG(df, isLLM=''):
    df_asso = pd.concat([df['R1{}'.format(isLLM)], df['R2{}'.format(isLLM)], df['R3{}'.format(isLLM)]], axis=0)
    df_cue = pd.concat([df['cue'], df['cue'], df['cue']], axis=0)
    df_EN = pd.concat([df['EN'], df['EN'], df['EN']], axis=0)
    # df_PR = pd.concat([df['PR'], df['PR'], df['PR']], axis=0)
    df_CN = pd.concat([df['CN'], df['CN'], df['CN']], axis=0)
    df_chap = pd.concat([df['chap'], df['chap'],df['chap']],axis=0)
    df_new = pd.DataFrame({
        'chap':df_chap,
        'EN_cue': df_EN,
        # 'PR_cue': df_PR,
        'CN_cue': df_CN,
        'cue': df_cue,
        'R1{}'.format(isLLM): df_asso,
    })
    df_tmp = df_new.loc[:, 'chap':'cue'].copy()
    df_tmp = df_tmp.drop_duplicates(keep='first')

    df_new = df_new.loc[(df_new['R1{}'.format(isLLM)] != '') & (df_new['R1{}'.format(isLLM)] != '\n')]
    df_new = df_new.groupby('cue')['R1{}'.format(isLLM)].value_counts().reset_index(name='Counts')
    df_new = pd.merge(df_new, df_tmp, on='cue', how='inner')  # 连接align和count

    df_new['sum'] = df_new.groupby('cue')['Counts'].transform('sum').to_frame()
    df_new['FSG'] = df_new['Counts'] / df_new['sum']
    df_symmetry = df_new[['R1{}'.format(isLLM), 'cue', 'Counts', 'FSG', 'sum']]
    df_tmp = pd.merge(df_new, df_symmetry, left_on=['cue', 'R1{}'.format(isLLM)],
                      right_on=['R1{}'.format(isLLM), 'cue'], how='left',
                      suffixes=('', '_sym'))

    df_new['BSG'] = df_tmp['Counts_sym'] / df_tmp['sum_sym']
    df_new.loc[pd.isnull(df_new['BSG']), 'BSG'] = 0.0  # 如果没有则取0
    # df_new['WSG'] = 0.8 * df_new['FSG'] + 0.2 * df_new['BSG']
    df_new['WSG'] = df_new['FSG']
    return df_new

def to_json(df,lang):
    result = []
    chapter_name = [
        'Physical world',
        'Kinship',
        'Animals',
        'The body',
        'Food and drink',
        'Clothing and grooming',
        'The house',
        'Agricultural and vegetation',
        'Action and technology',
        'Motion',
        'Possession',
        'Spatial relations',
        'Quantity',
        'Time',
        'Sense perception',
        'Emotion and values',
        'Cognition',
        'Speech and Language',
        'Society and politics',
        'Warfare and hunting',
        'Law',
        'Religion and belief'
    ]
    # 按 "EN cueword" 分组
    grouped = df.groupby("cue")
    print(df.columns,flush=True)
    # 遍历每个分组
    for cue, group in grouped:
        chap_id = group["chap"].iloc[0]
        try:
            en_cueword = group['EN'].iloc[0]
            cn_cueword = group["CN"].iloc[0]
        except:
            en_cueword = group['EN_cue'].iloc[0]
            cn_cueword = group['CN_cue'].iloc[0]
        
        asso_word_list = group["R1"].tolist()
        wsg_list = group["WSG"].tolist()
        
        # 通过 chapid 获取章节名称
        chap_name = chapter_name[chap_id - 1]  # 假设 chapid 从 1 开始索引
        # # 每个query重复五次
        # querys = query_template_EN(en_cueword) if lang!='CN' else query_template_CN(cn_cueword)
        # for query in querys:
        # # 构建单个 JSON 对象
        #     result.append({
        #         "chap": str(chap_id),
        #         "chap_name": chap_name,
        #         "EN cue": en_cueword,
        #         "CN cue": cn_cueword,
        #         "asso words": asso_word_list,
        #         "WSG": wsg_list,
        #         'query':query
        #     })
        
        # 不构建query
        # 过滤多词
        _asso_word_list = [word.strip() for word in asso_word_list if len(word.strip().split()) == 1]
        _wsg_list = [wsg_list[i] for i, word in enumerate(asso_word_list) if len(word.strip().split()) == 1]
        result.append({
            "chap": str(chap_id),
            "chap_name": chap_name,
            "EN cue": en_cueword,
            "CN cue": cn_cueword,
            "asso words": _asso_word_list,
            "WSG": _wsg_list,
        })
    with open(f'/home/daixunlian/word_asso/dataset/{lang}.json','w') as f:
        json.dump(result, f, ensure_ascii=False,indent=4)


def align():
    # align = pd.read_excel('/home/daixunlian/word_asso/dataset/alignment/alignment_standard.xlsx')
    CN, USA, UK, OC = merge('CN'), merge('USA'), merge('UK'), merge('OC')
    cue_set = set(CN['EN']) & set(USA['EN']) & set(UK['EN']) & set(OC['EN'])
    CN = CN.loc[CN['EN'].isin(cue_set)].reset_index(drop=True)
    USA = USA.loc[USA['EN'].isin(cue_set)].reset_index(drop=True)
    UK = UK.loc[UK['EN'].isin(cue_set)].reset_index(drop=True)
    OC = OC.loc[OC['EN'].isin(cue_set)].reset_index(drop=True)
    USA,UK,OC,CN = cal_WSG(USA),cal_WSG(UK),cal_WSG(OC),cal_WSG(CN)
    to_json(USA,'USA')
    to_json(UK,'UK')       
    to_json(OC,'OC')
    to_json(CN,'CN')
    # to_clean_participant(CN,'CN')
    # to_clean_participant(USA,'USA')
    # to_clean_participant(UK,'UK')
    # to_clean_participant(OC,'OC')

def main():
    data = []
    for lang in ['USA','UK','OC','CN']:
        with open(f'/home/daixunlian/word_asso/dataset/{lang}.json', 'r') as f:
            data.append(json.load(f))
    data_new = [[],[],[],[]]
    for i,d in enumerate(data[0]):

        CN_index = next((k for k, item in enumerate(data[3]) if (item['EN cue'] == data[0][i]['EN cue']) & (item['CN cue'] == data[0][i]['CN cue'])),None)

        if not CN_index:
            continue
        
        lent = min([len(data[k][i]['asso words']) for k in range(3)] + [len(data[3][CN_index]['asso words'])])

        for j in range(3):
            data_new[j].append(data[j][i])
            data_new[j][-1]['asso words'] = data[j][i]['asso words'][:lent] # 截断
        data_new[3].append(data[3][CN_index])
        data_new[3][-1]['asso words'] = data[3][CN_index]['asso words'][:lent]
        
    for i, lang in enumerate(['USA','UK','OC','CN']):
        with open(f'/home/daixunlian/word_asso/dataset/{lang}_2.json', 'w') as f:
            json.dump(data_new[i],f,indent=4,ensure_ascii=False)


if __name__ == '__main__':
    # align()
    # 截断数据
    main()