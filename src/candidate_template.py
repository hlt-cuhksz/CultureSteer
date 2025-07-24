def candidate_template(cue_word, lang, type=None):
    sys_template = None
    # culture_infor = f'You are a person with {lang2cultrue(lang)} cultural background. ' if lang != 'CN' else f'你是一个中国文化背景的人。'
    if lang != 'CN':
        culture_infor = f'You are a person with {lang2cultrue(lang)} cultural background.'
        template = f'When "{cue_word}" is mentioned, people often think of the following three words:'
        task_infor = 'You will be performing a word association task. Please directly answer the association word.'
        cross_infor = f'Before you respond, take a moment to think about how {lang2cultrue(lang)} culture is different from {", ".join(lang2diffculture(lang))} cultures.'
    else:
        culture_infor = f'你是一个中国文化背景的人。'
        template = f'当提起"{cue_word}",人们往往会想到的三个词是:'
        task_infor = '你将进行词联想任务，请直接说出你联想到的词。'
        cross_infor = f'在回答之前, 请你注意中国文化与美国、英国、大洋洲文化的不同。'
    if type:
        if type == 'inf':
            sys_template = culture_infor + task_infor
        elif type == 'cct':
            sys_template = culture_infor + task_infor + cross_infor
    # if sys_template:
    #     return sys_template, template# 如果非None则返回两个值
    # else: # 如果None则返回一个值
    return str(sys_template) + str(template)

def lang2cultrue(lang):
    if lang == 'USA':
        culture = 'United State'
    elif lang == 'UK':
        culture = 'United Kingdom'
    elif lang == 'OC':
        culture = 'Oceania'
    elif lang == 'CN':
        culture = 'Chinese'
    return culture

def lang2diffculture(lang):
    LANG = ['USA','UK','OC','CN']
    return [lang2cultrue(l) for l in LANG if l != lang]

if __name__ == '__main__':
    candidate_template('cat','USA','cct')
    candidate_template('猫','CN','cct')