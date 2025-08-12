cd ../src
export PYTHONPATH=/data/daixl/my_project/word_asso:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="4"

# baseline
model_names=('Llama' 'Qwen' 'CultureLLMLlama' 'CultureLLMQwen' 'SimLLMLlama' 'SimLLMQwen' 'CultureMergeLlama' 'CultureSPALlama' 'Llamacsp' 'Llamacct' 'Qwencsp' 'Qwencct' 'Llama3shot' 'Llama5shot' 'Qwen3shot' 'Qwen5shot')
# model_names=('Llama1shot' 'Llama3shot' 'Llama5shot' 'Qwen1shot' 'Qwen3shot' 'Qwen5shot')
# model_names=('CultureLLMQwen')

for model_name in "${model_names[@]}"; do
    python cal_candidate.py --lang USA --model_name "$model_name" --runner cal_s &
    python cal_candidate.py --lang UK --model_name "$model_name" --runner cal_s &
    python cal_candidate.py --lang OC --model_name "$model_name" --runner cal_s &
    python cal_candidate.py --lang CN --model_name "$model_name" --runner cal_s &
    wait
done

model_names=('Llama' 'Qwen')
for model_name in "${model_names[@]}"; do
    python cal_candidate_steer_model.py --lang USA --model_name "$model_name" --runner cal_s &
    python cal_candidate_steer_model.py --lang UK --model_name "$model_name" --runner cal_s & 
    python cal_candidate_steer_model.py --lang OC --model_name "$model_name" --runner cal_s &
    python cal_candidate_steer_model.py --lang CN --model_name "$model_name" --runner cal_s &
    wait
done
# python cal_candidate.py --lang CN --model_name Qwen3shot --runner cal_s
# python cal_candidate.py --lang CN --model_name CultureLLMLlama --runner cal_a
# python cal_candidate.py --lang USA --model_name CultureLLMLlama --runner cal_a

# python cal_candidate.py --lang CN --model_name CultureMergeLlama --runner cal_s & 
# python cal_candidate.py --lang USA --model_name CultureMergeLlama --runner cal_s &
# python cal_candidate.py --lang UK --model_name CultureMergeLlama --runner cal_s &
# python cal_candidate.py --lang OC --model_name CultureMergeLlama --runner cal_s &
# wait


#!/bin/bash

# Explicitly run all models in parallel for each language
# python cal_candidate.py --lang USA --model_name Llama1shot --runner cal_s &
# python cal_candidate.py --lang USA --model_name Llama3shot --runner cal_s &
# python cal_candidate.py --lang USA --model_name Llama5shot --runner cal_s &
# python cal_candidate.py --lang USA --model_name Qwen1shot --runner cal_s &
# python cal_candidate.py --lang USA --model_name Qwen3shot --runner cal_s &
# python cal_candidate.py --lang USA --model_name Qwen5shot --runner cal_s &

# python cal_candidate.py --lang UK --model_name Llama1shot --runner cal_s &
# python cal_candidate.py --lang UK --model_name Llama3shot --runner cal_s &
# python cal_candidate.py --lang UK --model_name Llama5shot --runner cal_s &
# python cal_candidate.py --lang UK --model_name Qwen1shot --runner cal_s &
# python cal_candidate.py --lang UK --model_name Qwen3shot --runner cal_s &
# python cal_candidate.py --lang UK --model_name Qwen5shot --runner cal_s &

# python cal_candidate.py --lang OC --model_name Llama1shot --runner cal_s &
# python cal_candidate.py --lang OC --model_name Llama3shot --runner cal_s &
# python cal_candidate.py --lang OC --model_name Llama5shot --runner cal_s &
# python cal_candidate.py --lang OC --model_name Qwen1shot --runner cal_s &
# python cal_candidate.py --lang OC --model_name Qwen3shot --runner cal_s &
# python cal_candidate.py --lang OC --model_name Qwen5shot --runner cal_s &

# python cal_candidate.py --lang CN --model_name Llama1shot --runner cal_s &
# python cal_candidate.py --lang CN --model_name Llama3shot --runner cal_s &
# python cal_candidate.py --lang CN --model_name Llama5shot --runner cal_s &
# python cal_candidate.py --lang CN --model_name Qwen1shot --runner cal_s &
# python cal_candidate.py --lang CN --model_name Qwen3shot --runner cal_s &
# python cal_candidate.py --lang CN --model_name Qwen5shot --runner cal_s &

wait  # Wait for all parallel processes to complete
