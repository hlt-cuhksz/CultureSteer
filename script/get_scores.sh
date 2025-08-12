cd ../src
export PYTHONPATH=/data/daixl/my_project/word_asso:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="4"
# model_names=('Llama' 'Qwen' 'CultureLLMLlama' 'CultureLLMQwen' 'SimLLMLlama' 'SimLLMQwen' 'CultureMergeLlama' 'CultureSPALlama' 'Llamacsp' 'Llamacct' 'Qwencsp' 'Qwencct' 'Llama3shot' 'Llama5shot' 'Qwen3shot' 'Qwen5shot')
# model_names=('CultureLLMQwen')
# baseline
model_names=('Llama1shot' 'Llama3shot' 'Llama5shot' 'Qwen1shot' 'Qwen3shot' 'Qwen5shot')
for model_name in "${model_names[@]}"; do
    python cal_candidate.py --lang USA --model_name "$model_name" --runner cal_p
    python cal_candidate.py --lang CN --model_name "$model_name" --runner cal_p
done

# model_names=('Llama' 'Qwen')
# # # steer
# for model_name in "${model_names[@]}"; do
#     python cal_candidate_steer_model.py --lang USA --model_name "$model_name" --runner cal_p
#     python cal_candidate_steer_model.py --lang UK --model_name "$model_name" --runner cal_p
#     python cal_candidate_steer_model.py --lang OC --model_name "$model_name" --runner cal_p
#     python cal_candidate_steer_model.py --lang CN --model_name "$model_name" --runner cal_p
# done







