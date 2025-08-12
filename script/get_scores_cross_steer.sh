cd ../src
export PYTHONPATH=/data/daixl/my_project/word_asso:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="1"

model_names=('Llama' 'Qwen')
langs=('USA' 'UK' 'OC' 'CN')

# for model_name in "${model_names[@]}"; do
#     for lang in "${langs[@]}"; do
#         # 遍历所有非当前 lang 的 cross_steer_lang
#         for cross_steer_lang in "${langs[@]}"; do
#             if [ "$cross_steer_lang" != "$lang" ]; then
#                 echo "Running: model=$model_name, lang=$lang, cross_steer_lang=$cross_steer_lang"
#                 python cal_candidate_steer_model.py \
#                     --lang "$lang" \
#                     --cross_steer_lang "$cross_steer_lang" \
#                     --model_name "$model_name" \
#                     --runner cal_p
#             fi
#         done
#     done
# done


for model_name in "${model_names[@]}"; do
    for lang in "${langs[@]}"; do
        python cal_candidate_steer_model.py --lang "$lang" --cross_steer_lang "wo" --model_name "$model_name" --runner cal_p
    done
done