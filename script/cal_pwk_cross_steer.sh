cd ../src
export PYTHONPATH=/data/daixl/my_project/word_asso:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="6"

# model_names=('Llama' 'Qwen')
#!/bin/bash


python cal_candidate_steer_model.py --lang USA --cross_steer_lang wo --model_name "Llama" --runner cal_s &
python cal_candidate_steer_model.py --lang UK --cross_steer_lang wo --model_name "Llama" --runner cal_s &
python cal_candidate_steer_model.py --lang OC --cross_steer_lang wo --model_name "Llama" --runner cal_s &
python cal_candidate_steer_model.py --lang CN --cross_steer_lang wo --model_name "Llama" --runner cal_s &
python cal_candidate_steer_model.py --lang USA --cross_steer_lang wo --model_name "Qwen" --runner cal_s &
python cal_candidate_steer_model.py --lang UK --cross_steer_lang wo --model_name "Qwen" --runner cal_s &
python cal_candidate_steer_model.py --lang OC --cross_steer_lang wo --model_name "Qwen" --runner cal_s &
python cal_candidate_steer_model.py --lang CN --cross_steer_lang wo --model_name "Qwen" --runner cal_s &
wait
# for model_name in "${model_names[@]}"; do
#     python cal_candidate_steer_model.py --lang USA --cross_steer_lang UK --model_name "$model_name" --runner cal_s &
#     python cal_candidate_steer_model.py --lang USA --cross_steer_lang OC --model_name "$model_name" --runner cal_s &
#     python cal_candidate_steer_model.py --lang USA --cross_steer_lang CN --model_name "$model_name" --runner cal_s &

#     python cal_candidate_steer_model.py --lang UK --cross_steer_lang USA --model_name "$model_name" --runner cal_s &
#     python cal_candidate_steer_model.py --lang UK --cross_steer_lang OC --model_name "$model_name" --runner cal_s &
#     python cal_candidate_steer_model.py --lang UK --cross_steer_lang CN --model_name "$model_name" --runner cal_s &

#     python cal_candidate_steer_model.py --lang OC --cross_steer_lang USA --model_name "$model_name" --runner cal_s &
#     python cal_candidate_steer_model.py --lang OC --cross_steer_lang UK --model_name "$model_name" --runner cal_s &
#     python cal_candidate_steer_model.py --lang OC --cross_steer_lang CN --model_name "$model_name" --runner cal_s &

#     python cal_candidate_steer_model.py --lang CN --cross_steer_lang USA --model_name "$model_name" --runner cal_s &
#     python cal_candidate_steer_model.py --lang CN --cross_steer_lang UK --model_name "$model_name" --runner cal_s &
#     python cal_candidate_steer_model.py --lang CN --cross_steer_lang OC --model_name "$model_name" --runner cal_s &

#     wait  # 等待当前 model_name 的所有任务完成
# done