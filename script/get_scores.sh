cd ../src
export CUDA_VISIBLE_DEVICES="4"

model_names=('Llama' 'Qwen')
# baseline
for model_name in "${model_names[@]}"; do
    python cal_candidate.py --lang USA --model_name "$model_name" --runner cal_p
    python cal_candidate.py --lang CN --model_name "$model_name" --runner cal_p
done

# steer
for model_name in "${model_names[@]}"; do
    python cal_candidate_steer_model.py --lang USA --model_name "$model_name" --runner cal_p
    python cal_candidate_steer_model.py --lang UK --model_name "$model_name" --runner cal_p
    python cal_candidate_steer_model.py --lang OC --model_name "$model_name" --runner cal_p
    python cal_candidate_steer_model.py --lang CN --model_name "$model_name" --runner cal_p
done







