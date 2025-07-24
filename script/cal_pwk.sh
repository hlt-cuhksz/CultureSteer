cd ../src
export CUDA_VISIBLE_DEVICES="4"

# baseline
model_names=('Llama' 'Qwen')
for model_name in "${model_names[@]}"; do
    python cal_candidate.py --lang USA --model_name "$model_name" --runner cal_s &
    python cal_candidate.py --lang UK --model_name "$model_name" --runner cal_s &
    python cal_candidate.py --lang OC --model_name "$model_name" --runner cal_s &
    python cal_candidate.py --lang CN --model_name "$model_name" --runner cal_s &
    wait
done

for model_name in "${model_names[@]}"; do
    python word_asso_task.py --lang USA --model_name "$model_name" --runner cal_s &
    python word_asso_task.py --lang UK --model_name "$model_name" --runner cal_s & 
    python word_asso_task.py --lang OC --model_name "$model_name" --runner cal_s &
    python word_asso_task.py --lang CN --model_name "$model_name" --runner cal_s &
    wait
done
