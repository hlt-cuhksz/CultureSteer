cd ../src
export CUDA_VISIBLE_DEVICES="6"
model_names=('Llama' 'Qwen')
for model_name in "${model_names[@]}"; do
    python train_or.py --model_name "$model_name"
done

