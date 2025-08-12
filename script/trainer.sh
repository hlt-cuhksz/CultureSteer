cd ../src
export PYTHONPATH=/data/daixl/my_project/word_asso:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="6"
model_names=('Llama' 'Qwen')
for model_name in "${model_names[@]}"; do
    python steer_trainer.py --model_name "$model_name" --batch_size 64
done

