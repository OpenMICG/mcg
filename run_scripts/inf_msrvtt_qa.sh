cd ..

export PYTHONPATH="$PYTHONPATH:$PWD"
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo $PYTHONPATH

STEP='best_step'

CONFIG_PATH='config/msrvtt.json'

TXT_DB='/path/to/your/data/msrvtt/txt/test.jsonl'
IMG_DB='/path/to/your/data/msrvtt/train-video/'

horovodrun -np 1 python src/tasks/run_video_qa.py \
      --do_inference 1 \
      --inference_split test \
      --inference_model_step $STEP \
      --inference_txt_db $TXT_DB \
      --inference_img_db $IMG_DB \
      --inference_batch_size 128 \
      --output_dir /path/to/your/data/mcg/finetune/msrvtt_qa/2023xxxxx/ \
      --config $CONFIG_PATH