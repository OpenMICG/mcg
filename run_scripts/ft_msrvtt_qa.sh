cd .. 

export PYTHONPATH="$PYTHONPATH:$PWD"
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo $PYTHONPATH

CONFIG_PATH='config/msrvtt_qa.json'

horovodrun -np 1 python src/tasks/run_video_qa.py \
      --debug 0 \
      --config $CONFIG_PATH \
      --output_dir /path/to/your/data/mcg/finetune/msrvtt_qa/$(date '+%Y%m%d%H%M%S')
