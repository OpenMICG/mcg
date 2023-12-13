cd ..

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

CONFIG_PATH='config/pretrain_mcg.json'

horovodrun -np 4 python src/pretrain/run_pretrain_sparse.py \
      --debug 0\
      --config $CONFIG_PATH \
      --output_dir /path/to/your/data/mcg/pretrain/$(date '+%Y%m%d%H%M%S')