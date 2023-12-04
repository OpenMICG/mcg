# MCG

Implementation of the MCG model

**Multi-granularity Contrastive Cross-modal Collaborative Generation for End-to-End Long-term Video Question Answering**

## Code structure

We present the official PyTorch code for MCG, with the complete code directory structured as follows:

```bash
./
├── config/	# Configuration files
│   ├── pretrain_mcg.json
│   ├── msvd_qa.json
│   ├── ...
│   └── timesformer_divst_8x32_224_k600.json
├── env/		# Environment requirements and setup scripts	
│   ├── install_pkg.sh
│   └── requirements.txt
├── src/		# MCG source code
│   ├── configs/
│   ├── datasets/
│   ├── __init__.py
│   ├── modeling/
│   ├── optimization/
│   ├── pretrain/
│   ├── tasks/
│   └── utils/
├── run_scripts/	# Pre-training and fine-tuning scripts
│   ├── pt_mcg.sh
│   ├── ...
│   └── ft_msvd_qa.sh
└── README.md
```

> **Note**: We haven't explicitly labeled the dataset directory in the structure. We encourage to keep dataset separate from the code and store it in a  specified data disk. Once you have downloaded the dataset, you can configure it in the configuration file in the `'/config'` directory.

## Setup & Data Preparation

### Install Dependencies

1. Creating conda environment

   ```bash
   conda create -n mcg python=3.8
   conda activate mcg
   ```

2. Run setup scripts

   ```bash
   cd env && bash install_pkg.sh
   ```

   > **Note**: We utilize Horovod as our distributed deep learning training framework.  Initial installation may pose some challenges;  please refer to the official [Horovod](https://github.com/horovod/horovod) GitHub repository for guidance.

### Pre-training Data Preparation

- **WebVid2M**
  - Download [WebVid2M](https://github.com/m-bain/frozen-in-time).
  - Put WebVid2M videos under your data path.
  - Change your `config/pretrain_mcg.json`
- **CC3M**
  - Download [CC-3M](https://github.com/igorbrigadir/DownloadConceptualCaptions).
  - Change `cc3m.json` with local image paths.
  - Change your `config/pretrain_mcg.json`
- **TGIF**
  - Download [TGIF](https://github.com/raingo/TGIF-Release).
  - Put TGIF dataset under your data path.
  - Change your `config/pretrain_mcg.json`

### Fine-tuning Data Preparation

- **MSRVTT-QA**

  - Download train_val_videos.zip and test_videos.zip from [here](https://www.mediafire.com/folder/h14iarbs62e7p/shared).

  - Check md5sum:

    ```bash
    51f2394d279cf84f1642defd9a651e6f  train_val_videos.zip
    0af68454cec9d586e92805739f3911d0  test_videos.zip
    ```

  - Unzip all the videos to your data path. (10k in total).

    ```bash
    unzip train_val_videos.zip -d /path/to/your/data/msrvtt/videos
    unzip test_videos.zip -d /path/to/your/data/msrvtt/videos
    ```

  - Download QA annotations from [here]()

- **MSVD-QA**

  - Download Video from official release:

    ```
    wget -nc https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar
    ```

  - Check md5sum:

    ```
    9bdb20fcf14d59524a6febca9f6a8d89  YouTubeClips.tar
    ```

  - Unzip all the videos to your data path. (1,970 videos in total).

    ```
    tar xvf YouTubeClips.tar -C /path/to/your/data/msvd/videos --strip-components=1
    ```

  - Download QA annotations from [here]()

- **NExT-QA**

  - Download the raw videos from [NExTVideo](https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view?usp=share_link). 

  - Download the QA annotations from[here](https://drive.google.com/drive/folders/14jSt4sGFQaZxBu4AGL2Svj34fUhcK2u0?usp=sharing) and map video file from [here](https://drive.google.com/drive/folders/1gKRR2es8-gRTyP25CvrrVtV6aN5UxttF?usp=sharing).

    - `['nextqa.zip']` contains annotations of QAs and GloVe Embeddings. 
    - As NExT-QA's videos are sourced from VidOR,  you may need the map file `['nextqa/map_vid_vidorID.json']`).


All the text annotation can be downloaded from this [Link](https://drive.google.com/file/d/1uSlZqe6Zf0YFKHHnkx_KG2RfhMGcm0b0/view?usp=sharing), with the complete annotions directory structured as follows:

```bash
./
├── fintune_data
│   ├── msrvttqa
│   │   ├── test.jsonl
│   │   ├── train_ans2label.json
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   ├── msvdqa
│   │   ├── test.jsonl
│   │   ├── train_ans2label.json
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   └── nextqa
│       ├── add_reference_answer_test.json
│       ├── glove_embed.npy
│       ├── map_vid_vidorID.json
│       ├── multi_vocab.pkl
│       ├── test.csv
│       ├── train.csv
│       ├── val.csv
│       └── vocab.pkl
└── pretrin_data
    ├── cc3m
    │   └── cc3m.json
    └── webvid2m
        ├── train.pkl
        └── val.pkl
```

## Pretraining

1. Download [WebVid2M](https://github.com/m-bain/frozen-in-time),  [CC-3M](https://github.com/igorbrigadir/DownloadConceptualCaptions) and [TGIF](https://github.com/raingo/TGIF-Release).

2. Configure your pretraining configuration file  `config/pretrain_mcg.json` with your dataset path and other hyper-parametes.

3. Modify your pre-training run scripts `run_scripts/pretrain_mcg.sh`

   ```bash
   cd ..
   
   export PYTHONPATH="$PYTHONPATH:$PWD"
   echo $PYTHONPATH
   
   CONFIG_PATH='config/pretrain_mcg.json'
   
   horovodrun -np 8 python src/pretrain/run_pretrain_sparse.py \		# change -np to GPUs numbers.
         --config $CONFIG_PATH \
         --output_dir /path/to/output_dir/pretrain/$(date '+%Y%m%d%H%M%S')
   ```

4. Training video-language model

   ```bash
   cd run_scripts && bash pretrain_mcg.sh
   ```

## Downstream Task Finetuning

Once you have completed the model pre-training, you can use the downstream datasets to fine-tune your weights . 



- We provide the fine-tuning and inference code for MSRVTT-QA, MSVD-QA, NExT-QA, you can run the following script for model training:

  ```bash
  cd run_scripts
  bash ft_msrvtt_qa.sh
  bash ft_msvd_qa.sh
  bash ft_next_qa.sh
  ```

  For example, with MSVD-QA:

  ```bash
  cd .. 
  
  export PYTHONPATH="$PYTHONPATH:$PWD"
  echo $PYTHONPATH
  
  CONFIG_PATH='config/msvd_qa.json'
  
  horovodrun -np 8 python src/tasks/run_video_qa.py \
        --debug 0 \
        --config $CONFIG_PATH \
        --output_dir /path/to/output_dir/finetune/msvd_qa/$(date '+%Y%m%d%H%M%S')
  
  ```

- Run inference with locally-finetuned checkpoints.

  ```
  cd run_scripts
  bash inf_msrvtt_qa.sh
  bash inf_msvd_qa.sh
  bash inf_next_qa.sh
  ```

  For example, with MSVD-QA:

  ```bash
  cd ..
  
  export PYTHONPATH="$PYTHONPATH:$PWD"
  echo $PYTHONPATH
  
  STEP='the_best_step'
  
  CONFIG_PATH='config/msvd_qa.json'
  OUTPUT_DIR='/path/to/output_dir/finetune/msvd_qa/the_finetuning_path/'
  
  TXT_DB='/path/to/dataset/msvd/txt/test.jsonl'
  IMG_DB='/path/to/dataset/msvd/train_video/'
  
  horovodrun -np 8 python src/tasks/run_video_qa.py \
        --do_inference 1 \
        --inference_split test \
        --inference_model_step $STEP \
        --inference_txt_db $TXT_DB \
        --inference_img_db $IMG_DB \
        --inference_batch_size 128 \
        --output_dir OUTPUT_DIR \
        --config $CONFIG_PATH
  ```

  - `OUTPUT_DIR` is the path after the `--output_dir` option in the finetuning script.
  - `$STEP` is a string, which tells the script to use the checkpoint `$OUTPUT_DIR/ckpt/model_step_$STEP.pt` for inference.

## Acknowledgement

The implementation of MCG relies on [ALPRO](https://github.com/salesforce/ALPRO),  [ClipBERT](https://github.com/jayleicn/ClipBERT), [TimesFormer](https://github.com/facebookresearch/TimeSformer/tree/main/timesformer/models), [Transformer](https://github.com/huggingface/transformers). We use [PyTorch](https://github.com/pytorch/pytorch) as our deep learning framework, with [Horovod](https://github.com/horovod/horovod) and [gradient-checkpoint](https://github.com/csrhddlam/pytorch-checkpoint) as our distributed multi-GPU training frameworks. We thank the original authors for their work and open source code.

