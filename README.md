# BDG(Distractor Generation)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fvoidful%2FBDG.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fvoidful%2FBDG?ref=badge_shield)

Code for "A BERT-based Distractor Generation Scheme with Multi-tasking and Negative Answer Training Strategies."  
[Paper](https://www.aclweb.org/anthology/2020.findings-emnlp.393/)

## V2
Updated result using BART. BART model is uploaded in HuggingFace model hub.
| model         | BLEU1 | BLEU2 | BLEU3 | BLEU4 | ROUGEL |
|---------------|-------|-------|-------|-------|--------|
| BERT DG       | 35.30 | 20.65 | 13.66 | 9.53  | 31.11  |
| BERT DG pm    | 39.81 | 24.81 | 17.66 | 13.56 | 34.01  |
| BERT DG an+pm | 39.52 | 24.29 | 17.28 | 13.28 | 33.40  |
| BART DG       | 40.76 | 26.40 | 19.14 | 14.65 | 35.53  |
| BART DG pm    | 41.85 | 27.45 | 20.47 | 16.33 | 37.15  |
| BART DG an+pm | 40.26 | 25.86 | 18.85 | 14.65 | 35.64  |
* higher is better

| model         | Count BLEU1 > 0.95 |
|---------------|--------------------|
| BERT DG       | 115                |
| BERT DG pm    | 57                 |
| BERT DG an+pm | 43                 |
| BART DG       | 110                |
| BART DG pm    | 60                 |
| BART DG an+pm | 23                 |
| Gold          | 12                 |
* lower is better

## Trained Model and Code Example
### BART
Distractor: https://huggingface.co/voidful/bart-distractor-generation  
Distractor PM: https://huggingface.co/voidful/bart-distractor-generation-pm  
Distractor AN+PM: https://huggingface.co/voidful/bart-distractor-generation-both  

### BERT 
Trained model available on release:  
https://github.com/voidful/BDG/releases/tag/v1.0

Colab notebook for using pre trained model:  
https://colab.research.google.com/drive/1yA3Rex9JHKJmc52E3YdsBQ4eQ_R6kEZB?usp=sharing

## Citation

If you make use of the code in this repository, please cite the following papers:

    @inproceedings{chung-etal-2020-BERT,
    title = "A {BERT}-based Distractor Generation Scheme with Multi-tasking and Negative Answer Training Strategies.",
    author = "Chung, Ho-Lam  and
      Chan, Ying-Hong  and
      Fan, Yao-Chung",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.393",
    pages = "4390--4400",
    abstract = "In this paper, we investigate the following two limitations for the existing distractor generation (DG) methods. First, the quality of the existing DG methods are still far from practical use. There are still room for DG quality improvement. Second, the existing DG designs are mainly for single distractor generation. However, for practical MCQ preparation, multiple distractors are desired. Aiming at these goals, in this paper, we present a new distractor generation scheme with multi-tasking and negative answer training strategies for effectively generating \textit{multiple} distractors. The experimental results show that (1) our model advances the state-of-the-art result from 28.65 to 39.81 (BLEU 1 score) and (2) the generated multiple distractors are diverse and shows strong distracting power for multiple choice question.",
    }


## Environment Setup
```bash
pip install -r requirement.txt
```

## Data Preprocessing   
Inside `data_preprocessing` folder.  
Download dataset [here](https://github.com/Yifan-Gao/Distractor-Generation-RACE), put it into `distractor` folder.    
run `convert_data.py` to do preprocessing.  
run `dataset_stat.py` for dataset statistics.  

## Train Distractor Generator
### BART
using tfkit==0.7.0 and transformers==4.4.2  
```bash
tfkit-train --savedir ./race_cqa_gen_d_bart/ --train ./race_train_updated_cqa_dsep_a_bart.csv --test ./race_test_updated_cqa_dsep_a_bart.csv --model seq2seq  --config facebook/bart-base --batch 9 --epoch 10 --grad_accum 2 --no_eval;
tfkit-train --savedir ./race_cqa_gen_d_bart_pm/ --train ./race_train_updated_cqa_dsep_a_bart.csv --test ./race_test_updated_cqa_dsep_a_bart.csv --model seq2seq  --config facebook/bart-base --batch 9 --epoch 10 --grad_accum 2 --no_eval --likelihood pos;
tfkit-train --savedir ./race_cqa_gen_d_bart_both/ --train ./race_train_updated_cqa_dsep_a_bart.csv --test ./race_test_updated_cqa_dsep_a_bart.csv --model seq2seq  --config facebook/bart-base --batch 9 --epoch 10 --grad_accum 2 --no_eval --likelihood both;
```

### BERT
using environment from `requirement.txt`   
run the following in main dir:  
### Train BDG Model
```bash
tfkit-train --maxlen 512 --savedir ./race_cqa_gen_d/ --train ./data_preprocessing/processed_data/race_train_updated_cqa_dsep_a.csv --test ./data_preprocessing/processed_data/race_test_updated_cqa_dsep_a.csv --model onebyone --tensorboard  --config bert-base-cased --batch 30 --epoch 6;
```
### Train BDG AN model
```bash
tfkit-train --maxlen 512 --savedir ./race_cqa_gen_d_an/ --train ./data_preprocessing/processed_data/race_train_updated_cqa_dsep_a.csv --test ./data_preprocessing/processed_data/race_test_updated_cqa_dsep_a.csv --model onebyone-neg --tensorboard  --config bert-base-cased --batch 30 --epoch 6;
```
### Train BDG PM model
```bash
tfkit-train --maxlen 512 --savedir ./race_cqa_gen_d_pm/ --train ./data_preprocessing/processed_data/race_train_updated_cqa_dsep_a.csv --test ./data_preprocessing/processed_data/race_test_updated_cqa_dsep_a.csv --model onebyone-pos --tensorboard  --config bert-base-cased --batch 30 --epoch 6;
```
### Train BDG AN+PM model
```bash
tfkit-train --maxlen 512 --savedir ./race_cqa_gen_d_both/ --train ./data_preprocessing/processed_data/race_train_updated_cqa_dsep_a.csv --test ./data_preprocessing/processed_data/race_test_updated_cqa_dsep_a.csv --model onebyone-both --tensorboard  --config bert-base-cased --batch 30 --epoch 6;
```
### Eval generator   
```bash
tfkit-eval --model model_path --valid ./data_preprocessing/processed_data/race_test_updated_cqa_dall.csv --metric nlg
```

## Distractor Analysis
Inside `distractor analysis` folder
-  `preprocess_model_result.py` for result preprocessing and statistics.
-  `normalize_jsonl_file.py` merge different model result with same question and context.
-  `create_rank_dataset.py` prepare data for Entropy Maximization.

## RACE MRC
### Preparation
```bash
git clone https://github.com/huggingface/transformers
cp our transformer file into huggingface/transformers
```

### Training Multiple Choice Question Answering Model
Based on the script [`run_multiple_choice.py`]().
Download race data
Train   
```bash
#training on 4 tesla V100(16GB) GPUS
export RACE_DIR=../RACE
python ./examples/run_multiple_choice.py \
--model_type roberta \
--task_name race \
--model_name_or_path roberta-base-openai-detector  \
--do_train  \
--do_eval \
--data_dir $RACE_DIR \
--learning_rate 1e-5 \
--num_train_epochs 10 \
--max_seq_length 512 \
--output_dir ./roberta-base-openai-race \
--per_gpu_eval_batch_size=9 \
--per_gpu_train_batch_size=9 \
--gradient_accumulation_steps 2 \
--save_steps 5000 \
--eval_all_checkpoints \
--seed 77 
```

### Eval QA & Get entropy ensemble result
```bash
export RACE_DIR=../multi_dist_normalized_jsonl/xxx.jsonl
python ./examples/run_multiple_choice.py \
--model_type roberta \
--task_name race \
--model_name_or_path ../roberta-base-openai-race/  \
--do_test \
--data_dir $RACE_DIR \
--max_seq_length 512 \
--per_gpu_eval_batch_size=3 \
--output_dir ./race_test_result \
--overwrite_cache
```




## License
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fvoidful%2FBDG.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fvoidful%2FBDG?ref=badge_large)