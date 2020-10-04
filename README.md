# BDG(BERT-based Distractor Generation)
Code for "A BERT-based Distractor Generation Scheme with Multi-tasking and Negative Answer Training Strategies."  

## Environment Setup
```bash
pip install -r requirement.txt
```

## Data Preprocessing   
Inside `data_preprocessing` folder.  
Download dataset [here](https://github.com/Yifan-Gao/Distractor-Generation-RACE), put it into `distractor` folder.    
run `convert_data.py` to do preprocessing.  
run `dataset_stat.py` for dataset statistics.  

## Train BERT-based Distractor Generator
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


