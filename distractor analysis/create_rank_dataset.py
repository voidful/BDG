import csv
import random
from collections import defaultdict
import jsonlines
import nlp2
import numpy as np
from tqdm import tqdm
from transformers import *
import itertools as it
from nlgeval import NLGEval

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


def tokenize_text(text):
    return tokenizer.convert_tokens_to_string(tokenizer.tokenize(text))
    # return text


n = NLGEval(
    metrics_to_omit=['METEOR', 'EmbeddingAverageCosineSimilairty', 'SkipThoughtCS', 'VectorExtremaCosineSimilarity',
                     'GreedyMatchingScore', 'CIDEr'])


def gen_jsonl(DATASET_FILE, FILE_TEST=None, GOLD_DICT=None, max_dist=3):
    DATASET_FILE.sort()
    all_data = []
    for dataset in DATASET_FILE:
        with open(dataset, 'r', encoding='utf8') as csvfile:
            all_data.extend(list(csv.reader(csvfile, quotechar='"')))

    # race one data format: {'answers': ['a'], 'options': [['a','b']], 'questions': ['q1'], 'article': "", 'id': 'middle2572.txt'}
    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p, i in enumerate(all_data):
        input_content = i[0].strip()
        distractor = i[1].strip()
        if "[ sep ]" in input_content:
            c = input_content.split("[ sep ]")
        else:
            c = input_content.split("[SEP]")  # c - 0,q - 1,a - 2,d1 - 3,d2 - 4
        # c = input_content.split("[SEP]")  # c - 0,q - 1,a - 2,d1 - 3,d2 - 4
        content = tokenize_text(c[0]).strip()
        question = tokenize_text(c[1]).strip()
        answer = tokenize_text(c[2]).strip()
        data_dict[content][question]['options'].append(tokenize_text(distractor).strip().lower())
        data_dict[content][question]['answers'] = answer.strip().lower()

    count = 1
    results = []
    optlist = ["a", "b", "c", "d"]

    for k, v in tqdm(data_dict.items()):
        for q, detail in v.items():
            if GOLD_DICT is not None:
                dis_num = len(GOLD_DICT[k][q]['options'])
            else:
                dis_num = max_dist
            dis_num = max_dist if dis_num > max_dist else dis_num
            for combin in set(it.combinations(detail['options'], dis_num)):
                options = list(combin) + [detail['answers']]
                keep = True
                for i in set(it.combinations(options, 2)):
                    metrics_dict = n.compute_individual_metrics([i[0]], i[1])
                    if metrics_dict['Bleu_1'] > 0.5:
                        keep = False
                        continue
                if keep:
                    random.shuffle(options)
                    ans = optlist[options.index(detail['answers'])]
                    results.append(
                        {'answers': [ans], 'options': [options], 'questions': [q], 'article': k, 'id': str(count)})
            count += 1

    if FILE_TEST is not None:
        print("total num of data", len(data_dict), len(results))
        with jsonlines.open(FILE_TEST, mode='w') as writer:
            writer.write_all(results)

    return data_dict


shuffle_options = True
# gold
GOLD = gen_jsonl(['race_test_updated_cqa_dsep_a.csv'])
max_dist = 3

print("===== rank_best =====")
rank_best = {
    "DATASET_FILE": [df for df in nlp2.get_files_from_dir("./" + 'rank_dist_jsonl/rank_bold') if 'csv' in df],
    "FILE_TEST": 'rank_dist_merge_jsonl/race_test_rank_bold_bleu1_0.5sim.jsonl',
    "GOLD_DICT": GOLD,
    "max_dist": max_dist
}
gen_jsonl(**rank_best)
