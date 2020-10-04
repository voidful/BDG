import csv
import os
import random
from collections import defaultdict
import jsonlines
import nlp2
from tqdm import tqdm
from transformers import *

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def tokenize_text(text):
    return tokenizer.convert_tokens_to_string(tokenizer.tokenize(text))
    # return text


def gen_jsonl(DATASET_FILE, FILE_TEST, GOLD_DICT=None, shuffle=False, max_dist_num=3):
    all_data = []
    for dataset in DATASET_FILE:
        with open(dataset, 'r', encoding='utf8') as csvfile:
            all_data.extend(list(csv.reader(csvfile, quotechar='"')))

    # race one data format: {'answers': ['a'], 'options': [['a','b']], 'questions': ['q1'], 'article': "", 'id': 'middle2572.txt'}
    if 'random' in FILE_TEST:
        all_distractor = []
        for i in all_data:
            all_distractor.append(i[1])
        random.shuffle(all_distractor)

    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p, i in enumerate(all_data):
        input_content = i[0].strip()
        distractor = i[1].strip()
        if "[ sep ]" in input_content:
            c = input_content.split("[ sep ]")
        else:
            c = input_content.split("[SEP]")  # c - 0,q - 1,a - 2,d1 - 3,d2 - 4
        content = tokenize_text(c[0]).strip()
        question = tokenize_text(c[1]).strip()
        answer = tokenize_text(c[2]).strip()
        if 'random' in FILE_TEST:
            for ran_samp in random.sample(all_distractor, 3):
                data_dict[content][question]['options'].append(tokenize_text(ran_samp).strip())
        else:
            data_dict[content][question]['options'].append(tokenize_text(distractor).strip())
        data_dict[content][question]['answers'] = answer

    count = 1
    len_option = []
    results = []
    optlist = ["a", "b", "c", "d"]

    for k, v in tqdm(data_dict.items()):
        for q, detail in v.items():
            len_option.append(len(detail['options']))
            if GOLD_DICT is not None:
                dis_num = len(GOLD_DICT[k][q]['options'])
                if dis_num == 0:
                    dis_num = max_dist_num
            else:
                dis_num = max_dist_num
            options = detail['options'][:max_dist_num if dis_num > max_dist_num else dis_num] + [detail['answers']]
            # do not shuffle when cal_token_score.py
            if shuffle:
                random.shuffle(options)
            ans = optlist[options.index(detail['answers'])]
            results.append({'answers': [ans], 'options': [options], 'questions': [q], 'article': k, 'id': str(count)})
            count += 1

    print("total num of data", len(data_dict), len(results))
    with jsonlines.open(FILE_TEST, mode='w') as writer:
        writer.write_all(results)

    return data_dict


def write_result(shuffle, input_folder, output_folder, dist_num):
    DATASET_FILE = ['race_test_updated_cqa_dsep_a.csv']
    # gold
    FILE_TEST = output_folder + '/race_test_gold.jsonl'
    print(FILE_TEST)
    Gold = gen_jsonl(DATASET_FILE, FILE_TEST, max_dist_num=dist_num)
    # random
    FILE_TEST = output_folder + '/race_test_random.jsonl'
    print(FILE_TEST)
    gen_jsonl(DATASET_FILE, FILE_TEST, max_dist_num=dist_num)
    # random
    FILE_TEST = output_folder + '/race_test_random.jsonl'
    print(FILE_TEST)
    gen_jsonl(DATASET_FILE, FILE_TEST, max_dist_num=dist_num)
    # predicted result
    input_folder = os.path.join(ROOT_DIR, input_folder)
    print('input_folder', input_folder)
    for f in nlp2.get_folders_form_dir(input_folder):
        dir = f.split("/")[-1]
        DATASET_FILE = [df for df in nlp2.get_files_from_dir(f) if 'csv' in df]
        DATASET_FILE.sort()
        FILE_TEST = output_folder + '/race_test_' + dir + '.jsonl'
        print(FILE_TEST)
        gen_jsonl(DATASET_FILE, FILE_TEST, Gold, shuffle=shuffle, max_dist_num=dist_num)


one_dist_config = {
    "shuffle": False,
    "input_folder": "./one_dist_model_result",
    "output_folder": './one_dist_jsonl',
    "dist_num": 2
}
write_result(**one_dist_config)

multi_dist_config = {
    "shuffle": True,
    "input_folder": "./multi_dist_model_result",
    "output_folder": './multi_dist_jsonl',
    "dist_num": 3
}
write_result(**multi_dist_config)


