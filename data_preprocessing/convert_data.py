import csv
from collections import defaultdict
from statistics import mean

import nlp2
from transformers import *
import jsonlines

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


input_folder = "./distractor"
output_folder = "./processed_data/"

inputFiles = [f for f in nlp2.get_files_from_dir(input_folder) if 'json' in f]

for inputFile in inputFiles:
    questions = defaultdict(dict)
    outfile_type = inputFile.split("/")[-1].replace(".json", "")
    print(inputFile, outfile_type)
    with jsonlines.open(inputFile, mode='r') as reader:
        count = 0
        for obj in reader:
            count += 1
            article = " ".join(obj['article']).strip()
            question = " ".join(obj['question']).strip()
            distractor = " ".join(obj['distractor']).strip()
            answer_text = " ".join(obj['answer_text']).strip()
            id = (article + question).replace(" ", '')
            questions[id]['context'] = article
            questions[id]['question'] = question
            questions[id]['answer'] = answer_text
            if 'distractor' in questions[id]:
                questions[id]['distractor'].append(distractor)
            else:
                questions[id]['distractor'] = [distractor]
        print(len(questions), count)

        count_over_512 = 0
        data_list = []
        data_list_wo_a = []
        data_sep_list = []
        data_sep_list_wo_a = []
        d_count = []
        for _, item in questions.items():
            c = item['context']
            q = item['question']
            a = item['answer']
            d = item['distractor']
            d_count.append(len(d))
            t_c = tokenizer.tokenize(c)
            t = tokenizer.tokenize(c + " [SEP] " + q + " [SEP] " + a + " [SEP] " + " [SEP] ".join(d))
            if len(t) > 512:
                t_oth = tokenizer.tokenize(" [SEP] " + q + " [SEP] " + a + " [SEP] " + " [SEP] ".join(d))
                remain = 512 - len(t_oth)
                t_c = t_c[:remain]
                if len(t_c + t_oth) > 512:
                    print(len(t))
                    count_over_512 += 1
                    continue
            t_in = t_c + tokenizer.tokenize(" [SEP] " + q + " [SEP] " + a)
            t_out = tokenizer.tokenize(" [SEP] ".join(d))
            for oned in d:
                data_sep_list.append(
                    [tokenizer.convert_tokens_to_string(t_in), tokenizer.convert_tokens_to_string([oned]), a])
                data_sep_list_wo_a.append(
                    [tokenizer.convert_tokens_to_string(t_in), tokenizer.convert_tokens_to_string([oned])])
            # cqa_dall_a
            data_list.append([tokenizer.convert_tokens_to_string(t_in), tokenizer.convert_tokens_to_string(t_out), a])
            data_list_wo_a.append([tokenizer.convert_tokens_to_string(t_in), tokenizer.convert_tokens_to_string(t_out)])

        print(count_over_512, mean(d_count))

        with open(output_folder + outfile_type + "_cqa_dall_a.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data_list)

        with open(output_folder + outfile_type + "_cqa_dall.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data_list_wo_a)

        with open(output_folder + outfile_type + "_cqa_dsep_a.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data_sep_list)

        with open(output_folder + outfile_type + "_cqa_dsep.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data_sep_list_wo_a)
