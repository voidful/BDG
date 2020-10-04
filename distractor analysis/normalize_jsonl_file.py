from collections import defaultdict
import json
import itertools as it
import jsonlines
import numpy as np
import os


def normalize(target_folder, remove_sim, too_sim_threshold, result_folder):
    data_dict = defaultdict(lambda: defaultdict(dict))
    output_dict = defaultdict(list)
    FILE_TEST = [i for i in os.listdir(target_folder) if ".jsonl" in i]
    # {'answers': ['a'], 'options': [['a','b']], 'questions': ['q1'], 'article': "", 'id': 'middle2572.txt'}
    print(FILE_TEST)
    print("====size====")
    # count total size of each prediction
    for FILE in FILE_TEST:
        with open(os.path.join(target_folder, FILE), 'r', encoding='utf8') as jsonlfile:
            for jlines in jsonlfile.readlines():
                jfile = json.loads(jlines)
                for q in jfile['questions']:
                    dict_id = jfile['article'].strip() + q.strip()
                    dict_id = dict_id.replace(" ", "").lower()
                    data_dict[dict_id][FILE] = jfile

    for _, testfiles in data_dict.items():
        if len(testfiles) == len(FILE_TEST):
            for fname, fcontent in testfiles.items():
                output_dict[fname].append(fcontent)
    print("Total", len(data_dict), len(output_dict))

    print("====similarity====")
    from nlgeval import NLGEval

    n = NLGEval(
        metrics_to_omit=['METEOR', 'EmbeddingAverageCosineSimilairty', 'SkipThoughtCS', 'VectorExtremaCosineSimilarity',
                         'GreedyMatchingScore', 'CIDEr'])
    for task, datas in output_dict.items():
        overall_dict = defaultdict(list)
        toosim_dict = defaultdict(list)
        overall_result = dict()
        toosim_result = dict()
        for v in datas:
            if len(v['options'][0]) <= 4:
                # {'Bleu_1': 0.19999999996000023, 'Bleu_2': 7.071067810274489e-09, 'Bleu_3': 2.5543647739782087e-11, 'Bleu_4': 1.699044244302013e-12, 'METEOR': 0.0547945205479452, 'ROUGE_L': 0.26180257510729615, 'CIDEr': 0.0, 'SkipThoughtCS': 0.41264296, 'EmbeddingAverageCosineSimilairty': 0.804388, 'VectorExtremaCosineSimilarity': 0.650115, 'GreedyMatchingScore': 0.655746}
                example_dict = defaultdict(list)
                if "two" in target_folder:  # answer with one option - check answer copying problem
                    opt = v['options'][0]
                    # [opt[1]] - ground truth/answer
                    metrics_dict = n.compute_individual_metrics([opt[1]], opt[0])
                    for mk, mv in metrics_dict.items():
                        if np.max(mv) > too_sim_threshold:
                            toosim_dict[mk].append(1)
                            if remove_sim:
                                if v in datas:
                                    del output_dict[task][datas.index(v)]
                                break
                        overall_dict[mk].append(mv)
                else:
                    for i in set(it.combinations(v['options'][0], 2)):
                        if len(i[0]) == 0 or len(i[1]) == 0:
                            continue
                        metrics_dict = n.compute_individual_metrics([i[0]], i[1])
                        for mk, mv in metrics_dict.items():
                            example_dict[mk].append(mv)

                    for mk, mv in example_dict.items():
                        if np.max(mv) > too_sim_threshold:
                            toosim_dict[mk].append(1)
                            if remove_sim:
                                if v in datas:
                                    del output_dict[task][datas.index(v)]
                                break
                        overall_dict[mk].append(np.mean(mv))

        for mk, mv in overall_dict.items():
            overall_result[mk] = np.mean(mv)
        for mk, mv in toosim_dict.items():
            toosim_result[mk] = np.sum(mv)
        print(task, overall_result, "\nToo Sim: ", toosim_result)

    data_dict = defaultdict(lambda: defaultdict(dict))
    for FILE, datas in output_dict.items():
        for data in datas:
            for q in data['questions']:
                dict_id = data['article'].strip() + q.strip()
                dict_id = dict_id.replace(" ", "").lower()
                data_dict[dict_id][FILE] = data

    normalized_dict = defaultdict(list)
    print("Total", len(data_dict))
    for _, testfiles in data_dict.items():
        if len(testfiles) == len(FILE_TEST):
            for fname, fcontent in testfiles.items():
                normalized_dict[fname].append(fcontent)

    print("====output====")
    for f, clist in normalized_dict.items():
        print("Normalized", f, len(clist))
        with jsonlines.open(os.path.join(result_folder, f), mode='w') as writer:
            writer.write_all(clist)


print("===== one =====")
# cal answer similarities
one_dist = {
    "target_folder": './one_dist_jsonl',
    "remove_sim": False,
    "too_sim_threshold": 0.95,
    "result_folder": "./one_dist_normalized_filtered_jsonl"

}
normalize(**one_dist)

print("\n===== multi =====")
# cal mean similarities in all options
multi_dist = {
    "target_folder": './multi_dist_jsonl',
    "remove_sim": False,
    "too_sim_threshold": 0.95,
    "result_folder": "./multi_dist_normalized_jsonl"

}
normalize(**multi_dist)
