import csv
import nlp2
from statistics import mean

inputFiles = [f for f in nlp2.get_files_from_dir('./processed_data') if 'csv' in f]

for inputFile in inputFiles:
    article_length = []
    question_length = []
    answer_length = []
    distractor_length = []
    distractor_num = []
    with open(inputFile, encoding="utf-8", errors='replace') as dataset_file:
        rows = csv.reader(dataset_file)
        for r in rows:
            article, question, answer = r[0].split("[SEP]")
            distractors = r[1].split("[SEP]")

            article = nlp2.split_sentence_to_array(article, True)
            question = nlp2.split_sentence_to_array(question, True)
            answer = nlp2.split_sentence_to_array(answer, True)

            article_length.append(len(article))
            question_length.append(len(question))
            answer_length.append(len(answer))
            distractor_num.append(len(distractors))
            for dist in distractors:
                dist = nlp2.split_sentence_to_array(dist, True)
                distractor_length.append(len(dist))

    print(f"====={inputFile}======")
    print("number of data", len(question_length))
    print("average article_length", mean(article_length))
    print("average question_length", mean(question_length))
    print("average distractor_length", mean(distractor_length))
    print("average distractor_num", mean(distractor_num))
    print("average ans_length", mean(answer_length))
