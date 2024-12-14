from dotenv import load_dotenv
load_dotenv()  # Openai token

import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
import evaluate
import spacy

from pydantic import BaseModel
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from deep_translator import GoogleTranslator
from tqdm import tqdm


bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")


def calc_classic_metrcis(data: pd.DataFrame):
    data_with_metrics = data.copy()

    answers = data_with_metrics["answers"]
    gold_answers = data_with_metrics["gold_answers"]

    print("calculating bert scores...")
    bert_scores = bertscore.compute(predictions=answers, references=gold_answers, lang="ru")
    data_with_metrics["bert_precision"] = bert_scores["precision"]
    data_with_metrics["bert_recall"] = bert_scores["recall"]
    data_with_metrics["bert_f1"] = bert_scores["f1"]

    print("calculating bleu scores...")
    bleu_scores = []
    for i in range(len(data_with_metrics)):
        item_bleu = bleu.compute(predictions=[answers[i]], references=[gold_answers[i]])["bleu"]
        bleu_scores.append(item_bleu)
    data_with_metrics["bleu"] = bleu_scores

    print("calculating rouge scores...")
    metric_names = ["precision", "recall", "fmeasure"]
    rouge_scores = {
        "rouge1": {metric_name: [] for metric_name in metric_names},
        "rouge2": {metric_name: [] for metric_name in metric_names},
        "rougeL": {metric_name: [] for metric_name in metric_names}
    }
    rouge = rouge_scorer.RougeScorer(
        list(rouge_scores.keys()),
        use_stemmer=True
    )

    for i in range(len(data)):
        for key, value in rouge.score(target=answers[i], prediction=gold_answers[i]).items():
            for metric_name in metric_names:
                rouge_scores[key][metric_name].append(getattr(value, metric_name))

    for key in rouge_scores:
        data_with_metrics[key] = rouge_scores[key]
        for metric_name in metric_names:
            data_with_metrics[key + "_" + metric_name] = rouge_scores[key][metric_name]

    return data_with_metrics


judge_prompt = """
Вы будете оценивать ответ финансового ассистента на вопрос пользователя, основываясь на следующих критериях. Для каждого критерия укажите свою оценку от 0 до 5 и предоставьте обоснование вашей оценки.

### Входные данные:
1. Вопрос пользователя: {question}
2. Ответ ассистента: {answer}
3. Верная информация: {gold_answer}

### Критерии оценки:

1. Правдивость:
   - Оцените, насколько ответ ассистента соответствует действительности и не содержит вымышленных или неточных данных. Например, если ассистент указывает сумму расходов, она должна быть в пределах реального контекста. Используйте сравнение с верной информацией, переданной во входных данных.

2. Помощь, полезность:
   - Оцените, насколько ответ ассистента полезен для пользователя и может помочь в решении его задачи. Например, все советы должны быть практичными и легко реализуемыми.

3. Калибровка:
   - Оцените, как ассистент демонстрирует неуверенность в тех случаях, когда правильный ответ неизвестен. Если ассистент содержит рекомендации по дополнительным ресурсам или действиям в условиях неопределенности, это положительно скажется на оценке.

4. Релевантность:
   - Оцените, насколько ответ соответствует заданному вопросу и его контексту. Все советы должны быть релевантными и отражать фактическую ситуацию пользователя.

5. Полнота:
    - Оцените, насколько всесторонне ответ охватывает вопрос. Если вопрос не подразумевает развернутого ответа, а ответ является лаконичным, оценка полноты должна быть высокой
"""


class CriterionSchema(BaseModel):
    mark: Annotated[int, "Оценка ответа судьей - единственное число"]
    reasoning: Annotated[str, "Обоснование, по которому выставлена именно такая оценка"]


class JudgeSchema(BaseModel):
    accuracy: Annotated[CriterionSchema, "Точная оценка правдивости"]
    usefulness: Annotated[CriterionSchema, "Точная оценка полезности"]
    calibration: Annotated[CriterionSchema, "Точная оценка калибровки"]
    relevance: Annotated[CriterionSchema, "Точная оценка релевантности"]
    recall: Annotated[CriterionSchema, "Точная оценка полноты"]


llm = ChatPromptTemplate.from_template(judge_prompt) | ChatOpenAI(model="gpt-4o-mini").with_structured_output(
    JudgeSchema)


def calc_judge_score(data: pd.DataFrame):
    judge_scores = {"judge_" + criterion: [] for criterion in JudgeSchema.__annotations__.keys()}

    print("calculating judge scores...")
    for i in tqdm(range(len(data))):
        judge_score = llm.invoke({
            "question": data["questions"][i],
            "answer": data["answers"][i],
            "gold_answer": data["gold_answers"][i]
        })

        print(judge_score)

        for criterion in JudgeSchema.__annotations__.keys():
            judge_scores["judge_" + criterion].append(getattr(getattr(judge_score, criterion), "mark"))  # TODO

    data_with_scores = data.copy()
    for criterion, values in judge_scores.items():
        data_with_scores[criterion] = values
    return data_with_scores


ner = spacy.load("en_core_web_lg")  # или "en_core_web_lg" или "en_core_web_trf"


def ner_accuracy(gold_answer: str, answer: str):
    translator = GoogleTranslator(source="ru", target="en")

    gold_answer = translator.translate(gold_answer)
    answer = translator.translate(answer)

    target_ents = ["DATE", "EVENT", "MONEY", "TIME"]

    gold_answer_by_ents = {ent_name: [] for ent_name in target_ents}
    answer_by_ents = {ent_name: [] for ent_name in target_ents}

    for ent in ner(gold_answer).ents:
        if ent.label_ in target_ents:
            gold_answer_by_ents[ent.label_].append(ent.text)

    for ent in ner(answer).ents:
        if ent.label_ in target_ents:
            answer_by_ents[ent.label_].append(ent.text)

    rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)

    gold_answer_by_ents = {key: " ".join(list(set(value))) for key, value in gold_answer_by_ents.items()}
    answer_by_ents = {key: " ".join(list(set(value))) for key, value in answer_by_ents.items()}

    precision = 0
    recall = 0
    f1 = 0
    for ent in target_ents:
        res = rouge.score(target=gold_answer_by_ents[ent], prediction=answer_by_ents[ent])["rouge1"]

        precision += res.precision if not np.isnan(res.precision) else 0
        recall += res.recall if not np.isnan(res.recall) else 0
        f1 += res.fmeasure if not np.isnan(res.fmeasure) else 0

    return {
        "precision": precision / len(target_ents),
        "recall": recall / len(target_ents),
        "f1": f1 / len(target_ents)
    }


def calc_ner_accuracy(data: pd.DataFrame):
    ner_precision = []
    ner_recall = []
    ner_f1 = []

    print("calculating NER accuracy scores...")
    for i in tqdm(range(len(data))):
        res = ner_accuracy(data["gold_answers"][i], data["answers"][i])

        ner_precision.append(res["precision"])
        ner_recall.append(res["recall"])
        ner_f1.append(res["f1"])

    data_with_metric = data.copy()
    data_with_metric["ner_precision"] = ner_precision
    data_with_metric["ner_recall"] = ner_recall
    data_with_metric["ner_f1"] = ner_f1
    return data_with_metric


if __name__ == "__main__":
    # filename = input("Filename:")
    table_names = ["baseline3.5.xlsx", "upgraded3.5.xlsx", "upgraded4-o.xlsx"]
    for filename in table_names:
        data = pd.read_excel(filename)
        filename = ".".join(filename.split(".")[:-1])

        data_with_metrics = calc_classic_metrcis(data)
        data_with_metrics = calc_ner_accuracy(data_with_metrics)
        data_with_metrics = calc_judge_score(data_with_metrics)

        data_with_metrics.to_csv(filename + "_metrics.csv", encoding="utf-8", index=False)
        print(data_with_metrics.head(3))




