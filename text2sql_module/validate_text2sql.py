import json
import argparse


def calculate_exact_match(predicted_sql, ground_truth_sql):
    """
    Проверяет, совпадает ли предсказанный SQL с эталонным.
    """
    return predicted_sql.strip().lower() == ground_truth_sql.strip().lower()


def validate_text2sql(predictions_path, ground_truth_path):
    """
    Валидирует предсказания модели, вычисляя метрику Exact Match.

    :param predictions_path: Путь к файлу с предсказанными SQL-запросами (JSON).
    :param ground_truth_path: Путь к файлу с эталонными данными (JSON).
    :return: Метрика Test EM.
    """
    # Загружаем данные
    with open(predictions_path, 'r') as pred_file:
        predictions = json.load(pred_file)

    with open(ground_truth_path, 'r') as gt_file:
        ground_truths = json.load(gt_file)

    assert len(predictions) == len(ground_truths), "Количество предсказаний и эталонных данных должно совпадать."

    exact_matches = 0

    # Сравнение предсказаний с эталонными данными
    for pred, gt in zip(predictions, ground_truths):
        predicted_sql = pred["predicted_sql"]
        ground_truth_sql = gt["ground_truth_sql"]

        if calculate_exact_match(predicted_sql, ground_truth_sql):
            exact_matches += 1

    total = len(ground_truths)
    test_em = exact_matches / total
    print(f"Exact Match (Test EM): {test_em:.2%} ({exact_matches}/{total})")
    return test_em


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Text2SQL model predictions.")
    parser.add_argument('--predictions', type=str, required=True, help="Path to predictions JSON file.")
    parser.add_argument('--ground_truth', type=str, required=True, help="Path to ground truth JSON file.")

    args = parser.parse_args()

    # Запуск валидации
    validate_text2sql(args.predictions, args.ground_truth)
