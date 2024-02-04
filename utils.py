import evaluate


def evaluate_explanations(predictions, references):
    print("eval being started")
    # rouge
    metric = evaluate.load("rouge")
    result = metric.compute(predictions=predictions, references=references, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    print("rouge done")

    # bleu
    metric = evaluate.load("bleu")
    result_bleu = metric.compute(predictions=predictions, references=references)
    result.update(result_bleu)
    print("bleu done")

    # bertscore
    metric = evaluate.load("bertscore")
    result_bertscore = metric.compute(predictions=predictions, references=references, lang="en")
    result_bertscore = {k: sum(v) / len(v) for k, v in result_bertscore.items() if k in ['precision', 'recall', 'f1']}
    result_bertscore["bertscore_precision"] = result_bertscore.pop("precision")
    result_bertscore["bertscore_recall"] = result_bertscore.pop("recall")
    result_bertscore["bertscore_f1"] = result_bertscore.pop("f1")
    result.update(result_bertscore)
    print("bertscore done")

    # meteor
    metric = evaluate.load("meteor")
    result_meteor = metric.compute(predictions=predictions, references=references)
    result.update(result_meteor)
    print("meteor done")

    # bleurt
    metric = evaluate.load('bleurt')
    result_bleurt = metric.compute(predictions=predictions, references=references)
    result_bleurt = {k: sum(v) / len(v) for k, v in result_bleurt.items()}
    result_bleurt["bleurt"] = result_bleurt.pop("scores")
    result.update(result_bleurt)
    print("bleurt done")

    return result
