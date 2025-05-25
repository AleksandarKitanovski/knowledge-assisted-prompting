import evaluate
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def classify_sentence(
    sentence: str,
    tokenizer: AutoTokenizer,
    classifier: AutoModelForSequenceClassification,
) -> int:
    tokenized_sentence = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    tokens, attention_mask = (
        tokenized_sentence["input_ids"],
        tokenized_sentence["attention_mask"],
    )
    logits = classifier(tokens, attention_mask).logits
    return int(logits.argmax())


def calculate_accuracy(
    sentences: pd.Series,
    tokenizer: AutoTokenizer,
    classifier: AutoModelForSequenceClassification,
) -> float:
    tqdm.pandas()
    results = sentences.progress_apply(
        lambda s: classify_sentence(s, tokenizer, classifier)
    )
    return float(results.sum() / len(results))


def calculate_sacrebleu(predictions: pd.Series, references: pd.Series) -> float:
    sacrebleu = evaluate.load("sacrebleu")
    results = sacrebleu.compute(predictions=predictions, references=references)
    return results["score"]


def calculate_avg_perplexity(sentences: pd.Series) -> float:
    perplexity = evaluate.load("perplexity", module_type="metric")
    result = perplexity.compute(predictions=sentences, model_id="gpt2")
    return float(result["mean_perplexity"])
