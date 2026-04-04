import argparse
import sqlite3

import pandas as pd
from evaluation.utils import (
    calculate_accuracy,
    calculate_avg_perplexity,
    calculate_sacrebleu,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_data(db_con: sqlite3.Connection, experiment: str) -> pd.DataFrame:
    return pd.read_sql(
        """
SELECT eo.id, eo.output, e.positive, e.negative
FROM experiment_output as eo
INNER JOIN example as e ON eo.example_id = e.id
WHERE eo.experiment_id = :experiment_id;
""",
        params={"experiment_id": experiment},
        con=db_con,
    )


def save_metrics(
    db_con: sqlite3.Connection,
    experiment: str,
    accuracy: float,
    sacrebleu: float,
    perplexity: float,
) -> None:
    cur = db_con.cursor()
    cur.execute(
        """
UPDATE experiment
SET accuracy = :accuracy,
    rsbleu = :sacrebleu,
    avg_perplexity = :perplexity
WHERE id = :experiment_id;
""",
        {
            "accuracy": accuracy,
            "sacrebleu": sacrebleu,
            "perplexity": perplexity,
            "experiment_id": experiment,
        },
    )


def run_evaluation(
    experiment: str, classifier_name: str, db: str, verbose: bool
) -> None:
    con = sqlite3.connect(db)

    data = get_data(con, experiment)
    tokenizer = AutoTokenizer.from_pretrained(classifier_name)
    classifier = AutoModelForSequenceClassification.from_pretrained(classifier_name)

    accuracy = calculate_accuracy(data["output"], tokenizer, classifier)
    sacrebleu = calculate_sacrebleu(data["output"], data["positive"])
    perplexity = calculate_avg_perplexity(data["output"])

    save_metrics(con, experiment, accuracy, sacrebleu, perplexity)

    if verbose:
        print(f"Accuracy: {accuracy:.2f}")
        print(f"SacreBLEU: {sacrebleu:.2f}")
        print(f"Perplexity (GPT-2): {perplexity:.2f}")


def main():
    parser = argparse.ArgumentParser(
        prog="evaluate_experiment.py",
        description="Evaluate the outputs of a text style transfer experiment.",
    )

    parser.add_argument(
        "--experiment",
        help="the id of the experiment to evaluate",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--database",
        help="the path to the sqlite database where the experiments are kept",
        type=str,
        default="data.db",
    )
    parser.add_argument(
        "--classifier",
        help="name of the classifier model to be used in the accuracy calculation (local or from huggingface)",
        type=str,
        default="aleks240/yelp_review_classifier",
    )
    parser.add_argument(
        "--verbose",
        help="print the results of the experiment",
        action="store_true",
    )

    args = parser.parse_args()

    run_evaluation(args.experiment, args.classifier, args.database, args.verbose)


if __name__ == "__main__":
    main()
