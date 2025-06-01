import importlib
import os
import sqlite3
from datetime import datetime
from uuid import uuid4

import ollama
import pandas as pd
import streamlit as st
from stqdm import stqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluation.utils import (
    calculate_avg_perplexity,
    calculate_sacrebleu,
    classify_sentence,
)
from ollama_interface.gateway import OllamaGateway


def fetch_data(split, con):
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT id, positive, negative 
        FROM example 
        WHERE split = :split
        """,
        {"split": split},
    ).fetchall()
    return pd.DataFrame(rows, columns=("Id", "Positive", "Negative"))


def fetch_flows():
    return [
        file
        for file in os.listdir("./flows")
        if file.endswith(".py") and not file.startswith("_")
    ]


def configuration(con):
    cur = con.cursor()
    splits = reversed(
        [
            row[0]
            for row in cur.execute(
                """
                SELECT DISTINCT split 
                FROM example
                """
            ).fetchall()
        ]
    )

    models = [model_desc.model for model_desc in ollama.list().models]

    config_cols = st.columns(4, vertical_alignment="top")

    with config_cols[0]:
        split = st.selectbox("Choose data split", options=splits)

    with config_cols[1]:
        model = st.selectbox(
            label="Ollama model",
            options=models,
        )

    with config_cols[2]:
        temperature = st.number_input(
            label="Temperature", min_value=0.0, max_value=1.0, step=0.01
        )

    with config_cols[3]:
        flow = st.selectbox(
            "Choose a TST Flow",
            options=fetch_flows(),
            format_func=lambda file: file.replace("_", " ")
            .replace("-", " ")
            .removesuffix(".py")
            .title(),
        )

    return (split, model, temperature, flow)


def show_data(title, data, col):
    data_container = col.expander(title, icon="📊", expanded=True)

    with data_container:
        st.dataframe(data.drop(columns=["Id"]))


def show_flow(file_name, col):
    with open(f"./flows/{file_name}") as f:
        flow_code = "".join(f.readlines())

    flow_container = col.expander("️Text Style Transfer Flow", icon="⌨️", expanded=True)
    with flow_container:
        st.code(flow_code, height=406)


def run_and_save_experiment(model, flow_file, temperature, con):
    with st.form(key="experiment", clear_on_submit=False):
        name = st.text_input("Experiment name")
        description = st.text_area("Description")
        if st.form_submit_button("Run"):
            cur = con.cursor()
            experiment_id = str(uuid4())
            cur.execute(
                """
                INSERT INTO experiment(id, timestamp, name, description, flow, model, temperature)
                VALUES (:id, :timestamp, :name, :description, :flow, :model, :temperature)
                """,
                {
                    "id": experiment_id,
                    "timestamp": str(datetime.now()),
                    "name": name,
                    "description": description,
                    "model": model,
                    "temperature": temperature,
                    "flow": flow_file,
                },
            )
            con.commit()
            return experiment_id

    return None


def evaluate_experiment(experiment_id, output_data, col, con):
    with col:
        st.subheader("Evaluate Experiment 🔍")
        if output_data is not None:
            tokenizer = AutoTokenizer.from_pretrained("yelp_review_classifier")
            classifier = AutoModelForSequenceClassification.from_pretrained(
                "yelp_review_classifier"
            )
            metrics = {}

            with st.spinner("Calculating Perplexity"):
                metrics["perplexity"] = calculate_avg_perplexity(output_data["Output"])

            with st.spinner("Calculating SacreBLEU"):
                metrics["rsbleu"] = calculate_sacrebleu(
                    output_data["Output"], output_data["Positive"]
                )

            metrics["accuracy"] = 0
            for sentence in stqdm(output_data["Output"].to_list()):
                metrics["accuracy"] += classify_sentence(
                    sentence, tokenizer, classifier
                )
            metrics["accuracy"] = metrics["accuracy"] / len(output_data)

            st.write(f"Average Perplexity: {metrics['perplexity']:.2f}")
            st.write(f"SacreBLEU: {metrics['rsbleu']:.2f}")
            st.write(f"Accuracy: {metrics['accuracy']:.2f}")

            cur = con.cursor()
            cur.execute(
                """
                UPDATE experiment
                SET avg_perplexity = :perplexity,
                    rsbleu = :rsbleu,
                    accuracy = :accuracy
                WHERE id = :experiment_id
                """,
                {
                    "experiment_id": experiment_id,
                    "perplexity": metrics["perplexity"],
                    "rsbleu": metrics["rsbleu"],
                    "accuracy": metrics["accuracy"],
                },
            )
            con.commit()


def save_experiment_outputs(output_data, experiment_id, con):
    cur = con.cursor()
    output_data.apply(
        lambda row: cur.execute(
            """
            INSERT INTO experiment_output(id, experiment_id, example_id, output)
            VALUES (:id, :experiment_id, :example_id, :output)
            """,
            {
                "id": str(uuid4()),
                "experiment_id": experiment_id,
                "example_id": row["Id"],
                "output": row["Output"],
            },
        ),
        axis=1,
    )
    con.commit()


def main():
    con = sqlite3.connect("./data.db")
    st.title("Text Sentiment Transfer Lab 🔬")

    # Setup Experiment
    st.header("Experiment Setup 🧪")
    split, model, temperature, flow_file = configuration(con)
    data = fetch_data(split, con)
    ollama_gateway = OllamaGateway(
        client=ollama.Client(), model=model, options={"temperature": temperature}
    )

    data_col, prompt_col = st.columns(2, vertical_alignment="top")
    show_data("Experiment Data", data, data_col)
    show_flow(flow_file, prompt_col)

    # Run & Save Experiment
    st.header("Run Experiment 🚀")
    experiment_id = run_and_save_experiment(
        model, flow_file.removesuffix(".py"), temperature, con
    )

    if "experiment_id" not in st.session_state:
        st.session_state["experiment_id"] = None

    if experiment_id is not None:
        st.session_state["experiment_id"] = experiment_id

    flow = importlib.import_module(f"flows.{flow_file.removesuffix('.py')}").flow
    results = []
    if st.session_state["experiment_id"]:
        with st.spinner("Processing..."):
            results: pd.Series = flow(data, ollama_gateway)

    results_col, evaluation_col = st.columns([4, 1])

    # Show Results
    with results_col:
        output_data = None
        st.subheader("Experiment Results 📋")
        if len(results) > 0:
            output_data = data.__deepcopy__()
            output_data["Output"] = results
            data_col = st.columns(1)[0]
            show_data("Experiment Results", output_data, data_col)
            save_experiment_outputs(output_data, experiment_id, con)

    # Evaluate Experiment
    evaluate_experiment(
        st.session_state["experiment_id"], output_data, evaluation_col, con
    )
    st.session_state["experiment_id"] = None


if __name__ == "__main__":
    main()
