from datetime import datetime
import sqlite3

import pandas as pd
import streamlit as st


def fetch_experiments(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT id, timestamp, name, description, model, temperature, flow, rsbleu, accuracy, avg_perplexity
        FROM experiment
        """,
        con=con,
        index_col="id",
    )


def delete_experiment(experiment_id: str, con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute(
        """
        DELETE
        FROM experiment
        WHERE id = :id
        """,
        {"id": experiment_id},
    )
    con.commit()


def main():
    con = sqlite3.connect("./data.db")
    experiments = fetch_experiments(con)

    st.title("Experiment Explorer 🔍")

    # Overview
    st.header("Experiments Overview 📓")
    st.dataframe(experiments)

    # Details
    st.header("Experiment Details 🔬")

    details_col, metrics_col, flow_col = st.columns([1, 1, 2])
    with details_col:
        st.subheader("Info")
        experiment_id = st.selectbox(
            "Select experiment",
            options=experiments.index,
            format_func=lambda exp_id: experiments.loc[exp_id]["name"],
            key="experiment",
        )
        experiment = experiments.loc[experiment_id]

        st.caption("Date & Time")
        with details_col.container(border=True):
            timestamp = datetime.fromisoformat(experiment["timestamp"])
            st.write(timestamp.strftime("%d %B, %Y - %H:%M"))

        st.caption("Name")
        with details_col.container(border=True):
            st.write(experiment["name"])

        st.caption("Description")
        with details_col.container(border=True):
            st.write(experiment["description"])

        st.caption("Model")
        with details_col.container(border=True):
            st.write(experiment["model"])

        st.caption("Temperature")
        with details_col.container(border=True):
            st.write(f"{experiment['temperature']}")

    with metrics_col:
        st.subheader("Metrics")
        baseline_id = st.selectbox(
            "Select baseline",
            options=experiments.index,
            format_func=lambda exp_id: experiments.loc[exp_id]["name"],
            key="baseline",
        )
        baseline = experiments.loc[baseline_id]

        st.metric(
            label="Average Perplexity",
            value=f"{experiment['avg_perplexity']:.2f}",
            delta=f"{experiment['avg_perplexity'] - baseline['avg_perplexity']:.2f}",
            delta_color="inverse",
            border=True,
        )
        st.metric(
            label="SacreBLEU",
            value=f"{experiment['rsbleu']:.2f}",
            delta=f"{experiment['rsbleu'] - baseline['rsbleu']:.2f}",
            border=True,
        )
        st.metric(
            label="Accuracy",
            value=f"{experiment['accuracy']:.2f}",
            delta=f"{experiment['accuracy'] - baseline['accuracy']:.2f}",
            border=True,
        )

    with open(f"./flows/{experiment['flow']}.py") as f:
        flow_code = "".join(f.readlines())

    with flow_col:
        st.subheader("Flow")
        st.code(flow_code)

    if st.button("Delete Experiment", type='primary'):
        delete_experiment(experiment_id, con)
        st.rerun()


if __name__ == "__main__":
    main()
