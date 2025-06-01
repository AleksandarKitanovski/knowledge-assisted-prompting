import pandas as pd
from stqdm import stqdm

from ollama_interface.gateway import OllamaGateway
from prompt.aggregate import PromptTemplate


def flow(data: pd.DataFrame, ollama_gateway: OllamaGateway) -> list:
    prompt_template = PromptTemplate(
        system="""You are an AI specialized in sentiment transfer from negative to positive.
Your task is to rewrite a given negative customer review to express a positive sentiment, while preserving the original information and meaning.
Follow these guidelines:
    * Do not remove or alter the core facts or topics mentioned.
    * Keep the rewritten sentence approximately the same length as the original (within a few words).
    * The output must sound natural, fluent, and human, like a genuine positive review.
    * Only change the tone and sentiment from negative to positive — not the content.
Return only the transformed sentence.""",
        template="""Sentiment transfer changes the sentiment of a sentence while keeping non-sentiment-related content unchanged.
Change the sentiment of the following English sentence from negative to positive without adding any extra information: {sentence}
Output: """,
    )

    results = []
    for sentence in stqdm(data["Negative"].to_list()):
        results.append(ollama_gateway.generate_response(prompt_template, sentence=sentence))

    return results
