import pandas as pd
from stqdm import stqdm

from ollama_interface.gateway import OllamaGateway
from prompt.aggregate import PromptTemplate
from wordnet.aggregate import WordNetObject
from wordnet.utilities import get_polarizing_words


def flow(data: pd.DataFrame, ollama_gateway: OllamaGateway) -> str:
    first_prompt_template = PromptTemplate(
        system="""You are an AI specialized in sentiment transfer from negative to positive.
Your task is to rewrite a given negative customer review to express a positive sentiment, while preserving the original information and meaning.
Follow these guidelines:
    * Do not remove or alter the core facts or topics mentioned.
    * Keep the rewritten sentence approximately the same length as the original (within a few words).
    * The output must sound natural, fluent, and human, like a genuine positive review.
    * Only change the tone and sentiment from negative to positive - not the content.
Return only the transformed sentence.""",
        template="""Sentiment transfer changes the sentiment of a sentence while keeping non-sentiment-related content unchanged.
Suggested words to change: {words}
Change the sentiment of the following English sentence from negative to positive without adding any extra information: {sentence}
Output: """,
    )

    second_prompt_template = PromptTemplate(
        system="""You are a professional editor for customer reviews. Your job is to improve reviews by fixing grammar and making the overall tone positive, while keeping the original meaning as much as possible.
Follow these rules:
    1. Fix grammar and clarity issues.
    2. If a review is negative, reword it to make it sound positive or constructive.
    3. Do not remove information, but you can soften strong language or highlight any positive aspects.
    4. Never change the subject of the review or add new facts.
    Keep the revised review natural and authentic, like something a real customer would say.""",
        template="""Please fix the grammar of this review and make it sound positive. Keep all the original information, but improve clarity and tone. If the review is negative, reword it to sound constructive or mildly positive without changing the facts.

Review: {sentence}
Output: """,
    )

    results = []
    for sentence in stqdm(data["Negative"].to_list()):
        polarizing_words: list[WordNetObject] = get_polarizing_words(
            sentence, cutoff=0.5
        )
        if len(polarizing_words) > 0:
            words = "\n" + "\n".join(
                polarizing_word.to_prompt_format()
                for polarizing_word in polarizing_words
            )
        else:
            words = "No suggestions."

        first_pass = ollama_gateway.generate_response(
            first_prompt_template, sentence=sentence, words=words
        )
        second_pass = ollama_gateway.generate_response(
            second_prompt_template, sentence=first_pass
        )

        results.append(second_pass)

    return results
