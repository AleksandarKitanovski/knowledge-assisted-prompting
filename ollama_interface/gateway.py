import json
from ollama import Client, GenerateResponse
from pydantic import BaseModel

from prompt.aggregate import PromptTemplate


class Response(BaseModel):
    output: str


class RelatedWords(BaseModel):
    synonyms: list[str]
    antonyms: list[str]


class OllamaGateway:
    def __init__(self, client: Client, model: str, options: dict[str, str]):
        self.__client = client
        self.__model = model
        self.__options = options

    def generate_response(self, prompt_template: PromptTemplate, **query_params) -> str:
        result: GenerateResponse = self.__client.generate(
            model=self.__model,
            format=Response.model_json_schema(),
            options=self.__options,
            system=prompt_template.system(),
            prompt=prompt_template.build_prompt(**query_params),
        )

        response = json.loads(result.response)
        return response["output"]

    def generate_synonyms_and_antonyms(
        self, prompt_template: PromptTemplate, **query_params
    ) -> str:
        result: GenerateResponse = self.__client.generate(
            model=self.__model,
            format=RelatedWords.model_json_schema(),
            options=self.__options,
            system=prompt_template.system(),
            prompt=prompt_template.build_prompt(**query_params),
        )

        response = json.loads(result.response)
        return response
