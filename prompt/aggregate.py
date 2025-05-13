class PromptTemplate:
    """
    A prompt template object that stores the system prompt, and a user prompt template.
    It is used to build prompts more easily.

    Example:
    ```
    prompt = PromptTemplate(system = "You are a helpful AI", template = "Answer the following question: {question}\\nAnswer:")
    prompt.build_prompt(question = "What is the capital of Macedonia?")
    ```
    """

    def __init__(self, system: str, template: str):
        self.__system = system
        self.__template = template

    def system(self) -> str:
        return self.__system

    def template(self) -> str:
        return self.__template

    def build_prompt(self, **query_params) -> str:
        """
        Build a prompt from the prompt template using the provided query.
        All the keys present in the template must be provided, and any keys
        provided which are not present in the template will be ignored.

        :param query_params: The key value pairs for the template.
        :returns str: The prompt built from the prompt template using the query.
        """
        return self.__template.format_map(query_params)
