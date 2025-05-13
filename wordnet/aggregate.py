class WordNetObject:
    """
    A WordNet object that stores a word, its position, and lists of its synonyms and antonyms.
    """
    def __init__(self, word: str, word_pos: str, synonyms: set[str], antonyms: set[str]):
        self.__word = word
        self.__word_pos = word_pos
        self.__synonyms = synonyms
        self.__antonyms = antonyms

    def word(self) -> str:
        self.__word

    def word_pos(self) -> str:
        self.__word_pos

    def synonyms(self) -> set[str]:
        self.__synonyms

    def antonyms(self) -> set[str]:
        self.__antonyms

    def to_prompt_format(self) -> str:
        """
        Transform the WordNetObject into prompt format.

        Example:

        ```
        Word: Great (adjective)
        Synonyms: [Excellent, Big] 
        Antonyms: [Terrible, Small]
        ```

        :returns str: A human readable version of the object.
        """
        return f"""Word: {self.__word} ({self.__word_pos})
Synonyms: [{", ".join(self.__synonyms)}]
Antonyms: [{", ".join(self.__antonyms)}]
"""

    def __str__(self) -> str:
        return self.__word

    def __repr__(self) -> str:
        return f"WordObject({self.__word}, {self.__synonyms}, {self.__antonyms})"

