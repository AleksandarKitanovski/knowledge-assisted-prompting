import networkx as nx
from nltk import pos_tag
from nltk.tokenize import word_tokenize as tokenize

from skg.aggregate import SKGObject
from wordnet.utilities import _map_universal_pos_to_wn_pos

_graph = nx.read_gml(
    "./skg/words_graph.gml"
)  # avoid reading the graph on every function call


def get_polarizing_words(sentence: str) -> list[SKGObject]:
    """
    Returns a list of polarizing word objects from the sentence which are found in our SKG.

    :param str sentence: The query sentence
    :returns dict: List of polarizing word objects
    """

    tokens = tokenize(sentence)
    tagged_tokens = set(pos_tag(tokens, tagset="universal"))
    tagged_words = {
        f"{token.lower()} ({_map_universal_pos_to_wn_pos(pos)})"
        for token, pos in tagged_tokens
        if _map_universal_pos_to_wn_pos(pos) != ""
    }

    words = []
    for tagged_word in tagged_words:
        synonyms = get_synonyms(tagged_word)
        antonyms = get_antonyms(tagged_word)
        words.append(SKGObject(tagged_word, synonyms, antonyms))

    return words


def get_antonyms(start_node: str) -> set[str]:
    if start_node not in _graph:
        return set()
    
    antonyms = set()
    for node in _graph[start_node]:
        edge = _graph[start_node][node]
        if edge["label"] == "antonym":
            antonyms.add(node)

    return antonyms


def get_synonyms(start_node: str) -> set[str]:
    if start_node not in _graph:
        return set()

    synonyms = set()
    for node in _graph[start_node]:
        edge = _graph[start_node][node]
        if edge["label"] == "synonym":
            synonyms.add(node)

    return synonyms
