from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

from wordnet.aggregate import WordNetObject


def get_polarizing_words(sentence: str, cutoff: float = 0.5) -> list[WordNetObject]:
    """
    Returns a list of polarizing word objects from the sentence which are found in SentiWordNet.
    Words which have a neg_score greater than cutoff are considered polarizing.

    :param str sentence: The query sentence
    :param float cutoff: The cutoff score to consider a word polarizing
    :returns dict: List of polarizing word objects
    """

    tagged_tokens = set(pos_tag(word_tokenize(sentence), tagset='universal'))
    tagged_tokens = [
        (token, _map_universal_pos_to_wn_pos(pos))
        for token, pos in tagged_tokens
    ]

    words = []
    for token, pos in tagged_tokens:
        if not _is_polarizing(token, pos, cutoff):
            continue

        synonyms = _get_word_synonyms(token, pos)
        antonyms = _get_word_antonyms(token, pos)
        words.append(WordNetObject(token, _map_wn_pos_to_human_readable(pos), synonyms, antonyms))

    return words


def _map_universal_pos_to_wn_pos(pos: str) -> str:
    pos_map = {"NOUN": wn.NOUN, "ADJ": wn.ADJ, "VERB": wn.VERB, "ADV": wn.ADV}

    return pos_map.get(pos, "")

def _map_wn_pos_to_human_readable(pos) -> str:
    pos_map = {wn.NOUN: "noun", wn.ADJ: "adjective", wn.VERB: "verb", wn.ADV: "adverb"}

    return pos_map.get(pos, "")

def _get_word_antonyms(token: str, pos: str) -> set[str]:
    synsets = wn.synsets(token, pos)
    antonyms = set()

    for synset in synsets:
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())

    return antonyms


def _get_word_synonyms(token: str, pos: str) -> set[str]:
    synsets = wn.synsets(token, pos)
    return set(synset.name().split(".")[0] for synset in synsets)


def _is_polarizing(token: str, pos: str, cutoff: float) -> bool:
    synsets = swn.senti_synsets(token, pos)
    for synset in synsets:
        if synset.neg_score() > cutoff:
            return True

    return False
