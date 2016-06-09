# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import re
import zipfile
import nltk



def unicode_compatible(cls):
    """
    Decorator for unicode compatible classes. Method ``__unicode__``
    has to be implemented to work decorator as expected.
    """
    cls.__str__ = lambda self: self.__unicode__().encode("utf8")

    return cls


def to_string(object):
    return to_unicode(object) 

def to_unicode(object):
    if isinstance(object, unicode):
        return object
    elif isinstance(object, bytes):
        return object.decode("utf8")
    else:
        # try decode instance to unicode
        return instance_to_unicode(object)

def instance_to_unicode(instance):
    if hasattr(instance, "__unicode__"):
        return unicode(instance)
    elif hasattr(instance, "__str__"):
        return bytes(instance).decode("utf8")


class Tokenizer(object):
    """Language dependent tokenizer of text document."""

    _WORD_PATTERN = re.compile(r"^[^\W\d_]+$", re.UNICODE)
    # feel free to contribute if you have better tokenizer for any of these languages :)
    LANGUAGE_ALIASES = {
        "slovak": "czech",
    }

    # improve tokenizer by adding specific abbreviations it has issues with
    # note the final point in these items must not be included
    LANGUAGE_EXTRA_ABREVS = {
        "english": ["e.g", "al", "i.e"],
        "german": ["al", "z.B", "Inc", "engl", "z. B", "vgl", "lat", "bzw", "S"],
    }

    def __init__(self, language):
        self._language = language

        tokenizer_language = self.LANGUAGE_ALIASES.get(language, language)
        self._sentence_tokenizer = self._sentence_tokenizer(tokenizer_language)

    @property
    def language(self):
        return self._language

    def _sentence_tokenizer(self, language):
        try:
            path = to_string("tokenizers/punkt/%s.pickle") % to_string(language)
            return nltk.data.load(path)
        except (LookupError, zipfile.BadZipfile):
            raise LookupError(
                "NLTK tokenizers are missing. Download them by following command: "
                '''python -c "import nltk; nltk.download('punkt')"'''
            )

    def to_sentences(self, paragraph):
        extra_abbreviations = self.LANGUAGE_EXTRA_ABREVS.get(self._language, [])
        self._sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
        sentences = self._sentence_tokenizer.tokenize(to_unicode(paragraph))
        return tuple(map(unicode.strip, sentences))

    def to_words(self, sentence):
        words = nltk.word_tokenize(to_unicode(sentence))
        return tuple(filter(self._is_word, words))

    def _is_word(self, word):
        return bool(Tokenizer._WORD_PATTERN.search(word))