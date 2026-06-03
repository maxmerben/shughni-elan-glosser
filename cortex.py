import json, os, re, pympi, warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from copy import copy
import converter as cv

DEFAULT_INDENT = 2


sgh_sentences_tiername_type = "sentence-txt-sgh"
ru_sentences_tiername_type = "sentence-txt-ru"
sgh_tokens_tiername_type = "word-txt-sgh"
sgh_morphs_tiername_type = "morph-txt-sgh"
eng_glosses_tiername_type = "morph-gls-en"
eng_pos_tiername_type = "morph-pos-en"
sgh_original_tiername_type = "orig-txt-sgh"
comment_tiername_type = "comment-txt-ru"

tier_types = {"sentences": sgh_sentences_tiername_type,
              "translations": ru_sentences_tiername_type,
              "tokens": sgh_tokens_tiername_type,
              "morphs": sgh_morphs_tiername_type,
              "glosses": eng_glosses_tiername_type,
              "pos": eng_pos_tiername_type,
              "original": sgh_original_tiername_type,
              "comment": comment_tiername_type}

linguistic_types_description = {
 "sentence-txt-sgh": {
  "GRAPHIC_REFERENCES": "false",
  "LINGUISTIC_TYPE_ID": "sentence-txt-sgh",
  "TIME_ALIGNABLE": "true"},
 "sentence-txt-ru": {"CONSTRAINTS": "Symbolic_Association",
  "GRAPHIC_REFERENCES": "false",
  "LINGUISTIC_TYPE_ID": "sentence-txt-ru",
  "TIME_ALIGNABLE": "false"},
 "word-txt-sgh": {"CONSTRAINTS": "Symbolic_Subdivision",
  "GRAPHIC_REFERENCES": "false",
  "LINGUISTIC_TYPE_ID": "word-txt-sgh",
  "TIME_ALIGNABLE": "false"},
 "morph-txt-sgh": {"CONSTRAINTS": "Symbolic_Subdivision",
  "GRAPHIC_REFERENCES": "false",
  "LINGUISTIC_TYPE_ID": "morph-txt-sgh",
  "TIME_ALIGNABLE": "false"},
 "morph-gls-en": {"CONSTRAINTS": "Symbolic_Association",
  "GRAPHIC_REFERENCES": "false",
  "LINGUISTIC_TYPE_ID": "morph-gls-en",
  "TIME_ALIGNABLE": "false"},
 "morph-pos-en": {"CONSTRAINTS": "Symbolic_Association",
  "GRAPHIC_REFERENCES": "false",
  "LINGUISTIC_TYPE_ID": "morph-pos-en",
  "TIME_ALIGNABLE": "false"},
 "orig-txt-sgh": {"CONSTRAINTS": "Symbolic_Association",
  "GRAPHIC_REFERENCES": "false",
  "LINGUISTIC_TYPE_ID": "orig-txt-sgh",
  "TIME_ALIGNABLE": "false"},
 "comment-txt-ru": {"CONSTRAINTS": "Symbolic_Association",
  "GRAPHIC_REFERENCES": "false",
  "LINGUISTIC_TYPE_ID": "comment-txt-ru",
  "TIME_ALIGNABLE": "false"}
}

PUNCT = "\.,\!\?…«»„“”‘'\"\/\\:;\[\]\(\)\{\}\<>@#\$\%\^\&\*\+—"
PREPUNCT = "«„“‘'\"\[\(\{\<"
POSTPUNCT = "\.,\!\?…»”:;’'\"\]\)\}>"

DEFAULT_ANNOTATION_DENSITY = 6000//6



def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = lambda msg, *args, **kwargs: f"Warning: {msg}\n"


########################


def dehyphen(text: str, equal=True):
    """
    Remove morpheme separators (hyphen-like symbols) from the input string.

    Parameters:
        text (str): The input string.
        equal (bool): If True, also remove equal signs ('=') (default: True).

    Returns:
        str: The text with morpheme separator symbols removed.
    """
    if equal:
        return re.sub("[\-−—=]+", "", text)
    else:
        return re.sub("[\-−—]+", "", text)


def depunct(text: str):
    """
    Remove punctuation characters in the beginning and in the end of the input string.

    Parameters:
        text (str): The input string.

    Returns:
        str: The string without punctuation in the beginning and in the end.
    """
    return re.sub(f"^[{PUNCT}]+|[{PUNCT}]+$", "", text)


def _join_morpheme_strings(morpheme_strings: list):
    """
    Join a list of morpheme strings into a single string with normalized separators. Morpheme strings are joined with morpheme separators, if they are provided.

    Parameters:
        morpheme_strings (List[str]): List of morpheme strings.

    Returns:
        str: The formatted morpheme string.
    """
    res = "-".join(morpheme_strings)
    res = re.sub("--", "-", res)
    return re.sub("-?==?-?", "=", res)


def _clean(text: str):
    """
    Clean whitespace in the beginning and in the end of the string and collapse double spaces to single spaces.

    Parameters:
        text (str): The input string.

    Returns:
        str: Cleaned string.
    """
    text = re.sub("^[ \n\t]+|[ \n\t]+$", "", text)
    return re.sub("  ", " ", text)
    

def _clean_morpheme(text: str):
    """
    Clean and split a morpheme into prefix, suffix and core morpheme string. Morpheme separators (hyphens and equal signs) in the beginning and the end of the string are extracted as prefix and suffix. Spaces are replaced with underscores.

    Parameters:
        text (str): The initial morpheme string.

    Returns:
        Tuple[str, str, str]: A tuple (prefix, morpheme_string, suffix), where:
            - prefix: '-' or '=' character in the beginning
            - morpheme_string: core morpheme string
            - suffix: '-' or '=' character in the end
    """
    text = re.sub("  +", "_", text)
    res = re.fullmatch("([\-=]*)(.*?)([\-=]*)", _clean(text))
    if res is None:
        return None, None, None
    return res.group(1), res.group(2), res.group(3)


def create_obligatory_tiers(eaf, speakers=["A"]):

    for speaker in speakers:

        tiername = f"{speaker}_{sgh_sentences_tiername_type}"
        if not tiername in eaf.tiers:
            eaf.add_tier(
                tiername, ling=sgh_sentences_tiername_type,
                parent=None, language="sgh")
            eaf.tiers[tiername] = (
                {},   # future sentence annotations
                {},
                {'LANG_REF': 'sgh',
                    'LINGUISTIC_TYPE_REF': sgh_sentences_tiername_type,
                    'PARTICIPANT': "A",
                    'TIER_ID': tiername},
                0     # 0 = Sentence tier
            )
        
        tiername = f"{speaker}_{ru_sentences_tiername_type}"

        if not tiername in eaf.tiers:
            parent_tiername = f"{speaker}_{sgh_sentences_tiername_type}"
            eaf.add_tier(
                tiername, ling=ru_sentences_tiername_type,
                parent=parent_tiername, language="ru")
            eaf.tiers[tiername] = (
                {},
                {},   # future translation annotations
                {'LANG_REF': "ru",
                    'LINGUISTIC_TYPE_REF': ru_sentences_tiername_type,
                    'PARENT_REF': parent_tiername,
                    'TIER_ID': tiername},
                1     # 1 = Translation tier
            )

        tiername = f"{speaker}_{sgh_original_tiername_type}"
        if not tiername in eaf.tiers:
            parent_tiername = f"{speaker}_{sgh_sentences_tiername_type}"
            eaf.add_tier(
                tiername, ling=sgh_original_tiername_type,
                parent=parent_tiername, language="sgh")
            eaf.tiers[tiername] = (
                {},
                {},   # future original annotations
                {'LANG_REF': "sgh",
                    'LINGUISTIC_TYPE_REF': sgh_original_tiername_type,
                    'PARENT_REF': parent_tiername,
                    'TIER_ID': tiername},
                6     # 6 = Original tier
            )
        
        tiername = f"{speaker}_{comment_tiername_type}"
        if not tiername in eaf.tiers:
            parent_tiername = f"{speaker}_{sgh_sentences_tiername_type}"
            eaf.add_tier(
                tiername, ling=comment_tiername_type,
                parent=parent_tiername, language="ru")
            eaf.tiers[tiername] = (
                {},
                {},   # future comment annotations
                {'LANG_REF': "ru",
                    'LINGUISTIC_TYPE_REF': comment_tiername_type,
                    'PARENT_REF': parent_tiername,
                    'TIER_ID': tiername},
                7     # 7 = Comment tier
            )
        
        tiername = f"{speaker}_{sgh_tokens_tiername_type}"
        if not tiername in eaf.tiers:
            parent_tiername = f"{speaker}_{sgh_sentences_tiername_type}"
            eaf.add_tier(
                tiername, ling=sgh_tokens_tiername_type,
                parent=parent_tiername, language="sgh")
            eaf.tiers[tiername] = [
                {},
                {},   # future token annotations
                {'LANG_REF': 'sgh',
                    'LINGUISTIC_TYPE_REF': sgh_tokens_tiername_type,
                    'PARENT_REF': parent_tiername,
                    'TIER_ID': tiername},
                2     # 2 = Tokens tier
            ]
        
        tiername = f"{speaker}_{sgh_morphs_tiername_type}"
        if not tiername in eaf.tiers:
            parent_tiername = f"{speaker}_{sgh_tokens_tiername_type}"
            eaf.add_tier(
                tiername, ling=sgh_morphs_tiername_type,
                parent=parent_tiername, language="sgh")
            eaf.tiers[tiername] = [
                {},
                {},   # future morph annotations
                {'LANG_REF': 'sgh',
                    'LINGUISTIC_TYPE_REF': sgh_morphs_tiername_type,
                    'PARENT_REF': parent_tiername,
                    'TIER_ID': tiername},
                3     # 3 = Morphs tier
            ]
        
        tiername = f"{speaker}_{eng_glosses_tiername_type}"
        if not tiername in eaf.tiers:
            parent_tiername = f"{speaker}_{sgh_morphs_tiername_type}"
            eaf.add_tier(
                tiername, ling=eng_glosses_tiername_type,
                parent=parent_tiername, language="en")
            eaf.tiers[tiername] = [
                {},
                {},   # future gloss annotations
                {'LANG_REF': 'en',
                    'LINGUISTIC_TYPE_REF': eng_glosses_tiername_type,
                    'PARENT_REF': parent_tiername,
                    'TIER_ID': tiername},
                4     # 4 = Glosses tier
            ]

        tiername = f"{speaker}_{eng_pos_tiername_type}"
        if not tiername in eaf.tiers:
            parent_tiername = f"{speaker}_{sgh_morphs_tiername_type}"
            eaf.add_tier(
                tiername, ling=eng_pos_tiername_type,
                parent=parent_tiername, language="en")
            eaf.tiers[tiername] = [
                {},
                {},   # future gloss annotations
                {'LANG_REF': 'en',
                    'LINGUISTIC_TYPE_REF': eng_pos_tiername_type,
                    'PARENT_REF': parent_tiername,
                    'TIER_ID': tiername},
                5     # 5 = POS tier
            ]


def create_annotation(eaf, content, convert=False, target="orig", base=None, force=False):
    """
    Create an annotation for a Sentence, a Token or a Morpheme on a corresponding EAF layer.

    Parameters:
        eaf (pympi.Eaf): The EAF object.
        content: a Sentence, a Token or a Morpheme to be added to the EAF object.
        convert (bool): Whether to convert a Sentence to the normalized orthography
            and store the original orthography in the `original` layer.
        force (bool): Whether to force a token into a different orthography or not.
    """
    
    if isinstance(content, Sentence):
        
        ### CREATING ANNOTATION FOR SHUGHNI SENTENCE
        
        content.sent_aid = eaf.generate_annotation_id()
        tiername = f"{content.meta['speaker']}_{sgh_sentences_tiername_type}"
        
        if not tiername in eaf.tiers:
            eaf.add_tier(
                tiername, ling=sgh_sentences_tiername_type,
                parent=None, language="sgh")
            eaf.tiers[tiername] = (
                {},   # future sentence annotations
                {},
                {'LANG_REF': 'sgh',
                    'LINGUISTIC_TYPE_REF': sgh_sentences_tiername_type,
                    'PARTICIPANT': "A",
                    'TIER_ID': tiername},
                0     # 0 = Sentence tier
            )
        
        eaf.annotations[content.sent_aid] = tiername

        if convert and content.original is None:
            content.original = content.text
            content.convert_orthography(target=target, base=base, force=force)

        off_start = eaf.generate_ts_id(time=content.timestamps[0])
        off_end = eaf.generate_ts_id(time=content.timestamps[1])
        eaf.tiers[tiername][0][content.sent_aid] = (
            off_start, off_end, content.text, None)
        
        if content.audio and "src_alignment" in content.audio:
            start, end = content.audio["src_alignment"][0]["src_id"].split("_")
            eaf.timeslots[off_start] = start
            eaf.timeslots[off_end] = end
        
        ### TRANSLATIONS
        
        if content.translation:
            for language in content.translation:
        
                content.trans_aid = eaf.generate_annotation_id()
                tiername = f"{content.meta['speaker']}_{ru_sentences_tiername_type}"

                if not tiername in eaf.tiers:
                    parent_tiername = f"{content.meta['speaker']}_{sgh_sentences_tiername_type}"
                    eaf.add_tier(
                        tiername, ling=ru_sentences_tiername_type,
                        parent=parent_tiername, language=language)
                    eaf.tiers[tiername] = (
                        {},
                        {},   # future translation annotations
                        {'LANG_REF': language,
                            'LINGUISTIC_TYPE_REF': ru_sentences_tiername_type,
                            'PARENT_REF': parent_tiername,
                            'TIER_ID': tiername},
                        1     # 1 = Translation tier
                    )

                eaf.annotations[content.trans_aid] = tiername
                
                eaf.tiers[tiername][1][content.trans_aid] = (
                    content.sent_aid, content.translation[language], None, None)

        ### ORIGINAL TEXT

        tiername = f"{content.meta['speaker']}_{sgh_original_tiername_type}"
        if content.original:
            content.original_aid = eaf.generate_annotation_id()
            eaf.annotations[content.original_aid] = tiername
            eaf.tiers[tiername][1][content.original_aid] = (
                content.sent_aid, content.original, None, None)

        ### COMMENT

        tiername = f"{content.meta['speaker']}_{comment_tiername_type}"
        if content.comment:
            content.comment_aid = eaf.generate_annotation_id()
            eaf.annotations[content.comment_aid] = tiername
            eaf.tiers[tiername][1][content.comment_aid] = (
                content.sent_aid, content.comment, None, None)
            
    elif isinstance(content, Token):
        
        # CREATING ANNOTATION FOR SHUGHNI TOKEN
        
        tiername = f"{content.back_sent.meta['speaker']}_{sgh_tokens_tiername_type}"
        content.token_aid = eaf.generate_annotation_id()

        eaf.annotations[content.token_aid] = tiername
        
        previous_token_aid = None
        if content._id != 1:
            previous_token_aid = content.back_sent[content._id - 2].token_aid
        
        eaf.tiers[tiername][1][content.token_aid] = (
            content.back_sent.sent_aid, content.token, previous_token_aid, None)
    
    elif isinstance(content, Morpheme):
    
        # CREATING ANNOTATION FOR SHUGHNI MORPHEME
        
        ### MORPH
        
        tiername = f"{content.back_sent.meta['speaker']}_{sgh_morphs_tiername_type}"
        
        if content.morph:
            content.morph_aid = eaf.generate_annotation_id()
            eaf.annotations[content.morph_aid] = tiername
            
            previous_morph_aid = None
            if content._id != 1:
                previous_morph_aid = content.back_ana[content._id - 2].morph_aid
            
            eaf.tiers[tiername][1][content.morph_aid] = (
                content.back_token.token_aid, content.morph_full, previous_morph_aid, None)
            
        ### GLOSS
        
        tiername = f"{content.back_sent.meta['speaker']}_{eng_glosses_tiername_type}"
        
        if content.gloss:
            content.gloss_aid = eaf.generate_annotation_id()
            eaf.annotations[content.gloss_aid] = tiername
            eaf.tiers[tiername][1][content.gloss_aid] = (
                content.morph_aid, content.gloss, None, None)
        
        ### POS TAG
        
        tiername = f"{content.back_sent.meta['speaker']}_{eng_pos_tiername_type}"
        
        if content.pos:
            content.pos_aid = eaf.generate_annotation_id()
            eaf.annotations[content.pos_aid] = tiername
            eaf.tiers[tiername][1][content.pos_aid] = (
                content.morph_aid, content.pos, None, None)

    else:
        raise TypeError("Function create_annotation() requires that the argument `content` is a Sentence, Token or a Morpheme!")

########################


def _ispunct(text: str, punct=PUNCT):
    """
    Check whether the input string consists only of punctuation and/or whitespace.

    Parameters:
        text (str): The string to check.
        punct (str): A regular expression with all necessary punctuation symbols.

    Returns:
        bool: True if the string contains only characters
              from Tokenizer.punct or whitespace.
    """
    return bool(re.fullmatch(f"[ \t\n{punct}]+", text))


def apply_text_filter(t, text_filter):
    add_text = True
    if text_filter:
        for key in text_filter:
            if key in t.__dict__:
                if not (t.__dict__[key] in text_filter[key]):
                    add_text = False
                    break
            elif key in t.__dict__["meta"]:
                if not (t.__dict__["meta"][key] in text_filter[key]):
                    add_text = False
                    break
            else:
                add_text = False
                break
    return add_text


def check_for_text_filter_integrity(text_filter):
    if text_filter:
        if (not isinstance(text_filter, dict)):
            raise ValueError("The `text_filter` argument in Text.search_morpheme() must be a dictionary!")
        for key in text_filter:
            if isinstance(text_filter[key], str):
                text_filter[key] = [text_filter[key]]
            elif not (isinstance(text_filter[key], list) or isinstance(text_filter[key], tuple)):
                raise ValueError("The values of `text_filter` must be lists or tuples!")


class Tokenizer():
    """
    Customizable Tokenizer object for segmenting text based on punctuation and whitespace.

    Supports various tokenization modes (e.g. separating punctuation, deleting it)
    and can return either string tokens or regex match objects.

    Attributes:
        mode (str): Tokenization mode, one of:
            - 'glue_punct': glue punctuation to surrounding text, delete whitespace
               e.g. 'hi Mark!' -> ['hi', 'Mark!']
            - 'separate_punct': separate punctuation into its own tokens, delete whitespace
               e.g. 'hi Mark!' -> ['hi', 'Mark', '!']
            - 'delete_punct': remove punctuation and whitespace entirely
               e.g. 'hi Mark!' -> ['hi', 'Mark']
            - 'separate_all': separate everything (whitespace, words, punctuation)
               e.g. 'hi Mark!' -> ['hi', ' ', 'Mark', '!']
        output (str): Output format, either 'text' (list of strings) or 'match' (match objects).
        punct (str): String of characters that are treated as punctuation.
        strip (bool): Whether to strip whitespace from the input before tokenizing.
        query (str): Compiled regular expression pattern used for tokenization.
    """
    
    def __init__(self, mode="glue_punct", punct=None, strip=True, output="text"):
        """
        Initialize a Tokenizer object.

        Parameters:
            mode (str): Tokenization mode. Must be one of: 'glue_punct',
                'separate_punct', 'delete_punct', 'separate_all' (default: 'glue_punct').
            punct (str or None): A regular expression with punctuation characters.
                Uses the global variable PUNCT if None (default: None).
            strip (bool): Whether to strip whitespace from the input text (default: True).
            output (str): Output format. 'text' returns strings, 'match' returns match objects
                (default: 'text').
        """
        
        if mode in ('glue_punct', 'separate_punct', 'delete_punct', 'separate_all'):
            self.mode = mode
        else:
            raise NameError("Unknown mode value! Choose one of these: 'glue_punct', 'separate_punct', 'delete_punct', 'separate_all'")
        
        if output in ('text', 'match'):
            self.output = output
        else:
            raise NameError("Unknown output value! Choose one of these: 'text', 'match'")

        self.punct = punct if punct is not None else PUNCT
        self.strip = strip

        if self.mode == "glue_punct":
            self.query = "[^ \t\n]+"
        elif self.mode == "separate_punct":
            self.query = f"[^ \t\n{self.punct}]+|[{self.punct}]+"
        elif self.mode == "delete_punct":
            self.query = f"[^ \t\n{self.punct}]+"
        elif self.mode == "separate_all":
            self.query = f"[ \t\n]+|[^ \t\n{self.punct}]+|[{self.punct}]+"

    def tokenize(self, text: str):
        """
        Tokenize the input text.

        Parameters:
            text (str): The input string to tokenize.

        Returns:
            list: A list of string tokens (if Tokenizer.output='text')
                or match objects (if Tokenizer.output='match').
        """
        
        if self.strip:
            text = text.strip()
        if self.output=="text":
            return re.findall(self.query, text)
        elif self.output=="match":
            return re.finditer(self.query, text)


class MorphemeSearch():
    """
    A list-like class with results of Morpheme search.

    Properties:
        as_ana (list): List of corresponding Analysis objects for each Morpheme.
        as_sent (list): List of corresponding Sentence objects for each Morpheme.
        as_glossing (list): Glossed string representation of corresponding
            Sentence objects for each Morpheme.
        morphemes (list): Frequency-sorted list of (morph, gloss) tuples.
        morphs (list): Frequency-sorted list of morphs.
        glosses (list): Frequency-sorted list of glosses.
        text_filter (dict or None): Dictionary with text filters applied in the search.

    Methods:
        __getitem__(index): Return the Morpheme at the given index.
        __repr__(): Formal string representation of the Morpheme list.
        __str__(): Informal string representation (same as __repr__).
        __len__(): Number of found Morphemes.
        __iter__(): Iterator over all Morphemes.
        append(new_morpheme): Add a single Morpheme to the list.
        extend(new_morphemes): Add multiple Morphemes to the list.
        regloss(gloss, pos): Replace the gloss of all Morphemes in the search results with the given value.
    """
    
    def __init__(self, morphemes=None, text_filter=None):
        """
        Initialize a MorphemeSearch object.

        Parameters:
            morphemes (list): A preset list of Morpheme objects (defaults to an empty list).
                Necessary for calls from other internal functions.
            text_filter (dict): Dictionary with text filters applied in the search (defaults to None).
        """
        self._morphemes = morphemes if morphemes else []
        self.text_filter = text_filter
    
    def __getitem__(self, index: int):
        """
        Return a Morpheme at the specified index.

        Parameters:
            index (int): The index of the Morpheme to retrieve.

        Returns:
            The Morpheme at the specified index, or None if the list is empty.
        """
        return self._morphemes[index] if self._morphemes else None
    
    def __repr__(self):
        """
        Return a formal string representation of the Morphemes list.

        Returns:
            str: A string representation.
        """
        return "[" + ",\n ".join([repr(m) for m in self._morphemes]) + "]"
    
    def __str__(self):
        return repr(self)

    def __len__(self):
        """
        Return number of found Morphemes.

        Returns:
            int: The number of Morphemes.
        """
        return len(self._morphemes) if self._morphemes else 0
    
    def __iter__(self):
        """
        Enable iteration over the Morphemes.

        Yields:
            Each Morpheme in the internal list.
        """
        for m in self._morphemes:
            yield m
    
    def __add__(self, other):
        if not isinstance(other, MorphemeSearch):
            raise TypeError
        return MorphemeSearch(morphemes=self._morphemes + other._morphemes)
    
    def append(self, new_morpheme):
        """
        Append a single Morpheme to the internal list of Morphemes.

        Parameters:
            new_morpheme (Morpheme): A Morpheme object to append.
        """
        if new_morpheme:
            if not isinstance(new_morpheme, Morpheme):
                raise TypeError("MorphemeSearch.append() requires a Morpheme object as an argument!")
            self._morphemes.append(new_morpheme)
    
    def extend(self, new_morphemes: list):
        """
        Extend the internal list of Morphemes with a new list of Morphemes.

        Parameters:
            new_morphemes (list of Morphemes): A list of Morpheme objects.
        """
        if new_morphemes:
            for m in new_morphemes:
                if not isinstance(m, Morpheme):
                    raise TypeError("MorphemeSearch.extend() requires a list of Morpheme objects as an argument!")
            self._morphemes.extend(new_morphemes)
    
    @property
    def as_ana(self):
        """
        Return a list of Analysis objects corresponding to the found Morphemes.

        Returns:
            list: A list of Analyses for each Morpheme.
        """
        return [m.back_ana for m in self._morphemes]
    
    @property
    def as_sent(self):
        """
        Return a list of Sentence objects corresponding to the found Morphemes.

        Returns:
            list: A list of Sentences for each Morpheme.
        """
        return [m.back_sent for m in self._morphemes]
    
    @property
    def as_glossing(self):
        """
        Return a list of glossed string representations of corresponding
        Sentence objects for each Morpheme.

        Returns:
            list: A list of glossed sentence strings.
        """
        return [m.back_sent.to_glossing() for m in self._morphemes]
    
    @property
    def morphemes(self):
        """
        Count and return the most common (morph_full, gloss) pairs.

        Returns:
            list of tuples: Each tuple is ((morph_full, gloss), count), sorted by frequency.
        """
        return Counter([(m.morph_full, m.gloss) for m in self._morphemes]).most_common()
    
    @property
    def morphs(self):
        """
        Count and return the most common morphs.

        Returns:
            list of tuples: Each tuple is (morph, count), sorted by frequency.
        """
        return Counter([m.morph_full for m in self._morphemes]).most_common()
    
    @property
    def glosses(self):
        """
        Count and return the most common glosses.

        Returns:
            list of tuples: Each tuple is (gloss, count), sorted by frequency.
        """
        return Counter([m.gloss for m in self._morphemes]).most_common()
    
    def to_df(self):
        """
        Return a pandas.DataFrame with Sentence objects for each Morpheme.

        Returns:
            pandas.DataFrame: A dataframe with glossed sentence strings.
        """
        token_ids = [m.back_token._id for m in self._morphemes]
        return pd.concat([self.as_sent[i].to_df(token_id = token_ids[i]) for i in range(len(self.as_sent))], ignore_index=True, sort=False)
    
    def regloss(self, gloss=None, pos=None):
        """
        Replace the gloss value of all Morphemes in the search results with a new value.

        Parameters:
            gloss (str): The new gloss string to apply to every Morpheme.
            pos (str): The new PoS string to apply to every Morpheme.
        """
        if gloss:
            for m in self._morphemes:
                m.gloss = gloss
        if pos:
            if pos == "":
                pos = None
            for m in self._morphemes:
                m.pos = pos


class MorphemeChainSearch():
    """
    A list-like class with results of Morpheme chain search.

    Properties:
        as_ana (list): List of corresponding Analysis objects for each Morpheme chain.
        as_sent (list): List of corresponding Sentence objects for each Morpheme chain.
        as_glossing (list): Glossed string representation of corresponding
            Sentence objects for each Morpheme chain.
        text_filter (dict or None): Dictionary with text filters applied in the search.

    Methods:
        __getitem__(index): Return the Morpheme chain at the given index.
        __repr__(): Formal string representation of the list of found Morpheme chains.
        __str__(): Informal string representation (same as __repr__).
        __len__(): Number of found Morpheme chains.
        __iter__(): Iterator over all Morpheme chains.
        append(new_chain): Add a single Morpheme chain to the list.
        extend(new_chains): Add multiple Morpheme chains to the list.
        regloss(gloss_index_string, pos_string): Replace the gloss of all Morpheme chains
            in the search results with the given analysis.
    """
    
    def __init__(self, chains=None, text_filter=None):
        """
        Initialize a MorphemeChainSearch object.

        Parameters:
            chains (list): A preset list of lists of Morpheme objects (defaults to an empty list).
                           Necessary for calls from other internal functions.
            text_filter (dict): Dictionary with text filters applied in the search (defaults to None).
        """
        self._chains = chains if chains else []
        self.text_filter = text_filter
    
    def __getitem__(self, index):
        """
        Return a Morpheme chain at the specified index.

        Parameters:
            index (int): The index of the Morpheme chain to retrieve.

        Returns:
            The Morpheme chain at the specified index, or None if the list is empty.
        """
        return self._chains[index] if self._chains else None
    
    def __repr__(self):
        """
        Return a formal string representation of the morpheme chains list.

        Returns:
            str: A string representation.
        """
        return "[" + ",\n ".join([repr(c) for c in self._chains]) + "]"
    
    def __str__(self):
        return repr(self)

    def __len__(self):
        """
        Return number of found morpheme chains.

        Returns:
            int: The number of morpheme chains.
        """
        return len(self._chains) if self._chains else 0
    
    def __iter__(self):
        """
        Enable iteration over the Morpheme chains.

        Yields:
            Each Morpheme chain in the internal list.
        """
        for c in self._chains:
            yield c
    
    def __add__(self, other):
        if not isinstance(other, MorphemeChainSearch):
            raise TypeError
        return MorphemeChainSearch(chains=self._chains + other._chains)
    
    def append(self, new_chain):
        """
        Append a new chain of Morphemes to the internal list of Morpheme chains.

        Parameters:
            new_chain (list, tuple): A chain of Morpheme objects to append.
        """
        if new_chain:
            if not (isinstance(new_chain, list) or isinstance(new_chain, tuple)):
                raise TypeError("MorphemeChainSearch.append() requires a list / tuple object as an argument!")
            self._chains.append(new_chain)
    
    def extend(self, new_chains):
        """
        Extend the internal list of Morpheme chains with a new list of MorphemeChains.

        Parameters:
            new_chains (list, tuple): A list of chains of Morpheme objects.
        """
        if new_chains:
            for c in new_chains:
                if not (isinstance(c, list) or isinstance(c, tuple)):
                    raise TypeError("MorphemeChainSearch.extend() requires a list of lists / tuples as an argument!")
            self._chains.extend(new_chains)
    
    @property
    def as_ana(self):
        """
        Return a list of Analysis objects corresponding to the found Morpheme chains.

        Returns:
            list: A list of Analyses for each Morpheme chain.
        """
        return [c[0].back_ana for c in self._chains]
    
    @property
    def as_sent(self):
        """
        Return a list of Sentence objects corresponding to the found Morpheme chains.

        Returns:
            list: A list of Sentences for each Morpheme chain.
        """
        return [c[0].back_sent for c in self._chains]
    
    @property
    def as_glossing(self):
        """
        Return a list of glossed string representations of corresponding
        Sentence objects for each Morpheme chain.

        Returns:
            list: A list of glossed sentence strings.
        """
        return [c[0].back_sent.to_glossing() for c in self._chains]
    
    @property
    def morphemechains(self):
        """
        Count and return the most common morpheme chains.

        Returns:
            list of tuples: Each tuple is (morpheme chain string, count), sorted by frequency.
        """
        return Counter([Analysis(c).gloss_index_string()[:-1] for c in self._chains]).most_common()
    
    @property
    def _is_homogeneous(self):
        """
        Check if all Morpheme chains have the same phonological content.

        Returns:
            bool: True if all Morpheme chains have the same phonological content.
        """
        morph_strings = set(["".join([m.morph for m in chain]).lower() for chain in self._chains])
        if len(morph_strings) > 1:
            return False
        return True
    
    def regloss(self, gloss_index_string: str, pos_string=None, force=False):
        """
        Replace the glossing of all Morpheme chains in the search results with a new analysis.

        Parameters:
            gloss_index_string (str): The new gloss index string to apply to every Morpheme chain,
                                        e.g. 'EMPH{ik=}-such{dis}'.
            pos_string (str or None): A string with POS values separated by hyphens, e.g. 'prt-pro'.
        """
        if (not self._is_homogeneous) and (not force):
            raise ValueError("Only MorphemeChainSearch with the same phonological content can be reglossed!")

        for res_chain in self._chains:
            res_chain_ids = [m._id for m in res_chain]
            if len(res_chain_ids)>0:
                cur_ana = res_chain[0].back_ana
                i = [m._id for m in cur_ana].index(res_chain_ids[0])
                
                new_morphemes = _parse_gloss_index_string(
                    gloss_index_string=gloss_index_string, pos_string=pos_string)
                for m in new_morphemes:
                    m.back_ana = cur_ana
                    m.back_token = res_chain[0].back_token
                    m.back_sent = res_chain[0].back_sent
                    m.back_text = res_chain[0].back_text
                
                for _id in res_chain_ids:
                    cur_ana.remove(_id=_id, renumerate=False)
                cur_ana.morphemes = cur_ana.morphemes[:i] + new_morphemes + cur_ana.morphemes[i:]
                cur_ana.numerate()


class TokenSearch():
    """
    A list-like class with results of Token search.

    Properties:
        as_ana (list): List of lists of corresponding Analysis objects for each Token.
        as_sent (list): List of corresponding Sentence objects for each Token.
        as_glossing (list): Glossed string representation of corresponding
            Sentence objects for each Token.
        tokens (list): Frequency-sorted list of Tokens.
        text_filter (dict or None): Dictionary with text filters applied in the search.

    Methods:
        __getitem__(index): Return the Token at the given index.
        __repr__(): Formal string representation of the Token list.
        __str__(): Informal string representation (same as __repr__).
        __len__(): Number of found Tokens.
        __iter__(): Iterator over all Tokens.
        append(new_token): Add a single Token to the list.
        extend(new_tokens): Add multiple Tokens to the list.
        regloss(new): Replace the glossing of all Tokens in the search results with the given analysis.
    """
    
    def __init__(self, tokens=None, text_filter=None):
        """
        Initialize a TokenSearch object.

        Parameters:
            tokens (list): A preset list of Token objects (defaults to an empty list).
                           Necessary for calls from other internal functions.
            text_filter (dict): Dictionary with text filters applied in the search (defaults to None).
        """
        self._tokens = tokens if tokens else []
        self.text_filter = text_filter
    
    def __getitem__(self, index):
        """
        Return a Token at the specified index.

        Parameters:
            index (int): The index of the Token to retrieve.

        Returns:
            The Token at the specified index, or None if the list is empty.
        """
        return self._tokens[index] if self._tokens else None
    
    def __repr__(self):
        """
        Return a formal string representation of the Tokens list.

        Returns:
            str: A string representation.
        """
        return "[" + ",\n ".join([repr(t) for t in self._tokens]) + "]"
    
    def __str__(self):
        return repr(self)
    
    def __len__(self):
        """
        Return number of found Tokens.

        Returns:
            int: The number of Tokens.
        """
        return len(self._tokens) if self._tokens else 0
    
    def __iter__(self):
        """
        Enable iteration over the Tokens.

        Yields:
            Each Token in the internal list.
        """
        for t in self._tokens:
            yield t
    
    def append(self, new_token):
        """
        Append a single Token to the internal list of Tokens.

        Parameters:
            new_token (Token): A Token object to append.
        """
        if new_token:
            if not isinstance(new_token, Token):
                raise TypeError("TokenSearch.append() requires a Token object as an argument!")
            self._tokens.append(new_token)
    
    def extend(self, new_tokens: list):
        """
        Extend the internal list of Tokens with a new list of Tokens.

        Parameters:
            new_tokens (list of Tokens): A list of Token objects.
        """
        if new_tokens:
            for t in new_tokens:
                if not isinstance(t, Token):
                    raise TypeError("TokenSearch.extend() requires a list of Token objects as an argument!")
            self._tokens.extend(new_tokens)
    
    @property
    def as_ana(self):
        """
        Return lists of Analysis objects corresponding to the found Tokens.

        Returns:
            list: A list of lists of Analyses for each Token.
        """
        return [t.ana for t in self._tokens]
    
    @property
    def as_sent(self):
        """
        Return a list of Sentence objects corresponding to the found Tokens.

        Returns:
            list: A list of Sentences for each Token.
        """
        return [t.back_sent for t in self._tokens]
    
    @property
    def as_glossing(self):
        """
        Return a list of glossed string representations of corresponding
        Sentence objects for each Token.

        Returns:
            list: A list of glossed sentence strings.
        """
        return [t.back_sent.to_glossing() for t in self._tokens]
    
    @property
    def tokens(self):
        """
        Count and return the most common token strings (Token.token).

        Returns:
            list of tuples: Each tuple is (token_string, count), sorted by frequency.
        """
        return Counter([t.token for t in self._tokens]).most_common()
    
    @property
    def analyses(self):
        """
        Count and return the most common analyses for these Tokens.
        Raises ValueError if some Tokens have more than one analysis.

        Returns:
            list of tuples: Each tuple is (Analysis, count), sorted by frequency.
        """
        spisok = []
        for t in self._tokens:
            if t.ana:
                if len(t.ana)>1:
                    raise ValueError("TokenSearch.analyses requires that all Tokens have only one analysis!")
                if len(t.ana)==1 and t.ana[0]:
                    spisok.append(t.ana[0].view())
        
        return Counter(spisok).most_common()
    
    def regloss(self, gloss_index_string: str, pos_string=None):
        """
        Replace the glossing of all Tokens in the search results with a new analysis.

        Parameters:
            gloss_index_string (str): The new gloss index string to apply to every Token,
                e.g. 'EMPH{ik=}-such{dis}'.
            pos_string (str or None): A string with POS values separated by hyphens, e.g. 'prt-pro'.
        """
        chains_to_regloss = MorphemeChainSearch()
        for t in self._tokens:
            if t.ana:
                if len(t.ana)>1:
                    t.ana = [t.ana[0]]
            else:
                t.ana = [Analysis(morph_string=t.token)]
            chains_to_regloss.append(t.ana[0].morphemes)
        chains_to_regloss.regloss(gloss_index_string=gloss_index_string, pos_string=pos_string)
    
    def to_df(self):
        """
        Return a pandas.DataFrame with Sentence objects for each Token.

        Returns:
            pandas.DataFrame: A dataframe with glossed sentence strings.
        """
        token_ids = [t._id for t in self._tokens]
        return pd.concat([self.as_sent[i].to_df(token_id = token_ids[i]) for i in range(len(self.as_sent))], ignore_index=True, sort=False)


class SentenceSearch():
    """
    A list-like class with results of Sentence search.

    Properties:
        as_glossing (list): Glossed string representation of Sentences.
        text_filter (dict or None): Dictionary with text filters applied in the search.

    Methods:
        __getitem__(index): Return the Sentence at the given index.
        __repr__(): Formal string representation of the Sentence list.
        __str__(): Informal string representation (same as __repr__).
        __len__(): Number of found Sentences.
        __iter__(): Iterator over all Sentences.
        append(new_sentence): Add a single Sentence to the list.
        extend(new_sentences): Add multiple Sentences to the list.
    """
    
    def __init__(self, sentences=None, text_filter=None):
        """
        Initialize a SentenceSearch object.

        Parameters:
            sentences (list): A preset list of Sentence objects (defaults to an empty list).
                           Necessary for calls from other internal functions.
            text_filter (dict): Dictionary with text filters applied in the search (defaults to None).
        """
        self._sentences = sentences if sentences else []
        self.text_filter = text_filter
    
    def __getitem__(self, index):
        """
        Return a Sentence at the specified index.

        Parameters:
            index (int): The index of the Sentence to retrieve.

        Returns:
            The Sentence at the specified index, or None if the list is empty.
        """
        return self._sentences[index] if self._sentences else None
    
    def __repr__(self):
        """
        Return a formal string representation of the Sentences list.

        Returns:
            str: A string representation.
        """
        return "[" + ",\n ".join([repr(t) for t in self._sentences]) + "]"
    
    def __str__(self):
        return repr(self)
    
    def __len__(self):
        """
        Return number of found Sentences.

        Returns:
            int: The number of Sentences.
        """
        return len(self._sentences) if self._sentences else 0
    
    def __iter__(self):
        """
        Enable iteration over the Sentences.

        Yields:
            Each Sentence in the internal list.
        """
        for t in self._sentences:
            yield t
    
    def append(self, new_sentence):
        """
        Append a single Sentence to the internal list of Sentences.

        Parameters:
            new_sentence (Sentence): A Sentence object to append.
        """
        if new_sentence:
            if not isinstance(new_sentence, Sentence):
                raise TypeError("SentenceSearch.append() requires a Sentence object as an argument!")
            self._sentences.append(new_sentence)
    
    def extend(self, new_sentences: list):
        """
        Extend the internal list of Sentences with a new list of Sentences.

        Parameters:
            new_sentences (list of Sentences): A list of Sentence objects.
        """
        if new_sentences:
            for t in new_sentences:
                if not isinstance(t, Sentence):
                    raise TypeError("SentenceSearch.extend() requires a list of Sentence objects as an argument!")
            self._sentences.extend(new_sentences)
    
    @property
    def as_glossing(self):
        """
        Return a list of glossed string representations of corresponding
        Sentence objects for each Sentence.

        Returns:
            list: A list of glossed sentence strings.
        """
        return [s.to_glossing() for s in self._sentences]

    def to_df(self):
        """
        Return a pandas.DataFrame with Sentences.

        Returns:
            pandas.DataFrame: A dataframe with glossed sentence strings.
        """
        return pd.concat([sent.to_df() for sent in self._sentences], ignore_index=True, sort=False)


class Morpheme():
    """
    Represents a single morpheme with morph (phonological form) and gloss (semantic meaning).

    Attributes:
        morph (str): Phonological form of the morpheme (without morpheme separators).
        gloss (str): Gloss (semantic label) of the morpheme.
        pos (str or None): Part-of-speech tag.
        morph_aid (int or None): EAF annotation id for the morph.
        gloss_aid (int or None): EAF annotation id for the gloss.
        pos_aid (int or None): EAF annotation id for the POS tag.
        _id (int or None): Identifier of the Morpheme’s position in the source Analysis.
        back_ana (Analysis or None): Back-reference to the source Analysis object.
        back_token (Token or None): Back-reference to the source Token object.
        back_sent (Sentence or None): Back-reference to the source Sentence object.
        back_text (Text or None): Back-reference to the source Text object.

    Properties:
        morph_full (str): Full morph including morpheme separator symbols.
        gloss_full (str): Full gloss including morpheme separator symbols.
        id (int or None): Identifier of the Morpheme’s position in the source analysis.
        is_lemma (bool): True if the gloss is a lemma (contains lowercase letters).
        is_grammeme (bool): True if the gloss is a grammeme tag (contains ONLY uppercase
            letters, digits and punctuation).

    Methods:
        __repr__(): Return a formal string representation of the Morpheme.
        __str__(): Informal string representation (same as __repr__).
        view(mode): Return a formatted string in the `gloss{morph}` format,
            e.g. 'EMPH{ik=}-such{dis}'.
        convert_orthography(converter, target, *kwargs):
            Convert the orthography of the morph using a converter.
    """
    
    def __init__(self, morph, gloss=None, morph_aid=None, gloss_aid=None,
                 pos=None, pos_aid=None, _id=None, back_ana=None, back_token=None,
                 back_sent=None, back_text=None):
        """
        Initialize a Morpheme object.

        Parameters:
            morph (str): Phonological form of the morpheme.
            gloss (str): Gloss (semantic label) of the morpheme (defaults to '_').
            morph_aid, gloss_aid, pos_aid: EAF annotation IDs for morph, gloss, and POS tag
                (default: None).
            pos (str): Part-of-speech tag (default: None).
            _id (int): Identifier of the Morpheme’s position in the source Analysis (default: None).
            back_ana (Analysis): Source Analysis object (default: None).
            back_token (Token): Source Token object (default: None).
            back_sent (Sentence): Source Sentence object (default: None).
            back_text (Text): Source Text object (default: None).
        """

        self._prefix, self._morph, self._suffix = _clean_morpheme(morph if morph else "_")
        _, self._gloss, _ = _clean_morpheme(gloss if gloss else "_")
        
        self.pos = pos
        
        self.morph_aid = morph_aid
        self.gloss_aid = gloss_aid
        self.pos_aid = pos_aid
        
        self._id = _id
        self.back_ana = back_ana
        self.back_token = back_token
        self.back_sent = back_sent
        self.back_text = back_text
    
    @property
    def morph(self):
        return self._morph

    @morph.setter
    def morph(self, value):
        self._prefix, self._morph, self._suffix = _clean_morpheme(value or "_")

    @property
    def gloss(self):
        return self._gloss

    @gloss.setter
    def gloss(self, value):
        _, self._gloss, _ = _clean_morpheme(value or "_")

    def __repr__(self):
        """
        Return a string representation of the morpheme in the 'dirty' mode of the method Morpheme.view().

        Returns:
            str: A string in the `gloss{morph}` format, e.g. 'EMPH{ik=}-such{dis}'.
        """
        return 'Morpheme("' + self.view(mode="dirty") + '")'
    
    def __str__(self):
        return repr(self)
    
    @property
    def morph_full(self):
        """
        Return the full morph including morpheme separator symbols.

        Returns:
            str: Full morph with morpheme separators.
        """
        return self._prefix + self._morph + self._suffix
    
    @property
    def gloss_full(self):
        """
        Return the full gloss including morpheme separator symbols.

        Returns:
            str: Full gloss with morpheme separators.
        """
        return self._prefix + self.gloss + self._suffix
    
    @property
    def pos_full(self):
        """
        Return the full POS tag including morpheme separator symbols.

        Returns:
            str: Full POS tag with morpheme separators.
        """
        return self._prefix + self.pos + self._suffix if self.pos else ""
    
    @property
    def id(self):
        """
        Return the identifier of the Morpheme’s position in the source analysis.

        Returns:
            int or None: The Morpheme’s identifier number.
        """
        return self._id
    
    @property
    def is_lemma(self):
        """
        Determine if the gloss appears to be a lemma, i.e. a lexical label (contains lowercase letters).

        Returns:
            bool: True if gloss contains lowercase letters, else False.
        """
        return False if re.fullmatch(".*[a-z]+.*", self.gloss) is None else True
    
    @property
    def is_grammeme(self):
        """
        Determine if the gloss appears to be a grammeme tag (contains ONLY uppercase
        letters, digits and punctuation).

        Returns:
            bool: True if gloss fully matches uppercase/digits/punctuation pattern, else False.
        """
        return False if re.fullmatch("[A-Z0-9/:_\.\[\]\(\)]+", self.gloss) is None else True

    @property
    def is_first(self):
        """
        Determine if the Morpheme is the first in the Analysis.

        Returns:
            bool: True if the Morpheme is the first in the Analysis,
                  False if it is not the first or there is no associated Analysis.
        """
        return True if self.back_ana and self._id == 1 else False
    
    @property
    def is_last(self):
        """
        Determine if the Morpheme is the last in the Analysis.

        Returns:
            bool: True if the Morpheme is the last in the Analysis,
                  False if it is not the last or there is no associated Analysis.
        """
        return True if self.back_ana and self._id == len(self.back_ana) else False
    
    def view(self, mode="clean", pos=True):
        """
        Return a formatted string representation of the Morpheme.

        Parameters:
            mode (str): Either 'clean' (without morpheme separators) or 'dirty'
                        (with morpheme separators).
            pos (bool): Include POS values or not.

        Returns:
            str: Formatted string in the `gloss{morph}` format, e.g. 'EMPH{ik=}-such{dis}'.
        """
        if pos:
            pos = "{" + self.pos + "}" if self.pos else ""
        else:
            pos = ""
        if mode == "clean":
            return self.gloss + "{" + self.morph + "}" + pos
        return self.gloss + "{" + self.morph_full + "}" + pos
    
    def convert_orthography(self, converter=None, target="orig", base=None, *kwargs):
        """
        Convert the morph of the Morpheme using a Converter in-place.

        Parameters:
            converter (Converter or None): An external Converter object
                (if None, a new Converter is created).
            target (str or None): Target orthography, one of the following options:
                'orig', 'cyr', 'lat', 'ipa' (default: 'orig').
            *kwargs: Additional arguments passed to the converter constructor.
        """
        if converter is None:
            converter = cv.Converter(target=target, base=base, *kwargs)
        self.morph = converter.convert(self.morph).text


def _parse_gloss_index_string(gloss_index_string: str, pos_string=None):
    """
    Parse a gloss index string and return a list of Morpheme objects.

    Parameters:
        gloss_index_string (str): Gloss index string to parse.
        pos_string (str or None): A string with POS values separated by hyphens, e.g. 'prt-pro'.

    Returns:
        list: List of Morphemes.
    """
    
    if gloss_index_string == "":
        return []
    
    true_result = []
    
    # PATTERN WITHOUT POS VALUES

    n = len(re.findall("{", gloss_index_string))
    
    if pos_string:
        pos_string = re.split("\-", pos_string)
        if len(pos_string) != n:
            raise ValueError("The `gloss_index_string` and `pos_string` do not match!")

    query = "(([^{}]+){([^{}]+)}-*)" + "(([^{}]+){([^{}]+)}-*)" * (n-1)
    result = re.fullmatch(query, gloss_index_string)
    if not result is None:
        result = result.groups()
        for i in range(1, len(result), 3):
            gloss, morph = result[i], result[i+1]
            pos = None if pos_string is None else pos_string[i//3]
            true_result.append(Morpheme(
                        gloss=gloss, morph=morph, pos = pos))
    
    else:
        # PATTERN WITH POS VALUES

        n //= 2

        query = "(([^{}]+){([^{}]+)}{([^{}]+)}-*)" + "(([^{}]+){([^{}]+)}{([^{}]+)}-*)" * (n-1)
        result = re.fullmatch(query, gloss_index_string)

        if result is None:
            raise ValueError("gloss_index_string has incorrect syntax!")
        else:
            result = result.groups()
            for i in range(1, len(result), 4):
                gloss, morph, pos = result[i], result[i+1], result[i+2]
                true_result.append(Morpheme(
                    gloss=gloss, morph=morph, pos = pos))
    
    
    return true_result


class Analysis():
    """
    Represents a morphological analysis consisting of a list of Morpheme objects.
    
    Can be initialized with one of the following arguments:
    - List of Morphemes as `morphemes`;
    - Morph string as `morph_string` (optionally, with `gloss_string`);
    - String in the gloss index format as `gloss_index_string`.

    Attributes:
        morphemes (list): A list of Morpheme objects that make up the Analysis.
        _id (int or None): Identifier of the Analysis’ number among Analyses of the source Token.
        back_token (Token or None): Back-reference to the source Token object.
        back_sent (Sentence or None): Back-reference to the source Sentence object.
        back_text (Text or None): Back-reference to the source Text object.

    Properties:
        id (int or None): Identifier of the Analysis’ number among Analyses of the source Token.
        morph_string (str): String of morphs, e.g. 'ik=dis'.
        gloss_string (str): String of glosses, e.g. 'EMPH=such'.
        pos_string (str): String of POS tags, e.g. 'prt=pro'.

    Methods:
        __getitem__(index): Return the Morpheme at a given index.
        __repr__(): Return a formal string representation of the Analysis in the `gloss{morph}` format.
        __str__(): Informal string representation (same as __repr__).
        __len__(): Return the number of morphemes in the Analysis.
        gloss_index_string (str): String of morphs and glosses
            in the strict `gloss_index` format, e.g. 'EMPH{ik=}-such{dis}-'.
        view(): String view of the Analysis in the `gloss{morph}` format.
        numerate(): Assign sequential ids to all Morphemes.
        remove(_id, safe=False, renumerate=True): Remove a Morpheme from the Analysis by id.
    """
    
    def __init__(self, morphemes=None,
                 gloss_string=None, morph_string=None, gloss_index_string=None,
                 pos_string=None, _id=None, back_token=None, back_sent=None, back_text=None):
        """
        Initialize an Analysis object from Morphemes or formatted strings.
        
        Provide one of the following:
        - List of Morphemes as `morphemes`;
        - String as `morph_string` (optionally, with `gloss_string`);
        - String as `gloss_index_string`.

        Parameters:
            morphemes (list): list of Morpheme objects (default: None).
            gloss_string (str): String of glosses separated by morpheme separators (default: None).
            morph_string (str): String of morphs separated by morpheme separators (default: None).
            gloss_index_string (str): String of morphs and glosses
                in the strict `gloss_index` format, e.g. 'EMPH{ik=}-such{dis}-' (default: None).
            pos_string (str): String of POS tags separated by hyphens, e.g. 'prt-pro'.
            _id (int or None): Identifier of the Analysis’ number among Analyses of the source Token
                (default: None).
            back_token (Token): Source Token object (default: None).
            back_sent (Sentence): Source Sentence object (default: None).
            back_text (Text): Source Text object (default: None).
        """
        
        self.morphemes = []
        
        if morphemes and not morph_string:
            if type(morphemes) in (tuple, list):
                self.morphemes = morphemes
            try:
                _gloss_string = _join_morpheme_strings([m.gloss for m in morphemes])
            except:
                raise TypeError
            self._morph_string = _join_morpheme_strings([m.morph for m in morphemes])
        
        elif morph_string and not morphemes:
            
            try:
                if gloss_index_string:
                    self.morphemes = _parse_gloss_index_string(
                        gloss_index_string=gloss_index_string, pos_string=pos_string)
                
                else:
                    lmorphs = re.split("[-=][-=]*", morph_string)
                    lglosses = None
                    
                    if gloss_string:
                        lglosses = re.split("[-=][-=]*", gloss_string)
                        if len(lglosses) != len(lmorphs):
                            raise ValueError(f"The number of morphs in 'morph_string' and the number of glosses in 'gloss_string' must be the same!\nProblem with token: '{morph_string}' [{gloss_string}]")
                        for i in range(len(lmorphs)):
                            self.morphemes.append(Morpheme(morph=lmorphs[i], gloss=lglosses[i], pos=pos_string[i] if pos_string else None))
                    else:
                        for i in range(len(lmorphs)):
                            self.morphemes.append(Morpheme(morph=lmorphs[i], pos=pos_string[i] if pos_string else None))
            except ValueError:
                warnings.warn(message=f"The token '{morph_string}' [{gloss_string}] had incorrect morphemic division and was not imported correctly.", stacklevel=4)
            if not gloss_string:
                pass
        
        elif morphemes and morph_string:
            raise ValueError("An Analysis instance must receive either a list of Morpheme objects as the argument 'morphemes' OR the argument 'morph_string', but not both!")
        
        self._id = _id
        self.back_token = back_token
        self.back_sent = back_sent
        self.back_text = back_text
        
        if gloss_index_string and not morphemes:
            self.morphemes = _parse_gloss_index_string(
                gloss_index_string=gloss_index_string, pos_string=pos_string)
            for m in self.morphemes:
                m.back_ana = self
                m.back_token = self.back_token
                m.back_sent = self.back_sent
                m.back_text = self.back_text
    
    def __getitem__(self, index):
        """
        Return the Morpheme at the specified index in the Analysis.

        Parameters:
            index (int): The index of the Morpheme to retrieve.

        Returns:
            Morpheme: The Morpheme object at the specified index.
        """
        return self.morphemes[index] if self.morphemes else None

    def get_id(self, id):
        for m in self.morphemes:
            if m._id == id:
                return m
    
    def __repr__(self):
        """
        Return a formal string representation of the Analysis.

        Returns:
            str: A string representation.
        """
        v = self.view()
        return "Analysis(" + (f'"{v}"' if v else '') + ")"
    
    def __str__(self):
        return repr(self)

    def __len__(self):
        """
        Return the number of Morphemes in the Analysis.

        Returns:
            int: The number of Morphemes.
        """
        return len(self.morphemes) if self.morphemes else 0
    
    def view(self, pos=True):
        """
        Return a full string showing the Analysis in the `gloss{morph}` format.

        Returns:
            str or None: Joined string of morphemes (or None if the Analysis has no Morphemes).
        """
        return "-".join([m.view(mode="dirty", pos=pos) for m in self.morphemes]) if self.morphemes else None
    
    @property
    def id(self):
        """
        Return the identifier of the Analysis’ number in the list of Analyses in the source Token.

        Returns:
            int or None: The Analysis’ identifier number.
        """
        return self._id
    
    @property
    def morph_string(self):
        """
        Return the string of morphs separated by morpheme separators.

        Returns:
            str: String of morphs with morpheme separators.
        """
        res = "-".join([m.morph_full for m in self.morphemes])
        res = re.sub("=-|-=", "=", re.sub("-+", "-", res))
        return res
    
    @property
    def gloss_string(self):
        """
        Return the string of glosses separated by morpheme separators.

        Returns:
            str: String of glosses with morpheme separators.
        """
        res = "-".join([m.gloss_full for m in self.morphemes])
        res = re.sub("=-|-=", "=", re.sub("-+", "-", res))
        return res
    
    def gloss_index_string(self, pos=True):
        """
        Return the string in the strict `gloss_index` format, e.g. 'EMPH{ik=}-such{dis}-'.
        
        Parameters:
            pos: Include POS values or not.

        Returns:
            str: String in the strict `gloss_index` format.
        """
        return "-".join([m.view(mode="dirty", pos=pos) for m in self.morphemes]) + "-"
    
    @property
    def pos_string(self):
        """
        Return the string of POS tags separated by morpheme separators.

        Returns:
            str: String of POS tags with morpheme separators.
        """
        res = "-".join([m.pos_full for m in self.morphemes])
        res = re.sub("=-|-=", "=", re.sub("-+", "-", res))
        return res
    
    def numerate(self):
        """
        Assign sequential ids to all Morphemes in the Analysis.
        """
        if self.morphemes:
            for i in range(len(self.morphemes)):
                self.morphemes[i]._id = i+1
    
    def remove(self, _id, safe=False, renumerate=True):
        """
        Remove a Morpheme from the Analysis by its id.

        Parameters:
            _id (int): The id of the Morpheme to remove.
            safe (bool): If True, silently ignore if id not found.
            renumerate (bool): If True, reassign ids after removal.
        """
        ana_ids = [m._id for m in self.morphemes]
        if safe:
            if _id not in ana_ids:
                return
        del self.morphemes[ana_ids.index(_id)]
        if renumerate:
            self.numerate()


def _find_consecutive_sequences(lists):
    if not lists:
        return []
    result, first_list = [], lists[0]
    ids_by_level = [set(obj._id for obj in sublist) for sublist in lists]
    for obj in first_list:
        x = obj._id
        sequence, valid = [obj], True
        for i in range(1, len(lists)):
            if (x + i) in ids_by_level[i]:
                match = next(o for o in lists[i] if o._id == x + i)
                sequence.append(match)
            else:
                valid = False
                break
        if valid:
            result.append(sequence)
    return result


class Token():
    """
    Represents a text token, optionally with a list of Analysis objects.

    Attributes:
        token (str): String with token’s textual representation (e.g. 'What?').
        ana (list or None): List with Analyses for the token (defaults to empty list).
        token_aid (int or None): EAF annotation id for the token.
        _id (int or None): Identifier of the Token’s number among Tokens of the source Sentence.
        back_sent (Sentence or None): Back-reference to the source Sentence object.
        back_text (Text or None): Back-reference to the source Text object.
        tsakorpus_features (dict or None): Optional tecnhical features for tsakorpus.

    Properties:
        id (int): Identifier of the Token’s number among Tokens of the source Sentence.
        morph_string (str): String of morphs, if the Token has one analysis only.
        gloss_string (str): String of glosses, if the Token has one analysis only.
        pos_string (str): String of POS tags, if the Token has one analysis only.
        multiple_ana (bool): True if there is more than one Analysis in the Token.

    Methods:
        __getitem__(index): Return the Analysis at a given index.
        __repr__(): Return a formal string representation of the Token.
        __str__(): Informal string representation (same as __repr__).
        gloss_index_string (str): String of morphs and glosses in the strict `gloss_index`
            format, e.g. 'EMPH{ik=}-such{dis}-', if the Token has one analysis only.
        numerate(): Assign sequential ids to all Analyses.
        remove(_id, safe=False, renumerate=True): Remove an Analysis from the Token by id.
        clean(): Clean whitespace in the beginning and in the end
            of the string and collapse double spaces to single spaces.
        search_morpheme(gloss=None, morph=None, pos=None, gloss_type=None, full=False):
            Search for a Morpheme in the Token.
        search_morphemechain(gloss_index_string=None, pos_string=None, _query=None):
            Search for a Morpheme chain in the Token.
        convert_orthography(converter, target, *kwargs):
            Convert the orthography of all morphs using a converter.
    """
    
    def __init__(self, token, ana=None, token_aid=None,
                 _id=None, back_sent=None, back_text=None,
                 tsakorpus_features=None):
        """
        Initialize a Token object.
        
        Provide a `token` string and, optionally, a list of Analyses as `ana`.

        Parameters:
            token (str): String with token’s textual representation (e.g. 'What?').
            ana (list): List with Analyses for the token (defaults to empty list).
            token_aid (int): EAF annotation id for the token (default: None).
            _id (int): Identifier of the Token’s number among Tokens of the source Sentence (default: None).
            back_sent (Sentence): Back-reference to the source Sentence object (default: None).
            back_text (Text): Back-reference to the source Text object (default: None).
        """
        
        self.token = token
        if ana is None:
            self.ana = []
        else:
            if isinstance(ana, list):
                for a in ana:
                    if a:
                        if not isinstance(a, Analysis):
                            raise TypeError(f"Token.ana must be a list of Analyses and not of `{type(a)}`!")
                self.ana = ana
            elif isinstance(ana, Analysis):
                self.ana = [ana]
            else:
                raise TypeError(f"Token.ana must be a list of Analyses and not `{type(ana)}`!")
        self.token_aid = token_aid
        
        self._id = _id
        self.back_sent = back_sent
        self.back_text = back_text
        
        self.tsakorpus_features = tsakorpus_features
        if tsakorpus_features is None:
            self.tsakorpus_features = {}
    
    def __getitem__(self, index):
        """
        Return the Analysis at the specified index in the list of Analyses in the Token.

        Parameters:
            index (int): The index of the Analysis to retrieve.

        Returns:
            Analysis: The Analysis object at the specified index.
        """
        return self.ana[index]

    def get_id(self, id):
        for a in self.ana:
            if a._id == id:
                return a
    
    def __repr__(self):
        """
        Return a formal string representation of the Token.

        Returns:
            str: A string representation.
        """
        return 'Token("' + self.token + '")'

    def __str__(self):
        return repr(self)
    
    @property
    def id(self):
        """
        Return the identifier of the Token’s number in the source Sentence.

        Returns:
            int or None: The Token’s identifier number.
        """
        return self._id
    
    @property
    def wtype(self):
        """
        Determine if the Token is a `word` or a `punct` (fot tsakorpus).

        Returns:
            str: The Token’s wtype.
        """
        if _ispunct(self.token):
            return "punct"
        return "word"
    
    @property
    def morph_string(self):
        """
        Return the string of morphs separated by morpheme separators.
            Recommended to use only if the Token has a single Analysis.

        Returns:
            str or None: String of morphs with morpheme separators (None if there are no Analyses).
        """
        if self.multiple_ana:
            warnings.warn(message=f"Multiple analyses in the sentence {self}!\n", stacklevel=4)
            return "?"
        if len(self.ana) == 0:
            return "_"
        return self[0].morph_string if len(self.ana) > 0 and self[0] else "_"
    
    @property
    def gloss_string(self):
        """
        Return the string of glosses separated by morpheme separators.
            Recommended to use only if the Token has a single Analysis.

        Returns:
            str or None: String of glosses with morpheme separators (None if there are no Analyses).
        """
        if self.multiple_ana:
            warnings.warn(message=f"Multiple analyses in the sentence {self}!\n", stacklevel=4)
            return "?"
        if len(self.ana) == 0:
            return "_"
        return self[0].gloss_string if len(self.ana) > 0 and self[0] else "_"
    
    def gloss_index_string(self, pos=True):
        """
        Return the string in the strict `gloss_index` format, e.g. 'EMPH{ik=}-such{dis}-'.
            Recommended to use only if the Token has a single Analysis.
        
        Parameters:
            pos: Include POS values or not.

        Returns:
            str or None: String in the strict `gloss_index` format (None if there are no Analyses).
        """
        if self.multiple_ana:
            warnings.warn(message=f"Multiple analyses in the sentence {self}!\n", stacklevel=4)
            return "?"
        if len(self.ana) == 0:
            return "_"
        return self[0].gloss_index_string(pos=pos) if len(self.ana) > 0 and self[0] else "_"
    
    @property
    def pos_string(self):
        """
        Return the string of POS tags separated by morpheme separators.
            Recommended to use only if the Token has a single Analysis.

        Returns:
            str or None: String of POS tags with morpheme separators (None if there are no Analyses).
        """
        if self.multiple_ana:
            warnings.warn(message=f"Multiple analyses in the sentence {self}!\n", stacklevel=4)
            return "?"
        if len(self.ana) == 0:
            return "_"
        return self[0].pos_string if len(self.ana) > 0 and self[0] else "_"
    
    @property
    def multiple_ana(self):
        """
        Determine if the Token has multiple Analyses.

        Returns:
            bool: True, if the Token has more than one Analysis.
        """
        return len(self.ana) > 1
    
    def morphemization_is_correct(self):
        """
        Check if Token.text is a “sum” of all its morphemes.

        Returns:
            bool or str: True if Token.text corresponds to its morphemes,
                otherwise False.
        """
        if len(self.ana) == 0 or self.ana[0] is None:
            return True
        token_text = depunct(re.sub("-", "=", self.token))
        morphline = self.ana[0].morph_string
        morphline = re.sub("-", "", morphline)

        if not (token_text == morphline or token_text.lower() == morphline):
            return False
        return True
        
        """
        token_seps = re.findall("[\-=]", token_text)
        morph_seps = re.findall("[\-=]", self.ana[0].morph_string)
        
        if len(token_seps) != len(morph_seps):
            return False
        
        for i in range(len(token_seps)):
            if not re.fullmatch(token_seps[i], morph_seps[i]):
                return False
        
        glued_morphs = "".join([m.morph for m in self.ana[0]])
        if include_hyphens:
            glued_morphs = re.sub("-", "", glued_morphs)
        if include_equals:
            glued_morphs = re.sub("=", "", glued_morphs)
        if glued_morphs not in (token_text, token_text.lower()):
            warnings.warn(message=f"The token text does not correspond to the list of morphemes!\n", stacklevel=4)
            return False
        """
    
    def numerate(self):
        """
        Assign sequential ids to all Analyses in the Token and to all Morphemes in each Analysis.
        """
        if self.ana:
            for i in range(len(self.ana)):
                if self.ana[i]:
                    self.ana[i]._id = i+1
                    self.ana[i].numerate()
    
    def remove(self, i, safe=False):
        """
        Remove an Analysis from the list of Analyses in the Token by its id.

        Parameters:
            _id (int): The id of the Analysis to remove.
            safe (bool): If True, silently ignore if id not found.
            renumerate (bool): If True, reassign ids after removal.
        """
        if safe and ((not self.ana) or (len(self.ana)<=i)):
            return
        if self.ana[i] and self.ana[i].morphemes:
            for j in reversed(range(len(self.ana[i].morphemes))):
                self.ana[i].remove(_id=self.ana[i][j]._id, safe=True)
        del self.ana[i]
        self.numerate()
    
    def clean(self):
        """
        Clean whitespace in the beginning and in the end
            of the string and collapse double spaces to single spaces.
        """
        self.token = _clean(self.token)
    
    def search_morpheme(self, gloss=None, morph=None, pos=None, gloss_type=None, full=False):
        """
        Search for a Morpheme in the Token’s Analyses.
        Regular expressions are allowed.
        
        Parameters:
            gloss (str or None): Gloss value to be a condition for the search.
                If None, all glosses will be found (default: None).
            morph (str or None): Morph value to be a condition for the search.
                If None, all morphs will be found (default: None).
            pos (str or None): POS tag value to be a condition for the search.
                If None, all POS tags will be found (default: None).
            gloss_type (str or None): gloss type to be a condition for the search.
                One of the following: 'grammeme', 'lemma' or 'other'.
                If None, glosses of all types will be found (default: None).
            full (bool): if True, morpheme separators will be considered
                during the search (default: False).
        
        Returns:
            MorphemeSearch: Results of the search.
        """
        found_morphemes = []
        if self.ana:
            for a in self.ana:
                if a and a.morphemes:
                    for m in a.morphemes:
                        if m:
                            if full:
                                if morph is not None and re.fullmatch(morph, m.morph_full, flags=re.IGNORECASE) is None:
                                    continue
                            else:
                                if morph is not None and re.fullmatch(morph, m.morph, flags=re.IGNORECASE) is None:
                                    continue
                            if gloss is not None and re.fullmatch(gloss, m.gloss) is None:
                                continue
                            if (gloss_type == "grammeme") and (not m.is_grammeme):
                                continue
                            elif (gloss_type == "lemma") and (not m.is_lemma):
                                continue
                            elif (gloss_type == "other") and (m.is_lemma or m.is_grammeme):
                                continue
                            if pos is not None:
                                if pos == "":
                                    if not m.pos is None:
                                        continue
                                elif (m.pos is None) or (re.fullmatch(pos, m.pos) is None):
                                    continue
                            found_morphemes.append(m)
        return MorphemeSearch(morphemes=found_morphemes)
    
    def search_morphemechain(self, gloss_index_string=None, pos_string=None, _query=None):
        """
        Search for a sequence of Morphemes in the Token’s Analyses.
        Regular expressions are allowed.
        
        Parameters:
            gloss_index_string (str or None): String in the `gloss{morph}` format,
                containing morphemes to be found. If None, all chains will be found (default: None).
            pos_string  (str or None): String with POS tags separated by hyphens (default: None).
            _query (list or None): A preset list of Morpheme objects (defaults to an empty list).
                Necessary for calls from other internal functions (default: None).
        
        Returns:
            MorphemeChainSearch: Results of the search.
        """
        results = []
        if _query is None:
            _query = Analysis(gloss_index_string = gloss_index_string, pos_string=pos_string)

        for i in range(len(_query)):
            results.append(self.search_morpheme(
                gloss=_query[i].gloss, morph=_query[i].morph, pos=_query[i].pos))
        return MorphemeChainSearch(chains = _find_consecutive_sequences(results))
    
    def convert_orthography(self, converter=None, target="orig", base=None, force=False, *kwargs):
        """
        Convert morphs of all analyses of the Token using a Converter in-place.

        Parameters:
            converter (optional): An external Converter object (if None, a new Converter is created).
            target (str): Target orthography ('orig', 'cyr', 'lat', 'ipa').
            force (bool): Whether to force a token into a different orthography or not.
            *kwargs: Additional arguments passed to the converter constructor.
        """
        if converter is None:
            converter = cv.Converter(target=target, base=base, *kwargs)
        res = converter.convert(self.token)

        if force==False:
            if res.base != "unknown" and res.base != converter.target:
                return
        
        self.token = res.text
        if self.ana:
            for a in self.ana:
                if a and a.morphemes:
                    for m in a.morphemes:
                        m.convert_orthography(converter)


class Sentence():
    """
    Represents a text sentence, optionally with a list of Token objects.

    Attributes:
        text (str): String with the sentence text.
        tokens (str or None): List of Token objects associated with the Sentence.
            If None, defaults to the automatic tokenization of Sentence.text.
        translation (str or None): String with the translation of the sentence.
        original (str or None): String with the sentence text in the original transcription.
        comment (str or None): String with the annotator’s comment for the Sentence.
        sent_aid (int or None): EAF annotation id for the sentence.
        trans_aid (int or None): EAF annotation id for the translation.
        original_aid (int or None): EAF annotation id for the original text.
        comment_aid (int or None): EAF annotation id for the comment.
        meta (dict or None): Dictionary with metadata about the Sentence.
        audio (dict or None): Dictionary with data about the audio of the Sentence.
        lang (str or None): String with the sentence’s language label.
        timestamps (tuple): Tuple with EAF timestamps of the Sentence (start, end).
        _id (int or None): Identifier of the Sentence’s number in the source Text.
        back_text (Text or None): Back-reference to the source Text object.

    Properties:
        id (int or None): Identifier of the Sentence’s number in the source Text.
        morph_string (str): String of morphs of tokens, if all Tokens have one Analysis only.
        gloss_string (str): String of glosses of tokens, if all Tokens have one Analysis only.
        pos_string (str): String of POS tags of tokens, if all Tokens have one Analysis only.
        multiple_ana (bool): True if there is more than one Analysis in some of the Tokens.
        morphemes (list): List of all morphemes in the Sentence.
        tokenization_is_correct (bool or str): Check if Sentence.tokens correspond to Sentence.text.
        morphemization_is_correct (bool or str): Check if all morphemes in Sentence.tokens correspond to the token text.

    Methods:
        __getitem__(index): Return the Token at a given index.
        __repr__(): Return a formal string representation of the Sentence.
        __str__(): Informal string representation (same as __repr__).
        gloss_index_string (str): String of morphs and glosses of tokens in the strict `gloss_index`
            format, e.g. 'EMPH{ik=}-such{dis}-', if all Tokens have one analysis only.
        numerate(): Assign sequential ids to all Tokens.
        remove(_id, safe=False, renumerate=True): Remove an Token from the Sentence by id.
        toolong(words=25, seconds=10): Checks if the length of the Sentence exceeds the given limits.
        clean(): Clean whitespace in the beginning and in the end
            of the string and collapse double spaces to single spaces
            for Sentence.text, Sentence.translation and all Sentence.tokens.
        search_morpheme(gloss=None, morph=None, pos=None, gloss_type=None, full=False):
            Search for a Morpheme in the Sentence.
        search_morphemechain(gloss_index_string=None, pos_string=None, _query=None):
            Search for a Morpheme chain in the Sentence.
        search_token(self, token, regex=True, ignore_morphemes=False, ignore_punct=True, ignore_case=True):
            Search for a Token in the Sentence.
        convert_orthography(converter, target, base=None, force=False, *kwargs):
            Convert the orthography of all morphs using a converter.
        tokenize_sentence(mode='glue_punct', punct=PUNCT, strip=True): Tokenize Sentence.text.
        to_print(speaker=False): Pretty print Sentence.text and Sentence.translation.
        to_glossing(): Pretty print a Sentence with glosses.
        glue_punct_tokens(): Glue standalone punctuation to adjacent tokens.
        split_sentence(last_token_i=None, last_token=None, last_token_text=None):
            Split the Sentence into two Sentence objects.
        make_off_values(): Create `off_start` and `off_end` values in Token.tsakorpus_features
            automatically.
    """
    
    def __init__(self, text, tokens=None, translation=None, original=None, comment=None,
                 sent_aid=None, trans_aid=None, original_aid=None, comment_aid=None,
                 meta=None, audio=None, lang=None, timestamps=(None, None),
                 _id=None, back_text=None):
        """
        Initialize a Token object.
        
        Provide a `text` string and, optionally, a list of Tokens as `tokens`.

        Parameters:
            text (str): String with the sentence text.
            tokens (list): List with Tokens for the sentence.
                If None, defaults to the automatic tokenization of Sentence.text.
            translation (str): String with the translation of the sentence (default: None).
            original (str): String with the sentence text in the original transcription (default: None).
            comment (str): String with the annotator’s comment for the Sentence (default: None).
            sent_aid (int): EAF annotation id for the sentence (default: None).
            trans_aid (int): EAF annotation id for the translation (default: None).
            original_aid (int): EAF annotation id for the original text (default: None).
            comment_aid (int): EAF annotation id for the comment (default: None).
            meta (dict): Dictionary with metadata about the Sentence (default: None).
            audio (dict): Dictionary with data about the audio of the Sentence (default: None).
            timestamps (tuple): Tuple with EAF timestamps of the Sentence (start, end)
                (default: (None, None)).
            _id (int): Identifier of the Sentence’s number in the source Text (default: None).
            back_text (Text): Back-reference to the source Text object (default: None).
        """
        
        self.text = text
        
        if type(tokens) in (list, tuple):
            for t in tokens:
                if not isinstance(t, Token):
                    raise TypeError("Sentence.tokens must be a list/tuple of Token objects!")
            self.tokens = tokens
        else:
            if tokens is None:
                self.tokens = [Token(token=t) for t in self.tokenize_sentence()]
            else:
                raise TypeError("Sentence.tokens must be a list or a tuple!")
        
        self.translation = translation
        self.original = original
        self.comment = comment

        self.sent_aid = sent_aid
        self.trans_aid = trans_aid
        self.original_aid = original_aid
        self.comment_aid = comment_aid
        
        self.meta = meta
        self.audio = audio
        self.lang = lang
        self.timestamps = timestamps
        
        self._id = _id
        self.back_text = back_text

    def __getitem__(self, index):
        """
        Return the Token at the specified index in the Sentence.

        Parameters:
            index (int): The index of the Token to retrieve.

        Returns:
            Token: The Token object at the specified index.
        """
        return self.tokens[index]

    def get_id(self, id):
        for t in self.tokens:
            if t._id == id:
                return t
        
    def __repr__(self):
        """
        Return a formal string representation of the Sentence.

        Returns:
            str: A string representation.
        """
        return 'Sentence("' + self.text + '")'
        
    def __str__(self):
        return repr(self)

    def __len__(self):
        """
        Return the number of Tokens in the Sentence.

        Returns:
            int: The number of Tokens.
        """
        return len(self.tokens)
    
    @property
    def id(self):
        """
        Return the identifier of the Sentence’s number in the source Text.

        Returns:
            int or None: The Sentence’s identifier number.
        """
        return self._id
    
    @property
    def morph_string(self):
        """
        Return the string of morphs of all Morphemes separated by morpheme separators.
            Recommended to use only if all Tokens have a single Analysis.

        Returns:
            str or None: String of morphs with morpheme separators (None if there are no Analyses).
        """
        return " ".join(t.morph_string for t in self.tokens)
    
    @property
    def gloss_string(self):
        """
        Return the string of glosses of all Morphemes separated by morpheme separators.
            Recommended to use only if all Tokens have a single Analysis.

        Returns:
            str or None: String of glosses with morpheme separators (None if there are no Analyses).
        """
        return " ".join(t.gloss_string for t in self.tokens)
    
    def gloss_index_string(self, pos=True):
        """
        Return the string in the strict `gloss_index` format, e.g. 'EMPH{ik=}-such{dis}-'
            from all Morphemes. Recommended to use only if all Tokens have a single Analysis.
        
        Parameters:
            pos: Include POS values or not.

        Returns:
            str or None: String in the strict `gloss_index` format (None if there are no Analyses).
        """
        return " ".join(t.gloss_index_string(pos=pos) for t in self.tokens)
    
    @property
    def pos_string(self):
        """
        Return the string of POS tags of all Morphemes separated by morpheme separators.
            Recommended to use only if all Tokens have a single Analysis.

        Returns:
            str or None: String of POS tags with morpheme separators (None if there are no Analyses).
        """
        return " ".join(t.pos_string for t in self.tokens)
    
    @property
    def multiple_ana(self):
        """
        Determine if some Tokens have multiple Analyses.

        Returns:
            bool: True, if at least one Token has more than one Analysis.
        """
        for t in self.tokens:
            if t.multiple_ana:
                return True
        return False
    
    @property
    def morphemes(self):
        """
        Return a sequential list with all Morphemes in all Tokens in the Sentence.

        Returns:
            list: List with all Morphemes.
        """
        res = []
        for t in self.tokens:
            if t.ana and t.ana[0]:
                res.extend(t.ana[0].morphemes)
        return res
    
    def tokenization_is_correct(self, quiet=False):
        """
        Check if Sentence.text is a “sum” of Sentence.tokens (with spaces in between).

        Returns:
            bool or str: True if Sentence.tokens correspond to Sentence.text,
                otherwise either "wrong number of tokens" or "wrong tokens".
        """
        text = Tokenizer(mode="glue_punct").tokenize(self.text)
        if len(text) != len(self.tokens):
            if not quiet:
                warnings.warn(message=f"The number of words in the sentence is not the same as the length of the list of tokens!\n", stacklevel=4)
            return "wrong number of tokens"
        if text != [t.token for t in self.tokens]:
            if not quiet:
                warnings.warn(message=f"The words in the sentence are not the same as the list of tokens!\n", stacklevel=4)
            return "wrong tokens"
        return True
    
    def where_morphemization_is_incorrect(self):
        spisok = []
        for token in self.tokens:
            if not token.morphemization_is_correct():
                spisok.append(token)
        return spisok
    
    def numerate(self, ignore_punct=False):
        """
        Assign sequential ids to all Tokens in the Sentence and to all Analyses in each Token.
        """
        if self.tokens:
            if ignore_punct:
                words = [t for t in self.tokens if t.wtype == "word"]
                punct = [t for t in self.tokens if t.wtype == "punct"]
                for i in range(len(words)):
                    if words[i]:
                        words[i]._id = i+1
                        words[i].numerate()
                for i in range(len(punct)):
                    if punct[i]:
                        punct[i]._id = None
                        punct[i].numerate()
            else:
                for i in range(len(self.tokens)):
                    if self.tokens[i]:
                        self.tokens[i]._id = i+1
                        self.tokens[i].numerate()
    
    def remove(self, _id, safe=False):
        """
        Remove a Token from the list of Tokens in the Sentence by its id.

        Parameters:
            _id (int): The id of the Token to remove.
            safe (bool): If True, silently ignore if id not found.
            renumerate (bool): If True, reassign ids after removal.
        """
        token_to_remove = None
        if self.tokens:
            for i in range(len(self.tokens)):
                if self.tokens[i]._id == _id:
                    token_to_remove = self.tokens[i]
                    break
        if not token_to_remove and safe:
            return

        if token_to_remove and token_to_remove.ana:
            for j in reversed(range(len(token_to_remove.ana))):
                token_to_remove.remove(j, safe=safe)
        del self.tokens[i]
        self.numerate()
    
    def toolong(self, words=25, seconds=None):
        """
        Check if the length of the Sentence exceeds the given limits.

        Parameters:
            tokens (int): The maximum number of Tokens.
            seconds (int): The maximum length of Sentence in seconds.
        
        Returns:
            bool: True if the Sentence exceeds the maximum number of Tokens or the maximum length of the Sentence.
        """
        if words and len(self.tokens) > words:
            return True
        if seconds and "para_alignment" in self.audio and "off_start" in self.audio["para_alignment"][0] and "off_end" in self.audio["para_alignment"][0]:
            if (self.audio["para_alignment"][0]["off_end"] - self.audio["para_alignment"][0]["off_start"]) / 1000 > seconds:
                return True
        elif seconds:
            warnings.warn(message=f"The Sentence does not have para_alignment for audio!\n", stacklevel=4)
    
    def clean(self):
        """
        Clean whitespace in the beginning and in the end
            of the string and collapse double spaces to single spaces.
            This is performed for the Sentence.text and all Sentence.tokens.
        """
        for i in range(len(self.tokens)):
            self.tokens[i].clean()
        self.text = _clean(self.text)
        if self.translation:
            for key in self.translation:
                self.translation[key] = _clean(self.translation[key])

    def search_morpheme(self, gloss=None, morph=None, pos=None, gloss_type=None, full=False):
        """
        Search for a Morpheme in all Analyses in the Sentence.
        Regular expressions are allowed.
        
        Parameters:
            gloss (str or None): Gloss value to be a condition for the search.
                If None, all glosses will be found (default: None).
            morph (str or None): Morph value to be a condition for the search.
                If None, all morphs will be found (default: None).
            pos (str or None): POS tag value to be a condition for the search.
                If None, all POS tags will be found (default: None).
            gloss_type (str or None): gloss type to be a condition for the search.
                One of the following: 'grammeme', 'lemma' or 'other'.
                If None, glosses of all types will be found (default: None).
            full (bool): if True, morpheme separators will be considered
                during the search (default: False).
        
        Returns:
            MorphemeSearch: Results of the search.
        """
        found_morphemes = MorphemeSearch()
        for t in self.tokens:
            if t:
                found_morphemes.extend(
                    t.search_morpheme(gloss=gloss, morph=morph,
                        pos=pos, gloss_type=gloss_type, full=full))
        return found_morphemes
    
    def search_morphemechain(self, gloss_index_string=None, pos_string=None, _query=None):
        """
        Search for a sequence of Morphemes in all Analyses in the Sentence.
        Regular expressions are allowed.
        
        Parameters:
            gloss_index_string (str or None): String in the `gloss{morph}` format,
                containing morphemes to be found. If None, all chains will be found (default: None).
            pos_string  (str or None): String with POS tags separated by hyphens (default: None).
            _query (list or None): A preset list of Morpheme objects (defaults to an empty list).
                Necessary for calls from other internal functions (default: None).
        
        Returns:
            MorphemeChainSearch: Results of the search.
        """
        if _query is None:
            _query = Analysis(gloss_index_string = gloss_index_string, pos_string=pos_string)
        found_morphemechains = MorphemeChainSearch()
        for t in self.tokens:
            found_morphemechains.extend(t.search_morphemechain(_query=_query))
        return found_morphemechains
    
    def search_token(self, token, regex=True, ignore_morphemes=False, ignore_punct=True, ignore_case=True):
        """
        Search for a Token in the Sentence.
        
        Parameters:
            token (str): Token string to be a condition for the search.
            regex (bool): If True, regular expressions are allowed (default: True).
            ignore_morphemes (bool): if True, morpheme separators are ignored
                during the search (default: False).
            ignore_punct (bool): if True, punctuation in the beginning and in the end
                of tokens is ignored during the search (default: True).
            ignore_case (bool): if True, case of tokens (uppercase / lowercase) is ignored
                during the search (default: True).
        
        Returns:
            TokenSearch: Results of the search.
        """
        found_tokens = TokenSearch()
        if ignore_morphemes:
            token = dehyphen(token, equal=True)
        if self.tokens:
            for t in self.tokens:
                cur_token = t.token
                if ignore_morphemes:
                    cur_token = dehyphen(cur_token, equal=True)
                if ignore_punct:
                    cur_token = depunct(cur_token)
                if regex:
                    if ignore_case:
                        res = re.fullmatch(token, cur_token, flags=re.IGNORECASE)
                    else:
                        res = re.fullmatch(token, cur_token)
                else:
                    if ignore_case:
                        res = (token.lower() == cur_token.lower())
                    else:
                        res = (token == cur_token)
                if res:
                    found_tokens.append(t)
        return found_tokens

    def convert_orthography(self, converter=None, target="orig", base=None, force=False, *kwargs):
        """
        Convert all morphs in the Sentence using a Converter in-place.

        Parameters:
            converter (optional): An external Converter object (if None, a new Converter is created).
            target (str): Target orthography ('orig', 'cyr', 'lat', 'ipa').
            force (bool): Whether to force a token into a different orthography or not.
            *kwargs: Additional arguments passed to the converter constructor.
        """
        if converter is None:
            converter = cv.Converter(target=target, base=base, *kwargs)
        self.text = converter.convert(self.text).text
        for t in self.tokens:
            t.convert_orthography(converter, base=base, force=force)
    
    def tokenize_sentence(self, mode="glue_punct", punct=PUNCT, strip=True, output="str", inplace=False):
        """
        Tokenize the sentence text using a specified Tokenizer mode.

        Parameters:
            mode (str): Tokenization mode. Must be one of: 'glue_punct',
                'separate_punct', 'delete_punct', 'separate_all' (default: 'glue_punct').
            punct (str): A regular expression with punctuation characters.
                (Defaults to the global variable PUNCT.)
            strip (bool): Whether to strip whitespace from the input text (default: True).
            output (str): The type of output. If "str", will return list of strings.
                If "Token", will return a list of Tokens (default: "str").
                If inplace == True, output is always "Token".
            inplace (bool): If True, tokens are assigned to the sentence in place.
                If False, tokens are not assigned but returned (default: False).

        Returns:
            list: A list of string tokens.
        """
        if inplace:
            output = "Token"
        tokens = Tokenizer(mode=mode, punct=punct, strip=strip, output="text").tokenize(self.text)
        if output == "Token":
            for i in range(len(tokens)):
                tokens[i] = Token(
                    token=tokens[i],
                    back_sent=self,
                    back_text=self.back_text
                )
        if inplace is False:
            return tokens
        self.tokens = tokens
        self.numerate()


    def to_print(self, speaker=False):
        """
        Return a pretty formatted string containing Sentence.text and Sentence.translation.

        Parameters:
            speaker (bool or str): If True, add speaker label from Sentence.meta['speaker'].
                If str, use the string as speaker label.

        Returns:
            str: Formatted string.
        """
        text = self.text.strip("\r\n")
        trans_string = ""
        if self.translation:
            trans_string = f"\n‘{list(self.translation.values())[0]}’"
        if speaker is True:
            if "speaker" in self.meta:
                speaker = f"[{self.meta['speaker']}]"
            speaker = speaker.strip("[](){}")
            speaker_size = len(speaker)
            return f"[{speaker}] {text}   {' '*speaker_size}{trans_string}"
        return f"{text}{trans_string}"
        
    def to_glossing(self):
        """
        Return a pretty formatted string with the Sentence’s glossing.

        Returns:
            str: Multi-line glossed sentence: raw text, morph line, gloss line, translation, and citation.
        """
        morph_string, gloss_string = [], []
        for t in self.tokens:
            if t and t.ana:
                if t.multiple_ana:
                    warnings.warn(message=f"Multiple analyses in the token {t}!\n", stacklevel=4)
                if t.ana[0]:
                    gloss_string.append(t.ana[0].gloss_string)
                    morph_string.append(t.ana[0].morph_string)
        morph_string, gloss_string = '\t'.join(morph_string), '\t'.join(gloss_string)
        trans_string = ""
        if self.translation:
            trans_string = f"\n‘{list(self.translation.values())[0]}’"
        if t.back_text.title is not None:
            text_title = t.back_text.title
        else:
            text_title = t.back_text.filename
        return f"{self.text}\n{morph_string}\n{gloss_string}{trans_string} [{text_title}, {t.back_sent._id}]"
    
    def print(self):
        print(self.to_glossing())
    
    def to_dict(self):
        morph_strings, gloss_strings, pos_strings = [], [], []
        for t in self.tokens:
            if t and t.ana:
                if t.multiple_ana:
                    warnings.warn(message=f"Multiple analyses in the token {t}!\n", stacklevel=4)
                if t.ana[0]:
                    gloss_strings.append(t.ana[0].gloss_string)
                    morph_strings.append(t.ana[0].morph_string)
                    pos_strings.append(t.ana[0].pos_string)
        translation = None
        if isinstance(self.translation, dict):
            translation = list(self.translation.values())[0]
        return {
            "id": self.id,
            "sentence": " ".join([t.token for t in self.tokens]),
            "morphs": morph_strings,
            "glosses": gloss_strings,
            "pos": pos_strings,
            "translation": translation,
            "original": self.original,
            "comment": self.comment
        }

    def to_latex(self, package="expex"):
        slovar = self.to_dict()
        
        gla = "\\gla " + slovar["sentence"] + " //\n"
        glb = "\\glb " + " ".join(slovar["morphs"]) + " //\n"
        glc = "\\glc " + " ".join(slovar["glosses"]) + " //\n"
        glft = ""
        if slovar["translation"]:
            glft = "\\glft ‘" + slovar["translation"] + "’ //\n"
        
        res = f"\\pex<ex{slovar['id']}>\n\\begingl\n" + gla + glb + glc + glft + "\\endgl \\xe"
        return re.sub("_", "\_", res)
    
    def glue_punct_tokens(self):
        """
        Glue standalone punctuation tokens to adjacent word tokens.
            Modifies Sentence.tokens in-place and reassigns token ids.
        """
        if self.tokens:
            tokens_to_remove = []
            for i in range(len(self.tokens)-1):
                if re.fullmatch(f"[{PREPUNCT}]+", self.tokens[i].token):
                    self.tokens[i+1].token = self.tokens[i].token + self.tokens[i+1].token
                    tokens_to_remove.append(self.tokens[i]._id)
                elif re.fullmatch(f"[{POSTPUNCT}]+", self.tokens[i+1].token):
                    self.tokens[i].token = self.tokens[i].token + self.tokens[i+1].token
                    tokens_to_remove.append(self.tokens[i+1]._id)
            for i in reversed(tokens_to_remove):
                self.remove(_id=i)
        self.numerate()
    
    def split_sentence(self, last_token_id=None, last_token=None, last_token_text=None, inplace=False):
        """
        Split the Sentence into two new Sentence objects at a specific Token.

        Provide one of the following:
            `last_token_id`, `last_token`, or `last_token_text`.

        Parameters:
            last_token_id (int): Index of the Token to split after.
            last_token (Token): Token object to split after.
            last_token_text (str): Raw text of the Token to split after.

        Returns:
            tuple: Two Sentence objects resulting from the split.
        """
        if (last_token is None) and (last_token_id is None) and (last_token_text is None):
            raise ValueError("Neither last_token_id nor last_token nor last_token_text were given!")
        if last_token is not None:
            if last_token_id is not None:
                raise ValueError("Both last_token_id and last_token were given!")
            if last_token_text is not None:
                raise ValueError("Both last_token and last_token_text were given!")
            if last_token not in self.tokens:
                raise ValueError("The last_token is not in the sentence!")
            last_token_id = last_token._id
        if last_token_text is not None:
            if last_token_id is not None:
                raise ValueError("Both last_token_id and last_token_text were given!")
            last_token_id = None
            for i in range(len(self.tokens)):
                if self.tokens[i].token == last_token_text:
                    last_token_id = i+1
                    break
            if not last_token_id:
                raise ValueError("This last_token_text was not found!")
        
        last_token_i = [t._id for t in self.tokens].index(last_token_id)
        
        custom_tokenizer = Tokenizer(mode="glue_punct")
        text = custom_tokenizer.tokenize(self.text)
        text1 = text[:last_token_i+1]
        text2 = text[last_token_i+1:]
        if len(text1 + text2) != len(self.tokens):
            raise ValueError(f"Different lengths of sentence ({len(text1 + text2)}) and list of tokens ({len(self.tokens)})!")
        midpoint = sum(self.timestamps)//2
        
        sent1 = Sentence(
            text=" ".join(text1),
            tokens=self.tokens[:last_token_i+1],
            translation=self.translation, original=self.original, comment=self.comment,
            sent_aid=self.sent_aid, trans_aid=self.trans_aid,
            original_aid=self.original_aid, comment_aid=self.comment_aid,
            meta=self.meta, audio=self.audio, timestamps=(self.timestamps[0], midpoint),
            _id=self._id, back_text=self.back_text
        )
        
        sent2 = Sentence(
            text=" ".join(text2),
            tokens=self.tokens[last_token_i+1:],
            translation=self.translation,
            sent_aid=self.sent_aid, trans_aid=self.trans_aid,
            original_aid=self.original_aid, comment_aid=self.comment_aid,
            meta=self.meta, audio=self.audio, timestamps=(midpoint, self.timestamps[1]),
            _id=self._id, back_text=self.back_text
        )
        
        for t in self.tokens[:last_token_i+1]:
            if t:
                t.back_sent = sent1
                if t.ana:
                    for a in t.ana:
                        if a:
                            a.back_sent = sent1
                            if a.morphemes:
                                for m in a.morphemes:
                                    if m:
                                        m.back_sent = sent1
        
        for t in self.tokens[last_token_i+1:]:
            if t:
                t.back_sent = sent2
                if t.ana:
                    for a in t.ana:
                        if a:
                            a.back_sent = sent2
                            if a.morphemes:
                                for m in a.morphemes:
                                    if m:
                                        m.back_sent = sent2
        
        if inplace:
            positional_i = self.back_text.sentences.index(self)
            self.back_text.sentences = self.back_text.sentences[:positional_i] + [sent1, sent2] + self.back_text.sentences[positional_i+1:]
            self.back_text.numerate()
            return

        return (sent1, sent2)

    def make_off_values(self, force=False):
        """
        Automatically create `off_start` and `off_end` values in Token.tsakorpus_features in-place.
        
        Only works if `Token.tokenization_is_correct()` for all Tokens, and the Sentence is clean.

        Parameters:
            force (bool): if True, error will be raised if it making the values is impossible
                          if False, no errors will be raised
        """
        if force == True:
            if self.tokenization_is_correct() != True:
                raise ValueError("Sentence.make_off_values() can only be performed if Sentence.tokenization_is_correct() is True!")
            if _clean(self.text) != self.text:
                raise ValueError("Sentence.make_off_values() can only be performed if Sentence has been “cleaned” by Sentence.clean()!")
        
        i = 0
        for token in self.tokens:
            token.tsakorpus_features["off_start"] = i
            token.tsakorpus_features["off_end"] = i+len(token.token)
            i += len(token.token) + 1

    def to_df(self, token_id=None):
        df = pd.DataFrame(columns=[
            "sent_id", "sentence", "morphs", "glosses", "pos", "translation",
            "subcorpus", "text.title", "text.filename", "text.author", "text.year"])
        
        df["sent_id"] = [self.id if isinstance(self.id, int) else ""]
        
        df["sentence"] = [self.text]
        df["morphs"] = [self.morph_string]
        df["glosses"] = [self.gloss_string]
        df["pos"] = [self.pos_string]
        if self.translation and len(self.translation) == 1:
            df["translation"] = [list(self.translation.values())[0]]
        else:
            df["translation"] = [self.translation]
        df["subcorpus"] = self.back_text.meta["subcorpus"] if self.back_text and "subcorpus" in self.back_text.meta else ""
        df["text.title"] = self.back_text.title if self.back_text and self.back_text.title else ""
        df["text.filename"] = self.back_text.filename if self.back_text else ""
        df["text.author"] = self.back_text.meta["author"] if self.back_text and "author" in self.back_text.meta else ""
        df["text.year"] = self.back_text.meta["year"] if self.back_text and "year" in self.back_text.meta else ""

        if token_id:
            df.insert(loc=0, column="token_id", value=[token_id])
            df.insert(loc=1, column="token", value=self.tokens[token_id-1].token)
        return df


class Text():
    def __init__(self, filename, abspath=None, folderpath=None, title=None, tiers=None,
                 metadata_path=None, _metadata=None, _id=None, load_file=True):
        self.filename = os.path.basename(filename)
        self._pure_filename = os.path.splitext(self.filename)[0]
        if abspath is None:
            folderpath = "" if folderpath is None else folderpath
            current_folder = os.path.dirname(os.path.abspath(__file__))
            abs_folderpath = os.path.join(current_folder, folderpath)
            abspath = os.path.join(abs_folderpath, filename)
        self.abspath = abspath
        
        self.filetype = os.path.splitext(self.filename)[1].strip(".")
        if (not self.filetype in ("eaf", "json", "csv", "tsv", "txt")):
            raise TypeError(f'File type "{self.filename}" is not supported!')
        
        self.sentences = []
        self.title = title
        
        self.meta = {}
        self.audio = {}
        
        if metadata_path is not None:
            if _metadata is not None:
                raise ValueError("For cx.Text(), provide either `metadata_path` or `_metadata`, but not both!")
            _metadata = None
            if metadata_path and metadata_path.endswith("csv"):
                _metadata = pd.read_csv(metadata_path)
        
        if _metadata is not None:
            if isinstance(_metadata, str):
                _metadata = pd.read_csv(_metadata)
            if isinstance(_metadata, pd.DataFrame):
                if self._pure_filename in list(_metadata["filename"]):
                    row = _metadata[_metadata["filename"]==self._pure_filename]
                    if "title" in row:
                        self.title = row.get("title").values[0]
                    if "title_en" in row:
                        self.title = row.get("title_en").values[0]
                    meta_fields = [
                        "glossing", "translation",
                        "media type", "type", "genre", "subcorpus", "duration", "year",
                        "dialect", "author", "gender", "birth year", "source",
                        "pages", "recorded by", "translated by", "glossed by", "glossing year",
                        "comment", "public comment"]
                    for field in meta_fields:
                        if field in row:
                            v = row.get(field).values[0]
                            try:
                                x = np.isnan(v)
                            except TypeError:
                                if field == "year":
                                    try:
                                        v = int(v)
                                    except:
                                        pass
                                self.meta[field] = v
                    
                else:
                    warnings.warn(message=f"Text '{self._pure_filename}' not found in the metadata dataframe!\n", stacklevel=4)
        
        self._id = _id
        
                
        # INITIALIZING FROM EAF

        self.eaf = None
        
        def initialize_eaf_text(self, tiers=None):
            self.eaf = pympi.Elan.Eaf(file_path=abspath)
            speakers = []

            if tiers:
                n = len(tiers)
                if n>8:
                    raise ValueError(f"Too many values were provided for the 'tiers' parameter for the text '{self.filename}'!")
                self.tier_types = tiers[:n] + list(tier_types.values())[n:]
            else:
                self.tier_types = list(tier_types.values())
            
            speaker_prefixes = []
            for tiername in self.eaf.tiers.keys():
                res = re.search(f"(.+)_({self.tier_types[0]})", tiername)
                if res:
                    speaker_prefixes.append(res[1])

            for speaker_prefix in speaker_prefixes:
                _sgh_sentences_tiername = f"{speaker_prefix}_{self.tier_types[0]}"
                _ru_sentences_tiername = f"{speaker_prefix}_{self.tier_types[1]}"
                _sgh_tokens_tiername = f"{speaker_prefix}_{self.tier_types[2]}"
                _sgh_morphs_tiername = f"{speaker_prefix}_{self.tier_types[3]}"
                _eng_glosses_tiername = f"{speaker_prefix}_{self.tier_types[4]}"
                _eng_pos_tiername = f"{speaker_prefix}_{self.tier_types[5]}"
                _sgh_original_tiername = f"{speaker_prefix}_{self.tier_types[6]}"
                _comment_tiername = f"{speaker_prefix}_{self.tier_types[7]}"

                try:
                    sentences = self.eaf.get_annotation_data_for_tier(_sgh_sentences_tiername)
                except KeyError:
                    if speaker_prefix == "A":
                        warnings.warn(message=f"In text {self.filename},\nwrong tiers: {', '.join(self.eaf.tiers)}!\n", stacklevel=4)
                    return None
                
                speakers.append(speaker_prefix)
                
                sent_items = tuple(self.eaf.tiers[_sgh_sentences_tiername][0].items())
                trans_items, tokens_items, morphs_items, glosses_items = tuple(), tuple(), tuple(), tuple()
                original_items, comment_items = tuple(), tuple()
                
                if _sgh_original_tiername in self.eaf.tiers:
                    original_items = tuple(self.eaf.tiers[_sgh_original_tiername][1].items())
                if _comment_tiername in self.eaf.tiers:
                    comment_items = tuple(self.eaf.tiers[_comment_tiername][1].items())
                if _ru_sentences_tiername in self.eaf.tiers:
                    trans_items = tuple(self.eaf.tiers[_ru_sentences_tiername][1].items())
                    trans_lang = self.eaf.tiers[_ru_sentences_tiername][2]["LANG_REF"]
                if _sgh_tokens_tiername in self.eaf.tiers:
                    tokens_items = tuple(self.eaf.tiers[_sgh_tokens_tiername][1].items())
                if _sgh_morphs_tiername in self.eaf.tiers:
                    morphs_items = tuple(self.eaf.tiers[_sgh_morphs_tiername][1].items())
                if _eng_glosses_tiername in self.eaf.tiers:
                    glosses_items = tuple(self.eaf.tiers[_eng_glosses_tiername][1].items())

                if _eng_pos_tiername in self.eaf.tiers:
                    pos_items = tuple(self.eaf.tiers[_eng_pos_tiername][1].items())
                else:
                    pos_items = tuple()
                    warnings.warn(message=f"The text {self.filename} does not have a POS layer!", stacklevel=4)
                
                for i in range(len(sent_items)):
                    
                    tokens = []
                    for t_item in tokens_items:
                        
                        if t_item[1][0] == sent_items[i][0]:
                        
                            morphs = []
                            for m_item in morphs_items:
                                if t_item[0] == m_item[1][0]:
                                    
                                    gloss, gloss_aid = None, None
                                    for g_item in glosses_items:
                                        if m_item[0] == g_item[1][0]:
                                            gloss, gloss_aid = g_item[1][1], g_item[0]
                                            break
                                    pos, pos_aid = None, None
                                    for p_item in pos_items:
                                        if m_item[0] == p_item[1][0]:
                                            pos, pos_aid = p_item[1][1], p_item[0]
                                            break
                                    morphs.append(Morpheme(
                                        morph=m_item[1][1], gloss=gloss,
                                        morph_aid=m_item[0], gloss_aid=gloss_aid,
                                        pos=pos, pos_aid=pos_aid))
                            ana = Analysis(morphs)
                            ana = [ana] if ana else [None]
                            tokens.append(Token(
                                token=t_item[1][1], ana=ana,
                                token_aid=t_item[0]))
                    
                    if i in range(len(trans_items)) and trans_items[i]:
                        translation, trans_aid = {trans_lang: trans_items[i][1][1]}, trans_items[i][0]
                    else:
                        translation, trans_aid = None, None
                    
                    if i in range(len(original_items)) and original_items[i]:
                        original, original_aid = original_items[i][1][1], original_items[i][0]
                    else:
                        original, original_aid = None, None
                    
                    if i in range(len(comment_items)) and comment_items[i]:
                        comment, comment_aid = comment_items[i][1][1], comment_items[i][0]
                    else:
                        comment, comment_aid = None, None
                    
                    off_start = self.eaf.timeslots[sent_items[i][1][0]]
                    off_end = self.eaf.timeslots[sent_items[i][1][1]]
                    
                    new_sentence = Sentence(
                        text = sent_items[i][1][2],
                        tokens = tokens,
                        translation = translation,
                        original = original,
                        comment = comment,
                        sent_aid = sent_items[i][0],
                        trans_aid = trans_aid,
                        original_aid = original_aid,
                        comment_aid = comment_aid,
                        meta = {"speaker": speaker_prefix},
                        audio = {
                            "para_alignment": [{
                                "off_start": off_start,
                                "off_end": off_end
                            }]
                        },
                        timestamps = (off_start, off_end)
                    )
                    new_sentence.make_off_values(force=False)
                    self.sentences.append(new_sentence)
                
                self.meta["speakers"] = speakers
        
        
        if self.filetype == "eaf" and load_file:
            initialize_eaf_text(self, tiers=tiers)
            if self.eaf:
                if len(self.eaf.media_descriptors)>0:
                    if "RELATIVE_MEDIA_URL" in self.eaf.media_descriptors[0]:
                        self.audio["filename"] = self.eaf.media_descriptors[0]["RELATIVE_MEDIA_URL"]
                    if "MEDIA_URL" in self.eaf.media_descriptors[0]:
                        self.audio["MEDIA_URL"] = self.eaf.media_descriptors[0]["MEDIA_URL"]
        


        # INITIALIZING FROM JSON
        
        self.json = None
        
        def initialize_json_text(self):
            self.json = json.load(open(abspath, "r", encoding="utf-8"))
            
            if "filename" in self.json["meta"]:
                self.json["meta"]["eaf_filename"] = self.json["meta"]["filename"]
                del self.json["meta"]["filename"]
            
            for key in self.json["meta"]:
                if not key in self.meta:
                    self.meta[key] = self.json["meta"][key]
            
            if "title" in self.meta and self.title is None:
                self.title = self.meta["title"]
            
            for sent in self.json["sentences"]:
                if sent["lang"] == 0:
                    translations = {}
                    for b_sent in self.json["sentences"]:
                        try:
                            if b_sent["lang"] != 0 and b_sent["para_alignment"][0]["para_id"] == sent["para_alignment"][0]["para_id"]:
                                translations[b_sent["lang"]] = b_sent["text"]
                        except:
                            print(b_sent)
                    
                    audio = {}
                    for feature in ("para_alignment", "src_alignment"):
                        if feature in sent:
                            audio[feature] = sent[feature]
                    
                    tokens = []
                    if "words" in sent:
                        for token in sent["words"]:
                            ana = []
                            if "ana" in token:
                                for a in token["ana"]:
                                    ana.append(Analysis(
                                        morph_string=a["parts"],
                                        gloss_string=a["gloss"] if "gloss" in a else None,
                                        gloss_index_string=a["gloss_index"] if "gloss_index" in a else None
                                    ))
                            tokens.append(Token(
                                token=token["wf"],
                                ana=ana,
                                tsakorpus_features={
                                    "off_start": token["off_start"],
                                    "off_end":   token["off_end"]
                                }
                            ))
                    
                    original = sent["original"] if "original" in sent else None
                    comment = sent["comment"] if "comment" in sent else None

                    self.sentences.append(Sentence(
                        text=sent["text"],
                        tokens=tokens,
                        translation=translations,
                        original=original, comment=comment,
                        meta=sent["meta"] if "meta" in sent else None,
                        audio=audio,
                        lang=sent["lang"]
                    ))
        
        
        if self.filetype == "json" and load_file:
            initialize_json_text(self)
        


        # INITIALIZING FROM TXT

        self.txt = None
        
        def initialize_txt_text(self):
            with open(abspath, "r", encoding="utf-8-sig") as f:
                self.txt = f.read()
            txt_lines = self.txt.split("\n\n")

            current_ts = 0
            for line in txt_lines:
                line = line.strip("\n").split("\n")
                sent_length = max(len(re.findall(" ", line[0])), 1) * DEFAULT_ANNOTATION_DENSITY
                begin_ts = current_ts
                end_ts = begin_ts + sent_length
                current_ts = end_ts

                self.sentences.append(Sentence(
                    text=line[0],
                    translation={"ru": line[1]},
                    original=line[0],
                    meta={"speaker": "A"},
                    audio = {
                        "para_alignment": [{
                            "off_start": begin_ts,
                            "off_end": end_ts
                        }]
                    },
                    timestamps = (begin_ts, end_ts)
                ))
            
            for sentence in self.sentences:
                sentence.tokens = []

        if self.filetype == "txt" and load_file:
            initialize_txt_text(self)
        

        ########
        
        
        self.numerate()
        
        if self.sentences:
            for s in self.sentences:
                if s:
                    s.back_text = self
                    if s.tokens:
                        for t in s.tokens:
                            if t:
                                t.back_sent = s
                                t.back_text = self
                                if t.ana:
                                    for a in t.ana:
                                        if a:
                                            a.back_token = t
                                            a.back_sent = s
                                            a.back_text = self
                                            if a.morphemes:
                                                for m in a.morphemes:
                                                    if m:
                                                        m.back_ana = a
                                                        m.back_token = t
                                                        m.back_sent = s
                                                        m.back_text = self


    def __getitem__(self, index):
        """
        Return the Sentence at the specified index in the Text.

        Parameters:
            index (int): The index of the Sentence to retrieve.

        Returns:
            Sentence: The Sentence object at the specified index.
        """
        return self.sentences[index]

    def get_id(self, id):
        for s in self.sentences:
            if s._id == id:
                return s

    def __repr__(self):
        """
        Return a formal string representation of the Text.

        Returns:
            str: A string representation.
        """
        if self.title:
            return 'Text("' + self.title + '")'
        return 'Text(filename="' + self.filename + '")'

    def __str__(self):
        return repr(self)
    
    def __len__(self):
        """
        Return the number of Sentences in the Text.

        Returns:
            int: The number of Sentences.
        """
        return len(self.sentences)
    
    @property
    def id(self):
        """
        Return the identifier of the Text’s number in the source Corpus.

        Returns:
            int or None: The Text’s identifier number.
        """
        return self._id
        
    @property
    def tokens(self):
        res = []
        for s in self.sentences:
            res.extend(s.tokens)
        return res
        
    @property
    def morphemes(self):
        res = []
        for s in self.sentences:
            if s:
                res.extend(s.morphemes)
        return res
    
    @property
    def multiple_ana(self):
        for t in self.tokens:
            if t.multiple_ana:
                return True
        return False

    @property
    def tier_names(self):
        return list(self.eaf.tiers.keys())
    
    def numerate(self, ignore_punct=False):
        """
        Assign sequential ids to all Sentences in the Text and all Tokens in each Sentence.
        """
        if self.sentences:
            for i in range(len(self.sentences)):
                if self.sentences[i]:
                    self.sentences[i]._id = i+1
                    self.sentences[i].numerate(ignore_punct=ignore_punct)
    
    def remove(self, _id, safe=False):
        """
        Remove a Sentence from the Text by its id.

        Parameters:
            _id (int): The id of the Sentence to remove.
            safe (bool): If True, silently ignore if id not found.
            renumerate (bool): If True, reassign ids after removal.
        """
        sentence_to_remove = None
        if self.sentences:
            for i in range(len(self.sentences)):
                if self.sentences[i]._id == _id:
                    sentence_to_remove = self.sentences[i]
                    break
        if not sentence_to_remove and safe:
            return

        if sentence_to_remove and sentence_to_remove.tokens:
            for j in reversed(range(len(sentence_to_remove.tokens))):
                sentence_to_remove.remove(j, safe=safe)
        del self.sentences[i]
        self.numerate()
    
    def toolong(self, words=25, seconds=None):
        return [s for s in self.sentences if s.toolong(words=words, seconds=seconds)]
    
    def where_tokenization_is_incorrect(self, mode=None):
        slovar = {
            "wrong number of tokens": [],
            "wrong tokens": []
        }
        if len(self.tokens) == 0:
            return slovar
        for s in self:
            result = s.tokenization_is_correct()
            if result != True:
                slovar[result].append(s)
        if mode in ("wrong number of tokens", "wrong tokens"):
            return slovar[mode]
        return slovar
    
    def where_morphemization_is_incorrect(self, mode=None):
        spisok = []
        for s in self:
            spisok.extend(s.where_morphemization_is_incorrect())
        return spisok
    
    def sort_sentences(self, by="time", inplace=True):
        if not by in ("time", "speaker"):
            raise ValueError(f"The value of the argument 'by' in the method Text.sort() should be 'time' or 'speaker'!")
        
        if inplace:
            sorted_sentences = self.sentences
        else:
            sorted_sentences = self.sentences.copy()

        if by == "time":
            try:
                sorted_sentences = sorted(sorted_sentences, key=lambda x: x.meta["speaker"])
            except KeyError:
                pass
            try:
                sorted_sentences = sorted(sorted_sentences, key=lambda x: x.audio["para_alignment"][0]["off_start"])
            except KeyError:
                raise ValueError(f"Not all sentences in this text are media-aligned correctly!")
        
        elif by == "speaker":
            try:
                sorted_sentences = sorted(sorted_sentences, key=lambda x: x.audio["para_alignment"][0]["off_start"])
            except KeyError:
                pass
            try:
                sorted_sentences = sorted(sorted_sentences, key=lambda x: x.meta["speaker"])
            except KeyError:
                raise ValueError(f"Not all sentences in this text are marked for speaker!")
        
        if not inplace:
            return sorted_sentences
    
    def clean(self):
        for i in range(len(self.sentences)):
            self.sentences[i].clean()
    
    def search_morpheme(self, gloss=None, morph=None, pos=None, gloss_type=None, full=False):
        """
        Search for a Morpheme in all Analyses in the Text.
        Regular expressions are allowed.
        
        Parameters:
            gloss (str or None): Gloss value to be a condition for the search.
                If None, all glosses will be found (default: None).
            morph (str or None): Morph value to be a condition for the search.
                If None, all morphs will be found (default: None).
            pos (str or None): POS tag value to be a condition for the search.
                If None, all POS tags will be found (default: None).
            gloss_type (str or None): gloss type to be a condition for the search.
                One of the following: 'grammeme', 'lemma' or 'other'.
                If None, glosses of all types will be found (default: None).
            full (bool): if True, morpheme separators will be considered
                during the search (default: False).
        
        Returns:
            MorphemeSearch: Results of the search.
        """

        found_morphemes = MorphemeSearch()
        for s in self.sentences:
            if s:
                found_morphemes.extend(
                    s.search_morpheme(gloss=gloss, morph=morph,
                        pos=pos, gloss_type=gloss_type, full=full))
        return found_morphemes
    
    def search_morphemechain(self, gloss_index_string=None, pos_string=None, _query=None):
        """
        Search for a sequence of Morphemes in all Analyses in the Text.
        Regular expressions are allowed.
        
        Parameters:
            gloss_index_string (str): String in the `gloss{morph}` format,
                containing morphemes to be found. If None, all chains will be found.
            pos_string  (str or None): String with POS tags separated by hyphens (default: None).
            _query (list or None): A preset list of Morpheme objects (defaults to an empty list).
                Necessary for calls from other internal functions (default: None).
        
        Returns:
            MorphemeChainSearch: Results of the search.
        """
        if _query is None:
            _query = Analysis(gloss_index_string = gloss_index_string, pos_string=pos_string)
        found_morphemechains = MorphemeChainSearch()
        for s in self.sentences:
            found_morphemechains.extend(s.search_morphemechain(_query=_query))
        return found_morphemechains
    
    def search_token(self, token, regex=True, ignore_morphemes=False, ignore_punct=True, ignore_case=True):
        """
        Search for a Token in the Text.
        
        Parameters:
            token (str): Token string to be a condition for the search.
            regex (bool): If True, regular expressions are allowed (default: True).
            ignore_morphemes (bool): if True, morpheme separators are ignored
                during the search (default: False).
            ignore_punct (bool): if True, punctuation in the beginning and in the end
                of tokens is ignored during the search (default: True).
            ignore_case (bool): if True, case of tokens (uppercase / lowercase) is ignored
                during the search (default: True).
        
        Returns:
            TokenSearch: Results of the search.
        """
        found_tokens = TokenSearch()
        for s in self.sentences:
            if s:
                found_tokens.extend(
                    s.search_token(
                        token=token, regex=regex, ignore_morphemes=ignore_morphemes,
                        ignore_punct=ignore_punct, ignore_case=ignore_case))
        return found_tokens
    
    def search_sentence(self, sentence, regex=True, ignore_morphemes=False, ignore_punct=True, ignore_case=True):
        """
        Search for a Sentence in the Text.
        
        Parameters:
            sentence (str): Sentence string to be a condition for the search.
            regex (bool): If True, regular expressions are allowed (default: True).
            ignore_morphemes (bool): if True, morpheme separators are ignored
                during the search (default: False).
            ignore_punct (bool): if True, punctuation in the beginning and in the end
                of tokens is ignored during the search (default: True).
            ignore_case (bool): if True, case of tokens (uppercase / lowercase) is ignored
                during the search (default: True).
        
        Returns:
            SentenceSearch: Results of the search.
        """
        found_sentences = SentenceSearch()
        if ignore_morphemes:
            sentence = dehyphen(sentence, equal=True)
        if self.sentences:
            for s in self.sentences:
                cur_sent = s.text
                if ignore_morphemes:
                    cur_sent = dehyphen(cur_sent, equal=True)
                if ignore_punct:
                    cur_sent = depunct(cur_sent)
                if regex:
                    if ignore_case:
                        res = re.fullmatch(sentence, cur_sent, flags=re.IGNORECASE)
                    else:
                        res = re.fullmatch(sentence, cur_sent)
                else:
                    if ignore_case:
                        res = (sentence.lower() == cur_sent.lower())
                    else:
                        res = (sentence == cur_sent)
                if res:
                    found_sentences.append(s)
        return found_sentences
    
    def morph_vocab(self, gloss_type=None):
        if gloss_type is None:
            return Counter([(m.morph_full, m.gloss) for m in self.morphemes])
        if gloss_type == "lemma":
            return Counter([(m.morph_full, m.gloss) for m in self.morphemes if m.is_lemma])
        elif gloss_type == "grammeme":
            return Counter([(m.morph_full, m.gloss) for m in self.morphemes if m.is_grammeme])
    
    def homophones(self, gloss_type=None):
        morphemes = sorted(self.morph_vocab(gloss_type=gloss_type).keys())
        morphemes_dict = {}
        for i in range(len(morphemes)):
            morph, gloss = morphemes[i]
            if not morph in morphemes_dict:
                morphemes_dict[morph] = [gloss]
            else:
                morphemes_dict[morph].append(gloss)
        homophones_dict = {}
        for morph in morphemes_dict:
            if len(morphemes_dict[morph]) > 1:
                homophones_dict[morph] = morphemes_dict[morph]
        return homophones_dict

    def convert_orthography(self, converter=None, target="orig", force=False, *kwargs):
        """
        Convert all morphs in the Text using a Converter in-place.

        Parameters:
            converter (optional): An external Converter object (if None, a new Converter is created).
            target (str): Target orthography ('orig', 'cyr', 'lat', 'ipa').
            force (bool): Whether to force a token into a different orthography or not.
            *kwargs: Additional arguments passed to the converter constructor.
        """
        if converter is None:
            converter = cv.Converter(target=target, *kwargs)
        for s in self.sentences:
            s.convert_orthography(converter, force=force)
    
    def to_print(self, speaker=False):
        """
        Return a pretty formatted text containing Sentence.text and Sentence.translation for each sentence.

        Parameters:
            speaker (bool or str): If True, add speaker label from Sentence.meta['speaker'].
                If str, use the string as speaker label.

        Returns:
            str: Formatted string.
        """
        sorted_sentences = self.sort_sentences(by="time", inplace=False)
        if sorted_sentences:
            return "\n\n".join([s.to_print(speaker=speaker) for s in sorted_sentences])
    
    def to_glossing(self):
        """
        Return a pretty formatted string with the Sentence’s glossing.

        Returns:
            str: Multi-line glossed sentence: raw text, morph line, gloss line, translation, and citation.
        """
        return "\n\n".join([s.to_glossing() for s in self.sentences])
    
    def to_latex(self, package="expex"):
        """
        Return a string with the Sentence’s glossing for LaTeX documents.

        Returns:
            str: LaTeX expression with the glossed sentence.
        """
        return "\n\n".join([s.to_latex(package=package) for s in self.sentences])
    
    def glue_punct_tokens(self):
        for s in self.sentences:
            s.glue_punct_tokens()
    
    def split_sentence(self, sentence_i, last_token_i=None, last_token=None, last_token_text=None):
        sentence = self.sentences[sentence_i]
        
        sent1, sent2 = sentence.split_sentence(
            last_token_i=last_token_i, last_token=last_token, last_token_text=last_token_text)
        
        self.sentences = self.sentences[:sentence_i] + [sent1, sent2] + self.sentences[sentence_i+1:]
        self.numerate()
    
    def clean_eaf(self):
        x = copy(self.eaf.tiers)
        self.eaf.remove_tiers(x)
        self.eaf.clean_time_slots()
        self.eaf.maxts, self.eaf.maxaid = 0, 0
        self.eaf.annotations = {}
    
    def clean_tier(self, tier):
        self.eaf.remove_all_annotations_from_tier(id_tier=tier)

    def define_speakers(self):
        if "speakers" in self.meta:
            return self.meta["speakers"]
        speakers = set()
        for s in self.sentences:
            if "speaker" in s.meta:
                speakers.add(s.meta["speaker"])
        return sorted(speakers)

    def update_eaf(self, convert_text=False, convert_tokens=False,
                   convert_morphs=False, target="orig", base=None, force=False,
                   keep_original=False):
        if not self.eaf:
            self.eaf = pympi.Elan.Eaf()
        
        self.clean_eaf()
        self.eaf.linguistic_types = linguistic_types_description

        if "speakers" not in self.meta:
            self.meta["speakers"] = ["A"]
        create_obligatory_tiers(eaf=self.eaf, speakers=self.meta["speakers"])
        
        for s in self.sentences:
            create_annotation(eaf=self.eaf, content=s, convert=convert_text,
                              target=target, base=base, force=force)
        
        text_vs_original = [s.text == s.original for s in self.sentences if s.original]
        if (not False in text_vs_original) and (not keep_original):
            for speaker in self.define_speakers():
                if f"{speaker}_{sgh_original_tiername_type}" in self.eaf.tiers:
                    self.eaf.remove_all_annotations_from_tier(f"{speaker}_{sgh_original_tiername_type}")
        
        for t in self.tokens:
            create_annotation(eaf=self.eaf, content=t, convert=convert_tokens,
                              target=target, base=base, force=force)
        
        for m in self.morphemes:
            create_annotation(eaf=self.eaf, content=m, convert=convert_morphs,
                              target=target, base=base, force=force)
    
    def to_eaf(self, filepath=None, folderpath="", filename=None, suffix="_new", update_eaf=False):
        if update_eaf:
            self.update_eaf()
        
        if filename is None:
            filename = self.filename
        if filepath is None:
            filepath = re.sub("\.[^\.]+$", "", filename)
            filepath = f"{filepath}{suffix}.eaf"
        path = os.path.join(folderpath, filepath)

        if os.path.isfile(path):
            os.unlink(path)
        
        pympi.Elan.to_eaf(path, self.eaf)
    
    def make_off_values(self):
        """
        Automatically create `off_start` and `off_end` values in Token.tsakorpus_features in-place.
        
        Only works if `Token.tokenization_is_correct()` for all Tokens, and the Sentence is clean.
        """
        for s in self.sentences:
            s.make_off_values()
    
    def make_timestamps(self, annotation_density=1000):
        """Automatically create timestamps for Sentences."""
        current_ts = 0
        for s in self.sentences:
            length = max(len(re.findall(" ", s.text)), 1) * annotation_density
            begin_ts = current_ts
            end_ts = begin_ts + length
            current_ts = end_ts
            s.timestamps = (begin_ts, end_ts)

    def update_json(self, force_off_values=False, force_tokenize=False):
        
        json_dict = {
            "meta": {},
            "sentences": []
        }

        if force_tokenize:
            for sentence in self.sentences:
                if len(sentence.tokens) == 0:
                    sentence.tokenize_sentence(inplace=True)
        
        json_dict["meta"].update(self.meta)

        if "year" in json_dict["meta"]:
            try:
                json_dict["meta"]["year"] = int(json_dict["meta"]["year"])
            except ValueError:
                res = re.sub(" \(\?\)", "", json_dict["meta"]["year"])
                res = re.split("[-–—]+", res)
                if len(res) == 2:
                    try:
                        json_dict["meta"]["year_from"] = res[0]
                        json_dict["meta"]["year_to"] = res[1]
                    except ValueError:
                        pass
                del json_dict["meta"]["year"]

        self.filename = self._pure_filename + ".json"
        json_dict["meta"]["filename"] = self._pure_filename + ".json"

        if self.title:
            json_dict["meta"]["title"] = self.title
        
        for feature in "title", "author":
            if feature not in json_dict["meta"]:
                json_dict["meta"][feature] = ""
                warnings.warn(message=f"The text '{self.filename}' has no {feature}!", stacklevel=4)
        
        def get_grpos(ana):
            grpos = []
            for m in ana.morphemes:
                if m.pos != None and m.pos not in grpos:
                    grpos.append(m.pos)
            return grpos
        
        def create_jsons_for_tokens(sentence, tokenize=False, force_off_values=True):
            if tokenize:
                sentence.tokens = sentence.tokenize_sentence(output="Token")
                sentence.numerate()
            words = []
            if sentence.tokens:
                for t in sentence.tokens:
                    try:
                        off_start = t.tsakorpus_features["off_start"]
                        off_end = t.tsakorpus_features["off_end"]
                    except:
                        sentence.make_off_values(force=force_off_values)
                        off_start = t.tsakorpus_features["off_start"]
                        off_end = t.tsakorpus_features["off_end"]
                    current_word = {
                        "wf": t.token,
                        "wtype": t.wtype,
                        "off_start": off_start,
                        "off_end": off_end,
                        "next_word": t._id,
                        "sentence_index": t._id-1,
                        "sentence_index_neg": len(sentence)-(t._id-1)
                    }
                    
                    if t.ana and len(t.ana)>0:
                        analyses = []
                        for ana in t.ana:
                            if ana:
                                lex = [m.morph_full for m in ana.morphemes if (m._prefix == "" and m._suffix == "")]
                                lex = lex[-1] if len(lex)>0 else ""
                                analyses.append({
                                    "gr.pos": get_grpos(ana),
                                    "parts": ana.morph_string,
                                    "gloss": ana.gloss_string,
                                    "lex": lex,
                                    #"trans_en": [m.gloss for m in ana.morphemes],
                                    "gloss_index": ana.gloss_index_string(pos=False)
                                })
                        if len(analyses) > 0:
                            current_word["ana"] = analyses
                    words.append(current_word)
            return words
        
        json_sentences = []
        json_translations = []

        media_aligned = False
        if "media type" in self.meta:
            if self.meta["media type"] in ("video", "audio"):
                media_aligned = True
        
        if media_aligned:
            json_dict["meta"]["adjusted"] = "yes"
        else:
            json_dict["meta"]["adjusted"] = "no"
        
        self.numerate()
        
        for s in self.sentences:
            if len(s.tokens) == 0 and len(s.text) == 0 and (not s.translation or len(s.translation) == 0):
                continue

            tokenize = True if len(s.tokens) == 0 and len(s.text) > 0 else False
            tokens = create_jsons_for_tokens(s, tokenize=tokenize, force_off_values=force_off_values)

            json_sentence = {
                "text": s.text,
                "words": tokens,
                "lang": 0,
                "meta": s.meta
            }
            
            para_alignment, src_alignment = {}, {}

            para_alignment["off_start"] = s.tokens[0].tsakorpus_features["off_start"]
            para_alignment["off_end"] = s.tokens[-1].tsakorpus_features["off_end"]
            
            para_alignment["para_id"] = s._id

            json_sentence["para_alignment"] = [para_alignment]

            if media_aligned:
                if "src_alignment" in s.audio:
                    off_start_src = float(s.audio["src_alignment"][0]["off_start_src"])
                    off_end_src = float(s.audio["src_alignment"][0]["off_end_src"])
                else:
                    off_start_src = float(s.audio["para_alignment"][0]["off_start"])
                    off_end_src = float(s.audio["para_alignment"][0]["off_end"])
                if "MEDIA_URL" in self.audio:
                    src_media = os.path.basename(self.audio["MEDIA_URL"])
                    src_filename, src_extension = os.path.splitext(src_media)[0], os.path.splitext(src_media)[1]
                    if src_filename != self._pure_filename:
                        warnings.warn(message=f"The media file name was changed from {src_filename} to {self._pure_filename}", stacklevel=4)
                        src_media = self._pure_filename + src_extension
                else:
                    src_media = ""
                
                src_alignment = {
                    "off_start_src": off_start_src / 1000,
                    "off_end_src": off_end_src / 1000,
                    "off_start_sent": para_alignment["off_start"],
                    "off_end_sent": para_alignment["off_end"],
                    "mtype": self.meta["media type"],
                    "src_id": f"{off_start_src}_{off_end_src}",
                    "src": src_media
                }

                json_sentence["src_alignment"] = [src_alignment]
            
            json_sentences.append(json_sentence)
            
            if s.translation:
                for key in s.translation:
                    trans_sent = Sentence(text=s.translation[key])
                    translation_tokens = create_jsons_for_tokens(
                        trans_sent, tokenize=True, force_off_values=True)
                    json_translation = {
                        "text": s.translation[key],
                        "words": translation_tokens,
                        "lang": 1 if key == "ru" else key,
                        "meta": s.meta,
                        "para_alignment": [para_alignment]
                    }
                    if media_aligned:
                        json_translation["src_alignment"] = json_sentence["src_alignment"]
                    json_translations.append(json_translation)
        
        json_dict["sentences"].extend(json_sentences)
        json_dict["sentences"].extend(json_translations)
            
        self.json = json_dict
    
    def to_json(self, filepath=None, folderpath="", filename=None, suffix="_new", update_json=False,
                force_off_values=False, force_tokenize=False):
        if update_json:
            self.update_json(force_off_values=force_off_values, force_tokenize=force_tokenize)

        if filename is None:
            filename = self.filename
        if filepath is None:
            filepath = re.sub("\.[^\.]+$", "", filename)
            filepath = f"{filepath}{suffix}.json"
        path = os.path.join(folderpath, filepath)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.json, f, ensure_ascii=False, indent=4)

    def to_df(self):
        return pd.concat([s.to_df() for s in self.sentences], ignore_index=True, sort=False)


class Corpus():
    def __init__(self, folderpath="", tiers=None, texts=None, text_filenames=None, load=True, metadata_path=None):
        self.folderpath = folderpath
        if load:
            current_folder = os.path.dirname(os.path.abspath(__file__))
            abs_folderpath = os.path.join(current_folder, folderpath)
        self.texts = []
        
        self.default_tier_types = tiers
        
        corpus_metadata = None
        if metadata_path and metadata_path.endswith("csv"):
            corpus_metadata = pd.read_csv(metadata_path)

        if not texts and not text_filenames and load:
            
            for f in tqdm(os.listdir(abs_folderpath)):
                abspath = os.path.join(abs_folderpath, f)
                if not (f.endswith(".eaf") or f.endswith(".json")):
                    continue
                if os.path.isfile(abspath):
                    new_text = Text(
                        filename=f,
                        abspath=abspath,
                        tiers=self.default_tier_types,
                        _metadata=corpus_metadata)
                    if new_text:
                        self.texts.append(new_text)
        
        elif texts:
            self.texts.extend(texts)
        
        elif text_filenames:
            for text_filename in text_filenames:
                self.texts.append(Text(
                    filename=text_filename,
                    folderpath=folderpath,
                    tiers=tiers,
                    metadata_path=metadata_path
                ))
        
        self.numerate()

    def __getitem__(self, index):
        """
        Return the Text at the specified index in the list of Texts in the Corpus.

        Parameters:
            index (int): The index of the Text to retrieve.

        Returns:
            Text: The Text object at the specified index.
        """
        return self.texts[index]

    def text(self, query):
        found = []
        for text in self.texts:
            if re.search(query, text.title, flags=re.IGNORECASE):
                found.append(text)
        return found

    def get_id(self, id):
        for t in self.texts:
            if t._id == id:
                return t

    def __repr__(self):
        """
        Return a formal string representation of the Corpus.

        Returns:
            str: A string representation.
        """
        if len(self) == 0:
            return "[]"
        return "[" + ",\n ".join([repr(t) for t in self.texts]) + "]"
    
    def __str__(self):
        return repr(self)
    
    def __len__(self):
        """
        Return the number of Texts in the Corpus.

        Returns:
            int: The number of Texts.
        """
        return len(self.texts)

    def append(self, text):
        self.texts.append(text)

    def extend(self, texts):
        self.texts.extend(texts)
    
    def numerate(self, ignore_punct=False):
        """
        Assign sequential ids to all Texts in the Corpus and all Sentences in each Text.
        """
        if self.texts:
            for i in range(len(self.texts)):
                if self.texts[i]:
                    self.texts[i]._id = i+1
                    self.texts[i].numerate(ignore_punct=ignore_punct)
    
    def remove(self, _id, safe=False):
        """
        Remove a Text from the list of Texts in the Corpus by its id.

        Parameters:
            _id (int): The id of the Text to remove.
            safe (bool): If True, silently ignore if id not found.
            renumerate (bool): If True, reassign ids after removal.
        """
        text_to_remove = None
        if self.texts:
            for i in range(len(self.texts)):
                if self.texts[i]._id == _id:
                    text_to_remove = self.texts[i]
                    break
        if not text_to_remove and safe:
            return

        if text_to_remove and text_to_remove.sentences:
            for j in reversed(range(len(text_to_remove.sentences))):
                text_to_remove.remove(j, safe=safe)
        del self.texts[i]
        self.numerate()
    
    def meta_values(self, key: str):
        return Counter([t.meta[key] for t in self.texts if key in t.meta])

    def search_morpheme(self, gloss=None, morph=None, pos=None, gloss_type=None, full=False, text_filter=None):
        """
        Search for a Morpheme in all Analyses in the Corpus.
        Regular expressions are allowed.
        
        Parameters:
            gloss (str or None): Gloss value to be a condition for the search.
                If None, all glosses will be found (default: None).
            morph (str or None): Morph value to be a condition for the search.
                If None, all morphs will be found (default: None).
            pos (str or None): POS tag value to be a condition for the search.
                If None, all POS tags will be found (default: None).
            gloss_type (str or None): gloss type to be a condition for the search.
                One of the following: 'grammeme', 'lemma' or 'other'.
                If None, glosses of all types will be found (default: None).
            full (bool): if True, morpheme separators will be considered
                during the search (default: False).
            text_filter (dict or None): dictionary with keys and lists of values to apply a text
                filter, e.g. {'subcorpus': ['Folklore', 'Oral texts']} will yield results
                only from texts whose 'subcorpus' value is either 'Folklore' or 'Oral texts'.
                If several keys are provided, both filters are applied.
        
        Returns:
            MorphemeSearch: Results of the search.
        """
        
        check_for_text_filter_integrity(text_filter)
        
        found_morphemes = MorphemeSearch(text_filter=text_filter)
        for t in self.texts:
            if t:
                add_text = apply_text_filter(t, text_filter)
                
                if add_text:
                    found_morphemes.extend(t.search_morpheme(
                        gloss=gloss, morph=morph, pos=pos,
                        gloss_type=gloss_type, full=full))
        return found_morphemes
    
    def search_morphemechain(self, gloss_index_string, pos_string=None, text_filter=None):
        """
        Search for a sequence of Morphemes in all Analyses in the Corpus.
        Regular expressions are allowed.
        
        Parameters:
            gloss_index_string (str): String in the `gloss{morph}` format,
                containing morphemes to be found. If None, all chains will be found.
            pos_string  (str or None): String with POS tags separated by hyphens (default: None).
            text_filter (dict or None): dictionary with keys and lists of values to apply a text
                filter, e.g. {'subcorpus': ['Folklore', 'Oral texts']} will yield results
                only from texts whose 'subcorpus' value is either 'Folklore' or 'Oral texts'.
                If several keys are provided, both filters are applied.
        
        Returns:
            MorphemeChainSearch: Results of the search.
        """
        
        check_for_text_filter_integrity(text_filter)

        _query = Analysis(gloss_index_string = gloss_index_string, pos_string=pos_string)
        found_morphemechains = MorphemeChainSearch(text_filter=text_filter)
        for t in self.texts:
            if t:
                add_text = apply_text_filter(t, text_filter)

                if add_text:
                    found_morphemechains.extend(t.search_morphemechain(_query=_query))
        return found_morphemechains
    
    def search_token(self, token, regex=True, ignore_morphemes=False, ignore_punct=True, ignore_case=True, text_filter=None):
        """
        Search for a Token in the Corpus.
        
        Parameters:
            token (str): Token string to be a condition for the search.
            regex (bool): If True, regular expressions are allowed (default: True).
            ignore_morphemes (bool): if True, morpheme separators are ignored
                during the search (default: False).
            ignore_punct (bool): if True, punctuation in the beginning and in the end
                of tokens is ignored during the search (default: True).
            ignore_case (bool): if True, case of tokens (uppercase / lowercase) is ignored
                during the search (default: True).
            text_filter (dict or None): dictionary with keys and lists of values to apply a text
                filter, e.g. {'subcorpus': ['Folklore', 'Oral texts']} will yield results
                only from texts whose 'subcorpus' value is either 'Folklore' or 'Oral texts'.
                If several keys are provided, both filters are applied.
        
        Returns:
            TokenSearch: Results of the search.
        """
        
        check_for_text_filter_integrity(text_filter)

        found_tokens = TokenSearch(text_filter=text_filter)
        for t in self.texts:
            if t:
                add_text = apply_text_filter(t, text_filter)

                if add_text:
                    for s in t.sentences:
                        if s:
                            found_tokens.extend(
                                s.search_token(
                                    token=token, regex=regex, ignore_morphemes=ignore_morphemes,
                                    ignore_punct=ignore_punct, ignore_case=ignore_case))
        return found_tokens
    
    def search_sentence(self, sentence, regex=True, ignore_morphemes=False,
                        ignore_punct=True, ignore_case=True, text_filter=None):
        """
        Search for a Sentence in the Corpus.
        
        Parameters:
            sentence (str): Sentence string to be a condition for the search.
            regex (bool): If True, regular expressions are allowed (default: True).
            ignore_morphemes (bool): if True, morpheme separators are ignored
                during the search (default: False).
            ignore_punct (bool): if True, punctuation in the beginning and in the end
                of tokens is ignored during the search (default: True).
            ignore_case (bool): if True, case of tokens (uppercase / lowercase) is ignored
                during the search (default: True).
            text_filter (dict or None): dictionary with keys and lists of values to apply a text
                filter, e.g. {'subcorpus': ['Folklore', 'Oral texts']} will yield results
                only from texts whose 'subcorpus' value is either 'Folklore' or 'Oral texts'.
                If several keys are provided, both filters are applied.
        
        Returns:
            SentenceSearch: Results of the search.
        """
        
        check_for_text_filter_integrity(text_filter)

        found_sentences = SentenceSearch(text_filter=text_filter)
        for t in self.texts:
            if t:
                add_text = apply_text_filter(t, text_filter)

                if add_text:
                    found_sentences.extend(
                        t.search_sentence(
                            sentence=sentence, regex=regex,
                            ignore_morphemes=ignore_morphemes, ignore_punct=ignore_punct,
                            ignore_case=ignore_case))
        return found_sentences
    
    def morph_vocab(self, gloss_type=None):
        vocab = Counter()
        for t in self.texts:
            new_vocab = t.morph_vocab(gloss_type)
            for item in new_vocab:
                vocab[item] += new_vocab[item]
        return vocab
    
    def homophones(self, gloss_type=None):
        morphemes = sorted(self.morph_vocab(gloss_type=gloss_type).keys())
        morphemes_dict = {}
        for i in range(len(morphemes)):
            morph, gloss = morphemes[i]
            if not morph in morphemes_dict:
                morphemes_dict[morph] = [gloss]
            else:
                morphemes_dict[morph].append(gloss)
        homophones_dict = {}
        for morph in morphemes_dict:
            if len(morphemes_dict[morph]) > 1:
                homophones_dict[morph] = morphemes_dict[morph]
        return homophones_dict
    
    def to_df(self):
        return pd.concat([t.to_df() for t in self.texts], ignore_index=True, sort=False)