

import inflection
try:
    import re2 as re
except ImportError:
    import re
from rapidfuzz import fuzz, utils, distance
import rapidfuzz
from functools import lru_cache
from typing import Callable, List
from enum import Enum
import numpy as np
from math import inf

class Dist(Enum):
    DAMERAU = distance.DamerauLevenshtein
    HAMMING = distance.Hamming
    INDEL = distance.Indel
    JARO = distance.Jaro
    WINKLER = distance.JaroWinkler
    LEVENSHTEIN = distance.Levenshtein
    LCS = distance.LCSseq
    OSA = distance.OSA
    PREFIX = distance.Prefix
    POSTFIX = distance.Postfix

class Compare(Enum):
    DISTANCE = "distance"
    SIMILARITY = "similarity"
    NORM_DISTANCE = "normalized_distance"
    NORM_SIMILARITY = "normalized_similarity"

def _build_similarity_func(dist: Dist = Dist.INDEL, compare: Compare = Compare.NORM_SIMILARITY):
    if hasattr(dist._value_, compare._value_): return getattr(dist._value_, compare._value_)
    raise ValueError("Distance/Compare function not defined")

def _get_processor(a):
    return a if isinstance(a, Callable) else (utils.default_process if (a is None) else utils.default_process)

def sort_list_by_similarity(target, choices, partial=True, descending=True, return_scores=True):
    result = similarity(target, choices, partial=partial, return_scores=return_scores)
    return result if descending else result[::-1]

def similarity_score(a, b, partial=True):
    result = similarity(a, b, partial=partial, return_scores=True)
    result = [x[1] for x in result]
    return result[0] if len(result) == 1 else result

def extract_most_similar(target, list_to_compare, partial=True, return_scores=False):
    return similarity(target, list_to_compare, partial=partial, top=1, return_scores=return_scores)[0]

def similarity(target, choices, *, top:int=None, return_scores:bool=True, partial:bool=True, clean_str:bool|Callable=True, dist:Dist=Dist.INDEL, compare:Compare= Compare.NORM_SIMILARITY, workers=1):
    choices = [choices] if isinstance(choices, str) else choices
    return _similarity(target, tuple(choices), top=top, return_scores=return_scores, partial=partial, clean_str=clean_str, dist=dist, compare=compare, workers=workers)

@lru_cache(maxsize=128)
def _similarity(target, choices, *, top:int=None, return_scores:bool=True, partial:bool=True, clean_str:bool|Callable=True, dist:Dist=Dist.INDEL, compare:Compare= Compare.NORM_SIMILARITY, workers=1):
    processor = _get_processor(clean_str)
    func = _build_similarity_func(dist, compare)
    if partial:
        result = [rapidfuzz.process.cdist([target[:i]], choices, scorer=func, processor=processor, workers=workers) for i in range(len(target), 0, -1)] + [rapidfuzz.process.cdist([target[i:]], choices, scorer=func, processor=processor, workers=workers) for i in range(0, len(target))]
        scores = np.amax(np.array(result), axis=0)[0]
        best = np.where(scores==np.max(scores))[0]
        if best.size > 1:
            besters = [x[0] if x else None for x in [np.where(x[0]==np.max(scores))[0].tolist() for x in result]]
            result = sorted(zip(choices, [besters.index(x) if x in besters else inf for x in list(range(len(choices)))], scores), reverse=False, key=lambda x: x[1])
            result = [(x[0], x[2]) for x in result]
            result = result if return_scores else [x[0] for x in result]
            return result if not top else result[:top]
    else:
        scores = rapidfuzz.process.cdist([target], choices, scorer=func, processor=processor, workers=workers)[0]

    result = sorted(zip(choices, scores.tolist()), reverse=True, key=lambda x: x[1])
    result = result if return_scores else [x[0] for x in result]
    return result if not top else result[:top]


def split_by_multiple(text, delimiters):
    pattern = '|'.join(map(re.escape, delimiters))
    return [x for x in re.split(pattern, text) if x]

def clean_camel(*args):
    if args is None: return
    s = "_".join(args)

    out = []
    out_append = out.append
    word_index = -1
    pos_in_word = 0
    in_word = False

    prev_is_lower = False
    prev_is_upper = False
    prev_is_digit = False

    n = len(s)
    i = 0

    while i < n:
        o = ord(s[i])

        if o == 95 or o == 45 or o == 32 or o == 9 or o == 10 or o == 13 or o == 11 or o == 12:
            in_word = False
            prev_is_lower = prev_is_upper = prev_is_digit = False
            i += 1
            continue

        is_upper = 65 <= o <= 90
        is_lower = 97 <= o <= 122
        is_digit = 48 <= o <= 57

        if not (is_upper or is_lower or is_digit):
            out_append(s[i])

            if o == 35:
                in_word = False
                pos_in_word = 0

            prev_is_lower = prev_is_upper = prev_is_digit = False
            i += 1
            continue

        if is_upper and i + 1 < n:
            on = ord(s[i + 1])
            next_is_lower = 97 <= on <= 122
        else:
            next_is_lower = False

        start_new_word = (
                (not in_word) or
                (is_upper and prev_is_lower) or
                (is_upper and prev_is_upper and next_is_lower) or
                (is_upper and prev_is_digit)
        )

        if start_new_word:
            word_index += 1
            pos_in_word = 0
            in_word = True

        if is_upper or is_lower:
            if word_index <= 0:
                if is_upper:
                    o += 32
            else:
                if pos_in_word == 0:
                    if is_lower:
                        o -= 32
                else:
                    if is_upper:
                        o += 32

        out_append(chr(o))

        prev_is_lower = is_lower
        prev_is_upper = is_upper
        prev_is_digit = is_digit
        pos_in_word += 1
        i += 1

    return ''.join(out)

def camelize(text, first=False):
    if text is None: return
    # If all uppercase, convert to lowercase first
    if text.isupper():
        text = text.lower()

    if not first and text[0].isupper():
        text = text[0].lower() + text[1:]

    # Default case: use humps
    return inflection.camelize(text, first)


def format_number(number, custom_config=None):
    default_config = {
        "showSign": False,
        "prefix": "",
        "postfix": "",
        "sigFigs": {
            "global": None,  # overrides all
            "normal": 4,  # 0 means all digits
            "thousand": 2,
            "million": 1,
            "billion": 1
        },
        "thresholds": {
            "thousand": 1000,
            "million": 1000000,
            "billion": 1000000000
        },
        "units": {
            "thousand": "k",
            "million": "mio",
            "billion": "bn"
        },
        "spacing": 0,
        'none_is_zero': False
    }

    # Helper functions
    def _format_with_sig_figs(num, sig_figs):
        if sig_figs == -1:
            return str(num)
        return f"{num:.{sig_figs}f}"

    # Main formatting logic
    if custom_config is None:
        custom_config = {}

    # Create a new config dictionary with default values
    config = dict(default_config)

    # Update top-level keys from custom_config
    for key in custom_config:
        if key in ["sigFigs", "thresholds", "units"]:
            # For nested dictionaries, merge them separately
            config[key] = dict(default_config[key])
            config[key].update(custom_config[key] or {})
        else:
            config[key] = custom_config[key]

    if number is None: return 0 if config['none_is_zero'] else None

    try:
        number = float(number)
    except (ValueError, TypeError):
        return str(number)

    abs_num = abs(number)
    sign = "-" if number < 0 else ("+" if config["showSign"] else "")
    prefix = config.get("prefix", "")
    postfix = config.get("postfix", "")
    spacing = " " * config.get("spacing", 0)

    if config["sigFigs"]["global"] == 0:
        config["sigFigs"]["global"] = -1

    if abs_num != abs_num or abs_num == float('inf'):  # Check for NaN or Infinity
        return str(number)

    if abs_num < config["thresholds"]["thousand"]:
        sig_figs = config["sigFigs"]["global"] or config["sigFigs"]["normal"]
        sig_figs = max(0, sig_figs)
        return f"{prefix}{sign}{_format_with_sig_figs(abs_num, sig_figs)}{postfix}"

    if abs_num < config["thresholds"]["million"]:
        sig_figs = config["sigFigs"]["global"] or config["sigFigs"]["thousand"]
        sig_figs = max(0, sig_figs)
        unit = config["units"]["thousand"] or ""
        return f"{prefix}{sign}{_format_with_sig_figs(abs_num / 1000, sig_figs)}{spacing}{unit}{postfix}"

    if abs_num < config["thresholds"]["billion"]:
        sig_figs = config["sigFigs"]["global"] or config["sigFigs"]["million"]
        sig_figs = max(0, sig_figs)
        unit = config["units"]["million"] or ""
        return f"{prefix}{sign}{_format_with_sig_figs(abs_num / 1000000, sig_figs)}{spacing}{unit}{postfix}"

    sig_figs = config["sigFigs"]["global"] or config["sigFigs"]["billion"]
    sig_figs = max(0, sig_figs)
    unit = config["units"]["billion"] or ""
    return f"{prefix}{sign}{_format_with_sig_figs(abs_num / 1000000000, sig_figs)}{spacing}{unit}{postfix}"


