# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

# Vietnamese diacritics (NFC). Unaccented Vietnamese is handled in LID, not here.
_VI_DIACRITICS = set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)

# Marks that are not also routine French/Spanish/Portuguese accents. Used as a
# *word-level* Vietnamese vote, never as a whole-sentence short-circuit.
_VI_DISTINCTIVE_DIACRITICS = set(
    "ăằắẳẵặơờớởỡợưừứửữựđảạẻẹẽỉịỏọủụỷỵỹĩũầấẩẫậềếểễệồốổỗộỳ"
    "ĂẰẮẲẴẶƠỜỚỞỠỢƯỪỨỬỮỰĐẢẠẺẸẼỈỊỎỌỦỤỶỴỸĨŨẦẤẨẪẬỀẾỂỄỆỒỐỔỖỘỲ"
)

_VI_CHAR_MAP = str.maketrans(
    {
        "à": "a",
        "á": "a",
        "ả": "a",
        "ã": "a",
        "ạ": "a",
        "ă": "a",
        "ằ": "a",
        "ắ": "a",
        "ẳ": "a",
        "ẵ": "a",
        "ặ": "a",
        "â": "a",
        "ầ": "a",
        "ấ": "a",
        "ẩ": "a",
        "ẫ": "a",
        "ậ": "a",
        "è": "e",
        "é": "e",
        "ẻ": "e",
        "ẽ": "e",
        "ẹ": "e",
        "ê": "e",
        "ề": "e",
        "ế": "e",
        "ể": "e",
        "ễ": "e",
        "ệ": "e",
        "ì": "i",
        "í": "i",
        "ỉ": "i",
        "ĩ": "i",
        "ị": "i",
        "ò": "o",
        "ó": "o",
        "ỏ": "o",
        "õ": "o",
        "ọ": "o",
        "ô": "o",
        "ồ": "o",
        "ố": "o",
        "ổ": "o",
        "ỗ": "o",
        "ộ": "o",
        "ơ": "o",
        "ờ": "o",
        "ớ": "o",
        "ở": "o",
        "ỡ": "o",
        "ợ": "o",
        "ù": "u",
        "ú": "u",
        "ủ": "u",
        "ũ": "u",
        "ụ": "u",
        "ư": "u",
        "ừ": "u",
        "ứ": "u",
        "ử": "u",
        "ữ": "u",
        "ự": "u",
        "ỳ": "y",
        "ý": "y",
        "ỷ": "y",
        "ỹ": "y",
        "ỵ": "y",
        "đ": "d",
    }
)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", nfc(text)).strip()


def has_vietnamese_diacritics(text: str) -> bool:
    return any(ch in _VI_DIACRITICS for ch in nfc(text))


def has_distinctive_vietnamese_diacritics(text: str) -> bool:
    """True for horn/breve/stroke/hook/dot or circumflex+tone, not é/à/ô alone."""
    return any(ch in _VI_DISTINCTIVE_DIACRITICS for ch in nfc(text))


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in text or "")


def strip_vietnamese_diacritics(text: str) -> str:
    folded = nfc(text).translate(_VI_CHAR_MAP)
    return unicodedata.normalize("NFC", folded)


def fold_marks(text: str) -> str:
    """Strip combining marks (NFKD). Language-agnostic; not a Vietnamese map."""
    s = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def has_non_ascii_letter(text: str) -> bool:
    return any(ch.isalpha() and not ch.isascii() for ch in text or "")


def normalize_for_copy(text: str) -> str:
    """Case/whitespace/punct fold used only to detect copied question text."""
    s = fold_marks(collapse_ws(text)).casefold()
    s = _PUNCT.sub(" ", s)
    return collapse_ws(s)


def tokenize_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[^\W_]+", nfc(text), flags=re.UNICODE) if w]


_LATEX_CMD = re.compile(r"\\[a-zA-Z]+")
_MATH_PUNCT = re.compile(r"[{}$^_\\\[\]().,:;=+\-*/<>|~]")


def is_numeric_or_symbol(text: str) -> bool:
    """True when the span is numbers / LaTeX / punctuation only (answer-lang exempt)."""
    s = collapse_ws(text)
    if not s:
        return True
    s = _LATEX_CMD.sub(" ", s)
    s = _MATH_PUNCT.sub(" ", s)
    s = collapse_ws(s)
    if not s:
        return True
    for ch in s:
        if ch.isalpha():
            return False
    return True
