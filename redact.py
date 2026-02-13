#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
from enum import Enum


class RedactionMode(Enum):
    PHRASE = "phrase"
    SINGLE = "single"


def redact_segment(result: list, start: int, end: int, redact_char: str):
    for i in range(start, end):
        if not result[i].isspace():
            result[i] = redact_char


def score_token(token):
    if token.ent_type_:
        return 1.0
    elif token.pos_ in ["NUM", "GPE", "LOC", "ORG", "PERSON"]:
        return 0.9
    elif token.pos_ in ["NOUN", "PROPN"]:
        return 0.75
    elif token.pos_ in ["VERB", "ADJ", "ADV"]:
        return 0.6
    else:
        return 0.1


def redact(text: str, redaction_mode: list[RedactionMode], level: int = 70) -> str:
    threshold = level / 100.0
    doc = nlp(text)
    result = list(text)
    if RedactionMode.PHRASE in redaction_mode:
        # redact entire noun phrases based on the specified level
        for chunk in doc.noun_chunks:
            importance = sum(score_token(tok) for tok in chunk) / len(chunk)
            if importance <= threshold:
                redact_segment(result, chunk.start_char, chunk.end_char, "█")
    if RedactionMode.SINGLE in redaction_mode:
        # redact individual tokens based on their importance and the specified level
        for token in doc:
            importance = score_token(token)
            if importance <= threshold:
                redact_segment(result, token.idx, token.idx + len(token), "█")
    return "".join(result)


def main() -> None:
    global nlp
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input file path")
    parser.add_argument("-o", "--output", type=str, help="Output file path")
    parser.add_argument("-l", "--level", type=int, default=80, help="Redaction level")
    parser.add_argument(
        "--mode",
        type=str,
        nargs="+",
        default=[RedactionMode.PHRASE.value],
        choices=[mode.value for mode in RedactionMode],
        help="Redaction mode(s)",
    )
    parser.add_argument(
        "-m", "--model", type=str, default="de_core_news_md", help="Spacy model to use"
    )
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            input_text = f.read()
    else:
        input_text = sys.stdin.read()

    import spacy

    nlp = spacy.load(args.model, disable=["lemmatizer"])
    redacted = redact(
        input_text, [RedactionMode(mode) for mode in args.mode], args.level
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(redacted)
    else:
        print(redacted)


if __name__ == "__main__":
    main()
