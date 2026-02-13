#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from typing import Set
import spacy
import sys
import argparse
from enum import Enum

class RedactionMode(Enum):
    PHRASE = "phrase"
    SINGLE = "single"

def redact_segment(result: list, start: int, end: int, redact_char: str):
    for i in range(start, end):
        if result[i].isalnum():
            result[i] = redact_char

def redact(text: str, level: int = 70, redaction_mode: RedactionMode = RedactionMode.PHRASE) -> str:
    threshold = (100 - level) / 100.0
    doc = nlp(text)
    result = list(text)
    if redaction_mode == RedactionMode.PHRASE:
        # randomly redact entire noun phrases based on the specified level
        for chunk in doc.noun_chunks:
            if level > random.randint(0, 100):
                redact_segment(result, chunk.start_char, chunk.end_char, "█")
    elif redaction_mode == RedactionMode.SINGLE:
        # redact individual tokens based on their importance and the specified level
        for token in doc:
            importance = 0.0
            # Level 1: hard facts (specific entities, if recognized)
            if token.ent_type_:
                importance = 1.0
            # Level 2: semantic cores (nouns & proper nouns)
            elif token.pos_ in ["NOUN", "PROPN", "NUM", "GPE", "LOC", "ORG", "PERSON"]:
                importance = 0.8
            # Level 3: descriptions & actions (verbs & adjectives)
            elif token.pos_ in ["VERB", "ADJ", "ADV", "NUM"]:
                importance = 0.6
            # Level 4: structural words (pronouns)
            elif token.pos_ in ["PRON"]:
                importance = 0.4
            # Everything else (articles, conjunctions, auxiliary verbs) remains at 0.1 or 0.0
            else:
                importance = 0.1
            if importance > threshold:
                start = token.idx
                end = token.idx + len(token.text)
                for i in range(start, end):
                    if not text[i].isspace():
                        result[i] = "█"
    return "".join(result)

def main() -> None:
    global nlp
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input file path")
    parser.add_argument("-l", "--level", type=int, default=80, help="Redaction level")
    parser.add_argument("-e", "--mode", type=str, default=RedactionMode.PHRASE.value, choices=[mode.value for mode in RedactionMode], help="Redaction mode")
    parser.add_argument("-m", "--model", type=str, default="de_core_news_md", help="Spacy model to use")
    args = parser.parse_args()
    
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            input_text = f.read()
    else:
        input_text = sys.stdin.read()
    
    nlp = spacy.load(args.model)
    redacted = redact(input_text, args.level, RedactionMode(args.mode))
    print(redacted)


if __name__ == "__main__":
    main()