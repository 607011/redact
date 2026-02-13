#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from typing import Set
import spacy
import sys
import argparse


ENTITY_LABELS: Set[str] = {
    "PERSON", "ORG", "GPE", "LOC", "NUM", "PROPN", "MISC"
}


def redact_segment(result: list, start: int, end: int, redact_char: str):
    for i in range(start, end):
        if result[i].isalnum():
            result[i] = redact_char

def redact(text: str, level: int = 70) -> str:
    doc = nlp(text)
    result = list(text)
    for chunk in doc.noun_chunks:
        if level > random.randint(0, 100):
            redact_segment(result, chunk.start_char, chunk.end_char, "█")
    return "".join(result)

def main() -> None:
    global nlp
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input file path")
    parser.add_argument("-l", "--level", type=int, default=100, help="Redaction level")
    parser.add_argument("-m", "--model", type=str, default="de_core_news_md", help="Spacy model to use")
    args = parser.parse_args()
    
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            input_text = f.read()
    else:
        input_text = sys.stdin.read()
    
    nlp = spacy.load(args.model)
    redacted = redact(input_text, args.level)
    print(redacted)


if __name__ == "__main__":
    main()