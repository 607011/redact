# redac’t

**Linguistically proven redactor for plain text files**

## Prerequisites

- Python 3.11 or newer
- Pipenv

## Prepare

Install modules:

```bash
pipenv install
```

Download language files, e.g. a basic model for English

```bash
pipenv run python -m spacy download en_core_web_sm
```

## Run

```bash
pipenv run ./redact.py \
  -i test.txt \
  --level 80 \
  --model en_core_web_sm
```
