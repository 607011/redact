FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download de_core_news_md
RUN python -m spacy download de_core_news_sm
RUN python -m spacy download de_core_news_lg
RUN python -m spacy download de_dep_news_trf
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_md
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download en_core_web_trf
RUN python -m spacy download fr_core_news_md

COPY server.py .

EXPOSE 8000

CMD ["python", "./server.py"]