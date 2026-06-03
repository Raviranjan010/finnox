from pytorch/pytorch:latest

RUN pip install numpy pandas nltk fastapi uvicorn celery[redis] redis httpx transformers spacy spacy-transformers shap numba
RUN python -m spacy download en_core_web_sm

COPY main.py /src/main.py
COPY tasks.py /src/tasks.py
COPY finbert /src/finbert
COPY models /src/models

EXPOSE  8080
CMD ["python3", "/src/main.py"]
