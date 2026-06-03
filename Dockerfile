from pytorch/pytorch:latest

RUN pip install numpy pandas nltk fastapi uvicorn celery[redis] redis httpx transformers

COPY main.py /src/main.py
COPY tasks.py /src/tasks.py
COPY finbert /src/finbert
COPY models /src/models

EXPOSE  8080
CMD ["python3", "/src/main.py"]
