FROM python:3.8.3 as base

WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


ENV PYTHONPATH /app
COPY . . 

CMD [ "bash" ]

# Fast API docker image
FROM base AS app-fastapi
CMD uvicorn --host 0.0.0.0 --port 8080 --workers 1 serving.fast_api:app
