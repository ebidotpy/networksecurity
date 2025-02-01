FROM python:3.12.8-alpine
WORKDIR /app
COPY . /app/

RUN apt-get update && pip install -r requirements.txt
CMD ["python3", "app.py"]


