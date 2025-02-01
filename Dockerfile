FROM python:3.12.8-slim
WORKDIR /app

# Change the shell to bash for the virtual environment activation
SHELL ["/bin/bash", "--", "-c"]  # Important: This is the correct way to change the shell

RUN python3 -m venv ./venv
RUN source ./venv/bin/activate

COPY requirements.txt .
RUN .venv/bin/pip install -r requirements.txt

COPY . .

RUN apt-get update && pip install -r requirements.txt
CMD [".venv/bin/python", "app.py"]


