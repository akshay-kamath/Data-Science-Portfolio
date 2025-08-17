
# Download dependencies to /root/.local.
FROM python:3.7-slim AS builder
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .
