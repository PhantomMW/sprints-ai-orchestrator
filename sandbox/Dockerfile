# Official slim Python 3.10 base image
# Standardized across development, CI, and local test runners
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 sandboxuser

WORKDIR /sandbox

USER sandboxuser

CMD ["python", "--version"]
