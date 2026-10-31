# Official slim Python 3.10 base image with verified multi-platform manifest digest
# Pinned for immutable builds and supply-chain reproducibility
ARG BASE_IMAGE=python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 sandboxuser

WORKDIR /sandbox

USER sandboxuser

CMD ["python", "--version"]
