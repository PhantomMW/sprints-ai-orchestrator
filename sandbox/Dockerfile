# Pinned official reproducible base image digest for python:3.10-slim (Debian Bookworm)
# Supply chain security: protects against silent upstream tag mutation or malicious drift
FROM python:3.10-slim@sha256:d8ca19faeec9a79fa4f9fb501ba6d8ebfc48dcf32f5d0fdf9fbb50e41cf86c28

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 sandboxuser

WORKDIR /sandbox

USER sandboxuser

CMD ["python", "--version"]
