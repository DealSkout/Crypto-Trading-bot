FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY cryptobot ./cryptobot
RUN pip install --no-cache-dir -e ".[claude]"
# config.yaml and .env are mounted at runtime; cache/ reports/ and the journal persist in /app/state
VOLUME ["/app/state"]
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "cryptobot", "paper"]
