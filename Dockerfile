FROM python:3.9-slim

WORKDIR /usr/src/app

ENV PYTHONPATH /usr/src/app

COPY . .

# Install Poetry
RUN pip install poetry

# Install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

CMD ["python"]
