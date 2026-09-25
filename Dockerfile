FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt requirements-sota.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-sota.txt

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "neuroforge.examples.run_demo"]
