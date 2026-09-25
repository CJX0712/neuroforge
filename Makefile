# NeuroForge Makefile —— 一键复现
# 干净环境：make install && make demo && make test && make benchmark
VENV ?= .venv
PY := $(VENV)/Scripts/python

install:
	python -m venv $(VENV) || $(PY) -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -r requirements-sota.txt

demo:
	$(PY) -m neuroforge.examples.run_demo

test:
	$(PY) -m pytest -q

benchmark:
	$(PY) scripts/benchmark.py

clean:
	rm -rf $(VENV) models logs __pycache__ .pytest_cache *.egg-info benchmark_report.json
