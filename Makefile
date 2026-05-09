.PHONY: install dev test clean uninstall

PY := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest

$(VENV)/bin/activate:
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

dev: install
	$(PYTHON) -m semanticsd serve

test: install
	$(PYTEST) -v

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

uninstall:
	bash scripts/install.sh --uninstall

.PHONY: dev-sandbox

dev-sandbox: install
	SEMANTICSD_HOME=$(PWD)/sandbox/.semanticsd \
	$(PYTHON) -c "from semanticsd import config, paths; paths.ensure_dirs(); \
	open(paths.config_path(),'w').write(config.DEFAULT_TOML.replace('directories = []','directories = [\"$(PWD)/sandbox\"]'))" && \
	SEMANTICSD_HOME=$(PWD)/sandbox/.semanticsd $(PYTHON) -m semanticsd serve
