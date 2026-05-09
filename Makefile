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
	$(PYTHON) -c "from semanticsd import config, paths; from semanticsd.db import connection, migrations; \
	paths.ensure_dirs(); \
	cfg = config.DEFAULT_TOML.replace('directories = []','directories = [\"$(PWD)/sandbox\"]').replace('http_port = 47600','http_port = 47601'); \
	open(paths.config_path(),'w').write(cfg); \
	migrations.apply(connection.get_connection(paths.db_path()))" && \
	SEMANTICSD_HOME=$(PWD)/sandbox/.semanticsd $(PYTHON) -m semanticsd serve
