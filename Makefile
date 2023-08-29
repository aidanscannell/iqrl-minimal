.PHONY: run clean paper

VENV = .venv
# VENV = .direnv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

$(VENV)/bin/activate: setup.py
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install laplace-torch==0.1a2
	$(PIP) install -e ".[experiments, dev]"

clean:
	rm -rf __pycache__
	rm -rf $(VENV)
	rm ${PAPER_DIR}/${AUX_DIR}

