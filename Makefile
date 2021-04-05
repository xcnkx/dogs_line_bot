.PHONY: setup
setup: 
	POETRY_VIRTUALENVS_IN_PROJECT=true poetry install

.PHONY: run
run: 
	./.venv/bin/python app.py