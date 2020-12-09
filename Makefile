.PHONY: clean test lint init docs

JOBS ?= 1

help:
	@echo "make"
	@echo "    clean"
	@echo "        Remove Python/build artifacts."
	@echo "    install"
	@echo "        Install xbot."
	@echo "    formatter"
	@echo "        Apply black formatting to code."
	@echo "    lint"
	@echo "        Lint code with flake8, and check if black formatter should be applied."
	@echo "    lint-docstrings"
	@echo "        Check docstring conventions in changed files."
	@echo "    types"
	@echo "        Check for type errors using mypy."
	@echo "    prepare-tests-ubuntu"
	@echo "        Install system requirements for running tests on Ubuntu and Debian based systems."
	@echo "    prepare-tests-macos"
	@echo "        Install system requirements for running tests on macOS."
	@echo "    prepare-tests-windows"
	@echo "        Install system requirements for running tests on Windows."
	@echo "    prepare-tests-files"
	@echo "        Download all additional project files needed to run tests."
	@echo "    prepare-spacy"
	@echo "        Download all additional resources needed to use spacy as part of Rasa."
	@echo "    prepare-mitie"
	@echo "        Download all additional resources needed to use mitie as part of Rasa."
	@echo "    test"
	@echo "        Run pytest on tests/."
	@echo "        Use the JOBS environment variable to configure number of workers (default: 1)."
	@echo "    livedocs"
	@echo "        Build the docs locally."
	@echo "    release"
	@echo "        Prepare a release."

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*.pyi' -exec rm -f {} +
	rm -rf build/
	rm -rf .mypy_cache/
	rm -rf dist/
	rm -rf docs/build

install:
	poetry run python -m pip install -U pip
	poetry install

lint:
	source ~/.bashrc
    # Ignore docstring errors when running on the entire project
	poetry run flake8 src tests --extend-ignore D
	poetry run black --check src tests
	make lint-docstrings

types:
	source ~/.bashrc
	pytype --disable=import-error src tests docs/conf.py
	mypy src tests docs/conf.py
	pytest --typeguard-packages=src

formatter:
	poetry run black src/xbot tests

safety:
	safety check --full-report

install-docs:
	cd docs/ && yarn install


# Compare against `master` if no branch was provided
BRANCH ?= master
lint-docstrings:
	# Lint docstrings only against the the diff to avoid too many errors.
	# Check only production code. Ignore other flake errors which are captured by `lint`
	# Diff of committed changes (shows only changes introduced by your branch
	ifneq ($(strip $(BRANCH)),)
		git diff $(BRANCH)...HEAD -- rasa | poetry run flake8 --select D --diff
	endif

	# Diff of uncommitted changes for running locally
	git diff HEAD -- rasa | poetry run flake8 --select D --diff

tests:
	# OMP_NUM_THREADS can improve overall performance using one thread by process (on tensorflow), avoiding overload
	OMP_NUM_THREADS=1 poetry run pytest tests --cov xbot

docs:
	cd docs/ && poetry run yarn pre-build && yarn build

livedocs:
	cd docs/ && poetry run yarn start

release:
	poetry run python scripts/release.py
