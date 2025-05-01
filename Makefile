# Check if uv is installed
UV := $(shell command -v uv 2> /dev/null)
ifeq ($(UV),)
$(error "uv command not found in PATH. Please install uv: https://github.com/astral-sh/uv")
endif

.ONESHELL:
ENV_NAME="pollinator_abundance" # You can keep this for descriptive messages

# Default target
.DEFAULT_GOAL := help

# Phony targets definition
.PHONY: help venv show fmt run clean

help: ## Show this help message.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

venv: ## Create .venv (Python 3.11) if missing, install package editable.
	@if [ ! -d .venv ]; then \
		echo "Creating Python 3.11 virtual environment '.venv' using uv..."; \
		uv venv -p 3.11 || { echo "Error: Failed to create venv with Python 3.11. Ensure Python 3.11 is available to uv."; exit 1; }; \
		echo "Virtual environment created."; \
	else \
		echo "Virtual environment '.venv' already exists."; \
	fi
	@echo "Installing local package in editable mode (pip install -e .)..."
	@uv pip install -e .
	@echo "Editable install complete. Use 'make show' to see details."

show: venv ## Show details about the current uv-managed environment.
	@echo "Current $(ENV_NAME) environment (managed by uv):"
	@echo "uv version:"
	@uv --version
	@echo "Python version:"
	@uv run python -V
	@echo "Installed packages:"
	@uv pip list

fmt: venv ## Format, lint, and type-check code using uv run.
	@echo "Running formatters, linters, and type checkers via 'uv run'..."
	@echo "--- Formatting (ruff format) ---"
	@uv run ruff format src/pollinator_abundance/
	@echo "--- Linting & Autofixing (ruff check) ---"
	@uv run ruff check --fix src/pollinator_abundance/
	@echo "--- Type Checking (mypy) ---"
	@uv run mypy src/pollinator_abundance/
	@echo "Formatting, linting, and type checking complete."

run: venv ## Run the main application script.
	@echo "Running the main application script (src/pollinator_abundance/main.py) using uv run..."
	@uv run python src/pollinator_abundance/main.py

clean: ## Remove the .venv directory.
	@echo "Removing .venv directory..."
	@rm -rf .venv
