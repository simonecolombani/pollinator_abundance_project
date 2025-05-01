# Pollinator Abundance Project

This project analyzes pollinator abundance and suitability within specified geographical areas. It calculates various Key Performance Indicators (KPIs) related to pollinators, such as Pollinator Abundance (PA), Nesting Suitability (NS), Floral Availability (FA), Nectar Potential (NP), and Mean Species Abundance (MSA), based on Corine Land Cover (CLC) data and specific bee species characteristics.

## Overview

The core functionality resides in the `handler.py` module, specifically the `pollinator_abundance_calculation` function. This function orchestrates a pipeline that:

1.  **Loads Data:** Reads pre-defined CLC data (from `constants.py`), bee species characteristics (hardcoded in `handler.py`), and image data (CLC maps as `.npy` files stored within the package's `data` directory).
2.  **Processes Images:** Uses functions from `image_processing.py` to handle image loading, merging (Region of Interest - ROI onto Context Area - CA), masking, resizing, and color mapping based on CLC values.
3.  **Calculates KPIs:** Leverages mathematical models (primarily from `math_v2.py`) to calculate FA, NS, PA, NP, and MSA based on the CLC map data and bee parameters (like foraging distance `alpha`). Bee-specific calculations are handled, potentially in parallel.
4.  **Generates Reports:** Utilizes `reporting.py` and `element.py` to create visual outputs (maps/images) for each calculated KPI, complete with legends, scales, and titles.

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.11**: This project requires Python version 3.11.
2.  **uv**: This project uses `uv` for environment and package management. You can install it by following the instructions on the [official uv GitHub repository](https://github.com/astral-sh/uv). The `Makefile` will check if `uv` is available in your `PATH`.
3.  **System Dependencies**: Some Python packages (like `opencv-python`) might require system-level libraries (e.g., C++ compilers, image format libraries). Ensure these are installed if you encounter installation issues.

## Installation

The `Makefile` simplifies the setup process. To create the virtual environment and install the necessary dependencies (listed in `pyproject.toml`), including the project package in editable mode:

```bash
make venv
```

This command will:
1. Check if `uv` is installed.
2. Create a Python 3.11 virtual environment named `.venv` using `uv` if it doesn't already exist.
3. Install the project package (`pollinator_abundance`) in editable mode (`pip install -e .`) along with its dependencies using `uv`.

Key dependencies include: `numpy`, `Pillow`, `opencv-python`, `requests`, `upolygon`, `mypy`, `ruff`.

## Project Structure

The main source code is located within the `src/pollinator_abundance/` directory. Key modules include:

* `main.py`: The main entry point for running the calculation.
* `handler.py`: Orchestrates the main calculation workflow.
* `constants.py`: Contains CLC data definitions.
* `math_v1.py`, `math_v2.py`: Core mathematical algorithms for KPI calculations.
* `image_processing.py`: Functions for image manipulation and processing.
* `reporting.py`: Functions for generating report images.
* `element.py`: Generates specific KPI report elements.
* `basic.py`: Basic utility functions.
* `logconf.py`: Logging configuration.
* `data/`: (Assumed location within the installed package) Contains necessary data files like `.npy` images and fonts.

## Usage

This project uses a `Makefile` to streamline common tasks.

**Running the Main Calculation:**

To execute the main analysis pipeline defined in `src/pollinator_abundance/main.py`:

```bash
make run
```

This command uses `uv run` to execute the script within the managed virtual environment. The script currently uses hardcoded parameters (like `plantation_id`, `roi_id`, `ca_id`) within `handler.py`.

**Other `make` commands:**

* `make help`: Displays a help message listing all available commands.
* `make venv`: Creates the virtual environment and installs dependencies.
* `make show`: Displays details about the current `uv`-managed environment.
* `make fmt`: Formats, lints, and type-checks the code using `ruff` and `mypy`.
* `make clean`: Removes the `.venv` directory.

**Example Workflow:**

1.  **Set up the environment:**
    ```bash
    make venv
    ```
2.  **Run the main calculation:**
    ```bash
    make run
    ```
    *(Note: Modify hardcoded parameters in `handler.py` if needed for different inputs).*
3.  **Format and check the code (during development):**
    ```bash
    make fmt
    ```

## Data

* **CLC Data:** Defined as Python lists of dictionaries in `constants.py` (`CLC_VALUES`, `CLC_VALUES_ROI`, `CLC_VALUES_CA`). These map CLC codes/colors to various attributes (fa, ns, msa, pn_mean, etc.).
* **Bee Data:** Characteristics for different bee species (e.g., nesting type, foraging distance `alpha`) are currently hardcoded as a multi-line string (`DATA_BEE_STR`) within `handler.py`.
* **Image Data:** The calculation relies on pre-processed CLC map images stored as NumPy arrays (`.npy` files) within the package's `data/` directory (e.g., `image_roi.npy`, `image_ca.npy`). Font files for reporting are also expected there.

## Configuration

The main calculation function (`handler.pollinator_abundance_calculation`) currently uses **hardcoded** values for inputs like:

* `plantation_id`, `roi_id`, `ca_id`
* Image alignment points (`alignment_point_x`, `alignment_point_y`)
* Pixel-to-meter ratios (`ratio_x`, `ratio_y`)
* Calculation resolution (`resolution` parameter mapping to `min_res`)

These might need to be modified directly in the code or refactored to accept dynamic inputs for different analysis scenarios.

## Output

The `make run` command executes the calculation pipeline. While the specific output mechanism isn't fully detailed (e.g., saving files vs. returning values), the code suggests the generation of:

* **Numerical Results:** Aggregated KPI values (PA, NS, FA, NP, MSA) for the ROI and CA, potentially stored in the `result_values` dictionary within the `handler`.
* **Image Reports:** Visual maps for different KPIs (CLC, NP, FA, MSA, NS, PA per nesting group, total PA, total NS) generated by `reporting.py` and `element.py`. The exact saving location/format is not specified in the provided code snippets but likely involves saving image files (e.g., PNG, WebP).

## Development

To ensure code quality and consistency, use the `fmt` command:

```bash
make fmt
```

This runs `ruff format`, `ruff check --fix`, and `mypy` on the `src/pollinator_abundance/` directory.

## Cleaning Up

To remove the virtual environment directory:

```bash
make clean
```

This will delete the `.venv` folder. You will need to run `make venv` again to recreate it.


### 
