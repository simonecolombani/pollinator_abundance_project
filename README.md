# Pollinator Abundance Project

This project analyzes pollinator abundance and suitability within specified geographical areas. It calculates various
Key Performance Indicators (KPIs) related to pollinators, such as Pollinator Abundance (PA), Nesting Suitability (NS),
Floral Availability (FA), Nectar Potential (NP), and Mean Species Abundance (MSA), based on Corine Land Cover (CLC) data
and specific bee species characteristics.

## Overview

The core functionality resides in the `handler.py` module, specifically the `pollinator_abundance_calculation` function.
This function orchestrates a pipeline that:

1. **Loads Data:** Reads pre-defined CLC data (from `constants.py`), bee species characteristics (hardcoded in
   `handler.py`), and image data (CLC maps as `.npy` files stored within the package's `data` directory).
2. **Processes Images:** Uses functions from `image_processing.py` to handle image loading, merging (Region of
   Interest - ROI onto Context Area - CA), masking, resizing, and color mapping based on CLC values.
3. **Calculates KPIs:** Leverages mathematical models (primarily from `math_v2.py`) to calculate FA, NS, PA, NP, and MSA
   based on the CLC map data and bee parameters (like foraging distance `alpha`). Bee-specific calculations are handled,
   potentially in parallel.
4. **Generates Reports:** Utilizes `reporting.py` and `element.py` to create visual outputs (maps/images) for each
   calculated KPI, complete with legends, scales, and titles.

## Prerequisites

Before you begin, ensure you have the following installed:

1. **Python 3.11**: This project requires Python version 3.11.
2. **uv**: This project uses `uv` for environment and package management. You can install it by following the
   instructions on the [official uv GitHub repository](https://github.com/astral-sh/uv). The `Makefile` will check if
   `uv` is available in your `PATH`.
3. **System Dependencies**: Some Python packages (like `opencv-python`) might require system-level libraries (e.g., C++
   compilers, image format libraries). Ensure these are installed if you encounter installation issues.

## Installation

The `Makefile` simplifies the setup process. To create the virtual environment and install the necessary dependencies (
listed in `pyproject.toml`), including the project package in editable mode:

```bash
make venv
```

This command will:

1. Check if `uv` is installed.
2. Create a Python 3.11 virtual environment named `.venv` using `uv` if it doesn't already exist.
3. Install the project package (`pollinator_abundance`) in editable mode (`pip install -e .`) along with its
   dependencies using `uv`.

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

This command uses `uv run` to execute the script within the managed virtual environment. The script currently uses
hardcoded parameters (like `plantation_id`, `roi_id`, `ca_id`) within `handler.py`.

**Other `make` commands:**

* `make help`: Displays a help message listing all available commands.
* `make venv`: Creates the virtual environment and installs dependencies.
* `make show`: Displays details about the current `uv`-managed environment.
* `make fmt`: Formats, lints, and type-checks the code using `ruff` and `mypy`.
* `make clean`: Removes the `.venv` directory.

**Example Workflow:**

1. **Set up the environment:**
   ```bash
   make venv
   ```
2. **Run the main calculation:**
   ```bash
   make run
   ```
   *(Note: Modify hardcoded parameters in `handler.py` if needed for different inputs).*
3. **Format and check the code (during development):**
   ```bash
   make fmt
   ```

## Data

* **CLC Data:** Defined as Python lists of dictionaries in `constants.py` (`CLC_VALUES`, `CLC_VALUES_ROI`,
  `CLC_VALUES_CA`). These map CLC codes/colors to various attributes (fa, ns, msa, pn_mean, etc.).
* **Bee Data:** Characteristics for different bee species (e.g., nesting type, foraging distance `alpha`) are currently
  hardcoded as a multi-line string (`DATA_BEE_STR`) within `handler.py`.
* **Image Data:** The calculation relies on pre-processed CLC map images stored as NumPy arrays (`.npy` files) within
  the package's `data/` directory (e.g., `image_roi.npy`, `image_ca.npy`). Font files for reporting are also expected
  there.

## Configuration

The main calculation function (`handler.pollinator_abundance_calculation`) currently uses **hardcoded** values for
inputs like:

* `plantation_id`, `roi_id`, `ca_id`
* Image alignment points (`alignment_point_x`, `alignment_point_y`)
* Pixel-to-meter ratios (`ratio_x`, `ratio_y`)
* Calculation resolution (`resolution` parameter mapping to `min_res`)

These might need to be modified directly in the code or refactored to accept dynamic inputs for different analysis
scenarios.

## Output

The `make run` command executes the calculation pipeline. While the specific output mechanism isn't fully detailed (
e.g., saving files vs. returning values), the code suggests the generation of:

* **Numerical Results:** Aggregated KPI values (PA, NS, FA, NP, MSA) for the ROI and CA, potentially stored in the
  `result_values` dictionary within the `handler`.
* **Image Reports:** Visual maps for different KPIs (CLC, NP, FA, MSA, NS, PA per nesting group, total PA, total NS)
  generated by `reporting.py` and `element.py`. The exact saving location/format is not specified in the provided code
  snippets but likely involves saving image files (e.g., PNG, WebP).

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

# Analysis of the code

### **Overview**

The code is part of a project that calculates Key Performance Indicators related to pollinator abundance and
suitability within specific geographical areas. It processes input data to generate results and reports for metrics like Pollinator Abundance ,Nesting Suitability and
others.

---

### **Key Components**

1. **`handler.py`**:
    - The `pollinator_abundance_calculation` function orchestrates the main workflow:
        - Loads input data (e.g., images, CLC values, bee species data).
        - Processes images for the **Region of Interest (ROI)** and **Context Area (CA)**.
        - Calculates KPIs using mathematical models.
        - Generates visual reports for the calculated KPIs.
    - Handles exceptions and logs the progress of the task.

2. **`element.py`**:
    - The `kpi_elements_generation` function generates specific KPI-related outputs:
        - Processes images for ROI and CA using masks.
        - Calculates KPI values for ROI and CA by applying mathematical models.
        - Creates visual reports (e.g., maps) for the KPIs using helper functions.
        - Handles both numerical and visual outputs for each KPI.

3. **`main.py`**:
    - Serves as the entry point for the project.
    - Calls the `pollinator_abundance_calculation` function to execute the analysis pipeline.

---

### **Workflow**

1. **Input Data**:
    - Images for ROI and CA (e.g., `.npy` files).
    - CLC data and bee species characteristics.
    - Parameters like alignment points, bounding boxes, and pixel-to-meter ratios.

2. **Image Processing**:
    - Merges ROI and CA images.
    - Applies masks to isolate specific areas.
    - Converts pixel-based data into real-world metrics.

3. **KPI Calculation**:
    - Uses mathematical models to compute KPIs like PA, NS, FA, etc.
    - Combines data from ROI and CA for more accurate results.

4. **Report Generation**:
    - Creates visual maps for each KPI.
    - Adds legends, scales, and titles to the reports.

5. **Output**:
    - Numerical results stored in a dictionary.
    - Reports

---

### **Strengths**

- Modular design with clear separation of concerns (e.g., `handler.py` for orchestration, `element.py` for KPI
  generation).
- Uses helper functions for image processing and reporting, improving code reusability.

---

### **Areas for improvement**

1. **Parallel execution**: (Done)
    - The kpi_elements_generation function could be executed in parallel for all KPIs to improve performance, especially
      if the calculations are independent. Using libraries like concurrent.futures can help achieve this.
    - At the begging of the function, we can create a thread pool executor and submit tasks for each KPI calculation.
      This will allow the function to run multiple KPI calculations concurrently, improving overall execution time. The
      only motivation to not run this function in parallel was the concurrent access to the **result_values** dictionary
      that is passed ad parameters but not used in the function. So we can run this function in parallel without
      problems.

2. **Error Handling**: (TBD)
    - While exceptions are caught, the error messages could be more descriptive to aid debugging. For example, instead
      of
      just catching a generic exception, specific exceptions (like `FileNotFoundError`, `ValueError`) could be caught
      and   
      logged with more context about the operation that failed.

3. **Functions parameters**: (TBD)
    - The `pollinator_abundance_calculation` function has many parameters, making it hard to read and maintain. It would
      be better to encapsulate related parameters into a configuration object or dictionary. This would make the
      function
      signature cleaner and easier to understand.
    - The `kpi_elements_generation` function also has many parameters. Similar to the previous point, consider using a
      configuration object or dictionary to group related parameters. Also some function parameters are not used in the
      function. These parameters should be removed from the function signature to improve readability and
      maintainability.

4. **Functions length**: (TBD)
    - The `pollinator_abundance_calculation` function is quite long and does multiple things. It would be beneficial to
      break it down into smaller, more focused functions. This would improve readability and make it easier to test
      individual components.
    -
5. **Cache**: (Done)
    - The `pollinator_abundance_calculation` and some sub-functions loads the images every time it is called. This can
      be optimized by
      using a cache to store the loaded images, so they are not reloaded every time the function is called. In this case
      can be
      done using the `functools.lru_cache` decorator or by implementing a custom caching mechanism.

---

### Execution time

To measure the execution time of the `pollinator_abundance_calculation` function, I noticed that the code already
includes a time calculation using the `time` library. However, I prefer to use the `timeit` library, as it is
specifically designed for precise benchmarking of code execution. This approach ensures more accurate and reliable
measurements, which can be useful for identifying performance bottlenecks and evaluating the impact of optimizations.

The analysis of the code execution time is done over 30 iterations on my pc, and the average time is calculated.

However, the execution time of the main functions `kpi_elements_generation` and `lambda_bee` is made with the time
library.

#### Optimizations

The following optimizations were made to improve the execution time of the `pollinator_abundance_calculation` function:

1. **Parallel execution**:  in the `pollinator_abundance_calculation` function, I used the `concurrent.futures` library
   to run the KPI calculations in parallel. This allows multiple calculations to be performed simultaneously, reducing
   the overall execution time.
2. **Using cache**: I used the `functools.lru_cache` in the function to load the images only once and cache the
   results. This avoids redundant image loading and processing, further improving performance.

#### Results of execution of the `pollinator_abundance_calculation`

| **Metric**           | **Before Optimizations** | **After Optimizations** |
|----------------------|--------------------------|-------------------------|
| Total Time (30 runs) | 155.80 seconds           | 125.70 seconds          |
| Average Time per Run | 5.19 seconds             | 4.19 seconds            |

With these optimizations, the average execution time of the `pollinator_abundance_calculation` function has been
reduced from **6.99 seconds** to **4.16 seconds**, resulting in a performance improvement.

#### Result of the execution of `kpi_elements_generation` and `lambda_bee`:

| **Metric**                | **Before Optimizations** | **After Optimizations** |
|---------------------------|--------------------------|-------------------------|
| `kpi_elements_generation` | 0.12 seconds             | 0.12 seconds            | (No optimization)
| `lambda_bee`              | 0.39 seconds             | 0.32 seconds            |

# Testing

Under the dir `tests/` you can find the tests for the project to test performance and output of the functions.

At the moment, the tests are not complete and only test the `pollinator_abundance_calculation` function. Specifically,
the tests check the execution time of the function and the output of the function.
There are two tests:

1. **`test_performance`**: This test checks that the execution time of the function is less than a certain threshold
   (5 seconds). This is done by using the `timeit` library to measure the execution time of the function over 30
   seconds.
2. **`test_output`**: This test checks that the output of the function is a dictionary and contains the expected keys.

## Run the tests

To run the test suite, you can use the following command:

```bash
make run_tests
```

# Wrapper

The project includes a server wrapper that allows you to run the project as a server. This is done using the fastapi
library

## Run the server

To run the server, you can use the following command:

```bash
make run_server
```

This will start the server on `http://localhost:8000` and you can access the API documentation at `http://localhost:8000/docs`.

## API Endpoints

The server exposes the following API endpoints:

- **`api/v1/calculate`**: This endpoint accepts a POST request with the following parameters:
- `plantation_id`: The ID of the plantation.
- `plantation_polygons_id`: The ID of the plantation polygons of interest.
- `resolution`: The requested resolution
- `ca_id`: The x coordinate of the alignment point.
- `roi_id`: The y coordinate of the alignment point.
- `override_bee`: Override the bee species flag.
- `how`: The method to use for the calculation.

This endpoint returns a JSON response with the results of the calculation. This is a simple example of how to use the
server wrapper to run the project as a server. This endpoint with a get request can be split into different endpoints
to avoids to pass all these parameters in the same request.

### N.B

Not function parameters are passed by request, for example 'ration_x', 'ratio_y', that are needed for the calculation of
the KPI are not passed but remain hardcoded in the code. This is a limitation of the current implementation example and
should be fixed in the future.

# Deployment

... Not implemented yet

# Future Work

1. The biggest improvements it to fix the results of the main function to avoid that the "result_values" dictionary is
   with all attribute's values hardcoded with None value.
2. Add more tests to cover all the functions in the project.
3. Add more API endpoints to cover all the functions in the project also with all the parameters and the data needed for
   the calculation.
4. Add all the parameters in the request to avoid hardcoded values in the code.
5. Add more documentation to the code and the project.

