import timeit

from pollinator_abundance.handler import pollinator_abundance_calculation


def simple_run():
    """Run the simple tun calculation."""
    print("Starting pollinator abundance calculation...")
    result = pollinator_abundance_calculation()
    print("Pollinator abundance calculation completed.")
    print("Result:", result.keys())


def run_with_times():
    result = timeit.timeit(simple_run, number=30)
    print(result)


if __name__ == "__main__":
    run_with_times()
