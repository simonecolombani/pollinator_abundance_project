from pollinator_abundance.handler import pollinator_abundance_calculation


if __name__ == "__main__":
    # Example usage
    print("Starting pollinator abundance calculation...")
    result = pollinator_abundance_calculation()
    print("Pollinator abundance calculation completed.")
    print("Result:", result.keys())
