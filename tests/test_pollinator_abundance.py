import unittest
import timeit
from pollinator_abundance.handler import pollinator_abundance_calculation


class TestPollinatorAbundance(unittest.TestCase):
    """Class to the pollinator abundance function"""

    def setUp(self):
        self.number_of_execution = 30
        self.max_allowed_time = 5.0  # seconds

    def test_correctness(self):
        result = pollinator_abundance_calculation()
        self.assertIsInstance(result, dict)
        self.assertIn("pa_image_total_normalized", result)
        self.assertIn("ns_image_total_normalized", result)

    def test_performance(self):
        execution_time = timeit.timeit(
            lambda: pollinator_abundance_calculation(
            ),
            number=self.number_of_execution
        )
        avg_time = execution_time / self.number_of_execution
        print(f"Average execution time: {avg_time:.2f} seconds")
        self.assertLess(avg_time, self.max_allowed_time,
                        f"Average execution time {avg_time:.2f} seconds exceeds the maximum allowed time of {self.max_allowed_time} seconds.")

        if __name__ == "__main__":
            unittest.main()
