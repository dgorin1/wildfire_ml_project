import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import importlib.util

# Add the pipeline directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pipeline')))

class Test01AssembleTrainingData(unittest.TestCase):

    @patch('script.get_weather_crs')
    @patch('script.process_year')
    def test_main_runs_without_errors(self, mock_process_year, mock_get_weather_crs):
        """
        Test that the main function of 01_assemble_training_data.py runs without raising exceptions.
        """
        # Mock the return values of the patched functions
        mock_get_weather_crs.return_value = "EPSG:4326"
        mock_process_year.return_value = None

        # Import the script to be tested
        spec = importlib.util.spec_from_file_location("script", "pipeline/01_assemble_training_data.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)

        # Run the main function
        try:
            script.main()
        except Exception as e:
            self.fail(f"script.main() raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
