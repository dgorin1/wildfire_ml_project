import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import importlib.util

# Add the pipeline directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pipeline')))

class Test00DownloadWeatherData(unittest.TestCase):

    @patch('script.xr.open_zarr')
    @patch('script.process_fire_worker')
    @patch('script.gpd.read_parquet')
    def test_main_runs_without_errors(self, mock_read_parquet, mock_process_fire_worker, mock_open_zarr):
        """
        Test that the main function of 00_download_weather_data.py runs without raising exceptions.
        """
        # Mock the return values of the patched functions
        mock_open_zarr.return_value.__enter__.return_value.rio.crs = "EPSG:4326"
        mock_process_fire_worker.return_value = "SUCCESS"
        mock_read_parquet.return_value = MagicMock()

        # Import the script to be tested
        spec = importlib.util.spec_from_file_location("script", "pipeline/00_download_weather_data.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)

        # Run the main function
        try:
            script.main()
        except Exception as e:
            self.fail(f"script.main() raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
