import pathlib
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from bot import relic_analyzer


class _ImmediateNavQueue:
    def put(self, item):
        _crop, _kwargs, future = item
        future.set_result([("cpu-nav", 1.0)])


class NavigationOcrRoutingTests(unittest.TestCase):
    def test_directml_gpu_mode_routes_navigation_to_cpu_worker(self):
        crop = np.zeros((8, 8, 3), dtype=np.uint8)
        with (
            mock.patch.object(relic_analyzer, "_navigation_uses_cuda", return_value=False),
            mock.patch.object(relic_analyzer, "_nav_thread_started", True),
            mock.patch.object(relic_analyzer, "_nav_queue", _ImmediateNavQueue()),
            mock.patch.object(relic_analyzer, "_get_reader") as gpu_reader,
        ):
            result = relic_analyzer.nav_ocr(crop, allowlist="0123456789")

        self.assertEqual(result, [("cpu-nav", 1.0)])
        gpu_reader.assert_not_called()

    def test_cuda_gpu_mode_keeps_original_gpu_navigation_path(self):
        crop = np.zeros((8, 8, 3), dtype=np.uint8)
        reader = mock.Mock()
        reader.readtext.return_value = [("cuda-nav", 1.0)]
        with (
            mock.patch.object(relic_analyzer, "_navigation_uses_cuda", return_value=True),
            mock.patch.object(relic_analyzer, "_get_reader", return_value=reader),
        ):
            result = relic_analyzer.nav_ocr(crop)

        self.assertEqual(result, [("cuda-nav", 1.0)])
        reader.readtext.assert_called_once_with(crop)


if __name__ == "__main__":
    unittest.main()
