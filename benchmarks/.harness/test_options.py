import sys
from pathlib import Path
import unittest

import numpy as np

from run_options import analytic_price, validate_samples

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'monte-carlo-analysis'))
from python_numpy_options import price_paths


class OptionsChecks(unittest.TestCase):
    def test_black_scholes_reference(self):
        self.assertAlmostEqual(analytic_price(), 9.413403383853016, places=11)

    def test_loops_and_vectorization_agree(self):
        z = np.array([-3., -1., 0., 1., 3.])
        np.testing.assert_allclose(price_paths(z, 'loop'), price_paths(z, 'vectorized'), rtol=1e-13)

    def test_accepts_valid_samples(self):
        self.assertEqual(len(validate_samples('SAMPLE rep=1 ms=2 price=9 stderr=.1', 1, 9, .1)), 1)

    def test_rejects_wrong_price_and_uncertainty(self):
        for output in ('SAMPLE rep=1 ms=2 price=8 stderr=.1',
                       'SAMPLE rep=1 ms=2 price=9 stderr=.2'):
            with self.assertRaises(ValueError):
                validate_samples(output, 1, 9, .1)

    def test_rejects_nonfinite_and_incomplete_results(self):
        for output in ('', 'SAMPLE rep=1 ms=nan price=9 stderr=.1',
                       'SAMPLE rep=2 ms=2 price=9 stderr=.1',
                       'SAMPLE rep=1 ms=2 price=9 stderr=.1\nSAMPLE rep=1 ms=2 price=9 stderr=.1'):
            with self.assertRaises(ValueError):
                validate_samples(output, 1, 9, .1)

    def test_float32_tolerance_still_rejects_observed_gpu_failures(self):
        with self.assertRaises(ValueError):
            validate_samples('SAMPLE rep=1 ms=1 price=9.393637657 stderr=2970.529296875',
                             1, 9.3936374, .0445682054, 2e-5)
        with self.assertRaises(ValueError):
            validate_samples('SAMPLE rep=1 ms=1 price=9.4113531 stderr=.0045304494',
                             1, 9.4113531, .0044611457, 2e-5)


if __name__ == '__main__':
    unittest.main()
