import unittest

from scripts.evaluate_prep_calibration_drift import evaluate_record_drift, evaluate_report_drift


class PrepCalibrationDriftTests(unittest.TestCase):
    def test_evaluate_record_drift_zero_when_in_envelope(self):
        fixture_spec = {
            "default_profile": "balanced",
            "profiles": {
                "balanced": {"acceptance_score_min": 0.5, "acceptance_score_max": 1.0}
            },
        }
        record = {
            "prep_calibration_profile": "balanced",
            "prep_acceptance_score": 0.75,
        }
        drift = evaluate_record_drift(record, fixture_spec)
        self.assertIsNotNone(drift)
        self.assertEqual(drift["drift_ratio"], 0.0)

    def test_evaluate_record_drift_positive_when_below_envelope(self):
        fixture_spec = {
            "default_profile": "balanced",
            "profiles": {
                "balanced": {"acceptance_score_min": 0.8, "acceptance_score_max": 1.0}
            },
        }
        record = {
            "prep_calibration_profile": "balanced",
            "prep_acceptance_score": 0.4,
        }
        drift = evaluate_record_drift(record, fixture_spec)
        self.assertIsNotNone(drift)
        self.assertGreater(drift["drift_ratio"], 0.0)

    def test_evaluate_report_drift_filters_unknown_fixtures(self):
        report = {
            "records": [
                {
                    "fixture_id": "known",
                    "prep_calibration_profile": "balanced",
                    "prep_acceptance_score": 0.7,
                },
                {
                    "fixture_id": "unknown",
                    "prep_calibration_profile": "balanced",
                    "prep_acceptance_score": 0.7,
                },
            ]
        }
        evidence = {
            "fixtures": {
                "known": {
                    "default_profile": "balanced",
                    "profiles": {
                        "balanced": {
                            "acceptance_score_min": 0.5,
                            "acceptance_score_max": 1.0,
                        }
                    },
                }
            }
        }
        rows = evaluate_report_drift(report, evidence)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fixture_id"], "known")


if __name__ == "__main__":
    unittest.main()
