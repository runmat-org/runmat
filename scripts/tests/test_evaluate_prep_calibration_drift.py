import unittest
from datetime import datetime, timezone

from scripts.evaluate_prep_calibration_drift import (
    build_recommendation_artifact,
    evaluate_record_drift,
    evaluate_report_drift,
    recommend_profile_shifts,
    validate_evidence,
    validate_recommendation_artifact,
)


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
        if drift is None:
            self.fail("drift result should not be None")
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
        if drift is None:
            self.fail("drift result should not be None")
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

    def test_validate_evidence_detects_staleness(self):
        evidence = {
            "schema_version": "prep-calibration-evidence/v1",
            "generated_at": "2026-01-01T00:00:00Z",
            "max_age_days": 5,
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
            },
        }
        now = datetime(2026, 1, 20, tzinfo=timezone.utc)
        status = validate_evidence(evidence, now=now)
        self.assertTrue(status["valid"])
        self.assertTrue(status["stale"])

    def test_recommend_profile_shift_is_deterministic(self):
        evidence = {
            "fixtures": {
                "known": {
                    "default_profile": "balanced",
                    "profiles": {
                        "fast": {"acceptance_score_min": 0.4, "acceptance_score_max": 1.0},
                        "balanced": {
                            "acceptance_score_min": 0.8,
                            "acceptance_score_max": 1.0,
                        },
                        "conservative": {
                            "acceptance_score_min": 0.9,
                            "acceptance_score_max": 1.0,
                        },
                    },
                }
            }
        }
        latest = {
            "records": [
                {
                    "fixture_id": "known",
                    "prep_calibration_profile": "balanced",
                    "prep_acceptance_score": 0.6,
                }
            ]
        }
        rolling = [
            {
                "records": [
                    {
                        "fixture_id": "known",
                        "prep_calibration_profile": "balanced",
                        "prep_acceptance_score": 0.65,
                    }
                ]
            },
            latest,
        ]
        first = recommend_profile_shifts(latest, rolling, evidence, drift_trigger=0.1)
        second = recommend_profile_shifts(latest, rolling, evidence, drift_trigger=0.1)
        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertEqual(first[0]["suggested_profile"], "conservative")

    def test_build_and_validate_recommendation_artifact(self):
        evidence = {
            "fixtures": {
                "known": {
                    "default_profile": "balanced",
                    "profiles": {
                        "balanced": {
                            "acceptance_score_min": 0.8,
                            "acceptance_score_max": 1.0,
                        },
                        "conservative": {
                            "acceptance_score_min": 0.9,
                            "acceptance_score_max": 1.0,
                        },
                    },
                }
            }
        }
        latest = {
            "records": [
                {
                    "fixture_id": "known",
                    "prep_calibration_profile": "balanced",
                    "prep_acceptance_score": 0.6,
                }
            ]
        }
        artifact = build_recommendation_artifact(latest, [latest], evidence, drift_trigger=0.1)
        status = validate_recommendation_artifact(artifact)
        self.assertTrue(status["valid"])
        self.assertIn("recommendations", artifact)


if __name__ == "__main__":
    unittest.main()
