import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.prep_calibration.promote_prep_calibration_evidence import (
    main as promote_main,
)


class PromotePrepCalibrationEvidenceTests(unittest.TestCase):
    def setUp(self):
        for key in [
            "RUNMAT_PREP_CALIBRATION_EVIDENCE_CANDIDATE",
            "RUNMAT_PREP_CALIBRATION_EVIDENCE_APPROVED",
            "RUNMAT_PREP_CALIBRATION_RECOMMENDATIONS_INPUT",
            "RUNMAT_PREP_CALIBRATION_PROMOTION_MAX_RECOMMENDATIONS",
        ]:
            os.environ.pop(key, None)

    def test_promotion_succeeds_when_recommendations_within_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate = tmp_path / "candidate.json"
            approved = tmp_path / "approved.json"
            recs = tmp_path / "recs.json"
            candidate.write_text(
                json.dumps(
                    {
                        "schema_version": "prep-calibration-evidence/v1",
                        "state": "candidate",
                        "generated_at": "2026-03-08T00:00:00Z",
                        "max_age_days": 365,
                        "fixtures": {
                            "f": {
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
                )
            )
            recs.write_text(
                json.dumps(
                    {
                        "schema_version": "prep-calibration-recommendations/v1",
                        "generated_at": "2026-03-08T00:00:00Z",
                        "recommendation_count": 0,
                        "recommendations": [],
                    }
                )
            )
            os.environ["RUNMAT_PREP_CALIBRATION_EVIDENCE_CANDIDATE"] = str(candidate)
            os.environ["RUNMAT_PREP_CALIBRATION_EVIDENCE_APPROVED"] = str(approved)
            os.environ["RUNMAT_PREP_CALIBRATION_RECOMMENDATIONS_INPUT"] = str(recs)
            os.environ["RUNMAT_PREP_CALIBRATION_PROMOTION_MAX_RECOMMENDATIONS"] = "0"
            code = promote_main()
            self.assertEqual(code, 0)
            self.assertTrue(approved.exists())

    def test_promotion_fails_when_recommendations_exceed_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate = tmp_path / "candidate.json"
            approved = tmp_path / "approved.json"
            recs = tmp_path / "recs.json"
            candidate.write_text(
                json.dumps(
                    {
                        "schema_version": "prep-calibration-evidence/v1",
                        "state": "candidate",
                        "generated_at": "2026-03-08T00:00:00Z",
                        "max_age_days": 365,
                        "fixtures": {
                            "f": {
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
                )
            )
            recs.write_text(
                json.dumps(
                    {
                        "schema_version": "prep-calibration-recommendations/v1",
                        "generated_at": "2026-03-08T00:00:00Z",
                        "recommendation_count": 3,
                        "recommendations": [
                            {
                                "fixture_id": "f",
                                "current_profile": "balanced",
                                "suggested_profile": "conservative",
                                "suggested_profile_shift": 1,
                            }
                        ],
                    }
                )
            )
            os.environ["RUNMAT_PREP_CALIBRATION_EVIDENCE_CANDIDATE"] = str(candidate)
            os.environ["RUNMAT_PREP_CALIBRATION_EVIDENCE_APPROVED"] = str(approved)
            os.environ["RUNMAT_PREP_CALIBRATION_RECOMMENDATIONS_INPUT"] = str(recs)
            os.environ["RUNMAT_PREP_CALIBRATION_PROMOTION_MAX_RECOMMENDATIONS"] = "0"
            code = promote_main()
            self.assertEqual(code, 1)
            self.assertFalse(approved.exists())


if __name__ == "__main__":
    unittest.main()
