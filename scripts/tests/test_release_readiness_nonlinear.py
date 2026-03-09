import unittest
import os

from scripts.release_readiness_nonlinear import evaluate_release_readiness


def report(passed=True, publishable=True, gpu_ms=100.0, prep_acceptance_passed=True):
    fixtures = [
        "nonlinear_assembly_gpu_provider",
        "nonlinear_assembly_stress_gpu_provider",
        "nonlinear_softening_proxy_gpu_provider",
        "nonlinear_load_path_mix_gpu_provider",
    ]
    return {
        "passed": passed,
        "records": [
            {
                "fixture_id": fixture,
                "publishable": publishable,
                "gpu_run_ms": gpu_ms,
                "prep_acceptance_passed": prep_acceptance_passed,
            }
            for fixture in fixtures
        ],
    }


class ReleaseReadinessTests(unittest.TestCase):
    def setUp(self):
        for key in [
            "RUNMAT_RELEASE_READINESS_PREP_WARN_ARTIFACT_COUNT",
            "RUNMAT_RELEASE_READINESS_PREP_FAIL_ARTIFACT_COUNT",
            "RUNMAT_RELEASE_READINESS_PREP_WARN_P95_AGE_SECONDS",
            "RUNMAT_RELEASE_READINESS_PREP_FAIL_P95_AGE_SECONDS",
            "RUNMAT_RELEASE_READINESS_PREP_MAX_REJECT_RATE",
            "RUNMAT_PREP_CREATED_COUNT",
            "RUNMAT_PREP_STALE_REJECT_COUNT",
            "RUNMAT_PREP_MISMATCH_REJECT_COUNT",
            "RUNMAT_RELEASE_READINESS_PREP_ACCEPTANCE_MIN_RATE",
            "RUNMAT_RELEASE_READINESS_PREP_ACCEPTANCE_REQUIRE",
        ]:
            os.environ.pop(key, None)

    def test_pass_when_all_signals_healthy(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        rolling = [report(passed=True, publishable=True, gpu_ms=95.0)]
        result = evaluate_release_readiness(latest, rolling, protected=False)
        self.assertEqual(result["verdict"], "warn")
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("ARTIFACT_REPLAY_UNVERIFIED", codes)
        self.assertIn("ARTIFACT_COMPAT_UNVERIFIED", codes)

    def test_fail_on_protected_branch_with_conformance_failure(self):
        latest = report(passed=False, publishable=False, gpu_ms=130.0)
        rolling = [report(passed=True, publishable=True, gpu_ms=90.0)]
        result = evaluate_release_readiness(latest, rolling, protected=True)
        self.assertEqual(result["verdict"], "fail")
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("CONFORMANCE_FAILED", codes)
        self.assertIn("NONLINEAR_FIXTURE_UNPUBLISHABLE", codes)

    def test_warn_on_non_protected_slowdown(self):
        latest = report(passed=True, publishable=True, gpu_ms=200.0)
        rolling = [report(passed=True, publishable=True, gpu_ms=100.0)]
        result = evaluate_release_readiness(latest, rolling, protected=False)
        self.assertIn(result["verdict"], {"warn", "fail"})
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("NONLINEAR_TREND_SLOWDOWN", codes)

    def test_prep_health_count_warns_non_protected(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        rolling = [report(passed=True, publishable=True, gpu_ms=95.0)]
        os.environ["RUNMAT_RELEASE_READINESS_PREP_WARN_ARTIFACT_COUNT"] = "2"
        os.environ["RUNMAT_RELEASE_READINESS_PREP_FAIL_ARTIFACT_COUNT"] = "5"
        result = evaluate_release_readiness(
            latest,
            rolling,
            protected=False,
            prep_health={"artifact_count": 3, "p95_age_seconds": 10.0},
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PREP_SLO_COUNT_EXCEEDED", codes)

    def test_prep_health_age_fails_protected(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        rolling = [report(passed=True, publishable=True, gpu_ms=95.0)]
        os.environ["RUNMAT_RELEASE_READINESS_PREP_WARN_P95_AGE_SECONDS"] = "10"
        os.environ["RUNMAT_RELEASE_READINESS_PREP_FAIL_P95_AGE_SECONDS"] = "20"
        result = evaluate_release_readiness(
            latest,
            rolling,
            protected=True,
            prep_health={"artifact_count": 1, "p95_age_seconds": 25.0},
        )
        self.assertEqual(result["verdict"], "fail")
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PREP_SLO_AGE_EXCEEDED", codes)

    def test_prep_reject_rate_reason_is_emitted(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        rolling = [report(passed=True, publishable=True, gpu_ms=95.0)]
        os.environ["RUNMAT_RELEASE_READINESS_PREP_MAX_REJECT_RATE"] = "0.1"
        os.environ["RUNMAT_PREP_CREATED_COUNT"] = "10"
        os.environ["RUNMAT_PREP_STALE_REJECT_COUNT"] = "2"
        os.environ["RUNMAT_PREP_MISMATCH_REJECT_COUNT"] = "1"
        result = evaluate_release_readiness(
            latest,
            rolling,
            protected=False,
            prep_health={"artifact_count": 1, "p95_age_seconds": 5.0},
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PREP_REJECT_RATE_HIGH", codes)

    def test_prep_acceptance_rate_low_reason_is_emitted(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0, prep_acceptance_passed=False)
        rolling = [report(passed=True, publishable=True, gpu_ms=95.0)]
        os.environ["RUNMAT_RELEASE_READINESS_PREP_ACCEPTANCE_MIN_RATE"] = "0.8"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PREP_ACCEPTANCE_RATE_LOW", codes)

    def test_prep_acceptance_missing_warns_when_required(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        for rec in latest["records"]:
            rec.pop("prep_acceptance_passed", None)
        rolling = [report(passed=True, publishable=True, gpu_ms=95.0)]
        os.environ["RUNMAT_RELEASE_READINESS_PREP_ACCEPTANCE_REQUIRE"] = "true"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PREP_ACCEPTANCE_MISSING", codes)


if __name__ == "__main__":
    unittest.main()
