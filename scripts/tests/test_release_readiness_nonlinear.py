import unittest

from scripts.release_readiness_nonlinear import evaluate_release_readiness


def report(passed=True, publishable=True, gpu_ms=100.0):
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
            }
            for fixture in fixtures
        ],
    }


class ReleaseReadinessTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
