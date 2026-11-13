import unittest
import hashlib
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from scripts.meshing.performance import PerformanceError, _summarize, validate_report


class PerformanceReportTests(unittest.TestCase):
    def policy(self):
        fixture = "crates/runmat-geometry/io/tests/fixtures/box.brep"
        return {
            "schema_version": 1,
            "policy_revision": 1,
            "approved_at": "2026-08-23",
            "approval_basis": "measured",
            "workload_id": "fixture",
            "fixture": fixture,
            "fixture_sha256": hashlib.sha256(Path(fixture).read_bytes()).hexdigest(),
            "expected_canonical_digest": "abc",
            "minimum_elements": 1,
            "maximum_report_age_hours": 24,
            "required_stages": ["curve_mesh"],
            "profiles": {"smoke": {"minimum_samples": 3, "warmup_samples": 1}},
            "limits": {
                "wall_time_p50_ms": 20,
                "wall_time_p95_ms": 30,
                "wall_time_p99_ms": 30,
                "peak_rss_p99_bytes": 200,
                "allocation_count_p99": 200,
                "allocated_bytes_p99": 2000,
                "peak_live_bytes_p99": 200,
                "minimum_element_throughput_per_second_p50": 50,
                "maximum_wall_time_tail_ratio": 2,
            },
        }

    def sample(self, wall_time):
        return {
            "wall_time_ms": wall_time,
            "peak_rss_bytes": 100,
            "allocation_count": 100,
            "allocated_bytes": 1000,
            "peak_live_bytes": 100,
            "process_count": 2,
            "element_count": 2,
            "element_throughput_per_second": 2000 / wall_time,
            "canonical_digest": "abc",
            "stages": {"curve_mesh": {"elapsed_time_ms": 2, "peak_memory_bytes": 50}},
        }

    def report(self):
        samples = [self.sample(10), self.sample(11), self.sample(12)]
        return {
            "schema_version": 1,
            "policy_revision": 1,
            "workload_id": "fixture",
            "profile": "smoke",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "binary_sha256": "a" * 64,
            "fixture_sha256": self.policy()["fixture_sha256"],
            "samples": samples,
            "summary": _summarize(samples),
        }

    def test_accepts_recomputed_fresh_measurements(self):
        validate_report(self.report(), self.policy(), "smoke")

    def test_rejects_summary_not_derived_from_samples(self):
        report = self.report()
        report["summary"]["wall_time_ms"]["p99"] = 1
        with self.assertRaisesRegex(PerformanceError, "does not match raw samples"):
            validate_report(report, self.policy(), "smoke")

    def test_rejects_missing_process_measurements(self):
        report = self.report()
        report["samples"][0]["allocation_count"] = 0
        report["summary"] = _summarize(report["samples"])
        with self.assertRaisesRegex(PerformanceError, "allocation_count must be positive"):
            validate_report(report, self.policy(), "smoke")

    def test_rejects_malformed_sample_without_crashing(self):
        report = self.report()
        report["samples"][0] = "not-an-object"
        with self.assertRaisesRegex(PerformanceError, "sample must be an object"):
            validate_report(report, self.policy(), "smoke")

    def test_rejects_performance_outside_slo(self):
        policy = deepcopy(self.policy())
        policy["limits"]["wall_time_p99_ms"] = 5
        with self.assertRaisesRegex(PerformanceError, "wall-time p99"):
            validate_report(self.report(), policy, "smoke")

    def test_rejects_changed_canonical_result(self):
        report = self.report()
        report["samples"][1]["canonical_digest"] = "different"
        with self.assertRaisesRegex(PerformanceError, "canonical digest changed"):
            validate_report(report, self.policy(), "smoke")

    def test_rejects_unversioned_extra_stage(self):
        report = self.report()
        report["samples"][0]["stages"]["unexpected"] = {
            "elapsed_time_ms": 1,
            "peak_memory_bytes": 1,
        }
        with self.assertRaisesRegex(PerformanceError, "stage inventory differs"):
            validate_report(report, self.policy(), "smoke")


if __name__ == "__main__":
    unittest.main()
