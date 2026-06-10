import json
import unittest
from pathlib import Path

from scripts.fea.governance.validate_external_reference_benchmark import (
    REQUIRED_METRICS_BY_FIXTURE,
)


class ExternalReferenceBaselineTests(unittest.TestCase):
    def test_external_reference_baseline_covers_validator_required_metrics(self):
        baseline_path = Path("scripts/fea/reference_data/m6_external_reference_baseline.json")
        payload = json.loads(baseline_path.read_text())
        metrics = payload.get("metrics")
        self.assertIsInstance(metrics, list)

        required_pairs = {
            (fixture_id, metric_name)
            for fixture_id, metric_names in REQUIRED_METRICS_BY_FIXTURE.items()
            for metric_name in metric_names
        }

        observed_pairs = set()
        for metric in metrics:
            fixture_id = metric.get("fixture_id")
            if not isinstance(fixture_id, str):
                continue
            source = metric.get("source", "field")
            metric_name = (
                metric.get("assertion_name")
                if source == "threshold_assertion"
                else metric.get("name")
            )
            if isinstance(metric_name, str):
                observed_pairs.add((fixture_id, metric_name))

        missing_pairs = required_pairs - observed_pairs
        self.assertFalse(
            missing_pairs,
            "missing external-reference baseline metrics required by validator: "
            f"{sorted(missing_pairs)}",
        )


if __name__ == "__main__":
    unittest.main()
