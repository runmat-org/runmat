"""Observation coherence tests; these do not execute or qualify native images."""

import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "record-native-paired-product.py"
SPEC = importlib.util.spec_from_file_location("native_pair_observer", SCRIPT)
OBSERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OBSERVER)


def observation():
    role = {
        "canonical_plan_identity": "plan",
        "selected_build": {"toolchain": "compiler", "sysroot": "sdk"},
        "inputs": {"object": "object", "runtime_archive": "runtime"},
    }
    product = {
        "manifest": {
            "binding": {"program_revision": "source", "selected_product": "product"},
            "application": "application-object",
            "tests": "test-object",
        },
        "application": role,
        "tests": copy.deepcopy(role),
    }
    return {
        "schema_version": 2,
        "scope": "external_physical_observation_of_completed_native_pair",
        "build_subject": OBSERVER.build_subject(product),
        "images": [
            {"role": name, "digest": name, "bytes": 42, "loader_dependencies": []}
            for name in ("application", "tests")
        ],
    }


class ObservationTests(unittest.TestCase):
    def test_comparison_separates_build_identity_image_bytes_and_dependencies(self):
        current = observation()
        result = OBSERVER.compare_observations(copy.deepcopy(current), current, b"prior")
        self.assertTrue(result["physical_products_reproduced"])
        for component in ("paired_manifest", "roles"):
            changed = copy.deepcopy(current)
            changed["build_subject"][component] = {"different": True}
            with self.subTest(component=component), self.assertRaisesRegex(
                ValueError, "different semantic or link-build inputs"
            ):
                OBSERVER.compare_observations(changed, current, b"prior")
        for field, replacement, retained in (
            ("digest", "different-image", "loader_dependencies_reproduced"),
            ("bytes", 43, "loader_dependencies_reproduced"),
            ("loader_dependencies", [{"install_name": "different"}], "image_bytes_reproduced"),
        ):
            changed = copy.deepcopy(current)
            changed["images"][0][field] = replacement
            result = OBSERVER.compare_observations(changed, current, b"prior")
            with self.subTest(field=field):
                self.assertFalse(result["physical_products_reproduced"])
                self.assertTrue(result[retained])
        changed = copy.deepcopy(current)
        changed["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "compatible physical observation"):
            OBSERVER.compare_observations(changed, current, b"prior")

    def test_rust_validation_digest_must_match_exact_observed_receipt_bytes(self):
        # Use existing physical paths only to exercise subprocess routing. The
        # subprocess is mocked; the real Rust mode is a separate product gate.
        with patch.object(OBSERVER.subprocess, "run") as run:
            run.return_value.stdout = (OBSERVER.digest(b"receipt") + "\n").encode()
            OBSERVER.validate_receipt(SCRIPT, b"receipt", SCRIPT)
            self.assertEqual(run.call_args.args[0][1], "--validate-receipt")
            run.return_value.stdout = (OBSERVER.digest(b"other") + "\n").encode()
            with self.assertRaisesRegex(ValueError, "exact observed bytes"):
                OBSERVER.validate_receipt(SCRIPT, b"receipt", SCRIPT)


if __name__ == "__main__":
    unittest.main()
