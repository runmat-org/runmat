"""Host qualification must not certify deterministic deployment by relabeling."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "qualify-native-paired-host.py"
SPEC = importlib.util.spec_from_file_location("native_host_qualification", SCRIPT)
QUALIFICATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFICATION)


class HostProfileTests(unittest.TestCase):
    def test_hosted_profile_is_explicit_and_cannot_be_relabelled(self):
        self.assertEqual(QUALIFICATION.hosted_profile({}), "hosted_native")
        self.assertEqual(
            QUALIFICATION.hosted_profile({"profile": "hosted_native"}), "hosted_native"
        )
        for profile in ("deterministic_deployed", "unknown", None):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                QUALIFICATION.hosted_profile({"profile": profile})

    def test_rejected_profile_creates_no_qualification_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "operator.json"
            config.write_text('{"schema_version":1,"profile":"deterministic_deployed"}')
            output = root / "qualification"
            args = SimpleNamespace(operator_config=config, out=output)
            with self.assertRaisesRegex(ValueError, "hosted SDK/loader"):
                QUALIFICATION.Qualification(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
