"""The suite's residual report must never be an executable calibration input."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("run_suite", Path(__file__).with_name("run_suite.py"))
run_suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_suite)


class CalibrationCommandTest(unittest.TestCase):
    def test_refresh_reuses_runtime_owner_and_preserves_dry_run(self):
        for dry_run in (False, True):
            with patch.object(Path, "exists", return_value=True), patch.object(
                run_suite.subprocess, "run", return_value=Mock(returncode=0)
            ) as invoke:
                run_suite._run_auto_calibrate(Path("residual-results.json"), dry_run, None)
            command = invoke.call_args.args[0]
            self.assertIn("--refresh", command)
            self.assertNotIn("residual-results.json", command)
            self.assertEqual("--dry-run" in command, dry_run)
