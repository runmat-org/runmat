import unittest
import os

from scripts.release_readiness_nonlinear import (
    evaluate_release_readiness,
    markdown_summary,
)


def report(
    passed=True,
    publishable=True,
    gpu_ms=100.0,
    prep_acceptance_passed=True,
    prep_acceptance_score=0.9,
    prep_calibration_profile="balanced",
    thermo_coupling_enabled=None,
    thermo_transient_severity=None,
    thermo_nonlinear_severity=None,
    thermo_constitutive_material_spread_ratio=None,
    thermo_assignment_heterogeneity_index=None,
):
    fixtures = [
        "nonlinear_assembly_gpu_provider",
        "nonlinear_assembly_stress_gpu_provider",
        "nonlinear_softening_proxy_gpu_provider",
        "nonlinear_load_path_mix_gpu_provider",
    ]
    records = [
        {
            "fixture_id": fixture,
            "publishable": publishable,
            "gpu_run_ms": gpu_ms,
            "prep_acceptance_passed": prep_acceptance_passed,
            "prep_acceptance_score": prep_acceptance_score,
            "prep_calibration_profile": prep_calibration_profile,
        }
        for fixture in fixtures
    ]
    if thermo_coupling_enabled is not None:
        for rec in records:
            rec["thermo_coupling_enabled"] = thermo_coupling_enabled
    if thermo_transient_severity is not None:
        for rec in records:
            rec["thermo_transient_severity"] = thermo_transient_severity
    if thermo_nonlinear_severity is not None:
        for rec in records:
            rec["thermo_nonlinear_severity"] = thermo_nonlinear_severity
    if thermo_constitutive_material_spread_ratio is not None:
        for rec in records:
            rec["thermo_constitutive_material_spread_ratio"] = (
                thermo_constitutive_material_spread_ratio
            )
    if thermo_assignment_heterogeneity_index is not None:
        for rec in records:
            rec["thermo_assignment_heterogeneity_index"] = (
                thermo_assignment_heterogeneity_index
            )

    return {
        "passed": passed,
        "records": records,
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
            "RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_MAX_DRIFT",
            "RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_REQUIRE_EVIDENCE",
            "RUNMAT_RELEASE_READINESS_PREP_RETRAIN_TRIGGER_DRIFT",
            "RUNMAT_RELEASE_READINESS_PREP_MAX_RECOMMENDATION_RATIO",
            "RUNMAT_RELEASE_READINESS_PREP_CANDIDATE_MAX_AGE_DAYS",
            "RUNMAT_RELEASE_READINESS_PREP_REQUIRE_RECOMMENDATION_ARTIFACT",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_TRANSIENT_SEVERITY",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_NONLINEAR_SEVERITY",
            "RUNMAT_RELEASE_READINESS_THERMO_MIN_ENABLED_RATE",
            "RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_METRICS",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_RATIO",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_INDEX",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_BREACH_RATE",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_BREACH_RATE",
            "GITHUB_REF_NAME",
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

    def test_prep_calibration_drift_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            prep_acceptance_passed=True,
            prep_acceptance_score=0.1,
            prep_calibration_profile="balanced",
        )
        rolling = [report(passed=True, publishable=True, gpu_ms=95.0)]
        evidence = {
            "fixtures": {
                "nonlinear_assembly_gpu_provider": {
                    "default_profile": "balanced",
                    "profiles": {
                        "balanced": {
                            "acceptance_score_min": 0.8,
                            "acceptance_score_max": 1.0,
                        }
                    },
                },
                "nonlinear_assembly_stress_gpu_provider": {
                    "default_profile": "balanced",
                    "profiles": {
                        "balanced": {
                            "acceptance_score_min": 0.8,
                            "acceptance_score_max": 1.0,
                        }
                    },
                },
                "nonlinear_softening_proxy_gpu_provider": {
                    "default_profile": "balanced",
                    "profiles": {
                        "balanced": {
                            "acceptance_score_min": 0.8,
                            "acceptance_score_max": 1.0,
                        }
                    },
                },
                "nonlinear_load_path_mix_gpu_provider": {
                    "default_profile": "balanced",
                    "profiles": {
                        "balanced": {
                            "acceptance_score_min": 0.8,
                            "acceptance_score_max": 1.0,
                        }
                    },
                },
            }
        }
        os.environ["RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_MAX_DRIFT"] = "0.2"
        result = evaluate_release_readiness(
            latest,
            rolling,
            protected=False,
            calibration_evidence=evidence,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PREP_CALIBRATION_DRIFT_HIGH", codes)

    def test_prep_calibration_evidence_stale_reason_is_emitted(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        rolling = [report(passed=True, publishable=True, gpu_ms=95.0)]
        evidence = {
            "schema_version": "prep-calibration-evidence/v1",
            "generated_at": "2025-01-01T00:00:00Z",
            "max_age_days": 1,
            "fixtures": {
                "nonlinear_assembly_gpu_provider": {
                    "default_profile": "balanced",
                    "profiles": {
                        "balanced": {
                            "acceptance_score_min": 0.3,
                            "acceptance_score_max": 1.0,
                        }
                    },
                }
            },
        }
        result = evaluate_release_readiness(
            latest,
            rolling,
            protected=False,
            calibration_evidence=evidence,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PREP_CALIBRATION_EVIDENCE_STALE", codes)

    def test_prep_calibration_retrain_recommended_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            prep_acceptance_score=0.55,
            prep_calibration_profile="balanced",
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                prep_acceptance_score=0.75,
                prep_calibration_profile="balanced",
            ),
            latest,
        ]
        evidence = {
            "schema_version": "prep-calibration-evidence/v1",
            "generated_at": "2026-03-08T00:00:00Z",
            "max_age_days": 365,
            "fixtures": {
                fixture: {
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
                for fixture in [
                    "nonlinear_assembly_gpu_provider",
                    "nonlinear_assembly_stress_gpu_provider",
                    "nonlinear_softening_proxy_gpu_provider",
                    "nonlinear_load_path_mix_gpu_provider",
                ]
            },
        }
        os.environ["RUNMAT_RELEASE_READINESS_PREP_MAX_RECOMMENDATION_RATIO"] = "0.1"
        os.environ["RUNMAT_RELEASE_READINESS_PREP_RETRAIN_TRIGGER_DRIFT"] = "0.05"
        result = evaluate_release_readiness(
            latest,
            rolling,
            protected=False,
            calibration_evidence=evidence,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PREP_CALIBRATION_RETRAIN_RECOMMENDED", codes)

    def test_prep_calibration_candidate_stale_reason_is_emitted(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        rolling = [report(passed=True, publishable=True, gpu_ms=95.0)]
        evidence = {
            "schema_version": "prep-calibration-evidence/v1",
            "state": "candidate",
            "generated_at": "2025-01-01T00:00:00Z",
            "max_age_days": 999,
            "fixtures": {
                "nonlinear_assembly_gpu_provider": {
                    "default_profile": "balanced",
                    "profiles": {
                        "balanced": {
                            "acceptance_score_min": 0.3,
                            "acceptance_score_max": 1.0,
                        }
                    },
                }
            },
        }
        os.environ["RUNMAT_RELEASE_READINESS_PREP_CANDIDATE_MAX_AGE_DAYS"] = "1"
        result = evaluate_release_readiness(
            latest,
            rolling,
            protected=False,
            calibration_evidence=evidence,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PREP_CALIBRATION_CANDIDATE_STALE", codes)

    def test_recommendation_artifact_missing_reason_is_emitted_when_required(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        rolling = [report(passed=True, publishable=True, gpu_ms=95.0)]
        os.environ["RUNMAT_RELEASE_READINESS_PREP_REQUIRE_RECOMMENDATION_ARTIFACT"] = "true"
        result = evaluate_release_readiness(
            latest,
            rolling,
            protected=False,
            calibration_evidence={
                "schema_version": "prep-calibration-evidence/v1",
                "generated_at": "2026-03-08T00:00:00Z",
                "max_age_days": 365,
                "fixtures": {},
            },
            recommendation_artifact=None,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PREP_CALIBRATION_RECOMMENDATION_ARTIFACT_MISSING", codes)

    def test_release_branch_profile_requires_recommendation_artifact_by_default(self):
        os.environ["GITHUB_REF_NAME"] = "release/1.2.3"
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
            calibration_evidence={
                "schema_version": "prep-calibration-evidence/v1",
                "state": "approved",
                "generated_at": "2026-03-08T00:00:00Z",
                "max_age_days": 365,
                "fixtures": {
                    "nonlinear_assembly_gpu_provider": {
                        "default_profile": "balanced",
                        "profiles": {
                            "balanced": {
                                "acceptance_score_min": 0.3,
                                "acceptance_score_max": 1.0,
                            }
                        },
                    }
                },
            },
            recommendation_artifact=None,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PREP_CALIBRATION_RECOMMENDATION_ARTIFACT_MISSING", codes)
        self.assertEqual(result.get("governance_profile"), "release")

    def test_feature_branch_profile_does_not_require_recommendation_artifact_by_default(self):
        os.environ["GITHUB_REF_NAME"] = "feature/sandbox"
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
            calibration_evidence={
                "schema_version": "prep-calibration-evidence/v1",
                "state": "approved",
                "generated_at": "2026-03-08T00:00:00Z",
                "max_age_days": 365,
                "fixtures": {
                    "nonlinear_assembly_gpu_provider": {
                        "default_profile": "balanced",
                        "profiles": {
                            "balanced": {
                                "acceptance_score_min": 0.3,
                                "acceptance_score_max": 1.0,
                            }
                        },
                    }
                },
            },
            recommendation_artifact=None,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertNotIn("PREP_CALIBRATION_RECOMMENDATION_ARTIFACT_MISSING", codes)
        self.assertEqual(result.get("governance_profile"), "feature")

    def test_thermo_transient_severity_high_reason_is_emitted(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        latest["records"].append(
            {
                "fixture_id": "thermo_mech_kickoff_gpu_provider",
                "publishable": True,
                "gpu_run_ms": 90.0,
                "thermo_coupling_enabled": True,
                "thermo_transient_severity": 0.45,
            }
        )
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_MAX_TRANSIENT_SEVERITY"] = "0.2"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_TRANSIENT_SEVERITY_HIGH", codes)

    def test_thermo_nonlinear_severity_high_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_nonlinear_severity=0.55,
        )
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_MAX_NONLINEAR_SEVERITY"] = "0.2"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_NONLINEAR_SEVERITY_HIGH", codes)

    def test_thermo_coupling_enabled_rate_low_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=False,
            thermo_nonlinear_severity=0.05,
        )
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_MIN_ENABLED_RATE"] = "0.5"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_COUPLING_ENABLED_RATE_LOW", codes)

    def test_thermo_metrics_missing_warn_when_required(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_METRICS"] = "true"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_COUPLING_METRICS_MISSING", codes)

    def test_thermo_spread_ratio_high_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_constitutive_material_spread_ratio=1.45,
        )
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_RATIO"] = "1.2"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_MATERIAL_SPREAD_RATIO_HIGH", codes)

    def test_thermo_assignment_heterogeneity_high_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_assignment_heterogeneity_index=0.42,
        )
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_INDEX"] = "0.2"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_ASSIGNMENT_HETEROGENEITY_HIGH", codes)

    def test_thermo_spread_breach_rate_high_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_constitutive_material_spread_ratio=1.4,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                thermo_coupling_enabled=True,
                thermo_constitutive_material_spread_ratio=1.35,
            ),
            report(
                passed=True,
                publishable=True,
                gpu_ms=92.0,
                thermo_coupling_enabled=True,
                thermo_constitutive_material_spread_ratio=1.1,
            ),
        ]
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_RATIO"] = "1.2"
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_BREACH_RATE"] = "0.5"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_SPREAD_BREACH_RATE_HIGH", codes)

    def test_thermo_heterogeneity_breach_rate_high_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_assignment_heterogeneity_index=0.4,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                thermo_coupling_enabled=True,
                thermo_assignment_heterogeneity_index=0.35,
            ),
            report(
                passed=True,
                publishable=True,
                gpu_ms=92.0,
                thermo_coupling_enabled=True,
                thermo_assignment_heterogeneity_index=0.1,
            ),
        ]
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_INDEX"] = "0.2"
        os.environ[
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_BREACH_RATE"
        ] = "0.5"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_HETEROGENEITY_BREACH_RATE_HIGH", codes)

    def test_markdown_summary_prints_thermo_posture_section(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_transient_severity=0.1,
            thermo_nonlinear_severity=0.2,
            thermo_constitutive_material_spread_ratio=1.1,
            thermo_assignment_heterogeneity_index=0.08,
        )
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        summary = markdown_summary(result)
        self.assertIn("### Thermo Posture", summary)
        self.assertIn("Thermo coupling enabled-rate", summary)
        self.assertIn("Max thermo transient severity", summary)
        self.assertIn("Max thermo nonlinear severity", summary)
        self.assertIn("Max thermo material spread ratio", summary)
        self.assertIn("Max thermo assignment heterogeneity index", summary)
        self.assertIn("Thermo spread breach rate", summary)
        self.assertIn("Thermo heterogeneity breach rate", summary)


if __name__ == "__main__":
    unittest.main()
