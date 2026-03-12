import unittest
import os

from scripts.analysis.governance.release_readiness_nonlinear import (
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
    thermo_spatial_coverage_ratio=None,
    thermo_field_extrapolation_ratio=None,
    thermo_field_artifact_id=None,
    thermo_field_artifact_approved=None,
    thermo_field_artifact_age_days=None,
    thermo_field_artifact_provenance_valid=None,
    electro_thermal_coupling_enabled=None,
    electro_transient_severity=None,
    electro_nonlinear_severity=None,
    electro_joule_heating_scale=None,
    electro_conductivity_spread_ratio=None,
    plastic_nonlinear_severity=None,
    contact_nonlinear_severity=None,
):
    fixtures = [
        "nonlinear_assembly_gpu_provider",
        "nonlinear_assembly_stress_gpu_provider",
        "nonlinear_softening_proxy_gpu_provider",
        "nonlinear_load_path_mix_gpu_provider",
        "nonlinear_contact_proxy_gpu_provider",
        "nonlinear_contact_frictionless_reference_gpu_provider",
        "nonlinear_contact_frictionless_reference_complex_gpu_provider",
        "nonlinear_plastic_hardening_reference_gpu_provider",
        "nonlinear_plastic_hardening_reference_complex_gpu_provider",
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
    if thermo_spatial_coverage_ratio is not None:
        for rec in records:
            rec["thermo_spatial_coverage_ratio"] = thermo_spatial_coverage_ratio
    if thermo_field_extrapolation_ratio is not None:
        for rec in records:
            rec["thermo_field_extrapolation_ratio"] = thermo_field_extrapolation_ratio
    if thermo_field_artifact_id is not None:
        for rec in records:
            rec["thermo_field_artifact_id"] = thermo_field_artifact_id
    if thermo_field_artifact_approved is not None:
        for rec in records:
            rec["thermo_field_artifact_approved"] = thermo_field_artifact_approved
    if thermo_field_artifact_age_days is not None:
        for rec in records:
            rec["thermo_field_artifact_age_days"] = thermo_field_artifact_age_days
    if thermo_field_artifact_provenance_valid is not None:
        for rec in records:
            rec["thermo_field_artifact_provenance_valid"] = (
                thermo_field_artifact_provenance_valid
            )
    if electro_thermal_coupling_enabled is not None:
        for rec in records:
            rec["electro_thermal_coupling_enabled"] = electro_thermal_coupling_enabled
    if electro_transient_severity is not None:
        for rec in records:
            rec["electro_transient_severity"] = electro_transient_severity
    if electro_nonlinear_severity is not None:
        for rec in records:
            rec["electro_nonlinear_severity"] = electro_nonlinear_severity
    if electro_joule_heating_scale is not None:
        for rec in records:
            rec["electro_joule_heating_scale"] = electro_joule_heating_scale
    if electro_conductivity_spread_ratio is not None:
        for rec in records:
            rec["electro_conductivity_spread_ratio"] = electro_conductivity_spread_ratio
    if plastic_nonlinear_severity is not None:
        for rec in records:
            rec["plastic_nonlinear_severity"] = plastic_nonlinear_severity
    if contact_nonlinear_severity is not None:
        for rec in records:
            rec["contact_nonlinear_severity"] = contact_nonlinear_severity

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
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_TREND_RATIO",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_TREND_RATIO",
            "RUNMAT_RELEASE_READINESS_THERMO_MIN_FIELD_COVERAGE_RATIO",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_EXTRAPOLATION_RATIO",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_COVERAGE_DROP_TREND_RATIO",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_EXTRAPOLATION_TREND_RATIO",
            "RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_ARTIFACT_BACKED",
            "RUNMAT_RELEASE_READINESS_THERMO_FIELD_ARTIFACT_MAX_AGE_DAYS",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TRANSIENT_SEVERITY",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_NONLINEAR_SEVERITY",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MIN_ENABLED_RATE",
            "RUNMAT_RELEASE_READINESS_ELECTRO_REQUIRE_METRICS",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_HEATING_SCALE",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_CONDUCTIVITY_SPREAD_RATIO",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_BREACH_RATE",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_SPREAD_BREACH_RATE",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_TREND_RATIO",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_SPREAD_TREND_RATIO",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_NONLINEAR_SEVERITY",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_BREACH_RATE",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_TREND_RATIO",
            "RUNMAT_RELEASE_READINESS_PLASTIC_REQUIRE_METRICS",
            "RUNMAT_RELEASE_READINESS_PLASTIC_REFERENCE_MAX_TREND_RATIO",
            "RUNMAT_RELEASE_READINESS_PLASTIC_PROMOTION_MIN_SAMPLES",
            "RUNMAT_RELEASE_READINESS_PLASTIC_PROMOTION_MAX_BLOCKERS",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_NONLINEAR_SEVERITY",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_BREACH_RATE",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_TREND_RATIO",
            "RUNMAT_RELEASE_READINESS_CONTACT_REQUIRE_METRICS",
            "RUNMAT_RELEASE_READINESS_CONTACT_REFERENCE_MAX_TREND_RATIO",
            "RUNMAT_RELEASE_READINESS_CONTACT_PROMOTION_MIN_SAMPLES",
            "RUNMAT_RELEASE_READINESS_CONTACT_PROMOTION_MAX_BLOCKERS",
            "RUNMAT_RELEASE_READINESS_REQUIRE_PROMOTION_READY",
            "RUNMAT_RELEASE_READINESS_PROMOTION_MAX_BLOCKER_REGRESSION",
            "RUNMAT_RELEASE_READINESS_REQUIRE_PROMOTION_CALIBRATION",
            "RUNMAT_RELEASE_READINESS_PROMOTION_MIN_ROLLING_REPORTS",
            "RUNMAT_RELEASE_READINESS_PROMOTION_CALIBRATION_MAX_AGE_DAYS",
            "RUNMAT_THERMO_FIELD_PROMOTION_REPORT",
            "RUNMAT_THERMO_FIELD_SIGNING_KEY",
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

    def test_feature_branch_profile_tolerates_pathological_spread_baseline(self):
        os.environ["GITHUB_REF_NAME"] = "feature/sandbox"
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_constitutive_material_spread_ratio=1.6,
        )
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertNotIn("THERMO_MATERIAL_SPREAD_RATIO_HIGH", codes)
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

    def test_electro_transient_severity_high_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            electro_thermal_coupling_enabled=True,
            electro_transient_severity=0.55,
        )
        os.environ["RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TRANSIENT_SEVERITY"] = "0.2"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("ELECTRO_TRANSIENT_SEVERITY_HIGH", codes)

    def test_electro_nonlinear_severity_high_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            electro_thermal_coupling_enabled=True,
            electro_nonlinear_severity=0.55,
        )
        os.environ["RUNMAT_RELEASE_READINESS_ELECTRO_MAX_NONLINEAR_SEVERITY"] = "0.2"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("ELECTRO_NONLINEAR_SEVERITY_HIGH", codes)

    def test_electro_metrics_missing_warn_when_required(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        os.environ["RUNMAT_RELEASE_READINESS_ELECTRO_REQUIRE_METRICS"] = "true"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("ELECTRO_COUPLING_METRICS_MISSING", codes)

    def test_electro_joule_heating_scale_high_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            electro_thermal_coupling_enabled=True,
            electro_joule_heating_scale=11.6,
        )
        os.environ["RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_HEATING_SCALE"] = "10.5"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("ELECTRO_JOULE_HEATING_SCALE_HIGH", codes)

    def test_electro_spread_ratio_high_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            electro_thermal_coupling_enabled=True,
            electro_conductivity_spread_ratio=2.1,
        )
        os.environ["RUNMAT_RELEASE_READINESS_ELECTRO_MAX_CONDUCTIVITY_SPREAD_RATIO"] = "1.8"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("ELECTRO_CONDUCTIVITY_SPREAD_RATIO_HIGH", codes)

    def test_electro_joule_breach_rate_high_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            electro_thermal_coupling_enabled=True,
            electro_joule_heating_scale=11.4,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                electro_thermal_coupling_enabled=True,
                electro_joule_heating_scale=11.0,
            ),
            report(
                passed=True,
                publishable=True,
                gpu_ms=92.0,
                electro_thermal_coupling_enabled=True,
                electro_joule_heating_scale=9.8,
            ),
        ]
        os.environ["RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_HEATING_SCALE"] = "10.5"
        os.environ["RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_BREACH_RATE"] = "0.5"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("ELECTRO_JOULE_BREACH_RATE_HIGH", codes)

    def test_electro_spread_trend_worsening_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            electro_thermal_coupling_enabled=True,
            electro_conductivity_spread_ratio=2.0,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                electro_thermal_coupling_enabled=True,
                electro_conductivity_spread_ratio=1.4,
            )
        ]
        os.environ["RUNMAT_RELEASE_READINESS_ELECTRO_MAX_SPREAD_TREND_RATIO"] = "1.3"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("ELECTRO_SPREAD_TREND_WORSENING", codes)

    def test_plastic_nonlinear_severity_high_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            plastic_nonlinear_severity=0.92,
        )
        os.environ["RUNMAT_RELEASE_READINESS_PLASTIC_MAX_NONLINEAR_SEVERITY"] = "0.8"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PLASTIC_NONLINEAR_SEVERITY_HIGH", codes)

    def test_plastic_metrics_missing_warn_when_required(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        os.environ["RUNMAT_RELEASE_READINESS_PLASTIC_REQUIRE_METRICS"] = "true"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PLASTIC_METRICS_MISSING", codes)

    def test_plastic_breach_rate_high_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            plastic_nonlinear_severity=0.9,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                plastic_nonlinear_severity=0.84,
            ),
            report(
                passed=True,
                publishable=True,
                gpu_ms=92.0,
                plastic_nonlinear_severity=0.5,
            ),
        ]
        os.environ["RUNMAT_RELEASE_READINESS_PLASTIC_MAX_NONLINEAR_SEVERITY"] = "0.8"
        os.environ["RUNMAT_RELEASE_READINESS_PLASTIC_MAX_BREACH_RATE"] = "0.5"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PLASTIC_BREACH_RATE_HIGH", codes)

    def test_plastic_trend_worsening_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            plastic_nonlinear_severity=0.88,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                plastic_nonlinear_severity=0.6,
            )
        ]
        os.environ["RUNMAT_RELEASE_READINESS_PLASTIC_MAX_TREND_RATIO"] = "1.3"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PLASTIC_TREND_WORSENING", codes)

    def test_plastic_reference_trend_worsening_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            plastic_nonlinear_severity=0.42,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                plastic_nonlinear_severity=0.3,
            )
        ]
        os.environ["RUNMAT_RELEASE_READINESS_PLASTIC_REFERENCE_MAX_TREND_RATIO"] = "1.3"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PLASTIC_REFERENCE_TREND_WORSENING", codes)

    def test_contact_nonlinear_severity_high_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            contact_nonlinear_severity=0.92,
        )
        os.environ["RUNMAT_RELEASE_READINESS_CONTACT_MAX_NONLINEAR_SEVERITY"] = "0.8"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("CONTACT_NONLINEAR_SEVERITY_HIGH", codes)

    def test_contact_metrics_missing_warn_when_required(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        os.environ["RUNMAT_RELEASE_READINESS_CONTACT_REQUIRE_METRICS"] = "true"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("CONTACT_METRICS_MISSING", codes)

    def test_contact_breach_rate_high_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            contact_nonlinear_severity=0.9,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                contact_nonlinear_severity=0.84,
            ),
            report(
                passed=True,
                publishable=True,
                gpu_ms=92.0,
                contact_nonlinear_severity=0.5,
            ),
        ]
        os.environ["RUNMAT_RELEASE_READINESS_CONTACT_MAX_NONLINEAR_SEVERITY"] = "0.8"
        os.environ["RUNMAT_RELEASE_READINESS_CONTACT_MAX_BREACH_RATE"] = "0.5"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("CONTACT_BREACH_RATE_HIGH", codes)

    def test_contact_trend_worsening_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            contact_nonlinear_severity=0.88,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                contact_nonlinear_severity=0.6,
            )
        ]
        os.environ["RUNMAT_RELEASE_READINESS_CONTACT_MAX_TREND_RATIO"] = "1.3"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("CONTACT_TREND_WORSENING", codes)

    def test_contact_reference_trend_worsening_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            contact_nonlinear_severity=0.45,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                contact_nonlinear_severity=0.3,
            )
        ]
        os.environ["RUNMAT_RELEASE_READINESS_CONTACT_REFERENCE_MAX_TREND_RATIO"] = "1.3"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("CONTACT_REFERENCE_TREND_WORSENING", codes)

    def test_promotion_ready_signals_true_when_reference_metrics_are_healthy(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            plastic_nonlinear_severity=0.22,
            contact_nonlinear_severity=0.22,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                plastic_nonlinear_severity=0.21,
                contact_nonlinear_severity=0.21,
            )
        ]
        os.environ["RUNMAT_RELEASE_READINESS_REQUIRE_PROMOTION_READY"] = "true"
        os.environ["RUNMAT_RELEASE_READINESS_PLASTIC_PROMOTION_MIN_SAMPLES"] = "2"
        os.environ["RUNMAT_RELEASE_READINESS_CONTACT_PROMOTION_MIN_SAMPLES"] = "2"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        self.assertTrue(result["plastic_promotion_ready"])
        self.assertTrue(result["contact_promotion_ready"])
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertNotIn("PLASTIC_PROMOTION_NOT_READY", codes)
        self.assertNotIn("CONTACT_PROMOTION_NOT_READY", codes)

    def test_promotion_not_ready_reason_when_reference_samples_insufficient(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            plastic_nonlinear_severity=0.22,
            contact_nonlinear_severity=0.22,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                plastic_nonlinear_severity=0.21,
                contact_nonlinear_severity=0.21,
            )
        ]
        os.environ["RUNMAT_RELEASE_READINESS_REQUIRE_PROMOTION_READY"] = "true"
        os.environ["RUNMAT_RELEASE_READINESS_PLASTIC_PROMOTION_MIN_SAMPLES"] = "3"
        os.environ["RUNMAT_RELEASE_READINESS_CONTACT_PROMOTION_MIN_SAMPLES"] = "3"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        self.assertFalse(result["plastic_promotion_ready"])
        self.assertFalse(result["contact_promotion_ready"])
        self.assertIn("sample_count<3", result["plastic_promotion_blockers"])
        self.assertIn("sample_count<3", result["contact_promotion_blockers"])
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PLASTIC_PROMOTION_NOT_READY", codes)
        self.assertIn("CONTACT_PROMOTION_NOT_READY", codes)

    def test_promotion_blocker_budget_exceeded_reasons_are_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            plastic_nonlinear_severity=0.9,
            contact_nonlinear_severity=0.9,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                plastic_nonlinear_severity=0.2,
                contact_nonlinear_severity=0.2,
            )
        ]
        os.environ["RUNMAT_RELEASE_READINESS_PLASTIC_PROMOTION_MAX_BLOCKERS"] = "0"
        os.environ["RUNMAT_RELEASE_READINESS_CONTACT_PROMOTION_MAX_BLOCKERS"] = "0"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PLASTIC_PROMOTION_BLOCKER_BUDGET_EXCEEDED", codes)
        self.assertIn("CONTACT_PROMOTION_BLOCKER_BUDGET_EXCEEDED", codes)

    def test_promotion_blocker_burndown_stalled_reasons_are_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            plastic_nonlinear_severity=0.9,
            contact_nonlinear_severity=0.9,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                plastic_nonlinear_severity=0.2,
                contact_nonlinear_severity=0.2,
            )
        ]
        os.environ["RUNMAT_RELEASE_READINESS_PROMOTION_MAX_BLOCKER_REGRESSION"] = "0"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PLASTIC_PROMOTION_BLOCKER_BURNDOWN_STALLED", codes)
        self.assertIn("CONTACT_PROMOTION_BLOCKER_BURNDOWN_STALLED", codes)

    def test_promotion_history_insufficient_reason_is_emitted_when_required(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        os.environ["RUNMAT_RELEASE_READINESS_REQUIRE_PROMOTION_READY"] = "true"
        os.environ["RUNMAT_RELEASE_READINESS_PROMOTION_MIN_ROLLING_REPORTS"] = "2"
        result = evaluate_release_readiness(latest, [], protected=False)
        self.assertFalse(result["promotion_history_sufficient"])
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PROMOTION_HISTORY_INSUFFICIENT", codes)
        self.assertIn("PLASTIC_PROMOTION_BLOCKER_BASELINE_MISSING", codes)
        self.assertIn("CONTACT_PROMOTION_BLOCKER_BASELINE_MISSING", codes)

    def test_promotion_calibration_stale_reason_is_emitted_when_required(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        rolling = [report(passed=True, publishable=True, gpu_ms=95.0)]
        os.environ["RUNMAT_RELEASE_READINESS_REQUIRE_PROMOTION_CALIBRATION"] = "true"
        os.environ["RUNMAT_RELEASE_READINESS_PROMOTION_CALIBRATION_MAX_AGE_DAYS"] = "7"
        calibration = {
            "generated_at": "2000-01-01T00:00:00Z",
            "by_profile": {
                "feature": {
                    "plastic_promotion_max_blockers": 1,
                    "contact_promotion_max_blockers": 1,
                    "promotion_max_blocker_regression": 0,
                }
            },
        }
        result = evaluate_release_readiness(
            latest,
            rolling,
            protected=False,
            promotion_calibration=calibration,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PROMOTION_CALIBRATION_STALE", codes)

    def test_promotion_calibration_is_applied_when_present(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            plastic_nonlinear_severity=0.2,
            contact_nonlinear_severity=0.2,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                plastic_nonlinear_severity=0.2,
                contact_nonlinear_severity=0.2,
            )
        ]
        calibration = {
            "by_profile": {
                "feature": {
                    "plastic_promotion_max_blockers": 0,
                    "contact_promotion_max_blockers": 0,
                    "promotion_max_blocker_regression": 0,
                }
            }
        }
        result = evaluate_release_readiness(
            latest,
            rolling,
            protected=False,
            promotion_calibration=calibration,
        )
        self.assertTrue(result["promotion_calibration_applied"])
        self.assertEqual(result["plastic_promotion_max_blockers"], 0)
        self.assertEqual(result["contact_promotion_max_blockers"], 0)

    def test_missing_promotion_calibration_reason_when_required(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        rolling = [report(passed=True, publishable=True, gpu_ms=95.0)]
        os.environ["RUNMAT_RELEASE_READINESS_REQUIRE_PROMOTION_CALIBRATION"] = "true"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("PROMOTION_CALIBRATION_MISSING", codes)

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

    def test_thermo_spread_trend_worsening_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_constitutive_material_spread_ratio=1.35,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                thermo_coupling_enabled=True,
                thermo_constitutive_material_spread_ratio=1.05,
            )
        ]
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_TREND_RATIO"] = "1.15"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_SPREAD_TREND_WORSENING", codes)

    def test_thermo_heterogeneity_trend_worsening_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_assignment_heterogeneity_index=0.34,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                thermo_coupling_enabled=True,
                thermo_assignment_heterogeneity_index=0.18,
            )
        ]
        os.environ[
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_TREND_RATIO"
        ] = "1.3"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_HETEROGENEITY_TREND_WORSENING", codes)

    def test_thermo_field_coverage_low_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_spatial_coverage_ratio=0.2,
        )
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_MIN_FIELD_COVERAGE_RATIO"] = "0.4"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_FIELD_COVERAGE_LOW", codes)

    def test_thermo_field_extrapolation_high_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_field_extrapolation_ratio=0.3,
        )
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_EXTRAPOLATION_RATIO"] = "0.1"
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_FIELD_EXTRAPOLATION_HIGH", codes)

    def test_thermo_field_coverage_trend_worsening_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_spatial_coverage_ratio=0.3,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                thermo_coupling_enabled=True,
                thermo_spatial_coverage_ratio=0.6,
            )
        ]
        os.environ[
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_COVERAGE_DROP_TREND_RATIO"
        ] = "1.5"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_FIELD_COVERAGE_TREND_WORSENING", codes)

    def test_thermo_field_extrapolation_trend_worsening_reason_is_emitted(self):
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_field_extrapolation_ratio=0.18,
        )
        rolling = [
            report(
                passed=True,
                publishable=True,
                gpu_ms=95.0,
                thermo_coupling_enabled=True,
                thermo_field_extrapolation_ratio=0.09,
            )
        ]
        os.environ[
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_EXTRAPOLATION_TREND_RATIO"
        ] = "1.5"
        result = evaluate_release_readiness(latest, rolling, protected=False)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_FIELD_EXTRAPOLATION_TREND_WORSENING", codes)

    def test_thermo_artifact_backed_required_reason_is_emitted(self):
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_ARTIFACT_BACKED"] = "true"
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
        )
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_FIELD_ARTIFACT_REQUIRED", codes)

    def test_thermo_artifact_unapproved_reason_is_emitted(self):
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_ARTIFACT_BACKED"] = "true"
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_field_artifact_id="artifact-1",
            thermo_field_artifact_approved=False,
        )
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_FIELD_ARTIFACT_UNAPPROVED", codes)

    def test_thermo_artifact_stale_reason_is_emitted(self):
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_ARTIFACT_BACKED"] = "true"
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_FIELD_ARTIFACT_MAX_AGE_DAYS"] = "7"
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_field_artifact_id="artifact-1",
            thermo_field_artifact_approved=True,
            thermo_field_artifact_age_days=10.0,
        )
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_FIELD_ARTIFACT_STALE", codes)

    def test_thermo_artifact_provenance_invalid_reason_is_emitted(self):
        os.environ["RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_ARTIFACT_BACKED"] = "true"
        latest = report(
            passed=True,
            publishable=True,
            gpu_ms=100.0,
            thermo_coupling_enabled=True,
            thermo_field_artifact_id="artifact-1",
            thermo_field_artifact_approved=True,
            thermo_field_artifact_provenance_valid=False,
        )
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_FIELD_ARTIFACT_PROVENANCE_INVALID", codes)

    def test_thermo_field_promotion_blocked_reason_is_emitted(self):
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
            thermo_promotion_report={
                "blocked": True,
                "reasons": ["THERMO_FIELD_PROMOTION_TREND_DRIFT_BLOCKED:1.4>1.2"],
            },
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_FIELD_PROMOTION_BLOCKED", codes)

    def test_release_profile_requires_non_default_thermo_signing_key(self):
        os.environ["GITHUB_REF_NAME"] = "main"
        latest = report(passed=True, publishable=True, gpu_ms=100.0)
        result = evaluate_release_readiness(
            latest,
            [report(passed=True, publishable=True, gpu_ms=95.0)],
            protected=False,
        )
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("THERMO_FIELD_SIGNING_KEY_UNSAFE", codes)

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
            thermo_spatial_coverage_ratio=0.7,
            thermo_field_extrapolation_ratio=0.01,
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
        self.assertIn("Thermo spread ratio threshold", summary)
        self.assertIn("Max thermo assignment heterogeneity index", summary)
        self.assertIn("Min thermo field coverage ratio", summary)
        self.assertIn("Thermo field coverage threshold", summary)
        self.assertIn("Max thermo field extrapolation ratio", summary)
        self.assertIn("Thermo field extrapolation threshold", summary)
        self.assertIn("Thermo field coverage drop trend ratio", summary)
        self.assertIn("Thermo field coverage drop trend threshold", summary)
        self.assertIn("Thermo field extrapolation trend ratio", summary)
        self.assertIn("Thermo field extrapolation trend threshold", summary)
        self.assertIn("Thermo artifact-backed required", summary)
        self.assertIn("Thermo field artifact max age days", summary)
        self.assertIn("Thermo signing key safe", summary)
        self.assertIn("Thermo spread breach rate", summary)
        self.assertIn("Thermo heterogeneity breach rate", summary)
        self.assertIn("Thermo spread trend ratio", summary)
        self.assertIn("Thermo heterogeneity trend ratio", summary)
        self.assertIn("### Promotion Evidence Quality", summary)
        self.assertIn("Promotion calibration applied/required", summary)
        self.assertIn("Promotion calibration age/max days", summary)
        self.assertIn("Promotion history sufficient (rolling/min)", summary)


if __name__ == "__main__":
    unittest.main()
