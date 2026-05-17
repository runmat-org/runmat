#!/usr/bin/env python3
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.analysis.prep_calibration.evaluate_prep_calibration_drift import (
    load_evidence,
    evaluate_report_drift,
    recommend_profile_shifts,
    validate_evidence,
    validate_recommendation_artifact,
)


NONLINEAR_FIXTURES = {
    "nonlinear_assembly_gpu_provider",
    "nonlinear_assembly_stress_gpu_provider",
    "nonlinear_softening_proxy_gpu_provider",
    "nonlinear_load_path_mix_gpu_provider",
}

KEY_PERFORMANCE_FIXTURES = {
    "electromagnetic_reference_homogeneous_gpu_provider",
    "electromagnetic_reference_heterogeneous_gpu_provider",
    "electromagnetic_reference_boundary_penalty_stress_gpu_provider",
    "electromagnetic_reference_multi_region_phased_source_gpu_provider",
    "acoustic_harmonic_gpu_provider",
    "cfd_steady_gpu_provider",
    "cht_coupled_gpu_provider",
    "fsi_coupled_gpu_provider",
}

EM_FIXTURES = {
    "electromagnetic_reference_homogeneous_gpu_provider",
    "electromagnetic_reference_heterogeneous_gpu_provider",
    "electromagnetic_reference_sparse_assignments_gpu_provider",
    "electromagnetic_reference_fallback_heavy_gpu_provider",
    "electromagnetic_reference_overlap_interference_gpu_provider",
    "electromagnetic_reference_boundary_kernel_gpu_provider",
    "electromagnetic_reference_boundary_penalty_stress_gpu_provider",
    "electromagnetic_reference_multi_region_phased_source_gpu_provider",
}


@dataclass
class Reason:
    code: str
    severity: str
    detail: str


def is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def governance_profile_name() -> str:
    ref_name = os.getenv("GITHUB_REF_NAME", "")
    if ref_name in {"main", "master"} or ref_name.startswith("release/"):
        return "release"
    if ref_name in {"develop", "dev"}:
        return "development"
    return "feature"


def profile_default(name: str, default: str) -> str:
    profile = governance_profile_name()
    profile_map = {
        "release": {
            "RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_MAX_DRIFT": "0.15",
            "RUNMAT_RELEASE_READINESS_PREP_MAX_RECOMMENDATION_RATIO": "0.25",
            "RUNMAT_RELEASE_READINESS_PREP_CANDIDATE_MAX_AGE_DAYS": "7",
            "RUNMAT_RELEASE_READINESS_PREP_REQUIRE_RECOMMENDATION_ARTIFACT": "true",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_TRANSIENT_SEVERITY": "0.25",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_NONLINEAR_SEVERITY": "0.25",
            "RUNMAT_RELEASE_READINESS_THERMO_MIN_ENABLED_RATE": "0.5",
            "RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_METRICS": "true",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_INDEX": "0.2",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_BREACH_RATE": "0.1",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_BREACH_RATE": "0.1",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_TREND_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_TREND_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_THERMO_MIN_FIELD_COVERAGE_RATIO": "0.55",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_EXTRAPOLATION_RATIO": "0.02",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_CLAMP_RATIO": "0.03",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_COVERAGE_DROP_TREND_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_EXTRAPOLATION_TREND_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_CLAMP_TREND_RATIO": "1.15",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_CLAMP_BREACH_RATE": "0.1",
            "RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_ARTIFACT_BACKED": "true",
            "RUNMAT_RELEASE_READINESS_THERMO_FIELD_ARTIFACT_MAX_AGE_DAYS": "14",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_RESIDUAL_NORM": "6.8",
            "RUNMAT_RELEASE_READINESS_THERMAL_MIN_TEMPERATURE_K": "292.0",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_TEMPERATURE_K": "375.0",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_CONDUCTIVITY_SPREAD_RATIO": "1.6",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_HEAT_CAPACITY_SPREAD_RATIO": "1.6",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPATIAL_GRADIENT_INDEX": "1.5",
            "RUNMAT_RELEASE_READINESS_THERMAL_MIN_MONOTONIC_RESPONSE_FRACTION": "0.8",
            "RUNMAT_RELEASE_READINESS_THERMAL_MIN_RESPONSE_REALIZATION_RATIO": "0.5",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_RESPONSE_REALIZATION_RATIO": "1.6",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPREAD_BREACH_RATE": "0.05",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPREAD_TREND_RATIO": "1.08",
            "RUNMAT_RELEASE_READINESS_THERMAL_REQUIRE_METRICS": "true",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TRANSIENT_SEVERITY": "0.25",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_NONLINEAR_SEVERITY": "0.25",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MIN_ENABLED_RATE": "0.5",
            "RUNMAT_RELEASE_READINESS_ELECTRO_REQUIRE_METRICS": "true",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_HEATING_SCALE": "10.5",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_CONDUCTIVITY_SPREAD_RATIO": "1.8",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_BREACH_RATE": "0.1",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_SPREAD_BREACH_RATE": "0.1",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_TREND_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_SPREAD_TREND_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MIN_TIME_SCALE_MEAN": "0.75",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TIME_SCALE_MEAN": "1.25",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TIME_SCALE_BREACH_RATE": "0.1",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TIME_SCALE_TREND_RATIO": "1.15",
            "RUNMAT_RELEASE_READINESS_EM_REQUIRE_METRICS": "true",
            "RUNMAT_RELEASE_READINESS_EM_MAX_ENERGY_IMBALANCE_RATIO": "0.25",
            "RUNMAT_RELEASE_READINESS_EM_MAX_FLUX_DIVERGENCE_PROXY": "0.35",
            "RUNMAT_RELEASE_READINESS_EM_MAX_REAL_RESIDUAL_NORM": "0.15",
            "RUNMAT_RELEASE_READINESS_EM_MAX_IMAG_RESIDUAL_NORM": "0.15",
            "RUNMAT_RELEASE_READINESS_EM_MIN_SOURCE_REALIZATION_RATIO": "0.55",
            "RUNMAT_RELEASE_READINESS_EM_MIN_SOURCE_REGION_COVERAGE_RATIO": "0.55",
            "RUNMAT_RELEASE_READINESS_EM_MIN_SOURCE_MATERIAL_ALIGNMENT_RATIO": "0.55",
            "RUNMAT_RELEASE_READINESS_EM_MAX_BREACH_RATE": "0.1",
            "RUNMAT_RELEASE_READINESS_EM_MAX_ENERGY_IMBALANCE_TREND_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_EM_MAX_FLUX_DIVERGENCE_TREND_RATIO": "1.15",
            "RUNMAT_RELEASE_READINESS_EM_MAX_DISPERSIVE_PHASE_ATTENUATION_TREND_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_EM_MAX_DISPERSIVE_PHASE_CONDUCTIVITY_ATTENUATION_TREND_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_EM_MIN_DISPERSIVE_PHASE_ATTENUATION_MEAN": "0.95",
            "RUNMAT_RELEASE_READINESS_EM_MIN_DISPERSIVE_PHASE_CONDUCTIVITY_ATTENUATION_RATIO": "0.95",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_NONLINEAR_SEVERITY": "0.65",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MIN_LOAD_REALIZATION_RATIO": "0.72",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_REALIZATION_RATIO": "0.98",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MIN_LOAD_AMPLIFICATION_RATIO": "1.3",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_AMPLIFICATION_RATIO": "1.8",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_BREACH_RATE": "0.1",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_REALIZATION_BREACH_RATE": "0.1",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_AMPLIFICATION_BREACH_RATE": "0.1",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_TREND_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_REALIZATION_DROP_TREND_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_AMPLIFICATION_TREND_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_PLASTIC_REQUIRE_METRICS": "true",
            "RUNMAT_RELEASE_READINESS_PLASTIC_REFERENCE_MAX_TREND_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_NONLINEAR_SEVERITY": "0.65",
            "RUNMAT_RELEASE_READINESS_CONTACT_MIN_LOAD_REALIZATION_RATIO": "0.75",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_REALIZATION_RATIO": "0.98",
            "RUNMAT_RELEASE_READINESS_CONTACT_MIN_LOAD_AMPLIFICATION_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_AMPLIFICATION_RATIO": "1.7",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_BREACH_RATE": "0.1",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_REALIZATION_BREACH_RATE": "0.1",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_AMPLIFICATION_BREACH_RATE": "0.1",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_TREND_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_REALIZATION_DROP_TREND_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_AMPLIFICATION_TREND_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_CONTACT_REQUIRE_METRICS": "true",
            "RUNMAT_RELEASE_READINESS_CONTACT_REFERENCE_MAX_TREND_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_PLASTIC_PROMOTION_MIN_SAMPLES": "2",
            "RUNMAT_RELEASE_READINESS_CONTACT_PROMOTION_MIN_SAMPLES": "2",
            "RUNMAT_RELEASE_READINESS_REQUIRE_PROMOTION_READY": "true",
            "RUNMAT_RELEASE_READINESS_PLASTIC_PROMOTION_MAX_BLOCKERS": "0",
            "RUNMAT_RELEASE_READINESS_CONTACT_PROMOTION_MAX_BLOCKERS": "0",
            "RUNMAT_RELEASE_READINESS_PROMOTION_MAX_BLOCKER_REGRESSION": "0",
            "RUNMAT_RELEASE_READINESS_PROMOTION_MIN_ROLLING_REPORTS": "4",
            "RUNMAT_RELEASE_READINESS_PROMOTION_CALIBRATION_MAX_AGE_DAYS": "7",
            "RUNMAT_RELEASE_READINESS_REQUIRE_KEY_PERF_FIXTURES": "true",
            "RUNMAT_RELEASE_READINESS_KEY_PERF_MIN_SPEEDUP_RATIO": "1.02",
            "RUNMAT_RELEASE_READINESS_KEY_PERF_MAX_SLOWDOWN_RATIO": "1.25",
        },
        "development": {
            "RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_MAX_DRIFT": "0.2",
            "RUNMAT_RELEASE_READINESS_PREP_MAX_RECOMMENDATION_RATIO": "0.5",
            "RUNMAT_RELEASE_READINESS_PREP_CANDIDATE_MAX_AGE_DAYS": "14",
            "RUNMAT_RELEASE_READINESS_PREP_REQUIRE_RECOMMENDATION_ARTIFACT": "false",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_TRANSIENT_SEVERITY": "0.3",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_NONLINEAR_SEVERITY": "0.3",
            "RUNMAT_RELEASE_READINESS_THERMO_MIN_ENABLED_RATE": "0.0",
            "RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_METRICS": "false",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_RATIO": "1.3",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_INDEX": "0.3",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_BREACH_RATE": "0.25",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_BREACH_RATE": "0.25",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_TREND_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_TREND_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_THERMO_MIN_FIELD_COVERAGE_RATIO": "0.45",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_EXTRAPOLATION_RATIO": "0.08",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_CLAMP_RATIO": "0.08",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_COVERAGE_DROP_TREND_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_EXTRAPOLATION_TREND_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_CLAMP_TREND_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_CLAMP_BREACH_RATE": "0.25",
            "RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_ARTIFACT_BACKED": "true",
            "RUNMAT_RELEASE_READINESS_THERMO_FIELD_ARTIFACT_MAX_AGE_DAYS": "21",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_RESIDUAL_NORM": "7.2",
            "RUNMAT_RELEASE_READINESS_THERMAL_MIN_TEMPERATURE_K": "288.0",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_TEMPERATURE_K": "390.0",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_CONDUCTIVITY_SPREAD_RATIO": "1.8",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_HEAT_CAPACITY_SPREAD_RATIO": "1.8",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPATIAL_GRADIENT_INDEX": "1.8",
            "RUNMAT_RELEASE_READINESS_THERMAL_MIN_MONOTONIC_RESPONSE_FRACTION": "0.7",
            "RUNMAT_RELEASE_READINESS_THERMAL_MIN_RESPONSE_REALIZATION_RATIO": "0.4",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_RESPONSE_REALIZATION_RATIO": "1.8",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPREAD_BREACH_RATE": "0.15",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPREAD_TREND_RATIO": "1.15",
            "RUNMAT_RELEASE_READINESS_THERMAL_REQUIRE_METRICS": "false",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TRANSIENT_SEVERITY": "0.3",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_NONLINEAR_SEVERITY": "0.3",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MIN_ENABLED_RATE": "0.0",
            "RUNMAT_RELEASE_READINESS_ELECTRO_REQUIRE_METRICS": "false",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_HEATING_SCALE": "11.0",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_CONDUCTIVITY_SPREAD_RATIO": "2.2",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_BREACH_RATE": "0.25",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_SPREAD_BREACH_RATE": "0.25",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_TREND_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_SPREAD_TREND_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MIN_TIME_SCALE_MEAN": "0.7",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TIME_SCALE_MEAN": "1.3",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TIME_SCALE_BREACH_RATE": "0.25",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TIME_SCALE_TREND_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_EM_REQUIRE_METRICS": "false",
            "RUNMAT_RELEASE_READINESS_EM_MAX_ENERGY_IMBALANCE_RATIO": "0.35",
            "RUNMAT_RELEASE_READINESS_EM_MAX_FLUX_DIVERGENCE_PROXY": "0.45",
            "RUNMAT_RELEASE_READINESS_EM_MAX_REAL_RESIDUAL_NORM": "0.2",
            "RUNMAT_RELEASE_READINESS_EM_MAX_IMAG_RESIDUAL_NORM": "0.2",
            "RUNMAT_RELEASE_READINESS_EM_MIN_SOURCE_REALIZATION_RATIO": "0.45",
            "RUNMAT_RELEASE_READINESS_EM_MIN_SOURCE_REGION_COVERAGE_RATIO": "0.45",
            "RUNMAT_RELEASE_READINESS_EM_MIN_SOURCE_MATERIAL_ALIGNMENT_RATIO": "0.45",
            "RUNMAT_RELEASE_READINESS_EM_MAX_BREACH_RATE": "0.25",
            "RUNMAT_RELEASE_READINESS_EM_MAX_ENERGY_IMBALANCE_TREND_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_EM_MAX_FLUX_DIVERGENCE_TREND_RATIO": "1.25",
            "RUNMAT_RELEASE_READINESS_EM_MAX_DISPERSIVE_PHASE_ATTENUATION_TREND_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_EM_MAX_DISPERSIVE_PHASE_CONDUCTIVITY_ATTENUATION_TREND_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_EM_MIN_DISPERSIVE_PHASE_ATTENUATION_MEAN": "0.85",
            "RUNMAT_RELEASE_READINESS_EM_MIN_DISPERSIVE_PHASE_CONDUCTIVITY_ATTENUATION_RATIO": "0.85",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_NONLINEAR_SEVERITY": "0.75",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MIN_LOAD_REALIZATION_RATIO": "0.68",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_REALIZATION_RATIO": "1.02",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MIN_LOAD_AMPLIFICATION_RATIO": "1.2",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_AMPLIFICATION_RATIO": "1.9",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_BREACH_RATE": "0.25",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_REALIZATION_BREACH_RATE": "0.25",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_AMPLIFICATION_BREACH_RATE": "0.25",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_TREND_RATIO": "1.16",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_REALIZATION_DROP_TREND_RATIO": "1.25",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_AMPLIFICATION_TREND_RATIO": "1.25",
            "RUNMAT_RELEASE_READINESS_PLASTIC_REQUIRE_METRICS": "false",
            "RUNMAT_RELEASE_READINESS_PLASTIC_REFERENCE_MAX_TREND_RATIO": "1.15",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_NONLINEAR_SEVERITY": "0.75",
            "RUNMAT_RELEASE_READINESS_CONTACT_MIN_LOAD_REALIZATION_RATIO": "0.7",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_REALIZATION_RATIO": "1.02",
            "RUNMAT_RELEASE_READINESS_CONTACT_MIN_LOAD_AMPLIFICATION_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_AMPLIFICATION_RATIO": "1.8",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_BREACH_RATE": "0.25",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_REALIZATION_BREACH_RATE": "0.25",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_AMPLIFICATION_BREACH_RATE": "0.25",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_TREND_RATIO": "1.16",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_REALIZATION_DROP_TREND_RATIO": "1.25",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_AMPLIFICATION_TREND_RATIO": "1.25",
            "RUNMAT_RELEASE_READINESS_CONTACT_REQUIRE_METRICS": "false",
            "RUNMAT_RELEASE_READINESS_CONTACT_REFERENCE_MAX_TREND_RATIO": "1.15",
            "RUNMAT_RELEASE_READINESS_PLASTIC_PROMOTION_MIN_SAMPLES": "2",
            "RUNMAT_RELEASE_READINESS_CONTACT_PROMOTION_MIN_SAMPLES": "2",
            "RUNMAT_RELEASE_READINESS_REQUIRE_PROMOTION_READY": "false",
            "RUNMAT_RELEASE_READINESS_PLASTIC_PROMOTION_MAX_BLOCKERS": "1",
            "RUNMAT_RELEASE_READINESS_CONTACT_PROMOTION_MAX_BLOCKERS": "1",
            "RUNMAT_RELEASE_READINESS_PROMOTION_MAX_BLOCKER_REGRESSION": "0",
            "RUNMAT_RELEASE_READINESS_PROMOTION_MIN_ROLLING_REPORTS": "2",
            "RUNMAT_RELEASE_READINESS_PROMOTION_CALIBRATION_MAX_AGE_DAYS": "14",
            "RUNMAT_RELEASE_READINESS_REQUIRE_KEY_PERF_FIXTURES": "false",
            "RUNMAT_RELEASE_READINESS_KEY_PERF_MIN_SPEEDUP_RATIO": "1.0",
            "RUNMAT_RELEASE_READINESS_KEY_PERF_MAX_SLOWDOWN_RATIO": "1.35",
        },
        "feature": {
            "RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_MAX_DRIFT": "0.25",
            "RUNMAT_RELEASE_READINESS_PREP_MAX_RECOMMENDATION_RATIO": "0.75",
            "RUNMAT_RELEASE_READINESS_PREP_CANDIDATE_MAX_AGE_DAYS": "21",
            "RUNMAT_RELEASE_READINESS_PREP_REQUIRE_RECOMMENDATION_ARTIFACT": "false",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_TRANSIENT_SEVERITY": "0.4",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_NONLINEAR_SEVERITY": "0.4",
            "RUNMAT_RELEASE_READINESS_THERMO_MIN_ENABLED_RATE": "0.0",
            "RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_METRICS": "false",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_RATIO": "1.6",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_INDEX": "0.5",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_BREACH_RATE": "0.5",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_BREACH_RATE": "0.5",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_THERMO_MIN_FIELD_COVERAGE_RATIO": "0.3",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_EXTRAPOLATION_RATIO": "0.18",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_CLAMP_RATIO": "0.2",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_COVERAGE_DROP_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_EXTRAPOLATION_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_CLAMP_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_CLAMP_BREACH_RATE": "0.5",
            "RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_ARTIFACT_BACKED": "false",
            "RUNMAT_RELEASE_READINESS_THERMO_FIELD_ARTIFACT_MAX_AGE_DAYS": "30",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_RESIDUAL_NORM": "8.0",
            "RUNMAT_RELEASE_READINESS_THERMAL_MIN_TEMPERATURE_K": "285.0",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_TEMPERATURE_K": "420.0",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_CONDUCTIVITY_SPREAD_RATIO": "2.2",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_HEAT_CAPACITY_SPREAD_RATIO": "2.2",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPATIAL_GRADIENT_INDEX": "2.5",
            "RUNMAT_RELEASE_READINESS_THERMAL_MIN_MONOTONIC_RESPONSE_FRACTION": "0.6",
            "RUNMAT_RELEASE_READINESS_THERMAL_MIN_RESPONSE_REALIZATION_RATIO": "0.3",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_RESPONSE_REALIZATION_RATIO": "2.2",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPREAD_BREACH_RATE": "0.4",
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPREAD_TREND_RATIO": "1.3",
            "RUNMAT_RELEASE_READINESS_THERMAL_REQUIRE_METRICS": "false",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TRANSIENT_SEVERITY": "0.4",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_NONLINEAR_SEVERITY": "0.4",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MIN_ENABLED_RATE": "0.0",
            "RUNMAT_RELEASE_READINESS_ELECTRO_REQUIRE_METRICS": "false",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_HEATING_SCALE": "12.5",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_CONDUCTIVITY_SPREAD_RATIO": "3.0",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_BREACH_RATE": "0.5",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_SPREAD_BREACH_RATE": "0.5",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_SPREAD_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MIN_TIME_SCALE_MEAN": "0.6",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TIME_SCALE_MEAN": "1.5",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TIME_SCALE_BREACH_RATE": "0.5",
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TIME_SCALE_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_EM_REQUIRE_METRICS": "false",
            "RUNMAT_RELEASE_READINESS_EM_MAX_ENERGY_IMBALANCE_RATIO": "0.5",
            "RUNMAT_RELEASE_READINESS_EM_MAX_FLUX_DIVERGENCE_PROXY": "0.6",
            "RUNMAT_RELEASE_READINESS_EM_MAX_REAL_RESIDUAL_NORM": "0.3",
            "RUNMAT_RELEASE_READINESS_EM_MAX_IMAG_RESIDUAL_NORM": "0.3",
            "RUNMAT_RELEASE_READINESS_EM_MIN_SOURCE_REALIZATION_RATIO": "0.3",
            "RUNMAT_RELEASE_READINESS_EM_MIN_SOURCE_REGION_COVERAGE_RATIO": "0.3",
            "RUNMAT_RELEASE_READINESS_EM_MIN_SOURCE_MATERIAL_ALIGNMENT_RATIO": "0.3",
            "RUNMAT_RELEASE_READINESS_EM_MAX_BREACH_RATE": "0.5",
            "RUNMAT_RELEASE_READINESS_EM_MAX_ENERGY_IMBALANCE_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_EM_MAX_FLUX_DIVERGENCE_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_EM_MAX_DISPERSIVE_PHASE_ATTENUATION_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_EM_MAX_DISPERSIVE_PHASE_CONDUCTIVITY_ATTENUATION_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_EM_MIN_DISPERSIVE_PHASE_ATTENUATION_MEAN": "0.7",
            "RUNMAT_RELEASE_READINESS_EM_MIN_DISPERSIVE_PHASE_CONDUCTIVITY_ATTENUATION_RATIO": "0.7",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_NONLINEAR_SEVERITY": "0.9",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MIN_LOAD_REALIZATION_RATIO": "0.6",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_REALIZATION_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MIN_LOAD_AMPLIFICATION_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_AMPLIFICATION_RATIO": "2.1",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_BREACH_RATE": "0.5",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_REALIZATION_BREACH_RATE": "0.5",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_AMPLIFICATION_BREACH_RATE": "0.5",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_TREND_RATIO": "1.28",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_REALIZATION_DROP_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_AMPLIFICATION_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_PLASTIC_REQUIRE_METRICS": "false",
            "RUNMAT_RELEASE_READINESS_PLASTIC_REFERENCE_MAX_TREND_RATIO": "1.26",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_NONLINEAR_SEVERITY": "0.9",
            "RUNMAT_RELEASE_READINESS_CONTACT_MIN_LOAD_REALIZATION_RATIO": "0.6",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_REALIZATION_RATIO": "1.1",
            "RUNMAT_RELEASE_READINESS_CONTACT_MIN_LOAD_AMPLIFICATION_RATIO": "1.0",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_AMPLIFICATION_RATIO": "2.0",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_BREACH_RATE": "0.5",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_REALIZATION_BREACH_RATE": "0.5",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_AMPLIFICATION_BREACH_RATE": "0.5",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_TREND_RATIO": "1.28",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_REALIZATION_DROP_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_AMPLIFICATION_TREND_RATIO": "1.35",
            "RUNMAT_RELEASE_READINESS_CONTACT_REQUIRE_METRICS": "false",
            "RUNMAT_RELEASE_READINESS_CONTACT_REFERENCE_MAX_TREND_RATIO": "1.26",
            "RUNMAT_RELEASE_READINESS_PLASTIC_PROMOTION_MIN_SAMPLES": "2",
            "RUNMAT_RELEASE_READINESS_CONTACT_PROMOTION_MIN_SAMPLES": "2",
            "RUNMAT_RELEASE_READINESS_REQUIRE_PROMOTION_READY": "false",
            "RUNMAT_RELEASE_READINESS_PLASTIC_PROMOTION_MAX_BLOCKERS": "2",
            "RUNMAT_RELEASE_READINESS_CONTACT_PROMOTION_MAX_BLOCKERS": "2",
            "RUNMAT_RELEASE_READINESS_PROMOTION_MAX_BLOCKER_REGRESSION": "1",
            "RUNMAT_RELEASE_READINESS_PROMOTION_MIN_ROLLING_REPORTS": "0",
            "RUNMAT_RELEASE_READINESS_PROMOTION_CALIBRATION_MAX_AGE_DAYS": "30",
            "RUNMAT_RELEASE_READINESS_REQUIRE_KEY_PERF_FIXTURES": "false",
            "RUNMAT_RELEASE_READINESS_KEY_PERF_MIN_SPEEDUP_RATIO": "0.9",
            "RUNMAT_RELEASE_READINESS_KEY_PERF_MAX_SLOWDOWN_RATIO": "1.5",
        },
    }
    return profile_map.get(profile, {}).get(name, default)


def is_protected_branch() -> bool:
    if not is_true(os.getenv("RUNMAT_ANALYSIS_ENFORCE_BASELINE_ON_PROTECTED", "false")):
        return False
    ref_name = os.getenv("GITHUB_REF_NAME", "")
    protected = {
        name.strip()
        for name in os.getenv("RUNMAT_ANALYSIS_PROTECTED_BRANCHES", "main,master,release").split(",")
        if name.strip()
    }
    if ref_name in protected:
        return True
    return any(ref_name.startswith(f"{name}/") for name in protected)


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def rolling_reports(rolling_dir: Path) -> List[dict]:
    if not rolling_dir.exists():
        return []
    reports = []
    for path in sorted(rolling_dir.glob("analysis_benchmark_report_rolling_*.json")):
        parsed = load_json(path)
        if isinstance(parsed, dict):
            reports.append(parsed)
    return reports


def parse_iso8601_utc(raw: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def prep_artifacts(prep_root: Path) -> List[dict]:
    prep_dir = prep_root / "prep"
    if not prep_dir.exists():
        return []
    artifacts = []
    for path in sorted(prep_dir.glob("*.json")):
        parsed = load_json(path)
        if isinstance(parsed, dict):
            artifacts.append(parsed)
    return artifacts


def prep_health_from_artifacts(artifacts: List[dict]) -> dict:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    ages = []
    for artifact in artifacts:
        created = artifact.get("created_at")
        if isinstance(created, str):
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                ages.append(max((now - dt).total_seconds(), 0.0))
            except Exception:
                continue
    ages.sort()
    if not ages:
        p95_age = 0.0
    elif len(ages) == 1:
        p95_age = ages[0]
    else:
        idx = round((len(ages) - 1) * 0.95)
        p95_age = ages[int(idx)]
    return {
        "artifact_count": len(artifacts),
        "p95_age_seconds": p95_age,
    }


def env_u64(name: str, default: int = 0) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def nonlinear_records(report: dict) -> Dict[str, dict]:
    return records_for_fixtures(report, NONLINEAR_FIXTURES)


def records_for_fixtures(report: dict, fixture_ids: set[str]) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    for rec in report.get("records", []):
        if not isinstance(rec, dict):
            continue
        fixture_id = rec.get("fixture_id")
        if isinstance(fixture_id, str) and fixture_id in fixture_ids:
            records[fixture_id] = rec
    return records


def report_records(report: dict) -> List[dict]:
    records = []
    for rec in report.get("records", []):
        if isinstance(rec, dict):
            records.append(rec)
    return records


def threshold_assertion_observed(record: dict, assertion_name: str) -> float | None:
    assertions = record.get("threshold_assertions")
    if not isinstance(assertions, list):
        return None
    for assertion in assertions:
        if not isinstance(assertion, dict):
            continue
        if assertion.get("name") != assertion_name:
            continue
        observed = assertion.get("observed")
        if isinstance(observed, (int, float)):
            value = float(observed)
            if math.isfinite(value):
                return value
        return None
    return None


def evaluate_release_readiness(
    latest: dict,
    rolling: List[dict],
    protected: bool,
    prep_health: dict | None = None,
    calibration_evidence: dict | None = None,
    recommendation_artifact: dict | None = None,
    thermo_promotion_report: dict | None = None,
    promotion_calibration: dict | None = None,
) -> dict:
    reasons: List[Reason] = []
    latest_passed = bool(latest.get("passed", False))
    if not latest_passed:
        reasons.append(
            Reason(
                code="CONFORMANCE_FAILED",
                severity="fail",
                detail="latest nonlinear conformance report is not passed",
            )
        )

    records = nonlinear_records(latest)
    missing = sorted(NONLINEAR_FIXTURES.difference(records.keys()))
    if missing:
        reasons.append(
            Reason(
                code="NONLINEAR_FIXTURE_MISSING",
                severity="fail" if protected else "warn",
                detail=f"missing nonlinear fixture records: {', '.join(missing)}",
            )
        )

    unpublished = sorted(
        fixture
        for fixture, rec in records.items()
        if rec.get("publishable") is False
    )
    if unpublished:
        reasons.append(
            Reason(
                code="NONLINEAR_FIXTURE_UNPUBLISHABLE",
                severity="fail" if protected else "warn",
                detail=f"non-publishable nonlinear fixtures: {', '.join(unpublished)}",
            )
        )

    max_slowdown_ratio = float(
        os.getenv("RUNMAT_RELEASE_READINESS_MAX_SLOWDOWN_RATIO", "1.25")
    )
    require_trends = is_true(os.getenv("RUNMAT_RELEASE_READINESS_REQUIRE_TRENDS", "false"))
    if not rolling:
        if protected or require_trends:
            reasons.append(
                Reason(
                    code="TREND_DATA_MISSING",
                    severity="warn",
                    detail="no rolling reports available for trend comparison",
                )
            )
    else:
        hist = {fixture: [] for fixture in NONLINEAR_FIXTURES}
        for report in rolling:
            for fixture, rec in nonlinear_records(report).items():
                value = rec.get("gpu_run_ms")
                if isinstance(value, (int, float)) and value > 0:
                    hist[fixture].append(float(value))
        for fixture, rec in records.items():
            current = rec.get("gpu_run_ms")
            history = hist.get(fixture, [])
            if not isinstance(current, (int, float)) or current <= 0 or not history:
                continue
            baseline = statistics.median(history)
            if baseline <= 0:
                continue
            ratio = float(current) / baseline
            if ratio > max_slowdown_ratio:
                reasons.append(
                    Reason(
                        code="NONLINEAR_TREND_SLOWDOWN",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"{fixture} gpu_run_ms slowdown ratio {ratio:.3f} exceeds "
                            f"{max_slowdown_ratio:.3f}"
                        ),
                    )
                )

    key_perf_require_fixtures = is_true(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_REQUIRE_KEY_PERF_FIXTURES",
            profile_default("RUNMAT_RELEASE_READINESS_REQUIRE_KEY_PERF_FIXTURES", "false"),
        )
    )
    key_perf_min_speedup_ratio = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_KEY_PERF_MIN_SPEEDUP_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_KEY_PERF_MIN_SPEEDUP_RATIO", "1.0"),
        )
    )
    key_perf_max_slowdown_ratio = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_KEY_PERF_MAX_SLOWDOWN_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_KEY_PERF_MAX_SLOWDOWN_RATIO", "1.35"),
        )
    )
    key_perf_records = records_for_fixtures(latest, KEY_PERFORMANCE_FIXTURES)
    if key_perf_require_fixtures:
        missing_key_perf = sorted(KEY_PERFORMANCE_FIXTURES.difference(key_perf_records.keys()))
        if missing_key_perf:
            reasons.append(
                Reason(
                    code="KEY_PERF_FIXTURE_MISSING",
                    severity="fail" if protected else "warn",
                    detail=(
                        "missing key-performance fixture records: "
                        + ", ".join(missing_key_perf)
                    ),
                )
            )
    for fixture, rec in key_perf_records.items():
        speedup = rec.get("gpu_speedup_ratio")
        if isinstance(speedup, (int, float)) and math.isfinite(float(speedup)):
            if float(speedup) < key_perf_min_speedup_ratio:
                reasons.append(
                    Reason(
                        code="KEY_PERF_SPEEDUP_LOW",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"{fixture} gpu_speedup_ratio {float(speedup):.3f} below "
                            f"minimum {key_perf_min_speedup_ratio:.3f}"
                        ),
                    )
                )
    if rolling and key_perf_records:
        hist = {fixture: [] for fixture in key_perf_records}
        for report in rolling:
            for fixture, rec in records_for_fixtures(report, set(hist.keys())).items():
                value = rec.get("gpu_run_ms")
                if isinstance(value, (int, float)) and value > 0:
                    hist[fixture].append(float(value))
        for fixture, rec in key_perf_records.items():
            current = rec.get("gpu_run_ms")
            history = hist.get(fixture, [])
            if not isinstance(current, (int, float)) or current <= 0 or not history:
                continue
            baseline = statistics.median(history)
            if baseline <= 0:
                continue
            ratio = float(current) / baseline
            if ratio > key_perf_max_slowdown_ratio:
                reasons.append(
                    Reason(
                        code="KEY_PERF_TREND_SLOWDOWN",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"{fixture} gpu_run_ms slowdown ratio {ratio:.3f} exceeds "
                            f"{key_perf_max_slowdown_ratio:.3f}"
                        ),
                    )
                )

    replay_ok = is_true(os.getenv("RUNMAT_RELEASE_READINESS_ARTIFACT_REPLAY_OK", "false"))
    compat_ok = is_true(os.getenv("RUNMAT_RELEASE_READINESS_ARTIFACT_COMPAT_OK", "false"))
    if not replay_ok:
        reasons.append(
            Reason(
                code="ARTIFACT_REPLAY_UNVERIFIED",
                severity="fail" if protected else "warn",
                detail="artifact replay verification did not run or failed",
            )
        )
    if not compat_ok:
        reasons.append(
            Reason(
                code="ARTIFACT_COMPAT_UNVERIFIED",
                severity="fail" if protected else "warn",
                detail="artifact compatibility verification did not run or failed",
            )
        )

    prep_acceptance_min_rate = float(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_ACCEPTANCE_MIN_RATE", "0.9")
    )
    prep_acceptance_require = is_true(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_ACCEPTANCE_REQUIRE", "false")
    )
    acceptance_flags = []
    for rec in records.values():
        accepted = rec.get("prep_acceptance_passed")
        if isinstance(accepted, bool):
            acceptance_flags.append(accepted)
    acceptance_rate = None
    if not acceptance_flags:
        if protected or prep_acceptance_require:
            reasons.append(
                Reason(
                    code="PREP_ACCEPTANCE_MISSING",
                    severity="warn",
                    detail="prep acceptance metrics missing from nonlinear fixture records",
                )
            )
    else:
        acceptance_rate = sum(1 for flag in acceptance_flags if flag) / len(acceptance_flags)
        if acceptance_rate < prep_acceptance_min_rate:
            reasons.append(
                Reason(
                    code="PREP_ACCEPTANCE_RATE_LOW",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"prep acceptance pass rate {acceptance_rate:.3f} below "
                        f"minimum {prep_acceptance_min_rate:.3f}"
                    ),
                )
            )

    prep_warn_count = env_u64("RUNMAT_RELEASE_READINESS_PREP_WARN_ARTIFACT_COUNT", 64)
    prep_fail_count = env_u64("RUNMAT_RELEASE_READINESS_PREP_FAIL_ARTIFACT_COUNT", 128)
    prep_warn_p95_age = float(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_WARN_P95_AGE_SECONDS", "604800")
    )
    prep_fail_p95_age = float(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_FAIL_P95_AGE_SECONDS", "1209600")
    )
    prep_require_health = is_true(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_REQUIRE_HEALTH", "false")
    )

    if prep_health is None:
        if protected or prep_require_health:
            reasons.append(
                Reason(
                    code="PREP_HEALTH_MISSING",
                    severity="warn",
                    detail="prep artifact health data missing for release readiness evaluation",
                )
            )
    else:
        artifact_count = int(prep_health.get("artifact_count", 0))
        p95_age = float(prep_health.get("p95_age_seconds", 0.0))

        if artifact_count >= prep_fail_count:
            reasons.append(
                Reason(
                    code="PREP_SLO_COUNT_EXCEEDED",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"prep artifact count {artifact_count} exceeds "
                        f"fail threshold {prep_fail_count}"
                    ),
                )
            )
        elif artifact_count >= prep_warn_count:
            reasons.append(
                Reason(
                    code="PREP_SLO_COUNT_EXCEEDED",
                    severity="warn",
                    detail=(
                        f"prep artifact count {artifact_count} exceeds "
                        f"warn threshold {prep_warn_count}"
                    ),
                )
            )

        if p95_age >= prep_fail_p95_age:
            reasons.append(
                Reason(
                    code="PREP_SLO_AGE_EXCEEDED",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"prep artifact p95 age {p95_age:.1f}s exceeds "
                        f"fail threshold {prep_fail_p95_age:.1f}s"
                    ),
                )
            )
        elif p95_age >= prep_warn_p95_age:
            reasons.append(
                Reason(
                    code="PREP_SLO_AGE_EXCEEDED",
                    severity="warn",
                    detail=(
                        f"prep artifact p95 age {p95_age:.1f}s exceeds "
                        f"warn threshold {prep_warn_p95_age:.1f}s"
                    ),
                )
            )

    created_count = env_u64("RUNMAT_PREP_CREATED_COUNT", 0)
    stale_reject_count = env_u64("RUNMAT_PREP_STALE_REJECT_COUNT", 0)
    mismatch_reject_count = env_u64("RUNMAT_PREP_MISMATCH_REJECT_COUNT", 0)
    reject_rate_threshold = float(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_MAX_REJECT_RATE", "0.25")
    )
    if created_count > 0:
        reject_rate = (stale_reject_count + mismatch_reject_count) / created_count
        if reject_rate > reject_rate_threshold:
            reasons.append(
                Reason(
                    code="PREP_REJECT_RATE_HIGH",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"prep reject rate {reject_rate:.3f} exceeds "
                        f"threshold {reject_rate_threshold:.3f}"
                    ),
                )
            )

    prep_calibration_max_drift = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_MAX_DRIFT",
            profile_default("RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_MAX_DRIFT", "0.2"),
        )
    )
    prep_calibration_require_evidence = is_true(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_REQUIRE_EVIDENCE", "false")
    )
    prep_calibration_max_observed_drift = None
    prep_calibration_recommendation_count = 0
    if calibration_evidence is None:
        if protected or prep_calibration_require_evidence:
            reasons.append(
                Reason(
                    code="PREP_CALIBRATION_EVIDENCE_MISSING",
                    severity="warn",
                    detail="prep calibration evidence artifact missing for drift evaluation",
                )
            )
    else:
        evidence_status = validate_evidence(calibration_evidence)
        if not evidence_status.get("valid", False):
            reasons.append(
                Reason(
                    code="PREP_CALIBRATION_EVIDENCE_INVALID",
                    severity="fail" if protected else "warn",
                    detail="prep calibration evidence artifact is invalid",
                )
            )
        elif evidence_status.get("stale", False):
            reasons.append(
                Reason(
                    code="PREP_CALIBRATION_EVIDENCE_STALE",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"prep calibration evidence age {evidence_status.get('age_days', 0.0):.1f}d "
                        f"exceeds max {evidence_status.get('max_age_days', 0.0):.1f}d"
                    ),
                )
            )
        if evidence_status.get("state") == "candidate":
            candidate_max_age_days = float(
                os.getenv(
                    "RUNMAT_RELEASE_READINESS_PREP_CANDIDATE_MAX_AGE_DAYS",
                    profile_default("RUNMAT_RELEASE_READINESS_PREP_CANDIDATE_MAX_AGE_DAYS", "14"),
                )
            )
            age_days = evidence_status.get("age_days")
            if isinstance(age_days, (int, float)) and age_days > candidate_max_age_days:
                reasons.append(
                    Reason(
                        code="PREP_CALIBRATION_CANDIDATE_STALE",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"candidate evidence age {age_days:.1f}d exceeds max "
                            f"{candidate_max_age_days:.1f}d"
                        ),
                    )
                )
        drift_rows = evaluate_report_drift(latest, calibration_evidence)
        if drift_rows:
            prep_calibration_max_observed_drift = max(
                row.get("drift_ratio", 0.0) for row in drift_rows
            )
            if prep_calibration_max_observed_drift > prep_calibration_max_drift:
                offending = [
                    row["fixture_id"]
                    for row in drift_rows
                    if row.get("drift_ratio", 0.0) > prep_calibration_max_drift
                ]
                reasons.append(
                    Reason(
                        code="PREP_CALIBRATION_DRIFT_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max calibration drift {prep_calibration_max_observed_drift:.3f} exceeds "
                            f"threshold {prep_calibration_max_drift:.3f}; fixtures: {', '.join(offending)}"
                        ),
                    )
                )

        recommendation_trigger = float(
            os.getenv("RUNMAT_RELEASE_READINESS_PREP_RETRAIN_TRIGGER_DRIFT", "0.1")
        )
        recommendations = recommend_profile_shifts(
            latest,
            rolling,
            calibration_evidence,
            drift_trigger=recommendation_trigger,
        )
        prep_calibration_recommendation_count = len(recommendations)
        recommendation_ratio = (
            prep_calibration_recommendation_count / len(records) if records else 0.0
        )
        max_recommendation_ratio = float(
            os.getenv(
                "RUNMAT_RELEASE_READINESS_PREP_MAX_RECOMMENDATION_RATIO",
                profile_default("RUNMAT_RELEASE_READINESS_PREP_MAX_RECOMMENDATION_RATIO", "0.5"),
            )
        )
        if recommendation_ratio > max_recommendation_ratio:
            fixtures = [item["fixture_id"] for item in recommendations]
            reasons.append(
                Reason(
                    code="PREP_CALIBRATION_RETRAIN_RECOMMENDED",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"recommended retrain ratio {recommendation_ratio:.3f} exceeds "
                        f"threshold {max_recommendation_ratio:.3f}; fixtures: {', '.join(fixtures)}"
                    ),
                )
            )

    recommendation_require = is_true(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PREP_REQUIRE_RECOMMENDATION_ARTIFACT",
            profile_default("RUNMAT_RELEASE_READINESS_PREP_REQUIRE_RECOMMENDATION_ARTIFACT", "false"),
        )
    )
    if recommendation_artifact is None:
        if protected or recommendation_require:
            reasons.append(
                Reason(
                    code="PREP_CALIBRATION_RECOMMENDATION_ARTIFACT_MISSING",
                    severity="warn",
                    detail="prep calibration recommendation artifact missing",
                )
            )
    else:
        recommendation_status = validate_recommendation_artifact(recommendation_artifact)
        if not recommendation_status.get("valid", False):
            reasons.append(
                Reason(
                    code="PREP_CALIBRATION_RECOMMENDATION_ARTIFACT_INVALID",
                    severity="fail" if protected else "warn",
                    detail="prep calibration recommendation artifact invalid",
                )
            )
        elif recommendation_status.get("stale", False):
            reasons.append(
                Reason(
                    code="PREP_CALIBRATION_RECOMMENDATION_ARTIFACT_STALE",
                    severity="warn",
                    detail="prep calibration recommendation artifact is stale",
                )
            )

    thermo_max_transient_severity_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_TRANSIENT_SEVERITY",
            profile_default("RUNMAT_RELEASE_READINESS_THERMO_MAX_TRANSIENT_SEVERITY", "0.3"),
        )
    )
    thermo_max_nonlinear_severity_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_NONLINEAR_SEVERITY",
            profile_default("RUNMAT_RELEASE_READINESS_THERMO_MAX_NONLINEAR_SEVERITY", "0.3"),
        )
    )
    thermo_min_enabled_rate = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MIN_ENABLED_RATE",
            profile_default("RUNMAT_RELEASE_READINESS_THERMO_MIN_ENABLED_RATE", "0.0"),
        )
    )
    thermo_require_metrics = is_true(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_METRICS",
            profile_default("RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_METRICS", "false"),
        )
    )
    thermo_max_spread_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_RATIO", "1.3"),
        )
    )
    thermo_max_heterogeneity_index_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_INDEX",
            profile_default(
                "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_INDEX", "0.3"
            ),
        )
    )
    thermo_max_spread_breach_rate_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_BREACH_RATE",
            profile_default(
                "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_BREACH_RATE", "0.25"
            ),
        )
    )
    thermo_max_heterogeneity_breach_rate_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_BREACH_RATE",
            profile_default(
                "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_BREACH_RATE", "0.25"
            ),
        )
    )
    thermo_min_field_coverage_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MIN_FIELD_COVERAGE_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_THERMO_MIN_FIELD_COVERAGE_RATIO", "0.45"),
        )
    )
    thermo_max_field_extrapolation_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_EXTRAPOLATION_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_EXTRAPOLATION_RATIO", "0.08"
            ),
        )
    )
    thermo_max_field_clamp_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_CLAMP_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_CLAMP_RATIO", "0.08"),
        )
    )
    thermo_max_field_coverage_drop_trend_ratio = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_COVERAGE_DROP_TREND_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_COVERAGE_DROP_TREND_RATIO",
                "1.2",
            ),
        )
    )
    thermo_max_field_extrapolation_trend_ratio = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_EXTRAPOLATION_TREND_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_EXTRAPOLATION_TREND_RATIO",
                "1.2",
            ),
        )
    )
    thermo_max_field_clamp_trend_ratio = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_CLAMP_TREND_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_CLAMP_TREND_RATIO", "1.2"
            ),
        )
    )
    thermo_max_field_clamp_breach_rate_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_CLAMP_BREACH_RATE",
            profile_default(
                "RUNMAT_RELEASE_READINESS_THERMO_MAX_FIELD_CLAMP_BREACH_RATE", "0.25"
            ),
        )
    )
    thermo_require_artifact_backed = is_true(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_ARTIFACT_BACKED",
            profile_default("RUNMAT_RELEASE_READINESS_THERMO_REQUIRE_ARTIFACT_BACKED", "false"),
        )
    )
    thermo_field_artifact_max_age_days = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_FIELD_ARTIFACT_MAX_AGE_DAYS",
            profile_default("RUNMAT_RELEASE_READINESS_THERMO_FIELD_ARTIFACT_MAX_AGE_DAYS", "30"),
        )
    )
    thermo_max_spread_trend_ratio = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_TREND_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_THERMO_MAX_SPREAD_TREND_RATIO", "1.2"),
        )
    )
    thermo_max_heterogeneity_trend_ratio = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_TREND_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_THERMO_MAX_HETEROGENEITY_TREND_RATIO", "1.2"
            ),
        )
    )
    thermal_max_residual_norm_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_RESIDUAL_NORM",
            profile_default("RUNMAT_RELEASE_READINESS_THERMAL_MAX_RESIDUAL_NORM", "8.0"),
        )
    )
    thermal_min_temperature_k_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMAL_MIN_TEMPERATURE_K",
            profile_default("RUNMAT_RELEASE_READINESS_THERMAL_MIN_TEMPERATURE_K", "280.0"),
        )
    )
    thermal_max_temperature_k_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_TEMPERATURE_K",
            profile_default("RUNMAT_RELEASE_READINESS_THERMAL_MAX_TEMPERATURE_K", "430.0"),
        )
    )
    thermal_max_conductivity_spread_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_CONDUCTIVITY_SPREAD_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_THERMAL_MAX_CONDUCTIVITY_SPREAD_RATIO", "2.5"
            ),
        )
    )
    thermal_max_heat_capacity_spread_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_HEAT_CAPACITY_SPREAD_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_THERMAL_MAX_HEAT_CAPACITY_SPREAD_RATIO", "2.5"
            ),
        )
    )
    thermal_max_spatial_gradient_index_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPATIAL_GRADIENT_INDEX",
            profile_default(
                "RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPATIAL_GRADIENT_INDEX", "2.5"
            ),
        )
    )
    thermal_min_monotonic_response_fraction_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMAL_MIN_MONOTONIC_RESPONSE_FRACTION",
            profile_default(
                "RUNMAT_RELEASE_READINESS_THERMAL_MIN_MONOTONIC_RESPONSE_FRACTION", "0.6"
            ),
        )
    )
    thermal_min_response_realization_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMAL_MIN_RESPONSE_REALIZATION_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_THERMAL_MIN_RESPONSE_REALIZATION_RATIO", "0.3"
            ),
        )
    )
    thermal_max_response_realization_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_RESPONSE_REALIZATION_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_THERMAL_MAX_RESPONSE_REALIZATION_RATIO", "2.2"
            ),
        )
    )
    thermal_max_spread_breach_rate_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPREAD_BREACH_RATE",
            profile_default("RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPREAD_BREACH_RATE", "0.5"),
        )
    )
    thermal_max_spread_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPREAD_TREND_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPREAD_TREND_RATIO", "1.35"),
        )
    )
    thermal_require_metrics = is_true(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_THERMAL_REQUIRE_METRICS",
            profile_default("RUNMAT_RELEASE_READINESS_THERMAL_REQUIRE_METRICS", "false"),
        )
    )
    electro_max_transient_severity_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TRANSIENT_SEVERITY",
            profile_default("RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TRANSIENT_SEVERITY", "0.3"),
        )
    )
    electro_max_nonlinear_severity_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_NONLINEAR_SEVERITY",
            profile_default("RUNMAT_RELEASE_READINESS_ELECTRO_MAX_NONLINEAR_SEVERITY", "0.3"),
        )
    )
    electro_min_enabled_rate = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_ELECTRO_MIN_ENABLED_RATE",
            profile_default("RUNMAT_RELEASE_READINESS_ELECTRO_MIN_ENABLED_RATE", "0.0"),
        )
    )
    electro_require_metrics = is_true(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_ELECTRO_REQUIRE_METRICS",
            profile_default("RUNMAT_RELEASE_READINESS_ELECTRO_REQUIRE_METRICS", "false"),
        )
    )
    electro_max_joule_heating_scale_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_HEATING_SCALE",
            profile_default("RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_HEATING_SCALE", "11.0"),
        )
    )
    electro_max_conductivity_spread_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_CONDUCTIVITY_SPREAD_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_CONDUCTIVITY_SPREAD_RATIO", "2.2"
            ),
        )
    )
    electro_max_joule_breach_rate_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_BREACH_RATE",
            profile_default("RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_BREACH_RATE", "0.25"),
        )
    )
    electro_max_spread_breach_rate_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_SPREAD_BREACH_RATE",
            profile_default("RUNMAT_RELEASE_READINESS_ELECTRO_MAX_SPREAD_BREACH_RATE", "0.25"),
        )
    )
    electro_max_joule_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_TREND_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_ELECTRO_MAX_JOULE_TREND_RATIO", "1.2"),
        )
    )
    electro_max_spread_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_SPREAD_TREND_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_ELECTRO_MAX_SPREAD_TREND_RATIO", "1.2"),
        )
    )
    electro_min_time_scale_mean_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_ELECTRO_MIN_TIME_SCALE_MEAN",
            profile_default("RUNMAT_RELEASE_READINESS_ELECTRO_MIN_TIME_SCALE_MEAN", "0.7"),
        )
    )
    electro_max_time_scale_mean_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TIME_SCALE_MEAN",
            profile_default("RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TIME_SCALE_MEAN", "1.3"),
        )
    )
    electro_max_time_scale_breach_rate_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TIME_SCALE_BREACH_RATE",
            profile_default("RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TIME_SCALE_BREACH_RATE", "0.25"),
        )
    )
    electro_max_time_scale_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TIME_SCALE_TREND_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_ELECTRO_MAX_TIME_SCALE_TREND_RATIO", "1.2"),
        )
    )
    em_require_metrics = is_true(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_EM_REQUIRE_METRICS",
            profile_default("RUNMAT_RELEASE_READINESS_EM_REQUIRE_METRICS", "false"),
        )
    )
    em_max_energy_imbalance_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_EM_MAX_ENERGY_IMBALANCE_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_EM_MAX_ENERGY_IMBALANCE_RATIO", "0.35"),
        )
    )
    em_max_flux_divergence_proxy_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_EM_MAX_FLUX_DIVERGENCE_PROXY",
            profile_default("RUNMAT_RELEASE_READINESS_EM_MAX_FLUX_DIVERGENCE_PROXY", "0.45"),
        )
    )
    em_max_real_residual_norm_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_EM_MAX_REAL_RESIDUAL_NORM",
            profile_default("RUNMAT_RELEASE_READINESS_EM_MAX_REAL_RESIDUAL_NORM", "0.2"),
        )
    )
    em_max_imag_residual_norm_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_EM_MAX_IMAG_RESIDUAL_NORM",
            profile_default("RUNMAT_RELEASE_READINESS_EM_MAX_IMAG_RESIDUAL_NORM", "0.2"),
        )
    )
    em_min_source_realization_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_EM_MIN_SOURCE_REALIZATION_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_EM_MIN_SOURCE_REALIZATION_RATIO", "0.45"),
        )
    )
    em_min_source_region_coverage_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_EM_MIN_SOURCE_REGION_COVERAGE_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_EM_MIN_SOURCE_REGION_COVERAGE_RATIO", "0.45"),
        )
    )
    em_min_source_material_alignment_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_EM_MIN_SOURCE_MATERIAL_ALIGNMENT_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_EM_MIN_SOURCE_MATERIAL_ALIGNMENT_RATIO",
                "0.45",
            ),
        )
    )
    em_max_breach_rate_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_EM_MAX_BREACH_RATE",
            profile_default("RUNMAT_RELEASE_READINESS_EM_MAX_BREACH_RATE", "0.25"),
        )
    )
    em_max_energy_imbalance_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_EM_MAX_ENERGY_IMBALANCE_TREND_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_EM_MAX_ENERGY_IMBALANCE_TREND_RATIO", "1.2"
            ),
        )
    )
    em_max_flux_divergence_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_EM_MAX_FLUX_DIVERGENCE_TREND_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_EM_MAX_FLUX_DIVERGENCE_TREND_RATIO", "1.25"
            ),
        )
    )
    em_max_dispersive_phase_attenuation_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_EM_MAX_DISPERSIVE_PHASE_ATTENUATION_TREND_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_EM_MAX_DISPERSIVE_PHASE_ATTENUATION_TREND_RATIO",
                "1.2",
            ),
        )
    )
    em_max_dispersive_phase_conductivity_attenuation_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_EM_MAX_DISPERSIVE_PHASE_CONDUCTIVITY_ATTENUATION_TREND_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_EM_MAX_DISPERSIVE_PHASE_CONDUCTIVITY_ATTENUATION_TREND_RATIO",
                "1.2",
            ),
        )
    )
    em_min_dispersive_phase_attenuation_mean_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_EM_MIN_DISPERSIVE_PHASE_ATTENUATION_MEAN",
            profile_default(
                "RUNMAT_RELEASE_READINESS_EM_MIN_DISPERSIVE_PHASE_ATTENUATION_MEAN", "0.85"
            ),
        )
    )
    em_min_dispersive_phase_conductivity_attenuation_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_EM_MIN_DISPERSIVE_PHASE_CONDUCTIVITY_ATTENUATION_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_EM_MIN_DISPERSIVE_PHASE_CONDUCTIVITY_ATTENUATION_RATIO",
                "0.85",
            ),
        )
    )
    plastic_max_nonlinear_severity_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_NONLINEAR_SEVERITY",
            profile_default("RUNMAT_RELEASE_READINESS_PLASTIC_MAX_NONLINEAR_SEVERITY", "0.75"),
        )
    )
    plastic_max_breach_rate_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_BREACH_RATE",
            profile_default("RUNMAT_RELEASE_READINESS_PLASTIC_MAX_BREACH_RATE", "0.25"),
        )
    )
    plastic_min_load_realization_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PLASTIC_MIN_LOAD_REALIZATION_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_PLASTIC_MIN_LOAD_REALIZATION_RATIO", "0.7"),
        )
    )
    plastic_max_load_realization_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_REALIZATION_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_REALIZATION_RATIO", "1.0"),
        )
    )
    plastic_min_load_amplification_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PLASTIC_MIN_LOAD_AMPLIFICATION_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_PLASTIC_MIN_LOAD_AMPLIFICATION_RATIO", "1.2"
            ),
        )
    )
    plastic_max_load_amplification_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_AMPLIFICATION_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_AMPLIFICATION_RATIO", "1.9"
            ),
        )
    )
    plastic_max_load_realization_breach_rate_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_REALIZATION_BREACH_RATE",
            profile_default(
                "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_REALIZATION_BREACH_RATE", "0.25"
            ),
        )
    )
    plastic_max_load_amplification_breach_rate_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_AMPLIFICATION_BREACH_RATE",
            profile_default(
                "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_AMPLIFICATION_BREACH_RATE", "0.25"
            ),
        )
    )
    plastic_max_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_TREND_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_PLASTIC_MAX_TREND_RATIO", "1.2"),
        )
    )
    plastic_max_load_realization_drop_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_REALIZATION_DROP_TREND_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_REALIZATION_DROP_TREND_RATIO", "1.2"
            ),
        )
    )
    plastic_max_load_amplification_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_AMPLIFICATION_TREND_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_LOAD_AMPLIFICATION_TREND_RATIO", "1.2"
            ),
        )
    )
    plastic_require_metrics = is_true(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PLASTIC_REQUIRE_METRICS",
            profile_default("RUNMAT_RELEASE_READINESS_PLASTIC_REQUIRE_METRICS", "false"),
        )
    )
    plastic_reference_max_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PLASTIC_REFERENCE_MAX_TREND_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_PLASTIC_REFERENCE_MAX_TREND_RATIO", "1.2"
            ),
        )
    )
    contact_max_nonlinear_severity_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_NONLINEAR_SEVERITY",
            profile_default("RUNMAT_RELEASE_READINESS_CONTACT_MAX_NONLINEAR_SEVERITY", "0.75"),
        )
    )
    contact_max_breach_rate_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_BREACH_RATE",
            profile_default("RUNMAT_RELEASE_READINESS_CONTACT_MAX_BREACH_RATE", "0.25"),
        )
    )
    contact_min_load_realization_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_CONTACT_MIN_LOAD_REALIZATION_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_CONTACT_MIN_LOAD_REALIZATION_RATIO", "0.72"),
        )
    )
    contact_max_load_realization_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_REALIZATION_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_REALIZATION_RATIO", "1.0"),
        )
    )
    contact_min_load_amplification_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_CONTACT_MIN_LOAD_AMPLIFICATION_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_CONTACT_MIN_LOAD_AMPLIFICATION_RATIO", "1.1"
            ),
        )
    )
    contact_max_load_amplification_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_AMPLIFICATION_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_AMPLIFICATION_RATIO", "1.8"
            ),
        )
    )
    contact_max_load_realization_breach_rate_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_REALIZATION_BREACH_RATE",
            profile_default(
                "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_REALIZATION_BREACH_RATE", "0.25"
            ),
        )
    )
    contact_max_load_amplification_breach_rate_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_AMPLIFICATION_BREACH_RATE",
            profile_default(
                "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_AMPLIFICATION_BREACH_RATE", "0.25"
            ),
        )
    )
    contact_max_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_TREND_RATIO",
            profile_default("RUNMAT_RELEASE_READINESS_CONTACT_MAX_TREND_RATIO", "1.2"),
        )
    )
    contact_max_load_realization_drop_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_REALIZATION_DROP_TREND_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_REALIZATION_DROP_TREND_RATIO", "1.2"
            ),
        )
    )
    contact_max_load_amplification_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_AMPLIFICATION_TREND_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_CONTACT_MAX_LOAD_AMPLIFICATION_TREND_RATIO", "1.2"
            ),
        )
    )
    contact_require_metrics = is_true(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_CONTACT_REQUIRE_METRICS",
            profile_default("RUNMAT_RELEASE_READINESS_CONTACT_REQUIRE_METRICS", "false"),
        )
    )
    contact_reference_max_trend_ratio_threshold = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_CONTACT_REFERENCE_MAX_TREND_RATIO",
            profile_default(
                "RUNMAT_RELEASE_READINESS_CONTACT_REFERENCE_MAX_TREND_RATIO", "1.2"
            ),
        )
    )
    plastic_promotion_min_samples = int(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PLASTIC_PROMOTION_MIN_SAMPLES",
            profile_default("RUNMAT_RELEASE_READINESS_PLASTIC_PROMOTION_MIN_SAMPLES", "2"),
        )
    )
    contact_promotion_min_samples = int(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_CONTACT_PROMOTION_MIN_SAMPLES",
            profile_default("RUNMAT_RELEASE_READINESS_CONTACT_PROMOTION_MIN_SAMPLES", "2"),
        )
    )
    require_promotion_ready = is_true(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_REQUIRE_PROMOTION_READY",
            profile_default("RUNMAT_RELEASE_READINESS_REQUIRE_PROMOTION_READY", "false"),
        )
    )
    plastic_promotion_max_blockers = int(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PLASTIC_PROMOTION_MAX_BLOCKERS",
            profile_default("RUNMAT_RELEASE_READINESS_PLASTIC_PROMOTION_MAX_BLOCKERS", "1"),
        )
    )
    contact_promotion_max_blockers = int(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_CONTACT_PROMOTION_MAX_BLOCKERS",
            profile_default("RUNMAT_RELEASE_READINESS_CONTACT_PROMOTION_MAX_BLOCKERS", "1"),
        )
    )
    promotion_max_blocker_regression = int(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PROMOTION_MAX_BLOCKER_REGRESSION",
            profile_default("RUNMAT_RELEASE_READINESS_PROMOTION_MAX_BLOCKER_REGRESSION", "0"),
        )
    )
    require_promotion_calibration = is_true(
        os.getenv("RUNMAT_RELEASE_READINESS_REQUIRE_PROMOTION_CALIBRATION", "false")
    )
    promotion_min_rolling_reports = int(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PROMOTION_MIN_ROLLING_REPORTS",
            profile_default("RUNMAT_RELEASE_READINESS_PROMOTION_MIN_ROLLING_REPORTS", "0"),
        )
    )
    promotion_calibration_max_age_days = float(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PROMOTION_CALIBRATION_MAX_AGE_DAYS",
            profile_default(
                "RUNMAT_RELEASE_READINESS_PROMOTION_CALIBRATION_MAX_AGE_DAYS", "30"
            ),
        )
    )
    promotion_calibration_applied = False
    promotion_calibration_age_days = None
    promotion_history_sufficient = len(rolling) >= promotion_min_rolling_reports

    profile = governance_profile_name()
    if isinstance(promotion_calibration, dict):
        generated_raw = promotion_calibration.get("generated_at")
        if isinstance(generated_raw, str):
            generated_at = parse_iso8601_utc(generated_raw)
            if generated_at is not None:
                promotion_calibration_age_days = (
                    datetime.now(timezone.utc) - generated_at
                ).total_seconds() / 86400.0
        if (
            promotion_calibration_age_days is not None
            and promotion_calibration_age_days > promotion_calibration_max_age_days
            and (protected or require_promotion_calibration)
        ):
            reasons.append(
                Reason(
                    code="PROMOTION_CALIBRATION_STALE",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"promotion calibration age {promotion_calibration_age_days:.2f}d exceeds "
                        f"max {promotion_calibration_max_age_days:.2f}d"
                    ),
                )
            )
        by_profile = promotion_calibration.get("by_profile")
        if isinstance(by_profile, dict):
            profile_entry = by_profile.get(profile)
            if isinstance(profile_entry, dict):
                plastic_override = profile_entry.get("plastic_promotion_max_blockers")
                contact_override = profile_entry.get("contact_promotion_max_blockers")
                regression_override = profile_entry.get("promotion_max_blocker_regression")
                if isinstance(plastic_override, (int, float)):
                    plastic_promotion_max_blockers = int(plastic_override)
                    promotion_calibration_applied = True
                if isinstance(contact_override, (int, float)):
                    contact_promotion_max_blockers = int(contact_override)
                    promotion_calibration_applied = True
                if isinstance(regression_override, (int, float)):
                    promotion_max_blocker_regression = int(regression_override)
                    promotion_calibration_applied = True
    elif protected or require_promotion_calibration:
        reasons.append(
            Reason(
                code="PROMOTION_CALIBRATION_MISSING",
                severity="fail" if protected else "warn",
                detail="promotion calibration artifact missing or invalid",
            )
        )

    if (protected or require_promotion_ready) and not promotion_history_sufficient:
        reasons.append(
            Reason(
                code="PROMOTION_HISTORY_INSUFFICIENT",
                severity="fail" if protected else "warn",
                detail=(
                    f"rolling report count {len(rolling)} below minimum {promotion_min_rolling_reports}"
                ),
            )
        )
    thermo_records = [
        rec
        for rec in report_records(latest)
        if isinstance(rec.get("thermo_coupling_enabled"), bool)
        or isinstance(rec.get("thermo_transient_severity"), (int, float))
        or isinstance(rec.get("thermo_nonlinear_severity"), (int, float))
        or isinstance(rec.get("thermo_constitutive_material_spread_ratio"), (int, float))
        or isinstance(rec.get("thermo_assignment_heterogeneity_index"), (int, float))
    ]
    thermo_coupling_enabled_rate = None
    thermo_max_transient_severity = None
    thermo_max_nonlinear_severity = None
    thermo_max_spread_ratio = None
    thermo_max_heterogeneity_index = None
    thermo_min_field_coverage_ratio = None
    thermo_max_field_extrapolation_ratio = None
    thermo_max_field_clamp_ratio = None
    thermo_field_coverage_drop_trend_ratio = None
    thermo_field_extrapolation_trend_ratio = None
    thermo_field_clamp_trend_ratio = None
    thermo_spread_breach_rate = None
    thermo_heterogeneity_breach_rate = None
    thermo_field_clamp_breach_rate = None
    thermo_spread_trend_ratio = None
    thermo_heterogeneity_trend_ratio = None
    thermal_max_residual_norm = None
    thermal_min_temperature_k = None
    thermal_max_temperature_k = None
    thermal_max_conductivity_spread_ratio = None
    thermal_max_heat_capacity_spread_ratio = None
    thermal_max_spatial_gradient_index = None
    thermal_min_monotonic_response_fraction = None
    thermal_min_response_realization_ratio = None
    thermal_max_response_realization_ratio = None
    thermal_spread_breach_rate = None
    thermal_spread_trend_ratio = None
    electro_coupling_enabled_rate = None
    electro_max_transient_severity = None
    electro_max_nonlinear_severity = None
    electro_max_joule_heating_scale = None
    electro_max_conductivity_spread_ratio = None
    electro_min_time_scale_mean = None
    electro_max_time_scale_mean = None
    electro_joule_breach_rate = None
    electro_spread_breach_rate = None
    electro_time_scale_breach_rate = None
    electro_joule_trend_ratio = None
    electro_spread_trend_ratio = None
    electro_time_scale_trend_ratio = None
    em_max_energy_imbalance_ratio = None
    em_max_flux_divergence_proxy = None
    em_max_real_residual_norm = None
    em_max_imag_residual_norm = None
    em_min_source_realization_ratio = None
    em_min_source_region_coverage_ratio = None
    em_min_source_material_alignment_ratio = None
    em_min_dispersive_phase_attenuation_mean = None
    em_min_dispersive_phase_conductivity_attenuation_ratio = None
    em_breach_rate = None
    em_energy_imbalance_trend_ratio = None
    em_flux_divergence_trend_ratio = None
    em_dispersive_phase_attenuation_trend_ratio = None
    em_dispersive_phase_conductivity_attenuation_trend_ratio = None
    plastic_max_nonlinear_severity = None
    plastic_load_realization_ratio_min = None
    plastic_load_realization_ratio_max = None
    plastic_load_amplification_ratio_min = None
    plastic_load_amplification_ratio_max = None
    plastic_breach_rate = None
    plastic_load_realization_breach_rate = None
    plastic_load_amplification_breach_rate = None
    plastic_trend_ratio = None
    plastic_load_realization_drop_trend_ratio = None
    plastic_load_amplification_trend_ratio = None
    plastic_reference_trend_ratio = None
    plastic_promotion_ready = False
    plastic_promotion_blockers = []
    plastic_promotion_sample_count = 0
    plastic_promotion_blocker_count = 0
    plastic_promotion_blocker_regression = None
    contact_max_nonlinear_severity = None
    contact_load_realization_ratio_min = None
    contact_load_realization_ratio_max = None
    contact_load_amplification_ratio_min = None
    contact_load_amplification_ratio_max = None
    contact_breach_rate = None
    contact_load_realization_breach_rate = None
    contact_load_amplification_breach_rate = None
    contact_trend_ratio = None
    contact_load_realization_drop_trend_ratio = None
    contact_load_amplification_trend_ratio = None
    contact_reference_trend_ratio = None
    contact_promotion_ready = False
    contact_promotion_blockers = []
    contact_promotion_sample_count = 0
    contact_promotion_blocker_count = 0
    contact_promotion_blocker_regression = None
    thermal_records = [
        rec
        for rec in report_records(latest)
        if isinstance(rec.get("thermal_max_residual_norm"), (int, float))
        or isinstance(rec.get("thermal_min_temperature_k"), (int, float))
        or isinstance(rec.get("thermal_max_temperature_k"), (int, float))
        or isinstance(rec.get("thermal_conductivity_spread_ratio"), (int, float))
        or isinstance(rec.get("thermal_heat_capacity_spread_ratio"), (int, float))
        or isinstance(rec.get("thermal_spatial_gradient_index"), (int, float))
        or isinstance(rec.get("thermal_monotonic_response_fraction"), (int, float))
        or isinstance(rec.get("thermal_response_realization_ratio"), (int, float))
    ]
    if not thermal_records:
        if protected or thermal_require_metrics:
            reasons.append(
                Reason(
                    code="THERMAL_METRICS_MISSING",
                    severity="warn",
                    detail="standalone thermal posture metrics missing from report records",
                )
            )
    else:
        thermal_residual_values = []
        thermal_min_temp_values = []
        thermal_max_temp_values = []
        thermal_conductivity_spread_values = []
        thermal_heat_capacity_spread_values = []
        thermal_spatial_gradient_values = []
        thermal_monotonic_response_values = []
        thermal_response_realization_values = []
        thermal_spread_values = []
        for rec in thermal_records:
            for key, target in [
                ("thermal_max_residual_norm", thermal_residual_values),
                ("thermal_min_temperature_k", thermal_min_temp_values),
                ("thermal_max_temperature_k", thermal_max_temp_values),
                ("thermal_conductivity_spread_ratio", thermal_conductivity_spread_values),
                ("thermal_heat_capacity_spread_ratio", thermal_heat_capacity_spread_values),
                ("thermal_spatial_gradient_index", thermal_spatial_gradient_values),
                (
                    "thermal_monotonic_response_fraction",
                    thermal_monotonic_response_values,
                ),
                (
                    "thermal_response_realization_ratio",
                    thermal_response_realization_values,
                ),
            ]:
                raw = rec.get(key)
                if isinstance(raw, (int, float)):
                    value = float(raw)
                    if math.isfinite(value):
                        target.append(value)
            spread_raw = rec.get("thermal_conductivity_spread_ratio")
            spread2_raw = rec.get("thermal_heat_capacity_spread_ratio")
            spread_candidates = []
            if isinstance(spread_raw, (int, float)) and math.isfinite(float(spread_raw)):
                spread_candidates.append(float(spread_raw))
            if isinstance(spread2_raw, (int, float)) and math.isfinite(float(spread2_raw)):
                spread_candidates.append(float(spread2_raw))
            if spread_candidates:
                thermal_spread_values.append(max(spread_candidates))

        if thermal_residual_values:
            thermal_max_residual_norm = max(thermal_residual_values)
            if thermal_max_residual_norm > thermal_max_residual_norm_threshold:
                reasons.append(
                    Reason(
                        code="THERMAL_RESIDUAL_NORM_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max thermal residual norm {thermal_max_residual_norm:.3f} exceeds "
                            f"threshold {thermal_max_residual_norm_threshold:.3f}"
                        ),
                    )
                )
        if thermal_min_temp_values:
            thermal_min_temperature_k = min(thermal_min_temp_values)
            if thermal_min_temperature_k < thermal_min_temperature_k_threshold:
                reasons.append(
                    Reason(
                        code="THERMAL_MIN_TEMPERATURE_LOW",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"min thermal temperature {thermal_min_temperature_k:.3f}K below "
                            f"threshold {thermal_min_temperature_k_threshold:.3f}K"
                        ),
                    )
                )
        if thermal_max_temp_values:
            thermal_max_temperature_k = max(thermal_max_temp_values)
            if thermal_max_temperature_k > thermal_max_temperature_k_threshold:
                reasons.append(
                    Reason(
                        code="THERMAL_MAX_TEMPERATURE_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max thermal temperature {thermal_max_temperature_k:.3f}K exceeds "
                            f"threshold {thermal_max_temperature_k_threshold:.3f}K"
                        ),
                    )
                )
        if thermal_conductivity_spread_values:
            thermal_max_conductivity_spread_ratio = max(thermal_conductivity_spread_values)
            if (
                thermal_max_conductivity_spread_ratio
                > thermal_max_conductivity_spread_ratio_threshold
            ):
                reasons.append(
                    Reason(
                        code="THERMAL_CONDUCTIVITY_SPREAD_RATIO_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max thermal conductivity spread ratio {thermal_max_conductivity_spread_ratio:.3f} exceeds "
                            f"threshold {thermal_max_conductivity_spread_ratio_threshold:.3f}"
                        ),
                    )
                )
        if thermal_heat_capacity_spread_values:
            thermal_max_heat_capacity_spread_ratio = max(thermal_heat_capacity_spread_values)
            if (
                thermal_max_heat_capacity_spread_ratio
                > thermal_max_heat_capacity_spread_ratio_threshold
            ):
                reasons.append(
                    Reason(
                        code="THERMAL_HEAT_CAPACITY_SPREAD_RATIO_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max thermal heat-capacity spread ratio {thermal_max_heat_capacity_spread_ratio:.3f} exceeds "
                            f"threshold {thermal_max_heat_capacity_spread_ratio_threshold:.3f}"
                        ),
                    )
                )
        if thermal_spatial_gradient_values:
            thermal_max_spatial_gradient_index = max(thermal_spatial_gradient_values)
            if thermal_max_spatial_gradient_index > thermal_max_spatial_gradient_index_threshold:
                reasons.append(
                    Reason(
                        code="THERMAL_SPATIAL_GRADIENT_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max thermal spatial gradient index {thermal_max_spatial_gradient_index:.3f} exceeds "
                            f"threshold {thermal_max_spatial_gradient_index_threshold:.3f}"
                        ),
                    )
                )
        if thermal_monotonic_response_values:
            thermal_min_monotonic_response_fraction = min(thermal_monotonic_response_values)
            if (
                thermal_min_monotonic_response_fraction
                < thermal_min_monotonic_response_fraction_threshold
            ):
                reasons.append(
                    Reason(
                        code="THERMAL_MONOTONIC_RESPONSE_LOW",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"min thermal monotonic response fraction {thermal_min_monotonic_response_fraction:.3f} below "
                            f"threshold {thermal_min_monotonic_response_fraction_threshold:.3f}"
                        ),
                    )
                )
        if thermal_response_realization_values:
            thermal_min_response_realization_ratio = min(thermal_response_realization_values)
            thermal_max_response_realization_ratio = max(thermal_response_realization_values)
            if (
                thermal_min_response_realization_ratio
                < thermal_min_response_realization_ratio_threshold
            ):
                reasons.append(
                    Reason(
                        code="THERMAL_RESPONSE_REALIZATION_RATIO_LOW",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"min thermal response realization ratio {thermal_min_response_realization_ratio:.3f} below "
                            f"threshold {thermal_min_response_realization_ratio_threshold:.3f}"
                        ),
                    )
                )
            if (
                thermal_max_response_realization_ratio
                > thermal_max_response_realization_ratio_threshold
            ):
                reasons.append(
                    Reason(
                        code="THERMAL_RESPONSE_REALIZATION_RATIO_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max thermal response realization ratio {thermal_max_response_realization_ratio:.3f} exceeds "
                            f"threshold {thermal_max_response_realization_ratio_threshold:.3f}"
                        ),
                    )
                )
        if thermal_spread_values:
            thermal_spread_breach_rate = (
                sum(
                    1
                    for value in thermal_spread_values
                    if value
                    > max(
                        thermal_max_conductivity_spread_ratio_threshold,
                        thermal_max_heat_capacity_spread_ratio_threshold,
                    )
                )
                / len(thermal_spread_values)
            )
            if thermal_spread_breach_rate > thermal_max_spread_breach_rate_threshold:
                reasons.append(
                    Reason(
                        code="THERMAL_SPREAD_BREACH_RATE_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"thermal spread breach rate {thermal_spread_breach_rate:.3f} exceeds "
                            f"threshold {thermal_max_spread_breach_rate_threshold:.3f}"
                        ),
                    )
                )

    if not thermo_records:
        if protected or thermo_require_metrics:
            reasons.append(
                Reason(
                    code="THERMO_COUPLING_METRICS_MISSING",
                    severity="warn",
                    detail="thermo coupling posture metrics missing from report records",
                )
            )
    else:
        enabled_values = [
            rec.get("thermo_coupling_enabled")
            for rec in thermo_records
            if isinstance(rec.get("thermo_coupling_enabled"), bool)
        ]
        if enabled_values:
            thermo_coupling_enabled_rate = (
                sum(1 for value in enabled_values if value) / len(enabled_values)
            )
            if thermo_coupling_enabled_rate < thermo_min_enabled_rate:
                reasons.append(
                    Reason(
                        code="THERMO_COUPLING_ENABLED_RATE_LOW",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"thermo coupling enabled rate {thermo_coupling_enabled_rate:.3f} below "
                            f"minimum {thermo_min_enabled_rate:.3f}"
                        ),
                    )
                )
        elif protected or thermo_require_metrics:
            reasons.append(
                Reason(
                    code="THERMO_COUPLING_ENABLED_RATE_MISSING",
                    severity="warn",
                    detail="thermo coupling enabled-rate metric missing from report records",
                )
            )

        if thermo_require_artifact_backed:
            missing_artifact_ids = sorted(
                rec.get("fixture_id", "<unknown>")
                for rec in thermo_records
                if rec.get("thermo_coupling_enabled") is True
                and not isinstance(rec.get("thermo_field_artifact_id"), str)
            )
            if missing_artifact_ids:
                reasons.append(
                    Reason(
                        code="THERMO_FIELD_ARTIFACT_REQUIRED",
                        severity="fail" if protected else "warn",
                        detail=(
                            "thermo-enabled fixtures missing artifact-backed field references: "
                            + ", ".join(missing_artifact_ids)
                        ),
                    )
                )

            unapproved_artifacts = sorted(
                rec.get("fixture_id", "<unknown>")
                for rec in thermo_records
                if rec.get("thermo_coupling_enabled") is True
                and isinstance(rec.get("thermo_field_artifact_id"), str)
                and rec.get("thermo_field_artifact_approved") is not True
            )
            if unapproved_artifacts:
                reasons.append(
                    Reason(
                        code="THERMO_FIELD_ARTIFACT_UNAPPROVED",
                        severity="fail" if protected else "warn",
                        detail=(
                            "thermo field artifacts are not approved for fixtures: "
                            + ", ".join(unapproved_artifacts)
                        ),
                    )
                )

            stale_artifacts = []
            for rec in thermo_records:
                if rec.get("thermo_coupling_enabled") is not True:
                    continue
                age_days = rec.get("thermo_field_artifact_age_days")
                if isinstance(age_days, (int, float)) and float(age_days) > thermo_field_artifact_max_age_days:
                    stale_artifacts.append(
                        f"{rec.get('fixture_id', '<unknown>')}({float(age_days):.1f}d)"
                    )
            if stale_artifacts:
                reasons.append(
                    Reason(
                        code="THERMO_FIELD_ARTIFACT_STALE",
                        severity="fail" if protected else "warn",
                        detail=(
                            "thermo field artifacts exceed max age days "
                            f"{thermo_field_artifact_max_age_days:.1f}: "
                            + ", ".join(sorted(stale_artifacts))
                        ),
                    )
                )

            invalid_provenance = sorted(
                rec.get("fixture_id", "<unknown>")
                for rec in thermo_records
                if rec.get("thermo_coupling_enabled") is True
                and isinstance(rec.get("thermo_field_artifact_id"), str)
                and rec.get("thermo_field_artifact_provenance_valid") is not True
            )
            if invalid_provenance:
                reasons.append(
                    Reason(
                        code="THERMO_FIELD_ARTIFACT_PROVENANCE_INVALID",
                        severity="fail" if protected else "warn",
                        detail=(
                            "thermo field artifact provenance is invalid for fixtures: "
                            + ", ".join(invalid_provenance)
                        ),
                    )
                )

        transient_values = []
        for rec in thermo_records:
            raw_value = rec.get("thermo_transient_severity")
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
                if math.isfinite(value):
                    transient_values.append(value)
        if transient_values:
            thermo_max_transient_severity = max(transient_values)
            if thermo_max_transient_severity > thermo_max_transient_severity_threshold:
                reasons.append(
                    Reason(
                        code="THERMO_TRANSIENT_SEVERITY_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max thermo transient severity {thermo_max_transient_severity:.3f} exceeds "
                            f"threshold {thermo_max_transient_severity_threshold:.3f}"
                        ),
                    )
                )

        nonlinear_values = []
        for rec in thermo_records:
            raw_value = rec.get("thermo_nonlinear_severity")
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
                if math.isfinite(value):
                    nonlinear_values.append(value)
        if nonlinear_values:
            thermo_max_nonlinear_severity = max(nonlinear_values)
            if thermo_max_nonlinear_severity > thermo_max_nonlinear_severity_threshold:
                reasons.append(
                    Reason(
                        code="THERMO_NONLINEAR_SEVERITY_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max thermo nonlinear severity {thermo_max_nonlinear_severity:.3f} exceeds "
                            f"threshold {thermo_max_nonlinear_severity_threshold:.3f}"
                        ),
                    )
                )

        spread_values = []
        for rec in thermo_records:
            raw_value = rec.get("thermo_constitutive_material_spread_ratio")
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
                if math.isfinite(value):
                    spread_values.append(value)
        if spread_values:
            thermo_max_spread_ratio = max(spread_values)
            if thermo_max_spread_ratio > thermo_max_spread_ratio_threshold:
                reasons.append(
                    Reason(
                        code="THERMO_MATERIAL_SPREAD_RATIO_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max thermo material spread ratio {thermo_max_spread_ratio:.3f} exceeds "
                            f"threshold {thermo_max_spread_ratio_threshold:.3f}"
                        ),
                    )
                )

        heterogeneity_values = []
        for rec in thermo_records:
            raw_value = rec.get("thermo_assignment_heterogeneity_index")
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
                if math.isfinite(value):
                    heterogeneity_values.append(value)
        if heterogeneity_values:
            thermo_max_heterogeneity_index = max(heterogeneity_values)
            if (
                thermo_max_heterogeneity_index
                > thermo_max_heterogeneity_index_threshold
            ):
                reasons.append(
                    Reason(
                        code="THERMO_ASSIGNMENT_HETEROGENEITY_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max thermo assignment heterogeneity index {thermo_max_heterogeneity_index:.3f} exceeds "
                            f"threshold {thermo_max_heterogeneity_index_threshold:.3f}"
                        ),
                    )
                )

        coverage_values = []
        for rec in thermo_records:
            raw_region_count = rec.get("thermo_region_delta_count")
            if isinstance(raw_region_count, (int, float)) and float(raw_region_count) <= 0:
                continue
            raw_value = rec.get("thermo_spatial_coverage_ratio")
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
                if math.isfinite(value):
                    coverage_values.append(value)
        if coverage_values:
            thermo_min_field_coverage_ratio = min(coverage_values)
            if (
                thermo_min_field_coverage_ratio
                < thermo_min_field_coverage_ratio_threshold
            ):
                reasons.append(
                    Reason(
                        code="THERMO_FIELD_COVERAGE_LOW",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"min thermo field coverage ratio {thermo_min_field_coverage_ratio:.3f} below "
                            f"threshold {thermo_min_field_coverage_ratio_threshold:.3f}"
                        ),
                    )
                )

        extrapolation_values = []
        for rec in thermo_records:
            raw_value = rec.get("thermo_field_extrapolation_ratio")
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
                if math.isfinite(value):
                    extrapolation_values.append(value)
        if extrapolation_values:
            thermo_max_field_extrapolation_ratio = max(extrapolation_values)
            if (
                thermo_max_field_extrapolation_ratio
                > thermo_max_field_extrapolation_ratio_threshold
            ):
                reasons.append(
                    Reason(
                        code="THERMO_FIELD_EXTRAPOLATION_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max thermo field extrapolation ratio {thermo_max_field_extrapolation_ratio:.3f} exceeds "
                            f"threshold {thermo_max_field_extrapolation_ratio_threshold:.3f}"
                        ),
                    )
                )

        clamp_values = []
        for rec in thermo_records:
            raw_value = rec.get("thermo_field_clamp_ratio")
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
                if math.isfinite(value):
                    clamp_values.append(value)
        if clamp_values:
            thermo_max_field_clamp_ratio = max(clamp_values)
            if thermo_max_field_clamp_ratio > thermo_max_field_clamp_ratio_threshold:
                reasons.append(
                    Reason(
                        code="THERMO_FIELD_CLAMP_RATIO_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max thermo field clamp ratio {thermo_max_field_clamp_ratio:.3f} exceeds "
                            f"threshold {thermo_max_field_clamp_ratio_threshold:.3f}"
                        ),
                    )
                )

    electro_records = [
        rec
        for rec in report_records(latest)
        if isinstance(rec.get("electro_thermal_coupling_enabled"), bool)
        or isinstance(rec.get("electro_transient_severity"), (int, float))
        or isinstance(rec.get("electro_nonlinear_severity"), (int, float))
        or isinstance(rec.get("electro_transient_time_scale_mean"), (int, float))
        or isinstance(rec.get("electro_nonlinear_time_scale_mean"), (int, float))
    ]
    if not electro_records:
        if protected or electro_require_metrics:
            reasons.append(
                Reason(
                    code="ELECTRO_COUPLING_METRICS_MISSING",
                    severity="warn",
                    detail="electro-thermal coupling posture metrics missing from report records",
                )
            )
    else:
        electro_enabled_values = [
            rec.get("electro_thermal_coupling_enabled")
            for rec in electro_records
            if isinstance(rec.get("electro_thermal_coupling_enabled"), bool)
        ]
        if electro_enabled_values:
            electro_coupling_enabled_rate = (
                sum(1 for value in electro_enabled_values if value)
                / len(electro_enabled_values)
            )
            if electro_coupling_enabled_rate < electro_min_enabled_rate:
                reasons.append(
                    Reason(
                        code="ELECTRO_COUPLING_ENABLED_RATE_LOW",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"electro-thermal coupling enabled rate {electro_coupling_enabled_rate:.3f} below "
                            f"minimum {electro_min_enabled_rate:.3f}"
                        ),
                    )
                )
        elif protected or electro_require_metrics:
            reasons.append(
                Reason(
                    code="ELECTRO_COUPLING_ENABLED_RATE_MISSING",
                    severity="warn",
                    detail="electro-thermal coupling enabled-rate metric missing from report records",
                )
            )

        electro_transient_values = []
        for rec in electro_records:
            raw_value = rec.get("electro_transient_severity")
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
                if math.isfinite(value):
                    electro_transient_values.append(value)
        if electro_transient_values:
            electro_max_transient_severity = max(electro_transient_values)
            if electro_max_transient_severity > electro_max_transient_severity_threshold:
                reasons.append(
                    Reason(
                        code="ELECTRO_TRANSIENT_SEVERITY_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max electro-thermal transient severity {electro_max_transient_severity:.3f} exceeds "
                            f"threshold {electro_max_transient_severity_threshold:.3f}"
                        ),
                    )
                )

        electro_nonlinear_values = []
        for rec in electro_records:
            raw_value = rec.get("electro_nonlinear_severity")
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
                if math.isfinite(value):
                    electro_nonlinear_values.append(value)
        if electro_nonlinear_values:
            electro_max_nonlinear_severity = max(electro_nonlinear_values)
            if electro_max_nonlinear_severity > electro_max_nonlinear_severity_threshold:
                reasons.append(
                    Reason(
                        code="ELECTRO_NONLINEAR_SEVERITY_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max electro-thermal nonlinear severity {electro_max_nonlinear_severity:.3f} exceeds "
                            f"threshold {electro_max_nonlinear_severity_threshold:.3f}"
                        ),
                    )
                )

        electro_joule_values = []
        for rec in electro_records:
            raw_value = rec.get("electro_joule_heating_scale")
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
                if math.isfinite(value):
                    electro_joule_values.append(value)
        if electro_joule_values:
            electro_max_joule_heating_scale = max(electro_joule_values)
            if electro_max_joule_heating_scale > electro_max_joule_heating_scale_threshold:
                reasons.append(
                    Reason(
                        code="ELECTRO_JOULE_HEATING_SCALE_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max electro-thermal Joule heating scale {electro_max_joule_heating_scale:.3f} exceeds "
                            f"threshold {electro_max_joule_heating_scale_threshold:.3f}"
                        ),
                    )
                )

        electro_spread_values = []
        for rec in electro_records:
            raw_value = rec.get("electro_conductivity_spread_ratio")
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
                if math.isfinite(value):
                    electro_spread_values.append(value)
        if electro_spread_values:
            electro_max_conductivity_spread_ratio = max(electro_spread_values)
            if (
                electro_max_conductivity_spread_ratio
                > electro_max_conductivity_spread_ratio_threshold
            ):
                reasons.append(
                    Reason(
                        code="ELECTRO_CONDUCTIVITY_SPREAD_RATIO_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            "max electro-thermal conductivity spread ratio "
                            f"{electro_max_conductivity_spread_ratio:.3f} exceeds "
                            f"threshold {electro_max_conductivity_spread_ratio_threshold:.3f}"
                        ),
                    )
                )

        electro_time_scale_values = []
        for rec in electro_records:
            for key in (
                "electro_transient_time_scale_mean",
                "electro_nonlinear_time_scale_mean",
            ):
                raw_value = rec.get(key)
                if isinstance(raw_value, (int, float)):
                    value = float(raw_value)
                    if math.isfinite(value):
                        electro_time_scale_values.append(value)
        if electro_time_scale_values:
            electro_min_time_scale_mean = min(electro_time_scale_values)
            electro_max_time_scale_mean = max(electro_time_scale_values)
            if (
                electro_min_time_scale_mean < electro_min_time_scale_mean_threshold
                or electro_max_time_scale_mean > electro_max_time_scale_mean_threshold
            ):
                reasons.append(
                    Reason(
                        code="ELECTRO_TIME_SCALE_MEAN_OUT_OF_RANGE",
                        severity="fail" if protected else "warn",
                        detail=(
                            "electro-thermal time-scale mean outside threshold envelope: "
                            f"min={electro_min_time_scale_mean:.3f} max={electro_max_time_scale_mean:.3f} "
                            f"allowed=[{electro_min_time_scale_mean_threshold:.3f}, {electro_max_time_scale_mean_threshold:.3f}]"
                        ),
                    )
                )

    em_records = [
        rec
        for rec in report_records(latest)
        if (
            isinstance(rec.get("fixture_id"), str) and rec.get("fixture_id") in EM_FIXTURES
        )
        or isinstance(rec.get("electromagnetic_enabled"), bool)
        or isinstance(rec.get("electromagnetic_energy_imbalance_ratio"), (int, float))
    ]
    if not em_records:
        if protected or em_require_metrics:
            reasons.append(
                Reason(
                    code="EM_METRICS_MISSING",
                    severity="warn",
                    detail="electromagnetic posture metrics missing from report records",
                )
            )
    else:
        metric_specs = [
            (
                "electromagnetic_energy_imbalance_ratio",
                "max",
                em_max_energy_imbalance_ratio_threshold,
                "EM_ENERGY_IMBALANCE_RATIO_HIGH",
                "max EM energy imbalance ratio",
            ),
            (
                "electromagnetic_flux_divergence_proxy",
                "max",
                em_max_flux_divergence_proxy_threshold,
                "EM_FLUX_DIVERGENCE_PROXY_HIGH",
                "max EM flux divergence proxy",
            ),
            (
                "electromagnetic_real_residual_norm",
                "max",
                em_max_real_residual_norm_threshold,
                "EM_REAL_RESIDUAL_NORM_HIGH",
                "max EM real residual norm",
            ),
            (
                "electromagnetic_imag_residual_norm",
                "max",
                em_max_imag_residual_norm_threshold,
                "EM_IMAG_RESIDUAL_NORM_HIGH",
                "max EM imaginary residual norm",
            ),
            (
                "electromagnetic_source_realization_ratio",
                "min",
                em_min_source_realization_ratio_threshold,
                "EM_SOURCE_REALIZATION_RATIO_LOW",
                "min EM source realization ratio",
            ),
            (
                "electromagnetic_source_region_coverage_ratio",
                "min",
                em_min_source_region_coverage_ratio_threshold,
                "EM_SOURCE_REGION_COVERAGE_RATIO_LOW",
                "min EM source-region coverage ratio",
            ),
            (
                "electromagnetic_source_material_alignment_ratio",
                "min",
                em_min_source_material_alignment_ratio_threshold,
                "EM_SOURCE_MATERIAL_ALIGNMENT_RATIO_LOW",
                "min EM source-material alignment ratio",
            ),
        ]
        phase_assertion_specs = [
            (
                "electromagnetic_reference_homogeneous_gpu_provider",
                "em_homogeneous_dispersive_phase_attenuation_mean",
                "EM_DISPERSIVE_PHASE_ATTENUATION_MEAN_LOW",
                "EM homogeneous dispersive phase attenuation mean",
                em_min_dispersive_phase_attenuation_mean_threshold,
            ),
            (
                "electromagnetic_reference_heterogeneous_gpu_provider",
                "em_heterogeneous_dispersive_phase_attenuation_mean",
                "EM_DISPERSIVE_PHASE_ATTENUATION_MEAN_LOW",
                "EM heterogeneous dispersive phase attenuation mean",
                em_min_dispersive_phase_attenuation_mean_threshold,
            ),
            (
                "electromagnetic_reference_homogeneous_gpu_provider",
                "em_homogeneous_dispersive_phase_conductivity_attenuation_ratio",
                "EM_DISPERSIVE_PHASE_CONDUCTIVITY_ATTENUATION_RATIO_LOW",
                "EM homogeneous dispersive phase conductivity attenuation ratio",
                em_min_dispersive_phase_conductivity_attenuation_ratio_threshold,
            ),
            (
                "electromagnetic_reference_heterogeneous_gpu_provider",
                "em_heterogeneous_dispersive_phase_conductivity_attenuation_ratio",
                "EM_DISPERSIVE_PHASE_CONDUCTIVITY_ATTENUATION_RATIO_LOW",
                "EM heterogeneous dispersive phase conductivity attenuation ratio",
                em_min_dispersive_phase_conductivity_attenuation_ratio_threshold,
            ),
        ]
        missing_metric_fields = []
        breaches = []
        for field, mode, threshold, code, label in metric_specs:
            values = []
            for rec in em_records:
                raw_value = rec.get(field)
                if isinstance(raw_value, (int, float)):
                    value = float(raw_value)
                    if math.isfinite(value):
                        values.append(value)
            if not values:
                missing_metric_fields.append(field)
                continue
            observed = max(values) if mode == "max" else min(values)
            if field == "electromagnetic_energy_imbalance_ratio":
                em_max_energy_imbalance_ratio = observed
            elif field == "electromagnetic_flux_divergence_proxy":
                em_max_flux_divergence_proxy = observed
            elif field == "electromagnetic_real_residual_norm":
                em_max_real_residual_norm = observed
            elif field == "electromagnetic_imag_residual_norm":
                em_max_imag_residual_norm = observed
            elif field == "electromagnetic_source_realization_ratio":
                em_min_source_realization_ratio = observed
            elif field == "electromagnetic_source_region_coverage_ratio":
                em_min_source_region_coverage_ratio = observed
            elif field == "electromagnetic_source_material_alignment_ratio":
                em_min_source_material_alignment_ratio = observed

            breached = observed > threshold if mode == "max" else observed < threshold
            breaches.append(breached)
            if breached:
                comparator = "exceeds" if mode == "max" else "below"
                reasons.append(
                    Reason(
                        code=code,
                        severity="fail" if protected else "warn",
                        detail=(
                            f"{label} {observed:.3f} {comparator} threshold {threshold:.3f}"
                        ),
                    )
                )

        for fixture_id, assertion_name, code, label, threshold in phase_assertion_specs:
            values = []
            for rec in em_records:
                if rec.get("fixture_id") != fixture_id:
                    continue
                observed = threshold_assertion_observed(rec, assertion_name)
                if observed is not None:
                    values.append(observed)
            if not values:
                missing_metric_fields.append(f"{fixture_id}.{assertion_name}")
                continue

            observed = min(values)
            if assertion_name.endswith("dispersive_phase_attenuation_mean"):
                if (
                    em_min_dispersive_phase_attenuation_mean is None
                    or observed < em_min_dispersive_phase_attenuation_mean
                ):
                    em_min_dispersive_phase_attenuation_mean = observed
            elif (
                em_min_dispersive_phase_conductivity_attenuation_ratio is None
                or observed < em_min_dispersive_phase_conductivity_attenuation_ratio
            ):
                em_min_dispersive_phase_conductivity_attenuation_ratio = observed

            breached = observed < threshold
            breaches.append(breached)
            if breached:
                reasons.append(
                    Reason(
                        code=code,
                        severity="fail" if protected else "warn",
                        detail=(
                            f"{label} {observed:.3f} below threshold {threshold:.3f}"
                        ),
                    )
                )

        if missing_metric_fields and (protected or em_require_metrics):
            reasons.append(
                Reason(
                    code="EM_CORE_METRICS_MISSING",
                    severity="warn",
                    detail=(
                        "missing EM metric fields: " + ", ".join(sorted(missing_metric_fields))
                    ),
                )
            )
        if breaches:
            em_breach_rate = sum(1 for breached in breaches if breached) / len(breaches)
            if em_breach_rate > em_max_breach_rate_threshold:
                reasons.append(
                    Reason(
                        code="EM_BREACH_RATE_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"EM metric breach rate {em_breach_rate:.3f} exceeds threshold "
                            f"{em_max_breach_rate_threshold:.3f}"
                        ),
                    )
                )

    plastic_records = [
        rec
        for rec in report_records(latest)
        if isinstance(rec.get("plastic_nonlinear_severity"), (int, float))
    ]
    if not plastic_records:
        if protected or plastic_require_metrics:
            reasons.append(
                Reason(
                    code="PLASTIC_METRICS_MISSING",
                    severity="warn",
                    detail="plastic nonlinear posture metrics missing from report records",
                )
            )
    else:
        plastic_values = []
        for rec in plastic_records:
            raw_value = rec.get("plastic_nonlinear_severity")
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
                if math.isfinite(value):
                    plastic_values.append(value)
        if plastic_values:
            plastic_max_nonlinear_severity = max(plastic_values)
            if plastic_max_nonlinear_severity > plastic_max_nonlinear_severity_threshold:
                reasons.append(
                    Reason(
                        code="PLASTIC_NONLINEAR_SEVERITY_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max plastic nonlinear severity {plastic_max_nonlinear_severity:.3f} exceeds "
                            f"threshold {plastic_max_nonlinear_severity_threshold:.3f}"
                        ),
                    )
                )

        plastic_load_realization_values = []
        plastic_load_amplification_values = []
        for rec in plastic_records:
            raw_realization = rec.get("plastic_load_realization_ratio")
            if isinstance(raw_realization, (int, float)):
                value = float(raw_realization)
                if math.isfinite(value):
                    plastic_load_realization_values.append(value)
            raw_amplification = rec.get("plastic_load_amplification_ratio")
            if isinstance(raw_amplification, (int, float)):
                value = float(raw_amplification)
                if math.isfinite(value):
                    plastic_load_amplification_values.append(value)
        if not plastic_load_realization_values or not plastic_load_amplification_values:
            if protected or plastic_require_metrics:
                reasons.append(
                    Reason(
                        code="PLASTIC_LOAD_PATH_METRICS_MISSING",
                        severity="warn",
                        detail="plastic load-path constitutive metrics missing from report records",
                    )
                )
        else:
            plastic_load_realization_ratio_min = min(plastic_load_realization_values)
            plastic_load_realization_ratio_max = max(plastic_load_realization_values)
            plastic_load_amplification_ratio_min = min(plastic_load_amplification_values)
            plastic_load_amplification_ratio_max = max(plastic_load_amplification_values)
            if (
                plastic_load_realization_ratio_min
                < plastic_min_load_realization_ratio_threshold
                or plastic_load_realization_ratio_max
                > plastic_max_load_realization_ratio_threshold
            ):
                reasons.append(
                    Reason(
                        code="PLASTIC_LOAD_REALIZATION_RATIO_OUT_OF_RANGE",
                        severity="fail" if protected else "warn",
                        detail=(
                            "plastic load realization ratio outside threshold envelope: "
                            f"min={plastic_load_realization_ratio_min:.3f} max={plastic_load_realization_ratio_max:.3f} "
                            f"allowed=[{plastic_min_load_realization_ratio_threshold:.3f}, {plastic_max_load_realization_ratio_threshold:.3f}]"
                        ),
                    )
                )
            if (
                plastic_load_amplification_ratio_min
                < plastic_min_load_amplification_ratio_threshold
                or plastic_load_amplification_ratio_max
                > plastic_max_load_amplification_ratio_threshold
            ):
                reasons.append(
                    Reason(
                        code="PLASTIC_LOAD_AMPLIFICATION_RATIO_OUT_OF_RANGE",
                        severity="fail" if protected else "warn",
                        detail=(
                            "plastic load amplification ratio outside threshold envelope: "
                            f"min={plastic_load_amplification_ratio_min:.3f} max={plastic_load_amplification_ratio_max:.3f} "
                            f"allowed=[{plastic_min_load_amplification_ratio_threshold:.3f}, {plastic_max_load_amplification_ratio_threshold:.3f}]"
                        ),
                    )
                )

    contact_records = [
        rec
        for rec in report_records(latest)
        if isinstance(rec.get("contact_nonlinear_severity"), (int, float))
    ]
    if not contact_records:
        if protected or contact_require_metrics:
            reasons.append(
                Reason(
                    code="CONTACT_METRICS_MISSING",
                    severity="warn",
                    detail="contact nonlinear posture metrics missing from report records",
                )
            )
    else:
        contact_values = []
        for rec in contact_records:
            raw_value = rec.get("contact_nonlinear_severity")
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
                if math.isfinite(value):
                    contact_values.append(value)
        if contact_values:
            contact_max_nonlinear_severity = max(contact_values)
            if contact_max_nonlinear_severity > contact_max_nonlinear_severity_threshold:
                reasons.append(
                    Reason(
                        code="CONTACT_NONLINEAR_SEVERITY_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max contact nonlinear severity {contact_max_nonlinear_severity:.3f} exceeds "
                            f"threshold {contact_max_nonlinear_severity_threshold:.3f}"
                        ),
                    )
                )

        contact_load_realization_values = []
        contact_load_amplification_values = []
        for rec in contact_records:
            raw_realization = rec.get("contact_load_realization_ratio")
            if isinstance(raw_realization, (int, float)):
                value = float(raw_realization)
                if math.isfinite(value):
                    contact_load_realization_values.append(value)
            raw_amplification = rec.get("contact_load_amplification_ratio")
            if isinstance(raw_amplification, (int, float)):
                value = float(raw_amplification)
                if math.isfinite(value):
                    contact_load_amplification_values.append(value)
        if not contact_load_realization_values or not contact_load_amplification_values:
            if protected or contact_require_metrics:
                reasons.append(
                    Reason(
                        code="CONTACT_LOAD_PATH_METRICS_MISSING",
                        severity="warn",
                        detail="contact load-path constitutive metrics missing from report records",
                    )
                )
        else:
            contact_load_realization_ratio_min = min(contact_load_realization_values)
            contact_load_realization_ratio_max = max(contact_load_realization_values)
            contact_load_amplification_ratio_min = min(contact_load_amplification_values)
            contact_load_amplification_ratio_max = max(contact_load_amplification_values)
            if (
                contact_load_realization_ratio_min
                < contact_min_load_realization_ratio_threshold
                or contact_load_realization_ratio_max
                > contact_max_load_realization_ratio_threshold
            ):
                reasons.append(
                    Reason(
                        code="CONTACT_LOAD_REALIZATION_RATIO_OUT_OF_RANGE",
                        severity="fail" if protected else "warn",
                        detail=(
                            "contact load realization ratio outside threshold envelope: "
                            f"min={contact_load_realization_ratio_min:.3f} max={contact_load_realization_ratio_max:.3f} "
                            f"allowed=[{contact_min_load_realization_ratio_threshold:.3f}, {contact_max_load_realization_ratio_threshold:.3f}]"
                        ),
                    )
                )
            if (
                contact_load_amplification_ratio_min
                < contact_min_load_amplification_ratio_threshold
                or contact_load_amplification_ratio_max
                > contact_max_load_amplification_ratio_threshold
            ):
                reasons.append(
                    Reason(
                        code="CONTACT_LOAD_AMPLIFICATION_RATIO_OUT_OF_RANGE",
                        severity="fail" if protected else "warn",
                        detail=(
                            "contact load amplification ratio outside threshold envelope: "
                            f"min={contact_load_amplification_ratio_min:.3f} max={contact_load_amplification_ratio_max:.3f} "
                            f"allowed=[{contact_min_load_amplification_ratio_threshold:.3f}, {contact_max_load_amplification_ratio_threshold:.3f}]"
                        ),
                    )
                )

    trend_reports = [latest] + rolling
    trend_thermo_records = []
    for report in trend_reports:
        for rec in report_records(report):
            if not isinstance(rec, dict):
                continue
            if isinstance(rec.get("thermo_constitutive_material_spread_ratio"), (int, float)) or isinstance(
                rec.get("thermo_assignment_heterogeneity_index"), (int, float)
            ):
                trend_thermo_records.append(rec)

    spread_breach_values = []
    heterogeneity_breach_values = []
    clamp_breach_values = []
    for rec in trend_thermo_records:
        spread = rec.get("thermo_constitutive_material_spread_ratio")
        if isinstance(spread, (int, float)):
            value = float(spread)
            if math.isfinite(value):
                spread_breach_values.append(value > thermo_max_spread_ratio_threshold)
        heterogeneity = rec.get("thermo_assignment_heterogeneity_index")
        if isinstance(heterogeneity, (int, float)):
            value = float(heterogeneity)
            if math.isfinite(value):
                heterogeneity_breach_values.append(
                    value > thermo_max_heterogeneity_index_threshold
                )
        clamp_ratio = rec.get("thermo_field_clamp_ratio")
        if isinstance(clamp_ratio, (int, float)):
            value = float(clamp_ratio)
            if math.isfinite(value):
                clamp_breach_values.append(value > thermo_max_field_clamp_ratio_threshold)

    if spread_breach_values:
        thermo_spread_breach_rate = (
            sum(1 for breached in spread_breach_values if breached)
            / len(spread_breach_values)
        )
        if thermo_spread_breach_rate > thermo_max_spread_breach_rate_threshold:
            reasons.append(
                Reason(
                    code="THERMO_SPREAD_BREACH_RATE_HIGH",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"thermo spread breach rate {thermo_spread_breach_rate:.3f} exceeds "
                        f"threshold {thermo_max_spread_breach_rate_threshold:.3f}"
                    ),
                )
            )

    if heterogeneity_breach_values:
        thermo_heterogeneity_breach_rate = (
            sum(1 for breached in heterogeneity_breach_values if breached)
            / len(heterogeneity_breach_values)
        )
        if (
            thermo_heterogeneity_breach_rate
            > thermo_max_heterogeneity_breach_rate_threshold
        ):
            reasons.append(
                Reason(
                    code="THERMO_HETEROGENEITY_BREACH_RATE_HIGH",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"thermo heterogeneity breach rate {thermo_heterogeneity_breach_rate:.3f} exceeds "
                        f"threshold {thermo_max_heterogeneity_breach_rate_threshold:.3f}"
                    ),
                    )
                )

    if clamp_breach_values:
        thermo_field_clamp_breach_rate = (
            sum(1 for breached in clamp_breach_values if breached)
            / len(clamp_breach_values)
        )
        if thermo_field_clamp_breach_rate > thermo_max_field_clamp_breach_rate_threshold:
            reasons.append(
                Reason(
                    code="THERMO_FIELD_CLAMP_BREACH_RATE_HIGH",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"thermo field clamp breach rate {thermo_field_clamp_breach_rate:.3f} exceeds "
                        f"threshold {thermo_max_field_clamp_breach_rate_threshold:.3f}"
                    ),
                )
            )

    trend_electro_records = []
    for report in trend_reports:
        for rec in report_records(report):
            if not isinstance(rec, dict):
                continue
            if isinstance(rec.get("electro_joule_heating_scale"), (int, float)) or isinstance(
                rec.get("electro_conductivity_spread_ratio"), (int, float)
            ) or isinstance(rec.get("electro_transient_time_scale_mean"), (int, float)) or isinstance(
                rec.get("electro_nonlinear_time_scale_mean"), (int, float)
            ):
                trend_electro_records.append(rec)

    electro_joule_breach_values = []
    electro_spread_breach_values = []
    electro_time_scale_breach_values = []
    for rec in trend_electro_records:
        joule = rec.get("electro_joule_heating_scale")
        if isinstance(joule, (int, float)):
            value = float(joule)
            if math.isfinite(value):
                electro_joule_breach_values.append(
                    value > electro_max_joule_heating_scale_threshold
                )
        spread = rec.get("electro_conductivity_spread_ratio")
        if isinstance(spread, (int, float)):
            value = float(spread)
            if math.isfinite(value):
                electro_spread_breach_values.append(
                    value > electro_max_conductivity_spread_ratio_threshold
                )
        for key in (
            "electro_transient_time_scale_mean",
            "electro_nonlinear_time_scale_mean",
        ):
            raw_time_scale = rec.get(key)
            if isinstance(raw_time_scale, (int, float)):
                value = float(raw_time_scale)
                if math.isfinite(value):
                    electro_time_scale_breach_values.append(
                        value < electro_min_time_scale_mean_threshold
                        or value > electro_max_time_scale_mean_threshold
                    )

    if electro_joule_breach_values:
        electro_joule_breach_rate = (
            sum(1 for breached in electro_joule_breach_values if breached)
            / len(electro_joule_breach_values)
        )
        if electro_joule_breach_rate > electro_max_joule_breach_rate_threshold:
            reasons.append(
                Reason(
                    code="ELECTRO_JOULE_BREACH_RATE_HIGH",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"electro-thermal Joule breach rate {electro_joule_breach_rate:.3f} exceeds "
                        f"threshold {electro_max_joule_breach_rate_threshold:.3f}"
                    ),
                )
            )

    if electro_spread_breach_values:
        electro_spread_breach_rate = (
            sum(1 for breached in electro_spread_breach_values if breached)
            / len(electro_spread_breach_values)
        )
        if electro_spread_breach_rate > electro_max_spread_breach_rate_threshold:
            reasons.append(
                Reason(
                    code="ELECTRO_SPREAD_BREACH_RATE_HIGH",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"electro-thermal spread breach rate {electro_spread_breach_rate:.3f} exceeds "
                        f"threshold {electro_max_spread_breach_rate_threshold:.3f}"
                    ),
                )
            )

    plastic_breach_values = []
    for report in trend_reports:
        for rec in report_records(report):
            raw_value = rec.get("plastic_nonlinear_severity")
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
                if math.isfinite(value):
                    plastic_breach_values.append(
                        value > plastic_max_nonlinear_severity_threshold
                    )

    if plastic_breach_values:
        plastic_breach_rate = (
            sum(1 for breached in plastic_breach_values if breached)
            / len(plastic_breach_values)
        )
        if plastic_breach_rate > plastic_max_breach_rate_threshold:
            reasons.append(
                Reason(
                    code="PLASTIC_BREACH_RATE_HIGH",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"plastic nonlinear severity breach rate {plastic_breach_rate:.3f} exceeds "
                        f"threshold {plastic_max_breach_rate_threshold:.3f}"
                    ),
                )
            )

    plastic_load_realization_breach_values = []
    plastic_load_amplification_breach_values = []
    for report in trend_reports:
        for rec in report_records(report):
            raw_realization = rec.get("plastic_load_realization_ratio")
            if isinstance(raw_realization, (int, float)):
                value = float(raw_realization)
                if math.isfinite(value):
                    plastic_load_realization_breach_values.append(
                        value < plastic_min_load_realization_ratio_threshold
                        or value > plastic_max_load_realization_ratio_threshold
                    )
            raw_amplification = rec.get("plastic_load_amplification_ratio")
            if isinstance(raw_amplification, (int, float)):
                value = float(raw_amplification)
                if math.isfinite(value):
                    plastic_load_amplification_breach_values.append(
                        value < plastic_min_load_amplification_ratio_threshold
                        or value > plastic_max_load_amplification_ratio_threshold
                    )

    if plastic_load_realization_breach_values:
        plastic_load_realization_breach_rate = (
            sum(1 for breached in plastic_load_realization_breach_values if breached)
            / len(plastic_load_realization_breach_values)
        )
        if (
            plastic_load_realization_breach_rate
            > plastic_max_load_realization_breach_rate_threshold
        ):
            reasons.append(
                Reason(
                    code="PLASTIC_LOAD_REALIZATION_BREACH_RATE_HIGH",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"plastic load realization breach rate {plastic_load_realization_breach_rate:.3f} exceeds "
                        f"threshold {plastic_max_load_realization_breach_rate_threshold:.3f}"
                    ),
                )
            )

    if plastic_load_amplification_breach_values:
        plastic_load_amplification_breach_rate = (
            sum(1 for breached in plastic_load_amplification_breach_values if breached)
            / len(plastic_load_amplification_breach_values)
        )
        if (
            plastic_load_amplification_breach_rate
            > plastic_max_load_amplification_breach_rate_threshold
        ):
            reasons.append(
                Reason(
                    code="PLASTIC_LOAD_AMPLIFICATION_BREACH_RATE_HIGH",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"plastic load amplification breach rate {plastic_load_amplification_breach_rate:.3f} exceeds "
                        f"threshold {plastic_max_load_amplification_breach_rate_threshold:.3f}"
                    ),
                )
            )

    contact_breach_values = []
    for report in trend_reports:
        for rec in report_records(report):
            raw_value = rec.get("contact_nonlinear_severity")
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
                if math.isfinite(value):
                    contact_breach_values.append(
                        value > contact_max_nonlinear_severity_threshold
                    )

    if contact_breach_values:
        contact_breach_rate = (
            sum(1 for breached in contact_breach_values if breached)
            / len(contact_breach_values)
        )
        if contact_breach_rate > contact_max_breach_rate_threshold:
            reasons.append(
                Reason(
                    code="CONTACT_BREACH_RATE_HIGH",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"contact nonlinear severity breach rate {contact_breach_rate:.3f} exceeds "
                        f"threshold {contact_max_breach_rate_threshold:.3f}"
                    ),
                )
            )

    contact_load_realization_breach_values = []
    contact_load_amplification_breach_values = []
    for report in trend_reports:
        for rec in report_records(report):
            raw_realization = rec.get("contact_load_realization_ratio")
            if isinstance(raw_realization, (int, float)):
                value = float(raw_realization)
                if math.isfinite(value):
                    contact_load_realization_breach_values.append(
                        value < contact_min_load_realization_ratio_threshold
                        or value > contact_max_load_realization_ratio_threshold
                    )
            raw_amplification = rec.get("contact_load_amplification_ratio")
            if isinstance(raw_amplification, (int, float)):
                value = float(raw_amplification)
                if math.isfinite(value):
                    contact_load_amplification_breach_values.append(
                        value < contact_min_load_amplification_ratio_threshold
                        or value > contact_max_load_amplification_ratio_threshold
                    )

    if contact_load_realization_breach_values:
        contact_load_realization_breach_rate = (
            sum(1 for breached in contact_load_realization_breach_values if breached)
            / len(contact_load_realization_breach_values)
        )
        if (
            contact_load_realization_breach_rate
            > contact_max_load_realization_breach_rate_threshold
        ):
            reasons.append(
                Reason(
                    code="CONTACT_LOAD_REALIZATION_BREACH_RATE_HIGH",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"contact load realization breach rate {contact_load_realization_breach_rate:.3f} exceeds "
                        f"threshold {contact_max_load_realization_breach_rate_threshold:.3f}"
                    ),
                )
            )

    if contact_load_amplification_breach_values:
        contact_load_amplification_breach_rate = (
            sum(1 for breached in contact_load_amplification_breach_values if breached)
            / len(contact_load_amplification_breach_values)
        )
        if (
            contact_load_amplification_breach_rate
            > contact_max_load_amplification_breach_rate_threshold
        ):
            reasons.append(
                Reason(
                    code="CONTACT_LOAD_AMPLIFICATION_BREACH_RATE_HIGH",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"contact load amplification breach rate {contact_load_amplification_breach_rate:.3f} exceeds "
                        f"threshold {contact_max_load_amplification_breach_rate_threshold:.3f}"
                    ),
                )
            )

    if rolling:
        latest_by_fixture = {}
        for rec in report_records(latest):
            fixture_id = rec.get("fixture_id")
            if isinstance(fixture_id, str):
                latest_by_fixture[fixture_id] = rec
        rolling_by_fixture = {}
        for report in rolling:
            for rec in report_records(report):
                fixture_id = rec.get("fixture_id")
                if isinstance(fixture_id, str):
                    rolling_by_fixture.setdefault(fixture_id, []).append(rec)

        def fixture_trend_ratio(field: str, latest_reducer=max, ratio_mode="increase"):
            ratios = []
            for fixture_id, latest_rec in latest_by_fixture.items():
                baseline_records = rolling_by_fixture.get(fixture_id, [])
                if not baseline_records:
                    continue
                latest_raw = latest_rec.get(field)
                if not isinstance(latest_raw, (int, float)):
                    continue
                latest_value = float(latest_raw)
                if not math.isfinite(latest_value):
                    continue
                baseline_values = []
                for rec in baseline_records:
                    raw = rec.get(field)
                    if isinstance(raw, (int, float)):
                        value = float(raw)
                        if math.isfinite(value):
                            baseline_values.append(value)
                if not baseline_values:
                    continue
                baseline_value = statistics.median(baseline_values)
                if baseline_value <= 0 or latest_value <= 0:
                    continue
                if ratio_mode == "drop":
                    ratios.append(baseline_value / latest_value)
                elif ratio_mode == "symmetric":
                    ratios.append(max(latest_value / baseline_value, baseline_value / latest_value))
                else:
                    ratios.append(latest_value / baseline_value)
            if not ratios:
                return None
            return max(ratios)

        def fixture_assertion_trend_ratio(assertion_name: str):
            ratios = []
            for fixture_id, latest_rec in latest_by_fixture.items():
                baseline_records = rolling_by_fixture.get(fixture_id, [])
                if not baseline_records:
                    continue
                latest_value = threshold_assertion_observed(latest_rec, assertion_name)
                if latest_value is None:
                    continue
                baseline_values = []
                for rec in baseline_records:
                    observed = threshold_assertion_observed(rec, assertion_name)
                    if observed is not None:
                        baseline_values.append(observed)
                if not baseline_values:
                    continue
                baseline_value = statistics.median(baseline_values)
                if baseline_value <= 0 or latest_value <= 0:
                    continue
                ratios.append(baseline_value / latest_value)
            if not ratios:
                return None
            return max(ratios)

        thermo_spread_trend_ratio = fixture_trend_ratio(
            "thermo_constitutive_material_spread_ratio"
        )
        if (
            thermo_spread_trend_ratio is not None
            and thermo_spread_trend_ratio > thermo_max_spread_trend_ratio
        ):
            reasons.append(
                Reason(
                    code="THERMO_SPREAD_TREND_WORSENING",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"thermo spread trend ratio {thermo_spread_trend_ratio:.3f} exceeds "
                        f"threshold {thermo_max_spread_trend_ratio:.3f}"
                    ),
                )
            )

        thermo_heterogeneity_trend_ratio = fixture_trend_ratio(
            "thermo_assignment_heterogeneity_index"
        )
        if (
            thermo_heterogeneity_trend_ratio is not None
            and thermo_heterogeneity_trend_ratio > thermo_max_heterogeneity_trend_ratio
        ):
            reasons.append(
                Reason(
                    code="THERMO_HETEROGENEITY_TREND_WORSENING",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"thermo heterogeneity trend ratio {thermo_heterogeneity_trend_ratio:.3f} exceeds "
                        f"threshold {thermo_max_heterogeneity_trend_ratio:.3f}"
                    ),
                )
            )

        latest_thermal_spread_values = []
        for rec in report_records(latest):
            spread_candidates = []
            raw_cond = rec.get("thermal_conductivity_spread_ratio")
            raw_cp = rec.get("thermal_heat_capacity_spread_ratio")
            if isinstance(raw_cond, (int, float)) and math.isfinite(float(raw_cond)):
                spread_candidates.append(float(raw_cond))
            if isinstance(raw_cp, (int, float)) and math.isfinite(float(raw_cp)):
                spread_candidates.append(float(raw_cp))
            if spread_candidates:
                latest_thermal_spread_values.append(max(spread_candidates))
        rolling_thermal_spread_values = []
        for report in rolling:
            for rec in report_records(report):
                spread_candidates = []
                raw_cond = rec.get("thermal_conductivity_spread_ratio")
                raw_cp = rec.get("thermal_heat_capacity_spread_ratio")
                if isinstance(raw_cond, (int, float)) and math.isfinite(float(raw_cond)):
                    spread_candidates.append(float(raw_cond))
                if isinstance(raw_cp, (int, float)) and math.isfinite(float(raw_cp)):
                    spread_candidates.append(float(raw_cp))
                if spread_candidates:
                    rolling_thermal_spread_values.append(max(spread_candidates))
        if latest_thermal_spread_values and rolling_thermal_spread_values:
            latest_thermal_spread = max(latest_thermal_spread_values)
            baseline_thermal_spread = statistics.median(rolling_thermal_spread_values)
            if baseline_thermal_spread > 0:
                thermal_spread_trend_ratio = latest_thermal_spread / baseline_thermal_spread
                if thermal_spread_trend_ratio > thermal_max_spread_trend_ratio_threshold:
                    reasons.append(
                        Reason(
                            code="THERMAL_SPREAD_TREND_WORSENING",
                            severity="fail" if protected else "warn",
                            detail=(
                                f"thermal spread trend ratio {thermal_spread_trend_ratio:.3f} exceeds "
                                f"threshold {thermal_max_spread_trend_ratio_threshold:.3f}"
                            ),
                        )
                    )

        coverage_drop_ratios = []
        extrapolation_trend_ratios = []
        for fixture_id, latest_rec in latest_by_fixture.items():
            baseline_records = rolling_by_fixture.get(fixture_id, [])
            if not baseline_records:
                continue
            latest_coverage = latest_rec.get("thermo_spatial_coverage_ratio")
            baseline_coverage_values = [
                float(rec.get("thermo_spatial_coverage_ratio"))
                for rec in baseline_records
                if isinstance(rec.get("thermo_spatial_coverage_ratio"), (int, float))
            ]
            if (
                isinstance(latest_coverage, (int, float))
                and baseline_coverage_values
                and float(latest_coverage) > 0
            ):
                coverage_drop_ratios.append(
                    statistics.median(baseline_coverage_values) / float(latest_coverage)
                )

            latest_extrapolation = latest_rec.get("thermo_field_extrapolation_ratio")
            baseline_extrapolation_values = [
                float(rec.get("thermo_field_extrapolation_ratio"))
                for rec in baseline_records
                if isinstance(rec.get("thermo_field_extrapolation_ratio"), (int, float))
            ]
            if (
                isinstance(latest_extrapolation, (int, float))
                and baseline_extrapolation_values
                and statistics.median(baseline_extrapolation_values) > 0
            ):
                extrapolation_trend_ratios.append(
                    float(latest_extrapolation)
                    / statistics.median(baseline_extrapolation_values)
                )

        if coverage_drop_ratios:
            thermo_field_coverage_drop_trend_ratio = max(coverage_drop_ratios)
            if (
                thermo_field_coverage_drop_trend_ratio
                > thermo_max_field_coverage_drop_trend_ratio
            ):
                reasons.append(
                    Reason(
                        code="THERMO_FIELD_COVERAGE_TREND_WORSENING",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"thermo field coverage drop trend ratio {thermo_field_coverage_drop_trend_ratio:.3f} exceeds "
                            f"threshold {thermo_max_field_coverage_drop_trend_ratio:.3f}"
                        ),
                    )
                )

        if extrapolation_trend_ratios:
            thermo_field_extrapolation_trend_ratio = max(extrapolation_trend_ratios)
            if (
                thermo_field_extrapolation_trend_ratio
                > thermo_max_field_extrapolation_trend_ratio
            ):
                reasons.append(
                    Reason(
                        code="THERMO_FIELD_EXTRAPOLATION_TREND_WORSENING",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"thermo field extrapolation trend ratio {thermo_field_extrapolation_trend_ratio:.3f} exceeds "
                            f"threshold {thermo_max_field_extrapolation_trend_ratio:.3f}"
                        ),
                    )
                )

        thermo_field_clamp_trend_ratio = fixture_trend_ratio("thermo_field_clamp_ratio")
        if (
            thermo_field_clamp_trend_ratio is not None
            and thermo_field_clamp_trend_ratio > thermo_max_field_clamp_trend_ratio
        ):
            reasons.append(
                Reason(
                    code="THERMO_FIELD_CLAMP_TREND_WORSENING",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"thermo field clamp trend ratio {thermo_field_clamp_trend_ratio:.3f} exceeds "
                        f"threshold {thermo_max_field_clamp_trend_ratio:.3f}"
                    ),
                )
            )

        electro_joule_trend_ratio = fixture_trend_ratio("electro_joule_heating_scale")
        if (
            electro_joule_trend_ratio is not None
            and electro_joule_trend_ratio > electro_max_joule_trend_ratio_threshold
        ):
            reasons.append(
                Reason(
                    code="ELECTRO_JOULE_TREND_WORSENING",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"electro-thermal Joule trend ratio {electro_joule_trend_ratio:.3f} exceeds "
                        f"threshold {electro_max_joule_trend_ratio_threshold:.3f}"
                    ),
                )
            )

        electro_spread_trend_ratio = fixture_trend_ratio("electro_conductivity_spread_ratio")
        if (
            electro_spread_trend_ratio is not None
            and electro_spread_trend_ratio > electro_max_spread_trend_ratio_threshold
        ):
            reasons.append(
                Reason(
                    code="ELECTRO_SPREAD_TREND_WORSENING",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"electro-thermal spread trend ratio {electro_spread_trend_ratio:.3f} exceeds "
                        f"threshold {electro_max_spread_trend_ratio_threshold:.3f}"
                    ),
                )
            )

        electro_time_scale_trend_candidates = [
            fixture_trend_ratio("electro_transient_time_scale_mean", ratio_mode="symmetric"),
            fixture_trend_ratio("electro_nonlinear_time_scale_mean", ratio_mode="symmetric"),
        ]
        electro_time_scale_trend_candidates = [
            value for value in electro_time_scale_trend_candidates if value is not None
        ]
        if electro_time_scale_trend_candidates:
            electro_time_scale_trend_ratio = max(electro_time_scale_trend_candidates)
            if electro_time_scale_trend_ratio > electro_max_time_scale_trend_ratio_threshold:
                reasons.append(
                    Reason(
                        code="ELECTRO_TIME_SCALE_TREND_WORSENING",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"electro-thermal time-scale trend ratio {electro_time_scale_trend_ratio:.3f} exceeds "
                            f"threshold {electro_max_time_scale_trend_ratio_threshold:.3f}"
                        ),
                    )
                )

        em_energy_imbalance_trend_ratio = fixture_trend_ratio(
            "electromagnetic_energy_imbalance_ratio"
        )
        if (
            em_energy_imbalance_trend_ratio is not None
            and em_energy_imbalance_trend_ratio
            > em_max_energy_imbalance_trend_ratio_threshold
        ):
            reasons.append(
                Reason(
                    code="EM_ENERGY_IMBALANCE_TREND_WORSENING",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"EM energy-imbalance trend ratio {em_energy_imbalance_trend_ratio:.3f} exceeds "
                        f"threshold {em_max_energy_imbalance_trend_ratio_threshold:.3f}"
                    ),
                )
            )

        em_flux_divergence_trend_ratio = fixture_trend_ratio(
            "electromagnetic_flux_divergence_proxy"
        )
        if (
            em_flux_divergence_trend_ratio is not None
            and em_flux_divergence_trend_ratio > em_max_flux_divergence_trend_ratio_threshold
        ):
            reasons.append(
                Reason(
                    code="EM_FLUX_DIVERGENCE_TREND_WORSENING",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"EM flux-divergence trend ratio {em_flux_divergence_trend_ratio:.3f} exceeds "
                        f"threshold {em_max_flux_divergence_trend_ratio_threshold:.3f}"
                    ),
                )
            )

        phase_attenuation_trend_candidates = [
            fixture_assertion_trend_ratio(
                "em_homogeneous_dispersive_phase_attenuation_mean"
            ),
            fixture_assertion_trend_ratio(
                "em_heterogeneous_dispersive_phase_attenuation_mean"
            ),
        ]
        phase_attenuation_trend_candidates = [
            value for value in phase_attenuation_trend_candidates if value is not None
        ]
        if phase_attenuation_trend_candidates:
            em_dispersive_phase_attenuation_trend_ratio = max(
                phase_attenuation_trend_candidates
            )
            if (
                em_dispersive_phase_attenuation_trend_ratio
                > em_max_dispersive_phase_attenuation_trend_ratio_threshold
            ):
                reasons.append(
                    Reason(
                        code="EM_DISPERSIVE_PHASE_ATTENUATION_TREND_WORSENING",
                        severity="fail" if protected else "warn",
                        detail=(
                            "EM dispersive phase attenuation trend ratio "
                            f"{em_dispersive_phase_attenuation_trend_ratio:.3f} exceeds "
                            f"threshold {em_max_dispersive_phase_attenuation_trend_ratio_threshold:.3f}"
                        ),
                    )
                )

        phase_conductivity_trend_candidates = [
            fixture_assertion_trend_ratio(
                "em_homogeneous_dispersive_phase_conductivity_attenuation_ratio"
            ),
            fixture_assertion_trend_ratio(
                "em_heterogeneous_dispersive_phase_conductivity_attenuation_ratio"
            ),
        ]
        phase_conductivity_trend_candidates = [
            value for value in phase_conductivity_trend_candidates if value is not None
        ]
        if phase_conductivity_trend_candidates:
            em_dispersive_phase_conductivity_attenuation_trend_ratio = max(
                phase_conductivity_trend_candidates
            )
            if (
                em_dispersive_phase_conductivity_attenuation_trend_ratio
                > em_max_dispersive_phase_conductivity_attenuation_trend_ratio_threshold
            ):
                reasons.append(
                    Reason(
                        code="EM_DISPERSIVE_PHASE_CONDUCTIVITY_ATTENUATION_TREND_WORSENING",
                        severity="fail" if protected else "warn",
                        detail=(
                            "EM dispersive phase conductivity attenuation trend ratio "
                            f"{em_dispersive_phase_conductivity_attenuation_trend_ratio:.3f} exceeds "
                            "threshold "
                            f"{em_max_dispersive_phase_conductivity_attenuation_trend_ratio_threshold:.3f}"
                        ),
                    )
                )

        plastic_trend_ratio = fixture_trend_ratio("plastic_nonlinear_severity")
        if (
            plastic_trend_ratio is not None
            and plastic_trend_ratio > plastic_max_trend_ratio_threshold
        ):
            reasons.append(
                Reason(
                    code="PLASTIC_TREND_WORSENING",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"plastic nonlinear severity trend ratio {plastic_trend_ratio:.3f} exceeds "
                        f"threshold {plastic_max_trend_ratio_threshold:.3f}"
                    ),
                )
            )

        plastic_load_realization_drop_trend_ratio = fixture_trend_ratio(
            "plastic_load_realization_ratio", ratio_mode="drop"
        )
        if (
            plastic_load_realization_drop_trend_ratio is not None
            and plastic_load_realization_drop_trend_ratio
            > plastic_max_load_realization_drop_trend_ratio_threshold
        ):
            reasons.append(
                Reason(
                    code="PLASTIC_LOAD_REALIZATION_DROP_TREND_WORSENING",
                    severity="fail" if protected else "warn",
                    detail=(
                        "plastic load realization drop trend ratio "
                        f"{plastic_load_realization_drop_trend_ratio:.3f} exceeds "
                        f"threshold {plastic_max_load_realization_drop_trend_ratio_threshold:.3f}"
                    ),
                )
            )

        plastic_load_amplification_trend_ratio = fixture_trend_ratio(
            "plastic_load_amplification_ratio"
        )
        if (
            plastic_load_amplification_trend_ratio is not None
            and plastic_load_amplification_trend_ratio
            > plastic_max_load_amplification_trend_ratio_threshold
        ):
            reasons.append(
                Reason(
                    code="PLASTIC_LOAD_AMPLIFICATION_TREND_WORSENING",
                    severity="fail" if protected else "warn",
                    detail=(
                        "plastic load amplification trend ratio "
                        f"{plastic_load_amplification_trend_ratio:.3f} exceeds "
                        f"threshold {plastic_max_load_amplification_trend_ratio_threshold:.3f}"
                    ),
                )
            )

        plastic_reference_fixture_ids = {
            "nonlinear_plastic_hardening_reference_gpu_provider",
            "nonlinear_plastic_hardening_reference_complex_gpu_provider",
        }
        latest_plastic_reference_values = []
        for rec in report_records(latest):
            if rec.get("fixture_id") not in plastic_reference_fixture_ids:
                continue
            raw_value = rec.get("plastic_nonlinear_severity")
            if isinstance(raw_value, (int, float)):
                latest_plastic_reference_values.append(float(raw_value))
        rolling_plastic_reference_values = []
        for report in rolling:
            for rec in report_records(report):
                if rec.get("fixture_id") not in plastic_reference_fixture_ids:
                    continue
                raw_value = rec.get("plastic_nonlinear_severity")
                if isinstance(raw_value, (int, float)):
                    rolling_plastic_reference_values.append(float(raw_value))
        if latest_plastic_reference_values and rolling_plastic_reference_values:
            latest_reference_plastic = statistics.median(latest_plastic_reference_values)
            baseline_reference_plastic = statistics.median(rolling_plastic_reference_values)
            if baseline_reference_plastic > 0:
                plastic_reference_trend_ratio = (
                    latest_reference_plastic / baseline_reference_plastic
                )
                if (
                    plastic_reference_trend_ratio
                    > plastic_reference_max_trend_ratio_threshold
                ):
                    reasons.append(
                        Reason(
                            code="PLASTIC_REFERENCE_TREND_WORSENING",
                            severity="fail" if protected else "warn",
                            detail=(
                                f"plastic reference trend ratio {plastic_reference_trend_ratio:.3f} exceeds "
                                f"threshold {plastic_reference_max_trend_ratio_threshold:.3f}"
                            ),
                        )
                    )

        contact_trend_ratio = fixture_trend_ratio("contact_nonlinear_severity")
        if (
            contact_trend_ratio is not None
            and contact_trend_ratio > contact_max_trend_ratio_threshold
        ):
            reasons.append(
                Reason(
                    code="CONTACT_TREND_WORSENING",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"contact nonlinear severity trend ratio {contact_trend_ratio:.3f} exceeds "
                        f"threshold {contact_max_trend_ratio_threshold:.3f}"
                    ),
                )
            )

        contact_load_realization_drop_trend_ratio = fixture_trend_ratio(
            "contact_load_realization_ratio", ratio_mode="drop"
        )
        if (
            contact_load_realization_drop_trend_ratio is not None
            and contact_load_realization_drop_trend_ratio
            > contact_max_load_realization_drop_trend_ratio_threshold
        ):
            reasons.append(
                Reason(
                    code="CONTACT_LOAD_REALIZATION_DROP_TREND_WORSENING",
                    severity="fail" if protected else "warn",
                    detail=(
                        "contact load realization drop trend ratio "
                        f"{contact_load_realization_drop_trend_ratio:.3f} exceeds "
                        f"threshold {contact_max_load_realization_drop_trend_ratio_threshold:.3f}"
                    ),
                )
            )

        contact_load_amplification_trend_ratio = fixture_trend_ratio(
            "contact_load_amplification_ratio"
        )
        if (
            contact_load_amplification_trend_ratio is not None
            and contact_load_amplification_trend_ratio
            > contact_max_load_amplification_trend_ratio_threshold
        ):
            reasons.append(
                Reason(
                    code="CONTACT_LOAD_AMPLIFICATION_TREND_WORSENING",
                    severity="fail" if protected else "warn",
                    detail=(
                        "contact load amplification trend ratio "
                        f"{contact_load_amplification_trend_ratio:.3f} exceeds "
                        f"threshold {contact_max_load_amplification_trend_ratio_threshold:.3f}"
                    ),
                )
            )

        reference_fixture_ids = {
            "nonlinear_contact_frictionless_reference_gpu_provider",
            "nonlinear_contact_frictionless_reference_complex_gpu_provider",
        }
        latest_contact_reference_values = []
        for rec in report_records(latest):
            if rec.get("fixture_id") not in reference_fixture_ids:
                continue
            raw_value = rec.get("contact_nonlinear_severity")
            if isinstance(raw_value, (int, float)):
                latest_contact_reference_values.append(float(raw_value))
        rolling_contact_reference_values = []
        for report in rolling:
            for rec in report_records(report):
                if rec.get("fixture_id") not in reference_fixture_ids:
                    continue
                raw_value = rec.get("contact_nonlinear_severity")
                if isinstance(raw_value, (int, float)):
                    rolling_contact_reference_values.append(float(raw_value))
        if latest_contact_reference_values and rolling_contact_reference_values:
            latest_reference_contact = statistics.median(latest_contact_reference_values)
            baseline_reference_contact = statistics.median(rolling_contact_reference_values)
            if baseline_reference_contact > 0:
                contact_reference_trend_ratio = (
                    latest_reference_contact / baseline_reference_contact
                )
                if (
                    contact_reference_trend_ratio
                    > contact_reference_max_trend_ratio_threshold
                ):
                    reasons.append(
                        Reason(
                            code="CONTACT_REFERENCE_TREND_WORSENING",
                            severity="fail" if protected else "warn",
                            detail=(
                                f"contact reference trend ratio {contact_reference_trend_ratio:.3f} exceeds "
                                f"threshold {contact_reference_max_trend_ratio_threshold:.3f}"
                            ),
                        )
                    )

    if electro_time_scale_breach_values:
        electro_time_scale_breach_rate = (
            sum(1 for breached in electro_time_scale_breach_values if breached)
            / len(electro_time_scale_breach_values)
        )
        if electro_time_scale_breach_rate > electro_max_time_scale_breach_rate_threshold:
            reasons.append(
                Reason(
                    code="ELECTRO_TIME_SCALE_BREACH_RATE_HIGH",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"electro-thermal time-scale breach rate {electro_time_scale_breach_rate:.3f} exceeds "
                        f"threshold {electro_max_time_scale_breach_rate_threshold:.3f}"
                    ),
                )
            )

    plastic_reference_fixture_ids = {
        "nonlinear_plastic_hardening_reference_gpu_provider",
        "nonlinear_plastic_hardening_reference_complex_gpu_provider",
    }
    contact_reference_fixture_ids = {
        "nonlinear_contact_frictionless_reference_gpu_provider",
        "nonlinear_contact_frictionless_reference_complex_gpu_provider",
    }
    plastic_promotion_sample_count = sum(
        1
        for rec in report_records(latest)
        if rec.get("fixture_id") in plastic_reference_fixture_ids
        and isinstance(rec.get("plastic_nonlinear_severity"), (int, float))
    )
    contact_promotion_sample_count = sum(
        1
        for rec in report_records(latest)
        if rec.get("fixture_id") in contact_reference_fixture_ids
        and isinstance(rec.get("contact_nonlinear_severity"), (int, float))
    )

    if plastic_promotion_sample_count < plastic_promotion_min_samples:
        plastic_promotion_blockers.append(f"sample_count<{plastic_promotion_min_samples}")
    if plastic_reference_trend_ratio is None:
        plastic_promotion_blockers.append("reference_trend_ratio_missing")
    elif plastic_reference_trend_ratio > plastic_reference_max_trend_ratio_threshold:
        plastic_promotion_blockers.append("reference_trend_ratio_high")
    if plastic_breach_rate is None:
        plastic_promotion_blockers.append("breach_rate_missing")
    elif plastic_breach_rate > plastic_max_breach_rate_threshold:
        plastic_promotion_blockers.append("breach_rate_high")
    plastic_promotion_ready = len(plastic_promotion_blockers) == 0

    if contact_promotion_sample_count < contact_promotion_min_samples:
        contact_promotion_blockers.append(f"sample_count<{contact_promotion_min_samples}")
    if contact_reference_trend_ratio is None:
        contact_promotion_blockers.append("reference_trend_ratio_missing")
    elif contact_reference_trend_ratio > contact_reference_max_trend_ratio_threshold:
        contact_promotion_blockers.append("reference_trend_ratio_high")
    if contact_breach_rate is None:
        contact_promotion_blockers.append("breach_rate_missing")
    elif contact_breach_rate > contact_max_breach_rate_threshold:
        contact_promotion_blockers.append("breach_rate_high")
    contact_promotion_ready = len(contact_promotion_blockers) == 0
    plastic_promotion_blocker_count = len(plastic_promotion_blockers)
    contact_promotion_blocker_count = len(contact_promotion_blockers)

    if plastic_promotion_blocker_count > plastic_promotion_max_blockers:
        reasons.append(
            Reason(
                code="PLASTIC_PROMOTION_BLOCKER_BUDGET_EXCEEDED",
                severity="fail" if protected else "warn",
                detail=(
                    f"plastic promotion blocker count {plastic_promotion_blocker_count} exceeds budget "
                    f"{plastic_promotion_max_blockers}"
                ),
            )
        )
    if contact_promotion_blocker_count > contact_promotion_max_blockers:
        reasons.append(
            Reason(
                code="CONTACT_PROMOTION_BLOCKER_BUDGET_EXCEEDED",
                severity="fail" if protected else "warn",
                detail=(
                    f"contact promotion blocker count {contact_promotion_blocker_count} exceeds budget "
                    f"{contact_promotion_max_blockers}"
                ),
            )
        )

    def _promotion_blocker_count_for_report(
        report: dict,
        fixture_ids: set[str],
        severity_field: str,
        min_samples: int,
        severity_threshold: float,
    ) -> int:
        values = []
        for rec in report_records(report):
            if rec.get("fixture_id") not in fixture_ids:
                continue
            raw = rec.get(severity_field)
            if isinstance(raw, (int, float)):
                values.append(float(raw))
        count = 0
        if len(values) < min_samples:
            count += 1
        if not values:
            count += 1
        else:
            breach_rate = sum(1 for value in values if value > severity_threshold) / len(values)
            if breach_rate > 0.0:
                count += 1
        return count

    if rolling:
        rolling_plastic_blockers = [
            _promotion_blocker_count_for_report(
                report,
                plastic_reference_fixture_ids,
                "plastic_nonlinear_severity",
                plastic_promotion_min_samples,
                plastic_max_nonlinear_severity_threshold,
            )
            for report in rolling
        ]
        if rolling_plastic_blockers:
            plastic_promotion_blocker_regression = (
                plastic_promotion_blocker_count
                - int(statistics.median(rolling_plastic_blockers))
            )
            if plastic_promotion_blocker_regression > promotion_max_blocker_regression:
                reasons.append(
                    Reason(
                        code="PLASTIC_PROMOTION_BLOCKER_BURNDOWN_STALLED",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"plastic promotion blocker regression {plastic_promotion_blocker_regression} exceeds "
                            f"allowed {promotion_max_blocker_regression}"
                        ),
                    )
                )

        rolling_contact_blockers = [
            _promotion_blocker_count_for_report(
                report,
                contact_reference_fixture_ids,
                "contact_nonlinear_severity",
                contact_promotion_min_samples,
                contact_max_nonlinear_severity_threshold,
            )
            for report in rolling
        ]
        if rolling_contact_blockers:
            contact_promotion_blocker_regression = (
                contact_promotion_blocker_count
                - int(statistics.median(rolling_contact_blockers))
            )
            if contact_promotion_blocker_regression > promotion_max_blocker_regression:
                reasons.append(
                    Reason(
                        code="CONTACT_PROMOTION_BLOCKER_BURNDOWN_STALLED",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"contact promotion blocker regression {contact_promotion_blocker_regression} exceeds "
                            f"allowed {promotion_max_blocker_regression}"
                        ),
                    )
                )
    elif protected or require_promotion_ready:
        reasons.append(
            Reason(
                code="PLASTIC_PROMOTION_BLOCKER_BASELINE_MISSING",
                severity="fail" if protected else "warn",
                detail="no rolling reports available to compute plastic promotion blocker regression",
            )
        )
        reasons.append(
            Reason(
                code="CONTACT_PROMOTION_BLOCKER_BASELINE_MISSING",
                severity="fail" if protected else "warn",
                detail="no rolling reports available to compute contact promotion blocker regression",
            )
        )

    if not plastic_promotion_ready and (protected or require_promotion_ready):
        reasons.append(
            Reason(
                code="PLASTIC_PROMOTION_NOT_READY",
                severity="fail" if protected else "warn",
                detail=(
                    "plastic promotion blockers: "
                    + ", ".join(sorted(plastic_promotion_blockers))
                ),
            )
        )
    if not contact_promotion_ready and (protected or require_promotion_ready):
        reasons.append(
            Reason(
                code="CONTACT_PROMOTION_NOT_READY",
                severity="fail" if protected else "warn",
                detail=(
                    "contact promotion blockers: "
                    + ", ".join(sorted(contact_promotion_blockers))
                ),
            )
        )

    if isinstance(thermo_promotion_report, dict) and thermo_promotion_report.get("blocked"):
        blocked_reasons = thermo_promotion_report.get("reasons")
        if isinstance(blocked_reasons, list) and blocked_reasons:
            detail = "; ".join(str(reason) for reason in blocked_reasons)
        else:
            detail = "thermo field promotion report indicates blocked status"
        reasons.append(
            Reason(
                code="THERMO_FIELD_PROMOTION_BLOCKED",
                severity="fail" if protected else "warn",
                detail=detail,
            )
        )

    signing_key = os.getenv("RUNMAT_THERMO_FIELD_SIGNING_KEY", "")
    if governance_profile_name() == "release":
        if not signing_key or signing_key == "runmat-dev-thermo-signing-key":
            reasons.append(
                Reason(
                    code="THERMO_FIELD_SIGNING_KEY_UNSAFE",
                    severity="fail" if protected else "warn",
                    detail=(
                        "release governance requires non-default thermo field signing key"
                    ),
                )
            )

    if any(reason.severity == "fail" for reason in reasons):
        verdict = "fail"
    elif reasons:
        verdict = "warn"
    else:
        verdict = "pass"

    return {
        "schema_version": "analysis-release-readiness/v1",
        "verdict": verdict,
        "protected_branch": protected,
        "reasons": [reason.__dict__ for reason in reasons],
        "latest_report_passed": latest_passed,
        "nonlinear_fixture_count": len(records),
        "prep_acceptance_rate": acceptance_rate,
        "prep_calibration_max_observed_drift": prep_calibration_max_observed_drift,
        "prep_calibration_recommendation_count": prep_calibration_recommendation_count,
        "thermo_coupling_enabled_rate": thermo_coupling_enabled_rate,
        "thermo_max_transient_severity": thermo_max_transient_severity,
        "thermo_max_nonlinear_severity": thermo_max_nonlinear_severity,
        "thermo_max_spread_ratio": thermo_max_spread_ratio,
        "thermo_max_heterogeneity_index": thermo_max_heterogeneity_index,
        "thermo_min_field_coverage_ratio": thermo_min_field_coverage_ratio,
        "thermo_max_field_extrapolation_ratio": thermo_max_field_extrapolation_ratio,
        "thermo_max_field_clamp_ratio": thermo_max_field_clamp_ratio,
        "thermo_field_coverage_drop_trend_ratio": thermo_field_coverage_drop_trend_ratio,
        "thermo_field_extrapolation_trend_ratio": thermo_field_extrapolation_trend_ratio,
        "thermo_field_clamp_trend_ratio": thermo_field_clamp_trend_ratio,
        "thermo_max_spread_ratio_threshold": thermo_max_spread_ratio_threshold,
        "thermo_min_field_coverage_ratio_threshold": thermo_min_field_coverage_ratio_threshold,
        "thermo_max_field_extrapolation_ratio_threshold": thermo_max_field_extrapolation_ratio_threshold,
        "thermo_max_field_clamp_ratio_threshold": thermo_max_field_clamp_ratio_threshold,
        "thermo_max_field_coverage_drop_trend_ratio": thermo_max_field_coverage_drop_trend_ratio,
        "thermo_max_field_extrapolation_trend_ratio": thermo_max_field_extrapolation_trend_ratio,
        "thermo_max_field_clamp_trend_ratio": thermo_max_field_clamp_trend_ratio,
        "thermo_require_artifact_backed": thermo_require_artifact_backed,
        "thermo_field_artifact_max_age_days": thermo_field_artifact_max_age_days,
        "thermo_signing_key_safe": bool(
            signing_key and signing_key != "runmat-dev-thermo-signing-key"
        ),
        "thermo_spread_breach_rate": thermo_spread_breach_rate,
        "thermo_heterogeneity_breach_rate": thermo_heterogeneity_breach_rate,
        "thermo_field_clamp_breach_rate": thermo_field_clamp_breach_rate,
        "thermo_max_field_clamp_breach_rate_threshold": thermo_max_field_clamp_breach_rate_threshold,
        "thermo_spread_trend_ratio": thermo_spread_trend_ratio,
        "thermo_heterogeneity_trend_ratio": thermo_heterogeneity_trend_ratio,
        "thermal_max_residual_norm": thermal_max_residual_norm,
        "thermal_max_residual_norm_threshold": thermal_max_residual_norm_threshold,
        "thermal_min_temperature_k": thermal_min_temperature_k,
        "thermal_min_temperature_k_threshold": thermal_min_temperature_k_threshold,
        "thermal_max_temperature_k": thermal_max_temperature_k,
        "thermal_max_temperature_k_threshold": thermal_max_temperature_k_threshold,
        "thermal_max_conductivity_spread_ratio": thermal_max_conductivity_spread_ratio,
        "thermal_max_conductivity_spread_ratio_threshold": thermal_max_conductivity_spread_ratio_threshold,
        "thermal_max_heat_capacity_spread_ratio": thermal_max_heat_capacity_spread_ratio,
        "thermal_max_heat_capacity_spread_ratio_threshold": thermal_max_heat_capacity_spread_ratio_threshold,
        "thermal_max_spatial_gradient_index": thermal_max_spatial_gradient_index,
        "thermal_max_spatial_gradient_index_threshold": thermal_max_spatial_gradient_index_threshold,
        "thermal_min_monotonic_response_fraction": thermal_min_monotonic_response_fraction,
        "thermal_min_monotonic_response_fraction_threshold": thermal_min_monotonic_response_fraction_threshold,
        "thermal_min_response_realization_ratio": thermal_min_response_realization_ratio,
        "thermal_min_response_realization_ratio_threshold": thermal_min_response_realization_ratio_threshold,
        "thermal_max_response_realization_ratio": thermal_max_response_realization_ratio,
        "thermal_max_response_realization_ratio_threshold": thermal_max_response_realization_ratio_threshold,
        "thermal_spread_breach_rate": thermal_spread_breach_rate,
        "thermal_max_spread_breach_rate_threshold": thermal_max_spread_breach_rate_threshold,
        "thermal_spread_trend_ratio": thermal_spread_trend_ratio,
        "thermal_max_spread_trend_ratio_threshold": thermal_max_spread_trend_ratio_threshold,
        "electro_coupling_enabled_rate": electro_coupling_enabled_rate,
        "electro_max_transient_severity": electro_max_transient_severity,
        "electro_max_nonlinear_severity": electro_max_nonlinear_severity,
        "electro_max_joule_heating_scale": electro_max_joule_heating_scale,
        "electro_max_conductivity_spread_ratio": electro_max_conductivity_spread_ratio,
        "electro_min_time_scale_mean": electro_min_time_scale_mean,
        "electro_max_time_scale_mean": electro_max_time_scale_mean,
        "electro_max_joule_heating_scale_threshold": electro_max_joule_heating_scale_threshold,
        "electro_max_conductivity_spread_ratio_threshold": electro_max_conductivity_spread_ratio_threshold,
        "electro_min_time_scale_mean_threshold": electro_min_time_scale_mean_threshold,
        "electro_max_time_scale_mean_threshold": electro_max_time_scale_mean_threshold,
        "electro_joule_breach_rate": electro_joule_breach_rate,
        "electro_spread_breach_rate": electro_spread_breach_rate,
        "electro_time_scale_breach_rate": electro_time_scale_breach_rate,
        "electro_max_joule_breach_rate_threshold": electro_max_joule_breach_rate_threshold,
        "electro_max_spread_breach_rate_threshold": electro_max_spread_breach_rate_threshold,
        "electro_max_time_scale_breach_rate_threshold": electro_max_time_scale_breach_rate_threshold,
        "electro_joule_trend_ratio": electro_joule_trend_ratio,
        "electro_spread_trend_ratio": electro_spread_trend_ratio,
        "electro_time_scale_trend_ratio": electro_time_scale_trend_ratio,
        "electro_max_joule_trend_ratio_threshold": electro_max_joule_trend_ratio_threshold,
        "electro_max_spread_trend_ratio_threshold": electro_max_spread_trend_ratio_threshold,
        "electro_max_time_scale_trend_ratio_threshold": electro_max_time_scale_trend_ratio_threshold,
        "em_max_energy_imbalance_ratio": em_max_energy_imbalance_ratio,
        "em_max_energy_imbalance_ratio_threshold": em_max_energy_imbalance_ratio_threshold,
        "em_max_flux_divergence_proxy": em_max_flux_divergence_proxy,
        "em_max_flux_divergence_proxy_threshold": em_max_flux_divergence_proxy_threshold,
        "em_max_real_residual_norm": em_max_real_residual_norm,
        "em_max_real_residual_norm_threshold": em_max_real_residual_norm_threshold,
        "em_max_imag_residual_norm": em_max_imag_residual_norm,
        "em_max_imag_residual_norm_threshold": em_max_imag_residual_norm_threshold,
        "em_min_source_realization_ratio": em_min_source_realization_ratio,
        "em_min_source_realization_ratio_threshold": em_min_source_realization_ratio_threshold,
        "em_min_source_region_coverage_ratio": em_min_source_region_coverage_ratio,
        "em_min_source_region_coverage_ratio_threshold": em_min_source_region_coverage_ratio_threshold,
        "em_min_source_material_alignment_ratio": em_min_source_material_alignment_ratio,
        "em_min_source_material_alignment_ratio_threshold": em_min_source_material_alignment_ratio_threshold,
        "em_min_dispersive_phase_attenuation_mean": em_min_dispersive_phase_attenuation_mean,
        "em_min_dispersive_phase_attenuation_mean_threshold": em_min_dispersive_phase_attenuation_mean_threshold,
        "em_min_dispersive_phase_conductivity_attenuation_ratio": em_min_dispersive_phase_conductivity_attenuation_ratio,
        "em_min_dispersive_phase_conductivity_attenuation_ratio_threshold": em_min_dispersive_phase_conductivity_attenuation_ratio_threshold,
        "em_breach_rate": em_breach_rate,
        "em_max_breach_rate_threshold": em_max_breach_rate_threshold,
        "em_energy_imbalance_trend_ratio": em_energy_imbalance_trend_ratio,
        "em_max_energy_imbalance_trend_ratio_threshold": em_max_energy_imbalance_trend_ratio_threshold,
        "em_flux_divergence_trend_ratio": em_flux_divergence_trend_ratio,
        "em_max_flux_divergence_trend_ratio_threshold": em_max_flux_divergence_trend_ratio_threshold,
        "em_dispersive_phase_attenuation_trend_ratio": em_dispersive_phase_attenuation_trend_ratio,
        "em_max_dispersive_phase_attenuation_trend_ratio_threshold": em_max_dispersive_phase_attenuation_trend_ratio_threshold,
        "em_dispersive_phase_conductivity_attenuation_trend_ratio": em_dispersive_phase_conductivity_attenuation_trend_ratio,
        "em_max_dispersive_phase_conductivity_attenuation_trend_ratio_threshold": em_max_dispersive_phase_conductivity_attenuation_trend_ratio_threshold,
        "plastic_max_nonlinear_severity": plastic_max_nonlinear_severity,
        "plastic_max_nonlinear_severity_threshold": plastic_max_nonlinear_severity_threshold,
        "plastic_load_realization_ratio_min": plastic_load_realization_ratio_min,
        "plastic_load_realization_ratio_max": plastic_load_realization_ratio_max,
        "plastic_min_load_realization_ratio_threshold": plastic_min_load_realization_ratio_threshold,
        "plastic_max_load_realization_ratio_threshold": plastic_max_load_realization_ratio_threshold,
        "plastic_load_amplification_ratio_min": plastic_load_amplification_ratio_min,
        "plastic_load_amplification_ratio_max": plastic_load_amplification_ratio_max,
        "plastic_min_load_amplification_ratio_threshold": plastic_min_load_amplification_ratio_threshold,
        "plastic_max_load_amplification_ratio_threshold": plastic_max_load_amplification_ratio_threshold,
        "plastic_breach_rate": plastic_breach_rate,
        "plastic_max_breach_rate_threshold": plastic_max_breach_rate_threshold,
        "plastic_load_realization_breach_rate": plastic_load_realization_breach_rate,
        "plastic_max_load_realization_breach_rate_threshold": plastic_max_load_realization_breach_rate_threshold,
        "plastic_load_amplification_breach_rate": plastic_load_amplification_breach_rate,
        "plastic_max_load_amplification_breach_rate_threshold": plastic_max_load_amplification_breach_rate_threshold,
        "plastic_trend_ratio": plastic_trend_ratio,
        "plastic_max_trend_ratio_threshold": plastic_max_trend_ratio_threshold,
        "plastic_load_realization_drop_trend_ratio": plastic_load_realization_drop_trend_ratio,
        "plastic_max_load_realization_drop_trend_ratio_threshold": plastic_max_load_realization_drop_trend_ratio_threshold,
        "plastic_load_amplification_trend_ratio": plastic_load_amplification_trend_ratio,
        "plastic_max_load_amplification_trend_ratio_threshold": plastic_max_load_amplification_trend_ratio_threshold,
        "plastic_reference_trend_ratio": plastic_reference_trend_ratio,
        "plastic_reference_max_trend_ratio_threshold": plastic_reference_max_trend_ratio_threshold,
        "plastic_promotion_ready": plastic_promotion_ready,
        "plastic_promotion_blockers": plastic_promotion_blockers,
        "plastic_promotion_sample_count": plastic_promotion_sample_count,
        "plastic_promotion_min_samples": plastic_promotion_min_samples,
        "plastic_promotion_blocker_count": plastic_promotion_blocker_count,
        "plastic_promotion_max_blockers": plastic_promotion_max_blockers,
        "plastic_promotion_blocker_regression": plastic_promotion_blocker_regression,
        "contact_max_nonlinear_severity": contact_max_nonlinear_severity,
        "contact_max_nonlinear_severity_threshold": contact_max_nonlinear_severity_threshold,
        "contact_load_realization_ratio_min": contact_load_realization_ratio_min,
        "contact_load_realization_ratio_max": contact_load_realization_ratio_max,
        "contact_min_load_realization_ratio_threshold": contact_min_load_realization_ratio_threshold,
        "contact_max_load_realization_ratio_threshold": contact_max_load_realization_ratio_threshold,
        "contact_load_amplification_ratio_min": contact_load_amplification_ratio_min,
        "contact_load_amplification_ratio_max": contact_load_amplification_ratio_max,
        "contact_min_load_amplification_ratio_threshold": contact_min_load_amplification_ratio_threshold,
        "contact_max_load_amplification_ratio_threshold": contact_max_load_amplification_ratio_threshold,
        "contact_breach_rate": contact_breach_rate,
        "contact_max_breach_rate_threshold": contact_max_breach_rate_threshold,
        "contact_load_realization_breach_rate": contact_load_realization_breach_rate,
        "contact_max_load_realization_breach_rate_threshold": contact_max_load_realization_breach_rate_threshold,
        "contact_load_amplification_breach_rate": contact_load_amplification_breach_rate,
        "contact_max_load_amplification_breach_rate_threshold": contact_max_load_amplification_breach_rate_threshold,
        "contact_trend_ratio": contact_trend_ratio,
        "contact_max_trend_ratio_threshold": contact_max_trend_ratio_threshold,
        "contact_load_realization_drop_trend_ratio": contact_load_realization_drop_trend_ratio,
        "contact_max_load_realization_drop_trend_ratio_threshold": contact_max_load_realization_drop_trend_ratio_threshold,
        "contact_load_amplification_trend_ratio": contact_load_amplification_trend_ratio,
        "contact_max_load_amplification_trend_ratio_threshold": contact_max_load_amplification_trend_ratio_threshold,
        "contact_reference_trend_ratio": contact_reference_trend_ratio,
        "contact_reference_max_trend_ratio_threshold": contact_reference_max_trend_ratio_threshold,
        "contact_promotion_ready": contact_promotion_ready,
        "contact_promotion_blockers": contact_promotion_blockers,
        "contact_promotion_sample_count": contact_promotion_sample_count,
        "contact_promotion_min_samples": contact_promotion_min_samples,
        "contact_promotion_blocker_count": contact_promotion_blocker_count,
        "contact_promotion_max_blockers": contact_promotion_max_blockers,
        "contact_promotion_blocker_regression": contact_promotion_blocker_regression,
        "require_promotion_ready": require_promotion_ready,
        "promotion_max_blocker_regression": promotion_max_blocker_regression,
        "promotion_calibration_applied": promotion_calibration_applied,
        "require_promotion_calibration": require_promotion_calibration,
        "promotion_calibration_age_days": promotion_calibration_age_days,
        "promotion_calibration_max_age_days": promotion_calibration_max_age_days,
        "promotion_min_rolling_reports": promotion_min_rolling_reports,
        "promotion_history_sufficient": promotion_history_sufficient,
        "reference_trend_ratcheted": True,
        "reference_trend_rationale": "rolling_median_reference_fixtures",
        "rolling_report_count": len(rolling),
        "governance_profile": governance_profile_name(),
    }


def markdown_summary(result: dict) -> str:
    lines = ["## Nonlinear Release Readiness", ""]
    lines.append(f"Verdict: **{result['verdict']}**")
    lines.append(f"Governance profile: **{result.get('governance_profile', 'unknown')}**")
    lines.append(f"Protected branch enforcement: **{result['protected_branch']}**")
    lines.append(
        "Reference trend ratchet: "
        f"**{result.get('reference_trend_ratcheted', False)}** "
        f"(basis=`{result.get('reference_trend_rationale', '-')}`, rolling_reports=`{result.get('rolling_report_count', 0)}`)"
    )
    lines.append(
        "Promotion calibration applied: "
        f"**{result.get('promotion_calibration_applied', False)}** "
        f"(required=`{result.get('require_promotion_calibration', False)}`)"
    )
    lines.append(
        "Promotion calibration age/max days: "
        f"`{result.get('promotion_calibration_age_days') if result.get('promotion_calibration_age_days') is not None else '-'}`/`{result.get('promotion_calibration_max_age_days') if result.get('promotion_calibration_max_age_days') is not None else '-'}`"
    )
    lines.append(
        "Promotion history sufficient: "
        f"**{result.get('promotion_history_sufficient', False)}** "
        f"(rolling=`{result.get('rolling_report_count', 0)}`, min=`{result.get('promotion_min_rolling_reports', 0)}`)"
    )
    lines.append("")
    lines.append("### Thermo Posture")
    lines.append(
        "- Thermo coupling enabled-rate: "
        f"`{result.get('thermo_coupling_enabled_rate') if result.get('thermo_coupling_enabled_rate') is not None else '-'}`"
    )
    lines.append(
        "- Max thermo transient severity: "
        f"`{result.get('thermo_max_transient_severity') if result.get('thermo_max_transient_severity') is not None else '-'}`"
    )
    lines.append(
        "- Max thermo nonlinear severity: "
        f"`{result.get('thermo_max_nonlinear_severity') if result.get('thermo_max_nonlinear_severity') is not None else '-'}`"
    )
    lines.append(
        "- Max thermo material spread ratio: "
        f"`{result.get('thermo_max_spread_ratio') if result.get('thermo_max_spread_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermo spread ratio threshold: "
        f"`{result.get('thermo_max_spread_ratio_threshold') if result.get('thermo_max_spread_ratio_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Max thermo assignment heterogeneity index: "
        f"`{result.get('thermo_max_heterogeneity_index') if result.get('thermo_max_heterogeneity_index') is not None else '-'}`"
    )
    lines.append(
        "- Min thermo field coverage ratio: "
        f"`{result.get('thermo_min_field_coverage_ratio') if result.get('thermo_min_field_coverage_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermo field coverage threshold: "
        f"`{result.get('thermo_min_field_coverage_ratio_threshold') if result.get('thermo_min_field_coverage_ratio_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Max thermo field extrapolation ratio: "
        f"`{result.get('thermo_max_field_extrapolation_ratio') if result.get('thermo_max_field_extrapolation_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermo field extrapolation threshold: "
        f"`{result.get('thermo_max_field_extrapolation_ratio_threshold') if result.get('thermo_max_field_extrapolation_ratio_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Max thermo field clamp ratio: "
        f"`{result.get('thermo_max_field_clamp_ratio') if result.get('thermo_max_field_clamp_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermo field clamp threshold: "
        f"`{result.get('thermo_max_field_clamp_ratio_threshold') if result.get('thermo_max_field_clamp_ratio_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Thermo field coverage drop trend ratio: "
        f"`{result.get('thermo_field_coverage_drop_trend_ratio') if result.get('thermo_field_coverage_drop_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermo field coverage drop trend threshold: "
        f"`{result.get('thermo_max_field_coverage_drop_trend_ratio') if result.get('thermo_max_field_coverage_drop_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermo field extrapolation trend ratio: "
        f"`{result.get('thermo_field_extrapolation_trend_ratio') if result.get('thermo_field_extrapolation_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermo field extrapolation trend threshold: "
        f"`{result.get('thermo_max_field_extrapolation_trend_ratio') if result.get('thermo_max_field_extrapolation_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermo field clamp trend ratio: "
        f"`{result.get('thermo_field_clamp_trend_ratio') if result.get('thermo_field_clamp_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermo field clamp trend threshold: "
        f"`{result.get('thermo_max_field_clamp_trend_ratio') if result.get('thermo_max_field_clamp_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermo artifact-backed required: "
        f"`{result.get('thermo_require_artifact_backed') if result.get('thermo_require_artifact_backed') is not None else '-'}`"
    )
    lines.append(
        "- Thermo field artifact max age days: "
        f"`{result.get('thermo_field_artifact_max_age_days') if result.get('thermo_field_artifact_max_age_days') is not None else '-'}`"
    )
    lines.append(
        "- Thermo signing key safe: "
        f"`{result.get('thermo_signing_key_safe') if result.get('thermo_signing_key_safe') is not None else '-'}`"
    )
    lines.append(
        "- Thermo spread breach rate: "
        f"`{result.get('thermo_spread_breach_rate') if result.get('thermo_spread_breach_rate') is not None else '-'}`"
    )
    lines.append(
        "- Thermo heterogeneity breach rate: "
        f"`{result.get('thermo_heterogeneity_breach_rate') if result.get('thermo_heterogeneity_breach_rate') is not None else '-'}`"
    )
    lines.append(
        "- Thermo field clamp breach rate: "
        f"`{result.get('thermo_field_clamp_breach_rate') if result.get('thermo_field_clamp_breach_rate') is not None else '-'}`"
    )
    lines.append(
        "- Thermo spread trend ratio: "
        f"`{result.get('thermo_spread_trend_ratio') if result.get('thermo_spread_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermo heterogeneity trend ratio: "
        f"`{result.get('thermo_heterogeneity_trend_ratio') if result.get('thermo_heterogeneity_trend_ratio') is not None else '-'}`"
    )
    lines.append("")
    lines.append("### Thermal Posture")
    lines.append(
        "- Max thermal residual norm: "
        f"`{result.get('thermal_max_residual_norm') if result.get('thermal_max_residual_norm') is not None else '-'}`"
    )
    lines.append(
        "- Thermal residual threshold: "
        f"`{result.get('thermal_max_residual_norm_threshold') if result.get('thermal_max_residual_norm_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Min thermal temperature (K): "
        f"`{result.get('thermal_min_temperature_k') if result.get('thermal_min_temperature_k') is not None else '-'}`"
    )
    lines.append(
        "- Thermal min temperature threshold (K): "
        f"`{result.get('thermal_min_temperature_k_threshold') if result.get('thermal_min_temperature_k_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Max thermal temperature (K): "
        f"`{result.get('thermal_max_temperature_k') if result.get('thermal_max_temperature_k') is not None else '-'}`"
    )
    lines.append(
        "- Thermal max temperature threshold (K): "
        f"`{result.get('thermal_max_temperature_k_threshold') if result.get('thermal_max_temperature_k_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Max thermal conductivity spread ratio: "
        f"`{result.get('thermal_max_conductivity_spread_ratio') if result.get('thermal_max_conductivity_spread_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermal conductivity spread threshold: "
        f"`{result.get('thermal_max_conductivity_spread_ratio_threshold') if result.get('thermal_max_conductivity_spread_ratio_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Max thermal heat-capacity spread ratio: "
        f"`{result.get('thermal_max_heat_capacity_spread_ratio') if result.get('thermal_max_heat_capacity_spread_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermal heat-capacity spread threshold: "
        f"`{result.get('thermal_max_heat_capacity_spread_ratio_threshold') if result.get('thermal_max_heat_capacity_spread_ratio_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Max thermal spatial gradient index: "
        f"`{result.get('thermal_max_spatial_gradient_index') if result.get('thermal_max_spatial_gradient_index') is not None else '-'}`"
    )
    lines.append(
        "- Thermal spatial gradient threshold: "
        f"`{result.get('thermal_max_spatial_gradient_index_threshold') if result.get('thermal_max_spatial_gradient_index_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Min thermal monotonic response fraction: "
        f"`{result.get('thermal_min_monotonic_response_fraction') if result.get('thermal_min_monotonic_response_fraction') is not None else '-'}`"
    )
    lines.append(
        "- Thermal monotonic response threshold: "
        f"`{result.get('thermal_min_monotonic_response_fraction_threshold') if result.get('thermal_min_monotonic_response_fraction_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Min thermal response realization ratio: "
        f"`{result.get('thermal_min_response_realization_ratio') if result.get('thermal_min_response_realization_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermal min response realization threshold: "
        f"`{result.get('thermal_min_response_realization_ratio_threshold') if result.get('thermal_min_response_realization_ratio_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Max thermal response realization ratio: "
        f"`{result.get('thermal_max_response_realization_ratio') if result.get('thermal_max_response_realization_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermal max response realization threshold: "
        f"`{result.get('thermal_max_response_realization_ratio_threshold') if result.get('thermal_max_response_realization_ratio_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Thermal spread breach rate: "
        f"`{result.get('thermal_spread_breach_rate') if result.get('thermal_spread_breach_rate') is not None else '-'}`"
    )
    lines.append(
        "- Thermal spread breach threshold: "
        f"`{result.get('thermal_max_spread_breach_rate_threshold') if result.get('thermal_max_spread_breach_rate_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Thermal spread trend ratio: "
        f"`{result.get('thermal_spread_trend_ratio') if result.get('thermal_spread_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Thermal spread trend threshold: "
        f"`{result.get('thermal_max_spread_trend_ratio_threshold') if result.get('thermal_max_spread_trend_ratio_threshold') is not None else '-'}`"
    )
    lines.append("")
    lines.append("### Electro-Thermal Posture")
    lines.append(
        "- Electro-thermal coupling enabled-rate: "
        f"`{result.get('electro_coupling_enabled_rate') if result.get('electro_coupling_enabled_rate') is not None else '-'}`"
    )
    lines.append(
        "- Max electro-thermal transient severity: "
        f"`{result.get('electro_max_transient_severity') if result.get('electro_max_transient_severity') is not None else '-'}`"
    )
    lines.append(
        "- Max electro-thermal nonlinear severity: "
        f"`{result.get('electro_max_nonlinear_severity') if result.get('electro_max_nonlinear_severity') is not None else '-'}`"
    )
    lines.append(
        "- Max electro-thermal Joule heating scale: "
        f"`{result.get('electro_max_joule_heating_scale') if result.get('electro_max_joule_heating_scale') is not None else '-'}`"
    )
    lines.append(
        "- Electro-thermal Joule threshold: "
        f"`{result.get('electro_max_joule_heating_scale_threshold') if result.get('electro_max_joule_heating_scale_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Max electro-thermal conductivity spread ratio: "
        f"`{result.get('electro_max_conductivity_spread_ratio') if result.get('electro_max_conductivity_spread_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Electro-thermal spread threshold: "
        f"`{result.get('electro_max_conductivity_spread_ratio_threshold') if result.get('electro_max_conductivity_spread_ratio_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Electro-thermal time-scale min/max: "
        f"`{result.get('electro_min_time_scale_mean') if result.get('electro_min_time_scale_mean') is not None else '-'}`/`{result.get('electro_max_time_scale_mean') if result.get('electro_max_time_scale_mean') is not None else '-'}`"
    )
    lines.append(
        "- Electro-thermal time-scale thresholds (min/max): "
        f"`{result.get('electro_min_time_scale_mean_threshold') if result.get('electro_min_time_scale_mean_threshold') is not None else '-'}`/`{result.get('electro_max_time_scale_mean_threshold') if result.get('electro_max_time_scale_mean_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Electro-thermal Joule breach rate: "
        f"`{result.get('electro_joule_breach_rate') if result.get('electro_joule_breach_rate') is not None else '-'}`"
    )
    lines.append(
        "- Electro-thermal spread breach rate: "
        f"`{result.get('electro_spread_breach_rate') if result.get('electro_spread_breach_rate') is not None else '-'}`"
    )
    lines.append(
        "- Electro-thermal time-scale breach rate: "
        f"`{result.get('electro_time_scale_breach_rate') if result.get('electro_time_scale_breach_rate') is not None else '-'}`"
    )
    lines.append(
        "- Electro-thermal Joule trend ratio: "
        f"`{result.get('electro_joule_trend_ratio') if result.get('electro_joule_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Electro-thermal spread trend ratio: "
        f"`{result.get('electro_spread_trend_ratio') if result.get('electro_spread_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Electro-thermal time-scale trend ratio: "
        f"`{result.get('electro_time_scale_trend_ratio') if result.get('electro_time_scale_trend_ratio') is not None else '-'}`"
    )
    lines.append("")
    lines.append("### Plasticity Posture")
    lines.append(
        "- Max plastic nonlinear severity: "
        f"`{result.get('plastic_max_nonlinear_severity') if result.get('plastic_max_nonlinear_severity') is not None else '-'}`"
    )
    lines.append(
        "- Plastic severity threshold: "
        f"`{result.get('plastic_max_nonlinear_severity_threshold') if result.get('plastic_max_nonlinear_severity_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Plastic breach rate: "
        f"`{result.get('plastic_breach_rate') if result.get('plastic_breach_rate') is not None else '-'}`"
    )
    lines.append(
        "- Plastic load realization min/max: "
        f"`{result.get('plastic_load_realization_ratio_min') if result.get('plastic_load_realization_ratio_min') is not None else '-'}`/`{result.get('plastic_load_realization_ratio_max') if result.get('plastic_load_realization_ratio_max') is not None else '-'}`"
    )
    lines.append(
        "- Plastic load realization thresholds (min/max): "
        f"`{result.get('plastic_min_load_realization_ratio_threshold') if result.get('plastic_min_load_realization_ratio_threshold') is not None else '-'}`/`{result.get('plastic_max_load_realization_ratio_threshold') if result.get('plastic_max_load_realization_ratio_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Plastic load amplification min/max: "
        f"`{result.get('plastic_load_amplification_ratio_min') if result.get('plastic_load_amplification_ratio_min') is not None else '-'}`/`{result.get('plastic_load_amplification_ratio_max') if result.get('plastic_load_amplification_ratio_max') is not None else '-'}`"
    )
    lines.append(
        "- Plastic load amplification thresholds (min/max): "
        f"`{result.get('plastic_min_load_amplification_ratio_threshold') if result.get('plastic_min_load_amplification_ratio_threshold') is not None else '-'}`/`{result.get('plastic_max_load_amplification_ratio_threshold') if result.get('plastic_max_load_amplification_ratio_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Plastic load realization breach rate: "
        f"`{result.get('plastic_load_realization_breach_rate') if result.get('plastic_load_realization_breach_rate') is not None else '-'}`"
    )
    lines.append(
        "- Plastic load amplification breach rate: "
        f"`{result.get('plastic_load_amplification_breach_rate') if result.get('plastic_load_amplification_breach_rate') is not None else '-'}`"
    )
    lines.append(
        "- Plastic trend ratio: "
        f"`{result.get('plastic_trend_ratio') if result.get('plastic_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Plastic load realization drop trend ratio: "
        f"`{result.get('plastic_load_realization_drop_trend_ratio') if result.get('plastic_load_realization_drop_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Plastic load amplification trend ratio: "
        f"`{result.get('plastic_load_amplification_trend_ratio') if result.get('plastic_load_amplification_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Plastic reference trend ratio: "
        f"`{result.get('plastic_reference_trend_ratio') if result.get('plastic_reference_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Plastic reference trend threshold: "
        f"`{result.get('plastic_reference_max_trend_ratio_threshold') if result.get('plastic_reference_max_trend_ratio_threshold') is not None else '-'}`"
    )
    lines.append("")
    lines.append("### Contact Posture")
    lines.append(
        "- Max contact nonlinear severity: "
        f"`{result.get('contact_max_nonlinear_severity') if result.get('contact_max_nonlinear_severity') is not None else '-'}`"
    )
    lines.append(
        "- Contact severity threshold: "
        f"`{result.get('contact_max_nonlinear_severity_threshold') if result.get('contact_max_nonlinear_severity_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Contact breach rate: "
        f"`{result.get('contact_breach_rate') if result.get('contact_breach_rate') is not None else '-'}`"
    )
    lines.append(
        "- Contact load realization min/max: "
        f"`{result.get('contact_load_realization_ratio_min') if result.get('contact_load_realization_ratio_min') is not None else '-'}`/`{result.get('contact_load_realization_ratio_max') if result.get('contact_load_realization_ratio_max') is not None else '-'}`"
    )
    lines.append(
        "- Contact load realization thresholds (min/max): "
        f"`{result.get('contact_min_load_realization_ratio_threshold') if result.get('contact_min_load_realization_ratio_threshold') is not None else '-'}`/`{result.get('contact_max_load_realization_ratio_threshold') if result.get('contact_max_load_realization_ratio_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Contact load amplification min/max: "
        f"`{result.get('contact_load_amplification_ratio_min') if result.get('contact_load_amplification_ratio_min') is not None else '-'}`/`{result.get('contact_load_amplification_ratio_max') if result.get('contact_load_amplification_ratio_max') is not None else '-'}`"
    )
    lines.append(
        "- Contact load amplification thresholds (min/max): "
        f"`{result.get('contact_min_load_amplification_ratio_threshold') if result.get('contact_min_load_amplification_ratio_threshold') is not None else '-'}`/`{result.get('contact_max_load_amplification_ratio_threshold') if result.get('contact_max_load_amplification_ratio_threshold') is not None else '-'}`"
    )
    lines.append(
        "- Contact load realization breach rate: "
        f"`{result.get('contact_load_realization_breach_rate') if result.get('contact_load_realization_breach_rate') is not None else '-'}`"
    )
    lines.append(
        "- Contact load amplification breach rate: "
        f"`{result.get('contact_load_amplification_breach_rate') if result.get('contact_load_amplification_breach_rate') is not None else '-'}`"
    )
    lines.append(
        "- Contact trend ratio: "
        f"`{result.get('contact_trend_ratio') if result.get('contact_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Contact load realization drop trend ratio: "
        f"`{result.get('contact_load_realization_drop_trend_ratio') if result.get('contact_load_realization_drop_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Contact load amplification trend ratio: "
        f"`{result.get('contact_load_amplification_trend_ratio') if result.get('contact_load_amplification_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Contact reference trend ratio: "
        f"`{result.get('contact_reference_trend_ratio') if result.get('contact_reference_trend_ratio') is not None else '-'}`"
    )
    lines.append(
        "- Contact reference trend threshold: "
        f"`{result.get('contact_reference_max_trend_ratio_threshold') if result.get('contact_reference_max_trend_ratio_threshold') is not None else '-'}`"
    )
    lines.append("")
    lines.append("### Promotion Readiness")
    lines.append(
        "- Promotion-ready required: "
        f"`{result.get('require_promotion_ready') if result.get('require_promotion_ready') is not None else '-'}`"
    )
    lines.append(
        "- Plastic promotion ready: "
        f"`{result.get('plastic_promotion_ready') if result.get('plastic_promotion_ready') is not None else '-'}`"
    )
    lines.append(
        "- Plastic promotion samples/min: "
        f"`{result.get('plastic_promotion_sample_count') if result.get('plastic_promotion_sample_count') is not None else '-'}`/`{result.get('plastic_promotion_min_samples') if result.get('plastic_promotion_min_samples') is not None else '-'}`"
    )
    lines.append(
        "- Plastic promotion blockers: "
        f"`{', '.join(result.get('plastic_promotion_blockers', [])) if result.get('plastic_promotion_blockers') else '-'}`"
    )
    lines.append(
        "- Plastic blocker count/max: "
        f"`{result.get('plastic_promotion_blocker_count') if result.get('plastic_promotion_blocker_count') is not None else '-'}`/`{result.get('plastic_promotion_max_blockers') if result.get('plastic_promotion_max_blockers') is not None else '-'}`"
    )
    lines.append(
        "- Plastic blocker regression/allowed: "
        f"`{result.get('plastic_promotion_blocker_regression') if result.get('plastic_promotion_blocker_regression') is not None else '-'}`/`{result.get('promotion_max_blocker_regression') if result.get('promotion_max_blocker_regression') is not None else '-'}`"
    )
    lines.append(
        "- Contact promotion ready: "
        f"`{result.get('contact_promotion_ready') if result.get('contact_promotion_ready') is not None else '-'}`"
    )
    lines.append(
        "- Contact promotion samples/min: "
        f"`{result.get('contact_promotion_sample_count') if result.get('contact_promotion_sample_count') is not None else '-'}`/`{result.get('contact_promotion_min_samples') if result.get('contact_promotion_min_samples') is not None else '-'}`"
    )
    lines.append(
        "- Contact promotion blockers: "
        f"`{', '.join(result.get('contact_promotion_blockers', [])) if result.get('contact_promotion_blockers') else '-'}`"
    )
    lines.append(
        "- Contact blocker count/max: "
        f"`{result.get('contact_promotion_blocker_count') if result.get('contact_promotion_blocker_count') is not None else '-'}`/`{result.get('contact_promotion_max_blockers') if result.get('contact_promotion_max_blockers') is not None else '-'}`"
    )
    lines.append(
        "- Contact blocker regression/allowed: "
        f"`{result.get('contact_promotion_blocker_regression') if result.get('contact_promotion_blocker_regression') is not None else '-'}`/`{result.get('promotion_max_blocker_regression') if result.get('promotion_max_blocker_regression') is not None else '-'}`"
    )
    lines.append("")
    lines.append("### Promotion Evidence Quality")
    lines.append(
        "- Promotion calibration applied/required: "
        f"`{result.get('promotion_calibration_applied') if result.get('promotion_calibration_applied') is not None else '-'}`/`{result.get('require_promotion_calibration') if result.get('require_promotion_calibration') is not None else '-'}`"
    )
    lines.append(
        "- Promotion calibration age/max days: "
        f"`{result.get('promotion_calibration_age_days') if result.get('promotion_calibration_age_days') is not None else '-'}`/`{result.get('promotion_calibration_max_age_days') if result.get('promotion_calibration_max_age_days') is not None else '-'}`"
    )
    lines.append(
        "- Promotion history sufficient (rolling/min): "
        f"`{result.get('promotion_history_sufficient') if result.get('promotion_history_sufficient') is not None else '-'}` (`{result.get('rolling_report_count') if result.get('rolling_report_count') is not None else '-'}`/`{result.get('promotion_min_rolling_reports') if result.get('promotion_min_rolling_reports') is not None else '-'}`)"
    )
    lines.append("")
    if result["reasons"]:
        lines.append("### Reasons")
        for reason in result["reasons"]:
            lines.append(
                f"- `{reason['severity']}` `{reason['code']}`: {reason['detail']}"
            )
    else:
        lines.append("- No release blockers or warnings")
    return "\n".join(lines)


def main() -> int:
    latest_path = Path(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_REPORT_PATH",
            "target/runmat-analysis-artifacts/analysis_benchmark_report.json",
        )
    )
    rolling_dir = Path(
        os.getenv("RUNMAT_ANALYSIS_BASELINE_DIR", "target/runmat-analysis-artifacts/rolling")
    )
    prep_root = Path(
        os.getenv("RUNMAT_GEOMETRY_PREP_ARTIFACT_ROOT", "target/runmat-prep-artifacts")
    )
    calibration_evidence_path = Path(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_EVIDENCE",
            "scripts/analysis/prep_calibration/evidence/prep_calibration_evidence.json",
        )
    )
    recommendation_artifact_path = Path(
        os.getenv(
            "RUNMAT_PREP_CALIBRATION_RECOMMENDATIONS_INPUT",
            "target/runmat-analysis-artifacts/prep_calibration_recommendations.json",
        )
    )
    thermo_promotion_report_path = Path(
        os.getenv(
            "RUNMAT_THERMO_FIELD_PROMOTION_REPORT",
            "target/runmat-analysis-artifacts/thermo_field_promotion_report.json",
        )
    )
    promotion_calibration_path = Path(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PROMOTION_CALIBRATION",
            "target/runmat-analysis-artifacts/promotion_threshold_calibration.json",
        )
    )
    output_path = Path(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_OUTPUT",
            "target/runmat-analysis-artifacts/nonlinear_release_readiness.json",
        )
    )

    latest = load_json(latest_path)
    if not isinstance(latest, dict):
        print("release readiness failed: latest conformance report missing or invalid")
        return 1

    prep_health = prep_health_from_artifacts(prep_artifacts(prep_root))
    result = evaluate_release_readiness(
        latest,
        rolling_reports(rolling_dir),
        is_protected_branch(),
        prep_health=prep_health,
        calibration_evidence=load_evidence(calibration_evidence_path),
        recommendation_artifact=load_json(recommendation_artifact_path),
        thermo_promotion_report=load_json(thermo_promotion_report_path),
        promotion_calibration=load_json(promotion_calibration_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(markdown_summary(result))
    return 1 if result["verdict"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
