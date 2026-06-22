use super::harness::with_harness_provider;
use super::manifest::default_options;
use super::*;
use runmat_analysis_core::AnalysisField;
use runmat_analysis_fea::{
    fea_fsi_fluid_velocity_field_id, fea_thermal_temperature_field_id,
    FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE, FEA_FIELD_CFD_VELOCITY, FEA_FIELD_CHT_FLUID_VELOCITY,
    FEA_FIELD_EM_VECTOR_POTENTIAL_REAL, FEA_FIELD_STRUCTURAL_DISPLACEMENT,
    FEA_FIELD_STRUCTURAL_STRESS, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
    FEA_FIELD_STRUCTURAL_VON_MISES,
};
use runmat_runtime::analysis::{
    AnalysisRunResult, ContactInterfaceOptions, ElectroRegionConductivityScale,
    ElectroThermalCouplingOptions, ElectroTimeProfilePoint, PlasticityConstitutiveOptions,
    ThermoMechanicalCouplingOptions, ThermoRegionTemperatureDelta, ThermoTimeProfilePoint,
};
use sha2::{Digest, Sha256};

fn thermo_field_payload_hash_for_value(payload: &serde_json::Value) -> String {
    let schema = payload
        .get("schema_version")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let geometry_id = payload
        .get("source_geometry_id")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let geometry_revision = payload
        .get("source_geometry_revision")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let source = payload
        .get("field_source")
        .and_then(|value| value.as_object())
        .cloned()
        .unwrap_or_default();
    let source_id = source
        .get("source_id")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let source_revision = source
        .get("revision")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let interpolation = source
        .get("interpolation_mode")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let expected_regions = source
        .get("expected_region_ids")
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str())
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    let region_terms = payload
        .get("region_temperature_deltas")
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .map(|entry| {
                    let region_id = entry
                        .get("region_id")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default();
                    let delta = entry
                        .get("temperature_delta_k")
                        .and_then(|value| value.as_f64())
                        .unwrap_or(0.0);
                    format!("{}:{:016x}", region_id, delta.to_bits())
                })
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    let time_terms = payload
        .get("time_profile")
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .map(|entry| {
                    let t = entry
                        .get("normalized_time")
                        .and_then(|value| value.as_f64())
                        .unwrap_or(0.0)
                        .to_bits();
                    let s = entry
                        .get("scale")
                        .and_then(|value| value.as_f64())
                        .unwrap_or(0.0)
                        .to_bits();
                    format!("{:016x}:{:016x}", t, s)
                })
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    let canonical = format!(
        "{}|{}|{}|{}|{}|{}|{}|{}|{}",
        schema,
        geometry_id,
        geometry_revision,
        source_id,
        source_revision,
        interpolation,
        expected_regions,
        region_terms,
        time_terms,
    );
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

fn thermo_field_signature(payload_hash: &str, approved_by: &str) -> String {
    let signing_key = std::env::var("RUNMAT_THERMO_FIELD_SIGNING_KEY")
        .unwrap_or_else(|_| "runmat-dev-thermo-signing-key".to_string());
    let mut hasher = Sha256::new();
    hasher.update(format!("{payload_hash}:{approved_by}:{signing_key}").as_bytes());
    format!("sigv1:sha256:{:x}", hasher.finalize())
}

fn primary_result_field_id(run_kind: AnalysisRunKind) -> String {
    match run_kind {
        AnalysisRunKind::Thermal => fea_thermal_temperature_field_id(0),
        AnalysisRunKind::Acoustic => FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE.to_string(),
        AnalysisRunKind::Electromagnetic => FEA_FIELD_EM_VECTOR_POTENTIAL_REAL.to_string(),
        AnalysisRunKind::Cfd => FEA_FIELD_CFD_VELOCITY.to_string(),
        AnalysisRunKind::Cht => FEA_FIELD_CHT_FLUID_VELOCITY.to_string(),
        AnalysisRunKind::Fsi => fea_fsi_fluid_velocity_field_id(0),
        AnalysisRunKind::LinearStatic
        | AnalysisRunKind::Modal
        | AnalysisRunKind::Transient
        | AnalysisRunKind::Nonlinear => FEA_FIELD_STRUCTURAL_DISPLACEMENT.to_string(),
    }
}

fn analysis_result_field<'a>(
    run_result: &'a AnalysisRunResult,
    field_id: &str,
) -> Option<&'a AnalysisField> {
    run_result
        .run
        .field(field_id)
        .or_else(|| {
            run_result.modal_results.as_ref().and_then(|modal| {
                modal
                    .mode_shapes
                    .iter()
                    .find(|field| field.field_id == field_id)
            })
        })
        .or_else(|| {
            run_result.thermal_results.as_ref().and_then(|thermal| {
                thermal
                    .temperature_snapshots
                    .iter()
                    .find(|field| field.field_id == field_id)
            })
        })
        .or_else(|| {
            run_result.transient_results.as_ref().and_then(|transient| {
                transient
                    .displacement_snapshots
                    .iter()
                    .find(|field| field.field_id == field_id)
            })
        })
        .or_else(|| {
            run_result.nonlinear_results.as_ref().and_then(|nonlinear| {
                nonlinear
                    .displacement_snapshots
                    .iter()
                    .find(|field| field.field_id == field_id)
            })
        })
        .or_else(|| {
            run_result
                .electromagnetic_results
                .as_ref()
                .and_then(|electromagnetic| {
                    [
                        &electromagnetic.vector_potential_real,
                        &electromagnetic.magnetic_flux_density_magnitude,
                    ]
                    .into_iter()
                    .find(|field| field.field_id == field_id)
                })
        })
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
}

fn env_bool(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

fn nonlinear_options_for_spec(spec: &FixtureSpec) -> AnalysisNonlinearRunOptions {
    let mut options = AnalysisNonlinearRunOptions::production_recommended();
    options.increment_count = spec.transient_step_count.unwrap_or(options.increment_count);

    if let Some(value) = env_usize("RUNMAT_NONLINEAR_INCREMENT_COUNT") {
        options.increment_count = value.max(1);
    }
    if let Some(value) = env_usize("RUNMAT_NONLINEAR_MAX_NEWTON_ITERS") {
        options.max_newton_iters = value.max(1);
    }
    if let Some(value) = env_f64("RUNMAT_NONLINEAR_TOLERANCE") {
        if value.is_finite() && value > 0.0 {
            options.tolerance = value;
        }
    }
    if let Some(value) = env_f64("RUNMAT_NONLINEAR_RESIDUAL_FACTOR") {
        if value.is_finite() && value >= 1.0 {
            options.residual_convergence_factor = value;
        }
    }
    if let Some(value) = env_f64("RUNMAT_NONLINEAR_INCREMENT_NORM_TOL") {
        if value.is_finite() && value > 0.0 {
            options.increment_norm_tolerance = value;
        }
    }
    if let Some(value) = env_bool("RUNMAT_NONLINEAR_LINE_SEARCH") {
        options.line_search = value;
    }
    if let Some(value) = env_usize("RUNMAT_NONLINEAR_MAX_BACKTRACKS") {
        options.max_line_search_backtracks = value;
    }
    if let Some(value) = env_f64("RUNMAT_NONLINEAR_LINE_SEARCH_REDUCTION") {
        if value.is_finite() && value > 0.0 && value < 1.0 {
            options.line_search_reduction = value;
        }
    }
    if let Some(value) = env_usize("RUNMAT_NONLINEAR_TANGENT_REFRESH_INTERVAL") {
        options.tangent_refresh_interval = value.max(1);
    }

    options
}

fn plasticity_for_fixture(spec_id: &str) -> Option<PlasticityConstitutiveOptions> {
    match spec_id {
        "nonlinear_plasticity_benchmark_gpu_provider" => Some(PlasticityConstitutiveOptions {
            enabled: true,
            yield_strain: 2.0e-4,
            hardening_modulus_ratio: 0.2,
            saturation_exponent: 4.0,
        }),
        "nonlinear_invalid_plasticity_options" => Some(PlasticityConstitutiveOptions {
            enabled: true,
            yield_strain: -1.0,
            hardening_modulus_ratio: 0.2,
            saturation_exponent: 4.0,
        }),
        "nonlinear_plastic_hardening_reference_gpu_provider" => {
            Some(PlasticityConstitutiveOptions {
                enabled: true,
                yield_strain: 0.03,
                hardening_modulus_ratio: 0.06,
                saturation_exponent: 1.0,
            })
        }
        "nonlinear_plastic_hardening_reference_complex_gpu_provider" => {
            Some(PlasticityConstitutiveOptions {
                enabled: true,
                yield_strain: 0.018,
                hardening_modulus_ratio: 0.09,
                saturation_exponent: 1.25,
            })
        }
        _ => None,
    }
}

fn contact_for_fixture(spec_id: &str) -> Option<ContactInterfaceOptions> {
    match spec_id {
        "nonlinear_contact_benchmark_gpu_provider" => Some(ContactInterfaceOptions {
            enabled: true,
            penalty_stiffness_scale: 0.15,
            max_penetration_ratio: 0.035,
            friction_coefficient: 0.9,
        }),
        "nonlinear_invalid_contact_options" => Some(ContactInterfaceOptions {
            enabled: true,
            penalty_stiffness_scale: 0.0,
            max_penetration_ratio: 0.01,
            friction_coefficient: 0.0,
        }),
        "nonlinear_contact_frictionless_reference_gpu_provider" => Some(ContactInterfaceOptions {
            enabled: true,
            penalty_stiffness_scale: 2.0,
            max_penetration_ratio: 0.01,
            friction_coefficient: 0.0,
        }),
        "nonlinear_contact_frictionless_reference_complex_gpu_provider" => {
            Some(ContactInterfaceOptions {
                enabled: true,
                penalty_stiffness_scale: 1.2,
                max_penetration_ratio: 0.014,
                friction_coefficient: 0.0,
            })
        }
        _ => None,
    }
}

fn thermo_coupling_for_fixture(spec_id: &str) -> Option<ThermoMechanicalCouplingOptions> {
    match spec_id {
        "thermo_mech_kickoff_gpu_provider" => Some(ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 65.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: vec![
                ThermoRegionTemperatureDelta {
                    region_id: "tip_steel".to_string(),
                    temperature_delta_k: 75.0,
                },
                ThermoRegionTemperatureDelta {
                    region_id: "mid_aluminum".to_string(),
                    temperature_delta_k: 55.0,
                },
            ],
            time_profile: vec![
                ThermoTimeProfilePoint {
                    normalized_time: 0.0,
                    scale: 0.6,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 1.0,
                    scale: 1.0,
                },
            ],
        }),
        "thermo_gradient_benign_gpu_provider" => Some(ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 55.0,
            thermal_expansion_coefficient: 1.0e-5,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: vec![
                ThermoRegionTemperatureDelta {
                    region_id: "tip_steel".to_string(),
                    temperature_delta_k: 60.0,
                },
                ThermoRegionTemperatureDelta {
                    region_id: "mid_aluminum".to_string(),
                    temperature_delta_k: 50.0,
                },
            ],
            time_profile: vec![
                ThermoTimeProfilePoint {
                    normalized_time: 0.0,
                    scale: 0.7,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 1.0,
                    scale: 1.0,
                },
            ],
        }),
        "thermo_gradient_pathological_gpu_provider" => Some(ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 220.0,
            thermal_expansion_coefficient: 2.5e-5,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: vec![
                ThermoRegionTemperatureDelta {
                    region_id: "tip_steel".to_string(),
                    temperature_delta_k: 260.0,
                },
                ThermoRegionTemperatureDelta {
                    region_id: "polymer_segment".to_string(),
                    temperature_delta_k: 120.0,
                },
            ],
            time_profile: vec![
                ThermoTimeProfilePoint {
                    normalized_time: 0.0,
                    scale: 0.2,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 0.4,
                    scale: 1.3,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 1.0,
                    scale: 0.9,
                },
            ],
        }),
        spec_id
            if spec_id == "thermo_ramp_smooth_gpu_provider"
                || spec_id.starts_with("thermal_standalone_ramp_") =>
        {
            Some(ThermoMechanicalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 70.0,
                thermal_expansion_coefficient: 1.1e-5,
                field_artifact_id: None,
                field_source: None,
                region_temperature_deltas: vec![
                    ThermoRegionTemperatureDelta {
                        region_id: "tip_steel".to_string(),
                        temperature_delta_k: 72.0,
                    },
                    ThermoRegionTemperatureDelta {
                        region_id: "mid_aluminum".to_string(),
                        temperature_delta_k: 68.0,
                    },
                ],
                time_profile: vec![
                    ThermoTimeProfilePoint {
                        normalized_time: 0.0,
                        scale: 0.3,
                    },
                    ThermoTimeProfilePoint {
                        normalized_time: 0.5,
                        scale: 0.7,
                    },
                    ThermoTimeProfilePoint {
                        normalized_time: 1.0,
                        scale: 1.0,
                    },
                ],
            })
        }
        "thermo_ramp_smooth_field_artifact_gpu_provider" => Some(ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 70.0,
            thermal_expansion_coefficient: 1.1e-5,
            field_artifact_id: Some("thermo_ramp_smooth_approved".to_string()),
            field_source: None,
            region_temperature_deltas: Vec::new(),
            time_profile: Vec::new(),
        }),
        "thermo_shock_oscillatory_gpu_provider" => Some(ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 140.0,
            thermal_expansion_coefficient: 2.0e-5,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: vec![
                ThermoRegionTemperatureDelta {
                    region_id: "tip_steel".to_string(),
                    temperature_delta_k: 210.0,
                },
                ThermoRegionTemperatureDelta {
                    region_id: "polymer_segment".to_string(),
                    temperature_delta_k: 90.0,
                },
            ],
            time_profile: vec![
                ThermoTimeProfilePoint {
                    normalized_time: 0.0,
                    scale: 0.4,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 0.25,
                    scale: 1.4,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 0.5,
                    scale: 0.5,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 0.75,
                    scale: 1.3,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 1.0,
                    scale: 0.6,
                },
            ],
        }),
        "thermo_shock_oscillatory_field_artifact_gpu_provider" => {
            Some(ThermoMechanicalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 140.0,
                thermal_expansion_coefficient: 2.0e-5,
                field_artifact_id: Some("thermo_shock_oscillatory_approved".to_string()),
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            })
        }
        _ => None,
    }
}

fn electro_coupling_for_fixture(spec_id: &str) -> Option<ElectroThermalCouplingOptions> {
    match spec_id {
        "electro_thermal_joule_benign_gpu_provider" => Some(ElectroThermalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_voltage_v: 36.0,
            base_electrical_conductivity_s_per_m: 3.8e7,
            resistive_heating_coefficient: 3.5e-4,
            region_conductivity_scales: vec![
                ElectroRegionConductivityScale {
                    region_id: "tip_steel".to_string(),
                    conductivity_scale: 1.05,
                },
                ElectroRegionConductivityScale {
                    region_id: "mid_aluminum".to_string(),
                    conductivity_scale: 0.98,
                },
            ],
            time_profile: vec![
                ElectroTimeProfilePoint {
                    normalized_time: 0.0,
                    current_scale: 0.7,
                },
                ElectroTimeProfilePoint {
                    normalized_time: 1.0,
                    current_scale: 1.0,
                },
            ],
        }),
        "electro_thermal_joule_pathological_gpu_provider" => Some(ElectroThermalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_voltage_v: 180.0,
            base_electrical_conductivity_s_per_m: 2.0e7,
            resistive_heating_coefficient: 9.5e-4,
            region_conductivity_scales: vec![
                ElectroRegionConductivityScale {
                    region_id: "shock_region_0".to_string(),
                    conductivity_scale: 1.8,
                },
                ElectroRegionConductivityScale {
                    region_id: "shock_region_1".to_string(),
                    conductivity_scale: 0.55,
                },
            ],
            time_profile: vec![
                ElectroTimeProfilePoint {
                    normalized_time: 0.0,
                    current_scale: 0.45,
                },
                ElectroTimeProfilePoint {
                    normalized_time: 0.3,
                    current_scale: 1.45,
                },
                ElectroTimeProfilePoint {
                    normalized_time: 0.6,
                    current_scale: 0.6,
                },
                ElectroTimeProfilePoint {
                    normalized_time: 1.0,
                    current_scale: 1.25,
                },
            ],
        }),
        "electro_thermal_invalid_voltage" => Some(ElectroThermalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_voltage_v: f64::NAN,
            base_electrical_conductivity_s_per_m: 3.8e7,
            resistive_heating_coefficient: 3.5e-4,
            region_conductivity_scales: Vec::new(),
            time_profile: Vec::new(),
        }),
        "electro_thermal_invalid_conductivity_scale" => Some(ElectroThermalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_voltage_v: 36.0,
            base_electrical_conductivity_s_per_m: 3.8e7,
            resistive_heating_coefficient: 3.5e-4,
            region_conductivity_scales: vec![ElectroRegionConductivityScale {
                region_id: "tip_steel".to_string(),
                conductivity_scale: 0.0,
            }],
            time_profile: Vec::new(),
        }),
        "electro_thermal_unmapped_region" => Some(ElectroThermalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_voltage_v: 36.0,
            base_electrical_conductivity_s_per_m: 3.8e7,
            resistive_heating_coefficient: 3.5e-4,
            region_conductivity_scales: vec![ElectroRegionConductivityScale {
                region_id: "electro_region_not_mapped".to_string(),
                conductivity_scale: 1.0,
            }],
            time_profile: Vec::new(),
        }),
        "nonlinear_load_path_mix_gpu_provider" => Some(ElectroThermalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_voltage_v: 82.0,
            base_electrical_conductivity_s_per_m: 2.6e7,
            resistive_heating_coefficient: 6.0e-4,
            region_conductivity_scales: vec![
                ElectroRegionConductivityScale {
                    region_id: "softening_material_region_0".to_string(),
                    conductivity_scale: 1.2,
                },
                ElectroRegionConductivityScale {
                    region_id: "softening_material_region_1".to_string(),
                    conductivity_scale: 0.85,
                },
            ],
            time_profile: vec![
                ElectroTimeProfilePoint {
                    normalized_time: 0.0,
                    current_scale: 0.55,
                },
                ElectroTimeProfilePoint {
                    normalized_time: 0.5,
                    current_scale: 1.2,
                },
                ElectroTimeProfilePoint {
                    normalized_time: 1.0,
                    current_scale: 0.9,
                },
            ],
        }),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
enum ElectromagneticFixtureKind {
    Homogeneous,
    Heterogeneous,
    SparseAssignments,
    MultiRegionAssignments,
    OverlapInterference,
    BoundaryKernel,
    BoundaryPenaltyStress,
    MultiRegionPhasedSource,
}

#[derive(Debug, Clone, Copy)]
struct ElectromagneticFixtureProfile {
    reference_frequency_hz: f64,
    applied_current_a: f64,
    kind: ElectromagneticFixtureKind,
}

fn electromagnetic_profile_for_fixture(spec_id: &str) -> Option<ElectromagneticFixtureProfile> {
    match spec_id {
        "electromagnetic_reference_homogeneous_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 60.0,
                applied_current_a: 120.0,
                kind: ElectromagneticFixtureKind::Homogeneous,
            })
        }
        "electromagnetic_missing_material"
        | "electromagnetic_missing_source"
        | "electromagnetic_missing_boundary" => Some(ElectromagneticFixtureProfile {
            reference_frequency_hz: 60.0,
            applied_current_a: 120.0,
            kind: ElectromagneticFixtureKind::Homogeneous,
        }),
        "electromagnetic_reference_heterogeneous_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 400.0,
                applied_current_a: 250.0,
                kind: ElectromagneticFixtureKind::Heterogeneous,
            })
        }
        "electromagnetic_reference_sparse_assignments_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 180.0,
                applied_current_a: 170.0,
                kind: ElectromagneticFixtureKind::SparseAssignments,
            })
        }
        "electromagnetic_reference_multiregion_assignments_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 260.0,
                applied_current_a: 210.0,
                kind: ElectromagneticFixtureKind::MultiRegionAssignments,
            })
        }
        "electromagnetic_reference_overlap_interference_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 320.0,
                applied_current_a: 240.0,
                kind: ElectromagneticFixtureKind::OverlapInterference,
            })
        }
        "electromagnetic_reference_boundary_kernel_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 520.0,
                applied_current_a: 180.0,
                kind: ElectromagneticFixtureKind::BoundaryKernel,
            })
        }
        "electromagnetic_reference_boundary_penalty_stress_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 650.0,
                applied_current_a: 210.0,
                kind: ElectromagneticFixtureKind::BoundaryPenaltyStress,
            })
        }
        "electromagnetic_reference_multi_region_phased_source_gpu_provider" => {
            Some(ElectromagneticFixtureProfile {
                reference_frequency_hz: 460.0,
                applied_current_a: 260.0,
                kind: ElectromagneticFixtureKind::MultiRegionPhasedSource,
            })
        }
        _ => None,
    }
}

fn electromagnetic_sweep_frequency_hz_for_fixture(spec_id: &str) -> Vec<f64> {
    let Some(profile) = electromagnetic_profile_for_fixture(spec_id) else {
        return Vec::new();
    };
    let f0 = profile.reference_frequency_hz.max(1.0);
    vec![f0 * 0.75, f0 * 0.9, f0, f0 * 1.1, f0 * 1.25]
}

fn authored_cfd_boundaries_for_fixture(
    spec_id: &str,
    inlet_velocity_m_per_s: f64,
) -> Vec<runmat_analysis_core::BoundaryCondition> {
    vec![
        runmat_analysis_core::BoundaryCondition {
            bc_id: format!("bc_cfd_inlet_{}", spec_id),
            region_id: "fluid_inlet".to_string(),
            kind: runmat_analysis_core::BoundaryConditionKind::CfdInletVelocity {
                velocity_m_per_s: inlet_velocity_m_per_s,
            },
        },
        runmat_analysis_core::BoundaryCondition {
            bc_id: format!("bc_cfd_outlet_{}", spec_id),
            region_id: "fluid_outlet".to_string(),
            kind: runmat_analysis_core::BoundaryConditionKind::CfdOutletPressure {
                pressure_pa: 0.0,
            },
        },
        runmat_analysis_core::BoundaryCondition {
            bc_id: format!("bc_cfd_wall_upper_{}", spec_id),
            region_id: "fluid_wall_upper".to_string(),
            kind: runmat_analysis_core::BoundaryConditionKind::CfdNoSlipWall,
        },
        runmat_analysis_core::BoundaryCondition {
            bc_id: format!("bc_cfd_wall_lower_{}", spec_id),
            region_id: "fluid_wall_lower".to_string(),
            kind: runmat_analysis_core::BoundaryConditionKind::CfdNoSlipWall,
        },
    ]
}

fn configure_model_for_fixture(spec_id: &str, model: &mut AnalysisModel) {
    if spec_id.starts_with("acoustic_harmonic_") {
        model.steps = vec![runmat_analysis_core::AnalysisStep {
            step_id: format!("step_acoustic_{}", spec_id),
            kind: runmat_analysis_core::AnalysisStepKind::Modal,
        }];
        for material in &mut model.materials {
            material.acoustic = Some(runmat_analysis_core::MaterialAcousticModel {
                density_kg_per_m3: 1.225,
                speed_of_sound_m_per_s: 343.0,
                damping_ratio: 0.024,
            });
        }
        if model.materials.is_empty() {
            model.materials.push(runmat_analysis_core::MaterialModel {
                material_id: "mat_acoustic_air".to_string(),
                name: "Acoustic Air".to_string(),
                mechanical: runmat_analysis_core::MaterialMechanicalModel {
                    youngs_modulus_pa: 1.0,
                    poisson_ratio: 0.0,
                },
                thermal: runmat_analysis_core::MaterialThermalModel::default(),
                acoustic: Some(runmat_analysis_core::MaterialAcousticModel {
                    density_kg_per_m3: 1.225,
                    speed_of_sound_m_per_s: 343.0,
                    damping_ratio: 0.024,
                }),
                electrical: None,
                plastic: None,
            });
        }
        model.boundary_conditions = vec![
            runmat_analysis_core::BoundaryCondition {
                bc_id: format!("bc_acoustic_rigid_{}", spec_id),
                region_id: "acoustic_wall".to_string(),
                kind: runmat_analysis_core::BoundaryConditionKind::AcousticRigidWall,
            },
            runmat_analysis_core::BoundaryCondition {
                bc_id: format!("bc_acoustic_radiation_{}", spec_id),
                region_id: "acoustic_open".to_string(),
                kind: runmat_analysis_core::BoundaryConditionKind::AcousticRadiation,
            },
            runmat_analysis_core::BoundaryCondition {
                bc_id: format!("bc_acoustic_impedance_{}", spec_id),
                region_id: "acoustic_liner".to_string(),
                kind: runmat_analysis_core::BoundaryConditionKind::AcousticImpedance {
                    specific_impedance_pa_s_per_m: 420.0,
                },
            },
        ];
        model.loads = vec![runmat_analysis_core::LoadCase {
            load_id: format!("load_acoustic_pressure_{}", spec_id),
            region_id: "acoustic_source".to_string(),
            kind: runmat_analysis_core::LoadKind::Pressure { magnitude_pa: 1.0 },
        }];
        match spec_id {
            "acoustic_harmonic_missing_source" => {
                model.loads = vec![runmat_analysis_core::LoadCase {
                    load_id: format!("load_acoustic_invalid_force_{}", spec_id),
                    region_id: "acoustic_source".to_string(),
                    kind: runmat_analysis_core::LoadKind::Force {
                        fx: 1.0,
                        fy: 0.0,
                        fz: 0.0,
                    },
                }];
            }
            "acoustic_harmonic_missing_boundary" => {
                model.boundary_conditions = vec![runmat_analysis_core::BoundaryCondition {
                    bc_id: format!("bc_acoustic_invalid_structural_{}", spec_id),
                    region_id: "acoustic_wall".to_string(),
                    kind: runmat_analysis_core::BoundaryConditionKind::Fixed,
                }];
            }
            _ => {}
        }
        model.thermo_mechanical = None;
        model.electro_thermal = None;
        model.interfaces.clear();
        model.cfd = None;
    }
    if spec_id.starts_with("thermal_standalone_ramp_") {
        model.steps = vec![runmat_analysis_core::AnalysisStep {
            step_id: format!("step_thermal_{}", spec_id),
            kind: runmat_analysis_core::AnalysisStepKind::Thermal,
        }];
        model.boundary_conditions = vec![
            runmat_analysis_core::BoundaryCondition {
                bc_id: format!("bc_thermal_temperature_{}", spec_id),
                region_id: "thermal_hot_wall".to_string(),
                kind: runmat_analysis_core::BoundaryConditionKind::ThermalPrescribedTemperature {
                    temperature_k: 343.15,
                },
            },
            runmat_analysis_core::BoundaryCondition {
                bc_id: format!("bc_thermal_flux_{}", spec_id),
                region_id: "thermal_flux_wall".to_string(),
                kind: runmat_analysis_core::BoundaryConditionKind::ThermalHeatFlux {
                    heat_flux_w_per_m2: 4500.0,
                },
            },
            runmat_analysis_core::BoundaryCondition {
                bc_id: format!("bc_thermal_convection_{}", spec_id),
                region_id: "thermal_open_wall".to_string(),
                kind: runmat_analysis_core::BoundaryConditionKind::ThermalConvection {
                    ambient_temperature_k: 308.15,
                    coefficient_w_per_m2k: 35.0,
                },
            },
        ];
        model.loads = vec![runmat_analysis_core::LoadCase {
            load_id: format!("load_thermal_heat_source_{}", spec_id),
            region_id: "thermal_core".to_string(),
            kind: runmat_analysis_core::LoadKind::HeatSource {
                volumetric_w_per_m3: 1.2e6,
            },
        }];
        match spec_id {
            "thermal_standalone_ramp_invalid_material" => {
                if let Some(material) = model.materials.first_mut() {
                    material.thermal.conductivity_w_per_mk = 0.0;
                }
            }
            "thermal_standalone_ramp_invalid_source" => {
                model.loads = vec![runmat_analysis_core::LoadCase {
                    load_id: format!("load_thermal_invalid_heat_source_{}", spec_id),
                    region_id: "thermal_core".to_string(),
                    kind: runmat_analysis_core::LoadKind::HeatSource {
                        volumetric_w_per_m3: f64::INFINITY,
                    },
                }];
            }
            "thermal_standalone_ramp_invalid_boundary" => {
                model.boundary_conditions = vec![runmat_analysis_core::BoundaryCondition {
                    bc_id: format!("bc_thermal_invalid_convection_{}", spec_id),
                    region_id: "thermal_open_wall".to_string(),
                    kind: runmat_analysis_core::BoundaryConditionKind::ThermalConvection {
                        ambient_temperature_k: 293.15,
                        coefficient_w_per_m2k: f64::NAN,
                    },
                }];
            }
            _ => {}
        }
        model.electro_thermal = None;
        model.electromagnetic = None;
        model.interfaces.clear();
        model.cfd = None;
    }
    if spec_id.starts_with("cfd_steady_") {
        model.steps = vec![runmat_analysis_core::AnalysisStep {
            step_id: format!("step_cfd_{}", spec_id),
            kind: runmat_analysis_core::AnalysisStepKind::Cfd,
        }];
        model.thermo_mechanical = None;
        model.electro_thermal = None;
        model.interfaces.clear();
        model.boundary_conditions = authored_cfd_boundaries_for_fixture(spec_id, 5.0);
        model.cfd = Some(runmat_analysis_core::CfdDomain {
            enabled: true,
            solve_family: runmat_analysis_core::CfdSolveFamily::SteadyState,
            reference_density_kg_per_m3: 1.225,
            dynamic_viscosity_pa_s: 1.81e-5,
            inlet_velocity_m_per_s: 5.0,
            turbulence_intensity: 0.06,
            time_profile: Vec::new(),
        });
    }
    if spec_id.starts_with("cfd_transient_") {
        model.steps = vec![runmat_analysis_core::AnalysisStep {
            step_id: format!("step_cfd_{}", spec_id),
            kind: runmat_analysis_core::AnalysisStepKind::Cfd,
        }];
        model.thermo_mechanical = None;
        model.electro_thermal = None;
        model.interfaces.clear();
        model.boundary_conditions = authored_cfd_boundaries_for_fixture(spec_id, 5.0);
        model.cfd = Some(runmat_analysis_core::CfdDomain {
            enabled: true,
            solve_family: runmat_analysis_core::CfdSolveFamily::Transient,
            reference_density_kg_per_m3: 1.225,
            dynamic_viscosity_pa_s: 1.81e-5,
            inlet_velocity_m_per_s: 5.0,
            turbulence_intensity: 0.06,
            time_profile: vec![
                runmat_analysis_core::CfdTimeProfilePoint {
                    normalized_time: 0.0,
                    inlet_scale: 0.65,
                },
                runmat_analysis_core::CfdTimeProfilePoint {
                    normalized_time: 1.0,
                    inlet_scale: 1.0,
                },
            ],
        });
    }
    if spec_id == "cfd_invalid_domain_options" || spec_id == "cfd_invalid_boundary_conditions" {
        model.steps = vec![runmat_analysis_core::AnalysisStep {
            step_id: format!("step_cfd_{}", spec_id),
            kind: runmat_analysis_core::AnalysisStepKind::Cfd,
        }];
        model.thermo_mechanical = None;
        model.electro_thermal = None;
        model.interfaces.clear();
        model.boundary_conditions = if spec_id == "cfd_invalid_boundary_conditions" {
            vec![runmat_analysis_core::BoundaryCondition {
                bc_id: format!("bc_cfd_inlet_only_{}", spec_id),
                region_id: "inlet".to_string(),
                kind: runmat_analysis_core::BoundaryConditionKind::CfdInletVelocity {
                    velocity_m_per_s: 5.0,
                },
            }]
        } else {
            authored_cfd_boundaries_for_fixture(spec_id, 5.0)
        };
        model.cfd = Some(runmat_analysis_core::CfdDomain {
            enabled: true,
            solve_family: runmat_analysis_core::CfdSolveFamily::SteadyState,
            reference_density_kg_per_m3: 1.225,
            dynamic_viscosity_pa_s: if spec_id == "cfd_invalid_domain_options" {
                0.0
            } else {
                1.81e-5
            },
            inlet_velocity_m_per_s: 5.0,
            turbulence_intensity: 0.06,
            time_profile: Vec::new(),
        });
    }
    if spec_id.starts_with("cht_coupled_") {
        model.steps = vec![
            runmat_analysis_core::AnalysisStep {
                step_id: format!("step_cht_flow_{}", spec_id),
                kind: runmat_analysis_core::AnalysisStepKind::Cfd,
            },
            runmat_analysis_core::AnalysisStep {
                step_id: format!("step_cht_thermal_{}", spec_id),
                kind: runmat_analysis_core::AnalysisStepKind::Thermal,
            },
        ];
        model.electro_thermal = None;
        model.interfaces.clear();
        model.cfd = Some(runmat_analysis_core::CfdDomain {
            enabled: true,
            solve_family: runmat_analysis_core::CfdSolveFamily::Transient,
            reference_density_kg_per_m3: 1.225,
            dynamic_viscosity_pa_s: 1.81e-5,
            inlet_velocity_m_per_s: 5.0,
            turbulence_intensity: 0.06,
            time_profile: vec![
                runmat_analysis_core::CfdTimeProfilePoint {
                    normalized_time: 0.0,
                    inlet_scale: 0.8,
                },
                runmat_analysis_core::CfdTimeProfilePoint {
                    normalized_time: 1.0,
                    inlet_scale: 1.0,
                },
            ],
        });
        if spec_id == "cht_coupled_channel_slab_cpu" {
            model.cfd = Some(runmat_analysis_core::CfdDomain {
                enabled: true,
                solve_family: runmat_analysis_core::CfdSolveFamily::Transient,
                reference_density_kg_per_m3: 1.225,
                dynamic_viscosity_pa_s: 1.81e-5,
                inlet_velocity_m_per_s: 4.1,
                turbulence_intensity: 0.052,
                time_profile: vec![
                    runmat_analysis_core::CfdTimeProfilePoint {
                        normalized_time: 0.0,
                        inlet_scale: 0.7,
                    },
                    runmat_analysis_core::CfdTimeProfilePoint {
                        normalized_time: 0.5,
                        inlet_scale: 0.9,
                    },
                    runmat_analysis_core::CfdTimeProfilePoint {
                        normalized_time: 1.0,
                        inlet_scale: 1.0,
                    },
                ],
            });
            model.interfaces = vec![runmat_analysis_core::AnalysisInterface {
                interface_id: format!("cht_channel_slab_interface_{spec_id}"),
                primary_region_id: "fluid_channel".to_string(),
                secondary_region_id: "solid_slab".to_string(),
                kind: runmat_analysis_core::AnalysisInterfaceKind::ConjugateHeatTransfer(
                    runmat_analysis_core::ConjugateHeatTransferInterfaceModel {
                        thermal_conductance_w_per_m2k: 750.0,
                        contact_resistance_m2k_per_w: 0.0,
                        relaxation_factor: 0.5,
                    },
                ),
            }];
        }
        if spec_id == "cht_coupled_invalid_cfd_domain" {
            if let Some(cfd_domain) = model.cfd.as_mut() {
                cfd_domain.dynamic_viscosity_pa_s = 0.0;
            }
        }
        model.thermo_mechanical = Some(runmat_analysis_core::ThermoMechanicalDomain {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 60.0,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: vec![
                runmat_analysis_core::ThermoRegionTemperatureDelta {
                    region_id: "tip".to_string(),
                    temperature_delta_k: 70.0,
                },
                runmat_analysis_core::ThermoRegionTemperatureDelta {
                    region_id: "root".to_string(),
                    temperature_delta_k: 45.0,
                },
            ],
            time_profile: vec![
                runmat_analysis_core::ThermoTimeProfilePoint {
                    normalized_time: 0.0,
                    scale: 0.5,
                },
                runmat_analysis_core::ThermoTimeProfilePoint {
                    normalized_time: 1.0,
                    scale: 1.0,
                },
            ],
        });
        if spec_id == "cht_coupled_invalid_interface_mapping" {
            model.interfaces = vec![runmat_analysis_core::AnalysisInterface {
                interface_id: format!("cht_invalid_contact_{spec_id}"),
                primary_region_id: "fluid_channel".to_string(),
                secondary_region_id: "solid_wall".to_string(),
                kind: runmat_analysis_core::AnalysisInterfaceKind::Contact(
                    runmat_analysis_core::ContactInterfaceModel {
                        penalty_stiffness_scale: 1.0,
                        max_penetration_ratio: 0.01,
                        friction_coefficient: 0.0,
                    },
                ),
            }];
        }
    }
    if spec_id.starts_with("fsi_coupled_") {
        model.steps = vec![
            runmat_analysis_core::AnalysisStep {
                step_id: format!("step_fsi_structure_{}", spec_id),
                kind: runmat_analysis_core::AnalysisStepKind::Transient,
            },
            runmat_analysis_core::AnalysisStep {
                step_id: format!("step_fsi_flow_{}", spec_id),
                kind: runmat_analysis_core::AnalysisStepKind::Cfd,
            },
        ];
        model.thermo_mechanical = None;
        model.electro_thermal = None;
        model.interfaces.clear();
        model.cfd = Some(runmat_analysis_core::CfdDomain {
            enabled: true,
            solve_family: runmat_analysis_core::CfdSolveFamily::Transient,
            reference_density_kg_per_m3: 1.225,
            dynamic_viscosity_pa_s: 1.81e-5,
            inlet_velocity_m_per_s: 4.0,
            turbulence_intensity: 0.06,
            time_profile: vec![
                runmat_analysis_core::CfdTimeProfilePoint {
                    normalized_time: 0.0,
                    inlet_scale: 0.6,
                },
                runmat_analysis_core::CfdTimeProfilePoint {
                    normalized_time: 1.0,
                    inlet_scale: 1.0,
                },
            ],
        });
        if spec_id == "fsi_coupled_pipe_plate_cpu" {
            model.cfd = Some(runmat_analysis_core::CfdDomain {
                enabled: true,
                solve_family: runmat_analysis_core::CfdSolveFamily::Transient,
                reference_density_kg_per_m3: 1.225,
                dynamic_viscosity_pa_s: 1.81e-5,
                inlet_velocity_m_per_s: 4.2,
                turbulence_intensity: 0.055,
                time_profile: vec![
                    runmat_analysis_core::CfdTimeProfilePoint {
                        normalized_time: 0.0,
                        inlet_scale: 0.5,
                    },
                    runmat_analysis_core::CfdTimeProfilePoint {
                        normalized_time: 0.5,
                        inlet_scale: 0.85,
                    },
                    runmat_analysis_core::CfdTimeProfilePoint {
                        normalized_time: 1.0,
                        inlet_scale: 1.0,
                    },
                ],
            });
            model.loads.push(runmat_analysis_core::LoadCase {
                load_id: format!("pipe_plate_pressure_preload_{spec_id}"),
                region_id: "pipe_wall".to_string(),
                kind: runmat_analysis_core::LoadKind::Pressure {
                    magnitude_pa: 2.5e4,
                },
            });
            model.interfaces = vec![runmat_analysis_core::AnalysisInterface {
                interface_id: format!("fsi_pipe_plate_interface_{spec_id}"),
                primary_region_id: "fluid_pipe".to_string(),
                secondary_region_id: "plate_wall".to_string(),
                kind: runmat_analysis_core::AnalysisInterfaceKind::FluidStructure(
                    runmat_analysis_core::FluidStructureInterfaceModel {
                        normal_stiffness_pa_per_m: 8.0e8,
                        damping_ratio: 0.04,
                        relaxation_factor: 0.5,
                    },
                ),
            }];
        }
        if spec_id == "fsi_coupled_invalid_cfd_domain" {
            if let Some(cfd_domain) = model.cfd.as_mut() {
                cfd_domain.dynamic_viscosity_pa_s = 0.0;
            }
        }
        if spec_id == "fsi_coupled_invalid_interface_mapping" {
            model.interfaces = vec![runmat_analysis_core::AnalysisInterface {
                interface_id: format!("fsi_invalid_contact_{spec_id}"),
                primary_region_id: "fluid_channel".to_string(),
                secondary_region_id: "structure_wall".to_string(),
                kind: runmat_analysis_core::AnalysisInterfaceKind::Contact(
                    runmat_analysis_core::ContactInterfaceModel {
                        penalty_stiffness_scale: 1.0,
                        max_penetration_ratio: 0.01,
                        friction_coefficient: 0.0,
                    },
                ),
            }];
        }
    }

    if let Some(profile) = electromagnetic_profile_for_fixture(spec_id) {
        model.steps = vec![runmat_analysis_core::AnalysisStep {
            step_id: format!("step_em_{}", spec_id),
            kind: runmat_analysis_core::AnalysisStepKind::Electromagnetic,
        }];
        model.thermo_mechanical = None;
        model.electro_thermal = None;
        model.interfaces.clear();

        if model.materials.len() < 3 {
            let base = model.materials.first().cloned().unwrap_or_else(|| {
                runmat_analysis_core::MaterialModel {
                    material_id: "mat_default".to_string(),
                    name: "Default".to_string(),
                    mechanical: runmat_analysis_core::MaterialMechanicalModel {
                        youngs_modulus_pa: 110e9,
                        poisson_ratio: 0.31,
                    },
                    thermal: runmat_analysis_core::MaterialThermalModel::default(),
                    acoustic: None,
                    electrical: None,
                    plastic: None,
                }
            });
            model.materials = vec![
                runmat_analysis_core::MaterialModel {
                    material_id: "mat_em_copper".to_string(),
                    name: "EM Copper".to_string(),
                    electrical: Some(runmat_analysis_core::MaterialElectricalModel {
                        reference_temperature_k: 293.15,
                        conductivity_s_per_m: 5.8e7,
                        resistive_heating_coefficient: 0.0039,
                        relative_permittivity: 2.0,
                        relative_permeability: 1.0,
                        conductivity_frequency_response: vec![
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: 50.0,
                                conductivity_scale: 1.05,
                                dispersive_loss_scale: Some(0.02),
                                relative_permittivity_scale: Some(1.01),
                                relative_permeability_scale: Some(1.0),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: 200.0,
                                conductivity_scale: 1.0,
                                dispersive_loss_scale: Some(0.025),
                                relative_permittivity_scale: Some(1.0),
                                relative_permeability_scale: Some(1.0),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: 1_000.0,
                                conductivity_scale: 0.94,
                                dispersive_loss_scale: Some(0.03),
                                relative_permittivity_scale: Some(0.99),
                                relative_permeability_scale: Some(1.0),
                            },
                        ],
                    }),
                    ..base.clone()
                },
                runmat_analysis_core::MaterialModel {
                    material_id: "mat_em_ferrite".to_string(),
                    name: "EM Ferrite".to_string(),
                    electrical: Some(runmat_analysis_core::MaterialElectricalModel {
                        reference_temperature_k: 293.15,
                        conductivity_s_per_m: 8.0e4,
                        resistive_heating_coefficient: 0.0020,
                        relative_permittivity: 14.0,
                        relative_permeability: 90.0,
                        conductivity_frequency_response: vec![
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: 50.0,
                                conductivity_scale: 1.32,
                                dispersive_loss_scale: Some(0.08),
                                relative_permittivity_scale: Some(1.15),
                                relative_permeability_scale: Some(1.35),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: 200.0,
                                conductivity_scale: 1.0,
                                dispersive_loss_scale: Some(0.11),
                                relative_permittivity_scale: Some(1.0),
                                relative_permeability_scale: Some(1.0),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: 1_000.0,
                                conductivity_scale: 0.72,
                                dispersive_loss_scale: Some(0.16),
                                relative_permittivity_scale: Some(0.85),
                                relative_permeability_scale: Some(0.70),
                            },
                        ],
                    }),
                    ..base.clone()
                },
                runmat_analysis_core::MaterialModel {
                    material_id: "mat_em_polymer".to_string(),
                    name: "EM Polymer".to_string(),
                    electrical: Some(runmat_analysis_core::MaterialElectricalModel {
                        reference_temperature_k: 293.15,
                        conductivity_s_per_m: 0.2,
                        resistive_heating_coefficient: 0.0010,
                        relative_permittivity: 3.5,
                        relative_permeability: 1.05,
                        conductivity_frequency_response: vec![
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: 50.0,
                                conductivity_scale: 0.88,
                                dispersive_loss_scale: Some(0.24),
                                relative_permittivity_scale: Some(0.95),
                                relative_permeability_scale: Some(0.98),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: 200.0,
                                conductivity_scale: 1.0,
                                dispersive_loss_scale: Some(0.26),
                                relative_permittivity_scale: Some(1.0),
                                relative_permeability_scale: Some(1.0),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: 1_000.0,
                                conductivity_scale: 1.24,
                                dispersive_loss_scale: Some(0.29),
                                relative_permittivity_scale: Some(1.10),
                                relative_permeability_scale: Some(1.04),
                            },
                        ],
                    }),
                    ..base
                },
            ];
        } else {
            for (idx, material) in model.materials.iter_mut().enumerate() {
                let (sigma, eps_r, mu_r) = match idx % 3 {
                    0 => (5.8e7, 2.0, 1.0),
                    1 => (8.0e4, 14.0, 90.0),
                    _ => (0.2, 3.5, 1.05),
                };
                let conductivity_frequency_response = match idx % 3 {
                    0 => vec![
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: 50.0,
                            conductivity_scale: 1.05,
                            dispersive_loss_scale: Some(0.02),
                            relative_permittivity_scale: Some(1.01),
                            relative_permeability_scale: Some(1.0),
                        },
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: 200.0,
                            conductivity_scale: 1.0,
                            dispersive_loss_scale: Some(0.025),
                            relative_permittivity_scale: Some(1.0),
                            relative_permeability_scale: Some(1.0),
                        },
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: 1_000.0,
                            conductivity_scale: 0.94,
                            dispersive_loss_scale: Some(0.03),
                            relative_permittivity_scale: Some(0.99),
                            relative_permeability_scale: Some(1.0),
                        },
                    ],
                    1 => vec![
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: 50.0,
                            conductivity_scale: 1.32,
                            dispersive_loss_scale: Some(0.08),
                            relative_permittivity_scale: Some(1.15),
                            relative_permeability_scale: Some(1.35),
                        },
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: 200.0,
                            conductivity_scale: 1.0,
                            dispersive_loss_scale: Some(0.11),
                            relative_permittivity_scale: Some(1.0),
                            relative_permeability_scale: Some(1.0),
                        },
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: 1_000.0,
                            conductivity_scale: 0.72,
                            dispersive_loss_scale: Some(0.16),
                            relative_permittivity_scale: Some(0.85),
                            relative_permeability_scale: Some(0.70),
                        },
                    ],
                    _ => vec![
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: 50.0,
                            conductivity_scale: 0.88,
                            dispersive_loss_scale: Some(0.24),
                            relative_permittivity_scale: Some(0.95),
                            relative_permeability_scale: Some(0.98),
                        },
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: 200.0,
                            conductivity_scale: 1.0,
                            dispersive_loss_scale: Some(0.26),
                            relative_permittivity_scale: Some(1.0),
                            relative_permeability_scale: Some(1.0),
                        },
                        runmat_analysis_core::ConductivityFrequencyPoint {
                            frequency_hz: 1_000.0,
                            conductivity_scale: 1.24,
                            dispersive_loss_scale: Some(0.29),
                            relative_permittivity_scale: Some(1.10),
                            relative_permeability_scale: Some(1.04),
                        },
                    ],
                };
                material.electrical = Some(runmat_analysis_core::MaterialElectricalModel {
                    reference_temperature_k: 293.15,
                    conductivity_s_per_m: sigma,
                    resistive_heating_coefficient: 0.0025,
                    relative_permittivity: eps_r,
                    relative_permeability: mu_r,
                    conductivity_frequency_response,
                });
            }
        }

        match profile.kind {
            ElectromagneticFixtureKind::Homogeneous => {
                for material in &mut model.materials {
                    material.electrical = Some(runmat_analysis_core::MaterialElectricalModel {
                        reference_temperature_k: 293.15,
                        conductivity_s_per_m: 5.8e7,
                        resistive_heating_coefficient: 0.0039,
                        relative_permittivity: 2.1,
                        relative_permeability: 1.0,
                        conductivity_frequency_response: vec![
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: 50.0,
                                conductivity_scale: 1.02,
                                dispersive_loss_scale: Some(0.015),
                                relative_permittivity_scale: Some(1.01),
                                relative_permeability_scale: Some(1.0),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: 60.0,
                                conductivity_scale: 1.0,
                                dispersive_loss_scale: Some(0.017),
                                relative_permittivity_scale: Some(1.0),
                                relative_permeability_scale: Some(1.0),
                            },
                            runmat_analysis_core::ConductivityFrequencyPoint {
                                frequency_hz: 1_000.0,
                                conductivity_scale: 0.95,
                                dispersive_loss_scale: Some(0.03),
                                relative_permittivity_scale: Some(0.99),
                                relative_permeability_scale: Some(1.0),
                            },
                        ],
                    });
                }
                let material_id = model
                    .materials
                    .first()
                    .map(|material| material.material_id.clone())
                    .unwrap_or_else(|| "mat_default".to_string());
                if model.material_assignments.is_empty() {
                    model
                        .material_assignments
                        .push(runmat_analysis_core::MaterialAssignment {
                            region_id: "em_region_homogeneous".to_string(),
                            expected_material_id: material_id.clone(),
                            assigned_material_id: material_id,
                            confidence: runmat_analysis_core::EvidenceConfidence::Verified,
                        });
                } else {
                    for assignment in &mut model.material_assignments {
                        assignment.region_id = "em_region_homogeneous".to_string();
                        assignment.expected_material_id = material_id.clone();
                        assignment.assigned_material_id = material_id.clone();
                        assignment.confidence = runmat_analysis_core::EvidenceConfidence::Verified;
                    }
                }
            }
            ElectromagneticFixtureKind::Heterogeneous => {
                if model.material_assignments.is_empty() {
                    model.material_assignments = model
                        .materials
                        .iter()
                        .enumerate()
                        .map(|(idx, material)| runmat_analysis_core::MaterialAssignment {
                            region_id: format!("em_region_{idx}"),
                            expected_material_id: material.material_id.clone(),
                            assigned_material_id: material.material_id.clone(),
                            confidence: match idx % 3 {
                                0 => runmat_analysis_core::EvidenceConfidence::Verified,
                                1 => runmat_analysis_core::EvidenceConfidence::Probable,
                                _ => runmat_analysis_core::EvidenceConfidence::Inferred,
                            },
                        })
                        .collect();
                } else {
                    for (idx, assignment) in model.material_assignments.iter_mut().enumerate() {
                        assignment.region_id = format!("em_region_{idx}");
                        let material_id = &model.materials[idx % model.materials.len()].material_id;
                        assignment.expected_material_id = material_id.clone();
                        assignment.assigned_material_id = material_id.clone();
                        assignment.confidence = match idx % 3 {
                            0 => runmat_analysis_core::EvidenceConfidence::Verified,
                            1 => runmat_analysis_core::EvidenceConfidence::Probable,
                            _ => runmat_analysis_core::EvidenceConfidence::Inferred,
                        };
                    }
                }
            }
            ElectromagneticFixtureKind::SparseAssignments => {
                let ids = model
                    .materials
                    .iter()
                    .map(|material| material.material_id.clone())
                    .collect::<Vec<_>>();
                model.material_assignments = (0..10)
                    .map(|idx| {
                        if idx < 4 {
                            let material_id = ids[idx % ids.len()].clone();
                            runmat_analysis_core::MaterialAssignment {
                                region_id: format!("em_sparse_region_{idx}"),
                                expected_material_id: material_id.clone(),
                                assigned_material_id: material_id,
                                confidence: if idx % 2 == 0 {
                                    runmat_analysis_core::EvidenceConfidence::Verified
                                } else {
                                    runmat_analysis_core::EvidenceConfidence::Probable
                                },
                            }
                        } else {
                            let material_id = ids[idx % ids.len()].clone();
                            runmat_analysis_core::MaterialAssignment {
                                region_id: format!("em_sparse_region_{idx}"),
                                expected_material_id: material_id.clone(),
                                assigned_material_id: material_id,
                                confidence: runmat_analysis_core::EvidenceConfidence::Inferred,
                            }
                        }
                    })
                    .collect();
            }
            ElectromagneticFixtureKind::MultiRegionAssignments => {
                let ids = model
                    .materials
                    .iter()
                    .map(|material| material.material_id.clone())
                    .collect::<Vec<_>>();
                model.material_assignments = (0..9)
                    .map(|idx| {
                        let material_id = ids[idx % ids.len()].clone();
                        runmat_analysis_core::MaterialAssignment {
                            region_id: format!("em_multiregion_region_{idx}"),
                            expected_material_id: material_id.clone(),
                            assigned_material_id: material_id,
                            confidence: if idx % 2 == 0 {
                                runmat_analysis_core::EvidenceConfidence::Probable
                            } else {
                                runmat_analysis_core::EvidenceConfidence::Inferred
                            },
                        }
                    })
                    .collect();
            }
            ElectromagneticFixtureKind::OverlapInterference => {
                if model.material_assignments.is_empty() {
                    model.material_assignments = model
                        .materials
                        .iter()
                        .enumerate()
                        .take(3)
                        .map(|(idx, material)| runmat_analysis_core::MaterialAssignment {
                            region_id: format!("em_overlap_region_{idx}"),
                            expected_material_id: material.material_id.clone(),
                            assigned_material_id: material.material_id.clone(),
                            confidence: runmat_analysis_core::EvidenceConfidence::Verified,
                        })
                        .collect();
                } else {
                    for (idx, assignment) in model.material_assignments.iter_mut().enumerate() {
                        assignment.region_id = format!("em_overlap_region_{}", idx % 3);
                        let material_id = &model.materials[idx % model.materials.len()].material_id;
                        assignment.expected_material_id = material_id.clone();
                        assignment.assigned_material_id = material_id.clone();
                        assignment.confidence = runmat_analysis_core::EvidenceConfidence::Verified;
                    }
                }
            }
            ElectromagneticFixtureKind::BoundaryKernel => {
                let ids = model
                    .materials
                    .iter()
                    .map(|material| material.material_id.clone())
                    .collect::<Vec<_>>();
                model.material_assignments = (0..32)
                    .map(|idx| {
                        let material_id = ids[idx % ids.len()].clone();
                        runmat_analysis_core::MaterialAssignment {
                            region_id: format!("em_boundary_region_{idx}"),
                            expected_material_id: material_id.clone(),
                            assigned_material_id: material_id,
                            confidence: if idx % 2 == 0 {
                                runmat_analysis_core::EvidenceConfidence::Verified
                            } else {
                                runmat_analysis_core::EvidenceConfidence::Probable
                            },
                        }
                    })
                    .collect();
            }
            ElectromagneticFixtureKind::BoundaryPenaltyStress => {
                let ids = model
                    .materials
                    .iter()
                    .map(|material| material.material_id.clone())
                    .collect::<Vec<_>>();
                model.material_assignments = (0..48)
                    .map(|idx| {
                        let material_id = ids[idx % ids.len()].clone();
                        runmat_analysis_core::MaterialAssignment {
                            region_id: format!("em_penalty_region_{idx}"),
                            expected_material_id: material_id.clone(),
                            assigned_material_id: material_id,
                            confidence: if idx % 4 == 0 {
                                runmat_analysis_core::EvidenceConfidence::Probable
                            } else {
                                runmat_analysis_core::EvidenceConfidence::Verified
                            },
                        }
                    })
                    .collect();
            }
            ElectromagneticFixtureKind::MultiRegionPhasedSource => {
                let ids = model
                    .materials
                    .iter()
                    .map(|material| material.material_id.clone())
                    .collect::<Vec<_>>();
                model.material_assignments = (0..16)
                    .map(|idx| {
                        let material_id = ids[idx % ids.len()].clone();
                        runmat_analysis_core::MaterialAssignment {
                            region_id: format!("em_phase_region_{idx}"),
                            expected_material_id: material_id.clone(),
                            assigned_material_id: material_id,
                            confidence: if idx % 3 == 0 {
                                runmat_analysis_core::EvidenceConfidence::Inferred
                            } else {
                                runmat_analysis_core::EvidenceConfidence::Verified
                            },
                        }
                    })
                    .collect();
            }
        }

        match profile.kind {
            ElectromagneticFixtureKind::Homogeneous => {
                model.boundary_conditions = vec![
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_ground_0".to_string(),
                        region_id: "em_region_homogeneous".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::VectorPotentialGround,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_ground_1".to_string(),
                        region_id: "em_region_homogeneous".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::VectorPotentialGround,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_insulation_0".to_string(),
                        region_id: "em_region_homogeneous".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::MagneticInsulation,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_insulation_1".to_string(),
                        region_id: "em_region_homogeneous".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::MagneticInsulation,
                    },
                ];
                model.loads = vec![
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_coil_0".to_string(),
                        region_id: "em_region_homogeneous".to_string(),
                        kind: runmat_analysis_core::LoadKind::CoilCurrent {
                            current_a: profile.applied_current_a,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_current_density_0".to_string(),
                        region_id: "em_region_homogeneous".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: 1.0,
                            jy: 0.0,
                            jz: 0.0,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_current_density_1".to_string(),
                        region_id: "em_region_homogeneous".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: 0.0,
                            jy: 1.0,
                            jz: 0.0,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                ];
            }
            ElectromagneticFixtureKind::Heterogeneous => {
                model.boundary_conditions = vec![
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_ground_hetero".to_string(),
                        region_id: "em_region_0".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::VectorPotentialGround,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_insulation_hetero_0".to_string(),
                        region_id: "em_region_1".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::MagneticInsulation,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_insulation_hetero_1".to_string(),
                        region_id: "em_region_2".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::MagneticInsulation,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_structural_fallback".to_string(),
                        region_id: "em_region_2".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::Fixed,
                    },
                ];
                model.loads = vec![
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_coil_hetero".to_string(),
                        region_id: "em_region_0".to_string(),
                        kind: runmat_analysis_core::LoadKind::CoilCurrent {
                            current_a: profile.applied_current_a * 0.85,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_current_density_hetero".to_string(),
                        region_id: "em_region_1".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: 0.8,
                            jy: 0.3,
                            jz: 0.1,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_structural_mixed".to_string(),
                        region_id: "em_region_2".to_string(),
                        kind: runmat_analysis_core::LoadKind::Force {
                            fx: 0.0,
                            fy: -50.0,
                            fz: 0.0,
                        },
                    },
                ];
            }
            ElectromagneticFixtureKind::SparseAssignments => {
                model.boundary_conditions = vec![
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_sparse_ground".to_string(),
                        region_id: "em_sparse_region_0".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::VectorPotentialGround,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_sparse_struct_0".to_string(),
                        region_id: "em_sparse_region_1".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::Fixed,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_sparse_struct_1".to_string(),
                        region_id: "em_sparse_region_2".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::Fixed,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_sparse_struct_2".to_string(),
                        region_id: "em_sparse_region_3".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::PrescribedDisplacement,
                    },
                ];
                model.loads = vec![
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_sparse_coil".to_string(),
                        region_id: "em_sparse_region_0".to_string(),
                        kind: runmat_analysis_core::LoadKind::CoilCurrent {
                            current_a: profile.applied_current_a * 0.4,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_sparse_current_density_0".to_string(),
                        region_id: "em_sparse_region_1".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: 0.0,
                            jy: 0.6,
                            jz: 0.2,
                            phase_rad: 0.2,
                            amplitude_scale: 0.8,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_sparse_current_density_1".to_string(),
                        region_id: "em_sparse_region_2".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: 0.3,
                            jy: 0.1,
                            jz: 0.4,
                            phase_rad: 0.4,
                            amplitude_scale: 0.7,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_sparse_current_density_2".to_string(),
                        region_id: "em_sparse_region_3".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: 0.1,
                            jy: 0.0,
                            jz: 0.7,
                            phase_rad: 0.6,
                            amplitude_scale: 0.6,
                        },
                    },
                ];
            }
            ElectromagneticFixtureKind::MultiRegionAssignments => {
                model.boundary_conditions = vec![
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_multiregion_ground".to_string(),
                        region_id: "em_multiregion_region_0".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::VectorPotentialGround,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_multiregion_insulation_0".to_string(),
                        region_id: "em_multiregion_region_1".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::MagneticInsulation,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_multiregion_insulation_1".to_string(),
                        region_id: "em_multiregion_region_2".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::MagneticInsulation,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_multiregion_ground_1".to_string(),
                        region_id: "em_multiregion_region_3".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::VectorPotentialGround,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_multiregion_insulation_2".to_string(),
                        region_id: "em_multiregion_region_4".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::MagneticInsulation,
                    },
                ];
                model.loads = vec![
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_multiregion_coil".to_string(),
                        region_id: "em_multiregion_region_0".to_string(),
                        kind: runmat_analysis_core::LoadKind::CoilCurrent {
                            current_a: profile.applied_current_a * 0.25,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_multiregion_current_density_0".to_string(),
                        region_id: "em_multiregion_region_1".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: 0.4,
                            jy: 0.2,
                            jz: 0.1,
                            phase_rad: 0.15,
                            amplitude_scale: 0.9,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_multiregion_current_density_1".to_string(),
                        region_id: "em_multiregion_region_2".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: 0.1,
                            jy: 0.5,
                            jz: 0.2,
                            phase_rad: 0.35,
                            amplitude_scale: 0.8,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_multiregion_current_density_2".to_string(),
                        region_id: "em_multiregion_region_3".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: 0.2,
                            jy: 0.1,
                            jz: 0.6,
                            phase_rad: 0.55,
                            amplitude_scale: 0.7,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_multiregion_current_density_3".to_string(),
                        region_id: "em_multiregion_region_4".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: 0.3,
                            jy: 0.3,
                            jz: 0.3,
                            phase_rad: 0.75,
                            amplitude_scale: 0.65,
                        },
                    },
                ];
            }
            ElectromagneticFixtureKind::OverlapInterference => {
                model.boundary_conditions = vec![
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_overlap_ground".to_string(),
                        region_id: "em_overlap_region_0".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::VectorPotentialGround,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_overlap_insulation_0".to_string(),
                        region_id: "em_overlap_region_1".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::MagneticInsulation,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_overlap_insulation_1".to_string(),
                        region_id: "em_overlap_region_2".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::MagneticInsulation,
                    },
                ];
                model.loads = vec![
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_overlap_coil_pos".to_string(),
                        region_id: "em_overlap_region_1".to_string(),
                        kind: runmat_analysis_core::LoadKind::CoilCurrent {
                            current_a: profile.applied_current_a,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_overlap_coil_neg".to_string(),
                        region_id: "em_overlap_region_1".to_string(),
                        kind: runmat_analysis_core::LoadKind::CoilCurrent {
                            current_a: -profile.applied_current_a * 0.92,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_overlap_cd_pos".to_string(),
                        region_id: "em_overlap_region_1".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: 1.0,
                            jy: 0.4,
                            jz: 0.0,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_overlap_cd_neg".to_string(),
                        region_id: "em_overlap_region_1".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: -0.8,
                            jy: -0.35,
                            jz: 0.0,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                ];
            }
            ElectromagneticFixtureKind::BoundaryKernel => {
                model.boundary_conditions = vec![
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_boundary_ground_mapped".to_string(),
                        region_id: "em_boundary_region_24".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::VectorPotentialGround,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_boundary_ground_unmapped_0".to_string(),
                        region_id: "em_boundary_unmapped_ground_0".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::VectorPotentialGround,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_boundary_ground_unmapped_1".to_string(),
                        region_id: "em_boundary_unmapped_ground_1".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::VectorPotentialGround,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_boundary_insulation_unmapped_0".to_string(),
                        region_id: "em_boundary_unmapped_insulation_0".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::MagneticInsulation,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_boundary_insulation_unmapped_1".to_string(),
                        region_id: "em_boundary_unmapped_insulation_1".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::MagneticInsulation,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_boundary_structural_noise".to_string(),
                        region_id: "em_boundary_region_3".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::Fixed,
                    },
                ];
                model.loads = vec![
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_boundary_coil_primary".to_string(),
                        region_id: "em_boundary_region_24".to_string(),
                        kind: runmat_analysis_core::LoadKind::CoilCurrent {
                            current_a: profile.applied_current_a * 0.9,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_boundary_current_density".to_string(),
                        region_id: "em_boundary_region_25".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: 0.9,
                            jy: -0.5,
                            jz: 0.2,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                ];
            }
            ElectromagneticFixtureKind::BoundaryPenaltyStress => {
                model.boundary_conditions = (0..18)
                    .flat_map(|idx| {
                        let mut conditions = vec![runmat_analysis_core::BoundaryCondition {
                            bc_id: format!("em_bc_penalty_ground_{idx}"),
                            region_id: format!("em_penalty_region_{}", idx + 8),
                            kind:
                                runmat_analysis_core::BoundaryConditionKind::VectorPotentialGround,
                        }];
                        if idx % 2 == 0 {
                            conditions.push(runmat_analysis_core::BoundaryCondition {
                                bc_id: format!("em_bc_penalty_insulation_{idx}"),
                                region_id: format!("em_penalty_region_{}", idx + 20),
                                kind:
                                    runmat_analysis_core::BoundaryConditionKind::MagneticInsulation,
                            });
                        }
                        conditions
                    })
                    .collect();
                model.loads = vec![
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_penalty_coil_0".to_string(),
                        region_id: "em_penalty_region_10".to_string(),
                        kind: runmat_analysis_core::LoadKind::CoilCurrent {
                            current_a: profile.applied_current_a,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_penalty_coil_1".to_string(),
                        region_id: "em_penalty_region_13".to_string(),
                        kind: runmat_analysis_core::LoadKind::CoilCurrent {
                            current_a: profile.applied_current_a * -0.65,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_penalty_density".to_string(),
                        region_id: "em_penalty_region_29".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: 1.2,
                            jy: -0.7,
                            jz: 0.15,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                ];
            }
            ElectromagneticFixtureKind::MultiRegionPhasedSource => {
                model.boundary_conditions = vec![
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_phase_ground_0".to_string(),
                        region_id: "em_phase_region_0".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::VectorPotentialGround,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_phase_ground_1".to_string(),
                        region_id: "em_phase_region_8".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::VectorPotentialGround,
                    },
                    runmat_analysis_core::BoundaryCondition {
                        bc_id: "em_bc_phase_insulation_0".to_string(),
                        region_id: "em_phase_region_5".to_string(),
                        kind: runmat_analysis_core::BoundaryConditionKind::MagneticInsulation,
                    },
                ];
                model.loads = vec![
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_phase_coil_0".to_string(),
                        region_id: "em_phase_region_1".to_string(),
                        kind: runmat_analysis_core::LoadKind::CoilCurrent {
                            current_a: profile.applied_current_a,
                            phase_rad: 0.0,
                            amplitude_scale: 1.0,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_phase_coil_1".to_string(),
                        region_id: "em_phase_region_3".to_string(),
                        kind: runmat_analysis_core::LoadKind::CoilCurrent {
                            current_a: -profile.applied_current_a * 0.8,
                            phase_rad: std::f64::consts::PI * 0.85,
                            amplitude_scale: 0.9,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_phase_coil_2".to_string(),
                        region_id: "em_phase_region_7".to_string(),
                        kind: runmat_analysis_core::LoadKind::CoilCurrent {
                            current_a: profile.applied_current_a * 0.55,
                            phase_rad: std::f64::consts::FRAC_PI_2,
                            amplitude_scale: 0.8,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_phase_density_0".to_string(),
                        region_id: "em_phase_region_10".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: 0.9,
                            jy: -0.45,
                            jz: 0.2,
                            phase_rad: std::f64::consts::PI / 3.0,
                            amplitude_scale: 0.9,
                        },
                    },
                    runmat_analysis_core::LoadCase {
                        load_id: "em_load_phase_density_1".to_string(),
                        region_id: "em_phase_region_12".to_string(),
                        kind: runmat_analysis_core::LoadKind::CurrentDensity {
                            jx: -0.85,
                            jy: 0.35,
                            jz: -0.1,
                            phase_rad: -std::f64::consts::FRAC_PI_2,
                            amplitude_scale: 0.85,
                        },
                    },
                ];
            }
        }

        model.electromagnetic = Some(runmat_analysis_core::ElectromagneticDomain {
            enabled: true,
            reference_frequency_hz: profile.reference_frequency_hz,
            applied_current_a: profile.applied_current_a,
        });
        match spec_id {
            "electromagnetic_missing_material" => {
                for material in &mut model.materials {
                    material.electrical = None;
                }
            }
            "electromagnetic_missing_source" => {
                model.loads = vec![runmat_analysis_core::LoadCase {
                    load_id: "em_invalid_structural_force".to_string(),
                    region_id: "em_region_homogeneous".to_string(),
                    kind: runmat_analysis_core::LoadKind::Force {
                        fx: 1.0,
                        fy: 0.0,
                        fz: 0.0,
                    },
                }];
            }
            "electromagnetic_missing_boundary" => {
                model.boundary_conditions = vec![runmat_analysis_core::BoundaryCondition {
                    bc_id: "em_invalid_structural_fixed".to_string(),
                    region_id: "em_region_homogeneous".to_string(),
                    kind: runmat_analysis_core::BoundaryConditionKind::Fixed,
                }];
            }
            _ => {}
        }
    }

    let mut thermo = thermo_coupling_for_fixture(spec_id);
    if thermo.is_none() && spec_id == "nonlinear_load_path_mix_gpu_provider" {
        thermo = Some(ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 75.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: Vec::new(),
            time_profile: Vec::new(),
        });
    }
    if let Some(value) = thermo {
        model.thermo_mechanical = Some(runmat_analysis_core::ThermoMechanicalDomain {
            enabled: value.enabled,
            reference_temperature_k: value.reference_temperature_k,
            applied_temperature_delta_k: value.applied_temperature_delta_k,
            field_artifact_id: value.field_artifact_id,
            field_source: value.field_source.map(|source| {
                runmat_analysis_core::ThermoFieldSource {
                    source_id: source.source_id,
                    revision: source.revision,
                    interpolation_mode: source.interpolation_mode.map(|mode| match mode {
                        runmat_runtime::analysis::ThermoFieldInterpolationMode::Linear => {
                            runmat_analysis_core::ThermoFieldInterpolationMode::Linear
                        }
                        runmat_runtime::analysis::ThermoFieldInterpolationMode::Step => {
                            runmat_analysis_core::ThermoFieldInterpolationMode::Step
                        }
                    }),
                    expected_region_ids: source.expected_region_ids,
                }
            }),
            region_temperature_deltas: value
                .region_temperature_deltas
                .into_iter()
                .map(|delta| runmat_analysis_core::ThermoRegionTemperatureDelta {
                    region_id: delta.region_id,
                    temperature_delta_k: delta.temperature_delta_k,
                })
                .collect(),
            time_profile: value
                .time_profile
                .into_iter()
                .map(|point| runmat_analysis_core::ThermoTimeProfilePoint {
                    normalized_time: point.normalized_time,
                    scale: point.scale,
                })
                .collect(),
        });
    }

    if let Some(electro) = electro_coupling_for_fixture(spec_id) {
        let mut material_region_ids = model
            .material_assignments
            .iter()
            .map(|assignment| assignment.region_id.clone())
            .collect::<Vec<_>>();
        if material_region_ids.is_empty() {
            let material_id = model
                .materials
                .first()
                .map(|material| material.material_id.clone())
                .unwrap_or_else(|| "mat_default".to_string());
            model
                .material_assignments
                .push(runmat_analysis_core::MaterialAssignment {
                    region_id: "electro_region_0".to_string(),
                    expected_material_id: material_id.clone(),
                    assigned_material_id: material_id,
                    confidence: runmat_analysis_core::EvidenceConfidence::Verified,
                });
            material_region_ids.push("electro_region_0".to_string());
        }

        let mut region_conductivity_scales =
            Vec::with_capacity(electro.region_conductivity_scales.len());
        for (index, scale) in electro.region_conductivity_scales.into_iter().enumerate() {
            let mapped_region_id = if spec_id == "electro_thermal_unmapped_region" {
                scale.region_id.clone()
            } else if index < material_region_ids.len() {
                material_region_ids[index].clone()
            } else {
                let synthesized_region_id = format!("electro_region_{index}");
                let assignment_seed =
                    model
                        .material_assignments
                        .first()
                        .cloned()
                        .unwrap_or_else(|| runmat_analysis_core::MaterialAssignment {
                            region_id: synthesized_region_id.clone(),
                            expected_material_id: model
                                .materials
                                .first()
                                .map(|material| material.material_id.clone())
                                .unwrap_or_else(|| "mat_default".to_string()),
                            assigned_material_id: model
                                .materials
                                .first()
                                .map(|material| material.material_id.clone())
                                .unwrap_or_else(|| "mat_default".to_string()),
                            confidence: runmat_analysis_core::EvidenceConfidence::Verified,
                        });
                model
                    .material_assignments
                    .push(runmat_analysis_core::MaterialAssignment {
                        region_id: synthesized_region_id.clone(),
                        ..assignment_seed
                    });
                material_region_ids.push(synthesized_region_id.clone());
                synthesized_region_id
            };
            region_conductivity_scales.push(runmat_analysis_core::ElectroRegionConductivityScale {
                region_id: mapped_region_id,
                conductivity_scale: scale.conductivity_scale,
            });
        }

        for material in &mut model.materials {
            material.electrical = Some(runmat_analysis_core::MaterialElectricalModel {
                reference_temperature_k: electro.reference_temperature_k,
                conductivity_s_per_m: electro.base_electrical_conductivity_s_per_m,
                resistive_heating_coefficient: electro.resistive_heating_coefficient,
                relative_permittivity: 1.0,
                relative_permeability: 1.0,
                conductivity_frequency_response: Vec::new(),
            });
        }
        model.electro_thermal = Some(runmat_analysis_core::ElectroThermalDomain {
            enabled: electro.enabled,
            reference_temperature_k: electro.reference_temperature_k,
            applied_voltage_v: electro.applied_voltage_v,
            region_conductivity_scales,
            time_profile: electro
                .time_profile
                .into_iter()
                .map(
                    |ElectroTimeProfilePoint {
                         normalized_time,
                         current_scale,
                     }| runmat_analysis_core::ElectroTimeProfilePoint {
                        normalized_time,
                        current_scale,
                    },
                )
                .collect(),
        });
    }

    if let Some(plastic) = plasticity_for_fixture(spec_id) {
        for material in &mut model.materials {
            material.plastic = Some(runmat_analysis_core::MaterialPlasticModel {
                yield_strain: plastic.yield_strain,
                hardening_modulus_ratio: plastic.hardening_modulus_ratio,
                saturation_exponent: plastic.saturation_exponent,
            });
        }
        model.loads = plastic_load_cases_for_fixture(spec_id);
    }

    if let Some(contact) = contact_for_fixture(spec_id) {
        let mut regions = model
            .material_assignments
            .iter()
            .map(|assignment| assignment.region_id.clone())
            .collect::<Vec<_>>();
        if regions.is_empty() {
            regions.push("root".to_string());
            regions.push("tip".to_string());
        }
        let primary = regions
            .first()
            .cloned()
            .unwrap_or_else(|| "root".to_string());
        let secondary = regions
            .iter()
            .find(|region| **region != primary)
            .cloned()
            .unwrap_or_else(|| "tip".to_string());
        model.interfaces = vec![runmat_analysis_core::AnalysisInterface {
            interface_id: format!("contact_{spec_id}"),
            primary_region_id: primary,
            secondary_region_id: secondary,
            kind: runmat_analysis_core::AnalysisInterfaceKind::Contact(
                runmat_analysis_core::ContactInterfaceModel {
                    penalty_stiffness_scale: contact.penalty_stiffness_scale,
                    max_penetration_ratio: contact.max_penetration_ratio,
                    friction_coefficient: contact.friction_coefficient,
                },
            ),
        }];
    }
}

fn plastic_load_cases_for_fixture(spec_id: &str) -> Vec<runmat_analysis_core::LoadCase> {
    let peak_force = match spec_id {
        "nonlinear_plasticity_benchmark_gpu_provider" => 1.2e7,
        "nonlinear_plastic_hardening_reference_gpu_provider" => 3.0e7,
        "nonlinear_plastic_hardening_reference_complex_gpu_provider" => 4.2e7,
        _ => 1.0e7,
    };
    vec![
        runmat_analysis_core::LoadCase {
            load_id: format!("plastic_axial_drive_{spec_id}"),
            region_id: "plastic_drive_tip".to_string(),
            kind: runmat_analysis_core::LoadKind::Force {
                fx: peak_force,
                fy: 0.0,
                fz: 0.0,
            },
        },
        runmat_analysis_core::LoadCase {
            load_id: format!("plastic_shear_drive_{spec_id}"),
            region_id: "plastic_drive_web".to_string(),
            kind: runmat_analysis_core::LoadKind::Force {
                fx: 0.0,
                fy: -0.35 * peak_force,
                fz: 0.12 * peak_force,
            },
        },
        runmat_analysis_core::LoadCase {
            load_id: format!("plastic_pressure_drive_{spec_id}"),
            region_id: "plastic_drive_face".to_string(),
            kind: runmat_analysis_core::LoadKind::Pressure {
                magnitude_pa: 0.08 * peak_force,
            },
        },
    ]
}

fn ensure_thermo_field_artifacts_for_fixture(spec_id: &str, model: &AnalysisModel) {
    let root = harness_thermo_field_artifact_root(&harness_artifact_root());
    let _ = fs::create_dir_all(&root);
    let write_artifact = |artifact_id: &str,
                          source_id: &str,
                          interpolation_mode: &str,
                          region_temperature_deltas: serde_json::Value,
                          time_profile: serde_json::Value| {
        let mut payload = serde_json::json!({
            "schema_version": "analysis_thermo_field_artifact/v1",
            "source_geometry_id": model.geometry_id,
            "source_geometry_revision": model.geometry_revision,
            "artifact_status": "approved",
            "created_at": "2026-03-10T00:00:00Z",
            "approved_at": "2026-03-10T00:05:00Z",
            "approved_by": "release-bot",
            "field_source": {
                "source_id": source_id,
                "revision": 1,
                "interpolation_mode": interpolation_mode,
                "expected_region_ids": [],
            },
            "region_temperature_deltas": region_temperature_deltas,
            "time_profile": time_profile,
        });
        let payload_hash = thermo_field_payload_hash_for_value(&payload);
        payload["payload_hash"] = serde_json::Value::String(payload_hash.clone());
        payload["signature"] =
            serde_json::Value::String(thermo_field_signature(&payload_hash, "release-bot"));
        let _ = fs::write(
            root.join(format!("{artifact_id}.json")),
            serde_json::to_vec_pretty(&payload).unwrap_or_default(),
        );
    };

    if spec_id == "thermo_ramp_smooth_field_artifact_gpu_provider" {
        write_artifact(
            "thermo_ramp_smooth_approved",
            "field/thermo-ramp-smooth",
            "linear",
            serde_json::json!([
                {"region_id": "tip_steel", "temperature_delta_k": 72.0},
                {"region_id": "mid_aluminum", "temperature_delta_k": 68.0}
            ]),
            serde_json::json!([
                {"normalized_time": 0.0, "scale": 0.3},
                {"normalized_time": 0.5, "scale": 0.7},
                {"normalized_time": 1.0, "scale": 1.0}
            ]),
        );
    }

    if spec_id == "thermo_shock_oscillatory_field_artifact_gpu_provider" {
        write_artifact(
            "thermo_shock_oscillatory_approved",
            "field/thermo-shock-oscillatory",
            "step",
            serde_json::json!([
                {"region_id": "tip_steel", "temperature_delta_k": 210.0},
                {"region_id": "polymer_segment", "temperature_delta_k": 90.0}
            ]),
            serde_json::json!([
                {"normalized_time": 0.0, "scale": 0.4},
                {"normalized_time": 0.25, "scale": 1.4},
                {"normalized_time": 0.5, "scale": 0.5},
                {"normalized_time": 0.75, "scale": 1.3},
                {"normalized_time": 1.0, "scale": 0.6}
            ]),
        );
    }
}

type FixtureRunEnvelope =
    runmat_runtime::operations::OperationEnvelope<runmat_runtime::analysis::AnalysisRunResult>;
type FixtureRunResult =
    Result<FixtureRunEnvelope, Box<runmat_runtime::operations::OperationErrorEnvelope>>;

fn boxed_fixture_run_result(
    result: Result<FixtureRunEnvelope, runmat_runtime::operations::OperationErrorEnvelope>,
) -> FixtureRunResult {
    result.map_err(Box::new)
}

fn run_fixture_cpu(spec: &FixtureSpec, model: &AnalysisModel) -> FixtureRunResult {
    boxed_fixture_run_result(match spec.run_kind {
        AnalysisRunKind::LinearStatic => analysis_run_linear_static_with_options(
            model,
            ComputeBackend::Cpu,
            default_options(),
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Modal => analysis_run_modal_with_options_op(
            model,
            ComputeBackend::Cpu,
            AnalysisModalRunOptions {
                mode_count: spec
                    .modal_mode_count
                    .unwrap_or(AnalysisModalRunOptions::default().mode_count),
                ..AnalysisModalRunOptions::balanced()
            },
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Acoustic => analysis_run_acoustic_with_options_op(
            model,
            ComputeBackend::Cpu,
            AnalysisAcousticRunOptions {
                mode_count: spec
                    .modal_mode_count
                    .unwrap_or(AnalysisAcousticRunOptions::default().mode_count),
                ..AnalysisAcousticRunOptions::default()
            },
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Transient => analysis_run_transient_with_options_op(
            model,
            ComputeBackend::Cpu,
            {
                let requested_bucket_rel_tol = std::env::var("RUNMAT_TRANSIENT_DT_BUCKET_REL_TOL")
                    .ok()
                    .and_then(|value| value.parse::<f64>().ok());
                AnalysisTransientRunOptions {
                    step_count: spec
                        .transient_step_count
                        .unwrap_or(AnalysisTransientRunOptions::default().step_count),
                    dt_bucket_rel_tolerance: requested_bucket_rel_tol.unwrap_or(
                        AnalysisTransientRunOptions::production_recommended()
                            .dt_bucket_rel_tolerance,
                    ),
                    ..AnalysisTransientRunOptions::production_recommended()
                }
            },
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Thermal => analysis_run_thermal_with_options_op(
            model,
            ComputeBackend::Cpu,
            AnalysisThermalRunOptions {
                step_count: spec
                    .transient_step_count
                    .unwrap_or(AnalysisThermalRunOptions::default().step_count),
                ..AnalysisThermalRunOptions::default()
            },
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Cfd => analysis_run_cfd_with_options_op(
            model,
            ComputeBackend::Cpu,
            AnalysisCfdRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                quality_policy: QualityPolicy::Balanced,
                time_step_s: 1.0e-3,
                step_count: spec
                    .transient_step_count
                    .unwrap_or(AnalysisCfdRunOptions::default().step_count),
                max_linear_iters: 128,
                tolerance: 1.0e-8,
                residual_warn_threshold: 1.0e-4,
                prep_context: None,
                prep_artifact_id: None,
                prep_calibration_profile: None,
            },
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Cht => analysis_run_cht_with_options_op(
            model,
            ComputeBackend::Cpu,
            AnalysisChtRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                quality_policy: QualityPolicy::Balanced,
                time_step_s: 1.0e-3,
                step_count: spec
                    .transient_step_count
                    .unwrap_or(AnalysisChtRunOptions::default().step_count),
                max_linear_iters: 128,
                tolerance: 1.0e-8,
                residual_warn_threshold: 1.0e-4,
                prep_context: None,
                prep_artifact_id: None,
                prep_calibration_profile: None,
            },
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Fsi => analysis_run_fsi_with_options_op(
            model,
            ComputeBackend::Cpu,
            AnalysisFsiRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                quality_policy: QualityPolicy::Balanced,
                time_step_s: 1.0e-3,
                step_count: spec
                    .transient_step_count
                    .unwrap_or(AnalysisFsiRunOptions::default().step_count),
                max_linear_iters: 128,
                tolerance: 1.0e-8,
                residual_warn_threshold: 1.0e-4,
                prep_context: None,
                prep_artifact_id: None,
                prep_calibration_profile: None,
            },
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Nonlinear => analysis_run_nonlinear_with_options_op(
            model,
            ComputeBackend::Cpu,
            nonlinear_options_for_spec(spec),
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Electromagnetic => analysis_run_electromagnetic_with_options_op(
            model,
            ComputeBackend::Cpu,
            AnalysisElectromagneticRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                quality_policy: QualityPolicy::Balanced,
                residual_target: 1.0e-6,
                harmonic_tolerance: 1.0e-7,
                harmonic_max_iterations: 96,
                prep_context: None,
                prep_artifact_id: None,
                prep_calibration_profile: None,
                sweep_enabled: !electromagnetic_sweep_frequency_hz_for_fixture(spec.id).is_empty(),
                sweep_frequency_hz: electromagnetic_sweep_frequency_hz_for_fixture(spec.id),
            },
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
    })
}

fn run_fixture_gpu(spec: &FixtureSpec, model: &AnalysisModel, mode: GpuMode) -> FixtureRunResult {
    let run = || match spec.run_kind {
        AnalysisRunKind::LinearStatic => analysis_run_linear_static_with_options(
            model,
            ComputeBackend::Gpu,
            default_options(),
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Modal => analysis_run_modal_with_options_op(
            model,
            ComputeBackend::Gpu,
            AnalysisModalRunOptions {
                mode_count: spec
                    .modal_mode_count
                    .unwrap_or(AnalysisModalRunOptions::default().mode_count),
                ..AnalysisModalRunOptions::balanced()
            },
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Acoustic => analysis_run_acoustic_with_options_op(
            model,
            ComputeBackend::Gpu,
            AnalysisAcousticRunOptions {
                mode_count: spec
                    .modal_mode_count
                    .unwrap_or(AnalysisAcousticRunOptions::default().mode_count),
                ..AnalysisAcousticRunOptions::default()
            },
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Transient => analysis_run_transient_with_options_op(
            model,
            ComputeBackend::Gpu,
            {
                let requested_bucket_rel_tol = std::env::var("RUNMAT_TRANSIENT_DT_BUCKET_REL_TOL")
                    .ok()
                    .and_then(|value| value.parse::<f64>().ok());
                AnalysisTransientRunOptions {
                    step_count: spec
                        .transient_step_count
                        .unwrap_or(AnalysisTransientRunOptions::default().step_count),
                    dt_bucket_rel_tolerance: requested_bucket_rel_tol.unwrap_or(
                        AnalysisTransientRunOptions::production_recommended()
                            .dt_bucket_rel_tolerance,
                    ),
                    ..AnalysisTransientRunOptions::production_recommended()
                }
            },
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Thermal => analysis_run_thermal_with_options_op(
            model,
            ComputeBackend::Gpu,
            AnalysisThermalRunOptions {
                step_count: spec
                    .transient_step_count
                    .unwrap_or(AnalysisThermalRunOptions::default().step_count),
                ..AnalysisThermalRunOptions::default()
            },
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Cfd => analysis_run_cfd_with_options_op(
            model,
            ComputeBackend::Gpu,
            AnalysisCfdRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                quality_policy: QualityPolicy::Balanced,
                time_step_s: 1.0e-3,
                step_count: spec
                    .transient_step_count
                    .unwrap_or(AnalysisCfdRunOptions::default().step_count),
                max_linear_iters: 128,
                tolerance: 1.0e-8,
                residual_warn_threshold: 1.0e-4,
                prep_context: None,
                prep_artifact_id: None,
                prep_calibration_profile: None,
            },
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Cht => analysis_run_cht_with_options_op(
            model,
            ComputeBackend::Gpu,
            AnalysisChtRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                quality_policy: QualityPolicy::Balanced,
                time_step_s: 1.0e-3,
                step_count: spec
                    .transient_step_count
                    .unwrap_or(AnalysisChtRunOptions::default().step_count),
                max_linear_iters: 128,
                tolerance: 1.0e-8,
                residual_warn_threshold: 1.0e-4,
                prep_context: None,
                prep_artifact_id: None,
                prep_calibration_profile: None,
            },
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Fsi => analysis_run_fsi_with_options_op(
            model,
            ComputeBackend::Gpu,
            AnalysisFsiRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                quality_policy: QualityPolicy::Balanced,
                time_step_s: 1.0e-3,
                step_count: spec
                    .transient_step_count
                    .unwrap_or(AnalysisFsiRunOptions::default().step_count),
                max_linear_iters: 128,
                tolerance: 1.0e-8,
                residual_warn_threshold: 1.0e-4,
                prep_context: None,
                prep_artifact_id: None,
                prep_calibration_profile: None,
            },
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Nonlinear => analysis_run_nonlinear_with_options_op(
            model,
            ComputeBackend::Gpu,
            nonlinear_options_for_spec(spec),
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Electromagnetic => analysis_run_electromagnetic_with_options_op(
            model,
            ComputeBackend::Gpu,
            AnalysisElectromagneticRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                quality_policy: QualityPolicy::Balanced,
                residual_target: 1.0e-6,
                harmonic_tolerance: 1.0e-7,
                harmonic_max_iterations: 96,
                prep_context: None,
                prep_artifact_id: None,
                prep_calibration_profile: None,
                sweep_enabled: !electromagnetic_sweep_frequency_hz_for_fixture(spec.id).is_empty(),
                sweep_frequency_hz: electromagnetic_sweep_frequency_hz_for_fixture(spec.id),
            },
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
    };
    boxed_fixture_run_result(match mode {
        GpuMode::WithProvider => with_harness_provider(run),
        GpuMode::WithoutProvider => {
            let _guard = ThreadProviderGuard::set(None);
            run()
        }
    })
}

fn parse_metric_value(message: &str, key: &str) -> Option<f64> {
    message
        .split_whitespace()
        .find_map(|token| token.strip_prefix(&format!("{key}=")))
        .and_then(|value| value.parse::<f64>().ok())
}

fn diagnostic_metric(
    run: &runmat_runtime::analysis::AnalysisRunResult,
    code: &str,
    key: &str,
) -> Option<f64> {
    run.run
        .diagnostics
        .iter()
        .find(|diag| diag.code == code)
        .and_then(|diag| parse_metric_value(&diag.message, key))
}

fn analysis_result_field_max_abs(run: &AnalysisRunResult, field_id: &str) -> Option<f64> {
    analysis_result_field(run, field_id)
        .and_then(AnalysisField::as_host_f64)
        .map(|values| {
            values
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max)
        })
}

#[allow(clippy::too_many_arguments)]
fn push_threshold_assertion(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    name: &str,
    source_diagnostic: &str,
    observed: Option<f64>,
    min_allowed: Option<f64>,
    max_allowed: Option<f64>,
) {
    let passed = observed
        .map(|value| {
            min_allowed.map(|min| value >= min).unwrap_or(true)
                && max_allowed.map(|max| value <= max).unwrap_or(true)
        })
        .unwrap_or(false);
    assertions.push(ThresholdAssertionRecord {
        name: name.to_string(),
        source_diagnostic: source_diagnostic.to_string(),
        observed,
        min_allowed,
        max_allowed,
        passed,
    });
    if !passed {
        failures.push(format!(
            "threshold assertion failed for fixture {}: {} observed={:?} min={:?} max={:?}",
            fixture_id, name, observed, min_allowed, max_allowed
        ));
    }
}

fn push_plastic_state_threshold_assertions(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
    prefix: &str,
) {
    push_nonlinear_state_topology_threshold_assertions(
        fixture_id, assertions, failures, run, prefix,
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_active_element_count"),
        "FEA_NONLINEAR_STATE",
        diagnostic_metric(run, "FEA_NONLINEAR_STATE", "plastic_active_element_count"),
        Some(1.0),
        None,
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_max_equivalent_plastic_strain"),
        "FEA_NONLINEAR_STATE",
        diagnostic_metric(run, "FEA_NONLINEAR_STATE", "max_equivalent_plastic_strain"),
        Some(1.0e-12),
        None,
    );
}

fn push_contact_state_threshold_assertions(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
    prefix: &str,
) {
    push_nonlinear_state_topology_threshold_assertions(
        fixture_id, assertions, failures, run, prefix,
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_active_entity_count"),
        "FEA_NONLINEAR_STATE",
        diagnostic_metric(run, "FEA_NONLINEAR_STATE", "contact_active_entity_count"),
        Some(1.0),
        None,
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_max_contact_pressure"),
        "FEA_NONLINEAR_STATE",
        diagnostic_metric(run, "FEA_NONLINEAR_STATE", "max_contact_pressure"),
        Some(1.0e-12),
        None,
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_min_contact_gap"),
        "FEA_NONLINEAR_STATE",
        diagnostic_metric(run, "FEA_NONLINEAR_STATE", "min_contact_gap"),
        Some(0.0),
        None,
    );
}

fn push_nonlinear_state_topology_threshold_assertions(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
    prefix: &str,
) {
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_topology_element_count"),
        "FEA_NONLINEAR_STATE_TOPOLOGY",
        diagnostic_metric(run, "FEA_NONLINEAR_STATE_TOPOLOGY", "element_count"),
        Some(1.0),
        None,
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_topology_active_recovery_edge_count"),
        "FEA_NONLINEAR_STATE_TOPOLOGY",
        diagnostic_metric(
            run,
            "FEA_NONLINEAR_STATE_TOPOLOGY",
            "active_recovery_edge_count",
        ),
        Some(1.0),
        None,
    );
}

fn push_plastic_known_answer_threshold_assertions(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
    prefix: &str,
) {
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_monotonic_equivalent_plastic_strain_fraction"),
        "FEA_PLASTIC_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_PLASTIC_KNOWN_ANSWER",
            "monotonic_equivalent_plastic_strain_fraction",
        ),
        Some(1.0),
        Some(1.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_active_element_coverage_ratio"),
        "FEA_PLASTIC_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_PLASTIC_KNOWN_ANSWER",
            "active_element_coverage_ratio",
        ),
        Some(1.0e-6),
        Some(1.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_final_to_peak_equivalent_plastic_strain_ratio"),
        "FEA_PLASTIC_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_PLASTIC_KNOWN_ANSWER",
            "final_to_peak_equivalent_plastic_strain_ratio",
        ),
        Some(0.999_999),
        Some(1.000_001),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_known_answer_coverage_ratio"),
        "FEA_PLASTIC_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_PLASTIC_KNOWN_ANSWER",
            "known_answer_coverage_ratio",
        ),
        Some(1.0),
        Some(1.0),
    );
}

fn push_contact_known_answer_threshold_assertions(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
    prefix: &str,
) {
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_pressure_gap_consistency_residual"),
        "FEA_CONTACT_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_CONTACT_KNOWN_ANSWER",
            "pressure_gap_consistency_residual",
        ),
        Some(0.0),
        Some(1.0e-12),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_active_entity_coverage_ratio"),
        "FEA_CONTACT_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_CONTACT_KNOWN_ANSWER",
            "active_entity_coverage_ratio",
        ),
        Some(1.0),
        Some(1.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_nonpenetration_gap_min"),
        "FEA_CONTACT_KNOWN_ANSWER",
        diagnostic_metric(run, "FEA_CONTACT_KNOWN_ANSWER", "nonpenetration_gap_min"),
        Some(0.0),
        None,
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_open_gap_pressure_residual"),
        "FEA_CONTACT_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_CONTACT_KNOWN_ANSWER",
            "open_gap_pressure_residual",
        ),
        Some(0.0),
        Some(1.0e-12),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_pressure_gap_complementarity_residual"),
        "FEA_CONTACT_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_CONTACT_KNOWN_ANSWER",
            "pressure_gap_complementarity_residual",
        ),
        Some(0.0),
        Some(1.0e-12),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_closed_entity_coverage_ratio"),
        "FEA_CONTACT_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_CONTACT_KNOWN_ANSWER",
            "closed_entity_coverage_ratio",
        ),
        Some(1.0),
        Some(1.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_friction_coefficient"),
        "FEA_CONTACT_KNOWN_ANSWER",
        diagnostic_metric(run, "FEA_CONTACT_KNOWN_ANSWER", "friction_coefficient"),
        Some(0.0),
        Some(0.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_known_answer_coverage_ratio"),
        "FEA_CONTACT_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_CONTACT_KNOWN_ANSWER",
            "known_answer_coverage_ratio",
        ),
        Some(1.0),
        Some(1.0),
    );
}

fn push_thermal_standalone_threshold_assertions(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
) {
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_max_residual_norm",
        "FEA_THERMAL_STABILITY",
        diagnostic_metric(run, "FEA_THERMAL_STABILITY", "max_residual_norm"),
        Some(0.0),
        Some(7.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_min_temperature_k",
        "FEA_THERMAL_STABILITY",
        diagnostic_metric(run, "FEA_THERMAL_STABILITY", "min_temperature_k"),
        Some(290.0),
        Some(360.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_max_temperature_k",
        "FEA_THERMAL_STABILITY",
        diagnostic_metric(run, "FEA_THERMAL_STABILITY", "max_temperature_k"),
        Some(300.0),
        Some(370.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_heat_balance_residual_ratio",
        "FEA_THERMAL_HEAT_BALANCE",
        diagnostic_metric(
            run,
            "FEA_THERMAL_HEAT_BALANCE",
            "heat_balance_residual_ratio",
        ),
        Some(0.0),
        Some(0.25),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_conductivity_spread_ratio",
        "FEA_THERMAL_CONSTITUTIVE",
        diagnostic_metric(run, "FEA_THERMAL_CONSTITUTIVE", "conductivity_spread_ratio"),
        Some(1.0),
        Some(1.5),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_heat_capacity_spread_ratio",
        "FEA_THERMAL_CONSTITUTIVE",
        diagnostic_metric(
            run,
            "FEA_THERMAL_CONSTITUTIVE",
            "heat_capacity_spread_ratio",
        ),
        Some(1.0),
        Some(1.5),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_spatial_gradient_index",
        "FEA_THERMAL_OUTCOME",
        diagnostic_metric(run, "FEA_THERMAL_OUTCOME", "spatial_gradient_index"),
        Some(0.4),
        Some(1.4),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_monotonic_response_fraction",
        "FEA_THERMAL_OUTCOME",
        diagnostic_metric(run, "FEA_THERMAL_OUTCOME", "monotonic_response_fraction"),
        Some(0.9),
        Some(1.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_response_realization_ratio",
        "FEA_THERMAL_OUTCOME",
        diagnostic_metric(
            run,
            "FEA_THERMAL_OUTCOME",
            "thermal_response_realization_ratio",
        ),
        Some(0.6),
        Some(1.2),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_slab_linear_profile_rms_ratio",
        "FEA_THERMAL_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_THERMAL_KNOWN_ANSWER",
            "slab_linear_profile_rms_ratio",
        ),
        Some(0.0),
        Some(0.12),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_slab_monotonic_edge_fraction",
        "FEA_THERMAL_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_THERMAL_KNOWN_ANSWER",
            "slab_monotonic_edge_fraction",
        ),
        Some(0.95),
        Some(1.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_lumped_response_error_ratio",
        "FEA_THERMAL_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_THERMAL_KNOWN_ANSWER",
            "lumped_response_error_ratio",
        ),
        Some(0.0),
        Some(0.45),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_source_response_sign_alignment",
        "FEA_THERMAL_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_THERMAL_KNOWN_ANSWER",
            "source_response_sign_alignment",
        ),
        Some(1.0),
        Some(1.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_source_coverage_ratio",
        "FEA_THERMAL_SOURCE_BOUNDARY_MODEL",
        diagnostic_metric(
            run,
            "FEA_THERMAL_SOURCE_BOUNDARY_MODEL",
            "thermal_source_coverage_ratio",
        ),
        Some(1.0),
        Some(1.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_boundary_coverage_ratio",
        "FEA_THERMAL_SOURCE_BOUNDARY_MODEL",
        diagnostic_metric(
            run,
            "FEA_THERMAL_SOURCE_BOUNDARY_MODEL",
            "thermal_boundary_coverage_ratio",
        ),
        Some(1.0),
        Some(1.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_prescribed_temperature_count",
        "FEA_THERMAL_SOURCE_BOUNDARY_MODEL",
        diagnostic_metric(
            run,
            "FEA_THERMAL_SOURCE_BOUNDARY_MODEL",
            "prescribed_temperature_count",
        ),
        Some(1.0),
        None,
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_heat_flux_boundary_count",
        "FEA_THERMAL_SOURCE_BOUNDARY_MODEL",
        diagnostic_metric(
            run,
            "FEA_THERMAL_SOURCE_BOUNDARY_MODEL",
            "heat_flux_boundary_count",
        ),
        Some(1.0),
        None,
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "thermal_standalone_convection_boundary_count",
        "FEA_THERMAL_SOURCE_BOUNDARY_MODEL",
        diagnostic_metric(
            run,
            "FEA_THERMAL_SOURCE_BOUNDARY_MODEL",
            "convection_boundary_count",
        ),
        Some(1.0),
        None,
    );
}

fn push_electro_thermal_source_coupling_threshold_assertions(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
    prefix: &str,
) {
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_joule_heat_realization_ratio"),
        "FEA_ET_THERMAL_SOURCE_COUPLING",
        diagnostic_metric(
            run,
            "FEA_ET_THERMAL_SOURCE_COUPLING",
            "joule_heat_realization_ratio",
        ),
        Some(0.999_999),
        Some(1.000_001),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_joule_source_coverage_ratio"),
        "FEA_ET_THERMAL_SOURCE_COUPLING",
        diagnostic_metric(
            run,
            "FEA_ET_THERMAL_SOURCE_COUPLING",
            "joule_source_coverage_ratio",
        ),
        Some(1.0),
        Some(1.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_thermal_temperature_source_alignment"),
        "FEA_ET_THERMAL_SOURCE_COUPLING",
        diagnostic_metric(
            run,
            "FEA_ET_THERMAL_SOURCE_COUPLING",
            "thermal_temperature_source_alignment",
        ),
        Some(0.999_999),
        Some(1.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        &format!("{prefix}_thermal_source_residual_ratio"),
        "FEA_ET_THERMAL_SOURCE_COUPLING",
        diagnostic_metric(
            run,
            "FEA_ET_THERMAL_SOURCE_COUPLING",
            "thermal_source_residual_ratio",
        ),
        Some(0.0),
        Some(1.0e-10),
    );
}

fn push_linear_structural_threshold_assertions(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
) {
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "structural_normalized_residual_norm",
        "FEA_STRUCTURAL_RESIDUAL",
        diagnostic_metric(run, "FEA_STRUCTURAL_RESIDUAL", "normalized_residual_norm"),
        Some(0.0),
        Some(1.0e-6),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "structural_total_strain_energy",
        "FEA_STRUCTURAL_ENERGY",
        diagnostic_metric(run, "FEA_STRUCTURAL_ENERGY", "total_strain_energy").or_else(|| {
            analysis_result_field_max_abs(run, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
        }),
        Some(0.0),
        None,
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "structural_work_energy_ratio",
        "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER",
            "work_energy_ratio",
        ),
        Some(0.999_999),
        Some(1.000_001),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "structural_work_energy_residual_ratio",
        "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER",
            "work_energy_residual_ratio",
        ),
        Some(0.0),
        Some(1.0e-8),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "structural_known_answer_coverage_ratio",
        "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER",
            "known_answer_coverage_ratio",
        ),
        Some(1.0),
        Some(1.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "structural_active_stiffness_edge_count",
        "FEA_STRUCTURAL_FIELD_RECOVERY",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_FIELD_RECOVERY",
            "active_stiffness_edge_count",
        ),
        Some(1.0),
        None,
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "structural_recovery_element_count",
        "FEA_STRUCTURAL_FIELD_RECOVERY",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_FIELD_RECOVERY",
            "recovery_element_count",
        ),
        Some(1.0),
        None,
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "structural_max_edge_displacement_jump",
        "FEA_STRUCTURAL_FIELD_RECOVERY",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_FIELD_RECOVERY",
            "max_edge_displacement_jump",
        ),
        Some(1.0e-12),
        Some(1.0e-2),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "structural_von_mises_peak_pa",
        FEA_FIELD_STRUCTURAL_VON_MISES,
        analysis_result_field_max_abs(run, FEA_FIELD_STRUCTURAL_VON_MISES),
        Some(1.0e5),
        Some(1.0e9),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "structural_stress_tensor_peak_pa",
        FEA_FIELD_STRUCTURAL_STRESS,
        analysis_result_field_max_abs(run, FEA_FIELD_STRUCTURAL_STRESS),
        Some(1.0e5),
        Some(1.0e9),
    );
}

fn push_structural_reference_kinematics_threshold_assertions(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    run: &AnalysisRunResult,
) {
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "structural_reference_transverse_displacement_leakage_ratio",
        "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
            "transverse_displacement_leakage_ratio",
        ),
        Some(0.0),
        Some(0.1),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "structural_reference_primary_stress_component_ratio",
        "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
            "primary_stress_component_ratio",
        ),
        Some(0.5),
        Some(1.0),
    );
    push_threshold_assertion(
        fixture_id,
        assertions,
        failures,
        "structural_reference_directional_coverage_ratio",
        "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
        diagnostic_metric(
            run,
            "FEA_STRUCTURAL_REFERENCE_KINEMATICS",
            "directional_reference_coverage_ratio",
        ),
        Some(1.0),
        Some(1.0),
    );
}

fn validate_fallback_event_schema(event: &str) -> bool {
    let parts: Vec<&str> = event.splitn(3, ':').collect();
    if parts.len() != 3 {
        return false;
    }
    let category_ok = matches!(
        parts[0],
        "BACKEND_NO_PROVIDER" | "BACKEND_UPLOAD_FAILED" | "SOLVER_BACKEND_FALLBACK"
    );
    let stage_ok = if parts[0] == "SOLVER_BACKEND_FALLBACK" {
        parts[1].starts_with("requested=")
    } else {
        is_namespace_field_id(parts[1])
    };
    let reason_ok = !parts[2].is_empty();
    category_ok && stage_ok && reason_ok
}

fn is_namespace_field_id(value: &str) -> bool {
    value.contains('.')
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '.')
}

fn compute_parity(left: &[f64], right: &[f64]) -> ParitySummary {
    let mut max_abs = 0.0;
    let mut max_rel = 0.0;
    for (lhs, rhs) in left.iter().zip(right.iter()) {
        let abs = (lhs - rhs).abs();
        let scale = lhs.abs().max(rhs.abs()).max(1.0);
        let rel = abs / scale;
        if abs > max_abs {
            max_abs = abs;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    ParitySummary {
        max_abs_diff: max_abs,
        max_rel_diff: max_rel,
    }
}

pub(super) fn run_fixture(
    spec: &FixtureSpec,
    filesystem_root: Option<&PathBuf>,
) -> FixtureRunRecord {
    let mut model = (spec.model)();
    configure_model_for_fixture(spec.id, &mut model);
    ensure_thermo_field_artifacts_for_fixture(spec.id, &model);
    let mut failures = Vec::new();

    let validate_start = Instant::now();
    let validate_result = analysis_validate(
        &model,
        UnitSystem::Meter,
        &ReferenceFrame::Global,
        OperationContext::new(Some(format!("trace-validate-{}", spec.id)), None),
    );
    let _validate_ms = validate_start.elapsed().as_secs_f64() * 1_000.0;

    let mut validate_ok = false;
    let mut validate_error_code = None;

    match validate_result {
        Ok(_) => {
            validate_ok = true;
            if let Some(expected) = spec.expect_validate_error {
                failures.push(format!(
                    "expected validate error code {expected}, but validate succeeded"
                ));
            }
        }
        Err(err) => {
            validate_error_code = Some(err.error_code.clone());
            if let Some(expected) = spec.expect_validate_error {
                if err.error_code != expected {
                    failures.push(format!(
                        "validate error mismatch: expected {expected}, got {}",
                        err.error_code
                    ));
                }
            } else {
                failures.push(format!(
                    "unexpected validate error code {} for fixture {}",
                    err.error_code, spec.id
                ));
            }
        }
    }

    let mut run_ok = false;
    let mut run_error_code = None;
    let mut cpu_run_ms = None;
    let mut gpu_run_ms = None;
    let mut gpu_fallback_events = Vec::new();
    let mut gpu_displacement_residency = None;
    let mut gpu_solver_host_sync_count = None;
    let mut gpu_solver_device_apply_k_ratio = None;
    let mut gpu_speedup_ratio = None;
    let mut gpu_solver_backend = None;
    let mut gpu_transient_cache_hit_ratio = None;
    let mut gpu_transient_cache_misses = None;
    let mut gpu_transient_cache_entries = None;
    let mut gpu_solver_prepared_build_ms = None;
    let mut gpu_solver_solve_ms = None;
    let mut gpu_solver_fallback_apply_count = None;
    let mut prep_calibration_profile = None;
    let mut prep_calibration_fingerprint = None;
    let mut prep_acceptance_score = None;
    let mut prep_acceptance_passed = None;
    let mut prep_acceptance_fingerprint = None;
    let mut thermo_coupling_enabled = None;
    let mut thermo_coupling_fingerprint = None;
    let mut thermo_constitutive_temperature_factor = None;
    let mut thermo_effective_modulus_scale = None;
    let mut thermo_constitutive_material_spread_ratio = None;
    let mut thermo_assignment_heterogeneity_index = None;
    let mut thermo_region_delta_count = None;
    let mut thermo_spatial_coverage_ratio = None;
    let mut thermo_field_extrapolation_ratio = None;
    let mut thermo_field_clamp_ratio = None;
    let thermo_field_artifact_id = model
        .thermo_mechanical
        .as_ref()
        .and_then(|domain| domain.field_artifact_id.clone());
    let mut thermo_field_artifact_approved = None;
    let mut thermo_field_artifact_age_days = None;
    let mut thermo_field_artifact_provenance_valid = None;
    if thermo_field_artifact_id.is_some() {
        thermo_field_artifact_approved = Some(true);
        thermo_field_artifact_age_days = Some(0.0);
        thermo_field_artifact_provenance_valid = Some(true);
    }
    let mut thermo_transient_severity = None;
    let mut thermo_nonlinear_severity = None;
    let mut electro_thermal_coupling_enabled = None;
    let mut electro_thermal_coupling_fingerprint = None;
    let mut electro_joule_heating_scale = None;
    let mut electro_conductivity_spread_ratio = None;
    let mut electro_transient_severity = None;
    let mut electro_transient_time_scale_mean = None;
    let mut electro_nonlinear_severity = None;
    let mut electro_nonlinear_time_scale_mean = None;
    let mut plastic_nonlinear_severity = None;
    let mut plastic_nonlinear_severity_mean = None;
    let mut plastic_load_realization_ratio = None;
    let mut plastic_load_amplification_ratio = None;
    let mut contact_nonlinear_severity = None;
    let mut contact_nonlinear_severity_mean = None;
    let mut contact_load_realization_ratio = None;
    let mut contact_load_amplification_ratio = None;
    let mut thermal_max_residual_norm = None;
    let mut thermal_min_temperature_k = None;
    let mut thermal_max_temperature_k = None;
    let mut thermal_conductivity_spread_ratio = None;
    let mut thermal_heat_capacity_spread_ratio = None;
    let mut thermal_spatial_gradient_index = None;
    let mut thermal_monotonic_response_fraction = None;
    let mut thermal_response_realization_ratio = None;
    let mut electromagnetic_enabled = None;
    let mut electromagnetic_formulation_coverage_ratio = None;
    let mut electromagnetic_magnetostatic_curl_curl_coverage_ratio = None;
    let mut electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio = None;
    let mut electromagnetic_full_wave_displacement_current_coverage_ratio = None;
    let mut electromagnetic_displacement_to_conduction_ratio = None;
    let mut electromagnetic_material_frequency_response_coverage_ratio = None;
    let mut electromagnetic_reference_frequency_hz = None;
    let mut electromagnetic_applied_current_a = None;
    let mut electromagnetic_solve_quality = None;
    let mut electromagnetic_conductivity_spread_ratio = None;
    let mut electromagnetic_relative_permittivity_spread_ratio = None;
    let mut electromagnetic_relative_permeability_spread_ratio = None;
    let mut electromagnetic_material_heterogeneity_index = None;
    let mut electromagnetic_assignment_coverage_ratio = None;
    let mut electromagnetic_assigned_coefficient_coverage_ratio = None;
    let mut electromagnetic_fallback_coefficient_ratio = None;
    let mut electromagnetic_region_coefficient_contrast_index = None;
    let mut electromagnetic_condition_number_estimate = None;
    let mut electromagnetic_source_realization_ratio = None;
    let mut electromagnetic_source_region_coverage_ratio = None;
    let mut electromagnetic_source_material_alignment_ratio = None;
    let mut electromagnetic_source_localization_ratio = None;
    let mut electromagnetic_source_overlap_ratio = None;
    let mut electromagnetic_source_interference_index = None;
    let mut electromagnetic_boundary_anchor_ratio = None;
    let mut electromagnetic_boundary_condition_localization_ratio = None;
    let mut electromagnetic_ground_anchor_effectiveness_ratio = None;
    let mut electromagnetic_insulation_leakage_ratio = None;
    let mut electromagnetic_flux_divergence_ratio = None;
    let mut electromagnetic_energy_imbalance_ratio = None;
    let mut electromagnetic_boundary_energy_ratio = None;
    let mut electromagnetic_boundary_penalty_conditioning_contribution = None;
    let mut electromagnetic_source_region_energy_consistency_ratio = None;
    let mut electromagnetic_real_residual_norm = None;
    let mut electromagnetic_imag_residual_norm = None;
    let mut electromagnetic_sweep_count = None;
    let mut electromagnetic_resonance_peak_frequency_hz = None;
    let mut electromagnetic_resonance_peak_flux_density = None;
    let mut electromagnetic_resonance_bandwidth_hz = None;
    let mut electromagnetic_resonance_quality_factor = None;
    let mut electromagnetic_resonance_flux_gain = None;
    let mut publishable = None;
    let mut parity = None;
    let mut threshold_assertions = Vec::new();

    if spec.expect_validate_error.is_none() && spec.expect_run_error.is_none() {
        let cpu_start = Instant::now();
        let cpu_result = run_fixture_cpu(spec, &model);
        cpu_run_ms = Some(cpu_start.elapsed().as_secs_f64() * 1_000.0);

        let cpu_envelope = match cpu_result {
            Ok(value) => value,
            Err(err) => {
                failures.push(format!(
                    "unexpected CPU run failure for fixture {}: {}",
                    spec.id, err.error_code
                ));
                return FixtureRunRecord {
                    fixture_id: spec.id.to_string(),
                    validate_ok,
                    validate_error_code,
                    run_ok,
                    run_error_code,
                    cpu_run_ms,
                    gpu_run_ms,
                    gpu_fallback_events,
                    gpu_displacement_residency,
                    gpu_solver_host_sync_count,
                    gpu_solver_device_apply_k_ratio,
                    gpu_speedup_ratio,
                    gpu_solver_backend,
                    gpu_transient_cache_hit_ratio,
                    gpu_transient_cache_misses,
                    gpu_transient_cache_entries,
                    gpu_solver_prepared_build_ms,
                    gpu_solver_solve_ms,
                    gpu_solver_fallback_apply_count,
                    prep_calibration_profile,
                    prep_calibration_fingerprint,
                    prep_acceptance_score,
                    prep_acceptance_passed,
                    prep_acceptance_fingerprint,
                    thermo_coupling_enabled,
                    thermo_coupling_fingerprint,
                    thermo_constitutive_temperature_factor,
                    thermo_effective_modulus_scale,
                    thermo_constitutive_material_spread_ratio,
                    thermo_assignment_heterogeneity_index,
                    thermo_region_delta_count,
                    thermo_spatial_coverage_ratio,
                    thermo_field_extrapolation_ratio,
                    thermo_field_clamp_ratio,
                    thermo_field_artifact_id,
                    thermo_field_artifact_approved,
                    thermo_field_artifact_age_days,
                    thermo_field_artifact_provenance_valid,
                    thermo_transient_severity,
                    thermo_nonlinear_severity,
                    electro_thermal_coupling_enabled,
                    electro_thermal_coupling_fingerprint,
                    electro_joule_heating_scale,
                    electro_conductivity_spread_ratio,
                    electro_transient_severity,
                    electro_transient_time_scale_mean,
                    electro_nonlinear_severity,
                    electro_nonlinear_time_scale_mean,
                    plastic_nonlinear_severity,
                    plastic_nonlinear_severity_mean,
                    plastic_load_realization_ratio,
                    plastic_load_amplification_ratio,
                    contact_nonlinear_severity,
                    contact_nonlinear_severity_mean,
                    contact_load_realization_ratio,
                    contact_load_amplification_ratio,
                    thermal_max_residual_norm,
                    thermal_min_temperature_k,
                    thermal_max_temperature_k,
                    thermal_conductivity_spread_ratio,
                    thermal_heat_capacity_spread_ratio,
                    thermal_spatial_gradient_index,
                    thermal_monotonic_response_fraction,
                    thermal_response_realization_ratio,
                    electromagnetic_enabled,
                    electromagnetic_formulation_coverage_ratio,
                    electromagnetic_magnetostatic_curl_curl_coverage_ratio,
                    electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio,
                    electromagnetic_full_wave_displacement_current_coverage_ratio,
                    electromagnetic_displacement_to_conduction_ratio,
                    electromagnetic_material_frequency_response_coverage_ratio,
                    electromagnetic_reference_frequency_hz,
                    electromagnetic_applied_current_a,
                    electromagnetic_solve_quality,
                    electromagnetic_conductivity_spread_ratio,
                    electromagnetic_relative_permittivity_spread_ratio,
                    electromagnetic_relative_permeability_spread_ratio,
                    electromagnetic_material_heterogeneity_index,
                    electromagnetic_assignment_coverage_ratio,
                    electromagnetic_assigned_coefficient_coverage_ratio,
                    electromagnetic_fallback_coefficient_ratio,
                    electromagnetic_region_coefficient_contrast_index,
                    electromagnetic_condition_number_estimate,
                    electromagnetic_source_realization_ratio,
                    electromagnetic_source_region_coverage_ratio,
                    electromagnetic_source_material_alignment_ratio,
                    electromagnetic_source_localization_ratio,
                    electromagnetic_source_overlap_ratio,
                    electromagnetic_source_interference_index,
                    electromagnetic_boundary_anchor_ratio,
                    electromagnetic_boundary_condition_localization_ratio,
                    electromagnetic_ground_anchor_effectiveness_ratio,
                    electromagnetic_insulation_leakage_ratio,
                    electromagnetic_flux_divergence_ratio,
                    electromagnetic_energy_imbalance_ratio,
                    electromagnetic_boundary_energy_ratio,
                    electromagnetic_boundary_penalty_conditioning_contribution,
                    electromagnetic_source_region_energy_consistency_ratio,
                    electromagnetic_real_residual_norm,
                    electromagnetic_imag_residual_norm,
                    electromagnetic_sweep_count,
                    electromagnetic_resonance_peak_frequency_hz,
                    electromagnetic_resonance_peak_flux_density,
                    electromagnetic_resonance_bandwidth_hz,
                    electromagnetic_resonance_quality_factor,
                    electromagnetic_resonance_flux_gain,
                    publishable,
                    parity,
                    threshold_assertions,
                    failures,
                };
            }
        };
        run_ok = true;
        publishable = Some(cpu_envelope.data.publishable);

        if let Some(expected_publishable) = spec.expected_publishable {
            if cpu_envelope.data.publishable != expected_publishable {
                failures.push(format!(
                    "cpu publishable mismatch: expected {expected_publishable}, got {}",
                    cpu_envelope.data.publishable
                ));
            }
        }

        if matches!(spec.run_kind, AnalysisRunKind::Modal) {
            let observed = cpu_envelope
                .data
                .modal_results
                .as_ref()
                .and_then(|modal| modal.residual_norms.iter().copied().reduce(f64::max));
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "modal_max_residual_norm",
                "FEA_MODAL_CONVERGENCE",
                observed,
                None,
                Some(1.0e-1),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "modal_adjacent_mode_pair_count",
                "FEA_MODAL_CLUSTER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_MODAL_CLUSTER",
                    "adjacent_mode_pair_count",
                ),
                spec.modal_mode_count
                    .and_then(|mode_count| mode_count.checked_sub(1))
                    .map(|count| count as f64),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "modal_cluster_coverage_ratio",
                "FEA_MODAL_CLUSTER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_MODAL_CLUSTER",
                    "cluster_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
        }
        if let Some(max_offdiag) = spec.max_modal_orthogonality_offdiag {
            let observed = diagnostic_metric(
                &cpu_envelope.data,
                "FEA_MODAL_ORTHOGONALITY",
                "max_m_orthogonality_offdiag",
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "modal_max_m_orthogonality_offdiag",
                "FEA_MODAL_ORTHOGONALITY",
                observed,
                None,
                Some(max_offdiag),
            );
        }
        if let Some(min_separation) = spec.min_modal_relative_frequency_separation {
            let observed = diagnostic_metric(
                &cpu_envelope.data,
                "FEA_MODAL_SEPARATION",
                "min_relative_frequency_separation",
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "modal_min_relative_frequency_separation",
                "FEA_MODAL_SEPARATION",
                observed,
                Some(min_separation),
                None,
            );
        }
        if spec.id.starts_with("acoustic_harmonic_") {
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_normalized_residual_norm",
                "FEA_ACOUSTIC_HELMHOLTZ_RESIDUAL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_HELMHOLTZ_RESIDUAL",
                    "normalized_residual_norm",
                ),
                None,
                Some(1.0e-3),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_drive_frequency_hz",
                "FEA_ACOUSTIC_HARMONIC_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_HARMONIC_RESPONSE",
                    "drive_frequency_hz",
                ),
                Some(50.0),
                Some(20_000.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_peak_pressure_pa",
                "FEA_ACOUSTIC_HARMONIC_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_HARMONIC_RESPONSE",
                    "peak_pressure_pa",
                ),
                Some(1.0e-12),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_domain_node_count",
                "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                    "domain_node_count",
                ),
                Some(3.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_domain_edge_count",
                "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                    "domain_edge_count",
                ),
                Some(2.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_domain_active_dimension_count",
                "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                    "domain_active_dimension_count",
                ),
                Some(2.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_boundary_node_count",
                "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_DOMAIN_ASSEMBLY",
                    "boundary_node_count",
                ),
                Some(2.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_material_coverage_ratio",
                "FEA_ACOUSTIC_HARMONIC_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_HARMONIC_RESPONSE",
                    "acoustic_material_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_boundary_coverage_ratio",
                "FEA_ACOUSTIC_BOUNDARY_MODEL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_BOUNDARY_MODEL",
                    "acoustic_boundary_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_radiation_boundary_count",
                "FEA_ACOUSTIC_BOUNDARY_MODEL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_BOUNDARY_MODEL",
                    "radiation_boundary_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_impedance_boundary_count",
                "FEA_ACOUSTIC_BOUNDARY_MODEL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_BOUNDARY_MODEL",
                    "impedance_boundary_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_frequency_response_sweep_count",
                "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                    "sweep_count",
                ),
                Some(3.0),
                Some(3.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_frequency_response_coverage_ratio",
                "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                    "response_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_sweep_bandwidth_hz",
                "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                    "sweep_bandwidth_hz",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_sweep_peak_pressure_pa",
                "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                    "sweep_peak_pressure_pa",
                ),
                Some(1.0e-12),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_sweep_max_residual_norm",
                "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_FREQUENCY_RESPONSE",
                    "sweep_max_residual_norm",
                ),
                None,
                Some(1.0e-3),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_tube_mode_alignment_error_ratio",
                "FEA_ACOUSTIC_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_KNOWN_ANSWER",
                    "tube_mode_alignment_error_ratio",
                ),
                Some(0.0),
                Some(0.5),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_tube_pressure_variation_ratio",
                "FEA_ACOUSTIC_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_KNOWN_ANSWER",
                    "tube_pressure_variation_ratio",
                ),
                Some(1.0e-12),
                Some(1.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_cavity_mode_spacing_ratio",
                "FEA_ACOUSTIC_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_KNOWN_ANSWER",
                    "cavity_mode_spacing_ratio",
                ),
                Some(1.0e-9),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_cavity_reference_mode_count",
                "FEA_ACOUSTIC_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_KNOWN_ANSWER",
                    "cavity_reference_mode_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "acoustic_known_answer_coverage_ratio",
                "FEA_ACOUSTIC_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_ACOUSTIC_KNOWN_ANSWER",
                    "known_answer_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
        }
        if matches!(
            spec.id,
            "cantilever_gpu_provider"
                | "cantilever_gpu_fallback"
                | "cantilever_load_sweep_gpu_provider"
                | "cantilever_large_load_sweep_gpu_provider"
                | "structural_axial_bar_reference_gpu_provider"
                | "structural_beam_bending_reference_gpu_provider"
        ) {
            push_linear_structural_threshold_assertions(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                &cpu_envelope.data,
            );
        }
        if matches!(
            spec.id,
            "structural_axial_bar_reference_gpu_provider"
                | "structural_beam_bending_reference_gpu_provider"
        ) {
            push_structural_reference_kinematics_threshold_assertions(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                &cpu_envelope.data,
            );
        }
        if let Some(max_residual) = spec.max_transient_residual_norm {
            let observed = diagnostic_metric(
                &cpu_envelope.data,
                "FEA_TRANSIENT_STABILITY",
                "max_residual_norm",
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "transient_max_residual_norm",
                "FEA_TRANSIENT_STABILITY",
                observed,
                None,
                Some(max_residual),
            );
        }
        if let Some(max_growth) = spec.max_transient_energy_growth_ratio {
            let observed = diagnostic_metric(
                &cpu_envelope.data,
                "FEA_TRANSIENT_ENERGY",
                "max_energy_growth_ratio",
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "transient_max_energy_growth_ratio",
                "FEA_TRANSIENT_ENERGY",
                observed,
                None,
                Some(max_growth),
            );
        }
        if spec.id.starts_with("transient_") {
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "transient_initial_total_energy",
                "FEA_TRANSIENT_ENERGY_BALANCE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_TRANSIENT_ENERGY_BALANCE",
                    "initial_total_energy",
                ),
                Some(0.0),
                Some(1.0e-12),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "transient_final_total_energy",
                "FEA_TRANSIENT_ENERGY_BALANCE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_TRANSIENT_ENERGY_BALANCE",
                    "final_total_energy",
                ),
                Some(0.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "transient_max_total_energy",
                "FEA_TRANSIENT_ENERGY_BALANCE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_TRANSIENT_ENERGY_BALANCE",
                    "max_total_energy",
                ),
                Some(0.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "transient_energy_balance_growth_ratio",
                "FEA_TRANSIENT_ENERGY_BALANCE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_TRANSIENT_ENERGY_BALANCE",
                    "energy_growth_ratio",
                ),
                Some(1.0),
                Some(spec.max_transient_energy_growth_ratio.unwrap_or(10.0)),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "transient_max_step_energy_jump_ratio",
                "FEA_TRANSIENT_ENERGY_BALANCE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_TRANSIENT_ENERGY_BALANCE",
                    "max_step_energy_jump_ratio",
                ),
                None,
                Some(10.0),
            );
        }
        let is_cfd_fixture =
            spec.id.starts_with("cfd_steady_") || spec.id.starts_with("cfd_transient_");
        if is_cfd_fixture {
            let expected_profile_point_count = if spec.id.starts_with("cfd_transient_") {
                2.0
            } else {
                0.0
            };
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_reference_density_kg_per_m3",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "density"),
                Some(1.20),
                Some(1.25),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_dynamic_viscosity_pa_s",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "viscosity"),
                Some(1.0e-5),
                Some(3.0e-5),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_inlet_velocity_m_per_s",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "inlet_velocity"),
                Some(4.0),
                Some(6.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_turbulence_intensity",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "turbulence_intensity"),
                Some(0.04),
                Some(0.08),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_reynolds_number",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "reynolds_number"),
                Some(2.0e5),
                Some(5.0e5),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_profile_point_count",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "profile_point_count"),
                Some(expected_profile_point_count),
                Some(expected_profile_point_count),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_max_momentum_residual",
                "FEA_CFD_RESIDUAL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_RESIDUAL",
                    "max_momentum_residual",
                ),
                None,
                Some(1.0e-4),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_max_continuity_residual",
                "FEA_CFD_RESIDUAL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_RESIDUAL",
                    "max_continuity_residual",
                ),
                None,
                Some(1.0e-4),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_mass_balance_residual",
                "FEA_CFD_ASSEMBLY",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_ASSEMBLY",
                    "mass_balance_residual",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_pressure_drop_pa",
                "FEA_CFD_ASSEMBLY",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_ASSEMBLY", "pressure_drop_pa"),
                Some(0.1),
                Some(10.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_control_volume_count",
                "FEA_CFD_ASSEMBLY",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_ASSEMBLY",
                    "control_volume_count",
                ),
                Some(2.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_inlet_boundary_count",
                "FEA_CFD_BOUNDARY_CONDITIONS",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_BOUNDARY_CONDITIONS",
                    "inlet_boundary_count",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_outlet_boundary_count",
                "FEA_CFD_BOUNDARY_CONDITIONS",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_BOUNDARY_CONDITIONS",
                    "outlet_boundary_count",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_wall_boundary_count",
                "FEA_CFD_BOUNDARY_CONDITIONS",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_BOUNDARY_CONDITIONS",
                    "wall_boundary_count",
                ),
                Some(2.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_boundary_coverage_ratio",
                "FEA_CFD_BOUNDARY_CONDITIONS",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_BOUNDARY_CONDITIONS",
                    "boundary_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_wall_boundary_coverage_ratio",
                "FEA_CFD_BOUNDARY_CONDITIONS",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_BOUNDARY_CONDITIONS",
                    "wall_boundary_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_pressure_correction_residual_ratio",
                "FEA_CFD_PRESSURE_CORRECTION",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_PRESSURE_CORRECTION",
                    "pressure_correction_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-4),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_velocity_correction_residual_ratio",
                "FEA_CFD_PRESSURE_CORRECTION",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_PRESSURE_CORRECTION",
                    "velocity_correction_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-4),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_pressure_drop_balance_ratio",
                "FEA_CFD_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_KNOWN_ANSWER",
                    "pressure_drop_balance_ratio",
                ),
                Some(0.999_999),
                Some(1.000_001),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_mass_flux_uniformity_ratio",
                "FEA_CFD_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_KNOWN_ANSWER",
                    "mass_flux_uniformity_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_pressure_monotonic_cell_fraction",
                "FEA_CFD_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_KNOWN_ANSWER",
                    "pressure_monotonic_cell_fraction",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cfd_known_answer_coverage_ratio",
                "FEA_CFD_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_KNOWN_ANSWER",
                    "known_answer_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            if spec.id.starts_with("cfd_transient_") {
                push_threshold_assertion(
                    spec.id,
                    &mut threshold_assertions,
                    &mut failures,
                    "cfd_transient_step_count",
                    "FEA_CFD_TRANSIENT_EVOLUTION",
                    diagnostic_metric(
                        &cpu_envelope.data,
                        "FEA_CFD_TRANSIENT_EVOLUTION",
                        "step_count",
                    ),
                    Some(12.0),
                    Some(12.0),
                );
                push_threshold_assertion(
                    spec.id,
                    &mut threshold_assertions,
                    &mut failures,
                    "cfd_transient_scale_min",
                    "FEA_CFD_TRANSIENT_EVOLUTION",
                    diagnostic_metric(
                        &cpu_envelope.data,
                        "FEA_CFD_TRANSIENT_EVOLUTION",
                        "transient_scale_min",
                    ),
                    Some(0.649_999),
                    Some(0.650_001),
                );
                push_threshold_assertion(
                    spec.id,
                    &mut threshold_assertions,
                    &mut failures,
                    "cfd_transient_scale_max",
                    "FEA_CFD_TRANSIENT_EVOLUTION",
                    diagnostic_metric(
                        &cpu_envelope.data,
                        "FEA_CFD_TRANSIENT_EVOLUTION",
                        "transient_scale_max",
                    ),
                    Some(1.0),
                    Some(1.0),
                );
                push_threshold_assertion(
                    spec.id,
                    &mut threshold_assertions,
                    &mut failures,
                    "cfd_transient_scale_variation",
                    "FEA_CFD_TRANSIENT_EVOLUTION",
                    diagnostic_metric(
                        &cpu_envelope.data,
                        "FEA_CFD_TRANSIENT_EVOLUTION",
                        "transient_scale_variation",
                    ),
                    Some(0.349_999),
                    Some(0.350_001),
                );
            }
        }
        if spec.id.starts_with("cht_coupled_") {
            let expected_cht_profile_point_count = if spec.id == "cht_coupled_channel_slab_cpu" {
                3.0
            } else {
                2.0
            };
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_reference_density_kg_per_m3",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "density"),
                Some(1.20),
                Some(1.25),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_dynamic_viscosity_pa_s",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "viscosity"),
                Some(1.0e-5),
                Some(3.0e-5),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_inlet_velocity_m_per_s",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "inlet_velocity"),
                Some(4.0),
                Some(6.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_turbulence_intensity",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "turbulence_intensity"),
                Some(0.04),
                Some(0.08),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_reynolds_number",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "reynolds_number"),
                Some(2.0e5),
                Some(5.0e5),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_profile_point_count",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "profile_point_count"),
                Some(expected_cht_profile_point_count),
                Some(expected_cht_profile_point_count),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_applied_temperature_delta_k",
                "FEA_CHT_COUPLING",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_COUPLING",
                    "applied_temperature_delta_k",
                ),
                Some(60.0),
                Some(60.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_step_count",
                "FEA_CHT_COUPLING",
                diagnostic_metric(&cpu_envelope.data, "FEA_CHT_COUPLING", "step_count"),
                Some(12.0),
                Some(12.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_time_step_s",
                "FEA_CHT_COUPLING",
                diagnostic_metric(&cpu_envelope.data, "FEA_CHT_COUPLING", "time_step_s"),
                Some(1.0e-3),
                Some(1.0e-3),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_max_momentum_residual",
                "FEA_CFD_RESIDUAL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_RESIDUAL",
                    "max_momentum_residual",
                ),
                None,
                Some(1.0e-4),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_max_continuity_residual",
                "FEA_CFD_RESIDUAL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_RESIDUAL",
                    "max_continuity_residual",
                ),
                None,
                Some(1.0e-4),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_max_thermal_residual",
                "FEA_THERMAL_STABILITY",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_THERMAL_STABILITY",
                    "max_residual_norm",
                ),
                None,
                Some(2.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_interface_face_count",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "interface_face_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_max_temperature_jump_k",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "max_temperature_jump_k",
                ),
                Some(0.0),
                Some(0.1),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_max_energy_residual",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "max_energy_residual",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_heat_flux_balance_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "heat_flux_balance_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_thermal_transport_residual_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "thermal_transport_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_interface_temperature_continuity_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "interface_temperature_continuity_ratio",
                ),
                Some(0.999),
                Some(1.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_advection_temperature_shift_k",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "max_advection_temperature_shift_k",
                ),
                Some(0.0),
                Some(0.1),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_interface_conductance_w_per_m2k",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "interface_conductance_w_per_m2k",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_flux_temperature_law_residual_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "flux_temperature_law_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_heat_flux_realization_residual_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "heat_flux_realization_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_coupled_interface_iteration_count",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "coupled_interface_iteration_count",
                ),
                Some(1.0),
                Some(64.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_coupled_interface_residual_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "coupled_interface_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_thermal_network_node_count",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "thermal_network_node_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_thermal_network_edge_count",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "thermal_network_edge_count",
                ),
                Some(0.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_thermal_network_residual_ratio",
                "FEA_CHT_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_INTERFACE_CLOSURE",
                    "thermal_network_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_heated_channel_energy_residual_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "heated_channel_energy_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_conjugate_slab_flux_law_residual_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "conjugate_slab_flux_law_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_known_answer_coupled_interface_residual_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "coupled_interface_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_known_answer_heat_flux_realization_residual_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "heat_flux_realization_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_known_answer_thermal_network_residual_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "thermal_network_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_known_answer_interface_temperature_continuity_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "interface_temperature_continuity_ratio",
                ),
                Some(0.999),
                Some(1.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_advection_shift_coverage_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "advection_shift_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "cht_known_answer_coverage_ratio",
                "FEA_CHT_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CHT_KNOWN_ANSWER",
                    "known_answer_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            if spec.id == "cht_coupled_channel_slab_cpu" {
                push_threshold_assertion(
                    spec.id,
                    &mut threshold_assertions,
                    &mut failures,
                    "cht_authored_interface_count",
                    "FEA_CHT_COUPLING",
                    diagnostic_metric(
                        &cpu_envelope.data,
                        "FEA_CHT_COUPLING",
                        "authored_interface_count",
                    ),
                    Some(1.0),
                    Some(1.0),
                );
            }
        }
        if spec.id.starts_with("fsi_coupled_") {
            let expected_fsi_profile_point_count = if spec.id == "fsi_coupled_pipe_plate_cpu" {
                3.0
            } else {
                2.0
            };
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_max_momentum_residual",
                "FEA_CFD_RESIDUAL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_RESIDUAL",
                    "max_momentum_residual",
                ),
                None,
                Some(1.0e-4),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_max_continuity_residual",
                "FEA_CFD_RESIDUAL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_CFD_RESIDUAL",
                    "max_continuity_residual",
                ),
                None,
                Some(1.0e-4),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_max_interface_residual",
                "FEA_FSI_INTERFACE_RESIDUAL",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_RESIDUAL",
                    "max_interface_residual",
                ),
                None,
                Some(1.0e-4),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_interface_node_count",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "interface_node_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_force_balance_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "force_balance_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_max_displacement_transfer_residual_m",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "max_displacement_transfer_residual_m",
                ),
                Some(0.0),
                Some(1.0e-12),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_max_coupling_iteration_count",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "max_coupling_iteration_count",
                ),
                Some(1.0),
                Some(128.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_pressure_feedback_residual_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "pressure_feedback_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_two_way_interface_residual_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "two_way_interface_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_structural_traction_update_residual_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "structural_traction_update_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_pressure_displacement_law_residual_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "pressure_displacement_law_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_structural_solve_residual_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "structural_solve_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_interface_work_j_per_m2",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "interface_work_j_per_m2",
                ),
                Some(1.0e-18),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_structural_strain_energy_j_per_m2",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "structural_strain_energy_j_per_m2",
                ),
                Some(1.0e-18),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_interface_work_energy_residual_ratio",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "interface_work_energy_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_structural_coupling_edge_count",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "structural_coupling_edge_count",
                ),
                Some(1.0),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_interface_stiffness_pa_per_m",
                "FEA_FSI_INTERFACE_CLOSURE",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_INTERFACE_CLOSURE",
                    "interface_stiffness_pa_per_m",
                ),
                Some(1.0e6),
                None,
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_pressure_loaded_wall_displacement_law_residual_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "pressure_loaded_wall_displacement_law_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_interface_traction_balance_residual_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "interface_traction_balance_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-9),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_known_answer_displacement_transfer_residual_m",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "interface_displacement_transfer_residual_m",
                ),
                Some(0.0),
                Some(1.0e-12),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_partitioned_pressure_feedback_residual_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "partitioned_pressure_feedback_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_known_answer_two_way_interface_residual_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "two_way_interface_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_known_answer_structural_traction_update_residual_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "structural_traction_update_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_known_answer_structural_solve_residual_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "structural_solve_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_known_answer_interface_work_energy_residual_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "interface_work_energy_residual_ratio",
                ),
                Some(0.0),
                Some(1.0e-8),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_known_answer_coverage_ratio",
                "FEA_FSI_KNOWN_ANSWER",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_KNOWN_ANSWER",
                    "known_answer_coverage_ratio",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_reference_density_kg_per_m3",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "density"),
                Some(1.20),
                Some(1.25),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_dynamic_viscosity_pa_s",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "viscosity"),
                Some(1.0e-5),
                Some(3.0e-5),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_inlet_velocity_m_per_s",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "inlet_velocity"),
                Some(3.5),
                Some(4.5),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_turbulence_intensity",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "turbulence_intensity"),
                Some(0.04),
                Some(0.08),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_reynolds_number",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "reynolds_number"),
                Some(2.0e5),
                Some(3.0e5),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_profile_point_count",
                "FEA_CFD_FLOW",
                diagnostic_metric(&cpu_envelope.data, "FEA_CFD_FLOW", "profile_point_count"),
                Some(expected_fsi_profile_point_count),
                Some(expected_fsi_profile_point_count),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_step_count",
                "FEA_FSI_COUPLING",
                diagnostic_metric(&cpu_envelope.data, "FEA_FSI_COUPLING", "step_count"),
                Some(12.0),
                Some(12.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_time_step_s",
                "FEA_FSI_COUPLING",
                diagnostic_metric(&cpu_envelope.data, "FEA_FSI_COUPLING", "time_step_s"),
                Some(1.0e-3),
                Some(1.0e-3),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_structural_step_count",
                "FEA_FSI_COUPLING",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_COUPLING",
                    "structural_step_count",
                ),
                Some(1.0),
                Some(1.0),
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "fsi_cfd_profile_point_count",
                "FEA_FSI_COUPLING",
                diagnostic_metric(
                    &cpu_envelope.data,
                    "FEA_FSI_COUPLING",
                    "cfd_profile_point_count",
                ),
                Some(expected_fsi_profile_point_count),
                Some(expected_fsi_profile_point_count),
            );
            if spec.id == "fsi_coupled_pipe_plate_cpu" {
                push_threshold_assertion(
                    spec.id,
                    &mut threshold_assertions,
                    &mut failures,
                    "fsi_authored_interface_count",
                    "FEA_FSI_COUPLING",
                    diagnostic_metric(
                        &cpu_envelope.data,
                        "FEA_FSI_COUPLING",
                        "authored_interface_count",
                    ),
                    Some(1.0),
                    Some(1.0),
                );
            }
        }
        if spec.id == "thermal_standalone_ramp_cpu" {
            push_thermal_standalone_threshold_assertions(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                &cpu_envelope.data,
            );
        }

        if let Some(gpu_mode) = spec.gpu_mode {
            let gpu_start = Instant::now();
            let gpu_result = run_fixture_gpu(spec, &model, gpu_mode);
            gpu_run_ms = Some(gpu_start.elapsed().as_secs_f64() * 1_000.0);

            match gpu_result {
                Ok(gpu_envelope) => {
                    run_ok = true;
                    publishable = Some(gpu_envelope.data.publishable);
                    gpu_fallback_events = gpu_envelope.data.provenance.fallback_events.clone();
                    gpu_solver_host_sync_count =
                        Some(gpu_envelope.data.provenance.solver_host_sync_count);
                    gpu_solver_device_apply_k_ratio =
                        Some(gpu_envelope.data.provenance.solver_device_apply_k_ratio);
                    gpu_solver_backend = Some(gpu_envelope.data.provenance.solver_backend.clone());
                    gpu_speedup_ratio = match (cpu_run_ms, gpu_run_ms) {
                        (Some(cpu_ms), Some(gpu_ms)) if gpu_ms > 0.0 => Some(cpu_ms / gpu_ms),
                        _ => None,
                    };
                    let cache_hits = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_CACHE",
                        "prepared_cache_hits",
                    );
                    let cache_misses = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_CACHE",
                        "prepared_cache_misses",
                    );
                    let cache_entries = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_CACHE",
                        "prepared_cache_entries",
                    );
                    gpu_transient_cache_misses = cache_misses;
                    gpu_transient_cache_entries = cache_entries;
                    gpu_transient_cache_hit_ratio = match (cache_hits, cache_misses) {
                        (Some(hits), Some(misses)) if (hits + misses) > 0.0 => {
                            Some(hits / (hits + misses))
                        }
                        _ => None,
                    };
                    gpu_solver_prepared_build_ms = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_MODAL_COST",
                        "prepared_build_ms",
                    )
                    .or_else(|| {
                        diagnostic_metric(&gpu_envelope.data, "FEA_EM_COST", "prepared_build_ms")
                    })
                    .or_else(|| {
                        diagnostic_metric(
                            &gpu_envelope.data,
                            "FEA_TRANSIENT_COST",
                            "prepared_build_ms",
                        )
                    })
                    .or_else(|| {
                        diagnostic_metric(
                            &gpu_envelope.data,
                            "FEA_NONLINEAR_COST",
                            "prepared_build_ms",
                        )
                    });
                    gpu_solver_solve_ms =
                        diagnostic_metric(&gpu_envelope.data, "FEA_MODAL_COST", "solve_ms")
                            .or_else(|| {
                                diagnostic_metric(&gpu_envelope.data, "FEA_EM_COST", "solve_ms")
                            })
                            .or_else(|| {
                                diagnostic_metric(
                                    &gpu_envelope.data,
                                    "FEA_TRANSIENT_COST",
                                    "solve_ms",
                                )
                            })
                            .or_else(|| {
                                diagnostic_metric(
                                    &gpu_envelope.data,
                                    "FEA_NONLINEAR_COST",
                                    "solve_ms",
                                )
                            })
                            .or(gpu_run_ms);
                    gpu_solver_fallback_apply_count = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_MODAL_COST",
                        "fallback_apply_count",
                    )
                    .or_else(|| {
                        diagnostic_metric(&gpu_envelope.data, "FEA_EM_COST", "fallback_apply_count")
                    })
                    .or_else(|| {
                        diagnostic_metric(
                            &gpu_envelope.data,
                            "FEA_TRANSIENT_COST",
                            "fallback_apply_count",
                        )
                    })
                    .or_else(|| {
                        diagnostic_metric(
                            &gpu_envelope.data,
                            "FEA_NONLINEAR_COST",
                            "fallback_apply_count",
                        )
                    });
                    let transient_adapt_scale_min = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_ADAPTIVITY",
                        "scale_min",
                    );
                    let transient_adapt_scale_max = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_ADAPTIVITY",
                        "scale_max",
                    );
                    let transient_adapt_scale_mean = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_ADAPTIVITY",
                        "scale_mean",
                    );
                    let transient_adapt_decrease_steps = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_ADAPTIVITY",
                        "decrease_steps",
                    );
                    let transient_bucket_rel_tolerance = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_BUCKETING",
                        "rel_tolerance",
                    );
                    let transient_physics_jump_ratio = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_PHYSICS",
                        "max_step_l2_jump_ratio",
                    );
                    let transient_physics_nonfinite = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_PHYSICS",
                        "nonfinite_displacement_count",
                    );
                    let nonlinear_converged_increments = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "converged_increments",
                    );
                    let nonlinear_total_increments = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "increments",
                    );
                    let nonlinear_failed_increments = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "failed_increments",
                    );
                    let nonlinear_max_residual_norm = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "max_residual_norm",
                    );
                    let nonlinear_max_increment_norm = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "max_increment_norm",
                    );
                    let nonlinear_line_search_backtracks = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "line_search_backtracks",
                    );
                    let nonlinear_max_backtracks_per_increment = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "max_line_search_backtracks_per_increment",
                    );
                    let nonlinear_tangent_rebuild_count = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "tangent_rebuild_count",
                    );
                    let nonlinear_iteration_spike_count = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "iteration_spike_count",
                    );
                    let nonlinear_convergence_stall_count = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "convergence_stall_count",
                    );
                    let nonlinear_backtrack_burst_count = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "backtrack_burst_count",
                    );

                    for event in &gpu_fallback_events {
                        if !validate_fallback_event_schema(event) {
                            failures.push(format!("invalid fallback event schema: {event}"));
                        }
                    }

                    let gpu_primary_field_id = primary_result_field_id(spec.run_kind);
                    if let Some(primary_field) =
                        analysis_result_field(&gpu_envelope.data, &gpu_primary_field_id)
                    {
                        gpu_displacement_residency = Some(match &primary_field.values {
                            AnalysisFieldValues::DeviceRef(_) => "device_ref".to_string(),
                            AnalysisFieldValues::HostF64(_) => "host_f64".to_string(),
                        });
                    } else {
                        failures.push(format!(
                            "primary field {gpu_primary_field_id} should be present for gpu fixture {}",
                            spec.id
                        ));
                    }

                    let gpu_expected_publishable = if spec.id.starts_with("acoustic_harmonic_") {
                        Some(false)
                    } else {
                        spec.expected_publishable
                    };
                    if let Some(expected_publishable) = gpu_expected_publishable {
                        if gpu_envelope.data.publishable != expected_publishable {
                            failures.push(format!(
                                "gpu publishable mismatch: expected {expected_publishable}, got {}",
                                gpu_envelope.data.publishable
                            ));
                        }
                    }

                    if let Some(expected_solver_backend) = spec.expected_solver_backend {
                        if gpu_envelope.data.provenance.solver_backend != expected_solver_backend {
                            failures.push(format!(
                                "solver_backend mismatch for fixture {}: expected={} got={}",
                                spec.id,
                                expected_solver_backend,
                                gpu_envelope.data.provenance.solver_backend
                            ));
                        }
                    }
                    if env_bool("RUNMAT_FEA_ENFORCE_SPEEDUP_GATES").unwrap_or(false) {
                        if let Some(min_speedup_ratio) = spec.min_gpu_speedup_ratio {
                            let observed = gpu_speedup_ratio.unwrap_or(0.0);
                            if observed < min_speedup_ratio {
                                failures.push(format!(
                                    "gpu speedup ratio below target for fixture {}: observed={} min={}",
                                    spec.id, observed, min_speedup_ratio
                                ));
                            }
                        }
                    }
                    if let Some(min_cache_hit_ratio) = spec.min_transient_cache_hit_ratio {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_prepared_cache_hit_ratio",
                            "FEA_TRANSIENT_CACHE",
                            gpu_transient_cache_hit_ratio,
                            Some(min_cache_hit_ratio),
                            None,
                        );
                    }
                    if let Some(max_cache_misses) = spec.max_transient_cache_misses {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_prepared_cache_misses",
                            "FEA_TRANSIENT_CACHE",
                            gpu_transient_cache_misses,
                            None,
                            Some(max_cache_misses),
                        );
                    }
                    if spec.id == "transient_long_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_adapt_scale_min",
                            "FEA_TRANSIENT_ADAPTIVITY",
                            transient_adapt_scale_min,
                            Some(0.65),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_adapt_scale_max",
                            "FEA_TRANSIENT_ADAPTIVITY",
                            transient_adapt_scale_max,
                            None,
                            Some(1.35),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_adapt_scale_mean",
                            "FEA_TRANSIENT_ADAPTIVITY",
                            transient_adapt_scale_mean,
                            Some(0.8),
                            Some(1.2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_adapt_decrease_steps",
                            "FEA_TRANSIENT_ADAPTIVITY",
                            transient_adapt_decrease_steps,
                            None,
                            Some(16.0),
                        );
                        let requested_bucket_tol =
                            std::env::var("RUNMAT_TRANSIENT_DT_BUCKET_REL_TOL")
                                .ok()
                                .and_then(|value| value.parse::<f64>().ok())
                                .unwrap_or(0.0)
                                .max(0.0);
                        if requested_bucket_tol > 0.0 {
                            push_threshold_assertion(
                                spec.id,
                                &mut threshold_assertions,
                                &mut failures,
                                "transient_bucket_rel_tolerance",
                                "FEA_TRANSIENT_BUCKETING",
                                transient_bucket_rel_tolerance,
                                Some(requested_bucket_tol * 0.99),
                                Some(requested_bucket_tol * 1.01 + 1.0e-12),
                            );
                            push_threshold_assertion(
                                spec.id,
                                &mut threshold_assertions,
                                &mut failures,
                                "transient_bucket_quality_residual",
                                "FEA_TRANSIENT_STABILITY",
                                diagnostic_metric(
                                    &gpu_envelope.data,
                                    "FEA_TRANSIENT_STABILITY",
                                    "max_residual_norm",
                                ),
                                None,
                                Some(1.0e-2),
                            );
                        }
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_physics_jump_ratio",
                            "FEA_TRANSIENT_PHYSICS",
                            transient_physics_jump_ratio,
                            None,
                            Some(3.5),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_physics_nonfinite_count",
                            "FEA_TRANSIENT_PHYSICS",
                            transient_physics_nonfinite,
                            None,
                            Some(0.0),
                        );
                    }
                    if spec.id == "transient_shock_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_shock_physics_jump_ratio",
                            "FEA_TRANSIENT_PHYSICS",
                            transient_physics_jump_ratio,
                            None,
                            Some(4.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_shock_physics_nonfinite_count",
                            "FEA_TRANSIENT_PHYSICS",
                            transient_physics_nonfinite,
                            None,
                            Some(0.0),
                        );
                    }
                    if spec.id == "thermo_mech_kickoff_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_thermal_strain_scale",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "thermal_strain_scale",
                            ),
                            Some(5.0e-4),
                            Some(5.0e-2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_thermal_load_scale",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "thermal_load_scale",
                            ),
                            Some(0.5),
                            Some(2.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_effective_modulus_scale",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "effective_modulus_scale",
                            ),
                            Some(0.85),
                            Some(1.2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_material_spread_ratio",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "constitutive_material_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.3),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_assignment_heterogeneity_index",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "assignment_heterogeneity_index",
                            ),
                            Some(0.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_transient_severity",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "severity_peak",
                            ),
                            Some(0.0),
                            Some(0.2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_transient_time_scale_mean",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "time_scale_mean",
                            ),
                            Some(0.6),
                            Some(1.1),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_constitutive_residual_ratio",
                            "FEA_TM_CONSISTENCY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_CONSISTENCY",
                                "constitutive_residual_ratio",
                            ),
                            Some(0.0),
                            Some(1.0e-8),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_thermal_strain_energy_density_mean",
                            "FEA_TM_CONSISTENCY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_CONSISTENCY",
                                "thermal_strain_energy_density_mean",
                            ),
                            Some(0.0),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_consistency_coverage_ratio",
                            "FEA_TM_CONSISTENCY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_CONSISTENCY",
                                "consistency_coverage_ratio",
                            ),
                            Some(0.9),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_temperature_field_node_count",
                            "FEA_TM_TEMPERATURE_FIELD",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TEMPERATURE_FIELD",
                                "node_count",
                            ),
                            Some(1.0),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_strain_temperature_residual_ratio",
                            "FEA_TM_TEMPERATURE_FIELD",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TEMPERATURE_FIELD",
                                "strain_temperature_residual_ratio",
                            ),
                            Some(0.0),
                            Some(1.0e-8),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_strain_temperature_coverage_ratio",
                            "FEA_TM_TEMPERATURE_FIELD",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TEMPERATURE_FIELD",
                                "strain_temperature_coverage_ratio",
                            ),
                            Some(0.9),
                            Some(1.0),
                        );
                    } else if spec.id == "thermo_gradient_benign_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_gradient_benign_spread_ratio",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "constitutive_material_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.18),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_gradient_benign_heterogeneity",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "assignment_heterogeneity_index",
                            ),
                            Some(0.0),
                            Some(0.22),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_gradient_benign_temporal_variation",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "temporal_variation",
                            ),
                            Some(0.0),
                            Some(0.35),
                        );
                    } else if spec.id == "thermo_gradient_pathological_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_gradient_pathological_spread_ratio",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "constitutive_material_spread_ratio",
                            ),
                            Some(1.04),
                            Some(1.55),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_gradient_pathological_heterogeneity",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "assignment_heterogeneity_index",
                            ),
                            Some(0.2),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_gradient_pathological_temporal_variation",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "temporal_variation",
                            ),
                            Some(0.2),
                            Some(1.0),
                        );
                    } else if spec.id == "thermo_ramp_smooth_gpu_provider"
                        || spec.id == "thermo_ramp_smooth_field_artifact_gpu_provider"
                    {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_ramp_smooth_constitutive_temperature_factor",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "constitutive_temperature_factor",
                            ),
                            Some(-0.1),
                            Some(0.1),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_ramp_smooth_effective_modulus_scale",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "effective_modulus_scale",
                            ),
                            Some(0.85),
                            Some(1.1),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_ramp_smooth_temporal_variation",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "temporal_variation",
                            ),
                            Some(0.2),
                            Some(0.4),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_ramp_smooth_spatial_gradient_index",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "spatial_gradient_index",
                            ),
                            Some(0.0),
                            Some(0.25),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_ramp_smooth_spatial_coverage_ratio",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "spatial_coverage_ratio",
                            ),
                            Some(0.35),
                            Some(0.7),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_ramp_smooth_field_extrapolation_ratio",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "field_extrapolation_ratio",
                            ),
                            Some(0.0),
                            Some(0.02),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_ramp_smooth_field_clamp_ratio",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "field_clamp_ratio",
                            ),
                            Some(0.0),
                            Some(0.02),
                        );
                    } else if spec.id == "thermo_shock_oscillatory_gpu_provider"
                        || spec.id == "thermo_shock_oscillatory_field_artifact_gpu_provider"
                    {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_shock_constitutive_temperature_factor",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "constitutive_temperature_factor",
                            ),
                            Some(-0.2),
                            Some(0.1),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_shock_effective_modulus_scale",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "effective_modulus_scale",
                            ),
                            Some(0.8),
                            Some(1.1),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_shock_oscillatory_temporal_variation",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "temporal_variation",
                            ),
                            Some(0.35),
                            Some(0.7),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_shock_oscillatory_spatial_gradient_index",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "spatial_gradient_index",
                            ),
                            Some(0.25),
                            Some(0.8),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_shock_oscillatory_spatial_coverage_ratio",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "spatial_coverage_ratio",
                            ),
                            Some(0.30),
                            Some(0.7),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_shock_oscillatory_field_extrapolation_ratio",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "field_extrapolation_ratio",
                            ),
                            Some(0.0),
                            Some(0.08),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_shock_oscillatory_field_clamp_ratio",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "field_clamp_ratio",
                            ),
                            Some(0.0),
                            Some(0.08),
                        );
                    } else if spec.id.starts_with("thermal_standalone_ramp_") {
                        push_thermal_standalone_threshold_assertions(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                        );
                    } else if spec.id == "electro_thermal_joule_benign_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_joule_heating_scale",
                            "FEA_ET_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_COUPLING",
                                "joule_heating_scale",
                            ),
                            Some(9.8),
                            Some(10.2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_conductivity_spread_ratio",
                            "FEA_ET_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_COUPLING",
                                "conductivity_spread_ratio",
                            ),
                            Some(1.03),
                            Some(1.12),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_transient_severity_peak",
                            "FEA_ET_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_TRANSIENT",
                                "severity_peak",
                            ),
                            Some(0.98),
                            Some(1.02),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_temporal_variation",
                            "FEA_ET_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_TRANSIENT",
                                "temporal_variation",
                            ),
                            Some(0.08),
                            Some(0.25),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_time_scale_mean",
                            "FEA_ET_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_TRANSIENT",
                                "time_scale_mean",
                            ),
                            Some(0.82),
                            Some(0.95),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_conductive_node_count",
                            "FEA_ET_DOMAIN_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_DOMAIN_TOPOLOGY",
                                "conductive_node_count",
                            ),
                            Some(2.0),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_mapped_voltage_boundary_count",
                            "FEA_ET_DOMAIN_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_DOMAIN_TOPOLOGY",
                                "mapped_voltage_boundary_count",
                            ),
                            Some(2.0),
                            Some(2.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_topology_component_count",
                            "FEA_ET_DOMAIN_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_DOMAIN_TOPOLOGY",
                                "topology_component_count",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_mapped_current_source_count",
                            "FEA_ET_DOMAIN_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_DOMAIN_TOPOLOGY",
                                "mapped_current_source_count",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_source_boundary_alignment_ratio",
                            "FEA_ET_DOMAIN_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_DOMAIN_TOPOLOGY",
                                "source_boundary_alignment_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_domain_conductance_coverage_ratio",
                            "FEA_ET_DOMAIN_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_DOMAIN_TOPOLOGY",
                                "domain_conductance_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_material_region_coverage_ratio",
                            "FEA_ET_DOMAIN_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_DOMAIN_TOPOLOGY",
                                "material_region_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_potential_residual_norm",
                            "FEA_ET_POTENTIAL_SOLVE",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_POTENTIAL_SOLVE",
                                "residual_norm",
                            ),
                            Some(0.0),
                            Some(1.0e-8),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_current_balance_residual",
                            "FEA_ET_POTENTIAL_SOLVE",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_POTENTIAL_SOLVE",
                                "current_balance_residual",
                            ),
                            Some(0.0),
                            Some(1.0e-8),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_potential_span_v",
                            "FEA_ET_POTENTIAL_SOLVE",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_POTENTIAL_SOLVE",
                                "potential_span_v",
                            ),
                            Some(35.9),
                            Some(36.1),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_conduction_edge_count",
                            "FEA_ET_POTENTIAL_SOLVE",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_POTENTIAL_SOLVE",
                                "edge_count",
                            ),
                            Some(1.0),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_topology_coverage_ratio",
                            "FEA_ET_POTENTIAL_SOLVE",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_POTENTIAL_SOLVE",
                                "topology_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_ohms_law_residual_ratio",
                            "FEA_ET_CONDUCTION_CONSERVATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_CONDUCTION_CONSERVATION",
                                "ohms_law_residual_ratio",
                            ),
                            Some(0.0),
                            Some(1.0e-8),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_joule_heat_balance_ratio",
                            "FEA_ET_CONDUCTION_CONSERVATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_CONDUCTION_CONSERVATION",
                                "joule_heat_balance_ratio",
                            ),
                            Some(0.999_999),
                            Some(1.000_001),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_potential_monotonic_edge_fraction",
                            "FEA_ET_CONDUCTION_CONSERVATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_CONDUCTION_CONSERVATION",
                                "potential_monotonic_edge_fraction",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_benign_conduction_graph_coverage_ratio",
                            "FEA_ET_CONDUCTION_CONSERVATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_CONDUCTION_CONSERVATION",
                                "conduction_graph_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_electro_thermal_source_coupling_threshold_assertions(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                            "electro_thermal_benign",
                        );
                    } else if spec.id == "electro_thermal_joule_pathological_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_joule_heating_scale",
                            "FEA_ET_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_COUPLING",
                                "joule_heating_scale",
                            ),
                            Some(9.8),
                            Some(10.2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_conductivity_spread_ratio",
                            "FEA_ET_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_COUPLING",
                                "conductivity_spread_ratio",
                            ),
                            Some(2.8),
                            Some(3.6),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_transient_severity_peak",
                            "FEA_ET_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_TRANSIENT",
                                "severity_peak",
                            ),
                            Some(0.9),
                            Some(1.2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_temporal_variation",
                            "FEA_ET_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_TRANSIENT",
                                "temporal_variation",
                            ),
                            Some(0.35),
                            Some(0.8),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_time_scale_mean",
                            "FEA_ET_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_TRANSIENT",
                                "time_scale_mean",
                            ),
                            Some(0.85),
                            Some(1.1),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_conductive_node_count",
                            "FEA_ET_DOMAIN_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_DOMAIN_TOPOLOGY",
                                "conductive_node_count",
                            ),
                            Some(2.0),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_mapped_voltage_boundary_count",
                            "FEA_ET_DOMAIN_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_DOMAIN_TOPOLOGY",
                                "mapped_voltage_boundary_count",
                            ),
                            Some(2.0),
                            Some(2.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_topology_component_count",
                            "FEA_ET_DOMAIN_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_DOMAIN_TOPOLOGY",
                                "topology_component_count",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_mapped_current_source_count",
                            "FEA_ET_DOMAIN_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_DOMAIN_TOPOLOGY",
                                "mapped_current_source_count",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_source_boundary_alignment_ratio",
                            "FEA_ET_DOMAIN_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_DOMAIN_TOPOLOGY",
                                "source_boundary_alignment_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_domain_conductance_coverage_ratio",
                            "FEA_ET_DOMAIN_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_DOMAIN_TOPOLOGY",
                                "domain_conductance_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_material_region_coverage_ratio",
                            "FEA_ET_DOMAIN_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_DOMAIN_TOPOLOGY",
                                "material_region_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_potential_residual_norm",
                            "FEA_ET_POTENTIAL_SOLVE",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_POTENTIAL_SOLVE",
                                "residual_norm",
                            ),
                            Some(0.0),
                            Some(1.0e-8),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_current_balance_residual",
                            "FEA_ET_POTENTIAL_SOLVE",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_POTENTIAL_SOLVE",
                                "current_balance_residual",
                            ),
                            Some(0.0),
                            Some(1.0e-8),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_potential_span_v",
                            "FEA_ET_POTENTIAL_SOLVE",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_POTENTIAL_SOLVE",
                                "potential_span_v",
                            ),
                            Some(179.9),
                            Some(180.1),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_conduction_edge_count",
                            "FEA_ET_POTENTIAL_SOLVE",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_POTENTIAL_SOLVE",
                                "edge_count",
                            ),
                            Some(1.0),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_topology_coverage_ratio",
                            "FEA_ET_POTENTIAL_SOLVE",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_POTENTIAL_SOLVE",
                                "topology_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_ohms_law_residual_ratio",
                            "FEA_ET_CONDUCTION_CONSERVATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_CONDUCTION_CONSERVATION",
                                "ohms_law_residual_ratio",
                            ),
                            Some(0.0),
                            Some(1.0e-8),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_joule_heat_balance_ratio",
                            "FEA_ET_CONDUCTION_CONSERVATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_CONDUCTION_CONSERVATION",
                                "joule_heat_balance_ratio",
                            ),
                            Some(0.999_999),
                            Some(1.000_001),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_potential_monotonic_edge_fraction",
                            "FEA_ET_CONDUCTION_CONSERVATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_CONDUCTION_CONSERVATION",
                                "potential_monotonic_edge_fraction",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_thermal_pathological_conduction_graph_coverage_ratio",
                            "FEA_ET_CONDUCTION_CONSERVATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_CONDUCTION_CONSERVATION",
                                "conduction_graph_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_electro_thermal_source_coupling_threshold_assertions(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                            "electro_thermal_pathological",
                        );
                    }
                    if spec.id == "nonlinear_assembly_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_converged_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_converged_increments,
                            Some(24.0),
                            Some(24.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_line_search_backtracks",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_line_search_backtracks,
                            Some(8.0),
                            Some(12.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_total_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_total_increments,
                            Some(24.0),
                            Some(24.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_failed_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_failed_increments,
                            None,
                            Some(0.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_max_increment_norm",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_max_increment_norm,
                            None,
                            Some(1.0e-4),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_iteration_spike_count",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_iteration_spike_count,
                            None,
                            Some(2.0),
                        );
                    }
                    if spec.id == "nonlinear_assembly_stress_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_converged_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_converged_increments,
                            Some(32.0),
                            Some(32.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_total_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_total_increments,
                            Some(32.0),
                            Some(32.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_failed_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_failed_increments,
                            None,
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_max_residual_norm",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_max_residual_norm,
                            None,
                            Some(1.0e-5),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_max_increment_norm",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_max_increment_norm,
                            None,
                            Some(1.0e-4),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_line_search_backtracks",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_line_search_backtracks,
                            Some(8.0),
                            Some(14.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_tangent_rebuild_count",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_tangent_rebuild_count,
                            Some(5.0),
                            Some(10.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_iteration_spike_count",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_iteration_spike_count,
                            None,
                            Some(3.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_stall_count",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_convergence_stall_count,
                            None,
                            Some(2.0),
                        );
                    }
                    if spec.id == "nonlinear_softening_benchmark_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_softening_total_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_total_increments,
                            Some(40.0),
                            Some(40.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_softening_failed_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_failed_increments,
                            None,
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_softening_stall_count",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_convergence_stall_count,
                            Some(0.0),
                            Some(2.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_softening_spike_count",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_iteration_spike_count,
                            Some(1.0),
                            Some(4.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_softening_backtrack_bursts",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_backtrack_burst_count,
                            Some(1.0),
                            Some(4.0),
                        );
                    }
                    if spec.id == "nonlinear_load_path_mix_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_path_mix_total_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_total_increments,
                            Some(36.0),
                            Some(36.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_path_mix_max_backtracks_per_increment",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_max_backtracks_per_increment,
                            Some(8.0),
                            Some(11.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_path_mix_backtrack_bursts",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_backtrack_burst_count,
                            Some(1.0),
                            Some(3.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_path_mix_spike_count",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_iteration_spike_count,
                            Some(0.0),
                            Some(2.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_path_mix_effective_modulus_scale",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "effective_modulus_scale",
                            ),
                            Some(0.93),
                            Some(1.02),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_path_mix_material_spread_ratio",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "constitutive_material_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.08),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_nonlinear_severity",
                            "FEA_TM_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_NONLINEAR",
                                "severity_peak",
                            ),
                            Some(0.0),
                            Some(0.08),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_nonlinear_time_scale_mean",
                            "FEA_TM_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_NONLINEAR",
                                "time_scale_mean",
                            ),
                            Some(0.9),
                            Some(1.1),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_nonlinear_field_clamp_ratio",
                            "FEA_TM_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_NONLINEAR",
                                "field_clamp_ratio",
                            ),
                            Some(0.0),
                            Some(0.05),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_nonlinear_joule_heating_scale",
                            "FEA_ET_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_COUPLING",
                                "joule_heating_scale",
                            ),
                            Some(9.8),
                            Some(10.2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_nonlinear_conductivity_spread_ratio",
                            "FEA_ET_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_COUPLING",
                                "conductivity_spread_ratio",
                            ),
                            Some(1.3),
                            Some(1.55),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_nonlinear_severity_peak",
                            "FEA_ET_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_NONLINEAR",
                                "severity_peak",
                            ),
                            Some(0.98),
                            Some(1.02),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_nonlinear_temporal_variation",
                            "FEA_ET_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_NONLINEAR",
                                "temporal_variation",
                            ),
                            Some(0.2),
                            Some(0.45),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_nonlinear_time_scale_mean",
                            "FEA_ET_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_NONLINEAR",
                                "time_scale_mean",
                            ),
                            Some(0.9),
                            Some(1.05),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electro_nonlinear_severity_mean",
                            "FEA_ET_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_ET_NONLINEAR",
                                "severity_mean",
                            ),
                            Some(0.9),
                            Some(0.98),
                        );
                    }
                    if spec.id == "nonlinear_plasticity_benchmark_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "plasticity_nonlinear_severity_peak",
                            "FEA_PLASTIC_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_PLASTIC_NONLINEAR",
                                "severity_peak",
                            ),
                            Some(0.82),
                            Some(0.9),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "plasticity_nonlinear_severity_mean",
                            "FEA_PLASTIC_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_PLASTIC_NONLINEAR",
                                "severity_mean",
                            ),
                            Some(0.65),
                            Some(0.8),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "plasticity_nonlinear_load_realization_ratio",
                            "FEA_PLASTIC_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_PLASTIC_NONLINEAR",
                                "load_realization_ratio",
                            ),
                            Some(0.825),
                            Some(0.835),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "plasticity_nonlinear_load_amplification_ratio",
                            "FEA_PLASTIC_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_PLASTIC_NONLINEAR",
                                "load_amplification_ratio",
                            ),
                            Some(1.51),
                            Some(1.525),
                        );
                        push_plastic_state_threshold_assertions(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                            "plasticity_nonlinear_state",
                        );
                    }
                    if spec.id == "nonlinear_plastic_hardening_reference_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "plasticity_hardening_reference_severity_peak",
                            "FEA_PLASTIC_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_PLASTIC_NONLINEAR",
                                "severity_peak",
                            ),
                            Some(0.18),
                            Some(0.28),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "plasticity_hardening_reference_severity_mean",
                            "FEA_PLASTIC_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_PLASTIC_NONLINEAR",
                                "severity_mean",
                            ),
                            Some(0.15),
                            Some(0.25),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "plasticity_hardening_reference_load_realization_ratio",
                            "FEA_PLASTIC_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_PLASTIC_NONLINEAR",
                                "load_realization_ratio",
                            ),
                            Some(0.825),
                            Some(0.835),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "plasticity_hardening_reference_load_amplification_ratio",
                            "FEA_PLASTIC_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_PLASTIC_NONLINEAR",
                                "load_amplification_ratio",
                            ),
                            Some(1.51),
                            Some(1.525),
                        );
                        push_plastic_state_threshold_assertions(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                            "plasticity_hardening_reference_state",
                        );
                        push_plastic_known_answer_threshold_assertions(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                            "plasticity_hardening_reference_known",
                        );
                    }
                    if spec.id == "nonlinear_plastic_hardening_reference_complex_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "plasticity_hardening_reference_complex_severity_peak",
                            "FEA_PLASTIC_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_PLASTIC_NONLINEAR",
                                "severity_peak",
                            ),
                            Some(0.33),
                            Some(0.43),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "plasticity_hardening_reference_complex_severity_mean",
                            "FEA_PLASTIC_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_PLASTIC_NONLINEAR",
                                "severity_mean",
                            ),
                            Some(0.27),
                            Some(0.36),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "plasticity_hardening_reference_complex_load_realization_ratio",
                            "FEA_PLASTIC_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_PLASTIC_NONLINEAR",
                                "load_realization_ratio",
                            ),
                            Some(0.825),
                            Some(0.835),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "plasticity_hardening_reference_complex_load_amplification_ratio",
                            "FEA_PLASTIC_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_PLASTIC_NONLINEAR",
                                "load_amplification_ratio",
                            ),
                            Some(1.51),
                            Some(1.525),
                        );
                        push_plastic_state_threshold_assertions(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                            "plasticity_hardening_reference_complex_state",
                        );
                        push_plastic_known_answer_threshold_assertions(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                            "plasticity_hardening_reference_complex_known",
                        );
                    }
                    if spec.id == "nonlinear_contact_benchmark_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "contact_nonlinear_severity_peak",
                            "FEA_CONTACT_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_CONTACT_NONLINEAR",
                                "severity_peak",
                            ),
                            Some(0.92),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "contact_nonlinear_severity_mean",
                            "FEA_CONTACT_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_CONTACT_NONLINEAR",
                                "severity_mean",
                            ),
                            Some(0.72),
                            Some(0.9),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "contact_nonlinear_load_realization_ratio",
                            "FEA_CONTACT_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_CONTACT_NONLINEAR",
                                "load_realization_ratio",
                            ),
                            Some(0.845),
                            Some(0.865),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "contact_nonlinear_load_amplification_ratio",
                            "FEA_CONTACT_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_CONTACT_NONLINEAR",
                                "load_amplification_ratio",
                            ),
                            Some(1.4),
                            Some(1.42),
                        );
                        push_contact_state_threshold_assertions(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                            "contact_nonlinear_state",
                        );
                    }
                    if spec.id == "nonlinear_contact_frictionless_reference_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "contact_frictionless_severity_peak",
                            "FEA_CONTACT_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_CONTACT_NONLINEAR",
                                "severity_peak",
                            ),
                            Some(0.23),
                            Some(0.31),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "contact_frictionless_severity_mean",
                            "FEA_CONTACT_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_CONTACT_NONLINEAR",
                                "severity_mean",
                            ),
                            Some(0.18),
                            Some(0.28),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "contact_frictionless_load_realization_ratio",
                            "FEA_CONTACT_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_CONTACT_NONLINEAR",
                                "load_realization_ratio",
                            ),
                            Some(0.845),
                            Some(0.865),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "contact_frictionless_load_amplification_ratio",
                            "FEA_CONTACT_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_CONTACT_NONLINEAR",
                                "load_amplification_ratio",
                            ),
                            Some(1.4),
                            Some(1.42),
                        );
                        push_contact_state_threshold_assertions(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                            "contact_frictionless_state",
                        );
                        push_contact_known_answer_threshold_assertions(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                            "contact_frictionless_known",
                        );
                    }
                    if spec.id == "nonlinear_contact_frictionless_reference_complex_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "contact_frictionless_complex_severity_peak",
                            "FEA_CONTACT_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_CONTACT_NONLINEAR",
                                "severity_peak",
                            ),
                            Some(0.36),
                            Some(0.46),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "contact_frictionless_complex_severity_mean",
                            "FEA_CONTACT_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_CONTACT_NONLINEAR",
                                "severity_mean",
                            ),
                            Some(0.3),
                            Some(0.39),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "contact_frictionless_complex_load_realization_ratio",
                            "FEA_CONTACT_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_CONTACT_NONLINEAR",
                                "load_realization_ratio",
                            ),
                            Some(0.845),
                            Some(0.865),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "contact_frictionless_complex_load_amplification_ratio",
                            "FEA_CONTACT_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_CONTACT_NONLINEAR",
                                "load_amplification_ratio",
                            ),
                            Some(1.4),
                            Some(1.42),
                        );
                        push_contact_state_threshold_assertions(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                            "contact_frictionless_complex_state",
                        );
                        push_contact_known_answer_threshold_assertions(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            &gpu_envelope.data,
                            "contact_frictionless_complex_known",
                        );
                    }
                    if spec.id.starts_with("electromagnetic_reference_") {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_formulation_coverage_ratio",
                            "FEA_EM_FORMULATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_FORMULATION",
                                "formulation_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_magnetostatic_curl_curl_coverage_ratio",
                            "FEA_EM_FORMULATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_FORMULATION",
                                "magnetostatic_curl_curl_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio",
                            "FEA_EM_FORMULATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_FORMULATION",
                                "magnetoquasistatic_eddy_current_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_full_wave_displacement_current_coverage_ratio",
                            "FEA_EM_FORMULATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_FORMULATION",
                                "full_wave_displacement_current_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_material_frequency_response_coverage_ratio",
                            "FEA_EM_FORMULATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_FORMULATION",
                                "material_frequency_response_coverage_ratio",
                            ),
                            Some(0.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_displacement_to_conduction_ratio",
                            "FEA_EM_FORMULATION",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_FORMULATION",
                                "displacement_to_conduction_ratio",
                            ),
                            Some(0.0),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_source_energy_diagnostic_coverage_ratio",
                            "FEA_EM_SOURCE_ENERGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_SOURCE_ENERGY",
                                "field_energy_integral",
                            )
                            .filter(|value| value.is_finite())
                            .map(|_| 1.0),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_source_energy_consistency_ratio",
                            "FEA_EM_SOURCE_ENERGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_SOURCE_ENERGY",
                                "source_region_energy_consistency_ratio",
                            ),
                            Some(0.2),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_source_energy_imbalance_ratio",
                            "FEA_EM_SOURCE_ENERGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_SOURCE_ENERGY",
                                "energy_imbalance_ratio",
                            ),
                            Some(0.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_sweep_known_reference_coverage_ratio",
                            "FEA_EM_SWEEP_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_SWEEP_KNOWN_ANSWER",
                                "reference_frequency_in_sweep_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_sweep_known_peak_frequency_error_ratio",
                            "FEA_EM_SWEEP_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_SWEEP_KNOWN_ANSWER",
                                "normalized_peak_frequency_error_ratio",
                            ),
                            Some(0.0),
                            Some(0.25),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_sweep_known_quality_factor",
                            "FEA_EM_SWEEP_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_SWEEP_KNOWN_ANSWER",
                                "resonance_quality_factor",
                            ),
                            Some(1.5),
                            Some(1.0e9),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "electromagnetic_sweep_known_answer_coverage_ratio",
                            "FEA_EM_SWEEP_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_SWEEP_KNOWN_ANSWER",
                                "sweep_known_answer_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                    }
                    if spec.id == "electromagnetic_reference_homogeneous_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_sigma_omega_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_frequency_scale_mean",
                            ),
                            Some(0.95),
                            Some(1.05),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_sigma_omega_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_frequency_scale_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.15),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_sigma_omega_response_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_frequency_response_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permittivity_frequency_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_frequency_scale_mean",
                            ),
                            Some(0.95),
                            Some(1.05),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permittivity_frequency_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_frequency_scale_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.1),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permittivity_frequency_response_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_frequency_response_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permeability_frequency_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_frequency_scale_mean",
                            ),
                            Some(0.95),
                            Some(1.05),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permeability_frequency_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_frequency_scale_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.1),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permeability_frequency_response_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_frequency_response_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_dispersive_loss_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_loss_scale_mean",
                            ),
                            Some(0.0),
                            Some(0.08),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_dispersive_loss_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_loss_scale_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.1),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_dispersive_phase_attenuation_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_phase_attenuation_mean",
                            ),
                            Some(0.99),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_dispersive_phase_attenuation_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_phase_attenuation_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.05),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_dispersive_coupling_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_conductivity_coupling_ratio",
                            ),
                            Some(0.0),
                            Some(0.05),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_dispersive_phase_conductivity_attenuation_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_phase_conductivity_attenuation_ratio",
                            ),
                            Some(0.99),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_conductivity_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.05),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permittivity_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.1),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_relative_permeability_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.1),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_material_heterogeneity_index",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "electromagnetic_material_heterogeneity_index",
                            ),
                            Some(0.0),
                            Some(0.02),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_assignment_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "assignment_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_assigned_coefficient_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "assigned_coefficient_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_source_realization_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_realization_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_source_region_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_region_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_source_material_alignment_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_material_alignment_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_boundary_anchor_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "boundary_anchor_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_flux_phasor_coherence_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "flux_phasor_coherence_ratio",
                            ),
                            Some(0.5),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_flux_divergence_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "flux_divergence_ratio",
                            ),
                            Some(0.0),
                            Some(0.35),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_energy_imbalance_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "energy_imbalance_ratio",
                            ),
                            Some(0.0),
                            Some(0.40),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_boundary_energy_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "boundary_energy_ratio",
                            ),
                            Some(0.12),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_edge_dof_count",
                            "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                                "edge_dof_count",
                            ),
                            Some(1.0),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_element_count",
                            "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                                "element_count",
                            ),
                            Some(1.0),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_oriented_edge_count",
                            "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_MAXWELL_EDGE_TOPOLOGY",
                                "oriented_edge_count",
                            ),
                            Some(1.0),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_gauge_anchor_count",
                            "FEA_EM_GAUGE",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_GAUGE",
                                "gauge_anchor_count",
                            ),
                            Some(1.0),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_gauge_anchor_residual_ratio",
                            "FEA_EM_GAUGE",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_GAUGE",
                                "gauge_anchor_residual_ratio",
                            ),
                            Some(0.0),
                            Some(1.0e-9),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_known_material_residual_ratio",
                            "FEA_EM_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_KNOWN_ANSWER",
                                "homogeneous_material_residual_ratio",
                            ),
                            Some(0.0),
                            Some(0.05),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_known_source_energy_consistency_residual_ratio",
                            "FEA_EM_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_KNOWN_ANSWER",
                                "source_energy_consistency_residual_ratio",
                            ),
                            Some(0.0),
                            Some(0.05),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_known_gauge_anchor_residual_ratio",
                            "FEA_EM_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_KNOWN_ANSWER",
                                "gauge_anchor_residual_ratio",
                            ),
                            Some(0.0),
                            Some(1.0e-9),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_known_flux_divergence_ratio",
                            "FEA_EM_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_KNOWN_ANSWER",
                                "flux_divergence_ratio",
                            ),
                            Some(0.0),
                            Some(0.35),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_homogeneous_known_answer_coverage_ratio",
                            "FEA_EM_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_KNOWN_ANSWER",
                                "known_answer_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                    }
                    if spec.id == "electromagnetic_reference_heterogeneous_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_sigma_omega_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_frequency_scale_mean",
                            ),
                            Some(0.85),
                            Some(1.35),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_sigma_omega_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_frequency_scale_spread_ratio",
                            ),
                            Some(1.2),
                            Some(2.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_sigma_omega_response_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_frequency_response_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permittivity_frequency_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_frequency_scale_mean",
                            ),
                            Some(0.85),
                            Some(1.35),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permittivity_frequency_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_frequency_scale_spread_ratio",
                            ),
                            Some(1.1),
                            Some(5.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permittivity_frequency_response_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_frequency_response_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permeability_frequency_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_frequency_scale_mean",
                            ),
                            Some(0.85),
                            Some(1.5),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permeability_frequency_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_frequency_scale_spread_ratio",
                            ),
                            Some(1.1),
                            Some(8.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permeability_frequency_response_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_frequency_response_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_dispersive_loss_scale_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_loss_scale_mean",
                            ),
                            Some(0.06),
                            Some(0.22),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_dispersive_loss_scale_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_loss_scale_spread_ratio",
                            ),
                            Some(1.0),
                            Some(20.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_dispersive_phase_attenuation_mean",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_phase_attenuation_mean",
                            ),
                            Some(0.9),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_dispersive_phase_attenuation_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_phase_attenuation_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_dispersive_coupling_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_conductivity_coupling_ratio",
                            ),
                            Some(0.0),
                            Some(0.35),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_dispersive_phase_conductivity_attenuation_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "dispersive_phase_conductivity_attenuation_ratio",
                            ),
                            Some(0.9),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_conductivity_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "conductivity_spread_ratio",
                            ),
                            Some(10.0),
                            Some(1.0e9),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permittivity_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permittivity_spread_ratio",
                            ),
                            Some(2.0),
                            Some(1.0e9),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_relative_permeability_spread_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "relative_permeability_spread_ratio",
                            ),
                            Some(5.0),
                            Some(1.0e12),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_material_heterogeneity_index",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "electromagnetic_material_heterogeneity_index",
                            ),
                            Some(0.25),
                            Some(2.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_region_contrast_index",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "region_coefficient_contrast_index",
                            ),
                            Some(1.0),
                            Some(10.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_assignment_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "assignment_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_source_realization_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_realization_ratio",
                            ),
                            Some(0.6),
                            Some(0.8),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_source_region_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_region_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_source_material_alignment_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_material_alignment_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_boundary_anchor_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "boundary_anchor_ratio",
                            ),
                            Some(0.7),
                            Some(0.85),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_flux_phasor_coherence_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "flux_phasor_coherence_ratio",
                            ),
                            Some(0.2),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_flux_divergence_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "flux_divergence_ratio",
                            ),
                            Some(0.0),
                            Some(0.35),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_heterogeneous_energy_imbalance_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "energy_imbalance_ratio",
                            ),
                            Some(0.1),
                            Some(1.0),
                        );
                    }
                    if spec.id == "electromagnetic_reference_sparse_assignments_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_sparse_assignment_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "assignment_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_sparse_assigned_coefficient_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "assigned_coefficient_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_sparse_source_realization_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_realization_ratio",
                            ),
                            Some(0.2),
                            Some(0.35),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_sparse_source_region_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_region_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_sparse_source_material_alignment_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_material_alignment_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_sparse_boundary_anchor_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "boundary_anchor_ratio",
                            ),
                            Some(0.2),
                            Some(0.35),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_sparse_energy_imbalance_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "energy_imbalance_ratio",
                            ),
                            Some(0.35),
                            Some(1.0),
                        );
                    }
                    if spec.id == "electromagnetic_reference_multiregion_assignments_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_multiregion_assignment_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "assignment_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_multiregion_assigned_coefficient_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "assigned_coefficient_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_multiregion_source_realization_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_realization_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_multiregion_source_region_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_region_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_multiregion_source_material_alignment_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_material_alignment_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_multiregion_boundary_anchor_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "boundary_anchor_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_multiregion_energy_imbalance_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "energy_imbalance_ratio",
                            ),
                            Some(0.35),
                            Some(1.0),
                        );
                    }
                    if spec.id == "electromagnetic_reference_overlap_interference_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_overlap_source_region_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_region_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_overlap_source_material_alignment_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_material_alignment_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_overlap_source_overlap_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_overlap_ratio",
                            ),
                            Some(0.35),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_overlap_source_interference_index",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_interference_index",
                            ),
                            Some(0.2),
                            Some(1.0),
                        );
                    }
                    if spec.id == "electromagnetic_reference_boundary_kernel_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_kernel_known_answer_coverage_ratio",
                            "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                                "boundary_known_answer_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_kernel_boundary_localization_ratio",
                            "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                                "boundary_condition_localization_ratio",
                            ),
                            Some(0.0),
                            Some(0.35),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_kernel_ground_anchor_effectiveness_ratio",
                            "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                                "ground_anchor_effectiveness_ratio",
                            ),
                            Some(0.0),
                            Some(0.6),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_kernel_insulation_leakage_ratio",
                            "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                                "insulation_leakage_ratio",
                            ),
                            Some(0.8),
                            Some(2.0),
                        );
                    }
                    if spec.id == "electromagnetic_reference_boundary_penalty_stress_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_penalty_known_answer_coverage_ratio",
                            "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                                "boundary_known_answer_coverage_ratio",
                            ),
                            Some(1.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_penalty_conditioning_contribution",
                            "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                                "boundary_penalty_conditioning_contribution",
                            ),
                            Some(0.35),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_penalty_anchor_ratio",
                            "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_BOUNDARY_KNOWN_ANSWER",
                                "boundary_anchor_ratio",
                            ),
                            Some(0.75),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_penalty_real_residual_norm",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "real_residual_norm",
                            ),
                            Some(0.0),
                            Some(0.95),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_boundary_penalty_imag_residual_norm",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "imag_residual_norm",
                            ),
                            Some(0.0),
                            Some(0.95),
                        );
                    }
                    if spec.id
                        == "electromagnetic_reference_multi_region_phased_source_gpu_provider"
                    {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_phased_source_region_coverage_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_region_coverage_ratio",
                            ),
                            Some(0.95),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_phased_source_energy_consistency_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_region_energy_consistency_ratio",
                            ),
                            Some(0.2),
                            Some(0.95),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_phased_source_overlap_ratio",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_overlap_ratio",
                            ),
                            Some(0.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "em_phased_source_interference_index",
                            "FEA_EM_STATIC",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_EM_STATIC",
                                "source_interference_index",
                            ),
                            Some(0.0),
                            Some(1.0),
                        );
                    }

                    let gpu_primary_field_id = primary_result_field_id(spec.run_kind);
                    let gpu_results = analysis_results_op(
                        &gpu_envelope.data,
                        AnalysisResultsQuery {
                            include_fields: vec![gpu_primary_field_id.clone()],
                            include_field_values: true,

                            include_diagnostics: false,
                            diagnostic_codes: Vec::new(),
                            include_modal_results: false,
                            mode_indices: Vec::new(),
                            include_transient_results: false,
                            transient_snapshot_indices: Vec::new(),
                            include_nonlinear_results: false,
                            include_electromagnetic_results: false,
                        },
                        OperationContext::new(Some(format!("trace-results-gpu-{}", spec.id)), None),
                    );
                    let gpu_results = match gpu_results {
                        Ok(value) => value,
                        Err(err) => {
                            failures.push(format!(
                                "fea.results failed for gpu fixture {}: {}",
                                spec.id, err.error_code
                            ));
                            return FixtureRunRecord {
                                fixture_id: spec.id.to_string(),
                                validate_ok,
                                validate_error_code,
                                run_ok,
                                run_error_code,
                                cpu_run_ms,
                                gpu_run_ms,
                                gpu_fallback_events,
                                gpu_displacement_residency,
                                gpu_solver_host_sync_count,
                                gpu_solver_device_apply_k_ratio,
                                gpu_speedup_ratio,
                                gpu_solver_backend,
                                gpu_transient_cache_hit_ratio,
                                gpu_transient_cache_misses,
                                gpu_transient_cache_entries,
                                gpu_solver_prepared_build_ms,
                                gpu_solver_solve_ms,
                                gpu_solver_fallback_apply_count,
                                prep_calibration_profile,
                                prep_calibration_fingerprint,
                                prep_acceptance_score,
                                prep_acceptance_passed,
                                prep_acceptance_fingerprint,
                                thermo_coupling_enabled,
                                thermo_coupling_fingerprint,
                                thermo_constitutive_temperature_factor,
                                thermo_effective_modulus_scale,
                                thermo_constitutive_material_spread_ratio,
                                thermo_assignment_heterogeneity_index,
                                thermo_region_delta_count,
                                thermo_spatial_coverage_ratio,
                                thermo_field_extrapolation_ratio,
                                thermo_field_clamp_ratio,
                                thermo_field_artifact_id,
                                thermo_field_artifact_approved,
                                thermo_field_artifact_age_days,
                                thermo_field_artifact_provenance_valid,
                                thermo_transient_severity,
                                thermo_nonlinear_severity,
                                electro_thermal_coupling_enabled,
                                electro_thermal_coupling_fingerprint,
                                electro_joule_heating_scale,
                                electro_conductivity_spread_ratio,
                                electro_transient_severity,
                                electro_transient_time_scale_mean,
                                electro_nonlinear_severity,
                                electro_nonlinear_time_scale_mean,
                                plastic_nonlinear_severity,
                                plastic_nonlinear_severity_mean,
                                plastic_load_realization_ratio,
                                plastic_load_amplification_ratio,
                                contact_nonlinear_severity,
                                contact_nonlinear_severity_mean,
                                contact_load_realization_ratio,
                                contact_load_amplification_ratio,
                                thermal_max_residual_norm,
                                thermal_min_temperature_k,
                                thermal_max_temperature_k,
                                thermal_conductivity_spread_ratio,
                                thermal_heat_capacity_spread_ratio,
                                thermal_spatial_gradient_index,
                                thermal_monotonic_response_fraction,
                                thermal_response_realization_ratio,
                                electromagnetic_enabled,
                                electromagnetic_formulation_coverage_ratio,
                                electromagnetic_magnetostatic_curl_curl_coverage_ratio,
                                electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio,
                                electromagnetic_full_wave_displacement_current_coverage_ratio,
                                electromagnetic_displacement_to_conduction_ratio,
                                electromagnetic_material_frequency_response_coverage_ratio,
                                electromagnetic_reference_frequency_hz,
                                electromagnetic_applied_current_a,
                                electromagnetic_solve_quality,
                                electromagnetic_conductivity_spread_ratio,
                                electromagnetic_relative_permittivity_spread_ratio,
                                electromagnetic_relative_permeability_spread_ratio,
                                electromagnetic_material_heterogeneity_index,
                                electromagnetic_assignment_coverage_ratio,
                                electromagnetic_assigned_coefficient_coverage_ratio,
                                electromagnetic_fallback_coefficient_ratio,
                                electromagnetic_region_coefficient_contrast_index,
                                electromagnetic_condition_number_estimate,
                                electromagnetic_source_realization_ratio,
                                electromagnetic_source_region_coverage_ratio,
                                electromagnetic_source_material_alignment_ratio,
                                electromagnetic_source_localization_ratio,
                                electromagnetic_source_overlap_ratio,
                                electromagnetic_source_interference_index,
                                electromagnetic_boundary_anchor_ratio,
                                electromagnetic_boundary_condition_localization_ratio,
                                electromagnetic_ground_anchor_effectiveness_ratio,
                                electromagnetic_insulation_leakage_ratio,
                                electromagnetic_flux_divergence_ratio,
                                electromagnetic_energy_imbalance_ratio,
                                electromagnetic_boundary_energy_ratio,
                                electromagnetic_boundary_penalty_conditioning_contribution,
                                electromagnetic_source_region_energy_consistency_ratio,
                                electromagnetic_real_residual_norm,
                                electromagnetic_imag_residual_norm,
                                electromagnetic_sweep_count,
                                electromagnetic_resonance_peak_frequency_hz,
                                electromagnetic_resonance_peak_flux_density,
                                electromagnetic_resonance_bandwidth_hz,
                                electromagnetic_resonance_quality_factor,
                                electromagnetic_resonance_flux_gain,
                                publishable,
                                parity,
                                threshold_assertions,
                                failures,
                            };
                        }
                    };
                    if gpu_results.operation != "fea.results"
                        || gpu_results.op_version != "fea.results/v1"
                    {
                        failures.push("fea.results contract version mismatch".to_string());
                    }

                    prep_calibration_profile =
                        gpu_results.data.summary.prep_calibration_profile.clone();
                    prep_calibration_fingerprint =
                        gpu_results.data.summary.prep_calibration_fingerprint;
                    prep_acceptance_score = gpu_results.data.summary.prep_acceptance_score;
                    prep_acceptance_passed = gpu_results.data.summary.prep_acceptance_passed;
                    prep_acceptance_fingerprint =
                        gpu_results.data.summary.prep_acceptance_fingerprint;
                    thermo_coupling_enabled = gpu_results.data.summary.thermo_coupling_enabled;
                    thermo_coupling_fingerprint =
                        gpu_results.data.summary.thermo_coupling_fingerprint;
                    thermo_constitutive_temperature_factor = gpu_results
                        .data
                        .summary
                        .thermo_constitutive_temperature_factor;
                    thermo_effective_modulus_scale =
                        gpu_results.data.summary.thermo_effective_modulus_scale;
                    thermo_constitutive_material_spread_ratio = gpu_results
                        .data
                        .summary
                        .thermo_constitutive_material_spread_ratio;
                    thermo_assignment_heterogeneity_index = gpu_results
                        .data
                        .summary
                        .thermo_assignment_heterogeneity_index;
                    thermo_region_delta_count = gpu_results.data.summary.thermo_region_delta_count;
                    thermo_spatial_coverage_ratio =
                        gpu_results.data.summary.thermo_spatial_coverage_ratio;
                    thermo_field_extrapolation_ratio =
                        gpu_results.data.summary.thermo_field_extrapolation_ratio;
                    thermo_field_clamp_ratio = gpu_results.data.summary.thermo_field_clamp_ratio;
                    thermo_transient_severity = gpu_results.data.summary.thermo_transient_severity;
                    thermo_nonlinear_severity = gpu_results.data.summary.thermo_nonlinear_severity;
                    electro_thermal_coupling_enabled =
                        gpu_results.data.summary.electro_thermal_coupling_enabled;
                    electro_thermal_coupling_fingerprint = gpu_results
                        .data
                        .summary
                        .electro_thermal_coupling_fingerprint;
                    electro_joule_heating_scale =
                        gpu_results.data.summary.electro_joule_heating_scale;
                    electro_conductivity_spread_ratio =
                        gpu_results.data.summary.electro_conductivity_spread_ratio;
                    electro_transient_severity =
                        gpu_results.data.summary.electro_transient_severity;
                    electro_transient_time_scale_mean =
                        gpu_results.data.summary.electro_transient_time_scale_mean;
                    electro_nonlinear_severity =
                        gpu_results.data.summary.electro_nonlinear_severity;
                    electro_nonlinear_time_scale_mean =
                        gpu_results.data.summary.electro_nonlinear_time_scale_mean;
                    plastic_nonlinear_severity =
                        gpu_results.data.summary.plastic_nonlinear_severity;
                    plastic_nonlinear_severity_mean =
                        gpu_results.data.summary.plastic_nonlinear_severity_mean;
                    plastic_load_realization_ratio =
                        gpu_results.data.summary.plastic_load_realization_ratio;
                    plastic_load_amplification_ratio =
                        gpu_results.data.summary.plastic_load_amplification_ratio;
                    contact_nonlinear_severity =
                        gpu_results.data.summary.contact_nonlinear_severity;
                    contact_nonlinear_severity_mean =
                        gpu_results.data.summary.contact_nonlinear_severity_mean;
                    contact_load_realization_ratio =
                        gpu_results.data.summary.contact_load_realization_ratio;
                    contact_load_amplification_ratio =
                        gpu_results.data.summary.contact_load_amplification_ratio;
                    thermal_max_residual_norm = gpu_results.data.summary.thermal_max_residual_norm;
                    thermal_min_temperature_k = gpu_results.data.summary.thermal_min_temperature_k;
                    thermal_max_temperature_k = gpu_results.data.summary.thermal_max_temperature_k;
                    thermal_conductivity_spread_ratio =
                        gpu_results.data.summary.thermal_conductivity_spread_ratio;
                    thermal_heat_capacity_spread_ratio =
                        gpu_results.data.summary.thermal_heat_capacity_spread_ratio;
                    thermal_spatial_gradient_index =
                        gpu_results.data.summary.thermal_spatial_gradient_index;
                    thermal_monotonic_response_fraction =
                        gpu_results.data.summary.thermal_monotonic_response_fraction;
                    thermal_response_realization_ratio =
                        gpu_results.data.summary.thermal_response_realization_ratio;
                    electromagnetic_enabled = gpu_results.data.summary.electromagnetic_enabled;
                    electromagnetic_formulation_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_formulation_coverage_ratio;
                    electromagnetic_magnetostatic_curl_curl_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_magnetostatic_curl_curl_coverage_ratio;
                    electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio;
                    electromagnetic_full_wave_displacement_current_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_full_wave_displacement_current_coverage_ratio;
                    electromagnetic_displacement_to_conduction_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_displacement_to_conduction_ratio;
                    electromagnetic_material_frequency_response_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_material_frequency_response_coverage_ratio;
                    electromagnetic_reference_frequency_hz = gpu_results
                        .data
                        .summary
                        .electromagnetic_reference_frequency_hz;
                    electromagnetic_applied_current_a =
                        gpu_results.data.summary.electromagnetic_applied_current_a;
                    electromagnetic_solve_quality =
                        gpu_results.data.summary.electromagnetic_solve_quality;
                    electromagnetic_conductivity_spread_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_conductivity_spread_ratio;
                    electromagnetic_relative_permittivity_spread_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_relative_permittivity_spread_ratio;
                    electromagnetic_relative_permeability_spread_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_relative_permeability_spread_ratio;
                    electromagnetic_material_heterogeneity_index = gpu_results
                        .data
                        .summary
                        .electromagnetic_material_heterogeneity_index;
                    electromagnetic_assignment_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_assignment_coverage_ratio;
                    electromagnetic_assigned_coefficient_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_assigned_coefficient_coverage_ratio;
                    electromagnetic_fallback_coefficient_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_fallback_coefficient_ratio;
                    electromagnetic_region_coefficient_contrast_index = gpu_results
                        .data
                        .summary
                        .electromagnetic_region_coefficient_contrast_index;
                    electromagnetic_condition_number_estimate = gpu_results
                        .data
                        .summary
                        .electromagnetic_condition_number_estimate;
                    electromagnetic_source_realization_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_source_realization_ratio;
                    electromagnetic_source_region_coverage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_source_region_coverage_ratio;
                    electromagnetic_source_material_alignment_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_source_material_alignment_ratio;
                    electromagnetic_source_localization_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_source_localization_ratio;
                    electromagnetic_source_overlap_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_source_overlap_ratio;
                    electromagnetic_source_interference_index = gpu_results
                        .data
                        .summary
                        .electromagnetic_source_interference_index;
                    electromagnetic_boundary_anchor_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_boundary_anchor_ratio;
                    electromagnetic_boundary_condition_localization_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_boundary_condition_localization_ratio;
                    electromagnetic_ground_anchor_effectiveness_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_ground_anchor_effectiveness_ratio;
                    electromagnetic_insulation_leakage_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_insulation_leakage_ratio;
                    electromagnetic_flux_divergence_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_flux_divergence_ratio;
                    electromagnetic_energy_imbalance_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_energy_imbalance_ratio;
                    electromagnetic_boundary_energy_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_boundary_energy_ratio;
                    electromagnetic_boundary_penalty_conditioning_contribution = gpu_results
                        .data
                        .summary
                        .electromagnetic_boundary_penalty_conditioning_contribution;
                    electromagnetic_source_region_energy_consistency_ratio = gpu_results
                        .data
                        .summary
                        .electromagnetic_source_region_energy_consistency_ratio;
                    electromagnetic_real_residual_norm =
                        gpu_results.data.summary.electromagnetic_real_residual_norm;
                    electromagnetic_imag_residual_norm =
                        gpu_results.data.summary.electromagnetic_imag_residual_norm;
                    electromagnetic_sweep_count =
                        gpu_results.data.summary.electromagnetic_sweep_count;
                    electromagnetic_resonance_peak_frequency_hz = gpu_results
                        .data
                        .summary
                        .electromagnetic_resonance_peak_frequency_hz;
                    electromagnetic_resonance_peak_flux_density = gpu_results
                        .data
                        .summary
                        .electromagnetic_resonance_peak_flux_density;
                    electromagnetic_resonance_bandwidth_hz = gpu_results
                        .data
                        .summary
                        .electromagnetic_resonance_bandwidth_hz;
                    electromagnetic_resonance_quality_factor = gpu_results
                        .data
                        .summary
                        .electromagnetic_resonance_quality_factor;
                    electromagnetic_resonance_flux_gain =
                        gpu_results.data.summary.electromagnetic_resonance_flux_gain;

                    if let Some(root) = filesystem_root {
                        runmat_runtime::analysis::storage::configure_artifact_store(
                            runmat_runtime::analysis::storage::AnalysisArtifactStoreConfig::Filesystem {
                                root: root.clone(),
                            },
                        )
                        .expect("reconfigure filesystem artifact store");

                        let persisted = analysis_results_by_run_id_op(
                            &gpu_envelope.data.run_id,
                            AnalysisResultsQuery {
                                include_fields: vec![gpu_primary_field_id],
                                include_field_values: true,

                                include_diagnostics: false,
                                diagnostic_codes: Vec::new(),
                                include_modal_results: false,
                                mode_indices: Vec::new(),
                                include_transient_results: false,
                                transient_snapshot_indices: Vec::new(),
                                include_nonlinear_results: false,
                                include_electromagnetic_results: false,
                            },
                            OperationContext::new(
                                Some(format!("trace-results-gpu-by-id-{}", spec.id)),
                                None,
                            ),
                        );
                        match persisted {
                            Ok(by_id) => {
                                if by_id.operation != "fea.results"
                                    || by_id.op_version != "fea.results/v1"
                                {
                                    failures.push(
                                        "fea.results by-run-id contract mismatch".to_string(),
                                    );
                                }
                            }
                            Err(err) => failures.push(format!(
                                "fea.results by-run-id failed for fixture {}: {}",
                                spec.id, err.error_code
                            )),
                        }
                    }

                    if let Some(max_sync) = spec.max_solver_host_sync_count {
                        let observed = gpu_envelope.data.provenance.solver_host_sync_count;
                        if observed > max_sync {
                            failures.push(format!(
                                "solver_host_sync_count exceeded for fixture {}: observed={} limit={}",
                                spec.id, observed, max_sync
                            ));
                        }
                    }
                    if let Some(min_ratio) = spec.min_solver_device_apply_k_ratio {
                        let observed = gpu_envelope.data.provenance.solver_device_apply_k_ratio;
                        if observed < min_ratio {
                            failures.push(format!(
                                "solver_device_apply_k_ratio below target for fixture {}: observed={} min={}",
                                spec.id, observed, min_ratio
                            ));
                        }
                    }

                    if let Some(expectation) = spec.residency_expectation {
                        match expectation {
                            ResidencyExpectation::DeviceRef => {
                                if !matches!(
                                    &gpu_results.data.fields[0].values,
                                    AnalysisFieldValues::DeviceRef(_)
                                ) {
                                    failures.push(
                                        "expected gpu displacement device_ref residency"
                                            .to_string(),
                                    );
                                }
                            }
                            ResidencyExpectation::HostFallback => {
                                if !matches!(
                                    &gpu_results.data.fields[0].values,
                                    AnalysisFieldValues::HostF64(_)
                                ) {
                                    failures.push(
                                        "expected gpu displacement host_f64 fallback".to_string(),
                                    );
                                }
                            }
                        }
                    }

                    if let Some(tol) = spec.parity_tolerance {
                        let cpu_primary_field_id = primary_result_field_id(spec.run_kind);
                        let cpu_results = analysis_results_op(
                            &cpu_envelope.data,
                            AnalysisResultsQuery {
                                include_fields: vec![cpu_primary_field_id],
                                include_field_values: true,
                                include_diagnostics: false,
                                diagnostic_codes: Vec::new(),
                                include_modal_results: false,
                                mode_indices: Vec::new(),
                                include_transient_results: false,
                                transient_snapshot_indices: Vec::new(),
                                include_nonlinear_results: false,
                                include_electromagnetic_results: false,
                            },
                            OperationContext::new(
                                Some(format!("trace-results-cpu-{}", spec.id)),
                                None,
                            ),
                        );
                        let cpu_results = match cpu_results {
                            Ok(value) => value,
                            Err(err) => {
                                failures.push(format!(
                                    "fea.results failed for cpu fixture {}: {}",
                                    spec.id, err.error_code
                                ));
                                return FixtureRunRecord {
                                    fixture_id: spec.id.to_string(),
                                    validate_ok,
                                    validate_error_code,
                                    run_ok,
                                    run_error_code,
                                    cpu_run_ms,
                                    gpu_run_ms,
                                    gpu_fallback_events,
                                    gpu_displacement_residency,
                                    gpu_solver_host_sync_count,
                                    gpu_solver_device_apply_k_ratio,
                                    gpu_speedup_ratio,
                                    gpu_solver_backend,
                                    gpu_transient_cache_hit_ratio,
                                    gpu_transient_cache_misses,
                                    gpu_transient_cache_entries,
                                    gpu_solver_prepared_build_ms,
                                    gpu_solver_solve_ms,
                                    gpu_solver_fallback_apply_count,
                                    prep_calibration_profile,
                                    prep_calibration_fingerprint,
                                    prep_acceptance_score,
                                    prep_acceptance_passed,
                                    prep_acceptance_fingerprint,
                                    thermo_coupling_enabled,
                                    thermo_coupling_fingerprint,
                                    thermo_constitutive_temperature_factor,
                                    thermo_effective_modulus_scale,
                                    thermo_constitutive_material_spread_ratio,
                                    thermo_assignment_heterogeneity_index,
                                    thermo_region_delta_count,
                                    thermo_spatial_coverage_ratio,
                                    thermo_field_extrapolation_ratio,
                                    thermo_field_clamp_ratio,
                                    thermo_field_artifact_id,
                                    thermo_field_artifact_approved,
                                    thermo_field_artifact_age_days,
                                    thermo_field_artifact_provenance_valid,
                                    thermo_transient_severity,
                                    thermo_nonlinear_severity,
                                    electro_thermal_coupling_enabled,
                                    electro_thermal_coupling_fingerprint,
                                    electro_joule_heating_scale,
                                    electro_conductivity_spread_ratio,
                                    electro_transient_severity,
                                    electro_transient_time_scale_mean,
                                    electro_nonlinear_severity,
                                    electro_nonlinear_time_scale_mean,
                                    plastic_nonlinear_severity,
                                    plastic_nonlinear_severity_mean,
                                    plastic_load_realization_ratio,
                                    plastic_load_amplification_ratio,
                                    contact_nonlinear_severity,
                                    contact_nonlinear_severity_mean,
                                    contact_load_realization_ratio,
                                    contact_load_amplification_ratio,
                                    thermal_max_residual_norm,
                                    thermal_min_temperature_k,
                                    thermal_max_temperature_k,
                                    thermal_conductivity_spread_ratio,
                                    thermal_heat_capacity_spread_ratio,
                                    thermal_spatial_gradient_index,
                                    thermal_monotonic_response_fraction,
                                    thermal_response_realization_ratio,
                                    electromagnetic_enabled,
                                    electromagnetic_formulation_coverage_ratio,
                                    electromagnetic_magnetostatic_curl_curl_coverage_ratio,
                                    electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio,
                                    electromagnetic_full_wave_displacement_current_coverage_ratio,
                                    electromagnetic_displacement_to_conduction_ratio,
                                    electromagnetic_material_frequency_response_coverage_ratio,
                                    electromagnetic_reference_frequency_hz,
                                    electromagnetic_applied_current_a,
                                    electromagnetic_solve_quality,
                                    electromagnetic_conductivity_spread_ratio,
                                    electromagnetic_relative_permittivity_spread_ratio,
                                    electromagnetic_relative_permeability_spread_ratio,
                                    electromagnetic_material_heterogeneity_index,
                                    electromagnetic_assignment_coverage_ratio,
                                    electromagnetic_assigned_coefficient_coverage_ratio,
                                    electromagnetic_fallback_coefficient_ratio,
                                    electromagnetic_region_coefficient_contrast_index,
                                    electromagnetic_condition_number_estimate,
                                    electromagnetic_source_realization_ratio,
                                    electromagnetic_source_region_coverage_ratio,
                                    electromagnetic_source_material_alignment_ratio,
                                    electromagnetic_source_localization_ratio,
                                    electromagnetic_source_overlap_ratio,
                                    electromagnetic_source_interference_index,
                                    electromagnetic_boundary_anchor_ratio,
                                    electromagnetic_boundary_condition_localization_ratio,
                                    electromagnetic_ground_anchor_effectiveness_ratio,
                                    electromagnetic_insulation_leakage_ratio,
                                    electromagnetic_flux_divergence_ratio,
                                    electromagnetic_energy_imbalance_ratio,
                                    electromagnetic_boundary_energy_ratio,
                                    electromagnetic_boundary_penalty_conditioning_contribution,
                                    electromagnetic_source_region_energy_consistency_ratio,
                                    electromagnetic_real_residual_norm,
                                    electromagnetic_imag_residual_norm,
                                    electromagnetic_sweep_count,
                                    electromagnetic_resonance_peak_frequency_hz,
                                    electromagnetic_resonance_peak_flux_density,
                                    electromagnetic_resonance_bandwidth_hz,
                                    electromagnetic_resonance_quality_factor,
                                    electromagnetic_resonance_flux_gain,
                                    publishable,
                                    parity,
                                    threshold_assertions,
                                    failures,
                                };
                            }
                        };

                        let cpu_values = cpu_results.data.fields[0].as_host_f64().unwrap_or(&[]);
                        let gpu_values = gpu_results.data.fields[0].as_host_f64().unwrap_or(&[]);

                        if cpu_values.is_empty() || gpu_values.is_empty() {
                            failures.push(
                                "parity check requested but host vectors were not available"
                                    .to_string(),
                            );
                        } else if cpu_values.len() != gpu_values.len() {
                            failures.push("parity vector length mismatch".to_string());
                        } else {
                            let summary = compute_parity(cpu_values, gpu_values);
                            if summary.max_abs_diff > tol.abs {
                                failures.push(format!(
                                    "parity abs diff {} exceeds {}",
                                    summary.max_abs_diff, tol.abs
                                ));
                            }
                            if summary.max_rel_diff > tol.rel {
                                failures.push(format!(
                                    "parity rel diff {} exceeds {}",
                                    summary.max_rel_diff, tol.rel
                                ));
                            }
                            parity = Some(summary);
                        }
                    }
                }
                Err(err) => {
                    run_error_code = Some(err.error_code.clone());
                    failures.push(format!(
                        "unexpected gpu run failure for fixture {}: {}",
                        spec.id, err.error_code
                    ));
                }
            }
        }
    }

    if let Some(expected_run_error) = spec.expect_run_error {
        let result = run_fixture_cpu(spec, &model);
        match result {
            Ok(_) => failures.push(format!(
                "expected run error code {expected_run_error}, but run succeeded"
            )),
            Err(err) => {
                run_error_code = Some(err.error_code.clone());
                if err.error_code != expected_run_error {
                    failures.push(format!(
                        "run error mismatch: expected {expected_run_error}, got {}",
                        err.error_code
                    ));
                }
            }
        }
    }

    FixtureRunRecord {
        fixture_id: spec.id.to_string(),
        validate_ok,
        validate_error_code,
        run_ok,
        run_error_code,
        cpu_run_ms,
        gpu_run_ms,
        gpu_fallback_events,
        gpu_displacement_residency,
        gpu_solver_host_sync_count,
        gpu_solver_device_apply_k_ratio,
        gpu_speedup_ratio,
        gpu_solver_backend,
        gpu_transient_cache_hit_ratio,
        gpu_transient_cache_misses,
        gpu_transient_cache_entries,
        gpu_solver_prepared_build_ms,
        gpu_solver_solve_ms,
        gpu_solver_fallback_apply_count,
        prep_calibration_profile,
        prep_calibration_fingerprint,
        prep_acceptance_score,
        prep_acceptance_passed,
        prep_acceptance_fingerprint,
        thermo_coupling_enabled,
        thermo_coupling_fingerprint,
        thermo_constitutive_temperature_factor,
        thermo_effective_modulus_scale,
        thermo_constitutive_material_spread_ratio,
        thermo_assignment_heterogeneity_index,
        thermo_region_delta_count,
        thermo_spatial_coverage_ratio,
        thermo_field_extrapolation_ratio,
        thermo_field_clamp_ratio,
        thermo_field_artifact_id,
        thermo_field_artifact_approved,
        thermo_field_artifact_age_days,
        thermo_field_artifact_provenance_valid,
        thermo_transient_severity,
        thermo_nonlinear_severity,
        electro_thermal_coupling_enabled,
        electro_thermal_coupling_fingerprint,
        electro_joule_heating_scale,
        electro_conductivity_spread_ratio,
        electro_transient_severity,
        electro_transient_time_scale_mean,
        electro_nonlinear_severity,
        electro_nonlinear_time_scale_mean,
        plastic_nonlinear_severity,
        plastic_nonlinear_severity_mean,
        plastic_load_realization_ratio,
        plastic_load_amplification_ratio,
        contact_nonlinear_severity,
        contact_nonlinear_severity_mean,
        contact_load_realization_ratio,
        contact_load_amplification_ratio,
        thermal_max_residual_norm,
        thermal_min_temperature_k,
        thermal_max_temperature_k,
        thermal_conductivity_spread_ratio,
        thermal_heat_capacity_spread_ratio,
        thermal_spatial_gradient_index,
        thermal_monotonic_response_fraction,
        thermal_response_realization_ratio,
        electromagnetic_enabled,
        electromagnetic_formulation_coverage_ratio,
        electromagnetic_magnetostatic_curl_curl_coverage_ratio,
        electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio,
        electromagnetic_full_wave_displacement_current_coverage_ratio,
        electromagnetic_displacement_to_conduction_ratio,
        electromagnetic_material_frequency_response_coverage_ratio,
        electromagnetic_reference_frequency_hz,
        electromagnetic_applied_current_a,
        electromagnetic_solve_quality,
        electromagnetic_conductivity_spread_ratio,
        electromagnetic_relative_permittivity_spread_ratio,
        electromagnetic_relative_permeability_spread_ratio,
        electromagnetic_material_heterogeneity_index,
        electromagnetic_assignment_coverage_ratio,
        electromagnetic_assigned_coefficient_coverage_ratio,
        electromagnetic_fallback_coefficient_ratio,
        electromagnetic_region_coefficient_contrast_index,
        electromagnetic_condition_number_estimate,
        electromagnetic_source_realization_ratio,
        electromagnetic_source_region_coverage_ratio,
        electromagnetic_source_material_alignment_ratio,
        electromagnetic_source_localization_ratio,
        electromagnetic_source_overlap_ratio,
        electromagnetic_source_interference_index,
        electromagnetic_boundary_anchor_ratio,
        electromagnetic_boundary_condition_localization_ratio,
        electromagnetic_ground_anchor_effectiveness_ratio,
        electromagnetic_insulation_leakage_ratio,
        electromagnetic_flux_divergence_ratio,
        electromagnetic_energy_imbalance_ratio,
        electromagnetic_boundary_energy_ratio,
        electromagnetic_boundary_penalty_conditioning_contribution,
        electromagnetic_source_region_energy_consistency_ratio,
        electromagnetic_real_residual_norm,
        electromagnetic_imag_residual_norm,
        electromagnetic_sweep_count,
        electromagnetic_resonance_peak_frequency_hz,
        electromagnetic_resonance_peak_flux_density,
        electromagnetic_resonance_bandwidth_hz,
        electromagnetic_resonance_quality_factor,
        electromagnetic_resonance_flux_gain,
        publishable,
        parity,
        threshold_assertions,
        failures,
    }
}
