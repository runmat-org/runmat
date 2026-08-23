use runmat_analysis_core::{AnalysisField, AnalysisFieldValues, AnalysisModel};
use runmat_analysis_fea::diagnostics::FeaDiagnostic;
use runmat_analysis_fea::{ComputeBackend, FeaProgressEvent, FeaRunResult};
use runmat_geometry_core::GeometryAsset;
use runmat_meshing_core::MeshingRequestSettings;
use runmat_meshing_evidence::MeshAuthoringSummary;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisValidateResult {
    pub valid: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisCreateModelIntentSpec {
    pub model_id: String,
    pub profile: AnalysisCreateModelProfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisCreateModelProfile {
    LinearStaticStructural,
    ThermoMechanicalCoupled,
    ElectroThermalCoupled,
    ThermalStandalone,
    ModalStructural,
    AcousticHarmonic,
    TransientStructural,
    NonlinearStructural,
    ElectromagneticStatic,
    CfdSteadyState,
    CfdTransient,
    ChtCoupled,
    FsiCoupled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalysisPhysicsProfileCatalogEntry {
    pub profile: AnalysisCreateModelProfile,
    pub label: &'static str,
    pub family: &'static str,
    pub target: &'static str,
    pub value: &'static str,
    pub default_outputs: &'static [AnalysisPhysicsProfileDefaultOutput],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalysisPhysicsProfileDefaultOutput {
    pub field: &'static str,
    pub location: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisRuntimePhysicsProfileCatalogEntry {
    pub profile: AnalysisCreateModelProfile,
    pub label: String,
    pub family: String,
    pub target: String,
    pub value: String,
    pub default_outputs: Vec<AnalysisRuntimePhysicsProfileDefaultOutput>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisRuntimePhysicsProfileDefaultOutput {
    pub field: String,
    pub location: String,
}

pub const ANALYSIS_PHYSICS_PROFILE_CATALOG: &[AnalysisPhysicsProfileCatalogEntry] = &[
    AnalysisPhysicsProfileCatalogEntry {
        profile: AnalysisCreateModelProfile::LinearStaticStructural,
        label: "Linear static structural",
        family: "structural",
        target: "solid mechanics",
        value: "static stress and displacement",
        default_outputs: &[
            AnalysisPhysicsProfileDefaultOutput {
                field: "structural.displacement",
                location: "nodes",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "structural.von_mises",
                location: "elements",
            },
        ],
    },
    AnalysisPhysicsProfileCatalogEntry {
        profile: AnalysisCreateModelProfile::ThermoMechanicalCoupled,
        label: "Thermo-mechanical",
        family: "coupled physics",
        target: "coupled solid mechanics and heat transfer",
        value: "coupled temperature, stress, and displacement",
        default_outputs: &[
            AnalysisPhysicsProfileDefaultOutput {
                field: "thermal.temperature",
                location: "nodes",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "structural.displacement",
                location: "nodes",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "structural.von_mises",
                location: "elements",
            },
        ],
    },
    AnalysisPhysicsProfileCatalogEntry {
        profile: AnalysisCreateModelProfile::ElectroThermalCoupled,
        label: "Electro-thermal",
        family: "coupled physics",
        target: "coupled electromagnetics and heat transfer",
        value: "resistive heating, temperature, and electrical fields",
        default_outputs: &[
            AnalysisPhysicsProfileDefaultOutput {
                field: "electro_thermal.temperature",
                location: "nodes",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "electro_thermal.joule_heat",
                location: "elements",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "electro_thermal.electric_potential",
                location: "nodes",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "electro_thermal.current_density",
                location: "elements",
            },
        ],
    },
    AnalysisPhysicsProfileCatalogEntry {
        profile: AnalysisCreateModelProfile::ThermalStandalone,
        label: "Thermal",
        family: "thermal",
        target: "heat transfer",
        value: "temperature and heat flux",
        default_outputs: &[
            AnalysisPhysicsProfileDefaultOutput {
                field: "thermal.temperature",
                location: "nodes",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "thermal.heat_flux",
                location: "elements",
            },
        ],
    },
    AnalysisPhysicsProfileCatalogEntry {
        profile: AnalysisCreateModelProfile::ModalStructural,
        label: "Modal structural",
        family: "modal",
        target: "solid mechanics",
        value: "natural frequencies and modes",
        default_outputs: &[
            AnalysisPhysicsProfileDefaultOutput {
                field: "structural.mode_shapes",
                location: "nodes",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "structural.natural_frequencies",
                location: "modes",
            },
        ],
    },
    AnalysisPhysicsProfileCatalogEntry {
        profile: AnalysisCreateModelProfile::AcousticHarmonic,
        label: "Acoustic harmonic",
        family: "acoustic",
        target: "acoustics",
        value: "frequency-domain pressure",
        default_outputs: &[
            AnalysisPhysicsProfileDefaultOutput {
                field: "acoustic.pressure",
                location: "nodes",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "acoustic.sound_pressure_level",
                location: "nodes",
            },
        ],
    },
    AnalysisPhysicsProfileCatalogEntry {
        profile: AnalysisCreateModelProfile::TransientStructural,
        label: "Transient structural",
        family: "structural",
        target: "solid mechanics",
        value: "time response",
        default_outputs: &[
            AnalysisPhysicsProfileDefaultOutput {
                field: "structural.displacement",
                location: "nodes",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "structural.von_mises",
                location: "elements",
            },
        ],
    },
    AnalysisPhysicsProfileCatalogEntry {
        profile: AnalysisCreateModelProfile::NonlinearStructural,
        label: "Nonlinear structural",
        family: "structural",
        target: "solid mechanics",
        value: "large deformation or nonlinear material",
        default_outputs: &[
            AnalysisPhysicsProfileDefaultOutput {
                field: "structural.displacement",
                location: "nodes",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "structural.von_mises",
                location: "elements",
            },
        ],
    },
    AnalysisPhysicsProfileCatalogEntry {
        profile: AnalysisCreateModelProfile::ElectromagneticStatic,
        label: "Electromagnetic",
        family: "electromagnetic",
        target: "electromagnetics",
        value: "electric and magnetic fields",
        default_outputs: &[
            AnalysisPhysicsProfileDefaultOutput {
                field: "em.electric_field",
                location: "elements",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "em.magnetic_flux_density",
                location: "elements",
            },
        ],
    },
    AnalysisPhysicsProfileCatalogEntry {
        profile: AnalysisCreateModelProfile::CfdSteadyState,
        label: "CFD steady state",
        family: "CFD",
        target: "fluid flow",
        value: "steady flow fields",
        default_outputs: &[
            AnalysisPhysicsProfileDefaultOutput {
                field: "fluid.velocity",
                location: "nodes",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "fluid.pressure",
                location: "nodes",
            },
        ],
    },
    AnalysisPhysicsProfileCatalogEntry {
        profile: AnalysisCreateModelProfile::CfdTransient,
        label: "CFD transient",
        family: "CFD",
        target: "fluid flow",
        value: "time-dependent flow fields",
        default_outputs: &[
            AnalysisPhysicsProfileDefaultOutput {
                field: "fluid.velocity",
                location: "nodes",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "fluid.pressure",
                location: "nodes",
            },
        ],
    },
    AnalysisPhysicsProfileCatalogEntry {
        profile: AnalysisCreateModelProfile::ChtCoupled,
        label: "Conjugate heat transfer",
        family: "coupled physics",
        target: "coupled fluid and solid heat transfer",
        value: "fluid and solid heat transfer",
        default_outputs: &[
            AnalysisPhysicsProfileDefaultOutput {
                field: "thermal.temperature",
                location: "nodes",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "fluid.velocity",
                location: "nodes",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "fluid.pressure",
                location: "nodes",
            },
        ],
    },
    AnalysisPhysicsProfileCatalogEntry {
        profile: AnalysisCreateModelProfile::FsiCoupled,
        label: "Fluid-structure interaction",
        family: "coupled physics",
        target: "coupled fluid and structural response",
        value: "fluid loads and structural response",
        default_outputs: &[
            AnalysisPhysicsProfileDefaultOutput {
                field: "structural.displacement",
                location: "nodes",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "structural.von_mises",
                location: "elements",
            },
            AnalysisPhysicsProfileDefaultOutput {
                field: "fluid.pressure",
                location: "nodes",
            },
        ],
    },
];

pub fn analysis_runtime_physics_profile_catalog() -> Vec<AnalysisRuntimePhysicsProfileCatalogEntry>
{
    ANALYSIS_PHYSICS_PROFILE_CATALOG
        .iter()
        .map(|entry| AnalysisRuntimePhysicsProfileCatalogEntry {
            profile: entry.profile,
            label: entry.label.to_string(),
            family: entry.family.to_string(),
            target: entry.target.to_string(),
            value: entry.value.to_string(),
            default_outputs: entry
                .default_outputs
                .iter()
                .map(|output| AnalysisRuntimePhysicsProfileDefaultOutput {
                    field: output.field.to_string(),
                    location: output.location.to_string(),
                })
                .collect(),
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionMode {
    Fp32,
    Fp64,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreconditionerMode {
    Auto,
    Jacobi,
    Amg,
    Ilu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityPolicy {
    Strict,
    Balanced,
    Exploratory,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoRegionTemperatureDelta {
    pub region_id: String,
    pub temperature_delta_k: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoTimeProfilePoint {
    pub normalized_time: f64,
    pub scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThermoFieldInterpolationMode {
    Linear,
    Step,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoFieldSource {
    pub source_id: String,
    pub revision: u32,
    #[serde(default)]
    pub interpolation_mode: Option<ThermoFieldInterpolationMode>,
    #[serde(default)]
    pub expected_region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoMechanicalCouplingOptions {
    pub enabled: bool,
    pub reference_temperature_k: f64,
    pub applied_temperature_delta_k: f64,
    pub thermal_expansion_coefficient: f64,
    #[serde(default)]
    pub field_artifact_id: Option<String>,
    #[serde(default)]
    pub field_source: Option<ThermoFieldSource>,
    #[serde(default)]
    pub region_temperature_deltas: Vec<ThermoRegionTemperatureDelta>,
    #[serde(default)]
    pub time_profile: Vec<ThermoTimeProfilePoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectroRegionConductivityScale {
    pub region_id: String,
    pub conductivity_scale: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectroTimeProfilePoint {
    pub normalized_time: f64,
    pub current_scale: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectroThermalCouplingOptions {
    pub enabled: bool,
    pub reference_temperature_k: f64,
    pub applied_voltage_v: f64,
    pub base_electrical_conductivity_s_per_m: f64,
    pub resistive_heating_coefficient: f64,
    #[serde(default)]
    pub region_conductivity_scales: Vec<ElectroRegionConductivityScale>,
    #[serde(default)]
    pub time_profile: Vec<ElectroTimeProfilePoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlasticityConstitutiveOptions {
    pub enabled: bool,
    pub yield_strain: f64,
    pub hardening_modulus_ratio: f64,
    pub saturation_exponent: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContactInterfaceOptions {
    pub enabled: bool,
    pub penalty_stiffness_scale: f64,
    pub max_penetration_ratio: f64,
    pub friction_coefficient: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityReasonCode {
    MaterialAssignmentConflict,
    SolverNotConverged,
    SolverBackendFallback,
    FieldPromotionFallback,
    FieldTopologyMismatch,
    SolidMeshNoVolumeElements,
    SolidMeshUnsupportedElementKind,
    SolidMeshUnmappedLoadRegion,
    SolidMeshUnmappedBoundaryConditionRegion,
    SolidMeshMaterialCoverageIncomplete,
    SolidMeshRenderTopologyIncomplete,
    SolidMeshValidationEvidenceMissing,
    SolidMeshValidationEvidenceFailed,
    SolidMeshQualityMinJacobianFailed,
    SolidMeshQualityAspectRatioWarn,
    SolidMeshBoundaryProjectionWarn,
    MissingSolidAnalysisMesh,
    ModalResidualExceeded,
    ModalOrthogonalityExceeded,
    ModalSeparationLow,
    TransientResidualExceeded,
    TransientStabilityExceeded,
    TransientStepFailure,
    ThermoMechanicalTransientStress,
    ThermoMechanicalConstitutiveSpreadHigh,
    ThermoMechanicalAssignmentHeterogeneityHigh,
    ThermoMechanicalGradientInstability,
    ThermoMechanicalFieldCoverageLow,
    ThermoMechanicalFieldExtrapolationHigh,
    ElectroThermalTransientStress,
    ElectroThermalNonlinearStress,
    ElectromagneticSolveQualityLow,
    ElectromagneticConductivitySpreadHigh,
    ElectromagneticMaterialHeterogeneityHigh,
    ElectromagneticAssignmentCoverageLow,
    ElectromagneticRegionContrastHigh,
    ElectromagneticConditioningHigh,
    ElectromagneticSourceRealizationLow,
    ElectromagneticSourceRegionCoverageLow,
    ElectromagneticSourceMaterialAlignmentLow,
    ElectromagneticSourceOverlapHigh,
    ElectromagneticSourceInterferenceHigh,
    ElectromagneticBoundaryLocalizationLow,
    ElectromagneticGroundAnchorEffectivenessLow,
    ElectromagneticInsulationLeakageHigh,
    ElectromagneticBoundaryAnchoringLow,
    ElectromagneticFluxDivergenceHigh,
    ElectromagneticEnergyImbalanceHigh,
    ElectromagneticBoundaryEnergyLow,
    ElectromagneticBoundaryPenaltyConditioningHigh,
    ElectromagneticSourceRegionEnergyConsistencyLow,
    ElectromagneticRealResidualHigh,
    ElectromagneticImagResidualHigh,
    ElectromagneticSweepCoverageLow,
    ElectromagneticResonanceSharpnessLow,
    PlasticityNonlinearStress,
    ContactNonlinearStress,
    NonlinearResidualExceeded,
    NonlinearIncrementFailure,
    ThermoMechanicalNonlinearStress,
    ThermalResidualExceeded,
    ThermalConstitutiveSpreadHigh,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityReason {
    pub code: QualityReasonCode,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub preconditioner_mode: PreconditionerMode,
    pub quality_policy: QualityPolicy,
    #[serde(default)]
    pub solver_mesh_artifact_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisElectromagneticRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub quality_policy: QualityPolicy,
    pub residual_target: f64,
    pub harmonic_tolerance: f64,
    pub harmonic_max_iterations: usize,
    #[serde(default)]
    pub solver_mesh_artifact_path: Option<String>,
    #[serde(default)]
    pub sweep_enabled: bool,
    #[serde(default)]
    pub sweep_frequency_hz: Vec<f64>,
}

impl Default for AnalysisElectromagneticRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            residual_target: 1.0e-6,
            harmonic_tolerance: 1.0e-7,
            harmonic_max_iterations: 96,
            solver_mesh_artifact_path: None,
            sweep_enabled: false,
            sweep_frequency_hz: Vec::new(),
        }
    }
}

impl Default for AnalysisRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
            solver_mesh_artifact_path: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisTransientRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub quality_policy: QualityPolicy,
    pub time_step_s: f64,
    pub min_time_step_s: f64,
    pub max_time_step_s: f64,
    pub step_count: usize,
    pub max_linear_iters: usize,
    pub tolerance: f64,
    pub residual_target: f64,
    pub adaptive_time_step: bool,
    pub max_step_retries: usize,
    pub adapt_min_scale: f64,
    pub adapt_max_scale: f64,
    pub adapt_growth_exponent: f64,
    pub adapt_retry_growth_cap: f64,
    pub adapt_nonconverged_shrink: f64,
    pub dt_bucket_rel_tolerance: f64,
    #[serde(default)]
    pub solver_mesh_artifact_path: Option<String>,
}

impl Default for AnalysisTransientRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            time_step_s: 1.0e-3,
            min_time_step_s: 1.0e-6,
            max_time_step_s: 2.0e-2,
            step_count: 10,
            max_linear_iters: 128,
            tolerance: 1.0e-8,
            residual_target: 1.0e-6,
            adaptive_time_step: true,
            max_step_retries: 4,
            adapt_min_scale: 0.8,
            adapt_max_scale: 1.25,
            adapt_growth_exponent: 0.35,
            adapt_retry_growth_cap: 1.05,
            adapt_nonconverged_shrink: 0.75,
            dt_bucket_rel_tolerance: 0.0,
            solver_mesh_artifact_path: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisAcousticRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub quality_policy: QualityPolicy,
    pub mode_count: usize,
    pub residual_warn_threshold: f64,
}

impl Default for AnalysisAcousticRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            mode_count: 3,
            residual_warn_threshold: 1.0e-3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisCfdRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub quality_policy: QualityPolicy,
    pub time_step_s: f64,
    pub step_count: usize,
    pub max_linear_iters: usize,
    pub tolerance: f64,
    pub residual_warn_threshold: f64,
    #[serde(default)]
    pub solver_mesh_artifact_path: Option<String>,
}

impl Default for AnalysisCfdRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            time_step_s: 1.0e-3,
            step_count: 12,
            max_linear_iters: 128,
            tolerance: 1.0e-8,
            residual_warn_threshold: 1.0e-5,
            solver_mesh_artifact_path: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisChtRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub quality_policy: QualityPolicy,
    pub time_step_s: f64,
    pub step_count: usize,
    pub max_linear_iters: usize,
    pub tolerance: f64,
    pub residual_warn_threshold: f64,
    #[serde(default)]
    pub solver_mesh_artifact_path: Option<String>,
}

impl Default for AnalysisChtRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            time_step_s: 1.0e-3,
            step_count: 12,
            max_linear_iters: 128,
            tolerance: 1.0e-8,
            residual_warn_threshold: 1.0e-4,
            solver_mesh_artifact_path: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisFsiRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub quality_policy: QualityPolicy,
    pub time_step_s: f64,
    pub step_count: usize,
    pub max_linear_iters: usize,
    pub tolerance: f64,
    pub residual_warn_threshold: f64,
    #[serde(default)]
    pub solver_mesh_artifact_path: Option<String>,
}

impl Default for AnalysisFsiRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            time_step_s: 1.0e-3,
            step_count: 12,
            max_linear_iters: 128,
            tolerance: 1.0e-8,
            residual_warn_threshold: 1.0e-4,
            solver_mesh_artifact_path: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisThermalRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub quality_policy: QualityPolicy,
    pub step_count: usize,
    pub time_step_s: f64,
    pub residual_warn_threshold: f64,
    #[serde(default)]
    pub solver_mesh_artifact_path: Option<String>,
}

impl Default for AnalysisThermalRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            step_count: 10,
            time_step_s: 1.0e-2,
            residual_warn_threshold: 1.0e-4,
            solver_mesh_artifact_path: None,
        }
    }
}

impl AnalysisTransientRunOptions {
    pub fn coarse() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp32,
            quality_policy: QualityPolicy::Exploratory,
            time_step_s: 5.0e-3,
            min_time_step_s: 5.0e-4,
            max_time_step_s: 2.0e-2,
            step_count: 6,
            max_linear_iters: 64,
            tolerance: 1.0e-6,
            residual_target: 1.0e-4,
            adaptive_time_step: true,
            max_step_retries: 2,
            adapt_min_scale: 0.75,
            adapt_max_scale: 1.3,
            adapt_growth_exponent: 0.3,
            adapt_retry_growth_cap: 1.02,
            adapt_nonconverged_shrink: 0.7,
            dt_bucket_rel_tolerance: 0.02,
            solver_mesh_artifact_path: None,
        }
    }

    pub fn balanced() -> Self {
        Self::default()
    }

    pub fn solid_recommended() -> Self {
        Self {
            quality_policy: QualityPolicy::Balanced,
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            dt_bucket_rel_tolerance: 0.01,
            ..Self::balanced()
        }
    }

    pub fn high_accuracy() -> Self {
        Self {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Strict,
            time_step_s: 5.0e-4,
            min_time_step_s: 5.0e-6,
            max_time_step_s: 2.0e-3,
            step_count: 24,
            max_linear_iters: 256,
            tolerance: 1.0e-10,
            residual_target: 1.0e-7,
            adaptive_time_step: true,
            max_step_retries: 8,
            adapt_min_scale: 0.85,
            adapt_max_scale: 1.2,
            adapt_growth_exponent: 0.45,
            adapt_retry_growth_cap: 1.03,
            adapt_nonconverged_shrink: 0.8,
            dt_bucket_rel_tolerance: 0.005,
            solver_mesh_artifact_path: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisModalRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub quality_policy: QualityPolicy,
    pub mode_count: usize,
    pub residual_warn_threshold: f64,
    #[serde(default)]
    pub solver_mesh_artifact_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisNonlinearRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub quality_policy: QualityPolicy,
    pub increment_count: usize,
    pub max_newton_iters: usize,
    pub tolerance: f64,
    pub residual_convergence_factor: f64,
    pub increment_norm_tolerance: f64,
    pub line_search: bool,
    pub max_line_search_backtracks: usize,
    pub line_search_reduction: f64,
    pub tangent_refresh_interval: usize,
    #[serde(default)]
    pub solver_mesh_artifact_path: Option<String>,
}

impl Default for AnalysisNonlinearRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            increment_count: 12,
            max_newton_iters: 24,
            tolerance: 1.0e-6,
            residual_convergence_factor: 5.0,
            increment_norm_tolerance: 1.0e-7,
            line_search: true,
            max_line_search_backtracks: 6,
            line_search_reduction: 0.5,
            tangent_refresh_interval: 2,
            solver_mesh_artifact_path: None,
        }
    }
}

impl AnalysisNonlinearRunOptions {
    pub fn coarse() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp32,
            quality_policy: QualityPolicy::Exploratory,
            increment_count: 8,
            max_newton_iters: 16,
            tolerance: 5.0e-6,
            residual_convergence_factor: 8.0,
            increment_norm_tolerance: 5.0e-7,
            line_search: false,
            max_line_search_backtracks: 0,
            line_search_reduction: 0.6,
            tangent_refresh_interval: 4,
            solver_mesh_artifact_path: None,
        }
    }

    pub fn balanced() -> Self {
        Self::default()
    }

    pub fn high_accuracy() -> Self {
        Self {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Strict,
            increment_count: 24,
            max_newton_iters: 40,
            tolerance: 1.0e-7,
            residual_convergence_factor: 3.0,
            increment_norm_tolerance: 5.0e-8,
            line_search: true,
            max_line_search_backtracks: 10,
            line_search_reduction: 0.5,
            tangent_refresh_interval: 1,
            solver_mesh_artifact_path: None,
        }
    }

    pub fn solid_recommended() -> Self {
        Self {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            increment_count: 24,
            max_newton_iters: 28,
            tolerance: 1.0e-6,
            residual_convergence_factor: 4.0,
            increment_norm_tolerance: 8.0e-8,
            line_search: true,
            max_line_search_backtracks: 8,
            line_search_reduction: 0.5,
            tangent_refresh_interval: 2,
            solver_mesh_artifact_path: None,
        }
    }
}

impl Default for AnalysisModalRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            mode_count: 3,
            residual_warn_threshold: 1.0e-3,
            solver_mesh_artifact_path: None,
        }
    }
}

impl AnalysisModalRunOptions {
    pub fn coarse() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp32,
            quality_policy: QualityPolicy::Exploratory,
            mode_count: 2,
            residual_warn_threshold: 5.0e-3,
            solver_mesh_artifact_path: None,
        }
    }

    pub fn balanced() -> Self {
        Self::default()
    }

    pub fn high_accuracy() -> Self {
        Self {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Strict,
            mode_count: 8,
            residual_warn_threshold: 5.0e-4,
            solver_mesh_artifact_path: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityGate {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Publishable,
    Degraded,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunProvenance {
    pub backend: ComputeBackend,
    pub solver_backend: String,
    pub solver_device_apply_k_ratio: f64,
    pub solver_host_sync_count: u32,
    pub precision_mode: String,
    pub deterministic_mode: bool,
    pub solver_method: String,
    pub preconditioner: String,
    pub quality_policy: String,
    pub fallback_events: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisRenderTopology {
    pub schema_version: String,
    pub source: AnalysisRenderTopologySource,
    pub meshes: Vec<AnalysisRenderMesh>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisRenderTopologySource {
    SolverPrep,
    AnalysisMesh,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisRenderMesh {
    pub mesh_id: String,
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<AnalysisRenderRegion>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vertex_volume_node_indices: Vec<Option<usize>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triangle_volume_element_indices: Vec<Option<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisRenderRegion {
    pub region_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triangle_ranges: Vec<AnalysisRenderTriangleRange>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisRenderTriangleRange {
    pub start: u32,
    pub count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisRunResult {
    pub run_id: String,
    pub run: FeaRunResult,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_topology: Option<AnalysisRenderTopology>,
    pub modal_results: Option<ModalResultsData>,
    #[serde(default)]
    pub thermal_results: Option<ThermalResultsData>,
    pub transient_results: Option<TransientResultsData>,
    pub nonlinear_results: Option<NonlinearResultsData>,
    #[serde(default)]
    pub electromagnetic_results: Option<ElectromagneticResultsData>,
    pub model_validity: QualityGate,
    pub solver_convergence: QualityGate,
    pub result_quality: QualityGate,
    pub run_status: RunStatus,
    pub publishable: bool,
    pub quality_reasons: Vec<QualityReason>,
    pub provenance: RunProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisArtifactRecord {
    pub run_id: String,
    pub created_at: String,
    pub op_version: String,
    pub field_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisResultsQuery {
    pub include_fields: Vec<String>,
    pub include_field_values: bool,
    pub include_diagnostics: bool,
    pub diagnostic_codes: Vec<String>,
    pub include_modal_results: bool,
    pub mode_indices: Vec<usize>,
    pub include_transient_results: bool,
    pub transient_snapshot_indices: Vec<usize>,
    pub include_nonlinear_results: bool,
    pub include_electromagnetic_results: bool,
}

impl Default for AnalysisResultsQuery {
    fn default() -> Self {
        Self {
            include_fields: Vec::new(),
            include_field_values: true,
            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
            include_electromagnetic_results: true,
        }
    }
}

impl AnalysisResultsQuery {
    pub fn metadata_only() -> Self {
        Self {
            include_field_values: false,
            include_modal_results: false,
            include_transient_results: false,
            include_nonlinear_results: false,
            include_electromagnetic_results: false,
            ..Self::default()
        }
    }

    pub fn field_values(field_id: impl Into<String>) -> Self {
        Self {
            include_fields: vec![field_id.into()],
            include_field_values: true,
            include_diagnostics: false,
            include_modal_results: false,
            include_transient_results: false,
            include_nonlinear_results: false,
            include_electromagnetic_results: false,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisFieldKind {
    Scalar,
    Vector,
    Tensor,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisFieldStorage {
    HostF64,
    DeviceRef,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisFieldLocation {
    Node,
    Element,
    Edge,
    BoundaryFace,
    InterfaceFace,
    Mode,
    Global,
    #[default]
    Unknown,
}

pub const ANALYSIS_FIELD_DEFAULT_PAGE_SIZE: usize = 4096;
pub const ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT: usize = 256;
pub const ANALYSIS_RUN_DATASET_SCHEMA_VERSION: u32 = 1;
pub const ANALYSIS_FIELD_DESCRIPTORS_SCHEMA_VERSION: u32 = 1;
pub const ANALYSIS_DIAGNOSTICS_SCHEMA_VERSION: u32 = 1;
pub const ANALYSIS_OBJECT_ARTIFACT_METADATA_SCHEMA_VERSION: u32 = 1;
pub const ANALYSIS_RUN_DATASET_KIND: &str = "finite_element_run_dataset";
pub const ANALYSIS_DATASET_ARTIFACT_KIND: &str = "finite_element_dataset";
pub const ANALYSIS_FIELD_DESCRIPTORS_ARTIFACT_KIND: &str = "finite_element_field_descriptors";
pub const ANALYSIS_DIAGNOSTICS_ARTIFACT_KIND: &str = "finite_element_diagnostics";
pub const ANALYSIS_ARTIFACT_MANIFEST_KIND: &str = "finite_element_artifact_manifest";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisDocumentKind {
    Study,
    Sweep,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisRuntimeCapabilities {
    pub supported_document_extensions: Vec<String>,
    pub supported_geometry_extensions: Vec<String>,
    #[serde(default)]
    pub physics_profiles: Vec<AnalysisRuntimePhysicsProfileCatalogEntry>,
    pub supports_check: bool,
    pub supports_run: bool,
    pub supports_results: bool,
    pub supports_live_progress: bool,
    pub visualization_backend: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisDocumentCheckResult {
    pub path: String,
    pub document_kind: AnalysisDocumentKind,
    pub valid: bool,
    pub validation: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<JsonValue>,
    pub diagnostics: Vec<JsonValue>,
    pub evidence_artifact_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisDocumentRunResult {
    pub path: String,
    pub document_kind: AnalysisDocumentKind,
    pub run: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<JsonValue>,
    pub field_descriptors: Vec<AnalysisFieldDescriptor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_summary: Option<JsonValue>,
    pub figure_handles: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_manifest: Option<JsonValue>,
    pub diagnostics: Vec<JsonValue>,
    pub progress_events: Vec<FeaProgressEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisFieldRequestOptions {
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<usize>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisFieldPageResult {
    pub run_id: String,
    pub field_id: String,
    pub descriptor: AnalysisFieldDescriptor,
    pub values: Vec<f64>,
    pub offset: usize,
    pub count: usize,
    pub total_count: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisRunDatasetStudyRef {
    pub path: String,
    pub document_kind: Option<AnalysisDocumentKind>,
    pub valid: Option<bool>,
    pub evidence_artifact_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisRunDatasetFieldPagingPolicy {
    pub mode: String,
    pub policy: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisRunDatasetPayload {
    pub schema_version: u32,
    pub kind: String,
    pub run_id: String,
    pub doc_path: String,
    pub study: AnalysisRunDatasetStudyRef,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analysis_profile: Option<AnalysisCreateModelProfile>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physics_family: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analysis_run_kind: Option<AnalysisRunKind>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available_output_summary: Option<String>,
    pub run: JsonValue,
    pub result_summary: JsonValue,
    pub fields: Vec<AnalysisFieldDescriptor>,
    pub field_paging: AnalysisRunDatasetFieldPagingPolicy,
    pub diagnostics: Vec<JsonValue>,
    pub artifact_manifest: JsonValue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisFieldDescriptorsArtifactPayload {
    pub schema_version: u32,
    pub run_id: String,
    pub fields: Vec<AnalysisFieldDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisDiagnosticsArtifactPayload {
    pub schema_version: u32,
    pub run_id: String,
    pub diagnostics: Vec<JsonValue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisObjectArtifactMetadata {
    pub schema_version: u32,
    pub artifact_id: String,
    pub src: String,
    pub bytes: u64,
    pub mime: String,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub handle: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisFieldPagingDescriptor {
    pub mode: String,
    pub value_unit: String,
    pub total_count: Option<usize>,
    pub page_size: usize,
    pub page_count: Option<usize>,
    pub default_materialize_limit: usize,
}

impl AnalysisFieldPagingDescriptor {
    pub fn for_value_count(value_count: usize) -> Self {
        Self {
            mode: "paged".to_string(),
            value_unit: "scalar_component".to_string(),
            total_count: Some(value_count),
            page_size: ANALYSIS_FIELD_DEFAULT_PAGE_SIZE,
            page_count: Some(value_count.div_ceil(ANALYSIS_FIELD_DEFAULT_PAGE_SIZE)),
            default_materialize_limit: ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisFieldStorageRef {
    pub kind: String,
    pub field_id: String,
    pub storage: AnalysisFieldStorage,
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisFieldDescriptor {
    pub field_id: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub quantity: String,
    #[serde(default)]
    pub topology_id: Option<String>,
    #[serde(default)]
    pub element_kind: Option<String>,
    pub class_name: String,
    pub kind: AnalysisFieldKind,
    pub dtype: String,
    #[serde(default)]
    pub unit: Option<String>,
    #[serde(default)]
    pub location: AnalysisFieldLocation,
    pub shape: Vec<usize>,
    pub element_count: usize,
    #[serde(default)]
    pub entity_count: usize,
    #[serde(default)]
    pub value_count: usize,
    pub component_count: Option<usize>,
    pub residency: String,
    pub storage: AnalysisFieldStorage,
    pub size_bytes: Option<u64>,
    pub paging: AnalysisFieldPagingDescriptor,
    #[serde(rename = "storageRef")]
    pub storage_ref: AnalysisFieldStorageRef,
}

impl AnalysisFieldDescriptor {
    pub fn from_field(field: &AnalysisField) -> Self {
        let storage = match &field.values {
            AnalysisFieldValues::HostF64(_) => AnalysisFieldStorage::HostF64,
            AnalysisFieldValues::DeviceRef(_) => AnalysisFieldStorage::DeviceRef,
        };
        let kind = infer_field_kind(&field.field_id, &field.shape);
        let class_name = match kind {
            AnalysisFieldKind::Scalar => "fea.ScalarField",
            AnalysisFieldKind::Vector => "fea.VectorField",
            AnalysisFieldKind::Tensor => "fea.TensorField",
            AnalysisFieldKind::Unknown => "fea.Field",
        }
        .to_string();
        let residency = match storage {
            AnalysisFieldStorage::HostF64 => "cpu",
            AnalysisFieldStorage::DeviceRef => "gpu",
        }
        .to_string();
        let location = infer_field_location(&field.field_id);
        let component_count = infer_component_count(&field.field_id, &field.shape);
        let value_count = field.element_count();
        let entity_count = infer_entity_count(&field.shape, location, component_count);
        let size_bytes = value_count
            .checked_mul(std::mem::size_of::<f64>())
            .map(|bytes| bytes as u64);
        Self {
            field_id: field.field_id.clone(),
            family: infer_field_family(&field.field_id).to_string(),
            quantity: infer_field_quantity(&field.field_id).to_string(),
            topology_id: infer_field_topology_id(&field.field_id, location).map(str::to_string),
            element_kind: infer_field_element_kind(&field.field_id, location).map(str::to_string),
            class_name,
            kind,
            dtype: "double".to_string(),
            unit: infer_field_unit(&field.field_id).map(str::to_string),
            location,
            shape: field.shape.clone(),
            element_count: value_count,
            entity_count,
            value_count,
            component_count,
            residency,
            storage,
            size_bytes,
            paging: AnalysisFieldPagingDescriptor::for_value_count(value_count),
            storage_ref: AnalysisFieldStorageRef {
                kind: "runtime_field".to_string(),
                field_id: field.field_id.clone(),
                storage,
                size_bytes,
            },
        }
    }
}

fn infer_entity_count(
    shape: &[usize],
    location: AnalysisFieldLocation,
    component_count: Option<usize>,
) -> usize {
    if matches!(
        location,
        AnalysisFieldLocation::Global | AnalysisFieldLocation::Mode
    ) {
        return shape.iter().product();
    }
    if let Some(first_dim) = shape.first().copied() {
        if shape.len() > 1 || component_count.is_some() {
            return first_dim;
        }
    }
    shape.iter().product()
}

fn infer_field_family(field_id: &str) -> &str {
    field_id
        .split_once('.')
        .map_or("unknown", |(family, _)| family)
}

fn infer_field_quantity(field_id: &str) -> &str {
    let Some((_, rest)) = field_id.split_once('.') else {
        return field_id;
    };
    let Some((quantity, suffix)) = rest.rsplit_once('.') else {
        return rest;
    };
    if suffix.chars().all(|ch| ch.is_ascii_digit()) {
        quantity
    } else {
        rest
    }
}

fn infer_field_unit(field_id: &str) -> Option<&'static str> {
    let normalized = field_id.to_ascii_lowercase();
    if normalized.contains("frequency_hz") || normalized.contains("frequency_response") {
        return Some("Hz");
    }
    if normalized.contains("eigenvalue") {
        return Some("rad^2/s^2");
    }
    if normalized.contains("sound_pressure_level_db") {
        return Some("dB");
    }
    if normalized.contains("temperature_gradient") {
        return Some("K/m");
    }
    if normalized.contains("temperature") || normalized.contains("temperature_jump") {
        return Some("K");
    }
    if normalized.contains("electric_potential") {
        return Some("V");
    }
    if normalized.contains("electric_field") {
        return Some("V/m");
    }
    if normalized.contains("electric_flux_density") {
        return Some("C/m^2");
    }
    if normalized.contains("current_density") {
        return Some("A/m^2");
    }
    if normalized.contains("vector_potential") {
        return Some("Wb/m");
    }
    if normalized.contains("magnetic_flux_density") {
        return Some("T");
    }
    if normalized.contains("magnetic_field") {
        return Some("A/m");
    }
    if normalized.contains("poynting_vector")
        || normalized.contains("boundary_heat_flux")
        || normalized.contains("interface_heat_flux")
        || normalized.ends_with(".heat_flux")
        || normalized.contains(".heat_flux.")
    {
        return Some("W/m^2");
    }
    if normalized.contains("power_loss_density")
        || normalized.contains("joule_heat")
        || normalized.contains("heat_source")
    {
        return Some("W/m^3");
    }
    if normalized.contains("energy_density") {
        return Some("J/m^3");
    }
    if normalized.contains("kinetic_energy")
        || normalized.contains("strain_energy")
        || normalized.contains("total_strain_energy")
    {
        return Some("J");
    }
    if normalized.contains("modal_mass") {
        return Some("kg");
    }
    if normalized.contains("modal_stiffness") {
        return Some("N/m");
    }
    if normalized.contains("wall_shear_stress")
        || normalized.contains("contact_pressure")
        || normalized.contains("interface_pressure")
        || normalized.contains("fluid_pressure")
        || normalized.contains(".pressure")
        || normalized.contains("stress")
        || normalized.contains("traction")
        || normalized.contains("von_mises")
    {
        return Some("Pa");
    }
    if normalized.contains("reaction_force")
        || normalized.contains("beam_axial_force")
        || normalized.contains("beam_shear_force")
        || normalized.contains("shell_membrane_force")
        || normalized.contains("shell_transverse_shear")
    {
        return Some("N");
    }
    if normalized.contains("reaction_moment")
        || normalized.contains("beam_torsion_moment")
        || normalized.contains("beam_bending_moment")
        || normalized.contains("shell_bending_moment")
    {
        return Some("N*m");
    }
    if normalized.contains("rotation") {
        return Some("rad");
    }
    if normalized.contains("contact_gap")
        || normalized.contains("displacement")
        || normalized.contains("mode_shape")
    {
        return Some("m");
    }
    if normalized.contains("velocity") {
        return Some("m/s");
    }
    if normalized.contains("acceleration") {
        return Some("m/s^2");
    }
    if normalized.contains("strain")
        || normalized.contains("residual")
        || normalized.contains("equation_scale")
        || normalized.contains("load_factor")
        || normalized.contains("coupling_iteration_count")
        || normalized.contains("reynolds_number")
        || normalized.contains("phase")
        || normalized.contains("orthogonality")
        || normalized.contains("participation_factor")
        || normalized.contains("relative_frequency_separation")
    {
        return Some("1");
    }
    None
}

fn infer_field_location(field_id: &str) -> AnalysisFieldLocation {
    let normalized = field_id.to_ascii_lowercase();
    if normalized.contains("nodal_von_mises") {
        return AnalysisFieldLocation::Node;
    }
    if normalized.contains("vector_potential") {
        return AnalysisFieldLocation::Edge;
    }
    if normalized.contains("wall_shear_stress") || normalized.contains("boundary_heat_flux") {
        return AnalysisFieldLocation::BoundaryFace;
    }
    if normalized.contains("interface_")
        || normalized.contains("contact_pressure")
        || normalized.contains("contact_gap")
    {
        return AnalysisFieldLocation::InterfaceFace;
    }
    if normalized.starts_with("modal.")
        && !normalized.starts_with("modal.mode_shape.")
        && !normalized.contains("orthogonality")
    {
        return AnalysisFieldLocation::Mode;
    }
    if normalized.contains("energy_density") {
        return AnalysisFieldLocation::Element;
    }
    if normalized.contains("residual")
        || normalized.contains("equation_scale")
        || normalized.contains("energy")
        || normalized.contains("load_factor")
        || normalized.contains("coupling_iteration_count")
        || normalized.contains("orthogonality")
    {
        return AnalysisFieldLocation::Global;
    }
    if normalized.starts_with("acoustic.") {
        return AnalysisFieldLocation::Node;
    }
    if normalized.contains("beam_") || normalized.contains("shell_") {
        return AnalysisFieldLocation::Element;
    }
    if normalized.contains("temperature_gradient")
        || normalized.contains("heat_flux")
        || normalized.contains("heat_source")
        || normalized.contains("joule_heat")
        || normalized.contains("magnetic_flux_density")
        || normalized.contains("magnetic_field")
        || normalized.contains("electric_field")
        || normalized.contains("electric_flux_density")
        || normalized.contains("current_density")
        || normalized.contains("poynting_vector")
        || normalized.contains("vorticity")
        || normalized.contains("pressure")
        || normalized.contains("stress")
        || normalized.contains("strain")
        || normalized.contains("von_mises")
    {
        return AnalysisFieldLocation::Element;
    }
    if normalized.contains("displacement")
        || normalized.contains("mode_shape")
        || normalized.contains("temperature")
        || normalized.starts_with("acoustic.")
        || normalized.contains("electric_potential")
    {
        return AnalysisFieldLocation::Node;
    }
    AnalysisFieldLocation::Element
}

fn infer_field_topology_id(
    field_id: &str,
    location: AnalysisFieldLocation,
) -> Option<&'static str> {
    let normalized = field_id.to_ascii_lowercase();
    if location == AnalysisFieldLocation::Global || location == AnalysisFieldLocation::Mode {
        return None;
    }
    if normalized.starts_with("structural.")
        || normalized.starts_with("thermo_mechanical.")
        || normalized.starts_with("transient.")
        || normalized.starts_with("nonlinear.")
    {
        return Some("analysis_mesh");
    }
    if normalized.starts_with("thermal.")
        || normalized.starts_with("electro_thermal.")
        || normalized.starts_with("em.")
        || normalized.starts_with("acoustic.")
        || normalized.starts_with("cfd.")
        || normalized.starts_with("cht.")
        || normalized.starts_with("fsi.")
    {
        return Some("analysis_topology");
    }
    None
}

fn infer_field_element_kind(
    field_id: &str,
    location: AnalysisFieldLocation,
) -> Option<&'static str> {
    let normalized = field_id.to_ascii_lowercase();
    if normalized.contains("beam_") {
        return Some("beam2");
    }
    if normalized.contains("shell_") {
        return Some("tri3");
    }
    if normalized.starts_with("structural.")
        && matches!(
            location,
            AnalysisFieldLocation::Element | AnalysisFieldLocation::BoundaryFace
        )
    {
        return Some("tetrahedron4");
    }
    None
}

fn infer_field_kind(field_id: &str, shape: &[usize]) -> AnalysisFieldKind {
    let normalized = field_id.to_ascii_lowercase();
    if normalized.contains("magnitude") {
        return AnalysisFieldKind::Scalar;
    }
    if normalized.contains("heat_flux") {
        return match shape {
            [_, components] if (2..=3).contains(components) => AnalysisFieldKind::Vector,
            _ => AnalysisFieldKind::Scalar,
        };
    }
    if normalized.contains("beam_shear_force")
        || normalized.contains("beam_bending_moment")
        || normalized.contains("beam_bending_stress")
        || normalized.contains("shell_membrane_force")
        || normalized.contains("shell_bending_moment")
        || normalized.contains("shell_transverse_shear")
    {
        return AnalysisFieldKind::Vector;
    }
    if normalized.contains("shell_von_mises") {
        return AnalysisFieldKind::Scalar;
    }
    if normalized.contains("beam_axial_force")
        || normalized.contains("beam_torsion_moment")
        || normalized.contains("beam_torsion_stress")
    {
        return AnalysisFieldKind::Scalar;
    }
    if normalized.contains("temperature_gradient")
        || normalized.contains("wall_shear_stress")
        || normalized.contains("vorticity")
        || normalized.contains("velocity")
        || normalized.contains("traction")
        || normalized.contains("magnetic_field")
        || normalized.contains("electric_field")
        || normalized.contains("current_density")
    {
        return AnalysisFieldKind::Vector;
    }
    if normalized.contains("von_mises")
        || normalized.contains("equivalent_plastic_strain")
        || normalized.contains("pressure")
        || normalized.contains("phase")
        || normalized.contains("reynolds_number")
        || normalized.contains("temperature")
        || normalized.contains("energy")
        || normalized.contains("residual")
    {
        return AnalysisFieldKind::Scalar;
    }
    if normalized.contains("stress") || normalized.contains("strain") {
        return AnalysisFieldKind::Tensor;
    }
    if normalized.contains("orthogonality") {
        return AnalysisFieldKind::Tensor;
    }
    if normalized.contains("displacement")
        || normalized.contains("rotation")
        || normalized.contains("mode_shape")
        || normalized.contains("reaction_force")
        || normalized.contains("reaction_moment")
        || normalized.contains("vector")
        || normalized.contains("flux")
    {
        return AnalysisFieldKind::Vector;
    }
    match shape {
        [] | [_] => AnalysisFieldKind::Scalar,
        [_, 1] => AnalysisFieldKind::Scalar,
        [_, 2] | [_, 3] => AnalysisFieldKind::Vector,
        [_, _, ..] => AnalysisFieldKind::Tensor,
    }
}

fn infer_component_count(field_id: &str, shape: &[usize]) -> Option<usize> {
    let normalized = field_id.to_ascii_lowercase();
    if normalized.contains("magnitude") {
        return None;
    }
    if normalized.contains("equivalent_plastic_strain") {
        return None;
    }
    if normalized.contains("heat_flux") {
        return match shape {
            [_, components] if (2..=3).contains(components) => Some(*components),
            _ => None,
        };
    }
    if normalized.contains("beam_shear_force")
        || normalized.contains("beam_bending_moment")
        || normalized.contains("beam_bending_stress")
    {
        return Some(2);
    }
    if normalized.contains("shell_membrane_force") || normalized.contains("shell_bending_moment") {
        return Some(3);
    }
    if normalized.contains("shell_transverse_shear") {
        return Some(2);
    }
    if normalized.contains("shell_von_mises") {
        return None;
    }
    if normalized.contains("beam_axial_force")
        || normalized.contains("beam_torsion_moment")
        || normalized.contains("beam_torsion_stress")
    {
        return None;
    }
    if normalized.contains("temperature_gradient")
        || normalized.contains("wall_shear_stress")
        || normalized.contains("vorticity")
        || normalized.contains("velocity")
        || normalized.contains("traction")
        || normalized.contains("magnetic_field")
        || normalized.contains("electric_field")
        || normalized.contains("current_density")
    {
        return Some(
            shape
                .last()
                .copied()
                .filter(|value| (2..=3).contains(value))
                .unwrap_or(3),
        );
    }
    if normalized.contains("von_mises")
        || normalized.contains("equivalent_plastic_strain")
        || normalized.contains("pressure")
        || normalized.contains("phase")
        || normalized.contains("reynolds_number")
        || normalized.contains("temperature")
        || normalized.contains("energy")
        || normalized.contains("residual")
    {
        return None;
    }
    if normalized.contains("displacement")
        || normalized.contains("rotation")
        || normalized.contains("mode_shape")
        || normalized.contains("reaction_force")
        || normalized.contains("reaction_moment")
        || normalized.contains("particle_velocity")
        || normalized.contains("vector")
        || normalized.contains("flux")
    {
        return Some(
            shape
                .last()
                .copied()
                .filter(|value| (2..=6).contains(value))
                .unwrap_or(3),
        );
    }
    if normalized.contains("stress") || normalized.contains("strain") {
        return Some(
            shape
                .last()
                .copied()
                .filter(|value| *value == 6)
                .unwrap_or(6),
        );
    }
    if normalized.contains("orthogonality") {
        return None;
    }
    match shape {
        [_, count] if (1..=6).contains(count) => Some(*count),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisResultsSummary {
    pub field_count: usize,
    pub total_elements: usize,
    pub mode_count: usize,
    pub available_mode_indices: Vec<usize>,
    pub min_frequency_hz: Option<f64>,
    pub max_frequency_hz: Option<f64>,
    pub max_modal_residual_norm: Option<f64>,
    pub first_mode_converged: Option<bool>,
    pub snapshot_count: usize,
    pub time_start_s: Option<f64>,
    pub time_end_s: Option<f64>,
    pub max_transient_residual_norm: Option<f64>,
    pub final_step_converged: Option<bool>,
    pub increment_count: usize,
    pub failed_increment_count: Option<usize>,
    pub max_nonlinear_residual_norm: Option<f64>,
    pub max_nonlinear_increment_norm: Option<f64>,
    pub max_nonlinear_iteration_count: Option<usize>,
    pub final_increment_converged: Option<bool>,
    pub nonlinear_line_search_backtracks: Option<usize>,
    pub nonlinear_max_backtracks_per_increment: Option<usize>,
    pub nonlinear_tangent_rebuild_count: Option<usize>,
    pub nonlinear_iteration_spike_count: Option<usize>,
    pub nonlinear_convergence_stall_count: Option<usize>,
    pub nonlinear_backtrack_burst_count: Option<usize>,
    pub thermo_coupling_enabled: Option<bool>,
    pub thermo_coupling_fingerprint: Option<u64>,
    pub thermo_constitutive_temperature_factor: Option<f64>,
    pub thermo_effective_modulus_scale: Option<f64>,
    pub thermo_constitutive_material_spread_ratio: Option<f64>,
    pub thermo_assignment_heterogeneity_index: Option<f64>,
    pub thermo_region_delta_count: Option<f64>,
    pub thermo_spatial_coverage_ratio: Option<f64>,
    pub thermo_field_extrapolation_ratio: Option<f64>,
    pub thermo_field_clamp_ratio: Option<f64>,
    pub thermo_transient_severity: Option<f64>,
    pub thermo_nonlinear_severity: Option<f64>,
    pub electro_thermal_coupling_enabled: Option<bool>,
    pub electro_thermal_coupling_fingerprint: Option<u64>,
    pub electro_joule_heating_scale: Option<f64>,
    pub electro_conductivity_spread_ratio: Option<f64>,
    pub electro_transient_severity: Option<f64>,
    pub electro_transient_time_scale_mean: Option<f64>,
    pub electro_nonlinear_severity: Option<f64>,
    pub electro_nonlinear_time_scale_mean: Option<f64>,
    pub plastic_nonlinear_severity: Option<f64>,
    pub plastic_nonlinear_severity_mean: Option<f64>,
    pub plastic_load_realization_ratio: Option<f64>,
    pub plastic_load_amplification_ratio: Option<f64>,
    pub contact_nonlinear_severity: Option<f64>,
    pub contact_nonlinear_severity_mean: Option<f64>,
    pub contact_load_realization_ratio: Option<f64>,
    pub contact_load_amplification_ratio: Option<f64>,
    pub thermal_max_residual_norm: Option<f64>,
    pub thermal_min_temperature_k: Option<f64>,
    pub thermal_max_temperature_k: Option<f64>,
    pub thermal_conductivity_spread_ratio: Option<f64>,
    pub thermal_heat_capacity_spread_ratio: Option<f64>,
    pub thermal_spatial_gradient_index: Option<f64>,
    pub thermal_monotonic_response_fraction: Option<f64>,
    pub thermal_response_realization_ratio: Option<f64>,
    pub electromagnetic_enabled: Option<bool>,
    pub electromagnetic_formulation_coverage_ratio: Option<f64>,
    pub electromagnetic_magnetostatic_curl_curl_coverage_ratio: Option<f64>,
    pub electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio: Option<f64>,
    pub electromagnetic_full_wave_displacement_current_coverage_ratio: Option<f64>,
    pub electromagnetic_displacement_to_conduction_ratio: Option<f64>,
    pub electromagnetic_material_frequency_response_coverage_ratio: Option<f64>,
    pub electromagnetic_reference_frequency_hz: Option<f64>,
    pub electromagnetic_applied_current_a: Option<f64>,
    pub electromagnetic_solve_quality: Option<f64>,
    pub electromagnetic_conductivity_spread_ratio: Option<f64>,
    pub electromagnetic_relative_permittivity_spread_ratio: Option<f64>,
    pub electromagnetic_relative_permeability_spread_ratio: Option<f64>,
    pub electromagnetic_material_heterogeneity_index: Option<f64>,
    pub electromagnetic_assignment_coverage_ratio: Option<f64>,
    pub electromagnetic_assigned_coefficient_coverage_ratio: Option<f64>,
    pub electromagnetic_region_coefficient_contrast_index: Option<f64>,
    pub electromagnetic_condition_number_estimate: Option<f64>,
    pub electromagnetic_source_realization_ratio: Option<f64>,
    pub electromagnetic_source_region_coverage_ratio: Option<f64>,
    pub electromagnetic_source_material_alignment_ratio: Option<f64>,
    pub electromagnetic_source_localization_ratio: Option<f64>,
    pub electromagnetic_source_overlap_ratio: Option<f64>,
    pub electromagnetic_source_interference_index: Option<f64>,
    pub electromagnetic_boundary_anchor_ratio: Option<f64>,
    pub electromagnetic_boundary_condition_localization_ratio: Option<f64>,
    pub electromagnetic_ground_anchor_effectiveness_ratio: Option<f64>,
    pub electromagnetic_insulation_leakage_ratio: Option<f64>,
    pub electromagnetic_flux_divergence_ratio: Option<f64>,
    pub electromagnetic_energy_imbalance_ratio: Option<f64>,
    pub electromagnetic_boundary_energy_ratio: Option<f64>,
    pub electromagnetic_boundary_penalty_conditioning_contribution: Option<f64>,
    pub electromagnetic_source_region_energy_consistency_ratio: Option<f64>,
    pub electromagnetic_real_residual_norm: Option<f64>,
    pub electromagnetic_imag_residual_norm: Option<f64>,
    pub electromagnetic_sweep_count: Option<f64>,
    pub electromagnetic_resonance_peak_frequency_hz: Option<f64>,
    pub electromagnetic_resonance_peak_flux_density: Option<f64>,
    pub electromagnetic_resonance_bandwidth_hz: Option<f64>,
    pub electromagnetic_resonance_quality_factor: Option<f64>,
    pub electromagnetic_resonance_flux_gain: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisResultsData {
    #[serde(default)]
    pub field_descriptors: Vec<AnalysisFieldDescriptor>,
    pub fields: Vec<AnalysisField>,
    pub modal_results: Option<ModalResultsData>,
    #[serde(default)]
    pub thermal_results: Option<ThermalResultsData>,
    pub transient_results: Option<TransientResultsData>,
    pub nonlinear_results: Option<NonlinearResultsData>,
    #[serde(default)]
    pub electromagnetic_results: Option<ElectromagneticResultsData>,
    pub diagnostics: Option<Vec<FeaDiagnostic>>,
    pub run_status: RunStatus,
    pub publishable: bool,
    pub quality_reasons: Vec<QualityReason>,
    pub provenance: RunProvenance,
    pub summary: AnalysisResultsSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisResultsCompareQuery {
    pub baseline_run_id: String,
    pub candidate_run_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisResultsCompareData {
    pub baseline_run_id: String,
    pub candidate_run_id: String,
    pub publishable_changed: bool,
    pub run_status_changed: bool,
    pub quality_reason_count_delta: i64,
    pub failed_increment_delta: Option<i64>,
    pub max_iteration_delta: Option<i64>,
    pub nonlinear_spike_count_delta: Option<i64>,
    pub nonlinear_stall_count_delta: Option<i64>,
    pub solve_ms_delta: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisTrendsQuery {
    pub window_size: usize,
}

impl Default for AnalysisTrendsQuery {
    fn default() -> Self {
        Self { window_size: 16 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisOutputRequest {
    pub id: String,
    pub field_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisStudySpec {
    pub study_id: String,
    pub geometry: GeometryAsset,
    pub create_model_intent: AnalysisCreateModelIntentSpec,
    #[serde(default)]
    pub model: Option<AnalysisModel>,
    pub run_kind: AnalysisRunKind,
    pub backend: ComputeBackend,
    #[serde(default)]
    pub meshing_settings: Option<MeshingRequestSettings>,
    #[serde(default)]
    pub outputs: Vec<AnalysisOutputRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solver_mesh_artifact_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meshing_evidence_artifact_path: Option<String>,
    #[serde(default)]
    pub linear_static_run_options: Option<AnalysisRunOptions>,
    #[serde(default)]
    pub modal_run_options: Option<AnalysisModalRunOptions>,
    #[serde(default)]
    pub acoustic_run_options: Option<AnalysisAcousticRunOptions>,
    #[serde(default)]
    pub thermal_run_options: Option<AnalysisThermalRunOptions>,
    #[serde(default)]
    pub transient_run_options: Option<AnalysisTransientRunOptions>,
    #[serde(default)]
    pub cfd_run_options: Option<AnalysisCfdRunOptions>,
    #[serde(default)]
    pub cht_run_options: Option<AnalysisChtRunOptions>,
    #[serde(default)]
    pub fsi_run_options: Option<AnalysisFsiRunOptions>,
    #[serde(default)]
    pub nonlinear_run_options: Option<AnalysisNonlinearRunOptions>,
    #[serde(default)]
    pub electromagnetic_run_options: Option<AnalysisElectromagneticRunOptions>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AnalysisStudyDiagramObservation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material_region_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundary_condition_region_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driving_condition_region_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structural_force_n: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisStudyAuthoringIntent {
    pub study_id: String,
    #[serde(default)]
    pub model_id: Option<String>,
    pub geometry: GeometryAsset,
    pub mesh_authoring_summary: MeshAuthoringSummary,
    pub profile: AnalysisCreateModelProfile,
    pub run_kind: AnalysisRunKind,
    pub backend: ComputeBackend,
    #[serde(default)]
    pub solver_mesh_artifact_path: Option<String>,
    #[serde(default)]
    pub meshing_evidence_artifact_path: Option<String>,
    #[serde(default)]
    pub material_region_id: Option<String>,
    #[serde(default)]
    pub boundary_condition_region_id: Option<String>,
    #[serde(default)]
    pub driving_condition_region_id: Option<String>,
    #[serde(default)]
    pub structural_force_n: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagram_observation: Option<AnalysisStudyDiagramObservation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisStudyAuthoringData {
    pub study: AnalysisStudySpec,
    pub evidence: AnalysisStudyAuthoringEvidence,
    pub evidence_artifact_path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisStudyAuthoringEvidence {
    pub schema_version: String,
    pub mesh_id: String,
    pub mesh_authoring_summary_schema_version: String,
    pub selected_material_region_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_boundary_condition_region_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_driving_condition_region_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_driving_condition_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_structural_force_n: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagram_artifact_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagram_source_mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagram_summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagram_confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solver_mesh_artifact_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meshing_evidence_artifact_path: Option<String>,
    pub material_region_source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundary_condition_region_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driving_condition_region_source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisStudyValidateResult {
    pub valid: bool,
    pub issue_codes: Vec<String>,
    #[serde(default)]
    pub issues: Vec<AnalysisStudyIssue>,
    pub evidence_artifact_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisStudyIssue {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisStudyPlanData {
    pub study_id: String,
    pub model_id: String,
    pub run_kind: AnalysisRunKind,
    pub backend: ComputeBackend,
    #[serde(default)]
    pub electromagnetic_run_options: Option<AnalysisElectromagneticRunOptions>,
    #[serde(default)]
    pub run_options: serde_json::Value,
    pub operation_sequence: Vec<String>,
    pub run_operation: String,
    pub run_op_version: String,
    pub study_fingerprint: String,
    pub evidence_artifact_path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisStudyRunData {
    pub study_id: String,
    pub model_id: String,
    pub model_profile: AnalysisCreateModelProfile,
    pub run_kind: AnalysisRunKind,
    pub backend: ComputeBackend,
    #[serde(default)]
    pub electromagnetic_run_options: Option<AnalysisElectromagneticRunOptions>,
    #[serde(default)]
    pub run_options: serde_json::Value,
    #[serde(default)]
    pub solver_mesh_artifact_path: Option<String>,
    #[serde(default)]
    pub meshing_evidence_artifact_path: Option<String>,
    pub study_fingerprint: String,
    pub operation_sequence: Vec<String>,
    pub run_operation: String,
    pub run_op_version: String,
    pub run_id: String,
    pub run_status: RunStatus,
    pub publishable: bool,
    pub solver_convergence: QualityGate,
    pub result_quality: QualityGate,
    #[serde(default)]
    pub quality_reasons: Vec<QualityReason>,
    pub provenance: RunProvenance,
    pub evidence_artifact_path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisStudySweepSpec {
    pub sweep_id: String,
    pub studies: Vec<AnalysisStudySpec>,
    #[serde(default = "default_study_sweep_fail_fast")]
    pub fail_fast: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisStudySweepValidateEntry {
    pub study_id: String,
    pub valid: bool,
    pub issue_codes: Vec<String>,
    #[serde(default)]
    pub issues: Vec<AnalysisStudyIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisStudySweepValidateData {
    pub sweep_id: String,
    pub valid: bool,
    pub issue_codes: Vec<String>,
    pub study_entries: Vec<AnalysisStudySweepValidateEntry>,
    pub evidence_artifact_path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisStudySweepPlanEntry {
    pub study_id: String,
    pub model_id: String,
    pub run_kind: AnalysisRunKind,
    pub backend: ComputeBackend,
    #[serde(default)]
    pub electromagnetic_run_options: Option<AnalysisElectromagneticRunOptions>,
    #[serde(default)]
    pub run_options: serde_json::Value,
    pub operation_sequence: Vec<String>,
    pub run_operation: String,
    pub run_op_version: String,
    pub study_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisStudySweepPlanData {
    pub sweep_id: String,
    pub study_count: usize,
    pub planned_count: usize,
    pub failed_count: usize,
    #[serde(default)]
    pub failure_entries: Vec<AnalysisStudySweepFailureEntry>,
    pub plan_entries: Vec<AnalysisStudySweepPlanEntry>,
    pub evidence_artifact_path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisStudySweepRunEntry {
    pub study_id: String,
    pub run_kind: AnalysisRunKind,
    pub run_id: String,
    pub run_status: RunStatus,
    pub publishable: bool,
    pub run_operation: String,
    pub run_op_version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisStudySweepData {
    pub sweep_id: String,
    pub study_count: usize,
    pub success_count: usize,
    pub failed_count: usize,
    #[serde(default)]
    pub failure_entries: Vec<AnalysisStudySweepFailureEntry>,
    pub run_entries: Vec<AnalysisStudySweepRunEntry>,
    pub evidence_artifact_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisStudySweepFailureEntry {
    pub study_id: String,
    pub study_index: usize,
    pub error_code: String,
    pub message: String,
}

fn default_study_sweep_fail_fast() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisRunKind {
    LinearStatic,
    Modal,
    Acoustic,
    Thermal,
    Transient,
    Cfd,
    Cht,
    Fsi,
    Nonlinear,
    Electromagnetic,
}

impl AnalysisCreateModelProfile {
    pub fn from_snake_case(value: &str) -> Option<Self> {
        serde_json::from_value(JsonValue::String(value.trim().to_ascii_lowercase())).ok()
    }

    pub fn as_snake_case(self) -> &'static str {
        match self {
            AnalysisCreateModelProfile::LinearStaticStructural => "linear_static_structural",
            AnalysisCreateModelProfile::ThermoMechanicalCoupled => "thermo_mechanical_coupled",
            AnalysisCreateModelProfile::ElectroThermalCoupled => "electro_thermal_coupled",
            AnalysisCreateModelProfile::ThermalStandalone => "thermal_standalone",
            AnalysisCreateModelProfile::ModalStructural => "modal_structural",
            AnalysisCreateModelProfile::AcousticHarmonic => "acoustic_harmonic",
            AnalysisCreateModelProfile::TransientStructural => "transient_structural",
            AnalysisCreateModelProfile::NonlinearStructural => "nonlinear_structural",
            AnalysisCreateModelProfile::ElectromagneticStatic => "electromagnetic_static",
            AnalysisCreateModelProfile::CfdSteadyState => "cfd_steady_state",
            AnalysisCreateModelProfile::CfdTransient => "cfd_transient",
            AnalysisCreateModelProfile::ChtCoupled => "cht_coupled",
            AnalysisCreateModelProfile::FsiCoupled => "fsi_coupled",
        }
    }

    pub fn catalog_entry(self) -> &'static AnalysisPhysicsProfileCatalogEntry {
        ANALYSIS_PHYSICS_PROFILE_CATALOG
            .iter()
            .find(|entry| entry.profile == self)
            .expect("analysis profile catalog must contain every profile variant")
    }

    pub fn derived_run_kind(self) -> AnalysisRunKind {
        debug_assert!(
            ANALYSIS_PHYSICS_PROFILE_CATALOG
                .iter()
                .any(|entry| entry.profile == self),
            "analysis profile catalog is missing {:?}",
            self
        );
        match self {
            AnalysisCreateModelProfile::LinearStaticStructural => AnalysisRunKind::LinearStatic,
            AnalysisCreateModelProfile::ThermoMechanicalCoupled => AnalysisRunKind::Transient,
            AnalysisCreateModelProfile::ElectroThermalCoupled => AnalysisRunKind::Transient,
            AnalysisCreateModelProfile::ThermalStandalone => AnalysisRunKind::Thermal,
            AnalysisCreateModelProfile::ModalStructural => AnalysisRunKind::Modal,
            AnalysisCreateModelProfile::AcousticHarmonic => AnalysisRunKind::Acoustic,
            AnalysisCreateModelProfile::TransientStructural => AnalysisRunKind::Transient,
            AnalysisCreateModelProfile::NonlinearStructural => AnalysisRunKind::Nonlinear,
            AnalysisCreateModelProfile::ElectromagneticStatic => AnalysisRunKind::Electromagnetic,
            AnalysisCreateModelProfile::CfdSteadyState
            | AnalysisCreateModelProfile::CfdTransient => AnalysisRunKind::Cfd,
            AnalysisCreateModelProfile::ChtCoupled => AnalysisRunKind::Cht,
            AnalysisCreateModelProfile::FsiCoupled => AnalysisRunKind::Fsi,
        }
    }
}

impl AnalysisRunKind {
    pub const fn as_snake_case(self) -> &'static str {
        match self {
            AnalysisRunKind::LinearStatic => "linear_static",
            AnalysisRunKind::Modal => "modal",
            AnalysisRunKind::Acoustic => "acoustic",
            AnalysisRunKind::Thermal => "thermal",
            AnalysisRunKind::Transient => "transient",
            AnalysisRunKind::Cfd => "cfd",
            AnalysisRunKind::Cht => "cht",
            AnalysisRunKind::Fsi => "fsi",
            AnalysisRunKind::Nonlinear => "nonlinear",
            AnalysisRunKind::Electromagnetic => "electromagnetic",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisTrendKindSummary {
    pub run_kind: AnalysisRunKind,
    pub sample_count: usize,
    pub median_solve_ms: Option<f64>,
    pub p95_solve_ms: Option<f64>,
    pub publishable_rate: f64,
    pub failed_increment_rate: Option<f64>,
    pub mean_spike_count: Option<f64>,
    pub mean_stall_count: Option<f64>,
    pub thermo_coupling_enabled_rate: Option<f64>,
    pub thermo_transient_warn_rate: Option<f64>,
    pub thermo_nonlinear_warn_rate: Option<f64>,
    pub thermo_spread_breach_rate: Option<f64>,
    pub thermo_heterogeneity_breach_rate: Option<f64>,
    pub electro_thermal_coupling_enabled_rate: Option<f64>,
    pub electro_transient_warn_rate: Option<f64>,
    pub electro_nonlinear_warn_rate: Option<f64>,
    pub plastic_nonlinear_warn_rate: Option<f64>,
    pub contact_nonlinear_warn_rate: Option<f64>,
    pub thermal_stability_warn_rate: Option<f64>,
    pub thermal_constitutive_warn_rate: Option<f64>,
    pub thermal_spread_breach_rate: Option<f64>,
    pub electromagnetic_solve_warn_rate: Option<f64>,
    pub electromagnetic_spread_breach_rate: Option<f64>,
    pub electromagnetic_heterogeneity_breach_rate: Option<f64>,
    pub electromagnetic_coverage_breach_rate: Option<f64>,
    pub electromagnetic_contrast_breach_rate: Option<f64>,
    pub electromagnetic_conditioning_breach_rate: Option<f64>,
    pub electromagnetic_source_realization_breach_rate: Option<f64>,
    pub electromagnetic_source_region_coverage_breach_rate: Option<f64>,
    pub electromagnetic_source_material_alignment_breach_rate: Option<f64>,
    pub electromagnetic_source_overlap_breach_rate: Option<f64>,
    pub electromagnetic_source_interference_breach_rate: Option<f64>,
    pub electromagnetic_boundary_anchor_breach_rate: Option<f64>,
    pub electromagnetic_boundary_localization_breach_rate: Option<f64>,
    pub electromagnetic_ground_effectiveness_breach_rate: Option<f64>,
    pub electromagnetic_insulation_leakage_breach_rate: Option<f64>,
    pub electromagnetic_divergence_breach_rate: Option<f64>,
    pub electromagnetic_energy_imbalance_breach_rate: Option<f64>,
    pub electromagnetic_boundary_energy_breach_rate: Option<f64>,
    pub electromagnetic_boundary_penalty_contribution_breach_rate: Option<f64>,
    pub electromagnetic_source_region_energy_consistency_breach_rate: Option<f64>,
    pub electromagnetic_real_residual_breach_rate: Option<f64>,
    pub electromagnetic_imag_residual_breach_rate: Option<f64>,
    pub electromagnetic_sweep_coverage_breach_rate: Option<f64>,
    pub electromagnetic_resonance_sharpness_breach_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisTrendsData {
    pub window_size: usize,
    pub summaries: Vec<AnalysisTrendKindSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalResultsData {
    pub modal_payload_version: String,
    pub eigenvalues_hz: Vec<f64>,
    pub mode_shapes: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
    pub mode_units: ModalFrequencyUnits,
    pub frequency_basis: ModalFrequencyBasis,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermalResultsData {
    pub thermal_payload_version: String,
    pub time_points_s: Vec<f64>,
    pub temperature_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub temperature_gradient_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub heat_flux_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub heat_source_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub boundary_heat_flux_snapshots: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
    pub reference_temperature_k: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientResultsData {
    pub transient_payload_version: String,
    pub time_points_s: Vec<f64>,
    pub displacement_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub rotation_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub velocity_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub angular_velocity_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub acceleration_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub angular_acceleration_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub von_mises_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub kinetic_energy_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub strain_energy_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub residual_norm_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub thermo_mechanical_temperature_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub thermo_mechanical_thermal_strain_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub thermo_mechanical_thermal_stress_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub thermo_mechanical_displacement_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub thermo_mechanical_von_mises_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub thermo_mechanical_coupling_residual_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub electro_thermal_temperature_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub electro_thermal_thermal_residual_snapshots: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
    pub integration_method: TransientIntegrationMethod,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearResultsData {
    pub nonlinear_payload_version: String,
    pub load_factors: Vec<f64>,
    pub displacement_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub rotation_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub von_mises_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub plastic_strain_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub equivalent_plastic_strain_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub contact_pressure_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub contact_gap_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub load_factor_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub residual_norm_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub thermo_mechanical_temperature_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub thermo_mechanical_thermal_strain_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub thermo_mechanical_thermal_stress_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub thermo_mechanical_displacement_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub thermo_mechanical_von_mises_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub thermo_mechanical_coupling_residual_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub electro_thermal_temperature_snapshots: Vec<AnalysisField>,
    #[serde(default)]
    pub electro_thermal_thermal_residual_snapshots: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
    #[serde(default)]
    pub increment_norms: Vec<f64>,
    #[serde(default)]
    pub iteration_counts: Vec<usize>,
    #[serde(default)]
    pub failed_increments: usize,
    #[serde(default)]
    pub line_search_backtracks: usize,
    #[serde(default)]
    pub max_line_search_backtracks_per_increment: usize,
    #[serde(default)]
    pub tangent_rebuild_count: usize,
    #[serde(default)]
    pub iteration_spike_count: usize,
    #[serde(default)]
    pub convergence_stall_count: usize,
    #[serde(default)]
    pub backtrack_burst_count: usize,
    pub method: NonlinearMethod,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectromagneticResultsData {
    pub electromagnetic_payload_version: String,
    pub reference_frequency_hz: f64,
    pub applied_current_a: f64,
    pub vector_potential_real: AnalysisField,
    pub vector_potential_imag: AnalysisField,
    pub magnetic_flux_density_real: AnalysisField,
    pub magnetic_flux_density_imag: AnalysisField,
    pub magnetic_flux_density_magnitude: AnalysisField,
    pub magnetic_field_real: AnalysisField,
    pub magnetic_field_imag: AnalysisField,
    pub current_density_real: AnalysisField,
    pub current_density_imag: AnalysisField,
    pub electric_field_real: AnalysisField,
    pub electric_field_imag: AnalysisField,
    pub power_loss_density: AnalysisField,
    pub energy_density: AnalysisField,
    pub residual_real: AnalysisField,
    pub residual_imag: AnalysisField,
    pub electric_flux_density_real: AnalysisField,
    pub electric_flux_density_imag: AnalysisField,
    pub poynting_vector_real: AnalysisField,
    pub poynting_vector_imag: AnalysisField,
    #[serde(default)]
    pub sweep_frequency_hz: Vec<f64>,
    #[serde(default)]
    pub sweep_peak_flux_density: Vec<f64>,
    #[serde(default)]
    pub sweep_solve_quality: Vec<f64>,
    #[serde(default)]
    pub resonance_peak_frequency_hz: Option<f64>,
    #[serde(default)]
    pub resonance_peak_flux_density: Option<f64>,
    #[serde(default)]
    pub resonance_bandwidth_hz: Option<f64>,
    #[serde(default)]
    pub resonance_quality_factor: Option<f64>,
    #[serde(default)]
    pub resonance_flux_gain: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransientIntegrationMethod {
    ImplicitEuler,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonlinearMethod {
    IncrementalNewtonRaphson,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModalFrequencyUnits {
    Hz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModalFrequencyBasis {
    NativeEigenSolve,
}

pub(crate) fn format_precision_mode(mode: PrecisionMode) -> String {
    match mode {
        PrecisionMode::Fp32 => "fp32".to_string(),
        PrecisionMode::Fp64 => "fp64".to_string(),
        PrecisionMode::Mixed => "mixed".to_string(),
    }
}

pub(crate) fn format_quality_policy(mode: QualityPolicy) -> String {
    match mode {
        QualityPolicy::Strict => "strict".to_string(),
        QualityPolicy::Balanced => "balanced".to_string(),
        QualityPolicy::Exploratory => "exploratory".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn physics_profile_catalog_covers_every_supported_profile_once() {
        let expected_profiles = [
            AnalysisCreateModelProfile::LinearStaticStructural,
            AnalysisCreateModelProfile::ThermoMechanicalCoupled,
            AnalysisCreateModelProfile::ElectroThermalCoupled,
            AnalysisCreateModelProfile::ThermalStandalone,
            AnalysisCreateModelProfile::ModalStructural,
            AnalysisCreateModelProfile::AcousticHarmonic,
            AnalysisCreateModelProfile::TransientStructural,
            AnalysisCreateModelProfile::NonlinearStructural,
            AnalysisCreateModelProfile::ElectromagneticStatic,
            AnalysisCreateModelProfile::CfdSteadyState,
            AnalysisCreateModelProfile::CfdTransient,
            AnalysisCreateModelProfile::ChtCoupled,
            AnalysisCreateModelProfile::FsiCoupled,
        ];

        assert_eq!(
            ANALYSIS_PHYSICS_PROFILE_CATALOG.len(),
            expected_profiles.len()
        );

        let mut seen = HashSet::new();
        for entry in ANALYSIS_PHYSICS_PROFILE_CATALOG {
            let serialized_profile = serde_json::to_value(entry.profile)
                .expect("profile should serialize")
                .as_str()
                .expect("profile should serialize as string")
                .to_string();
            assert!(
                expected_profiles.contains(&entry.profile),
                "catalog contains unsupported profile {:?}",
                entry.profile
            );
            assert_eq!(entry.profile.as_snake_case(), serialized_profile);
            assert_eq!(
                AnalysisCreateModelProfile::from_snake_case(entry.profile.as_snake_case()),
                Some(entry.profile)
            );
            assert!(
                seen.insert(entry.profile as u8),
                "catalog contains duplicate profile {:?}",
                entry.profile
            );
            assert!(!entry.label.trim().is_empty());
            assert!(!entry.family.trim().is_empty());
            assert!(!entry.target.trim().is_empty());
            assert!(!entry.value.trim().is_empty());
            assert!(
                !entry.default_outputs.is_empty(),
                "catalog profile {:?} needs at least one default output",
                entry.profile
            );
            for output in entry.default_outputs {
                assert!(!output.field.trim().is_empty());
                assert!(!output.location.trim().is_empty());
            }
        }

        for run_kind in [
            AnalysisRunKind::LinearStatic,
            AnalysisRunKind::Modal,
            AnalysisRunKind::Acoustic,
            AnalysisRunKind::Thermal,
            AnalysisRunKind::Transient,
            AnalysisRunKind::Cfd,
            AnalysisRunKind::Cht,
            AnalysisRunKind::Fsi,
            AnalysisRunKind::Nonlinear,
            AnalysisRunKind::Electromagnetic,
        ] {
            let serialized_run_kind = serde_json::to_value(run_kind)
                .expect("run kind should serialize")
                .as_str()
                .expect("run kind should serialize as string")
                .to_string();
            assert_eq!(run_kind.as_snake_case(), serialized_run_kind);
        }
    }

    #[test]
    fn runtime_physics_profile_catalog_exposes_full_family_set() {
        let runtime_catalog = analysis_runtime_physics_profile_catalog();

        assert_eq!(
            runtime_catalog.len(),
            ANALYSIS_PHYSICS_PROFILE_CATALOG.len(),
            "runtime profile catalog must serialize every supported physics profile"
        );

        let families: HashSet<&str> = runtime_catalog
            .iter()
            .map(|entry| entry.family.as_str())
            .collect();
        for family in [
            "structural",
            "modal",
            "thermal",
            "coupled physics",
            "electromagnetic",
            "acoustic",
            "CFD",
        ] {
            assert!(
                families.contains(family),
                "runtime profile catalog is missing {family}"
            );
        }

        for entry in runtime_catalog {
            assert!(!entry.label.trim().is_empty());
            assert!(!entry.target.trim().is_empty());
            assert!(!entry.value.trim().is_empty());
            assert!(!entry.default_outputs.is_empty());
        }
    }
}
