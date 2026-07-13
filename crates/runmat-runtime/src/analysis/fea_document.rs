use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use runmat_analysis_core::{
    AnalysisInterface, AnalysisModel, AnalysisModelId, AnalysisStep, BeamElementModel,
    BeamSectionModel, BoundaryCondition, BoundaryConditionKind, CfdDomain, ElectroThermalDomain,
    ElectromagneticDomain, EvidenceConfidence, LoadCase, LoadKind, MaterialAcousticModel,
    MaterialAssignment, MaterialElectricalModel, MaterialMechanicalModel, MaterialModel,
    MaterialPlasticModel, MaterialThermalModel, ReferenceFrame, ShellElementModel,
    ShellSectionModel, StructuralElement, StructuralElementKind, StructuralModel, StructuralNode,
    ThermoMechanicalDomain,
};
use runmat_analysis_fea::ComputeBackend;
use runmat_geometry_core::{GeometryAsset, UnitSystem};
use runmat_geometry_io::GeometryImportOptions;
use runmat_meshing_core::{
    MeshBackendKind, MeshElementOrder, MeshKindRequest, MeshProfile, MeshRefinementOptions,
    MeshTargetSize, MeshValidationPolicyOptions, QualityThresholds, RefinementConvergenceOptions,
    RefinementFocusLevel, RefinementFocusOptions, RefinementIndicatorMode,
    RefinementIndicatorOverrides, RefinementStrategy, VolumeElementKind, VolumeMeshingOptions,
};
use serde::de::{DeserializeOwned, Error as DeError};
use serde::{Deserialize, Deserializer};

use super::contracts::AnalysisOutputRequest;
use super::{
    analysis_create_model_op, AnalysisAcousticRunOptions, AnalysisCfdRunOptions,
    AnalysisChtRunOptions, AnalysisCreateModelIntentSpec, AnalysisCreateModelProfile,
    AnalysisElectromagneticRunOptions, AnalysisFsiRunOptions, AnalysisModalRunOptions,
    AnalysisNonlinearRunOptions, AnalysisRunKind, AnalysisRunOptions, AnalysisStudySpec,
    AnalysisStudySweepSpec, AnalysisThermalRunOptions, AnalysisTransientRunOptions,
};
use crate::operations::OperationContext;

const FEA_DOCUMENT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub enum FeaResolvedDocument {
    Study(Box<AnalysisStudySpec>),
    Sweep(AnalysisStudySweepSpec),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum RawFeaDocument {
    Study(Box<FeaStudyDocument>),
    Sweep(FeaSweepDocument),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaSweepDocument {
    version: u32,
    id: String,
    #[serde(default = "default_fail_fast")]
    fail_fast: bool,
    studies: Vec<FeaStudyDocument>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaStudyDocument {
    version: u32,
    id: String,
    geometry: FeaGeometryDocument,
    model: FeaModelDocument,
    run: FeaRunDocument,
    #[serde(default)]
    mesh: Option<FeaMeshDocument>,
    #[serde(default)]
    regions: BTreeMap<String, FeaRegionDocument>,
    #[serde(default)]
    materials: BTreeMap<String, FeaMaterialDocument>,
    #[serde(default)]
    material_assignments: Vec<FeaMaterialAssignmentDocument>,
    #[serde(default)]
    structural: Option<FeaStructuralDocument>,
    #[serde(default)]
    nodes: Vec<FeaStructuralNodeDocument>,
    #[serde(default)]
    elements: Vec<FeaStructuralElementDocument>,
    #[serde(default)]
    sections: Vec<FeaStructuralSectionDocument>,
    #[serde(default)]
    boundary_conditions: Vec<FeaBoundaryConditionDocument>,
    #[serde(default)]
    loads: Vec<FeaLoadDocument>,
    #[serde(default)]
    steps: Vec<FeaStepDocument>,
    #[serde(default)]
    outputs: Vec<FeaOutputDocument>,
    #[serde(default)]
    domains: FeaDomainsDocument,
    #[serde(default)]
    interfaces: Vec<AnalysisInterface>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaGeometryDocument {
    path: PathBuf,
    #[serde(default = "default_units")]
    units: UnitSystem,
    #[serde(default)]
    import: FeaGeometryImportDocument,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaGeometryImportDocument {
    #[serde(default)]
    max_triangles: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaModelDocument {
    #[serde(default)]
    id: Option<String>,
    profile: AnalysisCreateModelProfile,
    #[serde(default)]
    frame: Option<ReferenceFrame>,
    #[serde(default)]
    defaults: FeaModelDefaultsMode,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaMeshDocument {
    #[serde(default)]
    backend: MeshBackendKind,
    #[serde(default = "default_mesh_kind")]
    kind: MeshKindRequest,
    #[serde(default = "default_mesh_element")]
    element: VolumeElementKind,
    #[serde(default = "default_mesh_element_order")]
    element_order: MeshElementOrder,
    #[serde(default = "default_mesh_profile")]
    profile: MeshProfile,
    #[serde(default = "default_mesh_max_elements")]
    max_elements: usize,
    #[serde(default, deserialize_with = "deserialize_optional_mesh_target_size")]
    target_size: Option<MeshTargetSize>,
    #[serde(default)]
    min_size: Option<f64>,
    #[serde(default)]
    max_size: Option<f64>,
    #[serde(default)]
    growth_rate: Option<f64>,
    #[serde(default)]
    refinement: FeaMeshRefinementDocument,
    #[serde(default)]
    validation: FeaMeshValidationDocument,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaMeshValidationDocument {
    #[serde(default)]
    coverage: FeaMeshCoveragePreset,
    #[serde(default)]
    quality: FeaMeshQualityPreset,
    #[serde(default)]
    min_bounds_coverage_ratio: Option<f64>,
    #[serde(default)]
    min_volume_coverage_ratio: Option<f64>,
    #[serde(default)]
    min_boundary_area_ratio: Option<f64>,
    #[serde(default)]
    min_boundary_face_recovery_ratio: Option<f64>,
    #[serde(default)]
    min_boundary_edge_recovery_ratio: Option<f64>,
    #[serde(default)]
    max_volume_components: Option<usize>,
}

impl Default for FeaMeshValidationDocument {
    fn default() -> Self {
        Self {
            coverage: FeaMeshCoveragePreset::default(),
            quality: FeaMeshQualityPreset::default(),
            min_bounds_coverage_ratio: None,
            min_volume_coverage_ratio: None,
            min_boundary_area_ratio: None,
            min_boundary_face_recovery_ratio: None,
            min_boundary_edge_recovery_ratio: None,
            max_volume_components: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum FeaMeshCoveragePreset {
    Strict,
    #[default]
    Balanced,
    Relaxed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum FeaMeshQualityPreset {
    Strict,
    #[default]
    Balanced,
    Relaxed,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaMeshRefinementDocument {
    #[serde(default = "default_refinement_strategy")]
    strategy: RefinementStrategy,
    #[serde(default = "default_refinement_max_iterations")]
    max_iterations: usize,
    #[serde(default)]
    convergence: FeaRefinementConvergenceDocument,
    #[serde(default)]
    focus: FeaRefinementFocusDocument,
    #[serde(default)]
    indicators: BTreeMap<String, BTreeMap<String, RefinementIndicatorMode>>,
}

impl Default for FeaMeshRefinementDocument {
    fn default() -> Self {
        Self {
            strategy: default_refinement_strategy(),
            max_iterations: default_refinement_max_iterations(),
            convergence: FeaRefinementConvergenceDocument::default(),
            focus: FeaRefinementFocusDocument::default(),
            indicators: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaRefinementConvergenceDocument {
    #[serde(default = "default_field_change_tolerance")]
    field_change_tolerance: f64,
    #[serde(default = "default_energy_change_tolerance")]
    energy_change_tolerance: f64,
    #[serde(default, deserialize_with = "deserialize_optional_auto_f64")]
    residual_tolerance: Option<f64>,
}

impl Default for FeaRefinementConvergenceDocument {
    fn default() -> Self {
        Self {
            field_change_tolerance: default_field_change_tolerance(),
            energy_change_tolerance: default_energy_change_tolerance(),
            residual_tolerance: None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaRefinementFocusDocument {
    #[serde(default = "default_focus_fine")]
    loads: RefinementFocusLevel,
    #[serde(default = "default_focus_fine")]
    constraints: RefinementFocusLevel,
    #[serde(default = "default_focus_normal")]
    interfaces: RefinementFocusLevel,
    #[serde(default = "default_true")]
    curvature: bool,
    #[serde(default = "default_true")]
    small_features: bool,
}

impl Default for FeaRefinementFocusDocument {
    fn default() -> Self {
        Self {
            loads: default_focus_fine(),
            constraints: default_focus_fine(),
            interfaces: default_focus_normal(),
            curvature: true,
            small_features: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum FeaModelDefaultsMode {
    #[default]
    ProfileScaffold,
    None,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaRegionDocument {
    selector: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaMaterialDocument {
    #[serde(default)]
    name: Option<String>,
    mechanical: MaterialMechanicalModel,
    #[serde(default)]
    thermal: Option<MaterialThermalModel>,
    #[serde(default)]
    acoustic: Option<MaterialAcousticModel>,
    #[serde(default)]
    electrical: Option<MaterialElectricalModel>,
    #[serde(default)]
    plastic: Option<MaterialPlasticModel>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaMaterialAssignmentDocument {
    region: String,
    material: String,
    #[serde(default)]
    expected_material: Option<String>,
    #[serde(default = "default_assignment_confidence")]
    confidence: EvidenceConfidence,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaStructuralDocument {
    #[serde(default)]
    nodes: Vec<FeaStructuralNodeDocument>,
    #[serde(default)]
    elements: Vec<FeaStructuralElementDocument>,
    #[serde(default)]
    sections: Vec<FeaStructuralSectionDocument>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaStructuralNodeDocument {
    id: u32,
    coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaStructuralElementDocument {
    id: String,
    region: String,
    #[serde(rename = "type", alias = "kind")]
    element_type: FeaStructuralElementType,
    nodes: Vec<u32>,
    section: String,
    #[serde(default)]
    reference_axis: Option<[f64; 3]>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FeaStructuralElementType {
    Beam,
    Shell,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaStructuralSectionDocument {
    id: String,
    #[serde(rename = "type", alias = "kind", default)]
    section_type: FeaStructuralSectionType,
    #[serde(default)]
    area_m2: Option<f64>,
    #[serde(default)]
    iy_m4: Option<f64>,
    #[serde(default)]
    iz_m4: Option<f64>,
    #[serde(default)]
    torsion_j_m4: Option<f64>,
    #[serde(default)]
    thickness_m: Option<f64>,
    #[serde(default)]
    shear_correction: Option<f64>,
    #[serde(default)]
    drilling_stiffness_scale: Option<f64>,
    #[serde(default)]
    outer_fiber_y_m: f64,
    #[serde(default)]
    outer_fiber_z_m: f64,
    #[serde(default)]
    torsion_outer_radius_m: f64,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum FeaStructuralSectionType {
    #[default]
    Beam,
    BeamSection,
    Shell,
    ShellSection,
}

impl FeaStructuralSectionType {
    fn is_beam(self) -> bool {
        matches!(self, Self::Beam | Self::BeamSection)
    }

    fn is_shell(self) -> bool {
        matches!(self, Self::Shell | Self::ShellSection)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaBoundaryConditionDocument {
    id: String,
    region: String,
    #[serde(alias = "type")]
    kind: FeaBoundaryConditionKindDocument,
    #[serde(default)]
    rx: Option<f64>,
    #[serde(default)]
    ry: Option<f64>,
    #[serde(default)]
    rz: Option<f64>,
    #[serde(default)]
    specific_impedance_pa_s_per_m: Option<f64>,
    #[serde(default)]
    temperature_k: Option<f64>,
    #[serde(default)]
    heat_flux_w_per_m2: Option<f64>,
    #[serde(default)]
    ambient_temperature_k: Option<f64>,
    #[serde(default)]
    coefficient_w_per_m2k: Option<f64>,
    #[serde(default)]
    velocity_m_per_s: Option<f64>,
    #[serde(default)]
    pressure_pa: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum FeaBoundaryConditionKindDocument {
    Native(BoundaryConditionKind),
    Named(FeaBoundaryConditionType),
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FeaBoundaryConditionType {
    Fixed,
    PrescribedDisplacement,
    PrescribedRotation,
    MagneticInsulation,
    VectorPotentialGround,
    AcousticRigidWall,
    AcousticRadiation,
    AcousticImpedance,
    ThermalPrescribedTemperature,
    ThermalHeatFlux,
    ThermalConvection,
    CfdInletVelocity,
    CfdOutletPressure,
    CfdNoSlipWall,
    CfdSlipWall,
    CfdSymmetry,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaLoadDocument {
    id: String,
    region: String,
    #[serde(rename = "type", alias = "kind")]
    load_type: FeaLoadType,
    #[serde(default)]
    vector: Option<[f64; 3]>,
    #[serde(default)]
    force: Option<[f64; 3]>,
    #[serde(default)]
    moment: Option<[f64; 3]>,
    #[serde(default)]
    point: Option<[f64; 3]>,
    #[serde(default)]
    magnitude_pa: Option<f64>,
    #[serde(default)]
    current_a: Option<f64>,
    #[serde(default)]
    phase_rad: Option<f64>,
    #[serde(default)]
    amplitude_scale: Option<f64>,
    #[serde(default)]
    volumetric_w_per_m3: Option<f64>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FeaLoadType {
    Force,
    Moment,
    Torque,
    Wrench,
    Pressure,
    BodyForce,
    CurrentDensity,
    CoilCurrent,
    HeatSource,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaStepDocument {
    id: String,
    kind: runmat_analysis_core::AnalysisStepKind,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaOutputDocument {
    id: String,
    #[serde(rename = "field_id", alias = "field", alias = "name")]
    field_id: String,
    #[serde(default, alias = "target")]
    location: Option<String>,
    #[serde(default, rename = "kind", alias = "type")]
    kind: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaDomainsDocument {
    #[serde(default)]
    thermo_mechanical: Option<ThermoMechanicalDomain>,
    #[serde(default)]
    electro_thermal: Option<ElectroThermalDomain>,
    #[serde(default)]
    electromagnetic: Option<ElectromagneticDomain>,
    #[serde(default)]
    cfd: Option<CfdDomain>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaRunDocument {
    #[serde(default)]
    kind: Option<AnalysisRunKind>,
    #[serde(default = "default_backend")]
    backend: ComputeBackend,
    #[serde(default)]
    options: Option<serde_yaml::Value>,
}

#[derive(Debug, Clone, PartialEq)]
struct ResolvedStudyParts {
    spec: AnalysisStudySpec,
}

pub fn is_fea_file_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("fea"))
}

pub async fn load_fea_document_from_path_async(path: &Path) -> Result<FeaResolvedDocument, String> {
    if !is_fea_file_path(path) {
        return Err(format!(
            "unsupported FEA document extension: {}",
            path.display()
        ));
    }
    let input = runmat_filesystem::read_to_string_async(path)
        .await
        .map_err(|err| format!("failed to read FEA document {}: {err}", path.display()))?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new(""));
    parse_and_resolve_fea_document(&input, base_dir).await
}

pub async fn parse_and_resolve_fea_document(
    input: &str,
    base_dir: &Path,
) -> Result<FeaResolvedDocument, String> {
    let raw = serde_yaml::from_str::<RawFeaDocument>(input)
        .map_err(|err| format!("failed to parse FEA YAML: {err}"))?;
    match raw {
        RawFeaDocument::Study(study) => {
            let resolved = resolve_study(*study, base_dir).await?;
            Ok(FeaResolvedDocument::Study(Box::new(resolved.spec)))
        }
        RawFeaDocument::Sweep(sweep) => resolve_sweep(sweep, base_dir).await,
    }
}

async fn resolve_sweep(
    sweep: FeaSweepDocument,
    base_dir: &Path,
) -> Result<FeaResolvedDocument, String> {
    validate_version(sweep.version)?;
    if sweep.id.trim().is_empty() {
        return Err("sweep id must be non-empty".to_string());
    }
    let mut studies = Vec::with_capacity(sweep.studies.len());
    for study in sweep.studies {
        studies.push(resolve_study(study, base_dir).await?.spec);
    }
    Ok(FeaResolvedDocument::Sweep(AnalysisStudySweepSpec {
        sweep_id: sweep.id,
        studies,
        fail_fast: sweep.fail_fast,
    }))
}

async fn resolve_study(
    study: FeaStudyDocument,
    base_dir: &Path,
) -> Result<ResolvedStudyParts, String> {
    validate_version(study.version)?;
    if study.id.trim().is_empty() {
        return Err("study id must be non-empty".to_string());
    }

    let geometry = load_geometry(&study.geometry, base_dir).await?;
    let model_id = study
        .model
        .id
        .clone()
        .unwrap_or_else(|| format!("{}_model", sanitize_id(&study.id)));
    let intent = AnalysisCreateModelIntentSpec {
        model_id: model_id.clone(),
        profile: study.model.profile,
        prep_context: None,
    };
    let model = resolve_model(&study, &geometry, &intent)?;
    let run_kind = resolve_run_kind(study.model.profile, &study.run)?;
    let run_options = resolve_run_options(&study.run, run_kind)?;
    if let Some(mesh) = study.mesh.as_ref() {
        validate_refinement_indicator_applicability(
            &mesh.refinement.indicators,
            study.model.profile,
            run_kind,
            &study.domains,
        )?;
    }
    let mesh_options = resolve_mesh_options(study.mesh.as_ref(), study.model.profile, run_kind)?;
    let outputs = study.outputs.iter().map(resolve_output_request).collect();
    let spec = AnalysisStudySpec {
        study_id: study.id,
        geometry,
        create_model_intent: intent,
        model,
        run_kind,
        backend: study.run.backend,
        mesh_options,
        outputs,
        analysis_mesh_artifact_path: None,
        analysis_mesh_evidence_artifact_path: None,
        linear_static_run_options: run_options.linear_static,
        modal_run_options: run_options.modal,
        acoustic_run_options: run_options.acoustic,
        thermal_run_options: run_options.thermal,
        transient_run_options: run_options.transient,
        cfd_run_options: run_options.cfd,
        cht_run_options: run_options.cht,
        fsi_run_options: run_options.fsi,
        nonlinear_run_options: run_options.nonlinear,
        electromagnetic_run_options: run_options.electromagnetic,
    };
    Ok(ResolvedStudyParts { spec })
}

fn resolve_output_request(output: &FeaOutputDocument) -> AnalysisOutputRequest {
    AnalysisOutputRequest {
        id: output.id.clone(),
        field_id: output.field_id.clone(),
        location: output.location.clone(),
        kind: output.kind.clone(),
    }
}

fn resolve_mesh_options(
    mesh: Option<&FeaMeshDocument>,
    profile: AnalysisCreateModelProfile,
    run_kind: AnalysisRunKind,
) -> Result<Option<VolumeMeshingOptions>, String> {
    let Some(mesh) = mesh else {
        return Ok(default_mesh_options_for_study(profile, run_kind));
    };
    if mesh.max_elements == 0 {
        return Err("mesh.max_elements must be greater than zero".to_string());
    }
    if !matches!(mesh.kind, MeshKindRequest::Solid) {
        return Err(format!(
            "mesh.kind {:?} is not supported by the current analysis mesher; use mesh.kind: solid",
            mesh.kind
        ));
    }
    if !mesh.element.is_supported_for_solid_solve() {
        return Err(format!(
            "mesh.element {:?} is not supported for mesh.kind solid; use mesh.element: tetrahedron4",
            mesh.element
        ));
    }
    if let Some(min_size) = mesh.min_size {
        if !min_size.is_finite() || min_size <= 0.0 {
            return Err("mesh.min_size must be finite and positive".to_string());
        }
    }
    if let Some(max_size) = mesh.max_size {
        if !max_size.is_finite() || max_size <= 0.0 {
            return Err("mesh.max_size must be finite and positive".to_string());
        }
    }
    if let (Some(min_size), Some(max_size)) = (mesh.min_size, mesh.max_size) {
        if min_size > max_size {
            return Err("mesh.min_size must be less than or equal to mesh.max_size".to_string());
        }
    }
    if let Some(growth_rate) = mesh.growth_rate {
        if !growth_rate.is_finite() || growth_rate < 1.0 {
            return Err("mesh.growth_rate must be finite and at least 1.0".to_string());
        }
    }
    if !mesh
        .refinement
        .convergence
        .field_change_tolerance
        .is_finite()
        || mesh.refinement.convergence.field_change_tolerance <= 0.0
    {
        return Err(
            "mesh.refinement.convergence.field_change_tolerance must be finite and positive"
                .to_string(),
        );
    }
    if !mesh
        .refinement
        .convergence
        .energy_change_tolerance
        .is_finite()
        || mesh.refinement.convergence.energy_change_tolerance <= 0.0
    {
        return Err(
            "mesh.refinement.convergence.energy_change_tolerance must be finite and positive"
                .to_string(),
        );
    }
    if let Some(residual_tolerance) = mesh.refinement.convergence.residual_tolerance {
        if !residual_tolerance.is_finite() || residual_tolerance <= 0.0 {
            return Err(
                "mesh.refinement.convergence.residual_tolerance must be finite and positive"
                    .to_string(),
            );
        }
    }
    let validation = resolve_mesh_validation_policy(&mesh.validation)?;
    if matches!(mesh.refinement.strategy, RefinementStrategy::None)
        && mesh.refinement.max_iterations > 0
    {
        return Err(
            "mesh.refinement.max_iterations must be 0 when mesh.refinement.strategy is none"
                .to_string(),
        );
    }
    validate_refinement_indicators(&mesh.refinement.indicators)?;

    Ok(Some(VolumeMeshingOptions {
        backend: mesh.backend,
        kind: mesh.kind,
        element: mesh.element,
        element_order: mesh.element_order,
        profile: mesh.profile,
        max_elements: mesh.max_elements,
        target_size: mesh.target_size.unwrap_or(MeshTargetSize::Auto),
        min_size_m: mesh.min_size,
        max_size_m: mesh.max_size,
        growth_rate: mesh.growth_rate,
        refinement: MeshRefinementOptions {
            strategy: mesh.refinement.strategy,
            max_iterations: mesh.refinement.max_iterations,
            convergence: RefinementConvergenceOptions {
                field_change_tolerance: mesh.refinement.convergence.field_change_tolerance,
                energy_change_tolerance: mesh.refinement.convergence.energy_change_tolerance,
                residual_tolerance: mesh.refinement.convergence.residual_tolerance,
            },
            focus: RefinementFocusOptions {
                loads: mesh.refinement.focus.loads,
                constraints: mesh.refinement.focus.constraints,
                interfaces: mesh.refinement.focus.interfaces,
                curvature: mesh.refinement.focus.curvature,
                small_features: mesh.refinement.focus.small_features,
            },
            indicators: RefinementIndicatorOverrides {
                namespaces: mesh.refinement.indicators.clone(),
            },
        },
        validation,
    }))
}

fn resolve_mesh_validation_policy(
    validation: &FeaMeshValidationDocument,
) -> Result<MeshValidationPolicyOptions, String> {
    let mut policy = coverage_preset_options(validation.coverage);
    policy.quality = quality_preset_thresholds(validation.quality);
    apply_optional_unit_ratio(
        "mesh.validation.min_bounds_coverage_ratio",
        validation.min_bounds_coverage_ratio,
        &mut policy.min_bounds_coverage_ratio,
    )?;
    apply_optional_unit_ratio(
        "mesh.validation.min_volume_coverage_ratio",
        validation.min_volume_coverage_ratio,
        &mut policy.min_volume_coverage_ratio,
    )?;
    apply_optional_unit_ratio(
        "mesh.validation.min_boundary_area_ratio",
        validation.min_boundary_area_ratio,
        &mut policy.min_boundary_area_ratio,
    )?;
    apply_optional_unit_ratio(
        "mesh.validation.min_boundary_face_recovery_ratio",
        validation.min_boundary_face_recovery_ratio,
        &mut policy.min_boundary_face_recovery_ratio,
    )?;
    apply_optional_unit_ratio(
        "mesh.validation.min_boundary_edge_recovery_ratio",
        validation.min_boundary_edge_recovery_ratio,
        &mut policy.min_boundary_edge_recovery_ratio,
    )?;
    if matches!(validation.max_volume_components, Some(0)) {
        return Err("mesh.validation.max_volume_components must be greater than zero".to_string());
    }
    policy.max_volume_component_count = validation.max_volume_components;
    Ok(policy)
}

fn coverage_preset_options(preset: FeaMeshCoveragePreset) -> MeshValidationPolicyOptions {
    match preset {
        FeaMeshCoveragePreset::Strict => MeshValidationPolicyOptions {
            min_bounds_coverage_ratio: 1.0,
            min_volume_coverage_ratio: 1.0,
            min_boundary_area_ratio: 1.0,
            min_boundary_face_recovery_ratio: 1.0,
            min_boundary_edge_recovery_ratio: 1.0,
            ..MeshValidationPolicyOptions::default()
        },
        FeaMeshCoveragePreset::Balanced => MeshValidationPolicyOptions::default(),
        FeaMeshCoveragePreset::Relaxed => MeshValidationPolicyOptions {
            min_bounds_coverage_ratio: 0.80,
            min_volume_coverage_ratio: 0.80,
            min_boundary_area_ratio: 0.80,
            min_boundary_face_recovery_ratio: 0.95,
            min_boundary_edge_recovery_ratio: 0.95,
            ..MeshValidationPolicyOptions::default()
        },
    }
}

fn quality_preset_thresholds(preset: FeaMeshQualityPreset) -> QualityThresholds {
    match preset {
        FeaMeshQualityPreset::Strict => QualityThresholds {
            min_scaled_jacobian: 0.25,
            max_aspect_ratio: 10.0,
            max_boundary_projection_error_m: 1.0e-8,
            allow_inverted_elements: false,
        },
        FeaMeshQualityPreset::Balanced => QualityThresholds::default(),
        FeaMeshQualityPreset::Relaxed => QualityThresholds {
            min_scaled_jacobian: 0.05,
            max_aspect_ratio: 50.0,
            max_boundary_projection_error_m: 1.0e-4,
            allow_inverted_elements: false,
        },
    }
}

fn apply_optional_unit_ratio(
    label: &str,
    value: Option<f64>,
    target: &mut f64,
) -> Result<(), String> {
    if let Some(value) = value {
        validate_unit_ratio(label, value)?;
        *target = value;
    }
    Ok(())
}

fn validate_unit_ratio(label: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err(format!("{label} must be finite and in the range (0, 1]"));
    }
    Ok(())
}

fn default_mesh_options_for_study(
    profile: AnalysisCreateModelProfile,
    run_kind: AnalysisRunKind,
) -> Option<VolumeMeshingOptions> {
    (matches!(profile, AnalysisCreateModelProfile::LinearStaticStructural)
        && matches!(run_kind, AnalysisRunKind::LinearStatic))
    .then(VolumeMeshingOptions::default)
}

fn validate_refinement_indicators(
    indicators: &BTreeMap<String, BTreeMap<String, RefinementIndicatorMode>>,
) -> Result<(), String> {
    for (namespace, names) in indicators {
        let allowed = allowed_refinement_indicator_names(namespace)
            .ok_or_else(|| format!("unknown mesh.refinement.indicators namespace `{namespace}`"))?;
        for name in names.keys() {
            if !allowed.contains(&name.as_str()) {
                return Err(format!(
                    "unknown mesh.refinement.indicators.{namespace}.{name}"
                ));
            }
        }
    }
    Ok(())
}

fn validate_refinement_indicator_applicability(
    indicators: &BTreeMap<String, BTreeMap<String, RefinementIndicatorMode>>,
    profile: AnalysisCreateModelProfile,
    run_kind: AnalysisRunKind,
    domains: &FeaDomainsDocument,
) -> Result<(), String> {
    for namespace in indicators.keys() {
        if !refinement_indicator_namespace_applies(namespace, profile, run_kind, domains) {
            return Err(format!(
                "mesh.refinement.indicators.{namespace} does not apply to profile `{}` and run kind `{}`",
                profile.as_snake_case(),
                run_kind.as_snake_case()
            ));
        }
    }
    Ok(())
}

fn refinement_indicator_namespace_applies(
    namespace: &str,
    profile: AnalysisCreateModelProfile,
    run_kind: AnalysisRunKind,
    domains: &FeaDomainsDocument,
) -> bool {
    match namespace {
        "structural" => {
            matches!(
                profile,
                AnalysisCreateModelProfile::LinearStaticStructural
                    | AnalysisCreateModelProfile::TransientStructural
                    | AnalysisCreateModelProfile::NonlinearStructural
                    | AnalysisCreateModelProfile::ThermoMechanicalCoupled
                    | AnalysisCreateModelProfile::FsiCoupled
            ) || matches!(
                run_kind,
                AnalysisRunKind::LinearStatic
                    | AnalysisRunKind::Transient
                    | AnalysisRunKind::Nonlinear
                    | AnalysisRunKind::Fsi
            )
        }
        "modal" => {
            matches!(profile, AnalysisCreateModelProfile::ModalStructural)
                || matches!(run_kind, AnalysisRunKind::Modal)
        }
        "thermal" => {
            matches!(
                profile,
                AnalysisCreateModelProfile::ThermalStandalone
                    | AnalysisCreateModelProfile::ThermoMechanicalCoupled
                    | AnalysisCreateModelProfile::ElectroThermalCoupled
                    | AnalysisCreateModelProfile::ChtCoupled
            ) || matches!(run_kind, AnalysisRunKind::Thermal | AnalysisRunKind::Cht)
                || domains.thermo_mechanical.is_some()
                || domains.electro_thermal.is_some()
        }
        "thermo_mechanical" => {
            matches!(profile, AnalysisCreateModelProfile::ThermoMechanicalCoupled)
                || domains.thermo_mechanical.is_some()
        }
        "electro_thermal" => {
            matches!(profile, AnalysisCreateModelProfile::ElectroThermalCoupled)
                || domains.electro_thermal.is_some()
        }
        "electromagnetic" => {
            matches!(
                profile,
                AnalysisCreateModelProfile::ElectromagneticStatic
                    | AnalysisCreateModelProfile::ElectroThermalCoupled
            ) || matches!(run_kind, AnalysisRunKind::Electromagnetic)
                || domains.electromagnetic.is_some()
                || domains.electro_thermal.is_some()
        }
        "acoustic" => {
            matches!(profile, AnalysisCreateModelProfile::AcousticHarmonic)
                || matches!(run_kind, AnalysisRunKind::Acoustic)
        }
        "cfd" => {
            matches!(
                profile,
                AnalysisCreateModelProfile::CfdSteadyState
                    | AnalysisCreateModelProfile::CfdTransient
                    | AnalysisCreateModelProfile::ChtCoupled
                    | AnalysisCreateModelProfile::FsiCoupled
            ) || matches!(
                run_kind,
                AnalysisRunKind::Cfd | AnalysisRunKind::Cht | AnalysisRunKind::Fsi
            ) || domains.cfd.is_some()
        }
        "cht" => {
            matches!(profile, AnalysisCreateModelProfile::ChtCoupled)
                || matches!(run_kind, AnalysisRunKind::Cht)
        }
        "fsi" => {
            matches!(profile, AnalysisCreateModelProfile::FsiCoupled)
                || matches!(run_kind, AnalysisRunKind::Fsi)
        }
        _ => false,
    }
}

fn allowed_refinement_indicator_names(namespace: &str) -> Option<&'static [&'static str]> {
    match namespace {
        "structural" => Some(&[
            "stress_gradient",
            "strain_energy_density",
            "displacement_gradient",
            "load_regions",
            "constraint_regions",
            "plastic_strain",
            "contact_pressure",
            "contact_gap",
        ]),
        "modal" => Some(&[
            "mode_shape_curvature",
            "modal_strain_energy",
            "frequency_residual",
        ]),
        "thermal" => Some(&[
            "temperature_gradient",
            "heat_flux_gradient",
            "heat_source",
            "convection_regions",
            "prescribed_temperature_regions",
        ]),
        "thermo_mechanical" => Some(&[
            "thermal_gradient",
            "thermal_stress",
            "structural_von_mises",
            "strain_energy_density",
            "region_temperature_delta",
        ]),
        "electro_thermal" => Some(&[
            "electric_field_gradient",
            "current_density_gradient",
            "joule_heat_density",
            "thermal_gradient",
            "source_regions",
            "ground_regions",
        ]),
        "electromagnetic" => Some(&[
            "flux_density_gradient",
            "electric_field_gradient",
            "current_density_gradient",
            "energy_density",
            "source_regions",
            "ground_regions",
            "insulation_regions",
        ]),
        "acoustic" => Some(&[
            "pressure_gradient",
            "pressure_curvature",
            "wavelength",
            "impedance_regions",
            "source_regions",
        ]),
        "cfd" => Some(&[
            "velocity_gradient",
            "pressure_gradient",
            "vorticity",
            "wall_shear",
            "boundary_layer",
            "inlet_regions",
            "outlet_regions",
        ]),
        "cht" => Some(&[
            "interface_heat_flux_jump",
            "interface_temperature_jump",
            "solid_heat_flux_gradient",
            "fluid_boundary_layer",
        ]),
        "fsi" => Some(&[
            "interface_displacement_jump",
            "interface_traction_jump",
            "structural_stress_gradient",
            "fluid_pressure_gradient",
            "fluid_velocity_gradient",
        ]),
        _ => None,
    }
}

async fn load_geometry(
    geometry: &FeaGeometryDocument,
    base_dir: &Path,
) -> Result<GeometryAsset, String> {
    let path = resolve_document_path(base_dir, &geometry.path);
    let bytes = runmat_filesystem::read_async(&path)
        .await
        .map_err(|err| format!("failed to read geometry file {}: {err}", path.display()))?;
    let options = GeometryImportOptions {
        max_triangles: geometry.import.max_triangles.or(Some(16_000_000)),
        budget_policy: runmat_geometry_io::GeometryImportBudgetPolicy::Strict,
        units: geometry.units,
        tessellation_profile: Default::default(),
        relative_deflection: false,
    };
    crate::geometry::geometry_load_with_options_op(
        &path.to_string_lossy(),
        &bytes,
        options,
        OperationContext::new(None, None),
    )
    .map(|envelope| envelope.data)
    .map_err(|err| {
        format!(
            "failed to load geometry {}: {}",
            path.display(),
            err.message
        )
    })
}

fn resolve_model(
    study: &FeaStudyDocument,
    geometry: &GeometryAsset,
    intent: &AnalysisCreateModelIntentSpec,
) -> Result<Option<AnalysisModel>, String> {
    if !has_explicit_model_data(study)
        && study.model.defaults == FeaModelDefaultsMode::ProfileScaffold
    {
        return Ok(None);
    }

    let mut model = match study.model.defaults {
        FeaModelDefaultsMode::ProfileScaffold => {
            analysis_create_model_op(geometry, intent.clone(), OperationContext::new(None, None))
                .map(|envelope| envelope.data)
                .map_err(|err| format!("failed to create FEA model scaffold: {}", err.message))?
        }
        FeaModelDefaultsMode::None => empty_model(intent.model_id.clone(), geometry),
    };

    if let Some(frame) = &study.model.frame {
        model.frame = frame.clone();
    }
    if !study.materials.is_empty() {
        model.materials = study
            .materials
            .iter()
            .map(|(id, material)| resolve_material(id, material))
            .collect();
    }
    if !study.material_assignments.is_empty() {
        model.material_assignments = study
            .material_assignments
            .iter()
            .map(|assignment| resolve_material_assignment(assignment, geometry, &study.regions))
            .collect::<Result<Vec<_>, _>>()?;
    }
    if has_structural_model_data(study) {
        model.structural = Some(resolve_structural_model(study, geometry)?);
    }
    if !study.boundary_conditions.is_empty() {
        model.boundary_conditions = study
            .boundary_conditions
            .iter()
            .map(|bc| resolve_boundary_condition(bc, geometry, &study.regions))
            .collect::<Result<Vec<_>, _>>()?;
    }
    if !study.loads.is_empty() {
        model.loads = study
            .loads
            .iter()
            .map(|load| resolve_load(load, geometry, &study.regions))
            .collect::<Result<Vec<_>, _>>()?;
    }
    if !study.steps.is_empty() {
        model.steps = study
            .steps
            .iter()
            .map(|step| AnalysisStep {
                step_id: step.id.clone(),
                kind: step.kind.clone(),
            })
            .collect();
    }
    if study.domains.thermo_mechanical.is_some() {
        model.thermo_mechanical = study.domains.thermo_mechanical.clone();
    }
    if study.domains.electro_thermal.is_some() {
        model.electro_thermal = study.domains.electro_thermal.clone();
    }
    if study.domains.electromagnetic.is_some() {
        model.electromagnetic = study.domains.electromagnetic.clone();
    }
    if study.domains.cfd.is_some() {
        model.cfd = study.domains.cfd.clone();
    }
    if !study.interfaces.is_empty() {
        model.interfaces = study.interfaces.clone();
    }

    Ok(Some(model))
}

fn resolve_material(id: &str, material: &FeaMaterialDocument) -> MaterialModel {
    MaterialModel {
        material_id: id.to_string(),
        name: material.name.clone().unwrap_or_else(|| id.to_string()),
        mechanical: material.mechanical.clone(),
        thermal: material.thermal.clone().unwrap_or_default(),
        acoustic: material.acoustic.clone(),
        electrical: material.electrical.clone(),
        plastic: material.plastic.clone(),
    }
}

fn resolve_material_assignment(
    assignment: &FeaMaterialAssignmentDocument,
    geometry: &GeometryAsset,
    aliases: &BTreeMap<String, FeaRegionDocument>,
) -> Result<MaterialAssignment, String> {
    let region_id = resolve_region_ref(&assignment.region, geometry, aliases)?;
    Ok(MaterialAssignment {
        region_id,
        expected_material_id: assignment
            .expected_material
            .clone()
            .unwrap_or_else(|| assignment.material.clone()),
        assigned_material_id: assignment.material.clone(),
        confidence: assignment.confidence,
    })
}

fn resolve_structural_model(
    study: &FeaStudyDocument,
    geometry: &GeometryAsset,
) -> Result<StructuralModel, String> {
    let structural = study.structural.as_ref();
    let node_docs = structural
        .map(|value| value.nodes.as_slice())
        .unwrap_or(study.nodes.as_slice());
    let element_docs = structural
        .map(|value| value.elements.as_slice())
        .unwrap_or(study.elements.as_slice());
    let section_docs = structural
        .map(|value| value.sections.as_slice())
        .unwrap_or(study.sections.as_slice());

    Ok(StructuralModel {
        nodes: node_docs
            .iter()
            .map(|node| StructuralNode {
                node_id: node.id,
                coordinates_m: node.coordinates_m,
            })
            .collect(),
        elements: element_docs
            .iter()
            .map(|element| resolve_structural_element(element, geometry, &study.regions))
            .collect::<Result<Vec<_>, _>>()?,
        beam_sections: section_docs
            .iter()
            .filter(|section| section.section_type.is_beam())
            .map(resolve_beam_section)
            .collect::<Result<Vec<_>, _>>()?,
        shell_sections: section_docs
            .iter()
            .filter(|section| section.section_type.is_shell())
            .map(resolve_shell_section)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn resolve_structural_element(
    element: &FeaStructuralElementDocument,
    geometry: &GeometryAsset,
    aliases: &BTreeMap<String, FeaRegionDocument>,
) -> Result<StructuralElement, String> {
    let kind = match element.element_type {
        FeaStructuralElementType::Beam => {
            let node_ids: [u32; 2] = element.nodes.as_slice().try_into().map_err(|_| {
                format!(
                    "beam element {} must specify exactly two node ids",
                    element.id
                )
            })?;
            StructuralElementKind::Beam(BeamElementModel {
                node_ids,
                section_id: element.section.clone(),
                reference_axis: element.reference_axis.unwrap_or([0.0, 0.0, 1.0]),
            })
        }
        FeaStructuralElementType::Shell => {
            let node_ids: [u32; 3] = element.nodes.as_slice().try_into().map_err(|_| {
                format!(
                    "shell element {} must specify exactly three node ids",
                    element.id
                )
            })?;
            StructuralElementKind::Shell(ShellElementModel {
                node_ids,
                section_id: element.section.clone(),
                reference_axis: element.reference_axis.unwrap_or([1.0, 0.0, 0.0]),
            })
        }
    };
    Ok(StructuralElement {
        element_id: element.id.clone(),
        region_id: resolve_region_ref(&element.region, geometry, aliases)?,
        kind,
    })
}

fn resolve_beam_section(
    section: &FeaStructuralSectionDocument,
) -> Result<BeamSectionModel, String> {
    Ok(BeamSectionModel {
        section_id: section.id.clone(),
        area_m2: required_f64(section.area_m2, "beam_section.area_m2")?,
        iy_m4: required_f64(section.iy_m4, "beam_section.iy_m4")?,
        iz_m4: required_f64(section.iz_m4, "beam_section.iz_m4")?,
        torsion_j_m4: required_f64(section.torsion_j_m4, "beam_section.torsion_j_m4")?,
        outer_fiber_y_m: section.outer_fiber_y_m,
        outer_fiber_z_m: section.outer_fiber_z_m,
        torsion_outer_radius_m: section.torsion_outer_radius_m,
    })
}

fn resolve_shell_section(
    section: &FeaStructuralSectionDocument,
) -> Result<ShellSectionModel, String> {
    Ok(ShellSectionModel {
        section_id: section.id.clone(),
        thickness_m: required_f64(section.thickness_m, "shell_section.thickness_m")?,
        shear_correction: section.shear_correction.unwrap_or(5.0 / 6.0),
        drilling_stiffness_scale: section.drilling_stiffness_scale.unwrap_or(1.0e-4),
    })
}

fn resolve_boundary_condition(
    bc: &FeaBoundaryConditionDocument,
    geometry: &GeometryAsset,
    aliases: &BTreeMap<String, FeaRegionDocument>,
) -> Result<BoundaryCondition, String> {
    let kind = match &bc.kind {
        FeaBoundaryConditionKindDocument::Native(kind) => kind.clone(),
        FeaBoundaryConditionKindDocument::Named(kind) => {
            resolve_boundary_condition_kind(bc, *kind)?
        }
    };
    Ok(BoundaryCondition {
        bc_id: bc.id.clone(),
        region_id: resolve_region_ref(&bc.region, geometry, aliases)?,
        kind,
    })
}

fn resolve_boundary_condition_kind(
    bc: &FeaBoundaryConditionDocument,
    kind: FeaBoundaryConditionType,
) -> Result<BoundaryConditionKind, String> {
    Ok(match kind {
        FeaBoundaryConditionType::Fixed => BoundaryConditionKind::Fixed,
        FeaBoundaryConditionType::PrescribedDisplacement => {
            BoundaryConditionKind::PrescribedDisplacement
        }
        FeaBoundaryConditionType::PrescribedRotation => BoundaryConditionKind::PrescribedRotation {
            rx: required_f64(bc.rx, "boundary.prescribed_rotation.rx")?,
            ry: required_f64(bc.ry, "boundary.prescribed_rotation.ry")?,
            rz: required_f64(bc.rz, "boundary.prescribed_rotation.rz")?,
        },
        FeaBoundaryConditionType::MagneticInsulation => BoundaryConditionKind::MagneticInsulation,
        FeaBoundaryConditionType::VectorPotentialGround => {
            BoundaryConditionKind::VectorPotentialGround
        }
        FeaBoundaryConditionType::AcousticRigidWall => BoundaryConditionKind::AcousticRigidWall,
        FeaBoundaryConditionType::AcousticRadiation => BoundaryConditionKind::AcousticRadiation,
        FeaBoundaryConditionType::AcousticImpedance => BoundaryConditionKind::AcousticImpedance {
            specific_impedance_pa_s_per_m: required_f64(
                bc.specific_impedance_pa_s_per_m,
                "boundary.acoustic_impedance.specific_impedance_pa_s_per_m",
            )?,
        },
        FeaBoundaryConditionType::ThermalPrescribedTemperature => {
            BoundaryConditionKind::ThermalPrescribedTemperature {
                temperature_k: required_f64(
                    bc.temperature_k,
                    "boundary.thermal_prescribed_temperature.temperature_k",
                )?,
            }
        }
        FeaBoundaryConditionType::ThermalHeatFlux => BoundaryConditionKind::ThermalHeatFlux {
            heat_flux_w_per_m2: required_f64(
                bc.heat_flux_w_per_m2,
                "boundary.thermal_heat_flux.heat_flux_w_per_m2",
            )?,
        },
        FeaBoundaryConditionType::ThermalConvection => BoundaryConditionKind::ThermalConvection {
            ambient_temperature_k: required_f64(
                bc.ambient_temperature_k,
                "boundary.thermal_convection.ambient_temperature_k",
            )?,
            coefficient_w_per_m2k: required_f64(
                bc.coefficient_w_per_m2k,
                "boundary.thermal_convection.coefficient_w_per_m2k",
            )?,
        },
        FeaBoundaryConditionType::CfdInletVelocity => BoundaryConditionKind::CfdInletVelocity {
            velocity_m_per_s: required_f64(
                bc.velocity_m_per_s,
                "boundary.cfd_inlet_velocity.velocity_m_per_s",
            )?,
        },
        FeaBoundaryConditionType::CfdOutletPressure => BoundaryConditionKind::CfdOutletPressure {
            pressure_pa: required_f64(bc.pressure_pa, "boundary.cfd_outlet_pressure.pressure_pa")?,
        },
        FeaBoundaryConditionType::CfdNoSlipWall => BoundaryConditionKind::CfdNoSlipWall,
        FeaBoundaryConditionType::CfdSlipWall => BoundaryConditionKind::CfdSlipWall,
        FeaBoundaryConditionType::CfdSymmetry => BoundaryConditionKind::CfdSymmetry,
    })
}

fn resolve_load(
    load: &FeaLoadDocument,
    geometry: &GeometryAsset,
    aliases: &BTreeMap<String, FeaRegionDocument>,
) -> Result<LoadCase, String> {
    let kind = match load.load_type {
        FeaLoadType::Force => {
            let [fx, fy, fz] = load_vector(load, "force")?;
            LoadKind::Force { fx, fy, fz }
        }
        FeaLoadType::Moment | FeaLoadType::Torque => {
            let [mx, my, mz] = load_vector(load, "moment")?;
            LoadKind::Moment { mx, my, mz }
        }
        FeaLoadType::Wrench => {
            let [fx, fy, fz] = load_force(load)?;
            let [mx, my, mz] = load_moment(load)?;
            let [px, py, pz] = point_in_meters(load_point(load)?, geometry.units);
            LoadKind::Wrench {
                fx,
                fy,
                fz,
                mx,
                my,
                mz,
                px,
                py,
                pz,
            }
        }
        FeaLoadType::Pressure => LoadKind::Pressure {
            magnitude_pa: required_f64(load.magnitude_pa, "pressure.magnitude_pa")?,
        },
        FeaLoadType::BodyForce => {
            let [gx, gy, gz] = load_vector(load, "body_force")?;
            LoadKind::BodyForce { gx, gy, gz }
        }
        FeaLoadType::CurrentDensity => {
            let [jx, jy, jz] = load_vector(load, "current_density")?;
            LoadKind::CurrentDensity {
                jx,
                jy,
                jz,
                phase_rad: load.phase_rad.unwrap_or_default(),
                amplitude_scale: load.amplitude_scale.unwrap_or(1.0),
            }
        }
        FeaLoadType::CoilCurrent => LoadKind::CoilCurrent {
            current_a: required_f64(load.current_a, "coil_current.current_a")?,
            phase_rad: load.phase_rad.unwrap_or_default(),
            amplitude_scale: load.amplitude_scale.unwrap_or(1.0),
        },
        FeaLoadType::HeatSource => LoadKind::HeatSource {
            volumetric_w_per_m3: required_f64(
                load.volumetric_w_per_m3,
                "heat_source.volumetric_w_per_m3",
            )?,
        },
    };
    Ok(LoadCase {
        load_id: load.id.clone(),
        region_id: resolve_region_ref(&load.region, geometry, aliases)?,
        kind,
    })
}

#[derive(Debug, Default)]
struct ResolvedRunOptions {
    linear_static: Option<AnalysisRunOptions>,
    modal: Option<AnalysisModalRunOptions>,
    acoustic: Option<AnalysisAcousticRunOptions>,
    thermal: Option<AnalysisThermalRunOptions>,
    transient: Option<AnalysisTransientRunOptions>,
    cfd: Option<AnalysisCfdRunOptions>,
    cht: Option<AnalysisChtRunOptions>,
    fsi: Option<AnalysisFsiRunOptions>,
    nonlinear: Option<AnalysisNonlinearRunOptions>,
    electromagnetic: Option<AnalysisElectromagneticRunOptions>,
}

fn resolve_run_kind(
    profile: AnalysisCreateModelProfile,
    run: &FeaRunDocument,
) -> Result<AnalysisRunKind, String> {
    let derived = profile.derived_run_kind();
    if let Some(explicit) = run.kind {
        if explicit != derived {
            return Err(format!(
                "run.kind {:?} does not match the solver selected by model.profile {:?}; omit run.kind unless you need an advanced matching solver override",
                explicit, profile
            ));
        }
    }
    Ok(derived)
}

fn resolve_run_options(
    run: &FeaRunDocument,
    run_kind: AnalysisRunKind,
) -> Result<ResolvedRunOptions, String> {
    let Some(options) = run.options.clone() else {
        return Ok(ResolvedRunOptions::default());
    };
    let mut resolved = ResolvedRunOptions::default();
    match run_kind {
        AnalysisRunKind::LinearStatic => {
            resolved.linear_static = Some(parse_options(options, "linear_static options")?);
        }
        AnalysisRunKind::Modal => {
            resolved.modal = Some(parse_options(options, "modal options")?);
        }
        AnalysisRunKind::Acoustic => {
            resolved.acoustic = Some(parse_options(options, "acoustic options")?);
        }
        AnalysisRunKind::Thermal => {
            resolved.thermal = Some(parse_options(options, "thermal options")?);
        }
        AnalysisRunKind::Transient => {
            resolved.transient = Some(parse_options(options, "transient options")?);
        }
        AnalysisRunKind::Cfd => {
            resolved.cfd = Some(parse_options(options, "cfd options")?);
        }
        AnalysisRunKind::Cht => {
            resolved.cht = Some(parse_options(options, "cht options")?);
        }
        AnalysisRunKind::Fsi => {
            resolved.fsi = Some(parse_options(options, "fsi options")?);
        }
        AnalysisRunKind::Nonlinear => {
            resolved.nonlinear = Some(parse_options(options, "nonlinear options")?);
        }
        AnalysisRunKind::Electromagnetic => {
            resolved.electromagnetic = Some(parse_options(options, "electromagnetic options")?);
        }
    }
    Ok(resolved)
}

fn parse_options<T: DeserializeOwned>(
    options: serde_yaml::Value,
    label: &str,
) -> Result<T, String> {
    serde_yaml::from_value(options).map_err(|err| format!("invalid {label}: {err}"))
}

fn empty_model(model_id: String, geometry: &GeometryAsset) -> AnalysisModel {
    AnalysisModel {
        model_id: AnalysisModelId(model_id),
        geometry_id: geometry.geometry_id.clone(),
        geometry_revision: geometry.revision,
        units: geometry.units,
        frame: ReferenceFrame::Global,
        materials: Vec::new(),
        material_assignments: Vec::new(),
        structural: None,
        thermo_mechanical: None,
        electro_thermal: None,
        electromagnetic: None,
        cfd: None,
        interfaces: Vec::new(),
        boundary_conditions: Vec::new(),
        loads: Vec::new(),
        steps: Vec::new(),
    }
}

fn has_explicit_model_data(study: &FeaStudyDocument) -> bool {
    !study.materials.is_empty()
        || !study.material_assignments.is_empty()
        || !study.boundary_conditions.is_empty()
        || !study.loads.is_empty()
        || !study.steps.is_empty()
        || has_structural_model_data(study)
        || study.domains.thermo_mechanical.is_some()
        || study.domains.electro_thermal.is_some()
        || study.domains.electromagnetic.is_some()
        || study.domains.cfd.is_some()
        || !study.interfaces.is_empty()
        || study.model.frame.is_some()
}

fn has_structural_model_data(study: &FeaStudyDocument) -> bool {
    study.structural.as_ref().is_some_and(|structural| {
        !structural.nodes.is_empty()
            || !structural.elements.is_empty()
            || !structural.sections.is_empty()
    }) || !study.nodes.is_empty()
        || !study.elements.is_empty()
        || !study.sections.is_empty()
}

fn resolve_region_ref(
    reference: &str,
    geometry: &GeometryAsset,
    aliases: &BTreeMap<String, FeaRegionDocument>,
) -> Result<String, String> {
    if reference.strip_prefix("node:").is_some() || reference.parse::<u32>().is_ok() {
        return Ok(reference.to_string());
    }
    if let Some(alias) = aliases.get(reference) {
        return resolve_region_selector(&alias.selector, geometry);
    }
    resolve_region_selector(reference, geometry)
}

fn resolve_region_selector(selector: &str, geometry: &GeometryAsset) -> Result<String, String> {
    if let Some(id) = selector
        .strip_prefix("id:")
        .or_else(|| selector.strip_prefix("region:"))
    {
        return require_region_id(id, geometry);
    }
    if let Some(tag) = selector.strip_prefix("tag:") {
        return geometry
            .regions
            .iter()
            .find(|region| region.tag.as_deref() == Some(tag))
            .map(|region| region.region_id.clone())
            .ok_or_else(|| format!("region tag `{tag}` was not found in geometry"));
    }
    if let Some(name) = selector.strip_prefix("name:") {
        return geometry
            .regions
            .iter()
            .find(|region| region.name == name)
            .map(|region| region.region_id.clone())
            .ok_or_else(|| format!("region name `{name}` was not found in geometry"));
    }
    require_region_id(selector, geometry)
}

fn require_region_id(region_id: &str, geometry: &GeometryAsset) -> Result<String, String> {
    geometry
        .regions
        .iter()
        .find(|region| region.region_id == region_id)
        .map(|region| region.region_id.clone())
        .ok_or_else(|| format!("region id `{region_id}` was not found in geometry"))
}

fn load_vector(load: &FeaLoadDocument, label: &str) -> Result<[f64; 3], String> {
    load.vector
        .ok_or_else(|| format!("{label} load requires vector: [x, y, z]"))
}

fn load_force(load: &FeaLoadDocument) -> Result<[f64; 3], String> {
    load.force
        .ok_or_else(|| "wrench load requires force: [fx, fy, fz]".to_string())
}

fn load_moment(load: &FeaLoadDocument) -> Result<[f64; 3], String> {
    load.moment
        .ok_or_else(|| "wrench load requires moment: [mx, my, mz]".to_string())
}

fn load_point(load: &FeaLoadDocument) -> Result<[f64; 3], String> {
    load.point
        .ok_or_else(|| "wrench load requires point: [px, py, pz]".to_string())
}

fn point_in_meters(point: [f64; 3], units: UnitSystem) -> [f64; 3] {
    let scale = geometry_unit_scale_to_meters(units);
    [point[0] * scale, point[1] * scale, point[2] * scale]
}

fn geometry_unit_scale_to_meters(units: UnitSystem) -> f64 {
    match units {
        UnitSystem::Meter | UnitSystem::Unspecified => 1.0,
        UnitSystem::Millimeter => 1.0e-3,
        UnitSystem::Inch => 0.0254,
    }
}

fn required_f64(value: Option<f64>, label: &str) -> Result<f64, String> {
    value.ok_or_else(|| format!("{label} is required"))
}

fn resolve_document_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn validate_version(version: u32) -> Result<(), String> {
    if version == FEA_DOCUMENT_VERSION {
        Ok(())
    } else {
        Err(format!(
            "unsupported FEA document version {version}; expected {FEA_DOCUMENT_VERSION}"
        ))
    }
}

fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn default_units() -> UnitSystem {
    UnitSystem::Meter
}

fn default_backend() -> ComputeBackend {
    ComputeBackend::Cpu
}

fn default_mesh_kind() -> MeshKindRequest {
    MeshKindRequest::Solid
}

fn default_mesh_element() -> VolumeElementKind {
    VolumeElementKind::Tetrahedron4
}

fn default_mesh_element_order() -> MeshElementOrder {
    MeshElementOrder::Linear
}

fn default_mesh_profile() -> MeshProfile {
    MeshProfile::AnalysisReady
}

fn default_mesh_max_elements() -> usize {
    250_000
}

fn default_refinement_strategy() -> RefinementStrategy {
    RefinementStrategy::Auto
}

fn default_refinement_max_iterations() -> usize {
    4
}

fn default_field_change_tolerance() -> f64 {
    0.05
}

fn default_energy_change_tolerance() -> f64 {
    0.02
}

fn default_focus_fine() -> RefinementFocusLevel {
    RefinementFocusLevel::Fine
}

fn default_focus_normal() -> RefinementFocusLevel {
    RefinementFocusLevel::Normal
}

fn default_true() -> bool {
    true
}

fn default_fail_fast() -> bool {
    true
}

fn default_assignment_confidence() -> EvidenceConfidence {
    EvidenceConfidence::Verified
}

fn deserialize_optional_mesh_target_size<'de, D>(
    deserializer: D,
) -> Result<Option<MeshTargetSize>, D::Error>
where
    D: Deserializer<'de>,
{
    let Some(value) = Option::<serde_yaml::Value>::deserialize(deserializer)? else {
        return Ok(None);
    };
    match value {
        serde_yaml::Value::Null => Ok(None),
        serde_yaml::Value::String(value) if value.eq_ignore_ascii_case("auto") => Ok(None),
        serde_yaml::Value::Number(value) => {
            let value = value
                .as_f64()
                .ok_or_else(|| D::Error::custom("mesh.target_size must be a finite number"))?;
            if !value.is_finite() || value <= 0.0 {
                return Err(D::Error::custom(
                    "mesh.target_size must be finite and positive",
                ));
            }
            Ok(Some(MeshTargetSize::LengthM(value)))
        }
        _ => Err(D::Error::custom(
            "mesh.target_size must be auto, null, or a positive number in geometry units",
        )),
    }
}

fn deserialize_optional_auto_f64<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let Some(value) = Option::<serde_yaml::Value>::deserialize(deserializer)? else {
        return Ok(None);
    };
    match value {
        serde_yaml::Value::Null => Ok(None),
        serde_yaml::Value::String(value) if value.eq_ignore_ascii_case("auto") => Ok(None),
        serde_yaml::Value::Number(value) => {
            let value = value
                .as_f64()
                .ok_or_else(|| D::Error::custom("value must be a finite number"))?;
            if !value.is_finite() || value <= 0.0 {
                return Err(D::Error::custom("value must be finite and positive"));
            }
            Ok(Some(value))
        }
        _ => Err(D::Error::custom(
            "value must be auto, null, or a positive number",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::{
        GeometrySource, MeshDescriptor, MeshKind, Region, SourceGeometry, SourceGeometryKind,
        SurfaceMesh, TessellationProfile,
    };

    fn sample_geometry() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo:fea_document_test".to_string(),
            source: GeometrySource {
                path: "fixture.step".to_string(),
                sha256: "fixture".to_string(),
                importer_version: "test".to_string(),
            },
            source_geometry: SourceGeometry {
                kind: SourceGeometryKind::Cad,
                assembly: None,
                material_evidence: Vec::new(),
                cad_evaluators: Vec::new(),
            },
            tessellation_profile: TessellationProfile::default(),
            units: UnitSystem::Meter,
            revision: 1,
            meshes: vec![MeshDescriptor {
                mesh_id: "mesh_1".to_string(),
                kind: MeshKind::Surface,
                vertex_count: 3,
                element_count: 1,
            }],
            surface_meshes: vec![SurfaceMesh::new(
                "mesh_1",
                vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                vec![[0, 1, 2]],
            )],
            regions: vec![Region {
                region_id: "tip".to_string(),
                name: "Tip".to_string(),
                tag: Some("tip".to_string()),
                cad_ownership: None,
            }],
            region_entity_mappings: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    fn resolve_linear_static_mesh_options(
        mesh: Option<&FeaMeshDocument>,
    ) -> Result<Option<VolumeMeshingOptions>, String> {
        resolve_mesh_options(
            mesh,
            AnalysisCreateModelProfile::LinearStaticStructural,
            AnalysisRunKind::LinearStatic,
        )
    }

    #[test]
    fn fea_document_resolves_moment_and_torque_loads() {
        let geometry = sample_geometry();
        for (load_type, expected_id) in [("moment", "tip_moment"), ("torque", "tip_torque")] {
            let load: FeaLoadDocument = serde_yaml::from_str(&format!(
                r#"
id: {expected_id}
region: tag:tip
type: {load_type}
vector: [1.0, 2.0, 3.0]
"#
            ))
            .expect("load document should parse");

            let resolved = resolve_load(&load, &geometry, &BTreeMap::new())
                .expect("load should resolve against geometry");

            assert_eq!(resolved.load_id, expected_id);
            assert_eq!(resolved.region_id, "tip");
            assert!(matches!(
                resolved.kind,
                LoadKind::Moment {
                    mx: 1.0,
                    my: 2.0,
                    mz: 3.0
                }
            ));
        }
    }

    #[test]
    fn fea_document_moment_requires_vector() {
        let geometry = sample_geometry();
        let load: FeaLoadDocument = serde_yaml::from_str(
            r#"
id: tip_moment
region: tip
type: moment
"#,
        )
        .expect("load document should parse");

        let err = resolve_load(&load, &geometry, &BTreeMap::new())
            .expect_err("moment without vector should fail");

        assert!(err.contains("moment load requires vector: [x, y, z]"));
    }

    #[test]
    fn fea_document_resolves_wrench_load() {
        let geometry = sample_geometry();
        let load: FeaLoadDocument = serde_yaml::from_str(
            r#"
id: tip_wrench
region: tag:tip
type: wrench
force: [10.0, 20.0, 30.0]
moment: [1.0, 2.0, 3.0]
point: [0.1, 0.2, 0.3]
"#,
        )
        .expect("load document should parse");

        let resolved = resolve_load(&load, &geometry, &BTreeMap::new())
            .expect("load should resolve against geometry");

        assert_eq!(resolved.load_id, "tip_wrench");
        assert_eq!(resolved.region_id, "tip");
        assert!(matches!(
            resolved.kind,
            LoadKind::Wrench {
                fx: 10.0,
                fy: 20.0,
                fz: 30.0,
                mx: 1.0,
                my: 2.0,
                mz: 3.0,
                px: 0.1,
                py: 0.2,
                pz: 0.3,
            }
        ));
    }

    #[test]
    fn fea_document_scales_wrench_point_from_geometry_units_to_meters() {
        let mut geometry = sample_geometry();
        geometry.units = UnitSystem::Millimeter;
        let load: FeaLoadDocument = serde_yaml::from_str(
            r#"
id: tip_wrench
region: tag:tip
type: wrench
force: [10.0, 20.0, 30.0]
moment: [1.0, 2.0, 3.0]
point: [100.0, 200.0, 300.0]
"#,
        )
        .expect("load document should parse");

        let resolved = resolve_load(&load, &geometry, &BTreeMap::new())
            .expect("load should resolve against geometry");

        assert!(matches!(
            resolved.kind,
            LoadKind::Wrench {
                px,
                py,
                pz,
                ..
            } if (px - 0.1).abs() <= 1.0e-12
                && (py - 0.2).abs() <= 1.0e-12
                && (pz - 0.3).abs() <= 1.0e-12
        ));
    }

    #[test]
    fn fea_document_wrench_requires_force_moment_and_point() {
        let geometry = sample_geometry();
        let load: FeaLoadDocument = serde_yaml::from_str(
            r#"
id: tip_wrench
region: tip
type: wrench
force: [1.0, 0.0, 0.0]
moment: [0.0, 0.0, 1.0]
"#,
        )
        .expect("load document should parse");

        let err = resolve_load(&load, &geometry, &BTreeMap::new())
            .expect_err("wrench without point should fail");

        assert!(err.contains("wrench load requires point: [px, py, pz]"));
    }

    #[test]
    fn fea_document_moment_rejects_unknown_fields() {
        let err = serde_yaml::from_str::<FeaLoadDocument>(
            r#"
id: tip_moment
region: tip
type: moment
vector: [1.0, 2.0, 3.0]
units: n_m
"#,
        )
        .expect_err("unknown moment load fields should be rejected");

        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn fea_document_mesh_options_accept_auto_refinement_controls() {
        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
kind: solid
element: tetrahedron4
element_order: linear
profile: adaptive
max_elements: 250000
target_size: auto
min_size: 0.001
max_size: 0.02
growth_rate: 1.35
refinement:
  strategy: auto
  max_iterations: 4
  convergence:
    field_change_tolerance: 0.05
    energy_change_tolerance: 0.02
    residual_tolerance: auto
  focus:
    loads: fine
    constraints: fine
    interfaces: normal
    curvature: true
    small_features: true
  indicators:
    structural:
      stress_gradient: true
      displacement_gradient: false
      plastic_strain: auto
    modal:
      modal_strain_energy: on
      frequency_residual: off
    thermal:
      temperature_gradient: true
"#,
        )
        .expect("mesh document should parse");

        let options = resolve_linear_static_mesh_options(Some(&mesh))
            .expect("mesh options should resolve")
            .expect("mesh options should be present");

        assert_eq!(options.backend, MeshBackendKind::Auto);
        assert_eq!(options.kind, MeshKindRequest::Solid);
        assert_eq!(options.element, VolumeElementKind::Tetrahedron4);
        assert_eq!(options.element_order, MeshElementOrder::Linear);
        assert_eq!(options.profile, MeshProfile::Adaptive);
        assert_eq!(options.max_elements, 250_000);
        assert_eq!(options.target_size, MeshTargetSize::Auto);
        assert_eq!(options.min_size_m, Some(0.001));
        assert_eq!(options.max_size_m, Some(0.02));
        assert_eq!(options.growth_rate, Some(1.35));
        assert_eq!(options.refinement.strategy, RefinementStrategy::Auto);
        assert_eq!(options.refinement.max_iterations, 4);
        assert_eq!(options.refinement.convergence.field_change_tolerance, 0.05);
        assert_eq!(options.refinement.convergence.energy_change_tolerance, 0.02);
        assert_eq!(options.refinement.convergence.residual_tolerance, None);
        assert_eq!(options.validation.min_bounds_coverage_ratio, 0.90);
        assert_eq!(options.validation.min_volume_coverage_ratio, 0.90);
        assert_eq!(options.validation.min_boundary_area_ratio, 0.90);
        assert_eq!(options.validation.min_boundary_face_recovery_ratio, 1.0);
        assert_eq!(options.validation.min_boundary_edge_recovery_ratio, 1.0);
        assert_eq!(options.validation.quality, QualityThresholds::default());
        assert_eq!(options.refinement.focus.loads, RefinementFocusLevel::Fine);
        assert_eq!(
            options.refinement.focus.constraints,
            RefinementFocusLevel::Fine
        );
        assert_eq!(
            options.refinement.focus.interfaces,
            RefinementFocusLevel::Normal
        );
        assert!(options.refinement.focus.curvature);
        assert!(options.refinement.focus.small_features);
        let indicators = &options.refinement.indicators.namespaces;
        assert_eq!(
            indicators["structural"]["stress_gradient"],
            RefinementIndicatorMode::On
        );
        assert_eq!(
            indicators["structural"]["displacement_gradient"],
            RefinementIndicatorMode::Off
        );
        assert_eq!(
            indicators["structural"]["plastic_strain"],
            RefinementIndicatorMode::Auto
        );
        assert_eq!(
            indicators["modal"]["modal_strain_energy"],
            RefinementIndicatorMode::On
        );
        assert_eq!(
            indicators["modal"]["frequency_residual"],
            RefinementIndicatorMode::Off
        );
        assert_eq!(
            indicators["thermal"]["temperature_gradient"],
            RefinementIndicatorMode::On
        );
    }

    #[test]
    fn fea_document_mesh_options_accept_validation_policy_controls() {
        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
validation:
  coverage: strict
  quality: strict
  min_bounds_coverage_ratio: 0.95
  max_volume_components: 2
"#,
        )
        .expect("mesh document should parse validation policy");

        let options = resolve_linear_static_mesh_options(Some(&mesh))
            .expect("mesh options should resolve")
            .expect("mesh options should be present");

        assert_eq!(options.validation.min_bounds_coverage_ratio, 0.95);
        assert_eq!(options.validation.min_volume_coverage_ratio, 1.0);
        assert_eq!(options.validation.min_boundary_area_ratio, 1.0);
        assert_eq!(options.validation.min_boundary_face_recovery_ratio, 1.0);
        assert_eq!(options.validation.min_boundary_edge_recovery_ratio, 1.0);
        assert_eq!(options.validation.max_volume_component_count, Some(2));
        assert_eq!(
            options.validation.quality,
            QualityThresholds {
                min_scaled_jacobian: 0.25,
                max_aspect_ratio: 10.0,
                max_boundary_projection_error_m: 1.0e-8,
                allow_inverted_elements: false,
            }
        );
    }

    #[test]
    fn fea_document_mesh_options_accept_relaxed_validation_presets() {
        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
validation:
  coverage: relaxed
  quality: relaxed
"#,
        )
        .expect("mesh document should parse validation presets");

        let options = resolve_linear_static_mesh_options(Some(&mesh))
            .expect("mesh options should resolve")
            .expect("mesh options should be present");

        assert_eq!(options.validation.min_bounds_coverage_ratio, 0.80);
        assert_eq!(options.validation.min_volume_coverage_ratio, 0.80);
        assert_eq!(options.validation.min_boundary_area_ratio, 0.80);
        assert_eq!(options.validation.min_boundary_face_recovery_ratio, 0.95);
        assert_eq!(options.validation.min_boundary_edge_recovery_ratio, 0.95);
        assert_eq!(
            options.validation.quality,
            QualityThresholds {
                min_scaled_jacobian: 0.05,
                max_aspect_ratio: 50.0,
                max_boundary_projection_error_m: 1.0e-4,
                allow_inverted_elements: false,
            }
        );
    }

    #[test]
    fn fea_document_mesh_options_reject_invalid_validation_policy_controls() {
        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
validation:
  min_volume_coverage_ratio: 1.2
"#,
        )
        .expect("mesh document should parse before semantic validation");

        let err = resolve_linear_static_mesh_options(Some(&mesh))
            .expect_err("invalid validation ratio should fail");

        assert!(err.contains("mesh.validation.min_volume_coverage_ratio"));

        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
validation:
  max_volume_components: 0
"#,
        )
        .expect("mesh document should parse before semantic validation");

        let err = resolve_linear_static_mesh_options(Some(&mesh))
            .expect_err("invalid component count should fail");

        assert!(err.contains("mesh.validation.max_volume_components"));
    }

    #[test]
    fn fea_document_mesh_options_reject_invalid_size_envelope_controls() {
        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
min_size: 0.02
max_size: 0.01
growth_rate: 1.2
"#,
        )
        .expect("mesh document should parse before semantic validation");

        let err = resolve_linear_static_mesh_options(Some(&mesh))
            .expect_err("invalid size envelope should fail");
        assert!(err.contains("mesh.min_size"));

        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
min_size: 0.001
max_size: 0.01
growth_rate: 0.95
"#,
        )
        .expect("mesh document should parse before semantic validation");

        let err = resolve_linear_static_mesh_options(Some(&mesh))
            .expect_err("invalid growth rate should fail");
        assert!(err.contains("mesh.growth_rate"));
    }

    #[test]
    fn fea_document_defaults_linear_static_structural_mesh_options() {
        let options = resolve_linear_static_mesh_options(None)
            .expect("default structural mesh options should resolve")
            .expect("linear static structural study should default a solid analysis mesh");

        assert_eq!(options.kind, MeshKindRequest::Solid);
        assert_eq!(options.element, VolumeElementKind::Tetrahedron4);
        assert_eq!(options.profile, MeshProfile::AnalysisReady);
        assert_eq!(options.backend, MeshBackendKind::Auto);
        assert_eq!(options.refinement.strategy, RefinementStrategy::Auto);

        let modal_default = resolve_mesh_options(
            None,
            AnalysisCreateModelProfile::ModalStructural,
            AnalysisRunKind::Modal,
        )
        .expect("modal default should resolve");
        assert!(modal_default.is_none());
    }

    #[test]
    fn fea_document_mesh_options_accept_backend_selection() {
        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
backend: structured_grid_tetrahedron
"#,
        )
        .expect("mesh document should parse backend");

        let options = resolve_linear_static_mesh_options(Some(&mesh))
            .expect("mesh options should resolve")
            .expect("mesh options should be present");

        assert_eq!(options.backend, MeshBackendKind::StructuredGridTetrahedron);
    }

    #[test]
    fn fea_document_mesh_options_accept_physics_refinement_namespaces() {
        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
refinement:
  indicators:
    structural:
      load_regions: auto
      constraint_regions: auto
    modal:
      mode_shape_curvature: on
    thermal:
      convection_regions: auto
      prescribed_temperature_regions: auto
    thermo_mechanical:
      strain_energy_density: auto
      region_temperature_delta: auto
    electro_thermal:
      source_regions: auto
      ground_regions: auto
    electromagnetic:
      source_regions: auto
      ground_regions: auto
      insulation_regions: auto
    acoustic:
      pressure_curvature: auto
      impedance_regions: auto
      source_regions: auto
    cfd:
      inlet_regions: auto
      outlet_regions: auto
    cht:
      interface_heat_flux_jump: on
      interface_temperature_jump: on
      solid_heat_flux_gradient: auto
      fluid_boundary_layer: auto
    fsi:
      interface_displacement_jump: on
      interface_traction_jump: on
      structural_stress_gradient: auto
      fluid_pressure_gradient: auto
      fluid_velocity_gradient: auto
"#,
        )
        .expect("mesh document should parse");

        let options = resolve_linear_static_mesh_options(Some(&mesh))
            .expect("mesh options should resolve")
            .expect("mesh options should be present");

        let indicators = &options.refinement.indicators.namespaces;
        assert_eq!(
            indicators["structural"]["load_regions"],
            RefinementIndicatorMode::Auto
        );
        assert_eq!(
            indicators["thermal"]["convection_regions"],
            RefinementIndicatorMode::Auto
        );
        assert_eq!(
            indicators["thermo_mechanical"]["strain_energy_density"],
            RefinementIndicatorMode::Auto
        );
        assert_eq!(
            indicators["electromagnetic"]["insulation_regions"],
            RefinementIndicatorMode::Auto
        );
        assert_eq!(
            indicators["cht"]["interface_heat_flux_jump"],
            RefinementIndicatorMode::On
        );
        assert_eq!(
            indicators["fsi"]["interface_traction_jump"],
            RefinementIndicatorMode::On
        );
    }

    #[test]
    fn fea_document_mesh_options_accept_numeric_size_and_residual() {
        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
target_size: 0.002
refinement:
  strategy: adaptive
  max_iterations: 2
  convergence:
    residual_tolerance: 1.0e-6
"#,
        )
        .expect("mesh document should parse");

        let options = resolve_linear_static_mesh_options(Some(&mesh))
            .expect("mesh options should resolve")
            .expect("mesh options should be present");

        assert_eq!(options.target_size, MeshTargetSize::LengthM(0.002));
        assert_eq!(options.refinement.strategy, RefinementStrategy::Adaptive);
        assert_eq!(options.refinement.max_iterations, 2);
        assert_eq!(
            options.refinement.convergence.residual_tolerance,
            Some(1.0e-6)
        );
    }

    #[test]
    fn fea_document_mesh_options_reject_none_refinement_iterations() {
        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
refinement:
  strategy: none
  max_iterations: 1
"#,
        )
        .expect("mesh document should parse");

        let err = resolve_linear_static_mesh_options(Some(&mesh))
            .expect_err("none refinement should reject positive iteration count");

        assert!(err.contains("mesh.refinement.max_iterations"));
    }

    #[test]
    fn fea_document_mesh_options_reject_invalid_numeric_controls() {
        let err = serde_yaml::from_str::<FeaMeshDocument>(
            r#"
target_size: -0.002
"#,
        )
        .expect_err("negative target size should fail during parsing");

        assert!(err.to_string().contains("mesh.target_size"));
    }

    #[test]
    fn fea_document_mesh_options_reject_unsupported_mesh_kind() {
        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
kind: display_only
"#,
        )
        .expect("display-only mesh document should parse");

        let err = resolve_linear_static_mesh_options(Some(&mesh))
            .expect_err("display-only mesh kind is unsupported for analysis");

        assert!(err.contains("mesh.kind"));
        assert!(err.contains("solid"));
    }

    #[test]
    fn fea_document_mesh_options_reject_unsupported_solid_element() {
        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
element: hex8
"#,
        )
        .expect("mesh document should parse");

        let err = resolve_linear_static_mesh_options(Some(&mesh))
            .expect_err("hex8 solid element is not supported yet");

        assert!(err.contains("mesh.element"));
        assert!(err.contains("tetrahedron4"));
    }

    #[test]
    fn fea_document_mesh_options_reject_unknown_refinement_indicators() {
        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
refinement:
  indicators:
    structural:
      stress_gradient: true
      made_up_indicator: true
"#,
        )
        .expect("mesh document should parse before semantic validation");

        let err = resolve_linear_static_mesh_options(Some(&mesh))
            .expect_err("unknown refinement indicator should fail validation");

        assert!(err.contains("mesh.refinement.indicators.structural.made_up_indicator"));

        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
refinement:
  indicators:
    made_up_physics:
      stress_gradient: true
"#,
        )
        .expect("mesh document should parse before semantic validation");

        let err = resolve_linear_static_mesh_options(Some(&mesh))
            .expect_err("unknown refinement namespace should fail validation");

        assert!(err.contains("mesh.refinement.indicators namespace `made_up_physics`"));

        let mesh: FeaMeshDocument = serde_yaml::from_str(
            r#"
refinement:
  indicators:
    coupling:
      interface_jump: true
"#,
        )
        .expect("mesh document should parse before semantic validation");

        let err = resolve_linear_static_mesh_options(Some(&mesh))
            .expect_err("generic coupling namespace should fail validation");

        assert!(err.contains("mesh.refinement.indicators namespace `coupling`"));
    }

    #[test]
    fn fea_document_refinement_indicator_applicability_matches_profile_context() {
        let structural = BTreeMap::from([(
            "structural".to_string(),
            BTreeMap::from([("stress_gradient".to_string(), RefinementIndicatorMode::Auto)]),
        )]);
        validate_refinement_indicator_applicability(
            &structural,
            AnalysisCreateModelProfile::LinearStaticStructural,
            AnalysisRunKind::LinearStatic,
            &FeaDomainsDocument::default(),
        )
        .expect("structural indicators should apply to linear static structural studies");

        let unrelated = BTreeMap::from([(
            "thermal".to_string(),
            BTreeMap::from([(
                "temperature_gradient".to_string(),
                RefinementIndicatorMode::Auto,
            )]),
        )]);
        let err = validate_refinement_indicator_applicability(
            &unrelated,
            AnalysisCreateModelProfile::LinearStaticStructural,
            AnalysisRunKind::LinearStatic,
            &FeaDomainsDocument::default(),
        )
        .expect_err("thermal indicators should not apply to uncoupled structural studies");
        assert!(err.contains("mesh.refinement.indicators.thermal"));
        assert!(err.contains("linear_static_structural"));

        let coupled = BTreeMap::from([(
            "thermo_mechanical".to_string(),
            BTreeMap::from([("thermal_stress".to_string(), RefinementIndicatorMode::On)]),
        )]);
        validate_refinement_indicator_applicability(
            &coupled,
            AnalysisCreateModelProfile::ThermoMechanicalCoupled,
            AnalysisRunKind::Transient,
            &FeaDomainsDocument::default(),
        )
        .expect("thermo-mechanical indicators should apply to thermo-mechanical studies");
    }
}
