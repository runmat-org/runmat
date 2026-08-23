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
use runmat_meshing_core::{ElementOrder, MeshingRequestSettings};
use serde::de::DeserializeOwned;
use serde::Deserialize;

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
    #[serde(default = "default_mesh_element_order")]
    element_order: ElementOrder,
    #[serde(default = "default_mesh_max_elements")]
    maximum_elements: u64,
    #[serde(default = "default_mesh_target_edge_length_m")]
    target_edge_length_m: f64,
    #[serde(default = "default_mesh_maximum_chordal_deviation_m")]
    maximum_chordal_deviation_m: f64,
    #[serde(default = "default_mesh_maximum_grading_ratio")]
    maximum_grading_ratio: f64,
    #[serde(default)]
    deterministic_seed: u64,
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
    let meshing_settings =
        resolve_meshing_settings(study.mesh.as_ref(), study.model.profile, run_kind)?;
    let outputs = study.outputs.iter().map(resolve_output_request).collect();
    let spec = AnalysisStudySpec {
        study_id: study.id,
        geometry,
        create_model_intent: intent,
        model,
        run_kind,
        backend: study.run.backend,
        meshing_settings,
        outputs,
        solver_mesh_artifact_path: None,
        meshing_evidence_artifact_path: None,
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

fn resolve_meshing_settings(
    mesh: Option<&FeaMeshDocument>,
    profile: AnalysisCreateModelProfile,
    run_kind: AnalysisRunKind,
) -> Result<Option<MeshingRequestSettings>, String> {
    let required = matches!(profile, AnalysisCreateModelProfile::LinearStaticStructural)
        && matches!(run_kind, AnalysisRunKind::LinearStatic);
    let Some(mesh) = mesh else {
        return Ok(required.then(MeshingRequestSettings::default));
    };
    let mut settings = MeshingRequestSettings {
        element_order: mesh.element_order,
        deterministic_seed: mesh.deterministic_seed,
        target_edge_length_m: mesh.target_edge_length_m,
        maximum_chordal_deviation_m: mesh.maximum_chordal_deviation_m,
        maximum_grading_ratio: mesh.maximum_grading_ratio,
        ..MeshingRequestSettings::default()
    };
    settings.resources.maximum_elements = mesh.maximum_elements;
    if settings.resources.maximum_elements == 0
        || !settings.target_edge_length_m.is_finite()
        || settings.target_edge_length_m <= 0.0
        || !settings.maximum_chordal_deviation_m.is_finite()
        || settings.maximum_chordal_deviation_m <= 0.0
        || !settings.maximum_grading_ratio.is_finite()
        || settings.maximum_grading_ratio < 1.0
    {
        return Err("mesh settings require positive finite sizes, a grading ratio of at least 1, and a nonzero element budget".to_string());
    }
    Ok(Some(settings))
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

fn default_mesh_element_order() -> ElementOrder {
    ElementOrder::Tet4
}

fn default_mesh_max_elements() -> u64 {
    10_000_000
}

fn default_mesh_target_edge_length_m() -> f64 {
    MeshingRequestSettings::default().target_edge_length_m
}

fn default_mesh_maximum_chordal_deviation_m() -> f64 {
    MeshingRequestSettings::default().maximum_chordal_deviation_m
}

fn default_mesh_maximum_grading_ratio() -> f64 {
    MeshingRequestSettings::default().maximum_grading_ratio
}

fn default_fail_fast() -> bool {
    true
}

fn default_assignment_confidence() -> EvidenceConfidence {
    EvidenceConfidence::Verified
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
    fn fea_document_resolves_canonical_meshing_settings() {
        let mesh: FeaMeshDocument = serde_yaml::from_str(
            "element_order: tet10\ntarget_edge_length_m: 0.02\nmaximum_chordal_deviation_m: 0.0002\nmaximum_grading_ratio: 1.5\nmaximum_elements: 42000\ndeterministic_seed: 17\n",
        )
        .unwrap();
        let settings = resolve_meshing_settings(
            Some(&mesh),
            AnalysisCreateModelProfile::LinearStaticStructural,
            AnalysisRunKind::LinearStatic,
        )
        .unwrap()
        .unwrap();
        assert_eq!(settings.element_order, ElementOrder::Tet10);
        assert_eq!(settings.target_edge_length_m, 0.02);
        assert_eq!(settings.maximum_chordal_deviation_m, 0.0002);
        assert_eq!(settings.maximum_grading_ratio, 1.5);
        assert_eq!(settings.resources.maximum_elements, 42_000);
        assert_eq!(settings.deterministic_seed, 17);
    }

    #[test]
    fn fea_document_defaults_meshing_only_for_solid_structural_studies() {
        assert!(resolve_meshing_settings(
            None,
            AnalysisCreateModelProfile::LinearStaticStructural,
            AnalysisRunKind::LinearStatic,
        )
        .unwrap()
        .is_some());
        assert!(resolve_meshing_settings(
            None,
            AnalysisCreateModelProfile::ModalStructural,
            AnalysisRunKind::Modal,
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn fea_document_rejects_retired_mesh_fields() {
        for source in [
            "backend: auto",
            "kind: solid",
            "element: tetrahedron4",
            "profile: analysis_ready",
            "refinement: {}",
            "validation: {}",
        ] {
            assert!(
                serde_yaml::from_str::<FeaMeshDocument>(source).is_err(),
                "{source}"
            );
        }
    }

    #[test]
    fn fea_document_rejects_invalid_meshing_settings() {
        for source in [
            "target_edge_length_m: 0",
            "maximum_chordal_deviation_m: 0",
            "maximum_grading_ratio: 0.5",
            "maximum_elements: 0",
        ] {
            let mesh = serde_yaml::from_str::<FeaMeshDocument>(source).unwrap();
            assert!(resolve_meshing_settings(
                Some(&mesh),
                AnalysisCreateModelProfile::LinearStaticStructural,
                AnalysisRunKind::LinearStatic,
            )
            .is_err());
        }
    }
}
