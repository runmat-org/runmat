use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use runmat_analysis_core::{
    AnalysisBoundary, AnalysisConnector, AnalysisConstraint, AnalysisInterface, AnalysisModel,
    AnalysisModelId, AnalysisProcedure, AnalysisSequence, BeamElementModel, BeamSectionModel,
    CfdDomain, CoordinateFrame, ElectroThermalDomain, ElectromagneticDomain, EvidenceConfidence,
    ExcitationAction, ExcitationCase, ExcitationTarget, InitialCondition, MaterialAcousticModel,
    MaterialAssignment, MaterialElectricalModel, MaterialMechanicalModel, MaterialModel,
    MaterialPlasticModel, MaterialThermalModel, PhysicalSignal, PhysicsFamily, RegionTarget,
    ResponseSpectrum, ShellElementModel, ShellSectionModel, StructuralElement,
    StructuralElementKind, StructuralModel, StructuralNode, ThermoMechanicalDomain,
};
use runmat_analysis_fea::ComputeBackend;
use runmat_geometry_core::{GeometryAsset, UnitSystem};
use runmat_geometry_io::GeometryImportOptions;
use runmat_meshing_core::{ElementOrder, MeshingRequestSettings};
use serde::Deserialize;

use super::{
    analysis_create_model_op, AnalysisAcousticRunOptions, AnalysisCfdRunOptions,
    AnalysisChtRunOptions, AnalysisCreateModelIntentSpec, AnalysisElectromagneticRunOptions,
    AnalysisFsiRunOptions, AnalysisModalRunOptions, AnalysisModelTemplate,
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
    constraints: Vec<AnalysisConstraint>,
    #[serde(default)]
    boundaries: Vec<AnalysisBoundary>,
    #[serde(default)]
    connectors: Vec<AnalysisConnector>,
    #[serde(default)]
    interfaces: Vec<AnalysisInterface>,
    #[serde(default)]
    signals: Vec<PhysicalSignal>,
    #[serde(default)]
    excitation_cases: Vec<ExcitationCase>,
    #[serde(default)]
    excitations: Vec<ExcitationAction>,
    #[serde(default)]
    initial_conditions: Vec<InitialCondition>,
    #[serde(default)]
    response_spectra: Vec<ResponseSpectrum>,
    sequence: AnalysisSequence,
    #[serde(default)]
    domains: FeaDomainsDocument,
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
    template: AnalysisModelTemplate,
    #[serde(default)]
    frame: Option<CoordinateFrame>,
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
    TemplateScaffold,
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
    #[serde(rename = "type")]
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
    #[serde(rename = "type")]
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

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum FeaStructuralSectionType {
    Beam,
    Shell,
}

impl FeaStructuralSectionType {
    fn is_beam(self) -> bool {
        matches!(self, Self::Beam)
    }

    fn is_shell(self) -> bool {
        matches!(self, Self::Shell)
    }
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
    #[serde(default = "default_backend")]
    backend: ComputeBackend,
    #[serde(default)]
    solver: Option<FeaSolverOptionsDocument>,
}

#[derive(Debug, Deserialize)]
#[serde(
    tag = "procedure",
    content = "options",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum FeaSolverOptionsDocument {
    LinearStatic(AnalysisRunOptions),
    Modal(AnalysisModalRunOptions),
    Acoustic(AnalysisAcousticRunOptions),
    Thermal(AnalysisThermalRunOptions),
    Transient(AnalysisTransientRunOptions),
    Cfd(AnalysisCfdRunOptions),
    Cht(AnalysisChtRunOptions),
    Fsi(AnalysisFsiRunOptions),
    Nonlinear(AnalysisNonlinearRunOptions),
    Electromagnetic(AnalysisElectromagneticRunOptions),
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
        template: study.model.template,
    };
    let model = resolve_model(&study, &geometry, &intent)?;
    let run_kind = resolve_run_kind(&study.sequence)?;
    let run_options = resolve_run_options(&study.run, run_kind)?;
    let meshing_settings = resolve_meshing_settings(study.mesh.as_ref(), run_kind)?;
    let spec = AnalysisStudySpec {
        study_id: study.id,
        geometry,
        create_model_intent: intent,
        model,
        run_kind,
        backend: study.run.backend,
        meshing_settings,
        outputs: Vec::new(),
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

fn resolve_meshing_settings(
    mesh: Option<&FeaMeshDocument>,
    run_kind: AnalysisRunKind,
) -> Result<Option<MeshingRequestSettings>, String> {
    let required = matches!(run_kind, AnalysisRunKind::LinearStatic);
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
        && study.model.defaults == FeaModelDefaultsMode::TemplateScaffold
    {
        return Ok(None);
    }

    let mut model = match study.model.defaults {
        FeaModelDefaultsMode::TemplateScaffold => {
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
    model.constraints = study.constraints.clone();
    model.boundaries = study.boundaries.clone();
    model.connectors = study.connectors.clone();
    model.interfaces = study.interfaces.clone();
    model.signals = study.signals.clone();
    model.excitation_cases = study.excitation_cases.clone();
    model.excitations = study.excitations.clone();
    model.initial_conditions = study.initial_conditions.clone();
    model.response_spectra = study.response_spectra.clone();
    model.sequence = study.sequence.clone();
    resolve_typed_region_targets(&mut model, geometry, &study.regions)?;
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
        .unwrap_or(&[]);
    let element_docs = structural
        .map(|value| value.elements.as_slice())
        .unwrap_or(&[]);
    let section_docs = structural
        .map(|value| value.sections.as_slice())
        .unwrap_or(&[]);

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

fn resolve_run_kind(sequence: &AnalysisSequence) -> Result<AnalysisRunKind, String> {
    let [step] = sequence.steps.as_slice() else {
        return Err(format!(
            "version 1 FEA execution requires exactly one procedure step; sequence `{}` contains {}",
            sequence.sequence_id,
            sequence.steps.len()
        ));
    };
    match (step.family, step.procedure) {
        (PhysicsFamily::Structural, AnalysisProcedure::Static) => Ok(AnalysisRunKind::LinearStatic),
        (
            PhysicsFamily::Structural,
            AnalysisProcedure::ModalExtraction | AnalysisProcedure::PrestressedModalExtraction,
        ) => Ok(AnalysisRunKind::Modal),
        (PhysicsFamily::Structural, AnalysisProcedure::TransientResponse) => {
            Ok(AnalysisRunKind::Transient)
        }
        (PhysicsFamily::Structural, AnalysisProcedure::Nonlinear) => Ok(AnalysisRunKind::Nonlinear),
        (PhysicsFamily::Thermal, AnalysisProcedure::Static) => Ok(AnalysisRunKind::Thermal),
        (PhysicsFamily::Electromagnetic, AnalysisProcedure::Static | AnalysisProcedure::HarmonicResponse) => {
            Ok(AnalysisRunKind::Electromagnetic)
        }
        (PhysicsFamily::Acoustic, AnalysisProcedure::HarmonicResponse) => Ok(AnalysisRunKind::Acoustic),
        (PhysicsFamily::Fluid, AnalysisProcedure::Static | AnalysisProcedure::TransientResponse) => {
            Ok(AnalysisRunKind::Cfd)
        }
        (
            PhysicsFamily::Coupled(
                runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical
                | runmat_analysis_core::CoupledPhysicsFamily::Electrothermal,
            ),
            AnalysisProcedure::Static | AnalysisProcedure::TransientResponse,
        ) => Ok(AnalysisRunKind::Transient),
        (PhysicsFamily::Coupled(runmat_analysis_core::CoupledPhysicsFamily::ConjugateHeatTransfer), AnalysisProcedure::Static | AnalysisProcedure::TransientResponse) => {
            Ok(AnalysisRunKind::Cht)
        }
        (PhysicsFamily::Coupled(runmat_analysis_core::CoupledPhysicsFamily::FluidStructureInteraction), AnalysisProcedure::Static | AnalysisProcedure::TransientResponse) => {
            Ok(AnalysisRunKind::Fsi)
        }
        (family, procedure) => Err(format!(
            "procedure step `{}` selects unsupported version 1 execution combination {family:?}/{procedure:?}",
            step.step_id.as_str()
        )),
    }
}

fn resolve_run_options(
    run: &FeaRunDocument,
    run_kind: AnalysisRunKind,
) -> Result<ResolvedRunOptions, String> {
    let Some(options) = &run.solver else {
        return Ok(ResolvedRunOptions::default());
    };
    let mut resolved = ResolvedRunOptions::default();
    let selected = match options {
        FeaSolverOptionsDocument::LinearStatic(value) => {
            resolved.linear_static = Some(value.clone());
            AnalysisRunKind::LinearStatic
        }
        FeaSolverOptionsDocument::Modal(value) => {
            resolved.modal = Some(value.clone());
            AnalysisRunKind::Modal
        }
        FeaSolverOptionsDocument::Acoustic(value) => {
            resolved.acoustic = Some(value.clone());
            AnalysisRunKind::Acoustic
        }
        FeaSolverOptionsDocument::Thermal(value) => {
            resolved.thermal = Some(value.clone());
            AnalysisRunKind::Thermal
        }
        FeaSolverOptionsDocument::Transient(value) => {
            resolved.transient = Some(value.clone());
            AnalysisRunKind::Transient
        }
        FeaSolverOptionsDocument::Cfd(value) => {
            resolved.cfd = Some(value.clone());
            AnalysisRunKind::Cfd
        }
        FeaSolverOptionsDocument::Cht(value) => {
            resolved.cht = Some(value.clone());
            AnalysisRunKind::Cht
        }
        FeaSolverOptionsDocument::Fsi(value) => {
            resolved.fsi = Some(value.clone());
            AnalysisRunKind::Fsi
        }
        FeaSolverOptionsDocument::Nonlinear(value) => {
            resolved.nonlinear = Some(value.clone());
            AnalysisRunKind::Nonlinear
        }
        FeaSolverOptionsDocument::Electromagnetic(value) => {
            resolved.electromagnetic = Some(value.clone());
            AnalysisRunKind::Electromagnetic
        }
    };
    if selected != run_kind {
        return Err(format!(
            "run solver options select {selected:?}, but the sequence selects {run_kind:?}"
        ));
    }
    Ok(resolved)
}

fn empty_model(model_id: String, geometry: &GeometryAsset) -> AnalysisModel {
    AnalysisModel {
        model_id: AnalysisModelId(model_id.clone()),
        geometry_id: geometry.geometry_id.clone(),
        geometry_revision: geometry.revision,
        units: geometry.units,
        frame: CoordinateFrame::Global,
        materials: Vec::new(),
        material_assignments: Vec::new(),
        structural: None,
        thermo_mechanical: None,
        electro_thermal: None,
        electromagnetic: None,
        cfd: None,
        interfaces: Vec::new(),
        constraints: Vec::new(),
        boundaries: Vec::new(),
        connectors: Vec::new(),
        signals: Vec::new(),
        excitations: Vec::new(),
        excitation_cases: Vec::new(),
        initial_conditions: Vec::new(),
        response_spectra: Vec::new(),
        sequence: AnalysisSequence::from_steps(format!("{model_id}.sequence"), Vec::new()),
    }
}

fn has_explicit_model_data(study: &FeaStudyDocument) -> bool {
    !study.materials.is_empty()
        || !study.material_assignments.is_empty()
        || !study.constraints.is_empty()
        || !study.boundaries.is_empty()
        || !study.connectors.is_empty()
        || !study.interfaces.is_empty()
        || !study.signals.is_empty()
        || !study.excitation_cases.is_empty()
        || !study.excitations.is_empty()
        || !study.initial_conditions.is_empty()
        || !study.response_spectra.is_empty()
        || has_structural_model_data(study)
        || study.domains.thermo_mechanical.is_some()
        || study.domains.electro_thermal.is_some()
        || study.domains.electromagnetic.is_some()
        || study.domains.cfd.is_some()
        || study.model.frame.is_some()
}

fn has_structural_model_data(study: &FeaStudyDocument) -> bool {
    study.structural.as_ref().is_some_and(|structural| {
        !structural.nodes.is_empty()
            || !structural.elements.is_empty()
            || !structural.sections.is_empty()
    })
}

fn resolve_typed_region_targets(
    model: &mut AnalysisModel,
    geometry: &GeometryAsset,
    aliases: &BTreeMap<String, FeaRegionDocument>,
) -> Result<(), String> {
    let mut failure = None;
    for assignment in &mut model.material_assignments {
        resolve_region_id(&mut assignment.region_id, geometry, aliases, &mut failure);
    }
    if let Some(structural) = &mut model.structural {
        for element in &mut structural.elements {
            resolve_region_id(&mut element.region_id, geometry, aliases, &mut failure);
        }
    }
    for constraint in &mut model.constraints {
        constraint.visit_targets_mut(|target| {
            resolve_region_target(target, geometry, aliases, &mut failure)
        });
    }
    for boundary in &mut model.boundaries {
        resolve_region_target(&mut boundary.target, geometry, aliases, &mut failure);
    }
    for connector in &mut model.connectors {
        resolve_region_target(&mut connector.primary, geometry, aliases, &mut failure);
        if let Some(secondary) = &mut connector.secondary {
            resolve_region_target(secondary, geometry, aliases, &mut failure);
        }
    }
    for interface in &mut model.interfaces {
        resolve_region_id(
            &mut interface.primary_region_id,
            geometry,
            aliases,
            &mut failure,
        );
        resolve_region_id(
            &mut interface.secondary_region_id,
            geometry,
            aliases,
            &mut failure,
        );
    }
    for action in &mut model.excitations {
        if let ExcitationTarget::Region(target) = &mut action.excitation_mut().target {
            resolve_region_target(target, geometry, aliases, &mut failure);
        }
    }
    for initial_condition in &mut model.initial_conditions {
        if let Some(target) = initial_condition.region_target_mut() {
            resolve_region_target(target, geometry, aliases, &mut failure);
        }
    }
    if let Some(domain) = &mut model.thermo_mechanical {
        for region in &mut domain.region_temperature_deltas {
            resolve_region_id(&mut region.region_id, geometry, aliases, &mut failure);
        }
    }
    if let Some(domain) = &mut model.electro_thermal {
        for region in &mut domain.region_conductivity_scales {
            resolve_region_id(&mut region.region_id, geometry, aliases, &mut failure);
        }
    }
    failure.map_or(Ok(()), Err)
}

fn resolve_region_target(
    target: &mut RegionTarget,
    geometry: &GeometryAsset,
    aliases: &BTreeMap<String, FeaRegionDocument>,
    failure: &mut Option<String>,
) {
    resolve_region_id(&mut target.region_id, geometry, aliases, failure);
}

fn resolve_region_id(
    region_id: &mut String,
    geometry: &GeometryAsset,
    aliases: &BTreeMap<String, FeaRegionDocument>,
    failure: &mut Option<String>,
) {
    if failure.is_some() {
        return;
    }
    match resolve_region_ref(region_id, geometry, aliases) {
        Ok(resolved) => *region_id = resolved,
        Err(error) => *failure = Some(error),
    }
}

fn resolve_region_ref(
    reference: &str,
    geometry: &GeometryAsset,
    aliases: &BTreeMap<String, FeaRegionDocument>,
) -> Result<String, String> {
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

    fn minimal_study_yaml() -> serde_yaml::Value {
        let mut document = serde_yaml::Mapping::new();
        document.insert("kind".into(), "study".into());
        document.insert("version".into(), FEA_DOCUMENT_VERSION.into());
        document.insert("id".into(), "typed-study".into());
        document.insert(
            "geometry".into(),
            serde_yaml::from_str("path: fixture.step\n").expect("geometry YAML"),
        );
        let mut model = serde_yaml::Mapping::new();
        model.insert(
            "template".into(),
            serde_yaml::to_value(AnalysisModelTemplate::LinearStaticStructural)
                .expect("model template"),
        );
        model.insert("defaults".into(), "none".into());
        document.insert("model".into(), model.into());
        document.insert(
            "run".into(),
            serde_yaml::from_str("backend: cpu\n").expect("run YAML"),
        );
        document.insert(
            "sequence".into(),
            serde_yaml::to_value(AnalysisSequence::from_steps(
                "typed-sequence",
                vec![runmat_analysis_core::ProcedureStep::structural_static(
                    "solve",
                )],
            ))
            .expect("sequence"),
        );
        document.into()
    }

    #[test]
    fn typed_document_fields_deserialize_without_legacy_translation() {
        let mut document = minimal_study_yaml();
        let mapping = document.as_mapping_mut().expect("study mapping");
        mapping.insert(
            "constraints".into(),
            serde_yaml::to_value(vec![AnalysisConstraint::Structural(
                runmat_analysis_core::StructuralConstraint {
                    constraint_id: "fixed".into(),
                    frame: CoordinateFrame::Global,
                    kind: runmat_analysis_core::StructuralConstraintKind::Fixed {
                        target: RegionTarget::boundary("tip"),
                    },
                },
            )])
            .expect("constraints"),
        );

        let parsed: RawFeaDocument = serde_yaml::from_value(document).expect("typed document");
        let RawFeaDocument::Study(study) = parsed else {
            panic!("expected study")
        };
        assert_eq!(study.constraints[0].constraint_id(), "fixed");
        assert_eq!(study.sequence.steps[0].step_id.as_str(), "solve");
    }

    #[test]
    fn required_sequence_is_not_authored_model_data() {
        let mut document = minimal_study_yaml();
        document
            .as_mapping_mut()
            .expect("study mapping")
            .get_mut("model")
            .expect("model")
            .as_mapping_mut()
            .expect("model mapping")
            .insert("defaults".into(), "template_scaffold".into());

        let parsed: RawFeaDocument = serde_yaml::from_value(document).expect("study document");
        let RawFeaDocument::Study(study) = parsed else {
            panic!("expected study")
        };

        assert!(!study.sequence.steps.is_empty());
        assert!(!has_explicit_model_data(&study));
        assert_eq!(study.model.defaults, FeaModelDefaultsMode::TemplateScaffold);
        let intent = AnalysisCreateModelIntentSpec {
            model_id: "scaffold-model".into(),
            template: study.model.template,
        };
        assert!(resolve_model(&study, &sample_geometry(), &intent)
            .expect("resolve scaffold-only model")
            .is_none());
    }

    #[test]
    fn retired_document_keys_and_run_kind_are_rejected() {
        for key in [
            "boundary_conditions",
            "loads",
            "steps",
            "outputs",
            "nodes",
            "elements",
            "sections",
        ] {
            let mut document = minimal_study_yaml();
            document
                .as_mapping_mut()
                .expect("study mapping")
                .insert(key.into(), serde_yaml::Value::Sequence(Vec::new()));
            let error = serde_yaml::from_value::<RawFeaDocument>(document)
                .expect_err("retired key must fail");
            assert!(error.to_string().contains("unknown field"), "{error}");
        }

        let error = serde_yaml::from_str::<FeaRunDocument>("backend: cpu\nkind: linear_static\n")
            .expect_err("run.kind must fail");
        assert!(error.to_string().contains("unknown field"), "{error}");
        let error = serde_yaml::from_str::<FeaRunDocument>("backend: cpu\noptions: {}\n")
            .expect_err("untyped run options must fail");
        assert!(error.to_string().contains("unknown field"), "{error}");
    }

    #[test]
    fn structural_kind_aliases_are_rejected() {
        let element = "id: beam\nregion: tip\nkind: beam\nnodes: [1, 2]\nsection: section\n";
        assert!(serde_yaml::from_str::<FeaStructuralElementDocument>(element).is_err());
        let section = "id: section\nkind: beam\narea_m2: 1\niy_m4: 1\niz_m4: 1\ntorsion_j_m4: 1\n";
        assert!(serde_yaml::from_str::<FeaStructuralSectionDocument>(section).is_err());
        let retired_variant =
            "id: section\ntype: beam_section\narea_m2: 1\niy_m4: 1\niz_m4: 1\ntorsion_j_m4: 1\n";
        assert!(serde_yaml::from_str::<FeaStructuralSectionDocument>(retired_variant).is_err());
    }

    #[test]
    fn execution_kind_is_derived_from_one_sequence_step() {
        let sequence = AnalysisSequence::from_steps(
            "sequence",
            vec![runmat_analysis_core::ProcedureStep::structural_modal(
                "modes",
            )],
        );
        assert_eq!(resolve_run_kind(&sequence).unwrap(), AnalysisRunKind::Modal);

        let multiple = AnalysisSequence::from_steps(
            "sequence",
            vec![
                runmat_analysis_core::ProcedureStep::structural_static("static"),
                runmat_analysis_core::ProcedureStep::structural_modal("modes"),
            ],
        );
        assert!(resolve_run_kind(&multiple)
            .expect_err("multi-step execution is not implemented")
            .contains("exactly one procedure step"));
    }

    #[test]
    fn typed_solver_options_must_match_the_sequence() {
        let run = FeaRunDocument {
            backend: ComputeBackend::Cpu,
            solver: Some(FeaSolverOptionsDocument::LinearStatic(
                AnalysisRunOptions::default(),
            )),
        };
        assert!(resolve_run_options(&run, AnalysisRunKind::Modal)
            .expect_err("mismatched solver options must fail")
            .contains("sequence selects Modal"));
    }

    #[test]
    fn region_aliases_resolve_across_typed_model_targets() {
        let geometry = sample_geometry();
        let mut aliases = BTreeMap::new();
        aliases.insert(
            "loaded".into(),
            FeaRegionDocument {
                selector: "tag:tip".into(),
            },
        );
        let mut model = empty_model("model".into(), &geometry);
        model.constraints = vec![AnalysisConstraint::Structural(
            runmat_analysis_core::StructuralConstraint {
                constraint_id: "fixed".into(),
                frame: CoordinateFrame::Global,
                kind: runmat_analysis_core::StructuralConstraintKind::Fixed {
                    target: RegionTarget::boundary("loaded"),
                },
            },
        )];
        model.boundaries = vec![AnalysisBoundary {
            boundary_id: "insulation".into(),
            target: RegionTarget::boundary("loaded"),
            operator: runmat_analysis_core::BoundaryOperator::MagneticInsulation,
        }];
        model.connectors = vec![AnalysisConnector {
            connector_id: "spring".into(),
            primary: RegionTarget::authored("loaded"),
            secondary: Some(RegionTarget::authored("loaded")),
            kind: runmat_analysis_core::ConnectorKind::TranslationalSpring {
                stiffness: runmat_analysis_core::Stiffness::new(1.0),
            },
        }];
        model.interfaces = vec![AnalysisInterface {
            interface_id: "contact".into(),
            primary_region_id: "loaded".into(),
            secondary_region_id: "loaded".into(),
            kind: runmat_analysis_core::AnalysisInterfaceKind::Contact(
                runmat_analysis_core::ContactInterfaceModel {
                    penalty_stiffness_scale: 1.0,
                    max_penetration_ratio: 0.01,
                    friction_coefficient: 0.0,
                },
            ),
        }];
        model.excitations = vec![ExcitationAction::Force(
            runmat_analysis_core::AnalysisExcitation {
                excitation_id: runmat_analysis_core::ExcitationId("force".into()),
                case_id: runmat_analysis_core::ExcitationCaseId("case".into()),
                target: ExcitationTarget::Region(RegionTarget::authored("loaded")),
                frame: CoordinateFrame::Global,
                direction: None,
                distribution: runmat_analysis_core::ExcitationDistribution::PreserveResultant,
                kinematics: runmat_analysis_core::ExcitationKinematics::FixedFrame,
                signal_id: runmat_analysis_core::SignalId("force".into()),
                active_step_ids: Vec::new(),
                combination_group: None,
                projection: runmat_analysis_core::ProjectionRequirement {
                    preserve_resultant: true,
                    preserve_moment: true,
                    relative_tolerance: 1.0e-12,
                },
            },
        )];
        model.initial_conditions = vec![runmat_analysis_core::InitialCondition::Temperature(
            runmat_analysis_core::InitialConditionValue {
                initial_condition_id: "temperature".into(),
                target: runmat_analysis_core::InitialConditionTarget::Region(RegionTarget::volume(
                    "loaded",
                )),
                frame: CoordinateFrame::Global,
                value: runmat_analysis_core::InitialValue::Uniform(
                    runmat_analysis_core::Temperature::new(293.15),
                ),
                applies_before_step_id: "solve".into(),
            },
        )];

        resolve_typed_region_targets(&mut model, &geometry, &aliases).expect("resolve aliases");

        assert_eq!(model.constraints[0].target().unwrap().region_id, "tip");
        assert_eq!(model.boundaries[0].target.region_id, "tip");
        assert_eq!(model.connectors[0].primary.region_id, "tip");
        assert_eq!(
            model.connectors[0].secondary.as_ref().unwrap().region_id,
            "tip"
        );
        assert_eq!(model.interfaces[0].primary_region_id, "tip");
        assert!(matches!(
            &model.excitations[0].excitation().target,
            ExcitationTarget::Region(target) if target.region_id == "tip"
        ));
        assert_eq!(
            model.initial_conditions[0]
                .region_target_mut()
                .unwrap()
                .region_id,
            "tip"
        );
    }

    #[test]
    fn structural_node_syntax_is_not_inferred_as_a_region() {
        let error = resolve_region_ref("node:2", &sample_geometry(), &BTreeMap::new())
            .expect_err("node syntax is not a region selector");
        assert!(error.contains("node:2"));
    }

    #[test]
    fn fea_document_resolves_canonical_meshing_settings() {
        let mesh: FeaMeshDocument = serde_yaml::from_str(
            "element_order: tet10\ntarget_edge_length_m: 0.02\nmaximum_chordal_deviation_m: 0.0002\nmaximum_grading_ratio: 1.5\nmaximum_elements: 42000\ndeterministic_seed: 17\n",
        )
        .unwrap();
        let settings = resolve_meshing_settings(Some(&mesh), AnalysisRunKind::LinearStatic)
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
        assert!(
            resolve_meshing_settings(None, AnalysisRunKind::LinearStatic,)
                .unwrap()
                .is_some()
        );
        assert!(resolve_meshing_settings(None, AnalysisRunKind::Modal,)
            .unwrap()
            .is_none());
    }

    #[test]
    fn fea_document_rejects_retired_mesh_fields() {
        for source in [
            "backend: auto",
            "kind: solid",
            "element: tetrahedron4",
            "template: analysis_ready",
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
            assert!(resolve_meshing_settings(Some(&mesh), AnalysisRunKind::LinearStatic,).is_err());
        }
    }
}
