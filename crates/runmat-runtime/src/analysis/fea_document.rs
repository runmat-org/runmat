use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use runmat_analysis_core::{
    AnalysisBoundary, AnalysisConnector, AnalysisConstraint, AnalysisInterface, AnalysisModel,
    AnalysisModelId, AnalysisSequence, BeamElementModel, BeamSectionModel, CfdDomain,
    CoordinateFrame, ElectroThermalDomain, ElectromagneticDomain, EvidenceConfidence,
    ExcitationAction, ExcitationCase, ExcitationTarget, FormulationRequirement, InitialCondition,
    MaterialAcousticModel, MaterialAssignment, MaterialElectricalModel, MaterialMechanicalModel,
    MaterialModel, MaterialPlasticModel, MaterialThermalModel, PhysicalSignal, RegionTarget,
    ResponseSpectrum, ShellElementModel, ShellSectionModel, StructuralElement,
    StructuralElementKind, StructuralModel, StructuralNode, ThermoMechanicalDomain,
};
use runmat_analysis_fea::ComputeBackend;
use runmat_geometry_core::{GeometryAsset, UnitSystem};
use runmat_geometry_io::GeometryImportOptions;
use runmat_meshing_core::{ElementOrder, MeshingRequestSettings};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{
    analysis_create_model_op, AnalysisCreateModelIntentSpec, AnalysisExecutionPlan,
    AnalysisExecutionStep, AnalysisModelTemplate, AnalysisSolverOptions, AnalysisStudySpec,
    AnalysisStudySweepSpec,
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
    execution: Vec<FeaExecutionDocument>,
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
    selector: FeaRegionSelector,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeaRegionSelector {
    Id(String),
    Tag(String),
    Name(String),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaRegionSelectorRepresentation {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    tag: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

impl Serialize for FeaRegionSelector {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(1))?;
        match self {
            Self::Id(value) => map.serialize_entry("id", value)?,
            Self::Tag(value) => map.serialize_entry("tag", value)?,
            Self::Name(value) => map.serialize_entry("name", value)?,
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for FeaRegionSelector {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let representation = FeaRegionSelectorRepresentation::deserialize(deserializer)?;
        match (representation.id, representation.tag, representation.name) {
            (Some(value), None, None) => Ok(Self::Id(value)),
            (None, Some(value), None) => Ok(Self::Tag(value)),
            (None, None, Some(value)) => Ok(Self::Name(value)),
            _ => Err(serde::de::Error::custom(
                "region selector must contain exactly one of `id`, `tag`, or `name`",
            )),
        }
    }
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

#[derive(Debug)]
struct FeaExecutionDocument {
    step: String,
    backend: ComputeBackend,
    requirement: FormulationRequirement,
    solver_options: Option<AnalysisSolverOptions>,
}

#[derive(Deserialize)]
struct FeaExecutionDocumentFields {
    step: String,
    #[serde(default = "default_backend")]
    backend: ComputeBackend,
    #[serde(default = "default_formulation_requirement")]
    requirement: FormulationRequirement,
}

impl<'de> Deserialize<'de> for FeaExecutionDocument {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        validate_execution_object(&value).map_err(serde::de::Error::custom)?;
        let solver_options =
            parse_execution_solver_options(&value).map_err(serde::de::Error::custom)?;
        let fields: FeaExecutionDocumentFields =
            serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(Self {
            step: fields.step,
            backend: fields.backend,
            requirement: fields.requirement,
            solver_options,
        })
    }
}

pub(super) fn validate_execution_object(value: &serde_json::Value) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| "execution entry must be an object".to_string())?;
    for key in object.keys() {
        if !matches!(
            key.as_str(),
            "step" | "backend" | "requirement" | "solver" | "options"
        ) {
            return Err(format!("unknown field `{key}` in execution entry"));
        }
    }
    if object.contains_key("solver") != object.contains_key("options") {
        return Err("execution `solver` and `options` must be provided together".into());
    }
    Ok(())
}

pub(super) fn parse_execution_solver_options(
    value: &serde_json::Value,
) -> Result<Option<AnalysisSolverOptions>, String> {
    validate_execution_object(value)?;
    let object = value
        .as_object()
        .expect("validated execution entry must be an object");
    let Some(solver) = object.get("solver") else {
        return Ok(None);
    };
    let mut tagged = serde_json::Map::new();
    tagged.insert("solver".into(), solver.clone());
    tagged.insert(
        "options".into(),
        object
            .get("options")
            .expect("validated solver must have options")
            .clone(),
    );
    serde_json::from_value(tagged.into())
        .map(Some)
        .map_err(|error| format!("invalid execution solver/options: {error}"))
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
    let raw = parse_raw_fea_document(input)?;
    match raw {
        RawFeaDocument::Study(study) => {
            let resolved = resolve_study(*study, base_dir).await?;
            Ok(FeaResolvedDocument::Study(Box::new(resolved.spec)))
        }
        RawFeaDocument::Sweep(sweep) => resolve_sweep(sweep, base_dir).await,
    }
}

fn parse_raw_fea_document(input: &str) -> Result<RawFeaDocument, String> {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(input)
        .map_err(|err| format!("failed to parse FEA YAML: {err}"))?;
    parse_raw_fea_value(yaml)
}

fn parse_raw_fea_value(yaml: serde_yaml::Value) -> Result<RawFeaDocument, String> {
    let canonical = serde_json::to_value(yaml)
        .map_err(|err| format!("FEA YAML is not JSON-compatible: {err}"))?;
    serde_json::from_value(canonical)
        .map_err(|err| format!("failed to parse canonical FEA document: {err}"))
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
    let execution = resolve_execution(&study.execution, &study.sequence)?;
    let meshing_settings = resolve_meshing_settings(study.mesh.as_ref(), &execution)?;
    let spec = AnalysisStudySpec {
        study_id: study.id,
        geometry,
        create_model_intent: intent,
        model,
        execution,
        meshing_settings,
        solver_mesh_artifact_path: None,
        meshing_evidence_artifact_path: None,
    };
    Ok(ResolvedStudyParts { spec })
}

fn resolve_meshing_settings(
    mesh: Option<&FeaMeshDocument>,
    execution: &AnalysisExecutionPlan,
) -> Result<Option<MeshingRequestSettings>, String> {
    let required = execution.steps.iter().any(|entry| {
        matches!(
            entry.solver_options,
            AnalysisSolverOptions::StructuralLinear(_)
        )
    });
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

fn resolve_execution(
    authored: &[FeaExecutionDocument],
    sequence: &AnalysisSequence,
) -> Result<AnalysisExecutionPlan, String> {
    let mut resolved = Vec::with_capacity(authored.len());
    for entry in authored {
        let step = sequence
            .steps
            .iter()
            .find(|step| step.step_id.as_str() == entry.step)
            .ok_or_else(|| {
                format!(
                    "execution references procedure step `{}`, which is absent from sequence `{}`",
                    entry.step, sequence.sequence_id
                )
            })?;
        let solver_options = match &entry.solver_options {
            Some(options) if options.supports(step) => options.clone(),
            Some(_) => {
                return Err(format!(
                    "solver options are incompatible with procedure step `{}` ({:?}/{:?}/{:?})",
                    entry.step, step.family, step.procedure, step.solution_space
                ));
            }
            None => AnalysisSolverOptions::defaults_for(step)?,
        };
        resolved.push(AnalysisExecutionStep {
            step_id: step.step_id.clone(),
            backend: entry.backend,
            requirement: entry.requirement,
            solver_options,
        });
    }
    let execution = AnalysisExecutionPlan { steps: resolved };
    execution.validate_against(sequence)?;
    Ok(execution)
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
        resolve_region_target(&mut interface.primary, geometry, aliases, &mut failure);
        resolve_region_target(&mut interface.secondary, geometry, aliases, &mut failure);
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
    for step in &mut model.sequence.steps {
        for observation in &mut step.observations {
            let Some(spec) = observation.spec_mut() else {
                continue;
            };
            if let runmat_analysis_core::ObservationTarget::Spatial(ExcitationTarget::Region(
                target,
            )) = &mut spec.target
            {
                resolve_region_target(target, geometry, aliases, &mut failure);
            }
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
    require_region_id(reference, geometry)
}

fn resolve_region_selector(
    selector: &FeaRegionSelector,
    geometry: &GeometryAsset,
) -> Result<String, String> {
    match selector {
        FeaRegionSelector::Id(region_id) => require_region_id(region_id, geometry),
        FeaRegionSelector::Tag(tag) => require_unique_region_match(
            "tag",
            tag,
            geometry
                .regions
                .iter()
                .filter(|region| region.tag.as_deref() == Some(tag))
                .map(|region| region.region_id.as_str()),
        ),
        FeaRegionSelector::Name(name) => require_unique_region_match(
            "name",
            name,
            geometry
                .regions
                .iter()
                .filter(|region| region.name == *name)
                .map(|region| region.region_id.as_str()),
        ),
    }
}

fn require_unique_region_match<'a>(
    selector_kind: &str,
    selector_value: &str,
    matches: impl Iterator<Item = &'a str>,
) -> Result<String, String> {
    let mut region_ids = matches.map(str::to_string).collect::<Vec<_>>();
    region_ids.sort();
    match region_ids.as_slice() {
        [region_id] => Ok(region_id.clone()),
        [] => Err(format!(
            "region {selector_kind} `{selector_value}` was not found in geometry"
        )),
        _ => Err(format!(
            "region {selector_kind} `{selector_value}` is ambiguous: matched {} regions [{}]",
            region_ids.len(),
            region_ids.join(", ")
        )),
    }
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

fn default_formulation_requirement() -> FormulationRequirement {
    FormulationRequirement::Production
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
    use runmat_analysis_core::{
        AnalysisProcedure, Force, ProcedureStep, Signal, SignalApproximationEvidence, SignalId,
        SignalPrecision, SignalProvenance, SignalRepresentation, SignalShape, SignalValue,
    };
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
            regions: vec![
                Region {
                    region_id: "tip".to_string(),
                    name: "Tip".to_string(),
                    tag: Some("tip".to_string()),
                    cad_ownership: None,
                },
                Region {
                    region_id: "base".to_string(),
                    name: "Base".to_string(),
                    tag: Some("base".to_string()),
                    cad_ownership: None,
                },
            ],
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
            canonical_yaml(AnalysisModelTemplate::LinearStaticStructural),
        );
        model.insert("defaults".into(), "none".into());
        document.insert("model".into(), model.into());
        document.insert(
            "execution".into(),
            serde_yaml::from_str("- step: solve\n  backend: cpu\n").expect("execution YAML"),
        );
        document.insert(
            "sequence".into(),
            canonical_yaml(AnalysisSequence::from_steps(
                "typed-sequence",
                vec![runmat_analysis_core::ProcedureStep::structural_static(
                    "solve",
                )],
            )),
        );
        document.into()
    }

    fn documentation_example_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/fea/examples")
            .join(name)
    }

    #[test]
    fn canonical_documentation_examples_parse_and_resolve() {
        for name in [
            "structural-static-experimental.fea",
            "structural-transient-experimental.fea",
            "modal-response-spectrum-experimental.fea",
            "acoustic-production-rejected.fea",
        ] {
            let path = documentation_example_path(name);
            let resolved = pollster::block_on(load_fea_document_from_path_async(&path))
                .unwrap_or_else(|error| panic!("{} must resolve: {error}", path.display()));
            let FeaResolvedDocument::Study(study) = resolved else {
                panic!("{} must contain a study", path.display());
            };
            assert_eq!(
                study.execution.steps.len(),
                study.model.as_ref().unwrap().sequence.steps.len()
            );
        }
    }

    #[test]
    fn documentation_examples_expose_formulation_requirement_truth() {
        for name in [
            "structural-static-experimental.fea",
            "structural-transient-experimental.fea",
            "modal-response-spectrum-experimental.fea",
        ] {
            let path = documentation_example_path(name);
            let FeaResolvedDocument::Study(study) =
                pollster::block_on(load_fea_document_from_path_async(&path)).unwrap()
            else {
                panic!("{} must contain a study", path.display());
            };
            assert!(study
                .execution
                .steps
                .iter()
                .all(|step| step.requirement == FormulationRequirement::Experimental));
        }

        let path = documentation_example_path("acoustic-production-rejected.fea");
        let FeaResolvedDocument::Study(study) =
            pollster::block_on(load_fea_document_from_path_async(&path)).unwrap()
        else {
            panic!("{} must contain a study", path.display());
        };
        assert!(study
            .execution
            .steps
            .iter()
            .all(|step| step.requirement == FormulationRequirement::Production));
    }

    fn canonical_yaml(value: impl Serialize) -> serde_yaml::Value {
        let json = serde_json::to_value(value).expect("JSON-compatible contract");
        serde_yaml::to_value(json).expect("canonical YAML value")
    }

    #[test]
    fn typed_document_fields_deserialize_without_legacy_translation() {
        let mut document = minimal_study_yaml();
        let mapping = document.as_mapping_mut().expect("study mapping");
        mapping.insert(
            "constraints".into(),
            canonical_yaml(vec![AnalysisConstraint::Structural(
                runmat_analysis_core::StructuralConstraint {
                    constraint_id: "fixed".into(),
                    frame: CoordinateFrame::Global,
                    kind: runmat_analysis_core::StructuralConstraintKind::Fixed {
                        target: RegionTarget::boundary("tip"),
                    },
                },
            )]),
        );
        mapping.insert(
            "signals".into(),
            canonical_yaml(vec![PhysicalSignal::Force(Signal {
                signal_id: SignalId("push".into()),
                shape: SignalShape::Scalar,
                frame: Some(CoordinateFrame::Global),
                representation: SignalRepresentation::Constant(SignalValue::Scalar(Force::new(
                    10.0,
                ))),
                precision: SignalPrecision::Real64,
                evidence: SignalApproximationEvidence {
                    max_absolute_error: None,
                    max_relative_error: None,
                    derivative_max_absolute_error: None,
                    sample_count: 1,
                },
                provenance: SignalProvenance::Authored,
            })]),
        );

        let parsed = parse_raw_fea_value(document).expect("typed document");
        let RawFeaDocument::Study(study) = parsed else {
            panic!("expected study")
        };
        assert_eq!(study.constraints[0].constraint_id(), "fixed");
        assert_eq!(study.signals[0].signal_id().0, "push");
        assert_eq!(study.sequence.steps[0].step_id.as_str(), "solve");
    }

    #[test]
    fn template_scaffold_retains_the_authored_sequence() {
        let mut document = minimal_study_yaml();
        document
            .as_mapping_mut()
            .expect("study mapping")
            .get_mut("model")
            .expect("model")
            .as_mapping_mut()
            .expect("model mapping")
            .insert("defaults".into(), "template_scaffold".into());

        let parsed = parse_raw_fea_value(document).expect("study document");
        let RawFeaDocument::Study(study) = parsed else {
            panic!("expected study")
        };

        assert!(!study.sequence.steps.is_empty());
        assert_eq!(study.model.defaults, FeaModelDefaultsMode::TemplateScaffold);
        let intent = AnalysisCreateModelIntentSpec {
            model_id: "scaffold-model".into(),
            template: study.model.template,
        };
        let model = resolve_model(&study, &sample_geometry(), &intent)
            .expect("resolve scaffold model")
            .expect("resolved model");
        assert_eq!(model.sequence, study.sequence);
    }

    #[test]
    fn retired_document_keys_and_untyped_execution_are_rejected() {
        for key in [
            "boundary_conditions",
            "loads",
            "steps",
            "outputs",
            "nodes",
            "elements",
            "sections",
            "run",
        ] {
            let mut document = minimal_study_yaml();
            document
                .as_mapping_mut()
                .expect("study mapping")
                .insert(key.into(), serde_yaml::Value::Sequence(Vec::new()));
            let error = parse_raw_fea_value(document).expect_err("retired key must fail");
            assert!(error.to_string().contains("unknown field"), "{error}");
        }

        let error = serde_yaml::from_str::<FeaExecutionDocument>(
            "step: solve\nbackend: cpu\nkind: linear_static\n",
        )
        .expect_err("execution.kind must fail");
        assert!(error.to_string().contains("unknown field"), "{error}");
        let error = serde_yaml::from_str::<FeaExecutionDocument>(
            "step: solve\nbackend: cpu\noptions: {}\n",
        )
        .expect_err("untyped solver options must fail");
        assert!(
            error
                .to_string()
                .contains("`solver` and `options` must be provided together"),
            "{error}"
        );
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
    fn execution_identity_is_resolved_from_each_sequence_step() {
        let sequence = AnalysisSequence::from_steps(
            "sequence",
            vec![runmat_analysis_core::ProcedureStep::structural_modal(
                "modes",
            )],
        );
        let execution = resolve_execution(
            &[FeaExecutionDocument {
                step: "modes".into(),
                backend: ComputeBackend::Cpu,
                requirement: FormulationRequirement::Production,
                solver_options: None,
            }],
            &sequence,
        )
        .unwrap();
        assert_eq!(
            execution
                .procedure_step(&sequence, &execution.steps[0])
                .unwrap()
                .procedure,
            AnalysisProcedure::ModalExtraction
        );

        let multiple = AnalysisSequence::from_steps(
            "sequence",
            vec![
                runmat_analysis_core::ProcedureStep::structural_static("static"),
                runmat_analysis_core::ProcedureStep::structural_modal("modes"),
            ],
        );
        let execution = resolve_execution(
            &[
                FeaExecutionDocument {
                    step: "static".into(),
                    backend: ComputeBackend::Cpu,
                    requirement: FormulationRequirement::Production,
                    solver_options: None,
                },
                FeaExecutionDocument {
                    step: "modes".into(),
                    backend: ComputeBackend::Gpu,
                    requirement: FormulationRequirement::Production,
                    solver_options: None,
                },
            ],
            &multiple,
        )
        .expect("every sequence step has a typed execution entry");
        assert_eq!(execution.steps.len(), 2);
        assert_eq!(execution.steps[1].backend, ComputeBackend::Gpu);
    }

    #[test]
    fn typed_solver_options_must_match_the_sequence() {
        let execution = FeaExecutionDocument {
            step: "modes".into(),
            backend: ComputeBackend::Cpu,
            requirement: FormulationRequirement::Production,
            solver_options: Some(AnalysisSolverOptions::StructuralLinear(Default::default())),
        };
        let sequence = AnalysisSequence::from_steps(
            "sequence",
            vec![runmat_analysis_core::ProcedureStep::structural_modal(
                "modes",
            )],
        );
        assert!(resolve_execution(&[execution], &sequence)
            .expect_err("mismatched solver options must fail")
            .contains("incompatible"));
    }

    #[test]
    fn region_aliases_resolve_across_typed_model_targets() {
        let geometry = sample_geometry();
        let mut aliases = BTreeMap::new();
        aliases.insert(
            "loaded".into(),
            FeaRegionDocument {
                selector: FeaRegionSelector::Tag("tip".into()),
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
            primary: RegionTarget::boundary("loaded"),
            secondary: RegionTarget::boundary("base"),
            kind: runmat_analysis_core::AnalysisInterfaceKind::Contact(
                runmat_analysis_core::ContactInterfaceModel {
                    friction: runmat_analysis_core::ContactFriction::Frictionless,
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
        model.sequence = AnalysisSequence::from_steps(
            "sequence",
            vec![runmat_analysis_core::ProcedureStep::structural_static(
                "solve",
            )],
        );
        model.sequence.steps[0]
            .observations
            .push(runmat_analysis_core::ObservationRequest::Probe(
                runmat_analysis_core::ObservationSpec {
                    observation_id: runmat_analysis_core::ObservationId("temperature".into()),
                    quantity: runmat_analysis_core::ObservationQuantity::Temperature,
                    value_shape: runmat_analysis_core::ObservationValueShape::Scalar,
                    target: runmat_analysis_core::ObservationTarget::Spatial(
                        ExcitationTarget::Region(RegionTarget::boundary("loaded")),
                    ),
                    source_location: runmat_analysis_core::FieldLocation::Node,
                    output_location: runmat_analysis_core::FieldLocation::Node,
                    recovery: runmat_analysis_core::RecoveryPolicy::Native,
                    component: runmat_analysis_core::ObservationComponent::All,
                    frame: CoordinateFrame::Global,
                    sign_convention: runmat_analysis_core::SignConvention::Native,
                    complex_representation: runmat_analysis_core::ComplexRepresentation::Real,
                    sampling: runmat_analysis_core::SamplingPolicy::Final,
                    reduction: None,
                    storage: runmat_analysis_core::ObservationStoragePolicy {
                        precision: runmat_analysis_core::SignalPrecision::Real64,
                        compression: runmat_analysis_core::CompressionPolicy::None,
                        retention: runmat_analysis_core::RetentionPolicy::Canonical,
                        partial_access:
                            runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks,
                    },
                    purpose: runmat_analysis_core::ObservationPurpose::Visualization,
                    cost: runmat_analysis_core::ObservationCostEstimate {
                        operation_count: 0,
                        peak_resident_bytes: 0,
                        stored_bytes: 0,
                        transfer_bytes: 0,
                        collective_count: 0,
                    },
                },
            ));

        resolve_typed_region_targets(&mut model, &geometry, &aliases).expect("resolve aliases");

        assert_eq!(model.constraints[0].target().unwrap().region_id, "tip");
        assert_eq!(model.boundaries[0].target.region_id, "tip");
        assert_eq!(model.connectors[0].primary.region_id, "tip");
        assert_eq!(
            model.connectors[0].secondary.as_ref().unwrap().region_id,
            "tip"
        );
        assert_eq!(model.interfaces[0].primary.region_id, "tip");
        assert_eq!(model.interfaces[0].secondary.region_id, "base");
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
        assert!(matches!(
            model.sequence.steps[0].observations[0]
                .spec()
                .map(|spec| &spec.target),
            Some(runmat_analysis_core::ObservationTarget::Spatial(
                ExcitationTarget::Region(target)
            )) if target.region_id == "tip"
        ));
    }

    #[test]
    fn structural_node_syntax_is_not_inferred_as_a_region() {
        let error = resolve_region_ref("node:2", &sample_geometry(), &BTreeMap::new())
            .expect_err("node syntax is not a region selector");
        assert!(error.contains("node:2"));
    }

    #[test]
    fn region_selector_requires_a_typed_selector_variant() {
        let typed: FeaRegionDocument =
            serde_yaml::from_str("selector:\n  tag: tip\n").expect("typed selector");
        assert_eq!(typed.selector, FeaRegionSelector::Tag("tip".into()));

        for source in [
            "selector: tip\n",
            "selector: tag:tip\n",
            "selector:\n  region: tip\n",
            "selector:\n  id: tip\n  tag: tip\n",
        ] {
            serde_yaml::from_str::<FeaRegionDocument>(source)
                .expect_err("legacy region selector syntax must be rejected");
        }
    }

    #[test]
    fn direct_region_references_do_not_parse_legacy_prefixes() {
        let error = resolve_region_ref("tag:tip", &sample_geometry(), &BTreeMap::new())
            .expect_err("legacy prefixes are not direct region ids");
        assert!(error.contains("tag:tip"));
    }

    #[test]
    fn tag_and_name_selectors_reject_ambiguous_geometry_matches() {
        let mut geometry = sample_geometry();
        geometry.regions.push(Region {
            region_id: "tip_duplicate".into(),
            name: "Tip".into(),
            tag: Some("tip".into()),
            cad_ownership: None,
        });

        for selector in [
            FeaRegionSelector::Tag("tip".into()),
            FeaRegionSelector::Name("Tip".into()),
        ] {
            let error = resolve_region_selector(&selector, &geometry)
                .expect_err("non-unique selector must be rejected");
            assert!(error.contains("matched 2 regions"), "{error}");
            assert!(error.contains("tip"), "{error}");
            assert!(error.contains("tip_duplicate"), "{error}");
        }
    }

    #[test]
    fn fea_document_resolves_canonical_meshing_settings() {
        let mesh: FeaMeshDocument = serde_yaml::from_str(
            "element_order: tet10\ntarget_edge_length_m: 0.02\nmaximum_chordal_deviation_m: 0.0002\nmaximum_grading_ratio: 1.5\nmaximum_elements: 42000\ndeterministic_seed: 17\n",
        )
        .unwrap();
        let step = runmat_analysis_core::ProcedureStep::structural_static("solve");
        let execution = execution_for(step);
        let settings = resolve_meshing_settings(Some(&mesh), &execution)
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
            &execution_for(ProcedureStep::structural_static("s"))
        )
        .unwrap()
        .is_some());
        assert!(resolve_meshing_settings(
            None,
            &execution_for(ProcedureStep::structural_modal("m"))
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
            assert!(resolve_meshing_settings(
                Some(&mesh),
                &execution_for(ProcedureStep::structural_static("s")),
            )
            .is_err());
        }
    }

    fn execution_for(step: ProcedureStep) -> AnalysisExecutionPlan {
        AnalysisExecutionPlan::for_sequence(
            &AnalysisSequence::from_steps("test", vec![step]),
            ComputeBackend::Cpu,
        )
        .unwrap()
    }
}
