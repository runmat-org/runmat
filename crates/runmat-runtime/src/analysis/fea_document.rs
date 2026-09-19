use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use runmat_analysis_core::{
    AnalysisBoundary, AnalysisConnector, AnalysisConstraint, AnalysisInterface, AnalysisModel,
    AnalysisModelId, AnalysisSequence, BeamElementModel, BeamSectionModel, CfdDomain,
    CoordinateFrame, ElectroThermalDomain, ElectromagneticDomain, EvidenceConfidence,
    ExcitationAction, ExcitationCase, ExcitationTarget, FormulationRequirement, InitialCondition,
    MaterialAcousticModel, MaterialAssignment, MaterialElectricalModel, MaterialMechanicalModel,
    MaterialModel, MaterialPlasticModel, MaterialThermalModel, PhysicalSignal,
    RandomExcitationSpectrum, RegionTarget, ResponseSpectrum, ShellElementModel, ShellSectionModel,
    StructuralElement, StructuralElementKind, StructuralModel, StructuralNode, ThermalDomain,
    ThermoMechanicalDomain,
};
use runmat_analysis_fea::ComputeBackend;
use runmat_geometry_core::{GeometryAsset, UnitSystem};
use runmat_geometry_io::GeometryImportOptions;
use runmat_meshing_core::{ElementOrder, MeshingRequestSettings};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::contracts::validate_immutable_value_inventory;
use super::{
    AnalysisCreateModelIntentSpec, AnalysisExecutionPlan, AnalysisExecutionStep,
    AnalysisImmutableValueSpec, AnalysisModelSourceEvidence, AnalysisModelSourceFileEvidence,
    AnalysisModelSourceKind, AnalysisModelTemplate, AnalysisSolverOptions, AnalysisStudySpec,
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
    Model(Box<FeaStandaloneModelDocument>),
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
    immutable_values: Vec<AnalysisImmutableValueSpec>,
    sequence: AnalysisSequence,
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
    #[serde(default)]
    frame: Option<CoordinateFrame>,
    source: FeaModelSourceDocument,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaStandaloneModelDocument {
    version: u32,
    source: FeaModelSourceDocument,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FeaModelSourceDocument {
    Inline(Box<FeaModelDataDocument>),
    File(FeaModelFileSourceDocument),
    Compose(Vec<FeaModelSourceDocument>),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaModelFileSourceDocument {
    path: PathBuf,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FeaModelDataDocument {
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
    #[serde(default)]
    random_excitations: Vec<RandomExcitationSpectrum>,
    #[serde(default)]
    domains: FeaDomainsDocument,
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

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
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

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FeaMaterialDocument {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    mass_density_kg_per_m3: Option<f64>,
    #[serde(default)]
    mechanical: Option<MaterialMechanicalModel>,
    #[serde(default)]
    damping: Option<runmat_analysis_core::MechanicalDampingModel>,
    #[serde(default)]
    thermal: Option<MaterialThermalModel>,
    #[serde(default)]
    acoustic: Option<MaterialAcousticModel>,
    #[serde(default)]
    electrical: Option<MaterialElectricalModel>,
    #[serde(default)]
    plastic: Option<MaterialPlasticModel>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FeaMaterialAssignmentDocument {
    target: FeaPropertyRegionTargetDocument,
    material: String,
    #[serde(default)]
    expected_material: Option<String>,
    #[serde(default = "default_assignment_confidence")]
    confidence: EvidenceConfidence,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum FeaPropertyRegionTargetDocument {
    Beam(FeaPropertyRegionSelectionDocument),
    Shell(FeaPropertyRegionSelectionDocument),
    Solid(FeaPropertyRegionSelectionDocument),
    Cell(FeaPropertyRegionSelectionDocument),
    AllBeams,
    AllShells,
    AllSolids,
    AllCells,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FeaPropertyRegionSelectionDocument {
    region: String,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FeaStructuralDocument {
    #[serde(default)]
    nodes: Vec<FeaStructuralNodeDocument>,
    #[serde(default)]
    elements: Vec<FeaStructuralElementDocument>,
    #[serde(default)]
    sections: Vec<FeaStructuralSectionDocument>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FeaStructuralNodeDocument {
    id: u32,
    coordinates_m: [f64; 3],
    #[serde(default)]
    solid_attachment: Option<FeaConformalSolidAttachmentDocument>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FeaConformalSolidAttachmentDocument {
    geometry_vertex: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum FeaStructuralElementType {
    Beam,
    Shell,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
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

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FeaDomainsDocument {
    #[serde(default)]
    thermal: Option<ThermalDomain>,
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

#[derive(Debug)]
struct ResolvedModelSource {
    data: FeaModelDataDocument,
    kind: AnalysisModelSourceKind,
    files: Vec<AnalysisModelSourceFileEvidence>,
}

impl ResolvedModelSource {
    fn evidence(&self) -> Result<AnalysisModelSourceEvidence, String> {
        let encoded = serde_json::to_vec(&self.data)
            .map_err(|error| format!("failed to encode resolved model source: {error}"))?;
        Ok(AnalysisModelSourceEvidence {
            kind: self.kind,
            resolved_model_digest: runmat_execution::Digest::sha256(encoded),
            files: self.files.clone(),
        })
    }
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
        RawFeaDocument::Model(_) => Err(
            "a model-source document cannot be run directly; reference it from a study model source"
                .into(),
        ),
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

async fn resolve_model_source(
    source: &FeaModelSourceDocument,
    base_dir: &Path,
    source_root: &Path,
    logical_dir: &Path,
    stack: &mut Vec<PathBuf>,
) -> Result<ResolvedModelSource, String> {
    match source {
        FeaModelSourceDocument::Inline(data) => Ok(ResolvedModelSource {
            data: data.as_ref().clone(),
            kind: AnalysisModelSourceKind::Inline,
            files: Vec::new(),
        }),
        FeaModelSourceDocument::File(file) => {
            if file.path.as_os_str().is_empty()
                || file.path.is_absolute()
                || file.path.components().any(|component| {
                    matches!(
                        component,
                        std::path::Component::ParentDir
                            | std::path::Component::RootDir
                            | std::path::Component::Prefix(_)
                    )
                })
            {
                return Err(
                    "model file source path must be a non-empty, non-traversing relative path"
                        .into(),
                );
            }
            let path = resolve_document_path(base_dir, &file.path);
            if !is_fea_file_path(&path) {
                return Err(format!(
                    "model file source must use the .fea extension: {}",
                    path.display()
                ));
            }
            let canonical =
                runmat_filesystem::canonicalize_async(&path)
                    .await
                    .map_err(|error| {
                        format!(
                            "failed to resolve model file source {}: {error}",
                            path.display()
                        )
                    })?;
            if !canonical.starts_with(source_root) {
                return Err(format!(
                    "model file source resolves outside the study source tree: {}",
                    file.path.display()
                ));
            }
            if let Some(index) = stack.iter().position(|entry| entry == &canonical) {
                let mut cycle = stack[index..]
                    .iter()
                    .map(|entry| entry.display().to_string())
                    .collect::<Vec<_>>();
                cycle.push(canonical.display().to_string());
                return Err(format!("model source import cycle: {}", cycle.join(" -> ")));
            }
            stack.push(canonical.clone());
            let input = runmat_filesystem::read_to_string_async(&canonical)
                .await
                .map_err(|error| {
                    format!(
                        "failed to read model file source {}: {error}",
                        canonical.display()
                    )
                })?;
            let logical_path = logical_dir.join(&file.path);
            let file_evidence = AnalysisModelSourceFileEvidence {
                path: logical_path.to_string_lossy().replace('\\', "/"),
                content_digest: runmat_execution::Digest::sha256(input.as_bytes()),
            };
            let document = match parse_raw_fea_document(&input)? {
                RawFeaDocument::Model(document) => document,
                RawFeaDocument::Study(_) | RawFeaDocument::Sweep(_) => {
                    return Err(format!(
                        "model file source {} must declare `kind: model`",
                        canonical.display()
                    ))
                }
            };
            validate_version(document.version)?;
            let nested_base = canonical.parent().unwrap_or_else(|| Path::new(""));
            let nested_logical_dir = logical_path.parent().unwrap_or_else(|| Path::new(""));
            let resolved = Box::pin(resolve_model_source(
                &document.source,
                nested_base,
                source_root,
                nested_logical_dir,
                stack,
            ))
            .await;
            stack.pop();
            let mut resolved = resolved?;
            resolved.files.insert(0, file_evidence);
            resolved.kind = AnalysisModelSourceKind::File;
            Ok(resolved)
        }
        FeaModelSourceDocument::Compose(sources) => {
            if sources.is_empty() {
                return Err("composed model source requires at least one source".into());
            }
            let mut composed = FeaModelDataDocument::default();
            let mut files = Vec::new();
            for source in sources {
                let next = Box::pin(resolve_model_source(
                    source,
                    base_dir,
                    source_root,
                    logical_dir,
                    stack,
                ))
                .await?;
                merge_model_data(&mut composed, next.data)?;
                files.extend(next.files);
            }
            Ok(ResolvedModelSource {
                data: composed,
                kind: AnalysisModelSourceKind::Compose,
                files,
            })
        }
    }
}

fn merge_model_data(
    target: &mut FeaModelDataDocument,
    incoming: FeaModelDataDocument,
) -> Result<(), String> {
    merge_map(&mut target.regions, incoming.regions, "region")?;
    merge_map(&mut target.materials, incoming.materials, "material")?;
    merge_assignments(
        &mut target.material_assignments,
        incoming.material_assignments,
    )?;
    merge_structural(&mut target.structural, incoming.structural)?;
    merge_named(
        &mut target.constraints,
        incoming.constraints,
        |value| value.constraint_id().to_string(),
        "constraint",
    )?;
    merge_named(
        &mut target.boundaries,
        incoming.boundaries,
        |value| value.boundary_id.clone(),
        "boundary",
    )?;
    merge_named(
        &mut target.connectors,
        incoming.connectors,
        |value| value.connector_id.clone(),
        "connector",
    )?;
    merge_named(
        &mut target.interfaces,
        incoming.interfaces,
        |value| value.interface_id.clone(),
        "interface",
    )?;
    merge_named(
        &mut target.signals,
        incoming.signals,
        |value| value.signal_id().0.clone(),
        "signal",
    )?;
    merge_named(
        &mut target.excitation_cases,
        incoming.excitation_cases,
        |value| value.case_id.0.clone(),
        "excitation case",
    )?;
    merge_named(
        &mut target.excitations,
        incoming.excitations,
        |value| value.excitation().excitation_id.0.clone(),
        "excitation",
    )?;
    merge_named(
        &mut target.initial_conditions,
        incoming.initial_conditions,
        |value| value.initial_condition_id().to_string(),
        "initial condition",
    )?;
    merge_named(
        &mut target.response_spectra,
        incoming.response_spectra,
        |value| value.spectrum_id.0.clone(),
        "response spectrum",
    )?;
    merge_named(
        &mut target.random_excitations,
        incoming.random_excitations,
        |value| value.excitation_id.0.clone(),
        "random excitation",
    )?;
    merge_domains(&mut target.domains, incoming.domains)
}

fn merge_map<T: PartialEq>(
    target: &mut BTreeMap<String, T>,
    incoming: BTreeMap<String, T>,
    kind: &str,
) -> Result<(), String> {
    for (id, value) in incoming {
        match target.get(&id) {
            Some(existing) if existing != &value => {
                return Err(format!(
                    "conflicting {kind} definition `{id}` in model sources"
                ))
            }
            Some(_) => {}
            None => {
                target.insert(id, value);
            }
        }
    }
    Ok(())
}

fn merge_named<T: PartialEq>(
    target: &mut Vec<T>,
    incoming: Vec<T>,
    key: impl Fn(&T) -> String,
    kind: &str,
) -> Result<(), String> {
    for value in incoming {
        let id = key(&value);
        match target.iter().find(|existing| key(existing) == id) {
            Some(existing) if existing != &value => {
                return Err(format!(
                    "conflicting {kind} definition `{id}` in model sources"
                ))
            }
            Some(_) => {}
            None => target.push(value),
        }
    }
    Ok(())
}

fn merge_assignments(
    target: &mut Vec<FeaMaterialAssignmentDocument>,
    incoming: Vec<FeaMaterialAssignmentDocument>,
) -> Result<(), String> {
    for value in incoming {
        match target
            .iter()
            .find(|existing| existing.target == value.target)
        {
            Some(existing) if existing != &value => {
                return Err(
                    "conflicting material assignments target the same property region".into(),
                )
            }
            Some(_) => {}
            None => target.push(value),
        }
    }
    Ok(())
}

fn merge_structural(
    target: &mut Option<FeaStructuralDocument>,
    incoming: Option<FeaStructuralDocument>,
) -> Result<(), String> {
    let Some(incoming) = incoming else {
        return Ok(());
    };
    let target = target.get_or_insert_with(FeaStructuralDocument::default);
    merge_named(
        &mut target.nodes,
        incoming.nodes,
        |value| value.id.to_string(),
        "structural node",
    )?;
    merge_named(
        &mut target.elements,
        incoming.elements,
        |value| value.id.clone(),
        "structural element",
    )?;
    merge_named(
        &mut target.sections,
        incoming.sections,
        |value| value.id.clone(),
        "structural section",
    )
}

fn merge_domains(
    target: &mut FeaDomainsDocument,
    incoming: FeaDomainsDocument,
) -> Result<(), String> {
    merge_optional(&mut target.thermal, incoming.thermal, "thermal domain")?;
    merge_optional(
        &mut target.thermo_mechanical,
        incoming.thermo_mechanical,
        "thermo-mechanical domain",
    )?;
    merge_optional(
        &mut target.electro_thermal,
        incoming.electro_thermal,
        "electro-thermal domain",
    )?;
    merge_optional(
        &mut target.electromagnetic,
        incoming.electromagnetic,
        "electromagnetic domain",
    )?;
    merge_optional(&mut target.cfd, incoming.cfd, "CFD domain")
}

fn merge_optional<T: PartialEq>(
    target: &mut Option<T>,
    incoming: Option<T>,
    kind: &str,
) -> Result<(), String> {
    match (&*target, incoming) {
        (_, None) => Ok(()),
        (Some(existing), Some(value)) if existing != &value => {
            Err(format!("conflicting {kind} definitions in model sources"))
        }
        (Some(_), Some(_)) => Ok(()),
        (None, Some(value)) => {
            *target = Some(value);
            Ok(())
        }
    }
}

fn model_template_for_sequence(
    sequence: &AnalysisSequence,
) -> Result<AnalysisModelTemplate, String> {
    use runmat_analysis_core::{
        AnalysisProcedure as Procedure, CoupledPhysicsFamily, PhysicsFamily,
    };

    let step = sequence
        .steps
        .first()
        .ok_or_else(|| "FEA study sequence must contain at least one procedure step".to_string())?;
    Ok(match (step.family, step.procedure) {
        (
            PhysicsFamily::Structural,
            Procedure::HarmonicResponse | Procedure::PrestressedHarmonicResponse,
        ) => AnalysisModelTemplate::StructuralHarmonic,
        (PhysicsFamily::Structural, Procedure::TransientResponse) => {
            AnalysisModelTemplate::TransientStructural
        }
        (PhysicsFamily::Structural, Procedure::Nonlinear | Procedure::Adaptive) => {
            AnalysisModelTemplate::NonlinearStructural
        }
        (PhysicsFamily::Structural, Procedure::Static) => {
            AnalysisModelTemplate::LinearStaticStructural
        }
        (PhysicsFamily::Structural, _) => AnalysisModelTemplate::ModalStructural,
        (PhysicsFamily::Thermal, Procedure::TransientResponse) => {
            AnalysisModelTemplate::ThermalTransient
        }
        (PhysicsFamily::Thermal, _) => AnalysisModelTemplate::ThermalSteady,
        (PhysicsFamily::Electromagnetic, _) => AnalysisModelTemplate::ElectromagneticHarmonic,
        (PhysicsFamily::Acoustic, _) => AnalysisModelTemplate::AcousticHarmonic,
        (PhysicsFamily::Fluid, Procedure::TransientResponse) => AnalysisModelTemplate::CfdTransient,
        (PhysicsFamily::Fluid, _) => AnalysisModelTemplate::CfdSteadyState,
        (PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical), _) => {
            AnalysisModelTemplate::ThermoMechanicalCoupled
        }
        (PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer), _) => {
            AnalysisModelTemplate::ChtCoupled
        }
        (PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction), _) => {
            AnalysisModelTemplate::FsiCoupled
        }
        (PhysicsFamily::Coupled(CoupledPhysicsFamily::Electrothermal), _) => {
            return Err("electro-thermal study execution is not supported yet".into())
        }
    })
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
    let root_input = if base_dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        base_dir
    };
    let source_root = runmat_filesystem::canonicalize_async(root_input)
        .await
        .map_err(|error| format!("failed to resolve FEA document directory: {error}"))?;
    let model_source = resolve_model_source(
        &study.model.source,
        base_dir,
        &source_root,
        Path::new(""),
        &mut Vec::new(),
    )
    .await?;
    validate_immutable_value_inventory(&study.immutable_values, &model_source.data.signals)?;

    let geometry = load_geometry(&study.geometry, base_dir).await?;
    let model_id = study
        .model
        .id
        .clone()
        .unwrap_or_else(|| format!("{}_model", sanitize_id(&study.id)));
    let intent = AnalysisCreateModelIntentSpec {
        model_id: model_id.clone(),
        template: model_template_for_sequence(&study.sequence)?,
    };
    let model = resolve_model(&study, &model_source.data, &geometry, &intent)?;
    let execution = resolve_execution(&study.execution, &study.sequence)?;
    let meshing_settings = resolve_meshing_settings(study.mesh.as_ref(), &execution)?;
    let spec = AnalysisStudySpec {
        study_id: study.id,
        geometry,
        create_model_intent: intent,
        model_source_evidence: Some(model_source.evidence()?),
        model,
        execution,
        meshing_settings,
        immutable_values: study.immutable_values,
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
                | AnalysisSolverOptions::StructuralFrequency(_)
                | AnalysisSolverOptions::StructuralResponseSpectrum(_)
                | AnalysisSolverOptions::StructuralRandomResponse(_)
                | AnalysisSolverOptions::StructuralBuckling(_)
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
    data: &FeaModelDataDocument,
    geometry: &GeometryAsset,
    intent: &AnalysisCreateModelIntentSpec,
) -> Result<Option<AnalysisModel>, String> {
    let mut model = empty_model(intent.model_id.clone(), geometry);

    if let Some(frame) = &study.model.frame {
        model.frame = frame.clone();
    }
    if !data.materials.is_empty() {
        model.materials = data
            .materials
            .iter()
            .map(|(id, material)| resolve_material(id, material))
            .collect();
    }
    if !data.material_assignments.is_empty() {
        model.material_assignments = data
            .material_assignments
            .iter()
            .map(|assignment| resolve_material_assignment(assignment, geometry, &data.regions))
            .collect::<Result<Vec<_>, _>>()?;
    }
    if has_structural_model_data(data) {
        model.structural = Some(resolve_structural_model(data, geometry)?);
    }
    model.constraints = data.constraints.clone();
    model.boundaries = data.boundaries.clone();
    model.connectors = data.connectors.clone();
    model.interfaces = data.interfaces.clone();
    model.signals = data.signals.clone();
    model.excitation_cases = data.excitation_cases.clone();
    model.excitations = data.excitations.clone();
    model.initial_conditions = data.initial_conditions.clone();
    model.response_spectra = data.response_spectra.clone();
    model.random_excitations = data.random_excitations.clone();
    model.sequence = study.sequence.clone();
    resolve_typed_region_targets(&mut model, geometry, &data.regions)?;
    if data.domains.thermal.is_some() {
        model.thermal = data.domains.thermal.clone();
    }
    if data.domains.thermo_mechanical.is_some() {
        model.thermo_mechanical = data.domains.thermo_mechanical.clone();
    }
    if data.domains.electro_thermal.is_some() {
        model.electro_thermal = data.domains.electro_thermal.clone();
    }
    if data.domains.electromagnetic.is_some() {
        model.electromagnetic = data.domains.electromagnetic.clone();
    }
    if data.domains.cfd.is_some() {
        model.cfd = data.domains.cfd.clone();
    }
    validate_document_numerical_sources(&model)?;
    Ok(Some(model))
}

fn validate_document_numerical_sources(model: &AnalysisModel) -> Result<(), String> {
    for spectrum in &model.response_spectra {
        validate_document_numerical_source(
            &spectrum.ordinates,
            &format!("response spectrum `{}`", spectrum.spectrum_id.0),
        )?;
    }
    for spectrum in &model.random_excitations {
        validate_document_numerical_source(
            &spectrum.spectral_density,
            &format!("random excitation `{}`", spectrum.excitation_id.0),
        )?;
    }
    Ok(())
}

fn validate_document_numerical_source<T>(
    source: &runmat_analysis_core::NumericalDataSource<T>,
    label: &str,
) -> Result<(), String> {
    match source {
        runmat_analysis_core::NumericalDataSource::Authored { .. }
        | runmat_analysis_core::NumericalDataSource::Imported { .. } => Ok(()),
        runmat_analysis_core::NumericalDataSource::Generator(invocation)
            if matches!(
                &invocation.callable,
                runmat_analysis_core::GeneratorCallable::ProjectFunction { .. }
            ) => Ok(()),
        runmat_analysis_core::NumericalDataSource::Generator(_) => Err(format!(
            "{label} uses a runtime-bound callable; .fea generators must name a project function"
        )),
        runmat_analysis_core::NumericalDataSource::Materialized { .. } => Err(format!(
            "{label} contains a materialized generator result; .fea documents may author or import values, or declare a project-function generator"
        )),
    }
}

fn resolve_material(id: &str, material: &FeaMaterialDocument) -> MaterialModel {
    MaterialModel {
        material_id: id.to_string(),
        name: material.name.clone().unwrap_or_else(|| id.to_string()),
        provenance: runmat_analysis_core::MaterialProvenance::Authored,
        mass_density_kg_per_m3: material.mass_density_kg_per_m3,
        mechanical: material.mechanical.clone(),
        damping: material.damping,
        thermal: material.thermal.clone(),
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
    let target = match &assignment.target {
        FeaPropertyRegionTargetDocument::Beam(selection) => {
            let region = resolve_property_region_ref(
                &selection.region,
                geometry,
                aliases,
                is_importer_edge_alias,
            )?;
            runmat_analysis_core::PropertyRegionTarget::beam(region)
        }
        FeaPropertyRegionTargetDocument::Shell(selection) => {
            let region = resolve_property_region_ref(
                &selection.region,
                geometry,
                aliases,
                is_importer_face_alias,
            )?;
            runmat_analysis_core::PropertyRegionTarget::shell(region)
        }
        FeaPropertyRegionTargetDocument::Solid(selection) => {
            let region = resolve_property_region_ref(
                &selection.region,
                geometry,
                aliases,
                is_importer_solid_alias,
            )?;
            runmat_analysis_core::PropertyRegionTarget::solid(region)
        }
        FeaPropertyRegionTargetDocument::Cell(selection) => {
            let region = resolve_property_region_ref(
                &selection.region,
                geometry,
                aliases,
                is_importer_solid_alias,
            )?;
            runmat_analysis_core::PropertyRegionTarget::cell(region)
        }
        FeaPropertyRegionTargetDocument::AllBeams => {
            runmat_analysis_core::PropertyRegionTarget::all_beams()
        }
        FeaPropertyRegionTargetDocument::AllShells => {
            runmat_analysis_core::PropertyRegionTarget::all_shells()
        }
        FeaPropertyRegionTargetDocument::AllSolids => {
            runmat_analysis_core::PropertyRegionTarget::all_solids()
        }
        FeaPropertyRegionTargetDocument::AllCells => {
            runmat_analysis_core::PropertyRegionTarget::all_cells()
        }
    };
    Ok(MaterialAssignment {
        target,
        expected_material_id: assignment
            .expected_material
            .clone()
            .unwrap_or_else(|| assignment.material.clone()),
        assigned_material_id: assignment.material.clone(),
        confidence: assignment.confidence,
    })
}

fn resolve_property_region_ref(
    region: &str,
    geometry: &GeometryAsset,
    aliases: &BTreeMap<String, FeaRegionDocument>,
    is_exact_importer_alias: fn(&str) -> bool,
) -> Result<String, String> {
    if is_exact_importer_alias(region) {
        Ok(region.to_owned())
    } else {
        resolve_region_ref(region, geometry, aliases)
    }
}

fn resolve_structural_model(
    data: &FeaModelDataDocument,
    geometry: &GeometryAsset,
) -> Result<StructuralModel, String> {
    let structural = data.structural.as_ref();
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
            .map(|node| {
                Ok(StructuralNode {
                    node_id: node.id,
                    coordinates_m: node.coordinates_m,
                    solid_attachment: node
                        .solid_attachment
                        .as_ref()
                        .map(|attachment| {
                            let geometry_vertex_id = attachment.geometry_vertex.trim();
                            if geometry_vertex_id.is_empty() {
                                return Err(format!(
                                    "structural node {} has an empty solid_attachment.geometry_vertex",
                                    node.id
                                ));
                            }
                            Ok(runmat_analysis_core::ConformalSolidAttachment {
                                geometry_vertex_id: geometry_vertex_id.to_string(),
                            })
                        })
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
        elements: element_docs
            .iter()
            .map(|element| resolve_structural_element(element, geometry, &data.regions))
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
        thermal: None,
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
        random_excitations: Vec::new(),
        sequence: AnalysisSequence::from_steps(format!("{model_id}.sequence"), Vec::new()),
    }
}

fn has_structural_model_data(data: &FeaModelDataDocument) -> bool {
    data.structural.as_ref().is_some_and(|structural| {
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
        let Some(region_id) = assignment.selected_region_id() else {
            continue;
        };
        let importer_alias = match &assignment.target {
            runmat_analysis_core::PropertyRegionTarget::Beam(_) => {
                is_importer_edge_alias(region_id)
            }
            runmat_analysis_core::PropertyRegionTarget::Shell(_) => {
                is_importer_face_alias(region_id)
            }
            runmat_analysis_core::PropertyRegionTarget::Solid(_)
            | runmat_analysis_core::PropertyRegionTarget::Cell(_) => {
                is_importer_solid_alias(region_id)
            }
        };
        if !importer_alias {
            resolve_region_id(
                assignment
                    .target
                    .selected_region_id_mut()
                    .expect("selected material target has a region id"),
                geometry,
                aliases,
                &mut failure,
            );
        }
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

fn is_importer_solid_alias(region_id: &str) -> bool {
    is_importer_alias(region_id, "solid_")
}

fn is_importer_face_alias(region_id: &str) -> bool {
    is_importer_alias(region_id, "face_")
}

fn is_importer_edge_alias(region_id: &str) -> bool {
    is_importer_alias(region_id, "edge_")
}

fn is_importer_alias(region_id: &str, prefix: &str) -> bool {
    let Some(ordinal) = region_id.strip_prefix(prefix) else {
        return false;
    };
    ordinal.len() == 6 && ordinal.bytes().all(|byte| byte.is_ascii_digit()) && ordinal != "000000"
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
        AnalysisProcedure, Force, GeneratorArgument, GeneratorDomain, GeneratorEffectPolicy,
        ProcedureStep, Signal, SignalApproximationEvidence, SignalArtifactDomain,
        SignalArtifactRef, SignalExtrapolation, SignalGenerator, SignalId, SignalInterpolation,
        SignalPrecision, SignalProvenance, SignalRepresentation, SignalSamplingPolicy, SignalShape,
        SignalValue,
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
                exchange: None,
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
            "source".into(),
            serde_yaml::from_str("inline: {}\n").expect("inline model source YAML"),
        );
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

    fn inline_model_yaml(document: &mut serde_yaml::Value) -> &mut serde_yaml::Mapping {
        document
            .as_mapping_mut()
            .expect("study mapping")
            .get_mut("model")
            .expect("model")
            .as_mapping_mut()
            .expect("model mapping")
            .get_mut("source")
            .expect("model source")
            .as_mapping_mut()
            .expect("model source mapping")
            .get_mut("inline")
            .expect("inline model")
            .as_mapping_mut()
            .expect("inline model mapping")
    }

    fn inline_model(study: &FeaStudyDocument) -> &FeaModelDataDocument {
        let FeaModelSourceDocument::Inline(data) = &study.model.source else {
            panic!("expected inline model source")
        };
        data
    }

    fn inline_model_mut(study: &mut FeaStudyDocument) -> &mut FeaModelDataDocument {
        let FeaModelSourceDocument::Inline(data) = &mut study.model.source else {
            panic!("expected inline model source")
        };
        data
    }

    fn model_source_yaml(source: &str) -> FeaModelSourceDocument {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(source).unwrap();
        serde_json::from_value(serde_json::to_value(yaml).unwrap()).unwrap()
    }

    fn documentation_example_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/fea/examples")
            .join(name)
    }

    #[test]
    fn canonical_documentation_examples_parse_and_resolve() {
        let names = vec![
            "structural-static-experimental.fea",
            "structural-transient-experimental.fea",
            "modal-response-spectrum-experimental.fea",
            "acoustic-production-rejected.fea",
        ];
        #[cfg(feature = "occt-native")]
        let names = {
            let mut names = names;
            names.push("modal-random-response-experimental.fea");
            names
        };
        for name in names {
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
            if name == "structural-static-experimental.fea" {
                let evidence = study
                    .model_source_evidence
                    .as_ref()
                    .expect("file-authored model must retain source evidence");
                assert_eq!(evidence.kind, AnalysisModelSourceKind::File);
                assert_eq!(evidence.files.len(), 1);
                assert_eq!(evidence.files[0].path, "models/structural-static-load.fea");
            }
        }
    }

    #[test]
    fn response_spectrum_example_consumes_its_authored_spectrum() {
        let path = documentation_example_path("modal-response-spectrum-experimental.fea");
        let FeaResolvedDocument::Study(study) =
            pollster::block_on(load_fea_document_from_path_async(&path)).unwrap()
        else {
            panic!("{} must contain a study", path.display());
        };
        let model = study.model.as_ref().unwrap();
        assert_eq!(model.sequence.steps.len(), 2);
        assert_eq!(study.execution.steps.len(), 2);
        assert_eq!(
            model
                .sequence
                .steps
                .iter()
                .map(|step| step.procedure)
                .collect::<Vec<_>>(),
            vec![
                runmat_analysis_core::AnalysisProcedure::ModalExtraction,
                runmat_analysis_core::AnalysisProcedure::ModalResponseSpectrum,
            ]
        );
        let response = &model.sequence.steps[1];
        assert!(matches!(
            &response.independent_domain,
            runmat_analysis_core::IndependentDomain::ResponseSpectra { spectrum_ids }
                if spectrum_ids == &[runmat_analysis_core::ResponseSpectrumId("design_spectrum".into())]
        ));
        assert_eq!(response.state_inputs.len(), 1);
        assert_eq!(
            response.state_inputs[0].required_kind,
            runmat_analysis_core::StateKind::ModalBasis
        );
        assert_eq!(
            response.state_outputs[0].kind,
            runmat_analysis_core::StateKind::StructuralResponseEnvelope
        );
        assert_eq!(model.sequence.dependencies.len(), 1);
        assert_eq!(model.sequence.transitions.len(), 1);
    }

    #[test]
    fn documentation_examples_expose_formulation_requirement_truth() {
        let experimental_names = vec![
            "structural-static-experimental.fea",
            "structural-transient-experimental.fea",
            "modal-response-spectrum-experimental.fea",
        ];
        #[cfg(feature = "occt-native")]
        let experimental_names = {
            let mut names = experimental_names;
            names.push("modal-random-response-experimental.fea");
            names
        };
        for name in experimental_names {
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

    fn immutable_artifact_signal(logical_digest: runmat_execution::Digest) -> PhysicalSignal {
        PhysicalSignal::Force(Signal {
            signal_id: SignalId("generated-force".into()),
            shape: SignalShape::Scalar,
            frame: Some(CoordinateFrame::Global),
            representation: SignalRepresentation::Generator(SignalGenerator {
                invocation: runmat_analysis_core::GeneratorInvocation {
                    callable: runmat_analysis_core::GeneratorCallable::ProjectFunction {
                        qualified_name: "loads.scaled_force".into(),
                    },
                    arguments: vec![GeneratorArgument::Artifact(SignalArtifactRef {
                        artifact_id: "magnitudes".into(),
                        content_digest: logical_digest.to_string(),
                        schema: runmat_execution::value::EXTERNALIZED_VALUE_SCHEMA.into(),
                        domain: SignalArtifactDomain::Constant,
                        topology_digest: None,
                        sample_count: 2,
                    })],
                    effects: GeneratorEffectPolicy::DeclaredImmutableArtifacts,
                },
                domain: GeneratorDomain::TimeSeconds,
                sampling: SignalSamplingPolicy::Uniform {
                    start: 0.0,
                    end: 1.0,
                    sample_count: 2,
                },
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::Reject,
            }),
            precision: SignalPrecision::Real64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: None,
                max_relative_error: None,
                derivative_max_absolute_error: None,
                sample_count: 0,
            },
            provenance: SignalProvenance::Authored,
        })
    }

    #[test]
    fn immutable_signal_arguments_require_an_exact_closed_declaration_inventory() {
        let digest = runmat_execution::Digest::sha256(b"immutable-magnitudes");
        let mut document = minimal_study_yaml();
        inline_model_yaml(&mut document).insert(
            "signals".into(),
            canonical_yaml(vec![immutable_artifact_signal(digest)]),
        );
        let mapping = document.as_mapping_mut().expect("study mapping");
        mapping.insert(
            "immutable_values".into(),
            canonical_yaml(vec![AnalysisImmutableValueSpec {
                artifact_id: "magnitudes".into(),
                relative_path: "inputs/magnitudes.value.json".into(),
                logical_digest: digest.to_string(),
            }]),
        );
        let RawFeaDocument::Study(mut study) =
            parse_raw_fea_value(document).expect("typed study document")
        else {
            panic!("expected study")
        };
        validate_immutable_value_inventory(&study.immutable_values, &inline_model(&study).signals)
            .expect("exact declaration inventory");

        let declaration = study.immutable_values.pop().expect("declaration");
        assert!(validate_immutable_value_inventory(
            &study.immutable_values,
            &inline_model(&study).signals
        )
        .expect_err("missing declaration must fail")
        .contains("no immutable value declaration"));
        study.immutable_values.push(declaration);

        study.immutable_values[0].logical_digest =
            runmat_execution::Digest::sha256(b"substituted").to_string();
        assert!(validate_immutable_value_inventory(
            &study.immutable_values,
            &inline_model(&study).signals
        )
        .expect_err("substituted declaration must fail")
        .contains("differs"));
        study.immutable_values[0].logical_digest = digest.to_string();

        let PhysicalSignal::Force(signal) = &mut inline_model_mut(&mut study).signals[0] else {
            panic!("expected force signal")
        };
        let SignalRepresentation::Generator(generator) = &mut signal.representation else {
            panic!("expected generated signal")
        };
        let GeneratorArgument::Artifact(artifact) = &mut generator.invocation.arguments[0] else {
            panic!("expected immutable artifact argument")
        };
        artifact.schema = "application-specific-value".into();
        assert!(validate_immutable_value_inventory(
            &study.immutable_values,
            &inline_model(&study).signals
        )
        .expect_err("unsupported value schema must fail")
        .contains("unsupported immutable value schema"));
        let PhysicalSignal::Force(signal) = &mut inline_model_mut(&mut study).signals[0] else {
            unreachable!()
        };
        let SignalRepresentation::Generator(generator) = &mut signal.representation else {
            unreachable!()
        };
        let GeneratorArgument::Artifact(artifact) = &mut generator.invocation.arguments[0] else {
            unreachable!()
        };
        artifact.schema = runmat_execution::value::EXTERNALIZED_VALUE_SCHEMA.into();

        study.immutable_values.push(AnalysisImmutableValueSpec {
            artifact_id: "unused".into(),
            relative_path: "inputs/unused.value.json".into(),
            logical_digest: runmat_execution::Digest::sha256(b"unused").to_string(),
        });
        assert!(validate_immutable_value_inventory(
            &study.immutable_values,
            &inline_model(&study).signals
        )
        .expect_err("unused declaration must fail")
        .contains("referenced exactly"));
    }

    #[test]
    fn typed_document_fields_deserialize_without_legacy_translation() {
        let mut document = minimal_study_yaml();
        let mapping = inline_model_yaml(&mut document);
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
        assert_eq!(inline_model(&study).constraints[0].constraint_id(), "fixed");
        assert_eq!(inline_model(&study).signals[0].signal_id().0, "push");
        assert_eq!(study.sequence.steps[0].step_id.as_str(), "solve");
    }

    #[test]
    fn structural_solid_attachment_preserves_exact_cad_vertex_identity() {
        let mut document = minimal_study_yaml();
        inline_model_yaml(&mut document).insert(
                "structural".into(),
                serde_yaml::from_str(
                    "nodes:\n  - id: 7\n    coordinates_m: [0.0, 0.0, 0.0]\n    solid_attachment:\n      geometry_vertex: vertex_000007\n",
                )
                .expect("structural YAML"),
            );
        let RawFeaDocument::Study(study) = parse_raw_fea_value(document).expect("study document")
        else {
            panic!("expected study")
        };
        let intent = AnalysisCreateModelIntentSpec {
            model_id: "mixed-model".into(),
            template: model_template_for_sequence(&study.sequence).unwrap(),
        };
        let model = resolve_model(&study, inline_model(&study), &sample_geometry(), &intent)
            .expect("resolve model")
            .expect("model");
        assert_eq!(
            model.structural.as_ref().unwrap().nodes[0]
                .solid_attachment
                .as_ref()
                .unwrap()
                .geometry_vertex_id,
            "vertex_000007"
        );
    }

    #[test]
    fn structural_solid_attachment_rejects_empty_vertex_identity() {
        let mut document = minimal_study_yaml();
        inline_model_yaml(&mut document).insert(
                "structural".into(),
                serde_yaml::from_str(
                    "nodes:\n  - id: 7\n    coordinates_m: [0.0, 0.0, 0.0]\n    solid_attachment:\n      geometry_vertex: '   '\n",
                )
                .expect("structural YAML"),
            );
        let RawFeaDocument::Study(study) = parse_raw_fea_value(document).expect("study document")
        else {
            panic!("expected study")
        };
        let intent = AnalysisCreateModelIntentSpec {
            model_id: "mixed-model".into(),
            template: model_template_for_sequence(&study.sequence).unwrap(),
        };
        let error = resolve_model(&study, inline_model(&study), &sample_geometry(), &intent)
            .expect_err("empty persistent vertex identity must fail");
        assert!(error.contains("empty solid_attachment.geometry_vertex"));
    }

    #[test]
    fn retired_template_scaffold_switch_is_rejected() {
        let mut document = minimal_study_yaml();
        document
            .as_mapping_mut()
            .expect("study mapping")
            .get_mut("model")
            .expect("model")
            .as_mapping_mut()
            .expect("model mapping")
            .insert("defaults".into(), "template_scaffold".into());
        assert!(parse_raw_fea_value(document)
            .expect_err("hidden scaffold switch must be rejected")
            .contains("unknown field `defaults`"));
    }

    #[test]
    fn fea_numerical_generators_require_inspectable_project_functions() {
        let invocation = runmat_analysis_core::GeneratorInvocation {
            callable: runmat_analysis_core::GeneratorCallable::ProjectFunction {
                qualified_name: "spectra.design_input".into(),
            },
            arguments: Vec::new(),
            effects: GeneratorEffectPolicy::Pure,
        };
        let source = runmat_analysis_core::NumericalDataSource::<Vec<f64>>::Generator(invocation);
        validate_document_numerical_source(&source, "response spectrum `design`")
            .expect("named project functions are portable .fea sources");

        let bound = runmat_analysis_core::NumericalDataSource::<Vec<f64>>::Generator(
            runmat_analysis_core::GeneratorInvocation {
                callable: runmat_analysis_core::GeneratorCallable::BoundFunction {
                    function: runmat_types::ProgramFunctionId(7),
                    display_name: "anonymous spectrum closure".into(),
                },
                arguments: Vec::new(),
                effects: GeneratorEffectPolicy::Pure,
            },
        );
        assert!(
            validate_document_numerical_source(&bound, "response spectrum `design`")
                .expect_err("runtime-only closure identities cannot be serialized into .fea")
                .contains("must name a project function")
        );
    }

    #[test]
    fn fea_documents_cannot_claim_pre_materialized_generator_results() {
        let source = runmat_analysis_core::NumericalDataSource::Materialized {
            values: vec![1.0_f64],
            provenance: runmat_analysis_core::GeneratorProvenance {
                project_revision: "revision".into(),
                callable_id: "spectra.design_input".into(),
                frozen_input_digest: "sha256:inputs".into(),
            },
            content_digest: "sha256:values".into(),
        };
        assert!(
            validate_document_numerical_source(&source, "response spectrum `design`")
                .expect_err("materialized values are execution evidence, not authored input")
                .contains("contains a materialized generator result")
        );
    }

    #[test]
    fn model_file_sources_compose_by_typed_identity() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("shared.fea");
        std::fs::write(
            &source,
            r#"version: 1
kind: model
source:
  inline:
    regions:
      loaded:
        selector: { id: tip }
"#,
        )
        .unwrap();
        let authored = model_source_yaml(
            r#"compose:
  - file: { path: shared.fea }
  - inline:
      materials:
        steel:
          mass_density_kg_per_m3: 7850.0
          mechanical:
            youngs_modulus_pa: 200000000000.0
            poisson_ratio: 0.3
"#,
        );
        let resolved = pollster::block_on(resolve_model_source(
            &authored,
            directory.path(),
            &std::fs::canonicalize(directory.path()).unwrap(),
            Path::new(""),
            &mut Vec::new(),
        ))
        .unwrap();
        assert_eq!(resolved.data.regions.len(), 1);
        assert_eq!(resolved.data.materials.len(), 1);
        assert_eq!(resolved.files.len(), 1);
        assert_eq!(resolved.files[0].path, "shared.fea");
    }

    #[test]
    fn model_source_composition_rejects_conflicts() {
        let authored = model_source_yaml(
            r#"compose:
  - inline:
      regions:
        support: { selector: { id: base } }
  - inline:
      regions:
        support: { selector: { id: tip } }
"#,
        );
        let error = pollster::block_on(resolve_model_source(
            &authored,
            Path::new("."),
            &std::fs::canonicalize(".").unwrap(),
            Path::new(""),
            &mut Vec::new(),
        ))
        .expect_err("conflicting typed identity must fail");
        assert!(error.contains("conflicting region definition `support`"));
    }

    #[test]
    fn nested_model_file_sources_reject_cycles() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("a.fea"),
            "version: 1\nkind: model\nsource:\n  file: { path: b.fea }\n",
        )
        .unwrap();
        std::fs::write(
            directory.path().join("b.fea"),
            "version: 1\nkind: model\nsource:\n  file: { path: a.fea }\n",
        )
        .unwrap();
        let authored = model_source_yaml("file: { path: a.fea }\n");
        let error = pollster::block_on(resolve_model_source(
            &authored,
            directory.path(),
            &std::fs::canonicalize(directory.path()).unwrap(),
            Path::new(""),
            &mut Vec::new(),
        ))
        .expect_err("recursive model sources must fail");
        assert!(error.contains("model source import cycle"));
    }

    #[cfg(unix)]
    #[test]
    fn model_file_sources_cannot_escape_through_symlinks() {
        use std::os::unix::fs::symlink;

        let study_directory = tempfile::tempdir().unwrap();
        let outside_directory = tempfile::tempdir().unwrap();
        let outside_model = outside_directory.path().join("outside.fea");
        std::fs::write(
            &outside_model,
            "version: 1\nkind: model\nsource:\n  inline: {}\n",
        )
        .unwrap();
        symlink(&outside_model, study_directory.path().join("linked.fea")).unwrap();

        let authored = model_source_yaml("file: { path: linked.fea }\n");
        let error = pollster::block_on(resolve_model_source(
            &authored,
            study_directory.path(),
            &std::fs::canonicalize(study_directory.path()).unwrap(),
            Path::new(""),
            &mut Vec::new(),
        ))
        .expect_err("symlinked model source outside the study tree must fail");
        assert!(error.contains("outside the study source tree"));
    }

    #[test]
    fn standalone_thermal_domain_round_trips_through_the_document_boundary() {
        let mut document = minimal_study_yaml();
        let mut domains = serde_yaml::Mapping::new();
        domains.insert(
            "thermal".into(),
            canonical_yaml(ThermalDomain {
                reference_temperature: runmat_analysis_core::Temperature::new(318.15),
            }),
        );
        inline_model_yaml(&mut document).insert("domains".into(), domains.into());

        let parsed = parse_raw_fea_value(document).expect("thermal study document");
        let RawFeaDocument::Study(study) = parsed else {
            panic!("expected study")
        };
        let intent = AnalysisCreateModelIntentSpec {
            model_id: "thermal-model".into(),
            template: model_template_for_sequence(&study.sequence).unwrap(),
        };
        let model = resolve_model(&study, inline_model(&study), &sample_geometry(), &intent)
            .expect("resolve thermal model")
            .expect("resolved model");
        assert_eq!(model.thermal.unwrap().reference_temperature.kelvin, 318.15);
        assert!(model.thermo_mechanical.is_none());
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
                    cost: runmat_analysis_core::ObservationCostPolicy::Automatic,
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
    fn material_assignment_targets_are_closed_typed_domains_with_explicit_scope() {
        let parse = |source: &str| -> Result<FeaMaterialAssignmentDocument, String> {
            let yaml = serde_yaml::from_str::<serde_yaml::Value>(source)
                .map_err(|error| error.to_string())?;
            let canonical = serde_json::to_value(yaml).map_err(|error| error.to_string())?;
            serde_json::from_value(canonical).map_err(|error| error.to_string())
        };
        for (name, expected) in [
            ("all_beams", "beam"),
            ("all_shells", "shell"),
            ("all_solids", "solid"),
            ("all_cells", "cell"),
        ] {
            let parsed = parse(&format!("target: {name}\nmaterial: material\n"))
                .expect("whole-domain material target");
            let actual = match parsed.target {
                FeaPropertyRegionTargetDocument::AllBeams => "beam",
                FeaPropertyRegionTargetDocument::AllShells => "shell",
                FeaPropertyRegionTargetDocument::AllSolids => "solid",
                FeaPropertyRegionTargetDocument::AllCells => "cell",
                _ => panic!("whole-domain target parsed as a selected target"),
            };
            assert_eq!(actual, expected);
        }

        for kind in ["beam", "shell", "solid", "cell"] {
            let parsed = parse(&format!(
                "target:\n  {kind}:\n    region: selected_domain\nmaterial: material\n"
            ))
            .expect("selected material target");
            let (actual_kind, selection) = match parsed.target {
                FeaPropertyRegionTargetDocument::Beam(selection) => ("beam", selection),
                FeaPropertyRegionTargetDocument::Shell(selection) => ("shell", selection),
                FeaPropertyRegionTargetDocument::Solid(selection) => ("solid", selection),
                FeaPropertyRegionTargetDocument::Cell(selection) => ("cell", selection),
                _ => panic!("selected target parsed as a whole-domain target"),
            };
            assert_eq!(actual_kind, kind);
            assert_eq!(selection.region, "selected_domain");
        }

        parse("region: region_default\nmaterial: steel\n")
            .expect_err("untyped legacy region assignments must be rejected");
        parse("target: face\nmaterial: steel\n")
            .expect_err("boundary faces are not material domains");
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
