//! Public authoring operations, inputs, and result contracts.

use std::str::FromStr;

use runmat_analysis_core::{
    AnalysisProcedure, AnalysisSequence, CfdDomain, CoordinateFrame, ElectroThermalDomain,
    ElectromagneticDomain, EvidenceConfidence, FormulationRequirement, MaterialAcousticModel,
    MaterialElectricalModel, MaterialMechanicalModel, MaterialPlasticModel, MaterialThermalModel,
    PhysicsFamily, ProcedureStepId, ThermalDomain, ThermoMechanicalDomain,
};
use runmat_analysis_fea::ComputeBackend;
use runmat_geometry_core::UnitSystem;
use runmat_meshing_core::ElementOrder;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value as JsonValue;

use super::super::contracts::{AnalysisModelTemplate, AnalysisSolverOptions};
pub use super::super::fea_document::FeaRegionSelector;

pub const FEA_STUDY_DOCUMENT_OPERATION_NAMES: &[&str] = &[
    "get_summary",
    "create",
    "add_region",
    "update_region",
    "remove_region",
    "add_material",
    "update_material",
    "remove_material",
    "assign_material",
    "remove_material_assignment",
    "add_constraint",
    "update_constraint",
    "remove_constraint",
    "add_boundary",
    "update_boundary",
    "remove_boundary",
    "add_connector",
    "update_connector",
    "remove_connector",
    "add_interface",
    "update_interface",
    "remove_interface",
    "add_signal",
    "update_signal",
    "remove_signal",
    "add_excitation_case",
    "update_excitation_case",
    "remove_excitation_case",
    "add_excitation",
    "update_excitation",
    "remove_excitation",
    "add_initial_condition",
    "update_initial_condition",
    "remove_initial_condition",
    "add_response_spectrum",
    "update_response_spectrum",
    "remove_response_spectrum",
    "add_random_excitation",
    "update_random_excitation",
    "remove_random_excitation",
    "set_sequence",
    "set_domains",
    "set_mesh",
    "clear_mesh",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(usize)]
pub enum FeaStudyDocumentOperation {
    GetSummary,
    Create,
    AddRegion,
    UpdateRegion,
    RemoveRegion,
    AddMaterial,
    UpdateMaterial,
    RemoveMaterial,
    AssignMaterial,
    RemoveMaterialAssignment,
    AddConstraint,
    UpdateConstraint,
    RemoveConstraint,
    AddBoundary,
    UpdateBoundary,
    RemoveBoundary,
    AddConnector,
    UpdateConnector,
    RemoveConnector,
    AddInterface,
    UpdateInterface,
    RemoveInterface,
    AddSignal,
    UpdateSignal,
    RemoveSignal,
    AddExcitationCase,
    UpdateExcitationCase,
    RemoveExcitationCase,
    AddExcitation,
    UpdateExcitation,
    RemoveExcitation,
    AddInitialCondition,
    UpdateInitialCondition,
    RemoveInitialCondition,
    AddResponseSpectrum,
    UpdateResponseSpectrum,
    RemoveResponseSpectrum,
    AddRandomExcitation,
    UpdateRandomExcitation,
    RemoveRandomExcitation,
    SetSequence,
    SetDomains,
    SetMesh,
    ClearMesh,
}

impl FeaStudyDocumentOperation {
    pub const fn as_str(self) -> &'static str {
        FEA_STUDY_DOCUMENT_OPERATION_NAMES[self as usize]
    }
}

impl FromStr for FeaStudyDocumentOperation {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if !FEA_STUDY_DOCUMENT_OPERATION_NAMES.contains(&value) {
            return Err(format!(
                "unsupported finite element study operation: {value}"
            ));
        }
        serde_json::from_value(JsonValue::String(value.to_string()))
            .map_err(|_| format!("unsupported finite element study operation: {value}"))
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaStudyCreateInput {
    #[serde(default)]
    pub study_id: Option<String>,
    pub geometry: FeaStudyGeometryInput,
    pub model: FeaStudyModelInput,
    #[serde(default)]
    pub mesh: Option<FeaStudyMeshInput>,
    #[serde(default)]
    pub sequence: Option<AnalysisSequence>,
    #[serde(default)]
    pub execution: Option<Vec<FeaStudyExecutionInput>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaStudyGeometryInput {
    pub path: String,
    #[serde(default = "default_units")]
    pub units: UnitSystem,
    #[serde(default)]
    pub import: FeaStudyGeometryImportInput,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaStudyGeometryImportInput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_triangles: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaStudyModelInput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub template: AnalysisModelTemplate,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<CoordinateFrame>,
    #[serde(default)]
    pub defaults: FeaStudyModelDefaults,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaStudyModelDefaults {
    #[default]
    TemplateScaffold,
    None,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaStudyMeshInput {
    #[serde(default = "default_mesh_element_order")]
    pub element_order: ElementOrder,
    #[serde(default = "default_mesh_maximum_elements")]
    pub maximum_elements: u64,
    #[serde(default = "default_mesh_target_edge_length_m")]
    pub target_edge_length_m: f64,
    #[serde(default = "default_mesh_maximum_chordal_deviation_m")]
    pub maximum_chordal_deviation_m: f64,
    #[serde(default = "default_mesh_maximum_grading_ratio")]
    pub maximum_grading_ratio: f64,
    #[serde(default)]
    pub deterministic_seed: u64,
}

#[derive(Clone, Debug)]
pub struct FeaStudyExecutionInput {
    pub step: ProcedureStepId,
    pub backend: ComputeBackend,
    pub requirement: FormulationRequirement,
    pub solver_options: Option<AnalysisSolverOptions>,
}

#[derive(Deserialize)]
struct FeaStudyExecutionInputFields {
    step: ProcedureStepId,
    #[serde(default = "default_backend")]
    backend: ComputeBackend,
    #[serde(default = "default_formulation_requirement")]
    requirement: FormulationRequirement,
}

impl Serialize for FeaStudyExecutionInput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut object = serde_json::Map::new();
        object.insert(
            "step".into(),
            serde_json::to_value(&self.step).map_err(serde::ser::Error::custom)?,
        );
        object.insert(
            "backend".into(),
            serde_json::to_value(self.backend).map_err(serde::ser::Error::custom)?,
        );
        object.insert(
            "requirement".into(),
            serde_json::to_value(self.requirement).map_err(serde::ser::Error::custom)?,
        );
        if let Some(options) = &self.solver_options {
            let tagged = serde_json::to_value(options).map_err(serde::ser::Error::custom)?;
            object.extend(
                tagged
                    .as_object()
                    .ok_or_else(|| serde::ser::Error::custom("solver options must be an object"))?
                    .clone(),
            );
        }
        JsonValue::Object(object).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FeaStudyExecutionInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = JsonValue::deserialize(deserializer)?;
        super::super::fea_document::validate_execution_object(&value)
            .map_err(serde::de::Error::custom)?;
        let solver_options = super::super::fea_document::parse_execution_solver_options(&value)
            .map_err(serde::de::Error::custom)?;
        let fields: FeaStudyExecutionInputFields =
            serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(Self {
            step: fields.step,
            backend: fields.backend,
            requirement: fields.requirement,
            solver_options,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaStudyRegionInput {
    pub region_id: String,
    pub selector: FeaRegionSelector,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaStudyMaterialInput {
    pub material_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub mechanical: MaterialMechanicalModel,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal: Option<MaterialThermalModel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acoustic: Option<MaterialAcousticModel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub electrical: Option<MaterialElectricalModel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plastic: Option<MaterialPlasticModel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaStudyMaterialAssignmentInput {
    pub region: String,
    pub material: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_material: Option<String>,
    #[serde(default = "default_assignment_confidence")]
    pub confidence: EvidenceConfidence,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaStudyDomainsInput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal: Option<ThermalDomain>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermo_mechanical: Option<ThermoMechanicalDomain>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub electro_thermal: Option<ElectroThermalDomain>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub electromagnetic: Option<ElectromagneticDomain>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfd: Option<CfdDomain>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct IdInput {
    pub id: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AssignmentIdInput {
    pub region: String,
    pub material: String,
}

#[derive(Debug, Serialize)]
pub struct FeaStudyDocumentOperationOutput {
    pub source: String,
    pub result: FeaStudyDocumentOperationResult,
    pub write: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaStudySummary {
    pub study_id: Option<String>,
    pub geometry_path: Option<String>,
    pub geometry_units: Option<UnitSystem>,
    pub model_template: Option<AnalysisModelTemplate>,
    pub mesh: Option<FeaStudyMeshInput>,
    pub execution_backend: Option<ComputeBackend>,
    pub sequence_id: Option<String>,
    pub primary_family: Option<PhysicsFamily>,
    pub primary_procedure: Option<AnalysisProcedure>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaStudyRegionEntry {
    pub key: String,
    pub selector: Option<FeaRegionSelector>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaStudyMaterialEntry {
    pub key: String,
    pub name: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaStudyMaterialAssignmentEntry {
    pub index: usize,
    pub region: Option<String>,
    pub material: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaStudyNamedEntry {
    pub index: usize,
    pub id: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaStudySequenceSummary {
    pub sequence_id: String,
    pub step_count: usize,
    pub primary_family: Option<PhysicsFamily>,
    pub primary_procedure: Option<AnalysisProcedure>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaStudyReadiness {
    pub ready_to_solve: bool,
    pub blockers: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaStudyDocumentDiff {
    pub changed_sections: Vec<String>,
    pub before: FeaStudySummary,
    pub after: FeaStudySummary,
    pub source_length_before: usize,
    pub source_length_after: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaStudyDocumentCounts {
    pub regions: usize,
    pub materials: usize,
    pub material_assignments: usize,
    pub constraints: usize,
    pub boundaries: usize,
    pub connectors: usize,
    pub interfaces: usize,
    pub signals: usize,
    pub excitation_cases: usize,
    pub excitations: usize,
    pub initial_conditions: usize,
    pub response_spectra: usize,
    pub random_excitations: usize,
    pub sequence_steps: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaStudyDocumentOperationResult {
    pub path: String,
    pub document_kind: String,
    pub summary: FeaStudySummary,
    pub regions: Vec<FeaStudyRegionEntry>,
    pub materials: Vec<FeaStudyMaterialEntry>,
    pub material_assignments: Vec<FeaStudyMaterialAssignmentEntry>,
    pub constraints: Vec<FeaStudyNamedEntry>,
    pub boundaries: Vec<FeaStudyNamedEntry>,
    pub connectors: Vec<FeaStudyNamedEntry>,
    pub interfaces: Vec<FeaStudyNamedEntry>,
    pub signals: Vec<FeaStudyNamedEntry>,
    pub excitation_cases: Vec<FeaStudyNamedEntry>,
    pub excitations: Vec<FeaStudyNamedEntry>,
    pub initial_conditions: Vec<FeaStudyNamedEntry>,
    pub response_spectra: Vec<FeaStudyNamedEntry>,
    pub random_excitations: Vec<FeaStudyNamedEntry>,
    pub sequence: Option<FeaStudySequenceSummary>,
    pub counts: FeaStudyDocumentCounts,
    pub readiness: FeaStudyReadiness,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diff: Option<FeaStudyDocumentDiff>,
}

pub(super) fn default_units() -> UnitSystem {
    UnitSystem::Millimeter
}

pub(super) fn default_backend() -> ComputeBackend {
    ComputeBackend::Cpu
}

pub(super) fn default_formulation_requirement() -> FormulationRequirement {
    FormulationRequirement::Production
}

pub(super) fn default_mesh_element_order() -> ElementOrder {
    ElementOrder::Tet4
}

pub(super) fn default_mesh_maximum_elements() -> u64 {
    1_000_000
}

pub(super) fn default_mesh_target_edge_length_m() -> f64 {
    0.01
}

pub(super) fn default_mesh_maximum_chordal_deviation_m() -> f64 {
    0.0001
}

pub(super) fn default_mesh_maximum_grading_ratio() -> f64 {
    1.3
}

pub(super) fn default_assignment_confidence() -> EvidenceConfidence {
    EvidenceConfidence::Verified
}
