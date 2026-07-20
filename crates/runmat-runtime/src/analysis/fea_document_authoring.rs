use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use serde_yaml::{Mapping, Value as YamlValue};
use std::str::FromStr;

use super::contracts::{AnalysisCreateModelProfile, AnalysisRunKind};

pub const FEA_STUDY_DOCUMENT_OPERATION_NAMES: &[&str] = &[
    "get_summary",
    "create",
    "add_region",
    "update_region",
    "remove_region",
    "add_material",
    "update_material",
    "assign_material",
    "add_constraint",
    "update_constraint",
    "remove_constraint",
    "add_driving_condition",
    "update_driving_condition",
    "remove_driving_condition",
    "set_mesh",
    "set_outputs",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaStudyDocumentOperation {
    GetSummary,
    Create,
    AddRegion,
    UpdateRegion,
    RemoveRegion,
    AddMaterial,
    UpdateMaterial,
    AssignMaterial,
    AddConstraint,
    UpdateConstraint,
    RemoveConstraint,
    AddDrivingCondition,
    UpdateDrivingCondition,
    RemoveDrivingCondition,
    SetMesh,
    SetOutputs,
}

impl FeaStudyDocumentOperation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GetSummary => "get_summary",
            Self::Create => "create",
            Self::AddRegion => "add_region",
            Self::UpdateRegion => "update_region",
            Self::RemoveRegion => "remove_region",
            Self::AddMaterial => "add_material",
            Self::UpdateMaterial => "update_material",
            Self::AssignMaterial => "assign_material",
            Self::AddConstraint => "add_constraint",
            Self::UpdateConstraint => "update_constraint",
            Self::RemoveConstraint => "remove_constraint",
            Self::AddDrivingCondition => "add_driving_condition",
            Self::UpdateDrivingCondition => "update_driving_condition",
            Self::RemoveDrivingCondition => "remove_driving_condition",
            Self::SetMesh => "set_mesh",
            Self::SetOutputs => "set_outputs",
        }
    }
}

impl FromStr for FeaStudyDocumentOperation {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "get_summary" => Ok(Self::GetSummary),
            "create" => Ok(Self::Create),
            "add_region" => Ok(Self::AddRegion),
            "update_region" => Ok(Self::UpdateRegion),
            "remove_region" => Ok(Self::RemoveRegion),
            "add_material" => Ok(Self::AddMaterial),
            "update_material" => Ok(Self::UpdateMaterial),
            "assign_material" => Ok(Self::AssignMaterial),
            "add_constraint" => Ok(Self::AddConstraint),
            "update_constraint" => Ok(Self::UpdateConstraint),
            "remove_constraint" => Ok(Self::RemoveConstraint),
            "add_driving_condition" => Ok(Self::AddDrivingCondition),
            "update_driving_condition" => Ok(Self::UpdateDrivingCondition),
            "remove_driving_condition" => Ok(Self::RemoveDrivingCondition),
            "set_mesh" => Ok(Self::SetMesh),
            "set_outputs" => Ok(Self::SetOutputs),
            other => Err(format!(
                "unsupported finite element study operation: {other}"
            )),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct FeaStudyDocumentOperationOutput {
    pub source: String,
    pub result: FeaStudyDocumentOperationResult,
    pub write: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FeaStudySummary {
    #[serde(rename = "studyId")]
    pub study_id: Option<String>,
    #[serde(rename = "geometryPath")]
    pub geometry_path: Option<String>,
    #[serde(rename = "geometryUnits")]
    pub geometry_units: Option<String>,
    #[serde(rename = "modelProfile")]
    pub model_profile: Option<String>,
    #[serde(rename = "modelDefaults")]
    pub model_defaults: Option<String>,
    #[serde(rename = "meshProfile")]
    pub mesh_profile: Option<String>,
    #[serde(rename = "meshTargetSize")]
    pub mesh_target_size: Option<String>,
    #[serde(rename = "meshMaxElements")]
    pub mesh_max_elements: Option<String>,
    #[serde(rename = "runKind")]
    pub run_kind: Option<String>,
    #[serde(rename = "runBackend")]
    pub run_backend: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeaStudyRegionEntry {
    pub key: String,
    pub selector: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeaStudyMaterialEntry {
    pub key: String,
    pub name: Option<String>,
    #[serde(rename = "mechanicalSummary")]
    pub mechanical_summary: Option<String>,
    #[serde(rename = "thermalSummary")]
    pub thermal_summary: Option<String>,
    #[serde(rename = "electromagneticSummary")]
    pub electromagnetic_summary: Option<String>,
    #[serde(rename = "acousticSummary")]
    pub acoustic_summary: Option<String>,
    #[serde(rename = "fluidSummary")]
    pub fluid_summary: Option<String>,
    #[serde(rename = "youngsModulusPa")]
    pub youngs_modulus_pa: Option<String>,
    #[serde(rename = "poissonRatio")]
    pub poisson_ratio: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeaStudyMaterialAssignmentEntry {
    pub index: usize,
    pub region: Option<String>,
    pub material: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeaStudyBoundaryConditionEntry {
    pub index: usize,
    pub id: Option<String>,
    pub region: Option<String>,
    pub kind: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeaStudyDrivingConditionEntry {
    pub index: usize,
    pub id: Option<String>,
    pub region: Option<String>,
    #[serde(rename = "type")]
    pub condition_type: Option<String>,
    pub value: Option<String>,
    pub direction: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeaStudyStepEntry {
    pub index: usize,
    pub id: Option<String>,
    pub kind: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeaStudyOutputEntry {
    pub index: usize,
    pub id: Option<String>,
    pub field: Option<String>,
    pub location: Option<String>,
    pub kind: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeaStudyReadiness {
    #[serde(rename = "readyToSolve")]
    pub ready_to_solve: bool,
    pub blockers: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeaStudyDocumentDiff {
    #[serde(rename = "changedSections")]
    pub changed_sections: Vec<String>,
    pub before: FeaStudySummary,
    pub after: FeaStudySummary,
    #[serde(rename = "sourceLengthBefore")]
    pub source_length_before: usize,
    #[serde(rename = "sourceLengthAfter")]
    pub source_length_after: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeaStudyDocumentCounts {
    pub regions: usize,
    pub materials: usize,
    pub material_assignments: usize,
    pub boundary_conditions: usize,
    pub driving_conditions: usize,
    pub steps: usize,
    pub outputs: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeaStudyDocumentOperationResult {
    pub path: String,
    pub document_kind: String,
    pub summary: FeaStudySummary,
    pub regions: Vec<FeaStudyRegionEntry>,
    pub materials: Vec<FeaStudyMaterialEntry>,
    pub material_assignments: Vec<FeaStudyMaterialAssignmentEntry>,
    pub boundary_conditions: Vec<FeaStudyBoundaryConditionEntry>,
    pub driving_conditions: Vec<FeaStudyDrivingConditionEntry>,
    pub steps: Vec<FeaStudyStepEntry>,
    pub outputs: Vec<FeaStudyOutputEntry>,
    pub counts: FeaStudyDocumentCounts,
    pub readiness: FeaStudyReadiness,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diff: Option<FeaStudyDocumentDiff>,
}

pub fn apply_fea_study_document_operation(
    operation: &str,
    path: &str,
    source: Option<&str>,
    input: JsonValue,
) -> Result<FeaStudyDocumentOperationOutput, String> {
    let operation = FeaStudyDocumentOperation::from_str(operation)?;
    apply_fea_study_document_operation_typed(operation, path, source, input)
}

pub fn apply_fea_study_document_operation_typed(
    operation: FeaStudyDocumentOperation,
    path: &str,
    source: Option<&str>,
    input: JsonValue,
) -> Result<FeaStudyDocumentOperationOutput, String> {
    if operation == FeaStudyDocumentOperation::Create {
        let study_id = read_string(&input, &["study_id", "studyId"])
            .map(str::to_string)
            .unwrap_or_else(|| study_id_from_path(path));
        let geometry_path = required_string(&input, &["geometry_path", "geometryPath"])?;
        let geometry_units =
            read_string(&input, &["geometry_units", "geometryUnits"]).unwrap_or("millimeter");
        let model_profile = required_string(&input, &["model_profile", "modelProfile"])?;
        let next_source =
            create_fea_study_source(&study_id, geometry_path, geometry_units, model_profile)?;
        return Ok(FeaStudyDocumentOperationOutput {
            result: with_diff(path, "", &next_source, &["document"])?,
            source: next_source,
            write: true,
        });
    }

    let current_source = source.unwrap_or_default();
    if operation == FeaStudyDocumentOperation::GetSummary {
        return Ok(FeaStudyDocumentOperationOutput {
            source: current_source.to_string(),
            result: summarize_fea_study_document_contract(path, current_source)?,
            write: false,
        });
    }

    let mut document = parse_document(current_source)?;
    match operation {
        FeaStudyDocumentOperation::AddRegion => {
            let region_id = required_id(&input, &["region_id", "regionId"])?;
            let selector = required_string(&input, &["selector"])?;
            let mut region = Mapping::new();
            region.insert(yaml_string("selector"), yaml_string(selector));
            insert_map_entry(
                &mut document,
                "regions",
                &region_id,
                YamlValue::Mapping(region),
            )?;
        }
        FeaStudyDocumentOperation::UpdateRegion => {
            let region_id = required_id(&input, &["region_id", "regionId"])?;
            let selector = required_string(&input, &["selector"])?;
            let mut region = Mapping::new();
            region.insert(yaml_string("selector"), yaml_string(selector));
            replace_map_entry(
                &mut document,
                "regions",
                &region_id,
                YamlValue::Mapping(region),
            )?;
        }
        FeaStudyDocumentOperation::RemoveRegion => {
            let region_id = required_id(&input, &["region_id", "regionId"])?;
            remove_map_entry(&mut document, "regions", &region_id)?;
        }
        FeaStudyDocumentOperation::AddMaterial => {
            let material_id = required_id(&input, &["material_id", "materialId"])?;
            insert_map_entry(
                &mut document,
                "materials",
                &material_id,
                material_document(&input)?,
            )?;
        }
        FeaStudyDocumentOperation::UpdateMaterial => {
            let material_id = required_id(&input, &["material_id", "materialId"])?;
            replace_map_entry(
                &mut document,
                "materials",
                &material_id,
                material_document(&input)?,
            )?;
        }
        FeaStudyDocumentOperation::AssignMaterial => {
            append_sequence_entry(
                &mut document,
                "material_assignments",
                YamlValue::Mapping(mapping_from_pairs([
                    (
                        "region",
                        yaml_string(required_id(&input, &["region_id", "regionId"])?),
                    ),
                    (
                        "material",
                        yaml_string(required_id(&input, &["material_id", "materialId"])?),
                    ),
                ])),
            );
        }
        FeaStudyDocumentOperation::AddConstraint => {
            append_sequence_entry_with_unique_id(
                &mut document,
                "boundary_conditions",
                constraint_document(&input)?,
            )?;
        }
        FeaStudyDocumentOperation::UpdateConstraint => {
            replace_sequence_entry_by_id(
                &mut document,
                "boundary_conditions",
                &required_id(&input, &["constraint_id", "constraintId"])?,
                constraint_document(&input)?,
            )?;
        }
        FeaStudyDocumentOperation::RemoveConstraint => {
            remove_sequence_entry_by_id(
                &mut document,
                "boundary_conditions",
                &required_id(&input, &["constraint_id", "constraintId"])?,
            )?;
        }
        FeaStudyDocumentOperation::AddDrivingCondition => {
            append_sequence_entry_with_unique_id_labeled(
                &mut document,
                "loads",
                "driving_conditions",
                driving_condition_document(&input)?,
            )?;
        }
        FeaStudyDocumentOperation::UpdateDrivingCondition => {
            replace_sequence_entry_by_id_labeled(
                &mut document,
                "loads",
                "driving_conditions",
                &required_id(&input, &["driving_condition_id", "drivingConditionId"])?,
                driving_condition_document(&input)?,
            )?;
        }
        FeaStudyDocumentOperation::RemoveDrivingCondition => {
            remove_sequence_entry_by_id_labeled(
                &mut document,
                "loads",
                "driving_conditions",
                &required_id(&input, &["driving_condition_id", "drivingConditionId"])?,
            )?;
        }
        FeaStudyDocumentOperation::SetMesh => {
            replace_optional_map_entry(&mut document, "mesh", mesh_document(&input)?);
        }
        FeaStudyDocumentOperation::SetOutputs => {
            replace_sequence_block(&mut document, "outputs", output_documents(&input)?);
        }
        FeaStudyDocumentOperation::Create | FeaStudyDocumentOperation::GetSummary => {
            unreachable!("create and get_summary are handled before document mutation")
        }
    }

    let next_source = serialize_document(&document)?;
    Ok(FeaStudyDocumentOperationOutput {
        result: with_diff(
            path,
            current_source,
            &next_source,
            &[section_for_operation(operation)],
        )?,
        source: next_source,
        write: true,
    })
}

fn create_fea_study_source(
    study_id: &str,
    geometry_path: &str,
    geometry_units: &str,
    model_profile: &str,
) -> Result<String, String> {
    let normalized_model_profile = model_profile.trim().to_ascii_lowercase();
    let model_profile = AnalysisCreateModelProfile::from_snake_case(&normalized_model_profile)
        .ok_or_else(|| format!("unsupported physics model profile: {model_profile}"))?;
    let run_kind = model_profile.derived_run_kind();
    let study_id = normalize_yaml_key(study_id).ok_or_else(|| {
        "study_id must include at least one ASCII letter, number, or underscore".to_string()
    })?;
    let document = mapping_from_pairs([
        ("version", yaml_number(1.0)?),
        ("kind", yaml_string("study")),
        ("id", yaml_string(study_id)),
        (
            "geometry",
            YamlValue::Mapping(mapping_from_pairs([
                ("path", yaml_string(geometry_path)),
                ("units", yaml_string(geometry_units)),
            ])),
        ),
        (
            "model",
            YamlValue::Mapping(mapping_from_pairs([(
                "profile",
                yaml_string(model_profile.as_snake_case()),
            )])),
        ),
        (
            "run",
            YamlValue::Mapping(mapping_from_pairs([
                ("kind", yaml_string(run_kind.as_snake_case())),
                ("backend", yaml_string("cpu")),
            ])),
        ),
    ]);
    serialize_document(&YamlValue::Mapping(document))
}

pub fn summarize_fea_study_document(path: &str, source: &str) -> Result<JsonValue, String> {
    serde_json::to_value(summarize_fea_study_document_contract(path, source)?)
        .map_err(|err| err.to_string())
}

pub fn summarize_fea_study_document_contract(
    path: &str,
    source: &str,
) -> Result<FeaStudyDocumentOperationResult, String> {
    let document = parse_document(source)?;
    let document_kind =
        read_yaml_string(&document, &["kind"]).unwrap_or_else(|| "study".to_string());
    let summary = summarize_fea_study(&document);
    let regions = map_entries(&document, "regions", |key, value| FeaStudyRegionEntry {
        key,
        selector: nested_scalar(value, &["selector"]),
    });
    let materials = map_entries(&document, "materials", |key, value| {
        let youngs = nested_scalar(value, &["mechanical", "youngs_modulus_pa"]);
        let poisson = nested_scalar(value, &["mechanical", "poisson_ratio"]);
        let mechanical_summary = match (youngs.as_deref(), poisson.as_deref()) {
            (Some(youngs), Some(poisson)) => Some(format!(
                "E {}, nu {}",
                format_engineering_value(youngs),
                poisson
            )),
            (Some(youngs), None) => Some(format!("E {}", format_engineering_value(youngs))),
            (None, Some(poisson)) => Some(format!("nu {poisson}")),
            (None, None) => None,
        };
        FeaStudyMaterialEntry {
            key,
            name: nested_scalar(value, &["name"]),
            mechanical_summary,
            thermal_summary: material_group_summary(
                value,
                "thermal",
                &[
                    ("k", "thermal_conductivity_w_per_m_k"),
                    ("cp", "specific_heat_j_per_kg_k"),
                    ("rho", "density_kg_per_m3"),
                ],
            ),
            electromagnetic_summary: material_group_summary(
                value,
                "electromagnetic",
                &[
                    ("epsilon_r", "relative_permittivity"),
                    ("mu_r", "relative_permeability"),
                    ("sigma", "electrical_conductivity_s_per_m"),
                ],
            ),
            acoustic_summary: material_group_summary(
                value,
                "acoustic",
                &[
                    ("rho", "density_kg_per_m3"),
                    ("c", "speed_of_sound_m_per_s"),
                    ("Z", "acoustic_impedance_pa_s_per_m"),
                ],
            ),
            fluid_summary: material_group_summary(
                value,
                "fluid",
                &[
                    ("rho", "density_kg_per_m3"),
                    ("mu", "dynamic_viscosity_pa_s"),
                    ("p_ref", "reference_pressure_pa"),
                    ("T_ref", "reference_temperature_k"),
                ],
            ),
            youngs_modulus_pa: youngs,
            poisson_ratio: poisson,
        }
    });
    let material_assignments =
        sequence_entries(&document, "material_assignments", |index, value| {
            FeaStudyMaterialAssignmentEntry {
                index,
                region: nested_scalar(value, &["region"]),
                material: nested_scalar(value, &["material"]),
            }
        });
    let boundary_conditions = sequence_entries(&document, "boundary_conditions", |index, value| {
        FeaStudyBoundaryConditionEntry {
            index,
            id: nested_scalar(value, &["id"]),
            region: nested_scalar(value, &["region"]),
            kind: nested_scalar(value, &["kind"]).or_else(|| nested_scalar(value, &["type"])),
        }
    });
    let driving_conditions = sequence_entries(&document, "loads", |index, value| {
        FeaStudyDrivingConditionEntry {
            index,
            id: nested_scalar(value, &["id"]),
            region: nested_scalar(value, &["region"]),
            condition_type: nested_scalar(value, &["type"])
                .or_else(|| nested_scalar(value, &["kind"])),
            value: nested_scalar(value, &["vector"])
                .or_else(|| nested_scalar(value, &["force"]))
                .or_else(|| nested_scalar(value, &["moment"]))
                .or_else(|| nested_scalar(value, &["magnitude_pa"]))
                .or_else(|| nested_scalar(value, &["pressure_pa"]))
                .or_else(|| nested_scalar(value, &["velocity_m_per_s"]))
                .or_else(|| nested_scalar(value, &["mass_flow_kg_per_s"]))
                .or_else(|| nested_scalar(value, &["volumetric_flow_m3_per_s"]))
                .or_else(|| nested_scalar(value, &["temperature_k"]))
                .or_else(|| nested_scalar(value, &["heat_flux_w_per_m2"]))
                .or_else(|| nested_scalar(value, &["volumetric_w_per_m3"]))
                .or_else(|| nested_scalar(value, &["current_a"]))
                .or_else(|| nested_scalar(value, &["voltage_v"]))
                .or_else(|| nested_scalar(value, &["power_w"]))
                .or_else(|| nested_scalar(value, &["frequency_hz"]))
                .or_else(|| nested_scalar(value, &["amplitude_scale"])),
            direction: nested_scalar(value, &["direction"])
                .or_else(|| nested_scalar(value, &["normal"])),
        }
    });
    let steps = sequence_entries(&document, "steps", |index, value| FeaStudyStepEntry {
        index,
        id: nested_scalar(value, &["id"]),
        kind: nested_scalar(value, &["kind"]),
    });
    let outputs = sequence_entries(&document, "outputs", |index, value| FeaStudyOutputEntry {
        index,
        id: nested_scalar(value, &["id"]),
        field: nested_scalar(value, &["field"])
            .or_else(|| nested_scalar(value, &["field_id"]))
            .or_else(|| nested_scalar(value, &["name"])),
        location: nested_scalar(value, &["location"]).or_else(|| nested_scalar(value, &["target"])),
        kind: nested_scalar(value, &["kind"]).or_else(|| nested_scalar(value, &["type"])),
    });
    let readiness = study_readiness(
        &summary,
        &materials,
        &material_assignments,
        &boundary_conditions,
        &driving_conditions,
    );

    Ok(FeaStudyDocumentOperationResult {
        path: path.to_string(),
        document_kind,
        counts: FeaStudyDocumentCounts {
            regions: regions.len(),
            materials: materials.len(),
            material_assignments: material_assignments.len(),
            boundary_conditions: boundary_conditions.len(),
            driving_conditions: driving_conditions.len(),
            steps: steps.len(),
            outputs: outputs.len(),
        },
        summary,
        regions,
        materials,
        material_assignments,
        boundary_conditions,
        driving_conditions,
        steps,
        outputs,
        readiness,
        diff: None,
    })
}

fn with_diff(
    path: &str,
    before_source: &str,
    after_source: &str,
    changed_sections: &[&str],
) -> Result<FeaStudyDocumentOperationResult, String> {
    let mut result = summarize_fea_study_document_contract(path, after_source)?;
    let before = summarize_fea_study(&parse_document(before_source)?);
    let after = summarize_fea_study(&parse_document(after_source)?);
    result.diff = Some(FeaStudyDocumentDiff {
        changed_sections: changed_sections
            .iter()
            .map(|section| (*section).to_string())
            .collect(),
        before,
        after,
        source_length_before: before_source.len(),
        source_length_after: after_source.len(),
    });
    Ok(result)
}

fn summarize_fea_study(document: &YamlValue) -> FeaStudySummary {
    let model_profile = read_yaml_string(document, &["model", "profile"]);
    let explicit_run_kind = read_yaml_string(document, &["run", "kind"]);
    let run_kind = explicit_run_kind.or_else(|| {
        model_profile
            .as_deref()
            .and_then(derive_fea_run_kind_from_profile)
            .map(str::to_string)
    });
    FeaStudySummary {
        study_id: read_yaml_string(document, &["id"]),
        geometry_path: read_yaml_string(document, &["geometry", "path"]),
        geometry_units: read_yaml_string(document, &["geometry", "units"]),
        model_profile,
        model_defaults: read_yaml_string(document, &["model", "defaults"]),
        mesh_profile: read_yaml_string(document, &["mesh", "profile"]),
        mesh_target_size: read_yaml_string(document, &["mesh", "target_size"]),
        mesh_max_elements: read_yaml_string(document, &["mesh", "max_elements"]),
        run_kind,
        run_backend: read_yaml_string(document, &["run", "backend"]),
    }
}

fn derive_fea_run_kind_from_profile(profile: &str) -> Option<&'static str> {
    AnalysisCreateModelProfile::from_snake_case(profile)
        .map(AnalysisCreateModelProfile::derived_run_kind)
        .map(AnalysisRunKind::as_snake_case)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FeaStudyReadinessProfile {
    Structural,
    Modal,
    Thermal,
    Electromagnetic,
    Acoustic,
    Cfd,
    Coupled,
}

impl FeaStudyReadinessProfile {
    fn from_summary(summary: &FeaStudySummary) -> Option<Self> {
        let profile =
            AnalysisCreateModelProfile::from_snake_case(summary.model_profile.as_deref()?.trim())?;
        if profile.catalog_entry().family == "coupled physics" {
            return Some(Self::Coupled);
        }
        match profile.derived_run_kind() {
            AnalysisRunKind::LinearStatic
            | AnalysisRunKind::Transient
            | AnalysisRunKind::Nonlinear => Some(Self::Structural),
            AnalysisRunKind::Modal => Some(Self::Modal),
            AnalysisRunKind::Thermal => Some(Self::Thermal),
            AnalysisRunKind::Electromagnetic => Some(Self::Electromagnetic),
            AnalysisRunKind::Acoustic => Some(Self::Acoustic),
            AnalysisRunKind::Cfd => Some(Self::Cfd),
            AnalysisRunKind::Cht | AnalysisRunKind::Fsi => Some(Self::Coupled),
        }
    }

    const fn material_blocker(self) -> &'static str {
        match self {
            Self::Structural | Self::Modal => {
                "Define at least one material for the structural model."
            }
            Self::Thermal => "Define at least one material or medium for the thermal model.",
            Self::Electromagnetic => {
                "Define at least one material or medium for the electromagnetic model."
            }
            Self::Acoustic => "Define at least one material or medium for the acoustic model.",
            Self::Cfd => "Define at least one fluid material or medium for the CFD model.",
            Self::Coupled => "Define materials or media for each active coupled-physics domain.",
        }
    }

    const fn material_assignment_blocker(self) -> &'static str {
        match self {
            Self::Structural | Self::Modal => {
                "Assign material to at least one CAD region or selector."
            }
            Self::Thermal => "Assign thermal material or medium to at least one region.",
            Self::Electromagnetic => {
                "Assign electromagnetic material or medium to at least one region."
            }
            Self::Acoustic => "Assign acoustic material or medium to at least one region.",
            Self::Cfd => "Assign fluid material or medium to at least one flow region.",
            Self::Coupled => "Assign materials or media to the coupled-physics regions.",
        }
    }

    const fn boundary_blocker(self) -> Option<&'static str> {
        match self {
            Self::Structural => Some("Add at least one structural boundary condition."),
            Self::Modal => None,
            Self::Thermal => Some("Add at least one thermal boundary condition."),
            Self::Electromagnetic => Some("Add at least one electromagnetic boundary condition."),
            Self::Acoustic => Some("Add at least one acoustic boundary condition."),
            Self::Cfd => Some("Add at least one CFD boundary condition."),
            Self::Coupled => Some("Add boundary or interface conditions for the coupled domains."),
        }
    }

    const fn driving_condition_blocker(self) -> Option<&'static str> {
        match self {
            Self::Structural => Some("Add at least one structural driving condition."),
            Self::Modal => None,
            Self::Thermal => Some("Add at least one thermal source or driving condition."),
            Self::Electromagnetic => {
                Some("Add at least one electromagnetic source or driving condition.")
            }
            Self::Acoustic => Some("Add at least one acoustic source or excitation."),
            Self::Cfd => Some("Add at least one flow driver or source."),
            Self::Coupled => Some("Add driving conditions or sources for the coupled domains."),
        }
    }
}

fn study_readiness(
    summary: &FeaStudySummary,
    materials: &[FeaStudyMaterialEntry],
    material_assignments: &[FeaStudyMaterialAssignmentEntry],
    boundary_conditions: &[FeaStudyBoundaryConditionEntry],
    driving_conditions: &[FeaStudyDrivingConditionEntry],
) -> FeaStudyReadiness {
    let mut blockers = Vec::new();
    if summary.geometry_path.as_deref().map_or(true, str::is_empty) {
        blockers.push("Set geometry.path to the CAD or mesh file for this study.");
    }
    let model_profile = summary.model_profile.as_deref();
    if model_profile.map_or(true, str::is_empty) {
        blockers.push("Choose a physics model profile.");
    }
    if summary.run_backend.as_deref().map_or(true, str::is_empty) {
        blockers.push("Choose a run backend.");
    }
    if model_profile.is_some() && FeaStudyReadinessProfile::from_summary(summary).is_none() {
        blockers.push("Choose a supported physics model profile.");
    }
    if let Some(profile) = FeaStudyReadinessProfile::from_summary(summary) {
        if materials.is_empty() {
            blockers.push(profile.material_blocker());
        }
        if !has_material_assignment(material_assignments) {
            blockers.push(profile.material_assignment_blocker());
        }
        if let Some(blocker) = profile.boundary_blocker() {
            if !has_boundary_condition(boundary_conditions) {
                blockers.push(blocker);
            }
        }
        if let Some(blocker) = profile.driving_condition_blocker() {
            if !has_driving_condition(driving_conditions) {
                blockers.push(blocker);
            }
        }
    }
    FeaStudyReadiness {
        ready_to_solve: blockers.is_empty(),
        blockers,
    }
}

fn has_material_assignment(assignments: &[FeaStudyMaterialAssignmentEntry]) -> bool {
    assignments.iter().any(|assignment| {
        assignment
            .region
            .as_deref()
            .is_some_and(|value| !value.is_empty())
            && assignment
                .material
                .as_deref()
                .is_some_and(|value| !value.is_empty())
    })
}

fn has_boundary_condition(boundary_conditions: &[FeaStudyBoundaryConditionEntry]) -> bool {
    boundary_conditions.iter().any(|boundary| {
        boundary
            .region
            .as_deref()
            .is_some_and(|value| !value.is_empty())
            && boundary
                .kind
                .as_deref()
                .is_some_and(|value| !value.is_empty())
    })
}

fn has_driving_condition(driving_conditions: &[FeaStudyDrivingConditionEntry]) -> bool {
    driving_conditions.iter().any(|condition| {
        condition
            .region
            .as_deref()
            .is_some_and(|value| !value.is_empty())
            && condition
                .condition_type
                .as_deref()
                .is_some_and(|value| !value.is_empty())
    })
}

fn material_document(input: &JsonValue) -> Result<YamlValue, String> {
    let mut material = Mapping::new();
    if let Some(name) = read_string(input, &["name"]) {
        material.insert(yaml_string("name"), yaml_string(name));
    }
    insert_optional_material_group(
        &mut material,
        "mechanical",
        input,
        &[
            (
                "youngs_modulus_pa",
                &["youngs_modulus_pa", "youngsModulusPa"],
            ),
            ("poisson_ratio", &["poisson_ratio", "poissonRatio"]),
            (
                "density_kg_per_m3",
                &[
                    "mechanical_density_kg_per_m3",
                    "mechanicalDensityKgPerM3",
                    "density_kg_per_m3",
                    "densityKgPerM3",
                ],
            ),
        ],
    )?;
    insert_optional_material_group(
        &mut material,
        "thermal",
        input,
        &[
            (
                "thermal_conductivity_w_per_m_k",
                &[
                    "thermal_conductivity_w_per_m_k",
                    "thermalConductivityWPerMK",
                ],
            ),
            (
                "specific_heat_j_per_kg_k",
                &["specific_heat_j_per_kg_k", "specificHeatJPerKgK"],
            ),
            (
                "density_kg_per_m3",
                &["thermal_density_kg_per_m3", "thermalDensityKgPerM3"],
            ),
        ],
    )?;
    insert_optional_material_group(
        &mut material,
        "electromagnetic",
        input,
        &[
            (
                "relative_permittivity",
                &["relative_permittivity", "relativePermittivity"],
            ),
            (
                "relative_permeability",
                &["relative_permeability", "relativePermeability"],
            ),
            (
                "electrical_conductivity_s_per_m",
                &[
                    "electrical_conductivity_s_per_m",
                    "electricalConductivitySPerM",
                ],
            ),
        ],
    )?;
    insert_optional_material_group(
        &mut material,
        "acoustic",
        input,
        &[
            (
                "density_kg_per_m3",
                &["acoustic_density_kg_per_m3", "acousticDensityKgPerM3"],
            ),
            (
                "speed_of_sound_m_per_s",
                &["speed_of_sound_m_per_s", "speedOfSoundMPerS"],
            ),
            (
                "acoustic_impedance_pa_s_per_m",
                &["acoustic_impedance_pa_s_per_m", "acousticImpedancePaSPerM"],
            ),
        ],
    )?;
    insert_optional_material_group(
        &mut material,
        "fluid",
        input,
        &[
            (
                "density_kg_per_m3",
                &["fluid_density_kg_per_m3", "fluidDensityKgPerM3"],
            ),
            (
                "dynamic_viscosity_pa_s",
                &["dynamic_viscosity_pa_s", "dynamicViscosityPaS"],
            ),
            (
                "reference_pressure_pa",
                &["reference_pressure_pa", "referencePressurePa"],
            ),
            (
                "reference_temperature_k",
                &["reference_temperature_k", "referenceTemperatureK"],
            ),
        ],
    )?;
    if material.len() <= usize::from(material.contains_key(yaml_string("name"))) {
        return Err("material requires at least one physics material property".to_string());
    }
    Ok(YamlValue::Mapping(material))
}

fn insert_optional_material_group(
    material: &mut Mapping,
    group_name: &str,
    input: &JsonValue,
    fields: &[(&str, &[&str])],
) -> Result<(), String> {
    let mut group = Mapping::new();
    for (output_key, input_keys) in fields {
        let Some(value) = read_number(input, input_keys) else {
            continue;
        };
        group.insert(yaml_string(output_key), yaml_number(value)?);
    }
    if !group.is_empty() {
        material.insert(yaml_string(group_name), YamlValue::Mapping(group));
    }
    Ok(())
}

fn constraint_document(input: &JsonValue) -> Result<YamlValue, String> {
    Ok(YamlValue::Mapping(
        mapping_from_pairs([
            (
                "id",
                yaml_string(required_id(input, &["constraint_id", "constraintId"])?),
            ),
            (
                "region",
                yaml_string(required_id(input, &["region_id", "regionId"])?),
            ),
            ("kind", yaml_string(required_id(input, &["kind"])?)),
            ("rx", optional_yaml_number(input, &["rx"])?),
            ("ry", optional_yaml_number(input, &["ry"])?),
            ("rz", optional_yaml_number(input, &["rz"])?),
            (
                "temperature_k",
                optional_yaml_number(input, &["temperature_k", "temperatureK"])?,
            ),
            (
                "pressure_pa",
                optional_yaml_number(input, &["pressure_pa", "pressurePa"])?,
            ),
        ])
        .into_iter()
        .filter(|(_, value)| !matches!(value, YamlValue::Null))
        .collect(),
    ))
}

fn driving_condition_document(input: &JsonValue) -> Result<YamlValue, String> {
    Ok(YamlValue::Mapping(
        mapping_from_pairs([
            (
                "id",
                yaml_string(required_id(
                    input,
                    &["driving_condition_id", "drivingConditionId"],
                )?),
            ),
            (
                "region",
                yaml_string(required_id(input, &["region_id", "regionId"])?),
            ),
            ("type", yaml_string(required_id(input, &["type", "kind"])?)),
            ("vector", optional_vector(input, "vector")?),
            ("force", optional_vector(input, "force")?),
            ("moment", optional_vector(input, "moment")?),
            ("point", optional_vector(input, "point")?),
            (
                "magnitude_pa",
                optional_yaml_number(input, &["magnitude_pa", "magnitudePa"])?,
            ),
            (
                "pressure_pa",
                optional_yaml_number(input, &["pressure_pa", "pressurePa"])?,
            ),
            (
                "velocity_m_per_s",
                optional_yaml_number(input, &["velocity_m_per_s", "velocityMPerS"])?,
            ),
            (
                "mass_flow_kg_per_s",
                optional_yaml_number(input, &["mass_flow_kg_per_s", "massFlowKgPerS"])?,
            ),
            (
                "volumetric_flow_m3_per_s",
                optional_yaml_number(input, &["volumetric_flow_m3_per_s", "volumetricFlowM3PerS"])?,
            ),
            (
                "temperature_k",
                optional_yaml_number(input, &["temperature_k", "temperatureK"])?,
            ),
            (
                "heat_flux_w_per_m2",
                optional_yaml_number(input, &["heat_flux_w_per_m2", "heatFluxWPerM2"])?,
            ),
            (
                "current_a",
                optional_yaml_number(input, &["current_a", "currentA"])?,
            ),
            (
                "voltage_v",
                optional_yaml_number(input, &["voltage_v", "voltageV"])?,
            ),
            (
                "power_w",
                optional_yaml_number(input, &["power_w", "powerW"])?,
            ),
            (
                "frequency_hz",
                optional_yaml_number(input, &["frequency_hz", "frequencyHz"])?,
            ),
            (
                "phase_rad",
                optional_yaml_number(input, &["phase_rad", "phaseRad"])?,
            ),
            (
                "amplitude_scale",
                optional_yaml_number(input, &["amplitude_scale", "amplitudeScale"])?,
            ),
            (
                "volumetric_w_per_m3",
                optional_yaml_number(input, &["volumetric_w_per_m3", "volumetricWPerM3"])?,
            ),
        ])
        .into_iter()
        .filter(|(_, value)| !matches!(value, YamlValue::Null))
        .collect(),
    ))
}

fn output_documents(input: &JsonValue) -> Result<Vec<YamlValue>, String> {
    if let Some(outputs) = input.get("outputs").and_then(JsonValue::as_array) {
        return outputs.iter().map(output_document).collect();
    }
    Ok(vec![output_document(input)?])
}

fn output_document(input: &JsonValue) -> Result<YamlValue, String> {
    Ok(YamlValue::Mapping(
        mapping_from_pairs([
            (
                "id",
                yaml_string(required_id(input, &["output_id", "outputId", "id"])?),
            ),
            (
                "field",
                yaml_string(required_string(
                    input,
                    &["field_id", "fieldId", "field", "name"],
                )?),
            ),
            (
                "location",
                optional_yaml_string(input, &["location", "target"]),
            ),
            ("kind", optional_yaml_string(input, &["kind", "type"])),
        ])
        .into_iter()
        .filter(|(_, value)| !matches!(value, YamlValue::Null))
        .collect(),
    ))
}

fn mesh_document(input: &JsonValue) -> Result<YamlValue, String> {
    let mut mesh = Mapping::new();
    insert_optional_string(&mut mesh, "backend", input, &["backend"]);
    insert_optional_string(&mut mesh, "kind", input, &["kind"]);
    insert_optional_string(&mut mesh, "element", input, &["element"]);
    insert_optional_string(
        &mut mesh,
        "element_order",
        input,
        &["element_order", "elementOrder"],
    );
    insert_optional_string(&mut mesh, "profile", input, &["profile"]);
    insert_optional_number(
        &mut mesh,
        "max_elements",
        input,
        &["max_elements", "maxElements"],
    )?;
    insert_optional_number(
        &mut mesh,
        "target_size",
        input,
        &["target_size", "targetSize"],
    )?;
    insert_optional_number(&mut mesh, "min_size", input, &["min_size", "minSize"])?;
    insert_optional_number(&mut mesh, "max_size", input, &["max_size", "maxSize"])?;
    insert_optional_number(
        &mut mesh,
        "growth_rate",
        input,
        &["growth_rate", "growthRate"],
    )?;

    let refinement = mesh_refinement_document(input)?;
    if let YamlValue::Mapping(refinement) = &refinement {
        if !refinement.is_empty() {
            mesh.insert(
                yaml_string("refinement"),
                YamlValue::Mapping(refinement.clone()),
            );
        }
    }

    let validation = mesh_validation_document(input)?;
    if let YamlValue::Mapping(validation) = &validation {
        if !validation.is_empty() {
            mesh.insert(
                yaml_string("validation"),
                YamlValue::Mapping(validation.clone()),
            );
        }
    }

    Ok(YamlValue::Mapping(mesh))
}

fn mesh_refinement_document(input: &JsonValue) -> Result<YamlValue, String> {
    let mut refinement = Mapping::new();
    insert_optional_string(
        &mut refinement,
        "strategy",
        input,
        &["refinement_strategy", "refinementStrategy"],
    );
    insert_optional_number(
        &mut refinement,
        "max_iterations",
        input,
        &["refinement_max_iterations", "refinementMaxIterations"],
    )?;

    let mut convergence = Mapping::new();
    insert_optional_number(
        &mut convergence,
        "field_change_tolerance",
        input,
        &["field_change_tolerance", "fieldChangeTolerance"],
    )?;
    insert_optional_number(
        &mut convergence,
        "energy_change_tolerance",
        input,
        &["energy_change_tolerance", "energyChangeTolerance"],
    )?;
    insert_optional_number(
        &mut convergence,
        "residual_tolerance",
        input,
        &["residual_tolerance", "residualTolerance"],
    )?;
    if !convergence.is_empty() {
        refinement.insert(yaml_string("convergence"), YamlValue::Mapping(convergence));
    }

    let mut focus = Mapping::new();
    insert_optional_string(
        &mut focus,
        "loads",
        input,
        &["refinement_load_focus", "refinementLoadFocus"],
    );
    insert_optional_string(
        &mut focus,
        "constraints",
        input,
        &["refinement_constraint_focus", "refinementConstraintFocus"],
    );
    insert_optional_string(
        &mut focus,
        "interfaces",
        input,
        &["refinement_interface_focus", "refinementInterfaceFocus"],
    );
    insert_optional_bool(
        &mut focus,
        "curvature",
        input,
        &["refinement_curvature", "refinementCurvature"],
    )?;
    insert_optional_bool(
        &mut focus,
        "small_features",
        input,
        &["refinement_small_features", "refinementSmallFeatures"],
    )?;
    if !focus.is_empty() {
        refinement.insert(yaml_string("focus"), YamlValue::Mapping(focus));
    }

    Ok(YamlValue::Mapping(refinement))
}

fn mesh_validation_document(input: &JsonValue) -> Result<YamlValue, String> {
    let mut validation = Mapping::new();
    insert_optional_string(
        &mut validation,
        "coverage",
        input,
        &["validation_coverage", "validationCoverage"],
    );
    insert_optional_string(
        &mut validation,
        "quality",
        input,
        &["validation_quality", "validationQuality"],
    );
    insert_optional_number(
        &mut validation,
        "min_bounds_coverage_ratio",
        input,
        &["min_bounds_coverage_ratio", "minBoundsCoverageRatio"],
    )?;
    insert_optional_number(
        &mut validation,
        "min_volume_coverage_ratio",
        input,
        &["min_volume_coverage_ratio", "minVolumeCoverageRatio"],
    )?;
    insert_optional_number(
        &mut validation,
        "min_boundary_area_ratio",
        input,
        &["min_boundary_area_ratio", "minBoundaryAreaRatio"],
    )?;
    insert_optional_number(
        &mut validation,
        "min_boundary_face_recovery_ratio",
        input,
        &[
            "min_boundary_face_recovery_ratio",
            "minBoundaryFaceRecoveryRatio",
        ],
    )?;
    insert_optional_number(
        &mut validation,
        "min_boundary_edge_recovery_ratio",
        input,
        &[
            "min_boundary_edge_recovery_ratio",
            "minBoundaryEdgeRecoveryRatio",
        ],
    )?;
    insert_optional_number(
        &mut validation,
        "max_volume_components",
        input,
        &["max_volume_components", "maxVolumeComponents"],
    )?;
    Ok(YamlValue::Mapping(validation))
}

fn parse_document(source: &str) -> Result<YamlValue, String> {
    if source.trim().is_empty() {
        return Ok(YamlValue::Mapping(Mapping::new()));
    }
    serde_yaml::from_str(source).map_err(|err| format!("failed to parse FEA YAML: {err}"))
}

fn serialize_document(document: &YamlValue) -> Result<String, String> {
    serde_yaml::to_string(document)
        .map(|source| {
            let without_marker = source.strip_prefix("---\n").unwrap_or(&source);
            if without_marker.ends_with('\n') {
                without_marker.to_string()
            } else {
                format!("{without_marker}\n")
            }
        })
        .map_err(|err| format!("failed to serialize FEA YAML: {err}"))
}

fn insert_map_entry(
    document: &mut YamlValue,
    block_key: &str,
    entry_key: &str,
    value: YamlValue,
) -> Result<(), String> {
    let root = ensure_mapping(document);
    let block_name = block_key;
    let block_key = yaml_string(block_key);
    let block = root
        .entry(block_key)
        .or_insert_with(|| YamlValue::Mapping(Mapping::new()));
    let block = ensure_mapping(block);
    let key = yaml_string(entry_key);
    if block.contains_key(&key) {
        return Err(format!("{block_name} entry already exists: {entry_key}"));
    }
    block.insert(key, value);
    Ok(())
}

fn replace_map_entry(
    document: &mut YamlValue,
    block_key: &str,
    entry_key: &str,
    value: YamlValue,
) -> Result<(), String> {
    let block = require_mapping_block(document, block_key)?;
    let key = yaml_string(entry_key);
    if !block.contains_key(&key) {
        return Err(format!("{block_key} entry does not exist: {entry_key}"));
    }
    block.insert(key, value);
    Ok(())
}

fn remove_map_entry(
    document: &mut YamlValue,
    block_key: &str,
    entry_key: &str,
) -> Result<(), String> {
    let block = require_mapping_block(document, block_key)?;
    if block.remove(&yaml_string(entry_key)).is_none() {
        return Err(format!("{block_key} entry does not exist: {entry_key}"));
    }
    Ok(())
}

fn replace_optional_map_entry(document: &mut YamlValue, block_key: &str, value: YamlValue) {
    ensure_mapping(document).insert(yaml_string(block_key), value);
}

fn append_sequence_entry(document: &mut YamlValue, block_key: &str, value: YamlValue) {
    let root = ensure_mapping(document);
    let block_key = yaml_string(block_key);
    let block = root
        .entry(block_key)
        .or_insert_with(|| YamlValue::Sequence(Vec::new()));
    match block {
        YamlValue::Sequence(items) => items.push(value),
        _ => {
            *block = YamlValue::Sequence(vec![value]);
        }
    }
}

fn replace_sequence_block(document: &mut YamlValue, block_key: &str, values: Vec<YamlValue>) {
    ensure_mapping(document).insert(yaml_string(block_key), YamlValue::Sequence(values));
}

fn append_sequence_entry_with_unique_id(
    document: &mut YamlValue,
    block_key: &str,
    value: YamlValue,
) -> Result<(), String> {
    append_sequence_entry_with_unique_id_labeled(document, block_key, block_key, value)
}

fn append_sequence_entry_with_unique_id_labeled(
    document: &mut YamlValue,
    block_key: &str,
    public_block_label: &str,
    value: YamlValue,
) -> Result<(), String> {
    let id = nested_scalar(&value, &["id"])
        .ok_or_else(|| format!("{public_block_label} entry requires id"))?;
    let root = ensure_mapping(document);
    let block_key_value = yaml_string(block_key);
    let block = root
        .entry(block_key_value)
        .or_insert_with(|| YamlValue::Sequence(Vec::new()));
    let YamlValue::Sequence(items) = block else {
        return Err(format!("{public_block_label} must be a sequence"));
    };
    if items
        .iter()
        .any(|item| nested_scalar(item, &["id"]).as_deref() == Some(id.as_str()))
    {
        return Err(format!("{public_block_label} entry already exists: {id}"));
    }
    items.push(value);
    Ok(())
}

fn replace_sequence_entry_by_id(
    document: &mut YamlValue,
    block_key: &str,
    id: &str,
    value: YamlValue,
) -> Result<(), String> {
    replace_sequence_entry_by_id_labeled(document, block_key, block_key, id, value)
}

fn replace_sequence_entry_by_id_labeled(
    document: &mut YamlValue,
    block_key: &str,
    public_block_label: &str,
    id: &str,
    value: YamlValue,
) -> Result<(), String> {
    let items = require_sequence_block_labeled(document, block_key, public_block_label)?;
    let Some(existing) = items
        .iter_mut()
        .find(|item| nested_scalar(item, &["id"]).as_deref() == Some(id))
    else {
        return Err(format!("{public_block_label} entry does not exist: {id}"));
    };
    *existing = value;
    Ok(())
}

fn remove_sequence_entry_by_id(
    document: &mut YamlValue,
    block_key: &str,
    id: &str,
) -> Result<(), String> {
    remove_sequence_entry_by_id_labeled(document, block_key, block_key, id)
}

fn remove_sequence_entry_by_id_labeled(
    document: &mut YamlValue,
    block_key: &str,
    public_block_label: &str,
    id: &str,
) -> Result<(), String> {
    let items = require_sequence_block_labeled(document, block_key, public_block_label)?;
    let initial_len = items.len();
    items.retain(|item| nested_scalar(item, &["id"]).as_deref() != Some(id));
    if items.len() == initial_len {
        return Err(format!("{public_block_label} entry does not exist: {id}"));
    }
    Ok(())
}

fn ensure_mapping(value: &mut YamlValue) -> &mut Mapping {
    if !matches!(value, YamlValue::Mapping(_)) {
        *value = YamlValue::Mapping(Mapping::new());
    }
    match value {
        YamlValue::Mapping(mapping) => mapping,
        _ => unreachable!(),
    }
}

fn require_mapping_block<'a>(
    document: &'a mut YamlValue,
    block_key: &str,
) -> Result<&'a mut Mapping, String> {
    let root = ensure_mapping(document);
    let block = root
        .get_mut(&yaml_string(block_key))
        .ok_or_else(|| format!("{block_key} block does not exist"))?;
    match block {
        YamlValue::Mapping(mapping) => Ok(mapping),
        _ => Err(format!("{block_key} must be a mapping")),
    }
}

fn require_sequence_block_labeled<'a>(
    document: &'a mut YamlValue,
    block_key: &str,
    public_block_label: &str,
) -> Result<&'a mut Vec<YamlValue>, String> {
    let root = ensure_mapping(document);
    let block = root
        .get_mut(&yaml_string(block_key))
        .ok_or_else(|| format!("{public_block_label} block does not exist"))?;
    match block {
        YamlValue::Sequence(items) => Ok(items),
        _ => Err(format!("{public_block_label} must be a sequence")),
    }
}

fn map_entries<T, F>(document: &YamlValue, key: &str, mut map: F) -> Vec<T>
where
    F: FnMut(String, &YamlValue) -> T,
{
    document
        .as_mapping()
        .and_then(|root| root.get(&yaml_string(key)))
        .and_then(YamlValue::as_mapping)
        .map(|items| {
            items
                .iter()
                .filter_map(|(key, value)| key.as_str().map(|key| map(key.to_string(), value)))
                .collect()
        })
        .unwrap_or_default()
}

fn sequence_entries<T, F>(document: &YamlValue, key: &str, mut map: F) -> Vec<T>
where
    F: FnMut(usize, &YamlValue) -> T,
{
    document
        .as_mapping()
        .and_then(|root| root.get(&yaml_string(key)))
        .and_then(YamlValue::as_sequence)
        .map(|items| {
            items
                .iter()
                .enumerate()
                .map(|(index, value)| map(index, value))
                .collect()
        })
        .unwrap_or_default()
}

fn nested_scalar(value: &YamlValue, path: &[&str]) -> Option<String> {
    let mut current = value;
    for key in path {
        current = current.as_mapping()?.get(&yaml_string(key))?;
    }
    scalar_to_string(current)
}

fn read_yaml_string(value: &YamlValue, path: &[&str]) -> Option<String> {
    nested_scalar(value, path)
}

fn scalar_to_string(value: &YamlValue) -> Option<String> {
    match value {
        YamlValue::String(value) => Some(value.clone()),
        YamlValue::Number(value) => Some(value.to_string()),
        YamlValue::Bool(value) => Some(value.to_string()),
        YamlValue::Sequence(items) => Some(format!(
            "[{}]",
            items
                .iter()
                .filter_map(scalar_to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )),
        _ => None,
    }
}

fn mapping_from_pairs<const N: usize>(pairs: [(&str, YamlValue); N]) -> Mapping {
    pairs
        .into_iter()
        .map(|(key, value)| (yaml_string(key), value))
        .collect()
}

fn yaml_string(value: impl AsRef<str>) -> YamlValue {
    YamlValue::String(value.as_ref().to_string())
}

fn yaml_number(value: f64) -> Result<YamlValue, String> {
    serde_yaml::to_value(value).map_err(|err| err.to_string())
}

fn optional_yaml_number(input: &JsonValue, keys: &[&str]) -> Result<YamlValue, String> {
    read_number(input, keys)
        .map(yaml_number)
        .transpose()
        .map(|value| value.unwrap_or(YamlValue::Null))
}

fn optional_yaml_string(input: &JsonValue, keys: &[&str]) -> YamlValue {
    read_string(input, keys)
        .map(yaml_string)
        .unwrap_or(YamlValue::Null)
}

fn material_group_summary(
    value: &YamlValue,
    group_name: &str,
    fields: &[(&str, &str)],
) -> Option<String> {
    let parts = fields
        .iter()
        .filter_map(|(label, field)| {
            nested_scalar(value, &[group_name, field])
                .map(|value| format!("{label} {}", format_engineering_value(&value)))
        })
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| parts.join(", "))
}

fn insert_optional_string(
    mapping: &mut Mapping,
    output_key: &str,
    input: &JsonValue,
    input_keys: &[&str],
) {
    if let Some(value) = read_string(input, input_keys) {
        mapping.insert(yaml_string(output_key), yaml_string(value));
    }
}

fn insert_optional_number(
    mapping: &mut Mapping,
    output_key: &str,
    input: &JsonValue,
    input_keys: &[&str],
) -> Result<(), String> {
    if let Some(value) = read_number(input, input_keys) {
        mapping.insert(yaml_string(output_key), yaml_number(value)?);
    }
    Ok(())
}

fn insert_optional_bool(
    mapping: &mut Mapping,
    output_key: &str,
    input: &JsonValue,
    input_keys: &[&str],
) -> Result<(), String> {
    for key in input_keys {
        let Some(value) = input.get(*key) else {
            continue;
        };
        let Some(value) = value.as_bool() else {
            return Err(format!("{key} must be a boolean"));
        };
        mapping.insert(yaml_string(output_key), YamlValue::Bool(value));
        return Ok(());
    }
    Ok(())
}

fn optional_vector(input: &JsonValue, key: &str) -> Result<YamlValue, String> {
    let Some(value) = input.get(key) else {
        return Ok(YamlValue::Null);
    };
    let Some(items) = value.as_array() else {
        return Err(format!("{key} must be a 3-number vector"));
    };
    if items.len() != 3 {
        return Err(format!("{key} must be a 3-number vector"));
    }
    let values = items
        .iter()
        .map(|value| {
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("{key} must be a 3-number vector"))
                .and_then(yaml_number)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(YamlValue::Sequence(values))
}

fn required_string<'a>(input: &'a JsonValue, keys: &[&str]) -> Result<&'a str, String> {
    read_string(input, keys).ok_or_else(|| format!("missing required field: {}", keys.join(" or ")))
}

fn read_string<'a>(input: &'a JsonValue, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .filter_map(|key| input.get(*key)?.as_str())
        .map(str::trim)
        .find(|value| !value.is_empty())
}

fn required_id(input: &JsonValue, keys: &[&str]) -> Result<String, String> {
    let value = required_string(input, keys)?;
    normalize_yaml_key(value).ok_or_else(|| {
        format!(
            "{} must include at least one ASCII letter, number, or underscore",
            keys.join(" or ")
        )
    })
}

fn read_number(input: &JsonValue, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|key| {
        let value = input.get(*key)?;
        let numeric = value
            .as_f64()
            .or_else(|| value.as_str().and_then(|value| value.trim().parse().ok()))?;
        numeric.is_finite().then_some(numeric)
    })
}

fn normalize_yaml_key(value: impl AsRef<str>) -> Option<String> {
    let mut normalized = String::new();
    let mut last_was_separator = false;
    for ch in value.as_ref().trim().chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            normalized.push(ch);
            last_was_separator = false;
        } else if !last_was_separator {
            normalized.push('_');
            last_was_separator = true;
        }
    }
    let normalized = normalized.trim_matches('_');
    if normalized.is_empty() {
        return None;
    }
    let mut normalized = normalized.to_string();
    if normalized
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_digit())
    {
        normalized.insert(0, '_');
    }
    Some(normalized)
}

fn study_id_from_path(path: &str) -> String {
    let file_name = path
        .rsplit('/')
        .find(|part| !part.is_empty())
        .unwrap_or(path);
    let stem = file_name
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(file_name);
    normalize_yaml_key(stem).unwrap_or_else(|| "study".to_string())
}

fn section_for_operation(operation: FeaStudyDocumentOperation) -> &'static str {
    match operation {
        FeaStudyDocumentOperation::AddRegion
        | FeaStudyDocumentOperation::UpdateRegion
        | FeaStudyDocumentOperation::RemoveRegion => "regions",
        FeaStudyDocumentOperation::AddMaterial | FeaStudyDocumentOperation::UpdateMaterial => {
            "materials"
        }
        FeaStudyDocumentOperation::AssignMaterial => "material_assignments",
        FeaStudyDocumentOperation::AddConstraint
        | FeaStudyDocumentOperation::UpdateConstraint
        | FeaStudyDocumentOperation::RemoveConstraint => "boundary_conditions",
        FeaStudyDocumentOperation::AddDrivingCondition
        | FeaStudyDocumentOperation::UpdateDrivingCondition
        | FeaStudyDocumentOperation::RemoveDrivingCondition => "driving_conditions",
        FeaStudyDocumentOperation::SetMesh => "mesh",
        FeaStudyDocumentOperation::SetOutputs => "outputs",
        FeaStudyDocumentOperation::Create | FeaStudyDocumentOperation::GetSummary => "document",
    }
}

fn format_engineering_value(value: &str) -> String {
    let Ok(numeric) = value.parse::<f64>() else {
        return value.to_string();
    };
    if numeric.abs() >= 1_000_000_000.0 {
        format!("{:.3} GPa", numeric / 1_000_000_000.0)
    } else if numeric.abs() >= 1_000_000.0 {
        format!("{:.3} MPa", numeric / 1_000_000.0)
    } else {
        format!("{numeric:.3}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::contracts::ANALYSIS_PHYSICS_PROFILE_CATALOG;

    #[test]
    fn operation_names_are_the_rust_owned_contract() {
        let parsed = FEA_STUDY_DOCUMENT_OPERATION_NAMES
            .iter()
            .map(|name| {
                let operation =
                    FeaStudyDocumentOperation::from_str(name).expect("operation name should parse");
                operation.as_str()
            })
            .collect::<Vec<_>>();

        assert_eq!(parsed, FEA_STUDY_DOCUMENT_OPERATION_NAMES);
        assert_eq!(
            FeaStudyDocumentOperation::from_str("patch_yaml")
                .expect_err("unknown operation should fail"),
            "unsupported finite element study operation: patch_yaml"
        );
    }

    #[test]
    fn creates_and_summarizes_study_document() {
        let output = apply_fea_study_document_operation(
            "create",
            "/project/Wall Hook 30mm.fea",
            None,
            serde_json::json!({
                "geometry_path": "WallHook_30mm.stp",
                "geometry_units": "millimeter",
                "model_profile": "linear_static_structural"
            }),
        )
        .expect("create should succeed");

        assert!(output.write);
        assert!(output.source.contains("kind: study"));
        assert_eq!(
            output.result.summary.study_id.as_deref(),
            Some("Wall_Hook_30mm")
        );
        assert_eq!(
            output.result.summary.geometry_path.as_deref(),
            Some("WallHook_30mm.stp")
        );
        assert_eq!(
            output.result.summary.run_kind.as_deref(),
            Some("linear_static")
        );
        assert!(!output.result.readiness.ready_to_solve);
        assert_eq!(
            output.result.readiness.blockers[0],
            "Define at least one material for the structural model."
        );
    }

    #[test]
    fn document_authoring_accepts_every_supported_physics_profile() {
        for entry in ANALYSIS_PHYSICS_PROFILE_CATALOG {
            let profile = serde_json::to_value(entry.profile)
                .expect("profile serializes")
                .as_str()
                .expect("profile serializes as string")
                .to_string();
            let expected_run_kind = serde_json::to_value(entry.profile.derived_run_kind())
                .expect("run kind serializes")
                .as_str()
                .expect("run kind serializes as string")
                .to_string();
            assert!(
                !entry.default_outputs.is_empty(),
                "{profile} should publish default output fields"
            );

            let created = apply_fea_study_document_operation(
                "create",
                &format!("/project/{profile}.fea"),
                None,
                serde_json::json!({
                    "geometry_path": "geometry/assembly.step",
                    "geometry_units": "millimeter",
                    "model_profile": profile
                }),
            )
            .unwrap_or_else(|err| panic!("create {profile}: {err}"));

            assert_eq!(
                created.result.summary.model_profile.as_deref(),
                Some(profile.as_str())
            );
            assert_eq!(
                created.result.summary.run_kind.as_deref(),
                Some(expected_run_kind.as_str())
            );
            assert!(
                FeaStudyReadinessProfile::from_summary(&created.result.summary).is_some(),
                "{profile} should map to a readiness profile"
            );
        }
    }

    #[test]
    fn create_uses_neutral_study_id_fallback_for_unusable_path_stem() {
        let output = apply_fea_study_document_operation(
            "create",
            "/project/---.fea",
            None,
            serde_json::json!({
                "geometry_path": "part.step",
                "model_profile": "linear_static_structural"
            }),
        )
        .expect("create should succeed with path-derived fallback");

        assert_eq!(output.result.summary.study_id.as_deref(), Some("study"));
        assert!(output.source.contains("id: study"));
        assert!(!output.source.contains("fea_study"));
    }

    #[test]
    fn rejects_ids_that_do_not_normalize_to_stable_yaml_keys() {
        let invalid_study = apply_fea_study_document_operation(
            "create",
            "/project/valid.fea",
            None,
            serde_json::json!({
                "study_id": "---",
                "geometry_path": "part.step",
                "model_profile": "linear_static_structural"
            }),
        )
        .expect_err("invalid explicit study id should fail");
        assert!(invalid_study.contains("study_id must include"));

        let created = apply_fea_study_document_operation(
            "create",
            "/project/valid.fea",
            None,
            serde_json::json!({
                "geometry_path": "part.step",
                "model_profile": "linear_static_structural"
            }),
        )
        .expect("create");
        let invalid_region = apply_fea_study_document_operation(
            "add_region",
            "/project/valid.fea",
            Some(&created.source),
            serde_json::json!({"region_id": "---", "selector": "region:face_1"}),
        )
        .expect_err("invalid region id should fail");
        assert!(invalid_region.contains("region_id or regionId must include"));
    }

    #[test]
    fn applies_typed_document_operations() {
        let created = apply_fea_study_document_operation(
            "create",
            "/project/bracket.fea",
            None,
            serde_json::json!({
                "geometry_path": "bracket.step",
                "model_profile": "linear_static_structural"
            }),
        )
        .expect("create");
        let with_region = apply_fea_study_document_operation(
            "add_region",
            "/project/bracket.fea",
            Some(&created.source),
            serde_json::json!({"region_id": "fixed_face", "selector": "region:face_1"}),
        )
        .expect("region");
        let with_material = apply_fea_study_document_operation(
            "add_material",
            "/project/bracket.fea",
            Some(&with_region.source),
            serde_json::json!({
                "material_id": "aluminum",
                "name": "Aluminum",
                "youngs_modulus_pa": 69e9,
                "poisson_ratio": 0.33,
                "density_kg_per_m3": 2700.0
            }),
        )
        .expect("material");
        let with_assignment = apply_fea_study_document_operation(
            "assign_material",
            "/project/bracket.fea",
            Some(&with_material.source),
            serde_json::json!({"region_id": "fixed_face", "material_id": "aluminum"}),
        )
        .expect("assignment");
        let with_driving_condition = apply_fea_study_document_operation(
            "add_driving_condition",
            "/project/bracket.fea",
            Some(&with_assignment.source),
            serde_json::json!({
                "driving_condition_id": "tip_force",
                "region_id": "fixed_face",
                "type": "force",
                "vector": [0.0, 0.0, -100.0]
            }),
        )
        .expect("driving condition");
        let changed = apply_fea_study_document_operation(
            "update_driving_condition",
            "/project/bracket.fea",
            Some(&with_driving_condition.source),
            serde_json::json!({
                "driving_condition_id": "tip_force",
                "region_id": "fixed_face",
                "type": "force",
                "vector": [0.0, 0.0, -250.0]
            }),
        )
        .expect("update driving condition");

        assert_eq!(changed.result.counts.regions, 1);
        assert_eq!(changed.result.counts.materials, 1);
        assert_eq!(changed.result.counts.material_assignments, 1);
        assert_eq!(changed.result.counts.driving_conditions, 1);
        assert_eq!(
            changed.result.driving_conditions[0].id.as_deref(),
            Some("tip_force")
        );
        assert_eq!(
            changed.result.driving_conditions[0].value.as_deref(),
            Some("[0.0, 0.0, -250.0]")
        );
        assert!(!changed.result.readiness.ready_to_solve);
        assert_eq!(
            changed.result.readiness.blockers[0],
            "Add at least one structural boundary condition."
        );
        assert_eq!(
            changed.result.diff.as_ref().expect("diff").changed_sections[0],
            "driving_conditions"
        );
    }

    #[test]
    fn driving_condition_operations_preserve_non_structural_source_values() {
        let source = r#"
id: duct_acoustics
geometry:
  path: geometry/duct.step
model:
  profile: acoustic_harmonic
run:
  backend: cpu
regions:
  speaker_port:
    selector: region:inlet
materials:
  air:
    name: Air
material_assignments:
  - region: speaker_port
    material: air
boundary_conditions:
  - id: rigid_wall
    region: speaker_port
    kind: acoustic_impedance
"#;
        let changed = apply_fea_study_document_operation(
            "add_driving_condition",
            "/project/duct.fea",
            Some(source),
            serde_json::json!({
                "driving_condition_id": "speaker_pressure",
                "region_id": "speaker_port",
                "type": "acoustic_pressure",
                "pressure_pa": 2.0,
                "frequency_hz": 440.0,
                "phase_rad": 0.5,
                "amplitude_scale": 1.25
            }),
        )
        .expect("add acoustic source");

        assert!(changed.source.contains("type: acoustic_pressure"));
        assert!(changed.source.contains("pressure_pa: 2.0"));
        assert!(changed.source.contains("frequency_hz: 440.0"));
        assert!(changed.source.contains("phase_rad: 0.5"));
        assert_eq!(
            changed.result.driving_conditions[0]
                .condition_type
                .as_deref(),
            Some("acoustic_pressure")
        );
        assert_eq!(
            changed.result.driving_conditions[0].value.as_deref(),
            Some("2.0")
        );
        assert!(changed.result.readiness.ready_to_solve);
    }

    #[test]
    fn material_operations_preserve_family_specific_media_properties() {
        let source = r#"
id: media_test
geometry:
  path: geometry/part.step
model:
  profile: cfd_transient
run:
  backend: cpu
"#;

        let cases = [
            (
                "thermal_medium",
                serde_json::json!({
                    "material_id": "thermal_medium",
                    "name": "Thermal medium",
                    "thermal_conductivity_w_per_m_k": 16.2,
                    "specific_heat_j_per_kg_k": 500.0,
                    "thermal_density_kg_per_m3": 7800.0
                }),
                "thermal:",
                "thermal_conductivity_w_per_m_k: 16.2",
                "thermalSummary",
            ),
            (
                "dielectric",
                serde_json::json!({
                    "material_id": "dielectric",
                    "name": "Dielectric",
                    "relative_permittivity": 3.2,
                    "relative_permeability": 1.0,
                    "electrical_conductivity_s_per_m": 0.01
                }),
                "electromagnetic:",
                "relative_permittivity: 3.2",
                "electromagneticSummary",
            ),
            (
                "air_acoustic",
                serde_json::json!({
                    "material_id": "air_acoustic",
                    "name": "Air acoustic",
                    "acoustic_density_kg_per_m3": 1.225,
                    "speed_of_sound_m_per_s": 343.0
                }),
                "acoustic:",
                "speed_of_sound_m_per_s: 343.0",
                "acousticSummary",
            ),
            (
                "air_flow",
                serde_json::json!({
                    "material_id": "air_flow",
                    "name": "Air flow",
                    "fluid_density_kg_per_m3": 1.225,
                    "dynamic_viscosity_pa_s": 0.0000181,
                    "reference_pressure_pa": 101325.0
                }),
                "fluid:",
                "dynamic_viscosity_pa_s: 0.0000181",
                "fluidSummary",
            ),
        ];

        for (material_id, input, expected_group, expected_value, expected_summary_key) in cases {
            let changed = apply_fea_study_document_operation(
                "add_material",
                "/project/media.fea",
                Some(source),
                input,
            )
            .unwrap_or_else(|err| panic!("add material {material_id}: {err}"));
            assert!(
                changed.source.contains(expected_group),
                "{material_id} should write {expected_group}"
            );
            assert!(
                changed.source.contains(expected_value),
                "{material_id} should write {expected_value}"
            );
            let payload = serde_json::to_value(&changed.result.materials[0]).expect("material");
            assert!(
                payload
                    .get(expected_summary_key)
                    .and_then(JsonValue::as_str)
                    .is_some_and(|summary| !summary.is_empty()),
                "{material_id} should expose {expected_summary_key}"
            );
            assert!(
                changed.result.materials[0].youngs_modulus_pa.is_none(),
                "{material_id} should not require structural Young's modulus"
            );
            assert!(
                changed.result.materials[0].poisson_ratio.is_none(),
                "{material_id} should not require structural Poisson ratio"
            );
        }
    }

    #[test]
    fn material_operation_rejects_empty_material_definition() {
        let err = apply_fea_study_document_operation(
            "add_material",
            "/project/media.fea",
            Some("id: media_test\n"),
            serde_json::json!({
                "material_id": "empty",
                "name": "Empty"
            }),
        )
        .expect_err("empty material definitions should be rejected");
        assert!(err.contains("at least one physics material property"));
    }

    #[test]
    fn readiness_uses_profile_specific_requirements_across_supported_physics() {
        let cases = [
            (
                "linear_static_structural",
                "Add at least one structural driving condition.",
                false,
            ),
            (
                "transient_structural",
                "Add at least one structural driving condition.",
                false,
            ),
            (
                "nonlinear_structural",
                "Add at least one structural driving condition.",
                false,
            ),
            (
                "modal_structural",
                "Define at least one material for the structural model.",
                true,
            ),
            (
                "thermal_standalone",
                "Add at least one thermal source or driving condition.",
                false,
            ),
            (
                "electromagnetic_static",
                "Add at least one electromagnetic source or driving condition.",
                false,
            ),
            (
                "acoustic_harmonic",
                "Add at least one acoustic source or excitation.",
                false,
            ),
            (
                "cfd_steady_state",
                "Add at least one flow driver or source.",
                false,
            ),
            (
                "cfd_transient",
                "Add at least one flow driver or source.",
                false,
            ),
            (
                "thermo_mechanical_coupled",
                "Add driving conditions or sources for the coupled domains.",
                false,
            ),
            (
                "electro_thermal_coupled",
                "Add driving conditions or sources for the coupled domains.",
                false,
            ),
            (
                "cht_coupled",
                "Add driving conditions or sources for the coupled domains.",
                false,
            ),
            (
                "fsi_coupled",
                "Add driving conditions or sources for the coupled domains.",
                false,
            ),
        ];

        for (profile, expected_blocker, modal_without_driver) in cases {
            let source = format!(
                "version: 1\nkind: study\nid: profile_test\ngeometry:\n  path: part.step\nmodel:\n  profile: {profile}\nrun:\n  backend: cpu\n"
            );
            let result = summarize_fea_study_document_contract("/project/profile.fea", &source)
                .expect("summary");
            assert!(
                !result.readiness.ready_to_solve,
                "{profile} should not be ready with only geometry/profile/backend"
            );
            assert!(
                result.readiness.blockers.contains(&expected_blocker),
                "{profile} blockers should contain {expected_blocker:?}, got {:?}",
                result.readiness.blockers
            );
            assert_eq!(
                result.readiness.blockers.iter().any(|blocker| blocker
                    .contains("driving condition")
                    || blocker.contains("source")
                    || blocker.contains("driver")
                    || blocker.contains("excitation")),
                !modal_without_driver,
                "{profile} driving/source gate mismatch: {:?}",
                result.readiness.blockers
            );
        }
    }

    #[test]
    fn typed_authoring_creates_edits_and_checks_every_supported_physics_profile() {
        let cases = vec![
            (
                "linear_static_structural",
                "linear_static",
                serde_json::json!({
                    "youngs_modulus_pa": 70e9,
                    "poisson_ratio": 0.33,
                    "mechanical_density_kg_per_m3": 2700.0
                }),
                serde_json::json!({"kind": "fixed"}),
                Some(serde_json::json!({
                    "type": "force",
                    "vector": [0.0, 0.0, -125.0]
                })),
                "structural.displacement",
                false,
            ),
            (
                "thermo_mechanical_coupled",
                "transient",
                serde_json::json!({
                    "youngs_modulus_pa": 70e9,
                    "poisson_ratio": 0.33,
                    "mechanical_density_kg_per_m3": 2700.0,
                    "thermal_conductivity_w_per_m_k": 205.0,
                    "specific_heat_j_per_kg_k": 900.0,
                    "thermal_density_kg_per_m3": 2700.0
                }),
                serde_json::json!({"kind": "thermal_prescribed_temperature", "temperature_k": 293.15}),
                Some(serde_json::json!({
                    "type": "heat_source",
                    "volumetric_w_per_m3": 12000.0
                })),
                "thermal.temperature",
                false,
            ),
            (
                "electro_thermal_coupled",
                "transient",
                serde_json::json!({
                    "thermal_conductivity_w_per_m_k": 205.0,
                    "specific_heat_j_per_kg_k": 900.0,
                    "thermal_density_kg_per_m3": 2700.0,
                    "relative_permittivity": 1.0,
                    "relative_permeability": 1.0,
                    "electrical_conductivity_s_per_m": 3.5e7
                }),
                serde_json::json!({"kind": "electric_potential"}),
                Some(serde_json::json!({
                    "type": "current_source",
                    "current_a": 4.0
                })),
                "electro_thermal.temperature",
                true,
            ),
            (
                "thermal_standalone",
                "thermal",
                serde_json::json!({
                    "thermal_conductivity_w_per_m_k": 15.0,
                    "specific_heat_j_per_kg_k": 460.0,
                    "thermal_density_kg_per_m3": 7800.0
                }),
                serde_json::json!({"kind": "thermal_prescribed_temperature", "temperature_k": 293.15}),
                Some(serde_json::json!({
                    "type": "heat_source",
                    "volumetric_w_per_m3": 10000.0
                })),
                "thermal.temperature",
                true,
            ),
            (
                "modal_structural",
                "modal",
                serde_json::json!({
                    "youngs_modulus_pa": 200e9,
                    "poisson_ratio": 0.29,
                    "mechanical_density_kg_per_m3": 7850.0
                }),
                serde_json::json!({"kind": "fixed"}),
                None,
                "structural.mode_shapes",
                false,
            ),
            (
                "acoustic_harmonic",
                "acoustic",
                serde_json::json!({
                    "acoustic_density_kg_per_m3": 1.225,
                    "speed_of_sound_m_per_s": 343.0
                }),
                serde_json::json!({"kind": "acoustic_impedance", "pressure_pa": 0.0}),
                Some(serde_json::json!({
                    "type": "acoustic_pressure",
                    "pressure_pa": 2.0,
                    "frequency_hz": 440.0
                })),
                "acoustic.pressure",
                true,
            ),
            (
                "transient_structural",
                "transient",
                serde_json::json!({
                    "youngs_modulus_pa": 70e9,
                    "poisson_ratio": 0.33,
                    "mechanical_density_kg_per_m3": 2700.0
                }),
                serde_json::json!({"kind": "fixed"}),
                Some(serde_json::json!({
                    "type": "force",
                    "vector": [0.0, 0.0, -125.0]
                })),
                "structural.displacement",
                false,
            ),
            (
                "nonlinear_structural",
                "nonlinear",
                serde_json::json!({
                    "youngs_modulus_pa": 70e9,
                    "poisson_ratio": 0.33,
                    "mechanical_density_kg_per_m3": 2700.0
                }),
                serde_json::json!({"kind": "fixed"}),
                Some(serde_json::json!({
                    "type": "force",
                    "vector": [0.0, 0.0, -125.0]
                })),
                "structural.displacement",
                false,
            ),
            (
                "electromagnetic_static",
                "electromagnetic",
                serde_json::json!({
                    "relative_permittivity": 3.0,
                    "relative_permeability": 1.0,
                    "electrical_conductivity_s_per_m": 5.8e7
                }),
                serde_json::json!({"kind": "electric_potential"}),
                Some(serde_json::json!({
                    "type": "current_source",
                    "current_a": 2.5
                })),
                "em.electric_field",
                true,
            ),
            (
                "cfd_steady_state",
                "cfd",
                serde_json::json!({
                    "fluid_density_kg_per_m3": 1.225,
                    "dynamic_viscosity_pa_s": 0.0000181,
                    "reference_pressure_pa": 101325.0
                }),
                serde_json::json!({"kind": "wall"}),
                Some(serde_json::json!({
                    "type": "flow_velocity",
                    "velocity_m_per_s": 12.0
                })),
                "fluid.velocity",
                true,
            ),
            (
                "cfd_transient",
                "cfd",
                serde_json::json!({
                    "fluid_density_kg_per_m3": 1.225,
                    "dynamic_viscosity_pa_s": 0.0000181,
                    "reference_pressure_pa": 101325.0
                }),
                serde_json::json!({"kind": "wall"}),
                Some(serde_json::json!({
                    "type": "flow_velocity",
                    "velocity_m_per_s": 12.0
                })),
                "fluid.velocity",
                true,
            ),
            (
                "cht_coupled",
                "cht",
                serde_json::json!({
                    "fluid_density_kg_per_m3": 1.225,
                    "dynamic_viscosity_pa_s": 0.0000181,
                    "reference_pressure_pa": 101325.0,
                    "thermal_conductivity_w_per_m_k": 205.0,
                    "specific_heat_j_per_kg_k": 900.0,
                    "thermal_density_kg_per_m3": 2700.0
                }),
                serde_json::json!({"kind": "thermal_convection", "temperature_k": 293.15}),
                Some(serde_json::json!({
                    "type": "heat_source",
                    "volumetric_w_per_m3": 8000.0
                })),
                "thermal.temperature",
                true,
            ),
            (
                "fsi_coupled",
                "fsi",
                serde_json::json!({
                    "youngs_modulus_pa": 70e9,
                    "poisson_ratio": 0.33,
                    "mechanical_density_kg_per_m3": 2700.0,
                    "fluid_density_kg_per_m3": 1.225,
                    "dynamic_viscosity_pa_s": 0.0000181,
                    "reference_pressure_pa": 101325.0
                }),
                serde_json::json!({"kind": "wall"}),
                Some(serde_json::json!({
                    "type": "flow_velocity",
                    "velocity_m_per_s": 12.0
                })),
                "fluid.pressure",
                false,
            ),
        ];

        let covered_profiles = cases
            .iter()
            .map(|(profile, ..)| *profile)
            .collect::<std::collections::BTreeSet<_>>();
        let catalog_profiles = ANALYSIS_PHYSICS_PROFILE_CATALOG
            .iter()
            .map(|entry| entry.profile.as_snake_case())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            covered_profiles, catalog_profiles,
            "typed authoring readiness coverage must track the full supported physics profile catalog"
        );

        for (
            profile,
            run_kind,
            material_input,
            constraint_input,
            driver_input,
            output_field,
            expect_no_structural_material,
        ) in cases
        {
            let path = format!("/project/{profile}.fea");
            let created = apply_fea_study_document_operation(
                "create",
                &path,
                None,
                serde_json::json!({
                    "geometry_path": "geometry/assembly.step",
                    "geometry_units": "millimeter",
                    "model_profile": profile
                }),
            )
            .unwrap_or_else(|err| panic!("create {profile}: {err}"));
            assert_eq!(
                created.result.summary.model_profile.as_deref(),
                Some(profile)
            );
            assert_eq!(created.result.summary.run_kind.as_deref(), Some(run_kind));
            assert!(
                created.source.contains(&format!("profile: {profile}")),
                "{profile} source should retain model profile"
            );
            assert!(
                created.source.contains(&format!("kind: {run_kind}")),
                "{profile} source should persist derived run kind"
            );

            let with_region = apply_fea_study_document_operation(
                "add_region",
                &path,
                Some(&created.source),
                serde_json::json!({
                    "region_id": "domain",
                    "selector": "region:body_1"
                }),
            )
            .unwrap_or_else(|err| panic!("add region {profile}: {err}"));

            let mut material_input = material_input;
            material_input["material_id"] = JsonValue::String("medium".to_string());
            material_input["name"] = JsonValue::String("Study medium".to_string());
            let with_material = apply_fea_study_document_operation(
                "add_material",
                &path,
                Some(&with_region.source),
                material_input,
            )
            .unwrap_or_else(|err| panic!("add material {profile}: {err}"));

            let with_assignment = apply_fea_study_document_operation(
                "assign_material",
                &path,
                Some(&with_material.source),
                serde_json::json!({
                    "region_id": "domain",
                    "material_id": "medium"
                }),
            )
            .unwrap_or_else(|err| panic!("assign material {profile}: {err}"));

            let mut constraint_input = constraint_input;
            constraint_input["constraint_id"] = JsonValue::String("boundary".to_string());
            constraint_input["region_id"] = JsonValue::String("domain".to_string());
            let with_constraint = apply_fea_study_document_operation(
                "add_constraint",
                &path,
                Some(&with_assignment.source),
                constraint_input,
            )
            .unwrap_or_else(|err| panic!("add constraint {profile}: {err}"));

            let with_driver = if let Some(mut driver_input) = driver_input {
                driver_input["driving_condition_id"] = JsonValue::String("driver".to_string());
                driver_input["region_id"] = JsonValue::String("domain".to_string());
                apply_fea_study_document_operation(
                    "add_driving_condition",
                    &path,
                    Some(&with_constraint.source),
                    driver_input,
                )
                .unwrap_or_else(|err| panic!("add driver {profile}: {err}"))
            } else {
                with_constraint
            };

            let changed = apply_fea_study_document_operation(
                "set_outputs",
                &path,
                Some(&with_driver.source),
                serde_json::json!({
                    "output_id": "primary",
                    "field_id": output_field,
                    "location": "domain"
                }),
            )
            .unwrap_or_else(|err| panic!("set outputs {profile}: {err}"));

            assert!(
                changed.result.readiness.ready_to_solve,
                "{profile} should be ready after family-appropriate authoring: {:?}",
                changed.result.readiness.blockers
            );
            assert_eq!(changed.result.counts.regions, 1);
            assert_eq!(changed.result.counts.materials, 1);
            assert_eq!(changed.result.counts.material_assignments, 1);
            assert_eq!(changed.result.counts.boundary_conditions, 1);
            assert_eq!(
                changed.result.outputs[0].field.as_deref(),
                Some(output_field)
            );
            if expect_no_structural_material {
                assert!(
                    changed.result.materials[0].youngs_modulus_pa.is_none(),
                    "{profile} should not need structural Young's modulus"
                );
                assert!(
                    changed.result.materials[0].poisson_ratio.is_none(),
                    "{profile} should not need structural Poisson ratio"
                );
            }
        }
    }

    #[test]
    fn create_rejects_unknown_physics_profile() {
        let err = apply_fea_study_document_operation(
            "create",
            "/project/unknown.fea",
            None,
            serde_json::json!({
                "geometry_path": "part.step",
                "model_profile": "linear_stress_only"
            }),
        )
        .expect_err("create should reject unknown profiles");

        assert!(err.contains("unsupported physics model profile: linear_stress_only"));
    }

    #[test]
    fn updates_removes_and_sets_mesh_without_text_patching() {
        let created = apply_fea_study_document_operation(
            "create",
            "/project/bracket.fea",
            None,
            serde_json::json!({
                "geometry_path": "bracket.step",
                "model_profile": "linear_static_structural"
            }),
        )
        .expect("create");
        let with_region = apply_fea_study_document_operation(
            "add_region",
            "/project/bracket.fea",
            Some(&created.source),
            serde_json::json!({"region_id": "load_face", "selector": "region:face_1"}),
        )
        .expect("region");
        let duplicate_region = apply_fea_study_document_operation(
            "add_region",
            "/project/bracket.fea",
            Some(&with_region.source),
            serde_json::json!({"region_id": "load_face", "selector": "region:face_2"}),
        )
        .expect_err("add should not replace an existing region");
        assert!(duplicate_region.contains("regions entry already exists: load_face"));

        let updated_region = apply_fea_study_document_operation(
            "update_region",
            "/project/bracket.fea",
            Some(&with_region.source),
            serde_json::json!({"region_id": "load_face", "selector": "region:face_2"}),
        )
        .expect("update region");
        assert_eq!(
            updated_region.result.regions[0].selector.as_deref(),
            Some("region:face_2")
        );

        let missing_driving_condition_update = apply_fea_study_document_operation(
            "update_driving_condition",
            "/project/bracket.fea",
            Some(&updated_region.source),
            serde_json::json!({
                "driving_condition_id": "missing_force",
                "region_id": "load_face",
                "type": "force",
                "vector": [0.0, 0.0, -250.0]
            }),
        )
        .expect_err("update should require an existing driving condition");
        assert!(
            missing_driving_condition_update.contains("driving_conditions block does not exist")
                || missing_driving_condition_update
                    .contains("driving_conditions entry does not exist")
        );
        assert!(!missing_driving_condition_update.contains("loads"));

        let with_material = apply_fea_study_document_operation(
            "add_material",
            "/project/bracket.fea",
            Some(&updated_region.source),
            serde_json::json!({
                "material_id": "steel",
                "youngs_modulus_pa": 200e9,
                "poisson_ratio": 0.29
            }),
        )
        .expect("material");
        let updated_material = apply_fea_study_document_operation(
            "update_material",
            "/project/bracket.fea",
            Some(&with_material.source),
            serde_json::json!({
                "material_id": "steel",
                "name": "Structural steel",
                "youngs_modulus_pa": 210e9,
                "poisson_ratio": 0.30
            }),
        )
        .expect("update material");
        assert_eq!(
            updated_material.result.materials[0].name.as_deref(),
            Some("Structural steel")
        );

        let with_constraint = apply_fea_study_document_operation(
            "add_constraint",
            "/project/bracket.fea",
            Some(&updated_material.source),
            serde_json::json!({
                "constraint_id": "fixed_mount",
                "region_id": "load_face",
                "kind": "fixed"
            }),
        )
        .expect("constraint");
        let duplicate_constraint = apply_fea_study_document_operation(
            "add_constraint",
            "/project/bracket.fea",
            Some(&with_constraint.source),
            serde_json::json!({
                "constraint_id": "fixed_mount",
                "region_id": "load_face",
                "kind": "fixed"
            }),
        )
        .expect_err("add should not duplicate constraints");
        assert!(duplicate_constraint.contains("boundary_conditions entry already exists"));

        let updated_constraint = apply_fea_study_document_operation(
            "update_constraint",
            "/project/bracket.fea",
            Some(&with_constraint.source),
            serde_json::json!({
                "constraint_id": "fixed_mount",
                "region_id": "load_face",
                "kind": "prescribed_displacement",
                "rz": 0.0
            }),
        )
        .expect("update constraint");
        assert_eq!(
            updated_constraint.result.boundary_conditions[0]
                .kind
                .as_deref(),
            Some("prescribed_displacement")
        );

        let with_driving_condition = apply_fea_study_document_operation(
            "add_driving_condition",
            "/project/bracket.fea",
            Some(&updated_constraint.source),
            serde_json::json!({
                "driving_condition_id": "tip_force",
                "region_id": "load_face",
                "type": "force",
                "vector": [0.0, -10.0, 0.0]
            }),
        )
        .expect("add driving condition");
        let without_driving_condition = apply_fea_study_document_operation(
            "remove_driving_condition",
            "/project/bracket.fea",
            Some(&with_driving_condition.source),
            serde_json::json!({"driving_condition_id": "tip_force"}),
        )
        .expect("remove driving condition");
        assert_eq!(
            without_driving_condition.result.counts.driving_conditions,
            0
        );

        let with_mesh = apply_fea_study_document_operation(
            "set_mesh",
            "/project/bracket.fea",
            Some(&without_driving_condition.source),
            serde_json::json!({
                "backend": "auto",
                "kind": "solid",
                "element": "tetrahedron4",
                "profile": "balanced",
                "max_elements": 50000,
                "target_size": 0.002,
                "refinement_strategy": "adaptive",
                "refinement_max_iterations": 2,
                "field_change_tolerance": 0.05,
                "validation_coverage": "strict"
            }),
        )
        .expect("set mesh");
        assert!(with_mesh.source.contains("mesh:"));
        assert!(with_mesh.source.contains("max_elements: 50000"));
        assert_eq!(
            with_mesh
                .result
                .diff
                .as_ref()
                .expect("diff")
                .changed_sections[0],
            "mesh"
        );

        let with_outputs = apply_fea_study_document_operation(
            "set_outputs",
            "/project/bracket.fea",
            Some(&with_mesh.source),
            serde_json::json!({
                "outputs": [
                    {
                        "id": "displacement_view",
                        "field": "structural.displacement",
                        "location": "nodes",
                        "kind": "vector"
                    },
                    {
                        "id": "stress_view",
                        "field_id": "structural.von_mises",
                        "location": "elements",
                        "kind": "scalar"
                    }
                ]
            }),
        )
        .expect("set outputs");
        assert_eq!(with_outputs.result.counts.outputs, 2);
        assert_eq!(
            with_outputs.result.outputs[0].field.as_deref(),
            Some("structural.displacement")
        );
        assert_eq!(
            with_outputs
                .result
                .diff
                .as_ref()
                .expect("diff")
                .changed_sections[0],
            "outputs"
        );

        let without_constraint = apply_fea_study_document_operation(
            "remove_constraint",
            "/project/bracket.fea",
            Some(&with_outputs.source),
            serde_json::json!({"constraint_id": "fixed_mount"}),
        )
        .expect("remove constraint");
        assert_eq!(without_constraint.result.counts.boundary_conditions, 0);

        let without_region = apply_fea_study_document_operation(
            "remove_region",
            "/project/bracket.fea",
            Some(&without_constraint.source),
            serde_json::json!({"region_id": "load_face"}),
        )
        .expect("remove region");
        assert_eq!(without_region.result.counts.regions, 0);
    }
}
