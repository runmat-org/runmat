//! Canonical document construction and domain-aware mutations.

use runmat_analysis_core::{
    AnalysisBoundary, AnalysisConnector, AnalysisConstraint, AnalysisInterface, AnalysisSequence,
    ExcitationAction, ExcitationCase, FormulationRequirement, InitialCondition,
    MaterialAcousticModel, MaterialElectricalModel, MaterialMechanicalModel, MaterialPlasticModel,
    MaterialThermalModel, PhysicalSignal, RandomExcitationSpectrum, ResponseSpectrum,
};
use runmat_analysis_fea::ComputeBackend;
use serde::Serialize;
use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;

use super::super::contracts::AnalysisSolverOptions;
use super::yaml::*;
use super::*;

pub(super) fn mutate(
    operation: FeaStudyDocumentOperation,
    document: &mut YamlValue,
    input: JsonValue,
) -> Result<&'static str, String> {
    let add = |candidate| operation == candidate;
    let edits_model = !matches!(
        operation,
        FeaStudyDocumentOperation::SetSequence
            | FeaStudyDocumentOperation::SetMesh
            | FeaStudyDocumentOperation::ClearMesh
            | FeaStudyDocumentOperation::Create
            | FeaStudyDocumentOperation::GetSummary
    );
    let model_data = if edits_model {
        inline_model_data_mut(document)?
    } else {
        document
    };
    match operation {
        FeaStudyDocumentOperation::AddRegion | FeaStudyDocumentOperation::UpdateRegion => {
            let value: FeaStudyRegionInput = decode(input, operation.as_str())?;
            validate_id(&value.region_id, "region_id")?;
            let id = value.region_id.clone();
            mutate_map(
                model_data,
                "regions",
                id,
                to_yaml(&RegionValue {
                    selector: value.selector,
                })?,
                add(FeaStudyDocumentOperation::AddRegion),
            )?;
            Ok("regions")
        }
        FeaStudyDocumentOperation::RemoveRegion => {
            remove_map(model_data, "regions", &decode_id(input, "region")?)?;
            Ok("regions")
        }
        FeaStudyDocumentOperation::AddMaterial | FeaStudyDocumentOperation::UpdateMaterial => {
            let value: FeaStudyMaterialInput = decode(input, operation.as_str())?;
            validate_id(&value.material_id, "material_id")?;
            let id = value.material_id.clone();
            mutate_map(
                model_data,
                "materials",
                id,
                to_yaml(&MaterialValue::from(value))?,
                add(FeaStudyDocumentOperation::AddMaterial),
            )?;
            Ok("materials")
        }
        FeaStudyDocumentOperation::RemoveMaterial => {
            remove_map(model_data, "materials", &decode_id(input, "material")?)?;
            Ok("materials")
        }
        FeaStudyDocumentOperation::AssignMaterial => {
            let value: FeaStudyMaterialAssignmentInput = decode(input, operation.as_str())?;
            let target = to_yaml(&value.target)?;
            append_pair(model_data, &target, &value.material, to_yaml(&value)?)?;
            Ok("material_assignments")
        }
        FeaStudyDocumentOperation::RemoveMaterialAssignment => {
            let value: AssignmentIdInput = decode(input, operation.as_str())?;
            let target = to_yaml(&value.target)?;
            remove_pair(model_data, &target, &value.material)?;
            Ok("material_assignments")
        }
        FeaStudyDocumentOperation::AddConstraint | FeaStudyDocumentOperation::UpdateConstraint => {
            let value: AnalysisConstraint = decode(input, operation.as_str())?;
            let id = value.constraint_id().to_string();
            mutate_sequence(
                model_data,
                "constraints",
                &id,
                to_yaml(&value)?,
                add(FeaStudyDocumentOperation::AddConstraint),
                constraint_id,
            )?;
            Ok("constraints")
        }
        FeaStudyDocumentOperation::RemoveConstraint => {
            remove_named(model_data, "constraints", input, constraint_id)
        }
        FeaStudyDocumentOperation::AddBoundary | FeaStudyDocumentOperation::UpdateBoundary => {
            let value: AnalysisBoundary = decode(input, operation.as_str())?;
            let id = value.boundary_id.clone();
            mutate_sequence(
                model_data,
                "boundaries",
                &id,
                to_yaml(&value)?,
                add(FeaStudyDocumentOperation::AddBoundary),
                boundary_id,
            )?;
            Ok("boundaries")
        }
        FeaStudyDocumentOperation::RemoveBoundary => {
            remove_named(model_data, "boundaries", input, boundary_id)
        }
        FeaStudyDocumentOperation::AddConnector | FeaStudyDocumentOperation::UpdateConnector => {
            let value: AnalysisConnector = decode(input, operation.as_str())?;
            let id = value.connector_id.clone();
            mutate_sequence(
                model_data,
                "connectors",
                &id,
                to_yaml(&value)?,
                add(FeaStudyDocumentOperation::AddConnector),
                connector_id,
            )?;
            Ok("connectors")
        }
        FeaStudyDocumentOperation::RemoveConnector => {
            remove_named(model_data, "connectors", input, connector_id)
        }
        FeaStudyDocumentOperation::AddInterface | FeaStudyDocumentOperation::UpdateInterface => {
            let value: AnalysisInterface = decode(input, operation.as_str())?;
            let id = value.interface_id.clone();
            mutate_sequence(
                model_data,
                "interfaces",
                &id,
                to_yaml(&value)?,
                add(FeaStudyDocumentOperation::AddInterface),
                interface_id,
            )?;
            Ok("interfaces")
        }
        FeaStudyDocumentOperation::RemoveInterface => {
            remove_named(model_data, "interfaces", input, interface_id)
        }
        FeaStudyDocumentOperation::AddSignal | FeaStudyDocumentOperation::UpdateSignal => {
            let value: PhysicalSignal = decode(input, operation.as_str())?;
            let id = value.signal_id().0.clone();
            mutate_sequence(
                model_data,
                "signals",
                &id,
                to_yaml(&value)?,
                add(FeaStudyDocumentOperation::AddSignal),
                signal_id,
            )?;
            Ok("signals")
        }
        FeaStudyDocumentOperation::RemoveSignal => {
            remove_named(model_data, "signals", input, signal_id)
        }
        FeaStudyDocumentOperation::AddExcitationCase
        | FeaStudyDocumentOperation::UpdateExcitationCase => {
            let value: ExcitationCase = decode(input, operation.as_str())?;
            let id = value.case_id.0.clone();
            mutate_sequence(
                model_data,
                "excitation_cases",
                &id,
                to_yaml(&value)?,
                add(FeaStudyDocumentOperation::AddExcitationCase),
                excitation_case_id,
            )?;
            Ok("excitation_cases")
        }
        FeaStudyDocumentOperation::RemoveExcitationCase => {
            remove_named(model_data, "excitation_cases", input, excitation_case_id)
        }
        FeaStudyDocumentOperation::AddExcitation | FeaStudyDocumentOperation::UpdateExcitation => {
            let value: ExcitationAction = decode(input, operation.as_str())?;
            let id = value.excitation().excitation_id.0.clone();
            mutate_sequence(
                model_data,
                "excitations",
                &id,
                to_yaml(&value)?,
                add(FeaStudyDocumentOperation::AddExcitation),
                excitation_id,
            )?;
            Ok("excitations")
        }
        FeaStudyDocumentOperation::RemoveExcitation => {
            remove_named(model_data, "excitations", input, excitation_id)
        }
        FeaStudyDocumentOperation::AddInitialCondition
        | FeaStudyDocumentOperation::UpdateInitialCondition => {
            let value: InitialCondition = decode(input, operation.as_str())?;
            let id = value.initial_condition_id().to_string();
            mutate_sequence(
                model_data,
                "initial_conditions",
                &id,
                to_yaml(&value)?,
                add(FeaStudyDocumentOperation::AddInitialCondition),
                initial_condition_id,
            )?;
            Ok("initial_conditions")
        }
        FeaStudyDocumentOperation::RemoveInitialCondition => remove_named(
            model_data,
            "initial_conditions",
            input,
            initial_condition_id,
        ),
        FeaStudyDocumentOperation::AddResponseSpectrum
        | FeaStudyDocumentOperation::UpdateResponseSpectrum => {
            let value: ResponseSpectrum = decode(input, operation.as_str())?;
            let id = value.spectrum_id.0.clone();
            mutate_sequence(
                model_data,
                "response_spectra",
                &id,
                to_yaml(&value)?,
                add(FeaStudyDocumentOperation::AddResponseSpectrum),
                response_spectrum_id,
            )?;
            Ok("response_spectra")
        }
        FeaStudyDocumentOperation::RemoveResponseSpectrum => {
            remove_named(model_data, "response_spectra", input, response_spectrum_id)
        }
        FeaStudyDocumentOperation::AddRandomExcitation
        | FeaStudyDocumentOperation::UpdateRandomExcitation => {
            let value: RandomExcitationSpectrum = decode(input, operation.as_str())?;
            let id = value.excitation_id.0.clone();
            mutate_sequence(
                model_data,
                "random_excitations",
                &id,
                to_yaml(&value)?,
                add(FeaStudyDocumentOperation::AddRandomExcitation),
                random_excitation_id,
            )?;
            Ok("random_excitations")
        }
        FeaStudyDocumentOperation::RemoveRandomExcitation => remove_named(
            model_data,
            "random_excitations",
            input,
            random_excitation_id,
        ),
        FeaStudyDocumentOperation::SetSequence => {
            let value: AnalysisSequence = decode(input, operation.as_str())?;
            set_value(model_data, "sequence", to_yaml(&value)?);
            Ok("sequence")
        }
        FeaStudyDocumentOperation::SetDomains => {
            let value: FeaStudyDomainsInput = decode(input, operation.as_str())?;
            set_value(model_data, "domains", to_yaml(&value)?);
            Ok("domains")
        }
        FeaStudyDocumentOperation::SetMesh => {
            let value: FeaStudyMeshInput = decode(input, operation.as_str())?;
            validate_mesh(&value)?;
            set_value(model_data, "mesh", to_yaml(&value)?);
            Ok("mesh")
        }
        FeaStudyDocumentOperation::ClearMesh => {
            remove_value(model_data, "mesh");
            Ok("mesh")
        }
        FeaStudyDocumentOperation::Create | FeaStudyDocumentOperation::GetSummary => unreachable!(),
    }
}

#[derive(Serialize)]
struct RegionValue {
    selector: FeaRegionSelector,
}

#[derive(Serialize)]
struct MaterialValue {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mass_density_kg_per_m3: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mechanical: Option<MaterialMechanicalModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    damping: Option<runmat_analysis_core::MechanicalDampingModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thermal: Option<MaterialThermalModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    acoustic: Option<MaterialAcousticModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    electrical: Option<MaterialElectricalModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    plastic: Option<MaterialPlasticModel>,
}

impl From<FeaStudyMaterialInput> for MaterialValue {
    fn from(value: FeaStudyMaterialInput) -> Self {
        Self {
            name: value.name,
            mass_density_kg_per_m3: value.mass_density_kg_per_m3,
            mechanical: value.mechanical,
            damping: value.damping,
            thermal: value.thermal,
            acoustic: value.acoustic,
            electrical: value.electrical,
            plastic: value.plastic,
        }
    }
}

#[derive(Serialize)]
struct CreateDocument {
    version: u32,
    kind: &'static str,
    id: String,
    geometry: FeaStudyGeometryInput,
    model: FeaStudyModelInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    mesh: Option<FeaStudyMeshInput>,
    sequence: AnalysisSequence,
    execution: Vec<FeaStudyExecutionInput>,
}

pub(super) fn create_source(path: &str, input: FeaStudyCreateInput) -> Result<String, String> {
    if input.geometry.path.trim().is_empty() {
        return Err("geometry.path must be non-empty".into());
    }
    let id = input.study_id.unwrap_or_else(|| study_id_from_path(path));
    validate_id(&id, "study_id")?;
    if let Some(mesh) = &input.mesh {
        validate_mesh(mesh)?;
    }
    let sequence = input.sequence;
    let execution = match input.execution {
        Some(execution) => execution,
        None => sequence
            .steps
            .iter()
            .map(|step| {
                Ok(FeaStudyExecutionInput {
                    step: step.step_id.clone(),
                    backend: ComputeBackend::Cpu,
                    requirement: FormulationRequirement::Production,
                    solver_options: Some(AnalysisSolverOptions::defaults_for(step)?),
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    };
    validate_execution(&sequence, &execution)?;
    serialize_document(&to_yaml(&CreateDocument {
        version: 1,
        kind: "study",
        id,
        geometry: input.geometry,
        model: input.model,
        mesh: input.mesh,
        sequence,
        execution,
    })?)
}

fn validate_execution(
    sequence: &AnalysisSequence,
    execution: &[FeaStudyExecutionInput],
) -> Result<(), String> {
    if execution.is_empty() {
        return Err("execution must contain at least one procedure step".to_string());
    }
    let mut seen = std::collections::BTreeSet::new();
    for entry in execution {
        if !seen.insert(entry.step.clone()) {
            return Err(format!(
                "execution contains duplicate procedure step `{}`",
                entry.step.as_str()
            ));
        }
        let step = sequence
            .steps
            .iter()
            .find(|candidate| candidate.step_id == entry.step)
            .ok_or_else(|| {
                format!(
                    "execution references procedure step `{}` outside sequence `{}`",
                    entry.step.as_str(),
                    sequence.sequence_id
                )
            })?;
        if entry
            .solver_options
            .as_ref()
            .is_some_and(|options| !options.supports(step))
        {
            return Err(format!(
                "execution solver does not support procedure step `{}`",
                entry.step.as_str()
            ));
        }
    }
    Ok(())
}

fn validate_mesh(mesh: &FeaStudyMeshInput) -> Result<(), String> {
    if mesh.maximum_elements == 0 {
        return Err("mesh.maximum_elements must be greater than zero".into());
    }
    for (field, value) in [
        ("target_edge_length_m", mesh.target_edge_length_m),
        (
            "maximum_chordal_deviation_m",
            mesh.maximum_chordal_deviation_m,
        ),
        ("maximum_grading_ratio", mesh.maximum_grading_ratio),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("mesh.{field} must be finite and greater than zero"));
        }
    }
    Ok(())
}

fn study_id_from_path(path: &str) -> String {
    let stem = std::path::Path::new(path)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("study");
    let sanitized = stem
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized
        .chars()
        .any(|character| character.is_ascii_alphanumeric())
    {
        sanitized
    } else {
        "study".into()
    }
}
