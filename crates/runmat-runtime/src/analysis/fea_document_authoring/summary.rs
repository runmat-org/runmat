//! Stable document summaries for editors and other authoring clients.

use runmat_analysis_core::{
    AnalysisBoundary, AnalysisConnector, AnalysisConstraint, AnalysisInterface, AnalysisSequence,
    ExcitationAction, ExcitationCase, InitialCondition, PhysicalSignal, RandomExcitationSpectrum,
    ResponseSpectrum,
};
use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;

use super::yaml::*;
use super::*;

pub fn summarize_fea_study_document(path: &str, source: &str) -> Result<JsonValue, String> {
    serde_json::to_value(summarize_fea_study_document_contract(path, source)?)
        .map_err(|error| error.to_string())
}

pub fn summarize_fea_study_document_contract(
    path: &str,
    source: &str,
) -> Result<FeaStudyDocumentOperationResult, String> {
    let document = parse_document(source)?;
    let summary = summarize(&document);
    let empty_model = YamlValue::Mapping(serde_yaml::Mapping::new());
    let model = inline_model_data(&document).unwrap_or(&empty_model);
    let regions = map_entries(model, "regions", |key, value| FeaStudyRegionEntry {
        key,
        selector: typed_at(value, &["selector"]),
    });
    let materials = map_entries(model, "materials", |key, value| FeaStudyMaterialEntry {
        key,
        name: scalar(value, &["name"]),
    });
    let material_assignments = sequence_values(model, "material_assignments")
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let selected_kind = ["beam", "shell", "solid", "cell"]
                .into_iter()
                .find(|kind| scalar(value, &["target", kind, "region"]).is_some())
                .map(str::to_owned);
            let region = ["beam", "shell", "solid", "cell"]
                .into_iter()
                .find_map(|kind| scalar(value, &["target", kind, "region"]));
            let all_kind = scalar(value, &["target"]).and_then(|target| match target.as_str() {
                "all_beams" => Some("beam".to_string()),
                "all_shells" => Some("shell".to_string()),
                "all_solids" => Some("solid".to_string()),
                "all_cells" => Some("cell".to_string()),
                _ => None,
            });
            let target_scope = if selected_kind.is_some() {
                Some("region".to_string())
            } else {
                all_kind.as_ref().map(|_| "all".to_string())
            };
            FeaStudyMaterialAssignmentEntry {
                index,
                target_kind: selected_kind.or(all_kind),
                target_scope,
                region,
                material: scalar(value, &["material"]),
            }
        })
        .collect::<Vec<_>>();
    let constraints = named_entries::<AnalysisConstraint>(model, "constraints", |value| {
        value.constraint_id().to_string()
    });
    let boundaries =
        named_entries::<AnalysisBoundary>(model, "boundaries", |value| value.boundary_id.clone());
    let connectors =
        named_entries::<AnalysisConnector>(model, "connectors", |value| value.connector_id.clone());
    let interfaces =
        named_entries::<AnalysisInterface>(model, "interfaces", |value| value.interface_id.clone());
    let signals =
        named_entries::<PhysicalSignal>(model, "signals", |value| value.signal_id().0.clone());
    let excitation_cases =
        named_entries::<ExcitationCase>(model, "excitation_cases", |value| value.case_id.0.clone());
    let excitations = named_entries::<ExcitationAction>(model, "excitations", |value| {
        value.excitation().excitation_id.0.clone()
    });
    let initial_conditions =
        named_entries::<InitialCondition>(model, "initial_conditions", |value| {
            value.initial_condition_id().to_string()
        });
    let response_spectra = named_entries::<ResponseSpectrum>(model, "response_spectra", |value| {
        value.spectrum_id.0.clone()
    });
    let random_excitations =
        named_entries::<RandomExcitationSpectrum>(model, "random_excitations", |value| {
            value.excitation_id.0.clone()
        });
    let sequence = typed_at::<AnalysisSequence>(&document, &["sequence"]).map(|value| {
        let first = value.steps.first();
        FeaStudySequenceSummary {
            sequence_id: value.sequence_id,
            step_count: value.steps.len(),
            primary_family: first.map(|step| step.family),
            primary_procedure: first.map(|step| step.procedure),
        }
    });
    let counts = FeaStudyDocumentCounts {
        regions: regions.len(),
        materials: materials.len(),
        material_assignments: material_assignments.len(),
        constraints: constraints.len(),
        boundaries: boundaries.len(),
        connectors: connectors.len(),
        interfaces: interfaces.len(),
        signals: signals.len(),
        excitation_cases: excitation_cases.len(),
        excitations: excitations.len(),
        initial_conditions: initial_conditions.len(),
        response_spectra: response_spectra.len(),
        random_excitations: random_excitations.len(),
        sequence_steps: sequence.as_ref().map_or(0, |value| value.step_count),
    };
    let readiness = readiness(&summary, &counts);
    Ok(FeaStudyDocumentOperationResult {
        path: path.into(),
        document_kind: scalar(&document, &["kind"]).unwrap_or_else(|| "study".into()),
        summary,
        regions,
        materials,
        material_assignments,
        constraints,
        boundaries,
        connectors,
        interfaces,
        signals,
        excitation_cases,
        excitations,
        initial_conditions,
        response_spectra,
        random_excitations,
        sequence,
        counts,
        readiness,
        diff: None,
    })
}

fn summarize(document: &YamlValue) -> FeaStudySummary {
    let sequence = typed_at::<AnalysisSequence>(document, &["sequence"]);
    let first = sequence.as_ref().and_then(|value| value.steps.first());
    FeaStudySummary {
        study_id: scalar(document, &["id"]),
        geometry_path: scalar(document, &["geometry", "path"]),
        geometry_units: typed_at(document, &["geometry", "units"]),
        model_source: yaml_at(document, &["model", "source"])
            .and_then(YamlValue::as_mapping)
            .and_then(|source| source.keys().next())
            .and_then(YamlValue::as_str)
            .map(str::to_string),
        mesh: typed_at(document, &["mesh"]),
        execution_backend: yaml_at(document, &["execution"])
            .and_then(YamlValue::as_sequence)
            .and_then(|execution| execution.first())
            .and_then(|entry| typed_at(entry, &["backend"])),
        sequence_id: sequence.as_ref().map(|value| value.sequence_id.clone()),
        primary_family: first.map(|step| step.family),
        primary_procedure: first.map(|step| step.procedure),
    }
}

fn readiness(summary: &FeaStudySummary, counts: &FeaStudyDocumentCounts) -> FeaStudyReadiness {
    let mut blockers = Vec::new();
    if summary.geometry_path.as_deref().is_none_or(str::is_empty) {
        blockers.push("Select study geometry.".into());
    }
    if summary.model_source.is_none() {
        blockers.push("Select an explicit model source.".into());
    }
    if counts.sequence_steps == 0 {
        blockers.push("Define at least one canonical procedure step.".into());
    }
    if summary.model_source.as_deref() == Some("inline") {
        if counts.materials == 0 {
            blockers.push("Define at least one material or medium.".into());
        }
        if counts.material_assignments == 0 {
            blockers.push("Assign materials or media to model regions.".into());
        }
    } else if matches!(summary.model_source.as_deref(), Some("file" | "compose")) {
        blockers.push(
            "Resolve referenced model sources with `runmat check` before treating the study as ready."
                .into(),
        );
    }
    FeaStudyReadiness {
        ready_to_solve: blockers.is_empty(),
        blockers,
    }
}

pub(super) fn with_diff(
    path: &str,
    before: &str,
    after: &str,
    sections: &[&str],
) -> Result<FeaStudyDocumentOperationResult, String> {
    let mut result = summarize_fea_study_document_contract(path, after)?;
    result.diff = Some(FeaStudyDocumentDiff {
        changed_sections: sections.iter().map(|value| (*value).into()).collect(),
        before: summarize(&parse_document(before)?),
        after: summarize(&parse_document(after)?),
        source_length_before: before.len(),
        source_length_after: after.len(),
    });
    Ok(result)
}
