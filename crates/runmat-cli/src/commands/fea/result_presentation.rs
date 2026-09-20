use anyhow::{bail, Result};
use serde::Serialize;

use crate::presentation;

#[derive(Serialize)]
pub(super) struct LocalRunSummary {
    root_id: String,
    procedure: &'static str,
    study_definition_digest: String,
    mesh_digest: String,
    nodes: usize,
    elements: usize,
    boundary_faces: usize,
    step_run_ids: Vec<String>,
    result_manifest_artifact_path: String,
}

pub(super) fn local_summary(
    root_id: impl Into<String>,
    completed: &runmat_runtime::analysis::StructuralStudyResult,
    publication: &runmat_runtime::analysis::AnalysisStudyRunData,
) -> Result<LocalRunSummary> {
    Ok(LocalRunSummary {
        root_id: root_id.into(),
        procedure: numerical_kind(completed.numerical()),
        study_definition_digest: completed
            .definition_digest()
            .map_err(anyhow::Error::msg)?
            .to_string(),
        mesh_digest: stable_digest_hex(&completed.mesh().canonical_digest),
        nodes: completed.mesh().topology.nodes.len(),
        elements: completed.mesh().topology.volume_elements.len(),
        boundary_faces: completed.mesh().topology.boundary_faces.len(),
        step_run_ids: publication
            .steps
            .iter()
            .map(|step| step.run_id.clone())
            .collect(),
        result_manifest_artifact_path: publication
            .result_manifest_artifact_path
            .clone()
            .ok_or_else(|| {
                anyhow::anyhow!("typed study publication omitted its result manifest")
            })?,
    })
}

pub(crate) fn print_fea_result(
    study_id: &str,
    completed: &runmat_runtime::analysis::StructuralStudyResult,
    publication: &runmat_runtime::analysis::AnalysisStudyRunData,
    json: bool,
) -> Result<()> {
    let summary = local_summary(study_id, completed, publication)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }
    let styles = presentation::stdout();
    println!(
        "{} {}",
        styles.success("FEA run complete"),
        styles.identifier(study_id)
    );
    println!("  {} {}", styles.label("procedure:"), summary.procedure);
    println!(
        "  {} {}",
        styles.label("study definition:"),
        summary.study_definition_digest
    );
    println!("  {} {}", styles.label("mesh:"), summary.mesh_digest);
    println!(
        "  {} {}",
        styles.label("result runs:"),
        summary.step_run_ids.join(", ")
    );
    println!(
        "  {} {}",
        styles.label("result manifest:"),
        summary.result_manifest_artifact_path
    );
    println!(
        "  {} {} nodes, {} tetrahedra, {} boundary faces",
        styles.label("topology:"),
        summary.nodes,
        summary.elements,
        summary.boundary_faces
    );
    Ok(())
}

pub(crate) fn print_fea_sweep_result(
    sweep_id: &str,
    case_ids: &[String],
    result: &super::remote::MaterializedFeaSweep,
    json: bool,
) -> Result<()> {
    if result.entries.len() != case_ids.len() {
        bail!("materialized FEA sweep differs from its submitted case count");
    }
    let cases = result
        .entries
        .iter()
        .zip(case_ids)
        .map(|(entry, case_id)| {
            if entry.study_index >= case_ids.len() || &case_ids[entry.study_index] != case_id {
                bail!("materialized FEA sweep changed its authored case order");
            }
            Ok(match &entry.terminal {
                super::remote::MaterializedFeaSweepTerminal::Succeeded(study) => {
                    serde_json::json!({
                        "case_id": case_id,
                        "status": "succeeded",
                        "result": local_summary(case_id, &study.result, &study.publication)?,
                    })
                }
                super::remote::MaterializedFeaSweepTerminal::Failed(message) => serde_json::json!({
                    "case_id": case_id,
                    "status": "failed",
                    "message": message,
                }),
                super::remote::MaterializedFeaSweepTerminal::Cancelled => serde_json::json!({
                    "case_id": case_id,
                    "status": "cancelled",
                }),
                super::remote::MaterializedFeaSweepTerminal::Indeterminate(message) => {
                    serde_json::json!({
                        "case_id": case_id,
                        "status": "indeterminate",
                        "message": message,
                    })
                }
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "sweep_id": sweep_id,
                "cases": cases,
            }))?
        );
        return Ok(());
    }
    let styles = presentation::stdout();
    println!(
        "{} {} ({} cases)",
        styles.success("FEA sweep complete"),
        styles.identifier(sweep_id),
        cases.len()
    );
    for case in cases {
        let id = case["case_id"].as_str().unwrap_or("unknown");
        let status = case["status"].as_str().unwrap_or("unknown");
        match case.get("result") {
            Some(summary) => println!(
                "  {}: {} tetrahedra, {}",
                id,
                summary["elements"],
                summary["procedure"].as_str().unwrap_or("unknown")
            ),
            None => println!("  {id}: {status}"),
        }
    }
    Ok(())
}

pub(super) fn print_sweep_result(
    id: &str,
    summaries: &[LocalRunSummary],
    json: bool,
) -> Result<()> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "sweep_id": id,
                "cases": summaries,
            }))?
        );
    } else {
        let styles = presentation::stdout();
        println!(
            "{} {} ({} cases)",
            styles.success("FEA sweep complete"),
            styles.identifier(id),
            summaries.len()
        );
        for summary in summaries {
            println!(
                "  {}: {} tetrahedra, {}",
                summary.root_id, summary.elements, summary.procedure
            );
        }
    }
    Ok(())
}

fn stable_digest_hex(digest: &runmat_meshing_core::StableDigest) -> String {
    digest.0.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn numerical_kind(
    result: &runmat_runtime::analysis::StructuralStudyNumericalResult,
) -> &'static str {
    use runmat_runtime::analysis::StructuralStudyNumericalResult;
    match result {
        StructuralStudyNumericalResult::LinearSequence(_) => "structural_linear_sequence",
        StructuralStudyNumericalResult::StructuralModal(_) => "structural_modal",
        StructuralStudyNumericalResult::StructuralModalTransient(_) => "structural_modal_transient",
        StructuralStudyNumericalResult::StructuralModalResponseSpectrum(_) => {
            "structural_modal_response_spectrum"
        }
        StructuralStudyNumericalResult::StructuralRandomResponse(_) => "structural_random_response",
        StructuralStudyNumericalResult::StructuralBuckling(_) => "structural_buckling",
        StructuralStudyNumericalResult::StructuralPrestressedModal(_) => {
            "structural_prestressed_modal"
        }
        StructuralStudyNumericalResult::StructuralPrestressedFrequency(_) => {
            "structural_prestressed_frequency"
        }
        StructuralStudyNumericalResult::StructuralTransient(_) => "structural_transient",
        StructuralStudyNumericalResult::StructuralFrequency(_) => "structural_frequency",
    }
}
