use anyhow::{bail, Result};
use runmat_runtime::analysis::AnalysisStudyRunData;
use serde::Serialize;

use crate::presentation;

mod local;
use local::completed_summary;
pub(super) use local::print_completed_fea_result;

#[derive(Serialize)]
pub(super) struct LocalRunSummary {
    #[serde(flatten)]
    assessment_summary: AssessmentSummary,
    #[serde(flatten)]
    solve_placement: SolvePlacementSummary,
    /// Signed global heat balance of the final thermal solve, in watts.
    #[serde(skip_serializing_if = "Option::is_none")]
    thermal_energy_balance: Option<ThermalEnergyBalanceSummary>,
    root_id: String,
    procedure: &'static str,
    study_definition_digest: String,
    mesh_digest: String,
    nodes: usize,
    elements: usize,
    boundary_faces: usize,
    step_run_ids: Vec<String>,
    /// Definition run ids of the steps; each resolves to its latest committed attempt, while
    /// `step_run_ids` name this execution's attempts.
    definition_run_ids: Vec<String>,
    /// Every committed attempt of each definition run id, oldest first (this execution last).
    attempt_run_ids: Vec<Vec<String>>,
    result_manifest_artifact_path: String,
}

pub(super) fn definition_run_ids(step_run_ids: &[String]) -> Vec<String> {
    step_run_ids
        .iter()
        .map(|id| {
            runmat_runtime::analysis::split_attempt_run_id(id)
                .map_or_else(|| id.clone(), |(definition, _)| definition.to_owned())
        })
        .collect()
}

/// Where the static solves ran and how: the effective memory ceiling the placement admitted and
/// the solve path each static step selected under it. Provenance only, outside every identity.
#[derive(Serialize, Default)]
pub(super) struct SolvePlacementSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    effective_memory_ceiling_bytes: Option<u64>,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    static_solve_paths: std::collections::BTreeMap<
        String,
        runmat_analysis_fea::planning::resource::StaticSolvePath,
    >,
}

impl SolvePlacementSummary {
    fn of(numerical: &runmat_runtime::analysis::StructuralStudyNumericalResult) -> Self {
        match numerical {
            runmat_runtime::analysis::StructuralStudyNumericalResult::LinearSequence(sequence) => {
                Self {
                    effective_memory_ceiling_bytes: sequence.effective_memory_ceiling_bytes,
                    static_solve_paths: sequence
                        .steps
                        .iter()
                        .filter_map(|step| {
                            step.static_solve_path
                                .map(|path| (step.step_id.as_str().to_string(), path))
                        })
                        .collect(),
                }
            }
            _ => Self::default(),
        }
    }
}

#[derive(Serialize)]
struct AssessmentSummary {
    run_status: runmat_runtime::analysis::RunStatus,
    numerically_acceptable: bool,
    assessments: Vec<runmat_runtime::analysis::assessment::PublishedStudyAssessment>,
}

#[derive(Serialize)]
pub(super) struct ThermalEnergyBalanceSummary {
    applied_power_w: f64,
    boundary_loss_w: f64,
    storage_rate_w: f64,
    prescribed_boundary_power_w: f64,
    prescribed_heat_entering_w: f64,
    prescribed_heat_leaving_w: f64,
    residual_w: f64,
    relative_residual: f64,
}

impl From<&runmat_analysis_fea::contracts::ThermalEnergyBalance> for ThermalEnergyBalanceSummary {
    fn from(balance: &runmat_analysis_fea::contracts::ThermalEnergyBalance) -> Self {
        Self {
            applied_power_w: balance.applied_power_w.canonical_value(),
            boundary_loss_w: balance.boundary_loss_w.canonical_value(),
            storage_rate_w: balance.storage_rate_w.canonical_value(),
            prescribed_boundary_power_w: balance.prescribed_boundary_power_w.canonical_value(),
            prescribed_heat_entering_w: balance.prescribed_heat_flow.entering_w.canonical_value(),
            prescribed_heat_leaving_w: balance.prescribed_heat_flow.leaving_w.canonical_value(),
            residual_w: balance.residual_w.canonical_value(),
            relative_residual: balance.relative_residual,
        }
    }
}

impl From<&AnalysisStudyRunData> for AssessmentSummary {
    fn from(publication: &AnalysisStudyRunData) -> Self {
        Self {
            run_status: publication.run_status,
            numerically_acceptable: publication.numerically_acceptable,
            assessments: publication.assessments.clone(),
        }
    }
}

pub(super) fn local_summary(
    root_id: impl Into<String>,
    completed: &runmat_runtime::analysis::StructuralStudyResult,
    publication: &runmat_runtime::analysis::AnalysisStudyRunData,
) -> Result<LocalRunSummary> {
    Ok(LocalRunSummary {
        assessment_summary: publication.into(),
        solve_placement: SolvePlacementSummary::of(completed.numerical()),
        thermal_energy_balance: None,
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
        definition_run_ids: Vec::new(),
        attempt_run_ids: Vec::new(),
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
    completed: &runmat_execution_runner_native::CompletedFeaStudyResult,
    publication: &runmat_runtime::analysis::AnalysisStudyRunData,
    json: bool,
) -> Result<()> {
    print_completed_fea_result(study_id, completed, publication, json)
}

fn print_summary(study_id: &str, mut summary: LocalRunSummary, json: bool) -> Result<()> {
    summary.definition_run_ids = definition_run_ids(&summary.step_run_ids);
    summary.attempt_run_ids = summary
        .definition_run_ids
        .iter()
        .map(|definition| runmat_runtime::analysis::storage::run_attempts(definition))
        .collect::<Result<Vec<_>, _>>()
        .map_err(anyhow::Error::msg)?;
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
    if let Some(balance) = &summary.thermal_energy_balance {
        println!(
            "  {} applied {:.6} W, boundary loss {:.6} W, storage {:.6} W, prescribed-temperature reaction {:.6} W, residual {:.3e} W (relative {:.3e})",
            styles.label("energy balance:"),
            balance.applied_power_w,
            balance.boundary_loss_w,
            balance.storage_rate_w,
            balance.prescribed_boundary_power_w,
            balance.residual_w,
            balance.relative_residual
        );
        println!(
            "  {} {:.6} W in, {:.6} W out",
            styles.label("heat through prescribed temperatures:"),
            balance.prescribed_heat_entering_w,
            balance.prescribed_heat_leaving_w
        );
    }
    for assessment in &summary.assessment_summary.assessments {
        println!(
            "  {}: {:?}",
            assessment.definition.assessment_id, assessment.result.status
        );
        if let Some(issue) = &assessment.result.issue {
            println!("    evidence: {issue:?}");
        }
        if let Some(diagnostic) = &assessment.function_execution_diagnostic {
            println!("    execution: {diagnostic}");
        }
    }
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
                        "result": completed_summary(case_id, &study.result, &study.publication)?,
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
    let unsuccessful = result
        .entries
        .iter()
        .filter(|entry| {
            !matches!(
                entry.terminal,
                super::remote::MaterializedFeaSweepTerminal::Succeeded(_)
            )
        })
        .count();
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "sweep_id": sweep_id,
                "cases": cases,
            }))?
        );
        return sweep_outcome(sweep_id, unsuccessful, case_ids.len());
    }
    let styles = presentation::stdout();
    let heading = if unsuccessful == 0 {
        styles.success("FEA sweep complete")
    } else {
        styles.error("FEA sweep finished with unsuccessful cases")
    };
    println!(
        "{} {} ({} cases)",
        heading,
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
            None => match case["message"].as_str() {
                Some(message) => println!("  {id}: {status}: {message}"),
                None => println!("  {id}: {status}"),
            },
        }
    }
    sweep_outcome(sweep_id, unsuccessful, case_ids.len())
}

/// A sweep succeeds only when every case succeeds. The full per-case report is printed first;
/// any failed, cancelled or indeterminate case then fails the command (exit status 1), the same
/// convention `runmat test` uses for a run with failing tests.
fn sweep_outcome(sweep_id: &str, unsuccessful: usize, total: usize) -> Result<()> {
    if unsuccessful > 0 {
        bail!("FEA sweep {sweep_id}: {unsuccessful} of {total} cases did not succeed");
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
        StructuralStudyNumericalResult::ThermalPrestress { .. } => "thermal_prestress",
        StructuralStudyNumericalResult::LinearSequence(_) => "structural_linear_sequence",
        StructuralStudyNumericalResult::StructuralModal(_) => "structural_modal",
        StructuralStudyNumericalResult::StructuralModalTransient(_) => "structural_modal_transient",
        StructuralStudyNumericalResult::StructuralModalFrequency(_) => "structural_modal_frequency",
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

#[cfg(test)]
mod tests {
    use super::super::remote::{
        MaterializedFeaSweep, MaterializedFeaSweepEntry, MaterializedFeaSweepTerminal,
    };

    fn sweep(terminals: Vec<MaterializedFeaSweepTerminal>) -> MaterializedFeaSweep {
        MaterializedFeaSweep {
            entries: terminals
                .into_iter()
                .enumerate()
                .map(|(study_index, terminal)| MaterializedFeaSweepEntry {
                    study_index,
                    terminal,
                })
                .collect(),
        }
    }

    #[test]
    fn sweep_with_any_unsuccessful_case_fails_the_command() {
        let case_ids = vec!["a".to_string(), "b".to_string()];
        for json in [false, true] {
            let all_failed = sweep(vec![
                MaterializedFeaSweepTerminal::Failed("solver diverged".into()),
                MaterializedFeaSweepTerminal::Failed("solver diverged".into()),
            ]);
            let error = super::print_fea_sweep_result("s", &case_ids, &all_failed, json)
                .expect_err("a sweep whose cases all failed must fail the command");
            assert!(error.to_string().contains("2 of 2 cases did not succeed"));
            let partial = sweep(vec![
                MaterializedFeaSweepTerminal::Cancelled,
                MaterializedFeaSweepTerminal::Indeterminate("lost worker".into()),
            ]);
            assert!(super::print_fea_sweep_result("s", &case_ids, &partial, json).is_err());
        }
        assert!(super::sweep_outcome("s", 0, 2).is_ok());
        assert!(super::sweep_outcome("s", 1, 2).is_err());
    }
}
