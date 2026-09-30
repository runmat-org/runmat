//! Family-aware presentation of already admitted native local completions.
use super::{local_summary, print_summary, LocalRunSummary};
use anyhow::Result;
use runmat_execution_runner_native::CompletedFeaStudyResult;
use runmat_runtime::analysis::AnalysisStudyRunData;

pub(in super::super) fn completed_summary(
    root_id: impl Into<String>,
    completed: &CompletedFeaStudyResult,
    publication: &AnalysisStudyRunData,
) -> Result<LocalRunSummary> {
    match completed {
        CompletedFeaStudyResult::Structural(result) => local_summary(root_id, result, publication),
        CompletedFeaStudyResult::Acoustic(result) => Ok(LocalRunSummary {
            root_id: root_id.into(),
            procedure: "acoustic_harmonic",
            study_definition_digest: result
                .completion
                .compiled
                .definition_digest()
                .map_err(anyhow::Error::msg)?
                .to_string(),
            mesh_digest: super::stable_digest_hex(&result.mesh.canonical_digest),
            nodes: result.mesh.topology.nodes.len(),
            elements: result.mesh.topology.volume_elements.len(),
            boundary_faces: result.mesh.topology.boundary_faces.len(),
            step_run_ids: publication
                .steps
                .iter()
                .map(|step| step.run_id.clone())
                .collect(),
            result_manifest_artifact_path: publication
                .result_manifest_artifact_path
                .clone()
                .ok_or_else(|| {
                    anyhow::anyhow!("typed acoustic study publication omitted its result manifest")
                })?,
        }),
        CompletedFeaStudyResult::Thermal(result) => Ok(LocalRunSummary {
            root_id: root_id.into(),
            procedure: match result.completion.workload.procedure {
                runmat_analysis_fea::ThermalProcedureControl::Steady => "thermal_steady",
                runmat_analysis_fea::ThermalProcedureControl::BackwardEuler { .. } => {
                    "thermal_transient"
                }
            },
            study_definition_digest: result
                .completion
                .compiled
                .definition_digest()
                .map_err(anyhow::Error::msg)?
                .to_string(),
            mesh_digest: super::stable_digest_hex(&result.mesh.canonical_digest),
            nodes: result.mesh.topology.nodes.len(),
            elements: result.mesh.topology.volume_elements.len(),
            boundary_faces: result.mesh.topology.boundary_faces.len(),
            step_run_ids: publication
                .steps
                .iter()
                .map(|step| step.run_id.clone())
                .collect(),
            result_manifest_artifact_path: publication
                .result_manifest_artifact_path
                .clone()
                .ok_or_else(|| {
                    anyhow::anyhow!("typed thermal study publication omitted its result manifest")
                })?,
        }),
    }
}

pub(in super::super) fn print_completed_fea_result(
    study_id: &str,
    completed: &CompletedFeaStudyResult,
    publication: &AnalysisStudyRunData,
    json: bool,
) -> Result<()> {
    print_summary(
        study_id,
        completed_summary(study_id, completed, publication)?,
        json,
    )
}
