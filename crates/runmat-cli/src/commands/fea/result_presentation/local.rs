//! Family-aware presentation of already admitted native local completions.
use super::{local_summary, print_summary, LocalRunSummary, SolvePlacementSummary};
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
        CompletedFeaStudyResult::Electrical(result) => Ok(LocalRunSummary {
            assessment_summary: publication.into(),
            solve_placement: SolvePlacementSummary::default(),
            thermal_energy_balance: None,
            root_id: root_id.into(),
            procedure: match result
                .completion
                .workload
                .equation()
                .map_err(anyhow::Error::msg)?
            {
                runmat_analysis_core::StaticElectricalEquation::Electrostatics => "electrostatics",
                runmat_analysis_core::StaticElectricalEquation::SteadyConduction => {
                    "steady_conduction"
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
            definition_run_ids: Vec::new(),
            attempt_run_ids: Vec::new(),
            result_manifest_artifact_path: publication
                .result_manifest_artifact_path
                .clone()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "typed electrical study publication omitted its result manifest"
                    )
                })?,
        }),
        CompletedFeaStudyResult::AcousticResponse(result) => Ok(LocalRunSummary {
            thermal_energy_balance: None,
            assessment_summary: publication.into(),
            solve_placement: SolvePlacementSummary::default(),
            root_id: root_id.into(),
            procedure: match result.completion.workload.solver {
                runmat_analysis_fea::execution::AcousticResponseSolver::Modal { .. } => {
                    "acoustic_modal"
                }
                runmat_analysis_fea::execution::AcousticResponseSolver::Transient { .. } => {
                    "acoustic_transient"
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
            definition_run_ids: Vec::new(),
            attempt_run_ids: Vec::new(),
            result_manifest_artifact_path: publication
                .result_manifest_artifact_path
                .clone()
                .ok_or_else(|| {
                    anyhow::anyhow!("typed acoustic study publication omitted its result manifest")
                })?,
        }),
        CompletedFeaStudyResult::Magnetic(result) => Ok(LocalRunSummary {
            assessment_summary: publication.into(),
            solve_placement: SolvePlacementSummary::default(),
            thermal_energy_balance: None,
            root_id: root_id.into(),
            procedure: "magnetostatics",
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
            definition_run_ids: Vec::new(),
            attempt_run_ids: Vec::new(),
            result_manifest_artifact_path: publication
                .result_manifest_artifact_path
                .clone()
                .ok_or_else(|| {
                    anyhow::anyhow!("typed magnetic study publication omitted its result manifest")
                })?,
        }),
        CompletedFeaStudyResult::Maxwell(result) => Ok(LocalRunSummary {
            assessment_summary: publication.into(),
            solve_placement: SolvePlacementSummary::default(),
            thermal_energy_balance: None,
            root_id: root_id.into(),
            procedure: "harmonic_maxwell",
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
            definition_run_ids: Vec::new(),
            attempt_run_ids: Vec::new(),
            result_manifest_artifact_path: publication
                .result_manifest_artifact_path
                .clone()
                .ok_or_else(|| {
                    anyhow::anyhow!("typed Maxwell publication omitted its result manifest")
                })?,
        }),
        CompletedFeaStudyResult::Nonlinear(result) => Ok(LocalRunSummary {
            assessment_summary: publication.into(),
            solve_placement: SolvePlacementSummary::default(),
            thermal_energy_balance: None,
            root_id: root_id.into(),
            procedure: "quasistatic_j2_solid",
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
            definition_run_ids: Vec::new(),
            attempt_run_ids: Vec::new(),
            result_manifest_artifact_path: publication
                .result_manifest_artifact_path
                .clone()
                .ok_or_else(|| {
                    anyhow::anyhow!("typed nonlinear publication omitted its result manifest")
                })?,
        }),
        CompletedFeaStudyResult::Fluid(result) => Ok(LocalRunSummary {
            assessment_summary: publication.into(),
            solve_placement: SolvePlacementSummary::default(),
            thermal_energy_balance: None,
            root_id: root_id.into(),
            procedure: "steady_incompressible_fluid",
            study_definition_digest: result
                .completion
                .compiled()
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
            definition_run_ids: Vec::new(),
            attempt_run_ids: Vec::new(),
            result_manifest_artifact_path: publication
                .result_manifest_artifact_path
                .clone()
                .ok_or_else(|| {
                    anyhow::anyhow!("typed fluid publication omitted its result manifest")
                })?,
        }),
        CompletedFeaStudyResult::Acoustic(result) => Ok(LocalRunSummary {
            assessment_summary: publication.into(),
            solve_placement: SolvePlacementSummary::default(),
            thermal_energy_balance: None,
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
            definition_run_ids: Vec::new(),
            attempt_run_ids: Vec::new(),
            result_manifest_artifact_path: publication
                .result_manifest_artifact_path
                .clone()
                .ok_or_else(|| {
                    anyhow::anyhow!("typed acoustic study publication omitted its result manifest")
                })?,
        }),
        CompletedFeaStudyResult::LineParameters(result) => Ok(LocalRunSummary {
            assessment_summary: publication.into(),
            solve_placement: SolvePlacementSummary::default(),
            thermal_energy_balance: None,
            root_id: root_id.into(),
            procedure: "line_parameters",
            study_definition_digest: result
                .completion
                .compiled
                .definition_digest()
                .map_err(anyhow::Error::msg)?
                .to_string(),
            mesh_digest: super::stable_digest_hex(&result.mesh.canonical_digest),
            nodes: result.mesh.topology.nodes.len(),
            // A cross-section mesh is its triangles.
            elements: result.mesh.topology.boundary_faces.len(),
            boundary_faces: 0,
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
                    anyhow::anyhow!("line-parameter publication omitted its result manifest")
                })?,
        }),
        CompletedFeaStudyResult::Thermal(result) => Ok(LocalRunSummary {
            assessment_summary: publication.into(),
            solve_placement: SolvePlacementSummary::default(),
            thermal_energy_balance: result
                .completion
                .manifest
                .energy_balances
                .last()
                .map(super::ThermalEnergyBalanceSummary::from),
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
            definition_run_ids: Vec::new(),
            attempt_run_ids: Vec::new(),
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
