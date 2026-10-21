//! Ordered, metered completion replay without solver or worker-task re-execution.
//! Existing import-time source and numerical receipt consistency checks still run.
use super::*;
use crate::AnalysisWorkloadRequest;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ElectricalStudyResultPhase {
    Solve,
    Accuracy,
    State { ordinal: usize },
    Publication,
}

/// Preserves the original scheduler association; this is not a new scheduler attestation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectricalStudyResultClosure {
    pub phase: ElectricalStudyResultPhase,
    pub workload: AnalysisWorkloadRequest,
    pub response: ProgramExecutionResponse,
    pub execution: AnalysisPhaseExecutionEvidence,
}

impl ElectricalStudyResultClosure {
    pub(super) fn new(
        phase: ElectricalStudyResultPhase,
        host: &AnalysisHostWorkload,
        response: &ProgramExecutionResponse,
        accounting: &AnalysisPhaseExecutionEvidence,
    ) -> Self {
        Self {
            phase,
            workload: host.workload.clone(),
            response: response.clone(),
            execution: accounting.clone(),
        }
    }

    fn validate_for(
        &self,
        phase: ElectricalStudyResultPhase,
        host: &AnalysisHostWorkload,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != phase || self.workload != host.workload {
            return Err(AnalysisBridgeError::Identity(
                "electrical completion phase or workload differs from its reconstructed task",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ElectricalStudyReplayContext {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub policy: ElectricalStudyPlanningPolicy,
    pub mesh: SolverMeshArtifact,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

impl CompletedElectricalStudy {
    pub fn policy(&self) -> ElectricalStudyPlanningPolicy {
        self.replay.policy
    }

    pub fn result_closures(&self) -> &[ElectricalStudyResultClosure] {
        &self.closures
    }

    /// Original validated preparation, retained for durable artifact packaging.
    pub fn preparation(&self) -> AnalysisBridgeResult<ElectricalStudyPreparation> {
        self.compiled
            .validate_against(&self.replay.mesh)
            .map_err(|error| invalid(error.to_string()))?;
        self.result.numerical_receipt.validate_against(
            &self.workload,
            &self.result.result,
            &self.replay.mesh,
        )?;
        let workload = plan_electrical_study(&self.compiled, self.replay.policy)
            .map_err(|error| invalid(error.to_string()))?;
        if workload != self.workload {
            return Err(invalid(
                "completed electrical workload differs from authored plan",
            ));
        }
        Ok(ElectricalStudyPreparation {
            local_execution: self.replay.local_execution,
            policy: self.replay.policy,
            product: ElectricalStudyProduct {
                compiled: self.compiled.clone(),
                workload,
            },
            mesh: self.replay.mesh.clone(),
            frozen_project_digest: self.replay.frozen_project_digest,
            artifact_access: self.replay.artifact_access.clone(),
            program_revision: self.replay.program_revision.clone(),
            output_policy: self.replay.output_policy,
            required_capabilities: self.replay.required_capabilities.clone(),
        })
    }
}

/// Reconstruct every task from authored inputs and consume its saved response/accounting pair.
/// No unmetered shortcut: source imports, admission and terminal usage checks run unchanged.
pub fn import_electrical_study_completion(
    source: &impl CacheImport,
    preparation: ElectricalStudyPreparation,
    closures: Vec<ElectricalStudyResultClosure>,
) -> AnalysisBridgeResult<CompletedElectricalStudy> {
    let mut study = ElectricalStudyCoordinator::new(preparation)?;
    let mut closures = closures.into_iter();
    loop {
        // These transitions have no worker response. Rebuild them instead of trusting a label.
        if study.phase() == ElectricalStudyPhase::ReadyForBaseline {
            study.freeze_baseline()?;
        }
        if study.phase() == ElectricalStudyPhase::Complete {
            break;
        }
        let (phase, host) = match study.phase() {
            ElectricalStudyPhase::ReadyForSolve => (
                ElectricalStudyResultPhase::Solve,
                &study.prepare_solve()?.host,
            ),
            ElectricalStudyPhase::ReadyForAccuracy => (
                ElectricalStudyResultPhase::Accuracy,
                &study.prepare_accuracy()?.host,
            ),
            ElectricalStudyPhase::ReadyForState { ordinal } => (
                ElectricalStudyResultPhase::State { ordinal },
                &study.prepare_next_state()?.host,
            ),
            ElectricalStudyPhase::ReadyForPublication => (
                ElectricalStudyResultPhase::Publication,
                &study.prepare_publication()?.host,
            ),
            _ => {
                return Err(invalid(
                    "electrical completion replay reached an awaiting phase",
                ))
            }
        };
        let closure = closures
            .next()
            .ok_or_else(|| invalid("electrical completion omits an authored phase"))?;
        closure.validate_for(phase, host)?;
        match phase {
            ElectricalStudyResultPhase::Solve => {
                study.complete_solve(source, closure.response, closure.execution)?
            }
            ElectricalStudyResultPhase::Accuracy => {
                study.complete_accuracy(source, closure.response, closure.execution)?
            }
            ElectricalStudyResultPhase::State { .. } => {
                study.complete_next_state(source, closure.response, closure.execution)?
            }
            ElectricalStudyResultPhase::Publication => {
                study.complete_publication(source, closure.response, closure.execution)?
            }
        }
    }
    if closures.next().is_some() {
        return Err(invalid("electrical completion contains an extra phase"));
    }
    study.into_completed()
}
