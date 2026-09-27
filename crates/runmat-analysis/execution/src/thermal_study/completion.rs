//! Ordered, metered completion replay without solver or worker-task re-execution.
//! Existing import-time numerical consistency checks (including observation reduction) still run.
use super::*;
use crate::AnalysisWorkloadRequest;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ThermalStudyResultPhase {
    Solve,
    Observation,
    Accuracy,
    State { ordinal: usize },
    Publication,
}

/// Preserves the original scheduler association; this is not a new scheduler attestation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalStudyResultClosure {
    pub phase: ThermalStudyResultPhase,
    pub workload: AnalysisWorkloadRequest,
    pub response: ProgramExecutionResponse,
    pub accounting: AttemptResourceAccounting,
}

impl ThermalStudyResultClosure {
    pub(super) fn new(
        phase: ThermalStudyResultPhase,
        host: &AnalysisHostWorkload,
        response: &ProgramExecutionResponse,
        accounting: &AttemptResourceAccounting,
    ) -> Self {
        Self {
            phase,
            workload: host.workload.clone(),
            response: response.clone(),
            accounting: accounting.clone(),
        }
    }

    fn validate_for(
        &self,
        phase: ThermalStudyResultPhase,
        host: &AnalysisHostWorkload,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != phase || self.workload != host.workload {
            return Err(AnalysisBridgeError::Identity(
                "thermal completion phase or workload differs from its reconstructed task",
            ));
        }
        Ok(())
    }
}

pub(super) struct ThermalStudyReplayContext {
    pub mesh: SolverMeshArtifact,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

impl CompletedThermalStudy {
    pub fn result_closures(&self) -> &[ThermalStudyResultClosure] {
        &self.closures
    }

    /// Original validated preparation, retained for durable artifact packaging.
    pub fn preparation(&self) -> ThermalStudyPreparation {
        ThermalStudyPreparation {
            product: ThermalStudyProduct {
                compiled: self.compiled.clone(),
                workload: self.workload.clone(),
            },
            mesh: self.replay.mesh.clone(),
            frozen_project_digest: self.replay.frozen_project_digest,
            artifact_access: self.replay.artifact_access.clone(),
            program_revision: self.replay.program_revision.clone(),
            output_policy: self.replay.output_policy,
            required_capabilities: self.replay.required_capabilities.clone(),
        }
    }
}

/// Reconstruct every task from authored inputs and consume its saved response/accounting pair.
/// No unmetered shortcut: source imports, admission and terminal usage checks run unchanged.
pub fn import_thermal_study_completion(
    source: &impl CacheImport,
    preparation: ThermalStudyPreparation,
    closures: Vec<ThermalStudyResultClosure>,
) -> AnalysisBridgeResult<CompletedThermalStudy> {
    let mut study = ThermalStudyCoordinator::new(preparation)?;
    let mut closures = closures.into_iter();
    loop {
        // These transitions have no worker response. Rebuild them instead of trusting a label.
        if study.phase() == ThermalStudyPhase::ReadyForBaseline {
            study.freeze_baseline()?;
        }
        if study.phase() == ThermalStudyPhase::Complete {
            break;
        }
        let (phase, host) = match study.phase() {
            ThermalStudyPhase::ReadyForSolve => {
                (ThermalStudyResultPhase::Solve, &study.prepare_solve()?.host)
            }
            ThermalStudyPhase::ReadyForObservations => {
                let Some(task) = study.prepare_next_observation(source)? else {
                    continue;
                };
                (ThermalStudyResultPhase::Observation, task.host)
            }
            ThermalStudyPhase::ReadyForAccuracy => (
                ThermalStudyResultPhase::Accuracy,
                &study.prepare_accuracy()?.host,
            ),
            ThermalStudyPhase::ReadyForState { ordinal } => (
                ThermalStudyResultPhase::State { ordinal },
                &study.prepare_next_state()?.host,
            ),
            ThermalStudyPhase::ReadyForPublication => (
                ThermalStudyResultPhase::Publication,
                &study.prepare_publication()?.host,
            ),
            _ => {
                return Err(invalid(
                    "thermal completion replay reached an awaiting phase",
                ))
            }
        };
        let closure = closures
            .next()
            .ok_or_else(|| invalid("thermal completion omits an authored phase"))?;
        closure.validate_for(phase, host)?;
        match phase {
            ThermalStudyResultPhase::Solve => {
                study.complete_solve(source, closure.response, closure.accounting)?
            }
            ThermalStudyResultPhase::Observation => {
                study.complete_observation(source, closure.response, closure.accounting)?
            }
            ThermalStudyResultPhase::Accuracy => {
                study.complete_accuracy(source, closure.response, closure.accounting)?
            }
            ThermalStudyResultPhase::State { .. } => {
                study.complete_next_state(source, closure.response, closure.accounting)?
            }
            ThermalStudyResultPhase::Publication => {
                study.complete_publication(source, closure.response, closure.accounting)?
            }
        }
    }
    if closures.next().is_some() {
        return Err(invalid("thermal completion contains an extra phase"));
    }
    study.into_completed()
}
