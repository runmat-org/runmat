//! Ordered, metered completion replay without solver or worker-task re-execution.
//! Existing import-time source and numerical receipt consistency checks still run.
use super::*;
use crate::AnalysisWorkloadRequest;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum AcousticResponseStudyResultPhase {
    Solve,
    FieldRecovery,
    Observation,
    Accuracy,
    State { ordinal: usize },
    Publication,
}

/// Preserves the original scheduler association; this is not a new scheduler attestation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticResponseStudyResultClosure {
    pub phase: AcousticResponseStudyResultPhase,
    pub workload: AnalysisWorkloadRequest,
    pub response: ProgramExecutionResponse,
    pub execution: AnalysisPhaseExecutionEvidence,
}

impl AcousticResponseStudyResultClosure {
    pub(super) fn new(
        phase: AcousticResponseStudyResultPhase,
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
        phase: AcousticResponseStudyResultPhase,
        host: &AnalysisHostWorkload,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != phase || self.workload != host.workload {
            return Err(AnalysisBridgeError::Identity(
                "acoustic response completion phase or workload differs from its reconstructed task",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct AcousticResponseStudyReplayContext {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub policy: AcousticResponseStudyPlanningPolicy,
    pub mesh: SolverMeshArtifact,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

impl CompletedAcousticResponseStudy {
    pub fn policy(&self) -> AcousticResponseStudyPlanningPolicy {
        self.replay.policy
    }

    pub fn result_closures(&self) -> &[AcousticResponseStudyResultClosure] {
        &self.closures
    }

    /// Original validated preparation, retained for durable artifact packaging.
    pub fn preparation(&self) -> AnalysisBridgeResult<AcousticResponseStudyPreparation> {
        self.compiled
            .validate_against(&self.replay.mesh)
            .map_err(|error| invalid(error.to_string()))?;
        self.result.numerical_receipt.validate_against(
            &self.workload,
            &self.result.result,
            &self.replay.mesh,
        )?;
        let workload = plan_acoustic_response_study(&self.compiled, self.replay.policy)
            .map_err(|error| invalid(error.to_string()))?;
        if workload != self.workload {
            return Err(invalid(
                "completed acoustic response workload differs from authored plan",
            ));
        }
        Ok(AcousticResponseStudyPreparation {
            local_execution: self.replay.local_execution,
            policy: self.replay.policy,
            product: AcousticResponseStudyProduct {
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
/// Completion reuses the source imports, admission and terminal usage checks.
pub fn import_acoustic_response_study_completion(
    source: &impl CacheImport,
    preparation: AcousticResponseStudyPreparation,
    closures: Vec<AcousticResponseStudyResultClosure>,
) -> AnalysisBridgeResult<CompletedAcousticResponseStudy> {
    let mut study = AcousticResponseStudyCoordinator::new(preparation)?;
    let mut closures = closures.into_iter();
    loop {
        // These transitions have no worker response. Rebuild them instead of trusting a label.
        if study.phase() == AcousticResponseStudyPhase::ReadyForBaseline {
            study.freeze_baseline()?;
        }
        if study.phase() == AcousticResponseStudyPhase::Complete {
            break;
        }
        if study.phase() == AcousticResponseStudyPhase::ReadyForObservations
            && study.prepare_next_observation(source)?.is_none()
        {
            continue;
        }
        let (phase, host) = match study.phase() {
            AcousticResponseStudyPhase::ReadyForSolve => (
                AcousticResponseStudyResultPhase::Solve,
                &study.prepare_solve()?.host,
            ),
            AcousticResponseStudyPhase::ReadyForAccuracy => (
                AcousticResponseStudyResultPhase::Accuracy,
                &study.prepare_accuracy()?.host,
            ),
            AcousticResponseStudyPhase::ReadyForFieldRecovery => (
                AcousticResponseStudyResultPhase::FieldRecovery,
                &study.prepare_field_recovery()?.host,
            ),
            AcousticResponseStudyPhase::AwaitingObservation => (
                AcousticResponseStudyResultPhase::Observation,
                study.observations.pending()?.host,
            ),
            AcousticResponseStudyPhase::ReadyForState { ordinal } => (
                AcousticResponseStudyResultPhase::State { ordinal },
                &study.prepare_next_state()?.host,
            ),
            AcousticResponseStudyPhase::ReadyForPublication => (
                AcousticResponseStudyResultPhase::Publication,
                &study.prepare_publication()?.host,
            ),
            _ => {
                return Err(invalid(
                    "acoustic response completion replay reached an awaiting phase",
                ))
            }
        };
        let closure = closures
            .next()
            .ok_or_else(|| invalid("acoustic response completion omits an authored phase"))?;
        closure.validate_for(phase, host)?;
        match phase {
            AcousticResponseStudyResultPhase::Solve => {
                study.complete_solve(source, closure.response, closure.execution)?
            }
            AcousticResponseStudyResultPhase::FieldRecovery => {
                study.complete_field_recovery(source, closure.response, closure.execution)?
            }
            AcousticResponseStudyResultPhase::Observation => {
                study.complete_observation(source, closure.response, closure.execution)?
            }
            AcousticResponseStudyResultPhase::Accuracy => {
                study.complete_accuracy(source, closure.response, closure.execution)?
            }
            AcousticResponseStudyResultPhase::State { .. } => {
                study.complete_next_state(source, closure.response, closure.execution)?
            }
            AcousticResponseStudyResultPhase::Publication => {
                study.complete_publication(source, closure.response, closure.execution)?
            }
        }
    }
    if closures.next().is_some() {
        return Err(invalid(
            "acoustic response completion contains an extra phase",
        ));
    }
    study.into_completed()
}
