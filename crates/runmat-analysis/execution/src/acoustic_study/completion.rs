//! Ordered, metered completion replay without solver or worker-task re-execution.
//! Existing import-time source and numerical receipt consistency checks still run.
use super::*;
use crate::AnalysisWorkloadRequest;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum AcousticStudyResultPhase {
    Solve,
    Accuracy,
    State { ordinal: usize },
    Publication,
}

/// Preserves the original scheduler association; this is not a new scheduler attestation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticStudyResultClosure {
    pub phase: AcousticStudyResultPhase,
    pub workload: AnalysisWorkloadRequest,
    pub response: ProgramExecutionResponse,
    pub accounting: AttemptResourceAccounting,
}

impl AcousticStudyResultClosure {
    pub(super) fn new(
        phase: AcousticStudyResultPhase,
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
        phase: AcousticStudyResultPhase,
        host: &AnalysisHostWorkload,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != phase || self.workload != host.workload {
            return Err(AnalysisBridgeError::Identity(
                "acoustic completion phase or workload differs from its reconstructed task",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct AcousticStudyReplayContext {
    pub policy: AcousticStudyPlanningPolicy,
    pub mesh: SolverMeshArtifact,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

impl CompletedAcousticStudy {
    pub fn policy(&self) -> AcousticStudyPlanningPolicy {
        self.replay.policy
    }

    pub fn result_closures(&self) -> &[AcousticStudyResultClosure] {
        &self.closures
    }

    /// Original validated preparation, retained for durable artifact packaging.
    pub fn preparation(&self) -> AnalysisBridgeResult<AcousticStudyPreparation> {
        self.compiled
            .validate_against(&self.replay.mesh)
            .map_err(|error| invalid(error.to_string()))?;
        self.result.numerical_receipt.validate_against(
            &self.workload,
            &self.result.result,
            &self.replay.mesh,
        )?;
        let plan = plan_acoustic_study(&self.compiled, self.replay.policy)
            .map_err(|error| invalid(error.to_string()))?;
        let [frequency] = plan.frequencies() else {
            return Err(invalid(
                "completed acoustic plan must contain one frequency",
            ));
        };
        if frequency.workload() != &self.workload {
            return Err(invalid(
                "completed acoustic workload differs from authored plan",
            ));
        }
        Ok(AcousticStudyPreparation {
            policy: self.replay.policy,
            product: AcousticStudyProduct {
                compiled: self.compiled.clone(),
                plan,
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
pub fn import_acoustic_study_completion(
    source: &impl CacheImport,
    preparation: AcousticStudyPreparation,
    closures: Vec<AcousticStudyResultClosure>,
) -> AnalysisBridgeResult<CompletedAcousticStudy> {
    let mut study = AcousticStudyCoordinator::new(preparation)?;
    let mut closures = closures.into_iter();
    loop {
        // These transitions have no worker response. Rebuild them instead of trusting a label.
        if study.phase() == AcousticStudyPhase::ReadyForBaseline {
            study.freeze_baseline()?;
        }
        if study.phase() == AcousticStudyPhase::Complete {
            break;
        }
        let (phase, host) = match study.phase() {
            AcousticStudyPhase::ReadyForSolve => (
                AcousticStudyResultPhase::Solve,
                &study.prepare_solve()?.host,
            ),
            AcousticStudyPhase::ReadyForAccuracy => (
                AcousticStudyResultPhase::Accuracy,
                &study.prepare_accuracy()?.host,
            ),
            AcousticStudyPhase::ReadyForState { ordinal } => (
                AcousticStudyResultPhase::State { ordinal },
                &study.prepare_next_state()?.host,
            ),
            AcousticStudyPhase::ReadyForPublication => (
                AcousticStudyResultPhase::Publication,
                &study.prepare_publication()?.host,
            ),
            _ => {
                return Err(invalid(
                    "acoustic completion replay reached an awaiting phase",
                ))
            }
        };
        let closure = closures
            .next()
            .ok_or_else(|| invalid("acoustic completion omits an authored phase"))?;
        closure.validate_for(phase, host)?;
        match phase {
            AcousticStudyResultPhase::Solve => {
                study.complete_solve(source, closure.response, closure.accounting)?
            }
            AcousticStudyResultPhase::Accuracy => {
                study.complete_accuracy(source, closure.response, closure.accounting)?
            }
            AcousticStudyResultPhase::State { .. } => {
                study.complete_next_state(source, closure.response, closure.accounting)?
            }
            AcousticStudyResultPhase::Publication => {
                study.complete_publication(source, closure.response, closure.accounting)?
            }
        }
    }
    if closures.next().is_some() {
        return Err(invalid("acoustic completion contains an extra phase"));
    }
    study.into_completed()
}
