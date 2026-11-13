//! Ordered, metered completion replay without solver or worker-task re-execution.
//! Existing import-time source and numerical receipt consistency checks still run.
use super::*;
use crate::AnalysisWorkloadRequest;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum MaxwellStudyResultPhase {
    Solve,
    FieldRecovery,
    ObservationPartition { ordinal: usize },
    ObservationReduction { ordinal: usize },
    ObservationAssembly,
    Accuracy,
    State { ordinal: usize },
    Publication,
}

/// Preserves the original scheduler association; this is not a new scheduler attestation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellStudyResultClosure {
    pub phase: MaxwellStudyResultPhase,
    pub workload: AnalysisWorkloadRequest,
    pub response: ProgramExecutionResponse,
    pub execution: AnalysisPhaseExecutionEvidence,
}

impl MaxwellStudyResultClosure {
    pub(super) fn new(
        phase: MaxwellStudyResultPhase,
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
        phase: MaxwellStudyResultPhase,
        host: &AnalysisHostWorkload,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != phase || self.workload != host.workload {
            return Err(AnalysisBridgeError::Identity(
                "maxwell completion phase or workload differs from its reconstructed task",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct MaxwellStudyReplayContext {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub policy: MaxwellStudyPlanningPolicy,
    pub mesh: SolverMeshArtifact,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

impl CompletedMaxwellStudy {
    pub fn policy(&self) -> MaxwellStudyPlanningPolicy {
        self.replay.policy
    }

    pub fn result_closures(&self) -> &[MaxwellStudyResultClosure] {
        &self.closures
    }

    /// Original validated preparation, retained for durable artifact packaging.
    pub fn preparation(&self) -> AnalysisBridgeResult<MaxwellStudyPreparation> {
        self.compiled
            .validate_against(&self.replay.mesh)
            .map_err(|error| invalid(error.to_string()))?;
        self.result.numerical_receipt.validate_against(
            &self.workload,
            &self.result.result,
            &self.replay.mesh,
        )?;
        let workload = plan_maxwell_study(&self.compiled, self.replay.policy)
            .map_err(|error| invalid(error.to_string()))?;
        if workload != self.workload {
            return Err(invalid(
                "completed maxwell workload differs from authored plan",
            ));
        }
        Ok(MaxwellStudyPreparation {
            local_execution: self.replay.local_execution,
            policy: self.replay.policy,
            product: MaxwellStudyProduct {
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
pub fn import_maxwell_study_completion(
    source: &impl CacheImport,
    preparation: MaxwellStudyPreparation,
    closures: Vec<MaxwellStudyResultClosure>,
) -> AnalysisBridgeResult<CompletedMaxwellStudy> {
    let mut study = MaxwellStudyCoordinator::new(preparation)?;
    let mut closures = closures.into_iter();
    loop {
        if study.phase() == MaxwellStudyPhase::ReadyForBaseline {
            study.freeze_baseline()?;
        }
        if study.phase() == MaxwellStudyPhase::Complete {
            break;
        }
        if study.phase() == MaxwellStudyPhase::ReadyForObservationFanout {
            study.prepare_observation_fanout(source)?;
        }
        let closure = closures
            .next()
            .ok_or_else(|| invalid("maxwell completion omits an authored phase"))?;
        let (phase, host) = match study.phase() {
            MaxwellStudyPhase::ReadyForSolve => {
                (MaxwellStudyResultPhase::Solve, &study.prepare_solve()?.host)
            }
            MaxwellStudyPhase::ReadyForFieldRecovery => (
                MaxwellStudyResultPhase::FieldRecovery,
                &study.prepare_field_recovery()?.host,
            ),
            MaxwellStudyPhase::AwaitingObservationPartitions => {
                let MaxwellStudyResultPhase::ObservationPartition { ordinal } = closure.phase
                else {
                    return Err(invalid("maxwell completion omits a fanout partition"));
                };
                let task = study
                    .observation_state
                    .fanout
                    .as_ref()
                    .expect("prepared fanout")
                    .partitions
                    .get(ordinal)
                    .ok_or_else(|| invalid("maxwell replay partition ordinal outside fanout"))?;
                (closure.phase, &task.host)
            }
            MaxwellStudyPhase::ReadyForObservationReduction { ordinal } => (
                MaxwellStudyResultPhase::ObservationReduction { ordinal },
                &study.prepare_observation_reduction()?.host,
            ),
            MaxwellStudyPhase::ReadyForObservationAssembly => (
                MaxwellStudyResultPhase::ObservationAssembly,
                &study.prepare_observation_assembly()?.host,
            ),
            MaxwellStudyPhase::ReadyForAccuracy => (
                MaxwellStudyResultPhase::Accuracy,
                &study.prepare_accuracy()?.host,
            ),
            MaxwellStudyPhase::ReadyForState { ordinal } => (
                MaxwellStudyResultPhase::State { ordinal },
                &study.prepare_next_state()?.host,
            ),
            MaxwellStudyPhase::ReadyForPublication => (
                MaxwellStudyResultPhase::Publication,
                &study.prepare_publication()?.host,
            ),
            _ => {
                return Err(invalid(
                    "maxwell completion replay reached an awaiting phase",
                ))
            }
        };
        closure.validate_for(phase, host)?;
        match phase {
            MaxwellStudyResultPhase::Solve => {
                study.complete_solve(source, closure.response, closure.execution)?
            }
            MaxwellStudyResultPhase::FieldRecovery => {
                study.complete_field_recovery(source, closure.response, closure.execution)?
            }
            MaxwellStudyResultPhase::ObservationPartition { ordinal } => study
                .complete_observation_partition(
                    source,
                    ordinal,
                    closure.response,
                    closure.execution,
                )?,
            MaxwellStudyResultPhase::ObservationReduction { .. } => {
                study.complete_observation_reduction(source, closure.response, closure.execution)?
            }
            MaxwellStudyResultPhase::ObservationAssembly => {
                study.complete_observation_assembly(source, closure.response, closure.execution)?
            }
            MaxwellStudyResultPhase::Accuracy => {
                study.complete_accuracy(source, closure.response, closure.execution)?
            }
            MaxwellStudyResultPhase::State { .. } => {
                study.complete_next_state(source, closure.response, closure.execution)?
            }
            MaxwellStudyResultPhase::Publication => {
                study.complete_publication(source, closure.response, closure.execution)?
            }
        }
    }
    if closures.next().is_some() {
        return Err(invalid("maxwell completion contains an extra phase"));
    }
    study.into_completed()
}
