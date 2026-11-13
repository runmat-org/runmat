//! Ordered, metered completion replay without solver or worker-task re-execution.
//! Existing import-time source and numerical receipt consistency checks still run.
use super::*;
use crate::AnalysisWorkloadRequest;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ElectricalStudyResultPhase {
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
    pub machine_memory_bytes: u64,
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
        self.replay.policy.clone()
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
        let workload = plan_electrical_study(&self.compiled, self.replay.policy.clone())
            .map_err(|error| invalid(error.to_string()))?;
        if workload != self.workload {
            return Err(invalid(
                "completed electrical workload differs from authored plan",
            ));
        }
        Ok(ElectricalStudyPreparation {
            local_execution: self.replay.local_execution,
            machine_memory_bytes: self.replay.machine_memory_bytes,
            policy: self.replay.policy.clone(),
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
/// Completion reuses the source imports, admission and terminal usage checks.
pub fn import_electrical_study_completion(
    source: &impl CacheImport,
    preparation: ElectricalStudyPreparation,
    closures: Vec<ElectricalStudyResultClosure>,
) -> AnalysisBridgeResult<CompletedElectricalStudy> {
    let mut study = ElectricalStudyCoordinator::new(preparation)?;
    let mut closures = closures.into_iter();
    loop {
        if study.phase() == ElectricalStudyPhase::ReadyForBaseline {
            study.freeze_baseline()?;
        }
        if study.phase() == ElectricalStudyPhase::Complete {
            break;
        }
        if study.phase() == ElectricalStudyPhase::ReadyForObservationFanout {
            study.prepare_observation_fanout(source)?;
        }
        let closure = closures
            .next()
            .ok_or_else(|| invalid("electrical completion omits an authored phase"))?;
        let (phase, host) = match study.phase() {
            ElectricalStudyPhase::ReadyForSolve => (
                ElectricalStudyResultPhase::Solve,
                &study.prepare_solve()?.host,
            ),
            ElectricalStudyPhase::ReadyForFieldRecovery => (
                ElectricalStudyResultPhase::FieldRecovery,
                &study.prepare_field_recovery()?.host,
            ),
            ElectricalStudyPhase::AwaitingObservationPartitions => {
                let ElectricalStudyResultPhase::ObservationPartition { ordinal } = closure.phase
                else {
                    return Err(invalid("electrical completion omits a fanout partition"));
                };
                let task = study
                    .observation_state
                    .fanout
                    .as_ref()
                    .expect("prepared fanout")
                    .partitions
                    .get(ordinal)
                    .ok_or_else(|| invalid("electrical replay partition ordinal outside fanout"))?;
                (closure.phase, &task.host)
            }
            ElectricalStudyPhase::ReadyForObservationReduction { ordinal } => (
                ElectricalStudyResultPhase::ObservationReduction { ordinal },
                &study.prepare_observation_reduction()?.host,
            ),
            ElectricalStudyPhase::ReadyForObservationAssembly => (
                ElectricalStudyResultPhase::ObservationAssembly,
                &study.prepare_observation_assembly()?.host,
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
        closure.validate_for(phase, host)?;
        match phase {
            ElectricalStudyResultPhase::Solve => {
                study.complete_solve(source, closure.response, closure.execution)?
            }
            ElectricalStudyResultPhase::FieldRecovery => {
                study.complete_field_recovery(source, closure.response, closure.execution)?
            }
            ElectricalStudyResultPhase::ObservationPartition { ordinal } => study
                .complete_observation_partition(
                    source,
                    ordinal,
                    closure.response,
                    closure.execution,
                )?,
            ElectricalStudyResultPhase::ObservationReduction { .. } => {
                study.complete_observation_reduction(source, closure.response, closure.execution)?
            }
            ElectricalStudyResultPhase::ObservationAssembly => {
                study.complete_observation_assembly(source, closure.response, closure.execution)?
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
