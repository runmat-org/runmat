//! Ordered, metered completion replay without solver or worker-task re-execution.
//! Existing import-time source and numerical receipt consistency checks still run.
use super::*;
use crate::AnalysisWorkloadRequest;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum MagneticStudyResultPhase {
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
pub struct MagneticStudyResultClosure {
    pub phase: MagneticStudyResultPhase,
    pub workload: AnalysisWorkloadRequest,
    pub response: ProgramExecutionResponse,
    pub execution: AnalysisPhaseExecutionEvidence,
}

impl MagneticStudyResultClosure {
    pub(super) fn new(
        phase: MagneticStudyResultPhase,
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
        phase: MagneticStudyResultPhase,
        host: &AnalysisHostWorkload,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != phase || self.workload != host.workload {
            return Err(AnalysisBridgeError::Identity(
                "magnetic completion phase or workload differs from its reconstructed task",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct MagneticStudyReplayContext {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub policy: MagneticStudyPlanningPolicy,
    pub mesh: SolverMeshArtifact,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

impl CompletedMagneticStudy {
    pub fn policy(&self) -> MagneticStudyPlanningPolicy {
        self.replay.policy
    }

    pub fn result_closures(&self) -> &[MagneticStudyResultClosure] {
        &self.closures
    }

    /// Original validated preparation, retained for durable artifact packaging.
    pub fn preparation(&self) -> AnalysisBridgeResult<MagneticStudyPreparation> {
        self.compiled
            .validate_against(&self.replay.mesh)
            .map_err(|error| invalid(error.to_string()))?;
        self.result.numerical_receipt.validate_against(
            &self.workload,
            &self.result.result,
            &self.replay.mesh,
        )?;
        let workload = plan_magnetic_study(&self.compiled, self.replay.policy)
            .map_err(|error| invalid(error.to_string()))?;
        if workload != self.workload {
            return Err(invalid(
                "completed magnetic workload differs from authored plan",
            ));
        }
        Ok(MagneticStudyPreparation {
            local_execution: self.replay.local_execution,
            policy: self.replay.policy,
            product: MagneticStudyProduct {
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
pub fn import_magnetic_study_completion(
    source: &impl CacheImport,
    preparation: MagneticStudyPreparation,
    closures: Vec<MagneticStudyResultClosure>,
) -> AnalysisBridgeResult<CompletedMagneticStudy> {
    let mut study = MagneticStudyCoordinator::new(preparation)?;
    let mut closures = closures.into_iter();
    loop {
        if study.phase() == MagneticStudyPhase::ReadyForBaseline {
            study.freeze_baseline()?;
        }
        if study.phase() == MagneticStudyPhase::Complete {
            break;
        }
        if study.phase() == MagneticStudyPhase::ReadyForObservationFanout {
            study.prepare_observation_fanout(source)?;
        }
        let closure = closures
            .next()
            .ok_or_else(|| invalid("magnetic completion omits an authored phase"))?;
        let (phase, host) = match study.phase() {
            MagneticStudyPhase::ReadyForSolve => (
                MagneticStudyResultPhase::Solve,
                &study.prepare_solve()?.host,
            ),
            MagneticStudyPhase::ReadyForFieldRecovery => (
                MagneticStudyResultPhase::FieldRecovery,
                &study.prepare_field_recovery()?.host,
            ),
            MagneticStudyPhase::AwaitingObservationPartitions => {
                let MagneticStudyResultPhase::ObservationPartition { ordinal } = closure.phase
                else {
                    return Err(invalid("magnetic completion omits a fanout partition"));
                };
                let task = study
                    .observation_state
                    .fanout
                    .as_ref()
                    .expect("prepared fanout")
                    .partitions
                    .get(ordinal)
                    .ok_or_else(|| invalid("magnetic replay partition ordinal outside fanout"))?;
                (closure.phase, &task.host)
            }
            MagneticStudyPhase::ReadyForObservationReduction { ordinal } => (
                MagneticStudyResultPhase::ObservationReduction { ordinal },
                &study.prepare_observation_reduction()?.host,
            ),
            MagneticStudyPhase::ReadyForObservationAssembly => (
                MagneticStudyResultPhase::ObservationAssembly,
                &study.prepare_observation_assembly()?.host,
            ),
            MagneticStudyPhase::ReadyForAccuracy => (
                MagneticStudyResultPhase::Accuracy,
                &study.prepare_accuracy()?.host,
            ),
            MagneticStudyPhase::ReadyForState { ordinal } => (
                MagneticStudyResultPhase::State { ordinal },
                &study.prepare_next_state()?.host,
            ),
            MagneticStudyPhase::ReadyForPublication => (
                MagneticStudyResultPhase::Publication,
                &study.prepare_publication()?.host,
            ),
            _ => {
                return Err(invalid(
                    "magnetic completion replay reached an awaiting phase",
                ))
            }
        };
        closure.validate_for(phase, host)?;
        match phase {
            MagneticStudyResultPhase::Solve => {
                study.complete_solve(source, closure.response, closure.execution)?
            }
            MagneticStudyResultPhase::FieldRecovery => {
                study.complete_field_recovery(source, closure.response, closure.execution)?
            }
            MagneticStudyResultPhase::ObservationPartition { ordinal } => study
                .complete_observation_partition(
                    source,
                    ordinal,
                    closure.response,
                    closure.execution,
                )?,
            MagneticStudyResultPhase::ObservationReduction { .. } => {
                study.complete_observation_reduction(source, closure.response, closure.execution)?
            }
            MagneticStudyResultPhase::ObservationAssembly => {
                study.complete_observation_assembly(source, closure.response, closure.execution)?
            }
            MagneticStudyResultPhase::Accuracy => {
                study.complete_accuracy(source, closure.response, closure.execution)?
            }
            MagneticStudyResultPhase::State { .. } => {
                study.complete_next_state(source, closure.response, closure.execution)?
            }
            MagneticStudyResultPhase::Publication => {
                study.complete_publication(source, closure.response, closure.execution)?
            }
        }
    }
    if closures.next().is_some() {
        return Err(invalid("magnetic completion contains an extra phase"));
    }
    study.into_completed()
}
