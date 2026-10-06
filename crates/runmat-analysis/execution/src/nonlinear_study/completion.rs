//! Ordered cold replay rederives tasks and admits saved responses, never executes physics.
use super::*;
use serde::{Deserialize, Serialize};
mod projection;
pub use projection::OriginalNonlinearSolidBlockProjection;
mod retained;
mod workspace;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum NonlinearSolidStudyResultPhase {
    Block { ordinal: u64 },
    FieldRecovery,
    ObservationPartition { ordinal: usize },
    ObservationReduction { ordinal: usize },
    ObservationAssembly,
    Accuracy,
    State { ordinal: usize },
    Publication,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearSolidStudyResultClosure {
    pub phase: NonlinearSolidStudyResultPhase,
    pub workload: AnalysisWorkloadRequest,
    pub response: ProgramExecutionResponse,
    pub execution: AnalysisPhaseExecutionEvidence,
}
impl NonlinearSolidStudyResultClosure {
    pub(super) fn new(
        phase: NonlinearSolidStudyResultPhase,
        host: &AnalysisHostWorkload,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> Self {
        Self {
            phase,
            workload: host.workload.clone(),
            response,
            execution: accounting,
        }
    }
}

impl CompletedNonlinearSolidStudy {
    /// Import one original output block through its preceding task chain. Earlier decoded
    /// physics is released as replay advances; no fields or point history are reconstructed.
    pub fn import_original_block(
        &self,
        source: &impl CacheImport,
        ordinal: u64,
    ) -> AnalysisBridgeResult<ImportedNonlinearSolidBlock> {
        self.import_original_block_with_retained(source, ordinal, 0)
    }
    pub fn import_original_block_with_retained(
        &self,
        source: &impl CacheImport,
        ordinal: u64,
        additional_retained_bytes: u64,
    ) -> AnalysisBridgeResult<ImportedNonlinearSolidBlock> {
        let retained = self.retained_component_bytes()?;
        let mut study = NonlinearSolidStudyCoordinator::new(self.preparation()?)?;
        study.external_retained_bytes = retained
            .checked_add(additional_retained_bytes)
            .ok_or_else(|| limit("nonlinear prefix retained memory overflow"))?;
        projection::import_prefix(source, &mut study, &self.closures, ordinal)
    }
}

pub fn import_nonlinear_solid_study_completion(
    source: &impl CacheImport,
    preparation: NonlinearSolidStudyPreparation,
    closures: Vec<NonlinearSolidStudyResultClosure>,
) -> AnalysisBridgeResult<CompletedNonlinearSolidStudy> {
    import_with_retained(source, preparation, closures, 0)
}

pub(super) fn import_with_retained(
    source: &impl CacheImport,
    preparation: NonlinearSolidStudyPreparation,
    closures: Vec<NonlinearSolidStudyResultClosure>,
    additional_retained_bytes: u64,
) -> AnalysisBridgeResult<CompletedNonlinearSolidStudy> {
    let mut study = NonlinearSolidStudyCoordinator::new(preparation)?;
    // IntoIter keeps its original slot allocation even as owned closures transfer.
    let additional_retained_bytes = (closures.capacity() as u64)
        .checked_mul(std::mem::size_of::<NonlinearSolidStudyResultClosure>() as u64)
        .and_then(|n| n.checked_add(additional_retained_bytes))
        .ok_or_else(|| limit("nonlinear pending slot inventory overflow"))?;
    retained::admit_pending(
        &mut study,
        additional_retained_bytes,
        &(
            None::<&NonlinearSolidStudyResultClosure>,
            closures.as_slice(),
        ),
    )?;
    workspace::validate(&closures)?;
    study.import_workspace_bytes = Some(0);
    study.admit_retention(None, None)?;
    let mut closures = closures.into_iter();
    loop {
        retained::admit_pending(
            &mut study,
            additional_retained_bytes,
            &(
                None::<&NonlinearSolidStudyResultClosure>,
                closures.as_slice(),
            ),
        )?;
        if study.phase() == NonlinearSolidStudyPhase::ReadyForObservationFanout {
            study.prepare_observation_fanout(source)?;
            study.import_workspace_bytes = Some(0);
        }
        if study.phase() == NonlinearSolidStudyPhase::ReadyForBaseline {
            study.freeze_baseline()?;
        }
        if study.phase() == NonlinearSolidStudyPhase::Complete {
            break;
        }
        let closure = closures
            .next()
            .ok_or_else(|| invalid("nonlinear completion omits original phase"))?;
        retained::admit_pending(
            &mut study,
            additional_retained_bytes,
            &(Some(&closure), closures.as_slice()),
        )?;
        workspace::admit(source, &mut study, &closure)?;
        let (phase, host) = match study.phase() {
            NonlinearSolidStudyPhase::ReadyForBlock { ordinal } => (
                NonlinearSolidStudyResultPhase::Block { ordinal },
                &study.prepare_block()?.host,
            ),
            NonlinearSolidStudyPhase::ReadyForFieldRecovery => (
                NonlinearSolidStudyResultPhase::FieldRecovery,
                &study.prepare_field_recovery()?.host,
            ),
            NonlinearSolidStudyPhase::AwaitingObservationPartitions => {
                let NonlinearSolidStudyResultPhase::ObservationPartition { ordinal } =
                    closure.phase
                else {
                    return Err(invalid(
                        "original nonlinear observation partition phase absent",
                    ));
                };
                (closure.phase, &study.observation_partition(ordinal)?.host)
            }
            NonlinearSolidStudyPhase::ReadyForObservationReduction { ordinal } => (
                NonlinearSolidStudyResultPhase::ObservationReduction { ordinal },
                &study.prepare_observation_reduction()?.host,
            ),
            NonlinearSolidStudyPhase::ReadyForObservationAssembly => (
                NonlinearSolidStudyResultPhase::ObservationAssembly,
                &study.prepare_observation_assembly()?.host,
            ),
            NonlinearSolidStudyPhase::ReadyForAccuracy => (
                NonlinearSolidStudyResultPhase::Accuracy,
                &study.prepare_accuracy()?.host,
            ),
            NonlinearSolidStudyPhase::ReadyForState { ordinal } => (
                NonlinearSolidStudyResultPhase::State { ordinal },
                &study.prepare_next_state()?.host,
            ),
            NonlinearSolidStudyPhase::ReadyForPublication => (
                NonlinearSolidStudyResultPhase::Publication,
                &study.prepare_publication()?.host,
            ),
            _ => {
                return Err(invalid(
                    "nonlinear completion contains an invalid pending phase",
                ))
            }
        };
        if closure.phase != phase || closure.workload != host.workload {
            return Err(invalid(
                "nonlinear completion phase/workload differs from original task",
            ));
        }
        match phase {
            NonlinearSolidStudyResultPhase::Block { .. } => {
                study.complete_block(source, closure.response, closure.execution)?
            }
            NonlinearSolidStudyResultPhase::FieldRecovery => {
                study.complete_field_recovery(source, closure.response, closure.execution)?
            }
            NonlinearSolidStudyResultPhase::ObservationPartition { ordinal } => study
                .complete_observation_partition(
                    source,
                    ordinal,
                    closure.response,
                    closure.execution,
                )?,
            NonlinearSolidStudyResultPhase::ObservationReduction { .. } => {
                study.complete_observation_reduction(source, closure.response, closure.execution)?
            }
            NonlinearSolidStudyResultPhase::ObservationAssembly => {
                study.complete_observation_assembly(source, closure.response, closure.execution)?
            }
            NonlinearSolidStudyResultPhase::Accuracy => {
                study.complete_accuracy(source, closure.response, closure.execution)?
            }
            NonlinearSolidStudyResultPhase::State { .. } => {
                study.complete_next_state(source, closure.response, closure.execution)?
            }
            NonlinearSolidStudyResultPhase::Publication => {
                study.complete_publication(source, closure.response, closure.execution)?
            }
        }
        // Field fanout still consumes the original recovered source in the next
        // transition. Other completed phases retain no temporary decoder workspace.
        if phase != NonlinearSolidStudyResultPhase::FieldRecovery {
            study.import_workspace_bytes = Some(0);
        }
    }
    if closures.next().is_some() {
        return Err(invalid(
            "nonlinear completion contains unrelated trailing phases",
        ));
    }
    study.finish()
}
