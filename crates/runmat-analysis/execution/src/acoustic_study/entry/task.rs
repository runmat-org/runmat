//! Dispatch prepared entry phases without choosing placement or executing numerical work.
use super::*;

impl AcousticStudyEntryCoordinator {
    pub fn prepare_next(&mut self, source: &impl CacheImport) -> AnalysisBridgeResult<()> {
        match self.phase {
            AcousticStudyPhase::ReadyForSolve => {
                self.prepare_solve()?;
            }
            AcousticStudyPhase::ReadyForFieldRecovery => {
                self.prepare_field_recovery()?;
            }
            AcousticStudyPhase::ReadyForObservationFanout => {
                self.prepare_observation_fanout(source)?;
                // A frequency no authored observation samples continues with accuracy.
                if self.phase == AcousticStudyPhase::ReadyForAccuracy {
                    self.prepare_accuracy()?;
                } else {
                    self.prepare_observation_partition()?;
                }
            }
            AcousticStudyPhase::ReadyForObservationPartition { .. } => {
                self.prepare_observation_partition()?;
            }
            AcousticStudyPhase::ReadyForObservationReduction { .. } => {
                self.prepare_observation_reduction()?;
            }
            AcousticStudyPhase::ReadyForObservationAssembly => {
                self.prepare_observation_assembly()?;
            }
            AcousticStudyPhase::ReadyForAccuracy => {
                self.prepare_accuracy()?;
            }
            AcousticStudyPhase::ReadyForState { .. } => {
                self.prepare_next_state()?;
            }
            AcousticStudyPhase::ReadyForBaseline => {
                self.freeze_baseline()?;
                self.prepare_publication()?;
            }
            AcousticStudyPhase::ReadyForPublication => {
                self.prepare_publication()?;
            }
            _ => return Err(invalid("entry has no ready task phase")),
        }
        Ok(())
    }
    pub fn pending_task(
        &self,
    ) -> AnalysisBridgeResult<(
        &AnalysisHostWorkload,
        &crate::PreparedAnalysisProgram,
        &FeaResourcePlan,
    )> {
        match self.phase {
            AcousticStudyPhase::AwaitingSolve => {
                let task = self.solve_task.as_ref().expect("awaited solve");
                Ok((&task.host, &task.program, &task.resource_plan))
            }
            AcousticStudyPhase::AwaitingFieldRecovery => {
                let task = self.recovery_task.as_ref().expect("awaited recovery");
                Ok((&task.host, &task.program, &task.resource_plan))
            }
            AcousticStudyPhase::AwaitingObservationPartition { ordinal } => {
                let task = self.observation_state.partition(ordinal)?;
                Ok((&task.host, &task.program, &task.resource_plan))
            }
            AcousticStudyPhase::AwaitingObservationReduction { .. } => {
                let task = self
                    .observation_state
                    .reduction
                    .as_ref()
                    .expect("awaited reduction");
                Ok((&task.host, &task.program, &task.resource_plan))
            }
            AcousticStudyPhase::AwaitingObservationAssembly => {
                let task = self
                    .observation_state
                    .assembly
                    .as_ref()
                    .expect("awaited assembly");
                Ok((&task.host, &task.program, &task.resource_plan))
            }
            AcousticStudyPhase::AwaitingAccuracy => {
                let task = self.accuracy_task.as_ref().expect("awaited accuracy");
                Ok((&task.host, &task.program, &task.resource_plan))
            }
            AcousticStudyPhase::AwaitingState { .. } => {
                let task = self.state_task.as_ref().expect("awaited state");
                Ok((&task.host, &task.program, &task.resource_plan))
            }
            AcousticStudyPhase::AwaitingPublication => {
                let task = self.publication_task.as_ref().expect("awaited publication");
                Ok((&task.host, &task.program, &task.resource_plan))
            }
            _ => Err(invalid("entry has no pending task phase")),
        }
    }
    pub fn complete_next(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        match self.phase {
            AcousticStudyPhase::AwaitingSolve => self.complete_solve(source, response, accounting),
            AcousticStudyPhase::AwaitingFieldRecovery => {
                self.complete_field_recovery(source, response, accounting)
            }
            AcousticStudyPhase::AwaitingObservationPartition { .. } => {
                self.complete_observation_partition(source, response, accounting)
            }
            AcousticStudyPhase::AwaitingObservationReduction { .. } => {
                self.complete_observation_reduction(source, response, accounting)
            }
            AcousticStudyPhase::AwaitingObservationAssembly => {
                self.complete_observation_assembly(source, response, accounting)
            }
            AcousticStudyPhase::AwaitingAccuracy => {
                self.complete_accuracy(source, response, accounting)
            }
            AcousticStudyPhase::AwaitingState { .. } => {
                self.complete_next_state(source, response, accounting)
            }
            AcousticStudyPhase::AwaitingPublication => {
                self.complete_publication(source, response, accounting)
            }
            _ => Err(invalid("entry has no pending completion phase")),
        }
    }
}
