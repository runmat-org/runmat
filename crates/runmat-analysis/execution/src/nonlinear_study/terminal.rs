//! Delegate terminal phases to the common authority; retain original replay associations.
use super::*;

impl NonlinearSolidStudyCoordinator {
    pub fn prepare_accuracy(&mut self) -> AnalysisBridgeResult<&PreparedAccuracy> {
        let retained = self.admit_retention(None, None)?;
        let import_workspace = self.import_workspace_bytes;
        self.tail_mut()?
            .set_retained_context_bytes(retained, import_workspace);
        self.tail_mut()?.prepare_accuracy()
    }
    pub fn complete_accuracy(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        let task = self
            .tail
            .as_ref()
            .and_then(|t| t.accuracy_task.as_ref())
            .ok_or_else(|| invalid("nonlinear accuracy is not prepared"))?;
        let phase = NonlinearSolidStudyResultPhase::Accuracy;
        self.admit_retention(
            Some((phase, &task.host.workload, &response, &accounting)),
            Some(&task.resource_plan),
        )?;
        let closure = NonlinearSolidStudyResultClosure::new(
            phase,
            &task.host,
            response.clone(),
            accounting.clone(),
        );
        self.tail_mut()?
            .complete_accuracy(source, response, accounting)?;
        self.closures.push(closure);
        Ok(())
    }
    pub fn prepare_next_state(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedSemanticStatePublication> {
        let retained = self.admit_retention(None, None)?;
        let import_workspace = self.import_workspace_bytes;
        self.tail_mut()?
            .set_retained_context_bytes(retained, import_workspace);
        self.tail_mut()?.prepare_next_state()
    }
    pub fn complete_next_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        let task = self
            .tail
            .as_ref()
            .and_then(|t| t.state_task.as_ref())
            .ok_or_else(|| invalid("nonlinear state is not prepared"))?;
        let phase = NonlinearSolidStudyResultPhase::State {
            ordinal: self.states().len(),
        };
        self.admit_retention(
            Some((phase, &task.host.workload, &response, &accounting)),
            Some(&task.resource_plan),
        )?;
        let closure = NonlinearSolidStudyResultClosure::new(
            phase,
            &task.host,
            response.clone(),
            accounting.clone(),
        );
        self.tail_mut()?
            .complete_next_state(source, response, accounting)?;
        self.closures.push(closure);
        Ok(())
    }
    pub fn freeze_baseline(&mut self) -> AnalysisBridgeResult<&PreparedResourceUsage> {
        self.tail_mut()?.freeze_baseline()
    }
    pub fn prepare_publication(&mut self) -> AnalysisBridgeResult<&PreparedResultPublication> {
        let retained = self.admit_retention(None, None)?;
        let import_workspace = self.import_workspace_bytes;
        self.tail_mut()?
            .set_retained_context_bytes(retained, import_workspace);
        self.tail_mut()?.prepare_publication()
    }
    pub fn complete_publication(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        let task = self
            .tail
            .as_ref()
            .and_then(|t| t.publication_task.as_ref())
            .ok_or_else(|| invalid("nonlinear publication is not prepared"))?;
        let phase = NonlinearSolidStudyResultPhase::Publication;
        self.admit_retention(
            Some((phase, &task.host.workload, &response, &accounting)),
            Some(&task.resource_plan),
        )?;
        let closure = NonlinearSolidStudyResultClosure::new(
            phase,
            &task.host,
            response.clone(),
            accounting.clone(),
        );
        self.tail_mut()?
            .complete_publication(source, response, accounting)?;
        self.closures.push(closure);
        Ok(())
    }
}
