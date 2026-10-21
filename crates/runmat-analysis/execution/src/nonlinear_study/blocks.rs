use super::*;
use crate::phase_accounting::validate_analysis_phase_accounting as validate_accounting;

impl NonlinearSolidStudyCoordinator {
    pub fn prepare_block(&mut self) -> AnalysisBridgeResult<&PreparedNonlinearSolidBlockTask> {
        let NonlinearSolidStudyPhase::ReadyForBlock { ordinal } = self.phase() else {
            return Err(invalid("nonlinear study is not ready for a block"));
        };
        let p = &self.preparation;
        let workload = &p.product.workload;
        let request = match &self.block {
            None => NonlinearSolidBlockRequest::initial(
                workload,
                Digest::from_bytes(*p.mesh.canonical_digest.bytes()),
                p.accepted_block_limit,
            )?,
            Some(previous) => NonlinearSolidBlockRequest::continue_original(
                workload,
                &previous.block,
                p.accepted_block_limit,
            )?,
        };
        if request.block_ordinal != ordinal {
            return Err(invalid("original block ordinal differs"));
        }
        self.admit_retention(None, None)?;
        let task = prepare_nonlinear_solid_block_task(NonlinearSolidBlockPreparation {
            local_execution: p.local_execution,
            authored_study_digest: self.definition,
            frozen_project_digest: p.frozen_project_digest,
            program_revision: p.program_revision.clone(),
            artifact_access: p.artifact_access.clone(),
            workload: workload.clone(),
            mesh: p.mesh.clone(),
            request,
            predecessor: self.block.as_ref().map(|b| b.objects.clone()),
            output_policy: p.output_policy,
        })?;
        self.admit_retention(None, Some(&task.resource_plan))?;
        self.block_task = Some(task);
        self.phase = NonlinearSolidStudyPhase::AwaitingBlock { ordinal };
        Ok(self.block_task.as_ref().expect("prepared block"))
    }

    pub fn complete_block(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        let NonlinearSolidStudyPhase::AwaitingBlock { ordinal } = self.phase() else {
            return Err(invalid("nonlinear study has no awaited block"));
        };
        let task = self.block_task.as_ref().expect("awaited block");
        let successor = ordinal
            .checked_add(1)
            .ok_or_else(|| invalid("nonlinear block ordinal overflow"))?;
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let phase = NonlinearSolidStudyResultPhase::Block { ordinal };
        self.admit_retention(
            Some((phase, &task.host.workload, &response, &accounting)),
            Some(&task.resource_plan),
        )?;
        let imported = import_nonlinear_solid_block(source, response.clone(), task)?;
        let task_attempts = accounting.resource_graph(&task.resource_plan)?.attempts;
        let mut attempts = self.attempts.clone();
        attempts.extend(task_attempts.iter().cloned());
        let tail = if imported.block.complete
            && self.preparation.product.workload.model.sequence.steps[0]
                .observations
                .is_empty()
        {
            Some(self.build_original_tail(
                &task.host,
                &task.resource_plan,
                &imported,
                &attempts,
                None,
                accounting.graph_wall_millis(),
            )?)
        } else {
            None
        };
        let source_host = task.host.clone();
        let source_plan = task.resource_plan.clone();
        let complete = imported.block.complete;
        // All fallible source/accounting/admission work precedes visible lifecycle mutation.
        let closure =
            NonlinearSolidStudyResultClosure::new(phase, &task.host, response, accounting);
        self.attempts.extend(task_attempts);
        self.closures.push(closure);
        self.block = Some(imported);
        // Replay retains immutable references; input/program byte copies from the
        // completed task must not keep older decoded windows alive during the tail.
        self.block_task = None;
        self.tail = tail;
        if complete {
            self.source_host = Some(source_host);
            self.source_plan = Some(source_plan);
        }
        if self.tail.is_none() {
            self.phase = if complete {
                NonlinearSolidStudyPhase::ReadyForFieldRecovery
            } else {
                NonlinearSolidStudyPhase::ReadyForBlock { ordinal: successor }
            };
        }
        Ok(())
    }
}
