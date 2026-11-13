//! Governed program lifecycle. Scheduling decisions and original history belong to Driver.
use super::filesystem::{atomic_json, unix_millis};
use super::model::{
    DurableExecutionContext, DurableProgramAttempt, JobAttachment, LocalJobRecord, LocalJobState,
    ProgramBatchSubmission,
};
use super::scheduler::DurableScheduler;
use super::service::LocalSupervisor;
use super::store::JobStore;
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution::state::TaskState;
use runmat_execution_runner::{AttemptReport, DriverAction, DriverCommand};
use std::collections::VecDeque;

impl LocalSupervisor {
    pub async fn execution_context(
        &self,
        key: Option<&str>,
    ) -> NativeExecutionResult<DurableExecutionContext> {
        self.scheduler.lock().await.context(key)
    }

    pub async fn submit_program(
        &self,
        submission: ProgramBatchSubmission,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        self.submit_program_with_organization(
            submission,
            runmat_runtime::analysis::fea_run_admission::current_local_fea_organization_hint(),
        )
        .await
    }

    pub(super) async fn submit_program_with_organization(
        &self,
        submission: ProgramBatchSubmission,
        organization: runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        submission.validate()?;
        let mut store = self.store.lock().await;
        let mut scheduler = self.scheduler.lock().await;
        scheduler.validate_context(&submission.task)?;
        let artifact_id = submission
            .task
            .request
            .program_artifact_id()
            .ok_or_else(|| {
                crate::NativeExecutionError::Protocol(
                    "durable program submission has no program artifact".into(),
                )
            })?;
        let stored = crate::protocol::StoredProgram::Direct(Box::new(
            crate::protocol::StoredDirectProgram {
                recipe: submission.recipe.clone(),
                artifact: submission.artifact.clone(),
            },
        ));
        // Reject an unreadable worker artifact before publishing a job or admitting
        // scheduler work. The provider uses this same configured read ceiling.
        let encoded = runmat_canonical_codec::encode_json_bounded(
            &stored,
            u64::from(self.config.execution.max_message_bytes),
        )?;
        let (record, created) = store.create_program(&submission, unix_millis(), organization)?;
        if !created {
            // A duplicate returns the original authority; it never dispatches new physics.
            return Ok((record, true));
        }
        let artifacts =
            crate::local_store::ArtifactStore::new(self.config.paths.root.join("artifacts"))?;
        artifacts.put(artifact_id, &encoded)?;
        let mut actions = scheduler.tick()?;
        actions.extend(scheduler.admit(record.handle.id, &submission.task)?);
        self.dispatch_program_actions(&mut store, &mut scheduler, actions)
            .await?;
        Ok((store.record(record.handle.id)?, false))
    }

    pub(super) async fn reconcile_program(
        &self,
        store: &mut JobStore,
        record: &mut LocalJobRecord,
    ) -> NativeExecutionResult<()> {
        let submission = store
            .retained_submission(record.handle.id)?
            .ok_or_else(|| invalid("program reconciliation lacks its original request"))?;
        let mut scheduler = self.scheduler.lock().await;
        // Admission interrupted before durable publication can be completed from
        // the frozen caller submission. Existing admitted tasks are never reminted.
        let mut actions = scheduler.admit(record.handle.id, submission.task())?;
        let history = scheduler
            .history(record.handle.id)?
            .ok_or_else(|| invalid("durable program lacks original scheduler history"))?;
        for attempt in history
            .attempts
            .iter()
            .filter(|attempt| attempt.terminal.is_none())
        {
            let mut completed = store.program_report(record.handle.id, &attempt.request)?;
            if completed.is_none() {
                let owned =
                    super::ownership::is_owned(&store.job_dir(record.handle.id), &attempt.request)?;
                if owned {
                    if let Some(marker) = store.driver_marker(record.handle.id)? {
                        if record.state == LocalJobState::Starting
                            || record.driver_process_id.is_none()
                        {
                            record.state = LocalJobState::Running;
                            record.driver_process_id = Some(marker.process_id);
                            record.updated_unix_millis = unix_millis();
                            store.write_record(record)?;
                        }
                    }
                } else {
                    // The provider publishes while owning the lease. Re-read after
                    // observing release so publication cannot race a false loss report.
                    completed = store.program_report(record.handle.id, &attempt.request)?;
                    if completed.is_none() {
                        if record.state == LocalJobState::Queued {
                            actions.push(DriverAction::Launch(attempt.request.clone()));
                        } else if store.driver_marker(record.handle.id)?.is_some()
                            || unix_millis().saturating_sub(record.updated_unix_millis)
                                >= super::service::STARTING_RECOVERY_GRACE_MILLIS
                        {
                            let mut report = runmat_execution_runner::port::BackendReport::for_request(
                                &attempt.request,
                                AttemptReport::Lost {
                                    message: "original durable provider ownership ended without a committed report; physical worker quiescence is unknown".into(),
                                },
                            );
                            report.console = super::console_journal::read_transcript(
                                &store.job_dir(record.handle.id),
                                &attempt.request,
                            )?
                            .map(Box::new);
                            actions.extend(scheduler.command(
                                DriverCommand::WorkerLostWithReport {
                                    worker_id: attempt.request.worker_id,
                                    report: Box::new(report),
                                },
                            )?);
                        }
                    }
                }
            }
            if let Some(completed) = completed {
                // Reaffirm visible report retention before committing its original
                // fenced evidence, including after uncertain producer delivery.
                super::program_store::write_program_completion(
                    &store.job_dir(record.handle.id),
                    completed.clone(),
                )?;
                actions.extend(scheduler.command(DriverCommand::BackendReport(completed.report))?);
            }
        }
        self.dispatch_program_actions(store, &mut scheduler, actions)
            .await?;
        let history = scheduler
            .history(record.handle.id)?
            .ok_or_else(|| invalid("durable routing lost its original task"))?;
        let mut current = store.record(record.handle.id)?;
        let active = history
            .attempts
            .iter()
            .any(|attempt| attempt.terminal.is_none());
        current.state = match history.task.state {
            TaskState::Succeeded => LocalJobState::Succeeded,
            TaskState::Failed => LocalJobState::Failed,
            TaskState::Indeterminate => LocalJobState::Indeterminate,
            TaskState::Cancelled if active => LocalJobState::Cancelling,
            TaskState::Cancelled => LocalJobState::Cancelled,
            _ => current.state,
        };
        if current.state.is_terminal() {
            current.driver_process_id = None;
            current.message = history
                .attempts
                .last()
                .and_then(|a| a.terminal.as_ref())
                .and_then(|t| t.diagnostic.clone());
        }
        if current != *record {
            current.updated_unix_millis = unix_millis();
            store.write_record(&mut current)?;
        }
        *record = current;
        Ok(())
    }

    pub(super) async fn attach_program(
        &self,
        store: &JobStore,
        attachment: &mut JobAttachment,
    ) -> NativeExecutionResult<()> {
        if store
            .retained_submission(attachment.record.handle.id)?
            .is_none()
        {
            return Ok(());
        }
        let scheduler = self.scheduler.lock().await;
        let history = scheduler
            .history(attachment.record.handle.id)?
            .ok_or_else(|| invalid("durable attachment lacks its admitted task"))?;
        history.validate()?;
        if let Some(last) = history.attempts.last() {
            if last.terminal.is_some() {
                attachment.resources = last
                    .terminal
                    .as_ref()
                    .and_then(|t| t.resources.as_deref())
                    .cloned();
                if let Some(completed) =
                    store.program_report(attachment.record.handle.id, &last.request)?
                {
                    attachment.response = Some(completed.response);
                    attachment.latest_progress = completed.latest_progress;
                }
            }
        }
        attachment.execution = Some(history);
        Ok(())
    }

    pub(super) async fn cancel_program(
        &self,
        store: &mut JobStore,
        mut record: LocalJobRecord,
        reason: runmat_execution::CancellationReason,
    ) -> NativeExecutionResult<LocalJobRecord> {
        if record.state.is_terminal() {
            return Ok(record);
        }
        let mut scheduler = self.scheduler.lock().await;
        let history = scheduler
            .history(record.handle.id)?
            .ok_or_else(|| invalid("durable cancellation lacks original admission"))?;
        let now_millis = unix_millis().max(scheduler.driver.now_millis());
        let actions = scheduler.command(DriverCommand::CancelTask {
            task_id: history.task.id(),
            reason,
            now_millis,
        })?;
        self.dispatch_program_actions(store, &mut scheduler, actions)
            .await?;
        record = store.record(record.handle.id)?;
        record.state = if history.attempts.is_empty() {
            LocalJobState::Cancelled
        } else {
            LocalJobState::Cancelling
        };
        record.updated_unix_millis = unix_millis();
        store.write_record(&mut record)?;
        Ok(record)
    }

    pub(super) async fn dispatch_program_actions(
        &self,
        store: &mut JobStore,
        scheduler: &mut DurableScheduler,
        actions: Vec<DriverAction>,
    ) -> NativeExecutionResult<()> {
        let mut actions = VecDeque::from(actions);
        while let Some(action) = actions.pop_front() {
            match action {
                DriverAction::Launch(request) => {
                    let id = scheduler.route(request.task_id)?;
                    let record = store.record(id)?;
                    if super::ownership::is_owned(&store.job_dir(id), &request)? {
                        continue;
                    }
                    let assignment = DurableProgramAttempt {
                        schema_version: 1,
                        request: request.clone(),
                        config: self.config.execution.clone(),
                        job_directory: store.job_dir(id),
                    };
                    store.assign_program(id, &assignment)?;
                    actions.extend(scheduler.command(DriverCommand::BackendReport(
                        runmat_execution_runner::port::BackendReport::for_request(
                            &request,
                            AttemptReport::Started,
                        ),
                    ))?);
                    let record = self.launch(store, record).await?;
                    if record.state == LocalJobState::Failed {
                        let report = runmat_execution_runner::port::BackendReport::for_request(
                            &request,
                            AttemptReport::Lost {
                                message: record.message.unwrap_or_else(|| {
                                    "durable provider launch unconfirmed".into()
                                }),
                            },
                        );
                        actions.extend(scheduler.command(DriverCommand::WorkerLostWithReport {
                            worker_id: request.worker_id,
                            report: Box::new(report),
                        })?);
                    }
                }
                DriverAction::Cancel(request) | DriverAction::Terminate(request) => {
                    let id = scheduler.route(request.task_id)?;
                    store.require_program_assignment(id, &request)?;
                    // The original fenced cancellation reaches the provider's normal
                    // process-control seam. Escalation remains a shared Driver decision.
                    let cause = scheduler
                        .driver
                        .task_record(request.task_id)?
                        .cancellation
                        .ok_or_else(|| {
                            invalid("scheduler stop has no original cancellation cause")
                        })?
                        .cause;
                    atomic_json(
                        &super::program_store::cancel_path(&store.job_dir(id), request.id),
                        &crate::process_provider::cancellation::CancellationCommand {
                            request,
                            cause,
                        },
                    )?;
                    // Both escalation and cancellation reach the original provider,
                    // which owns its live ChildProcess and observes termination.
                    // Saved PID metadata is never a signalling capability.
                }
                DriverAction::Checkpoint
                | DriverAction::ResizePool { .. }
                | DriverAction::GarbageCollectResults { .. } => {}
            }
        }
        Ok(())
    }
}
fn invalid(message: impl Into<String>) -> NativeExecutionError {
    NativeExecutionError::Protocol(message.into())
}

#[cfg(test)]
mod tests;
