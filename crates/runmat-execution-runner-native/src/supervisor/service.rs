use std::sync::Arc;

use runmat_execution::JobId;
use runmat_process_host::environment::EnvironmentPolicy;
use runmat_process_host::{is_process_alive, ChildLifetime, HiddenMode, HostCommand, StdioPolicy};
use tokio::sync::Mutex;

use super::auth::acquire_supervisor_lock;
use super::driver_files::write_driver_marker;
use super::filesystem::{secure_directory, unix_millis};
use super::model::{BatchSubmission, GraphBatchSubmission, LocalJobRecord, LocalJobState};
use super::protocol::{SupervisorCommand, SupervisorResponse};
use super::store::JobStore;
use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) const STARTING_RECOVERY_GRACE_MILLIS: u64 = 10_000;

use super::LocalSupervisorConfig;

pub struct LocalSupervisor {
    // This guard outlives all service-owned reconciliation and connection futures.
    _authority: runmat_filesystem::ExclusiveFileLock,
    pub(super) config: LocalSupervisorConfig,
    pub(super) store: Mutex<JobStore>,
    pub(super) scheduler: Mutex<super::scheduler::DurableScheduler>,
}

impl LocalSupervisor {
    pub fn open(config: LocalSupervisorConfig) -> NativeExecutionResult<Arc<Self>> {
        config.validate()?;
        secure_directory(&config.paths.root)?;
        let authority = acquire_supervisor_lock(&config.paths)?;
        let store = JobStore::open(config.paths.clone())?;
        let scheduler = super::scheduler::DurableScheduler::open(&config, &authority)?;
        Ok(Arc::new(Self {
            _authority: authority,
            config,
            store: Mutex::new(store),
            scheduler: Mutex::new(scheduler),
        }))
    }

    pub fn object_store(&self) -> NativeExecutionResult<crate::FilesystemObjectStore> {
        crate::FilesystemObjectStore::open(
            self.config.paths.root.join("objects"),
            self.config.execution.max_object_bytes,
        )
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))
    }

    pub async fn submit(
        &self,
        submission: BatchSubmission,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        self.submit_with_organization(
            submission,
            runmat_runtime::analysis::fea_run_admission::current_local_fea_organization_hint(),
        )
        .await
    }

    async fn submit_with_organization(
        &self,
        submission: BatchSubmission,
        organization: runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        let mut store = self.store.lock().await;
        let (record, created) = store.create_script(submission, unix_millis(), organization)?;
        if !created {
            return Ok((record, true));
        }
        let record = self.launch(&mut store, record).await?;
        Ok((record, false))
    }

    pub async fn submit_graph(
        &self,
        submission: GraphBatchSubmission,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        self.submit_graph_with_organization(
            submission,
            runmat_runtime::analysis::fea_run_admission::current_local_fea_organization_hint(),
        )
        .await
    }

    async fn submit_graph_with_organization(
        &self,
        submission: GraphBatchSubmission,
        organization: runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        // Canonical archive decode and selected-V3 admission can be expensive;
        // complete it before entering the durable-store critical section.
        let validated = submission.into_validated()?;
        if let Some(task) = &validated.submission.task {
            let request = &validated.submission.request;
            let admitted = request
                .validate_and_admit_bundles()
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
            crate::graph_submission::validate_scratch_bundles(&admitted, task)?;
            let artifacts =
                crate::local_store::ArtifactStore::new(self.config.paths.root.join("artifacts"))?;
            #[derive(serde::Serialize)]
            struct BorrowedStoredGraph<'a> {
                request: &'a runmat_execution_artifact::GraphExecutionRequest,
            }
            let stored = artifacts.put_streamed_graph(
                &BorrowedStoredGraph { request },
                runmat_execution::resource::MAX_GRAPH_PROGRAM_WIRE_BYTES,
            )?;
            crate::graph_submission::validate_floor(request, task, stored.length)?;
        }
        let mut store = self.store.lock().await;
        if let Some(task) = &validated.submission.task {
            let mut scheduler = self.scheduler.lock().await;
            scheduler.validate_context(task)?;
            // Preserve the complete original scheduler submission, not merely
            // the request's derived callable or a replacement resource grant.
            let task = task.clone();
            let (record, created) =
                store.create_graph_validated(validated, unix_millis(), organization)?;
            if !created {
                return Ok((record, true));
            }
            let mut actions = scheduler.tick()?;
            actions.extend(scheduler.admit(record.handle.id, &task)?);
            self.dispatch_program_actions(&mut store, &mut scheduler, actions)
                .await?;
            return Ok((store.record(record.handle.id)?, false));
        }
        let (record, created) =
            store.create_graph_validated(validated, unix_millis(), organization)?;
        if !created {
            return Ok((record, true));
        }
        let record = self.launch(&mut store, record).await?;
        Ok((record, false))
    }

    pub async fn list(&self) -> NativeExecutionResult<Vec<LocalJobRecord>> {
        self.reconcile().await?;
        self.store.lock().await.list()
    }

    pub async fn show(&self, job_id: JobId) -> NativeExecutionResult<LocalJobRecord> {
        self.reconcile().await?;
        self.store.lock().await.record(job_id)
    }

    pub async fn attach(
        &self,
        job_id: JobId,
        stdout_offset: u64,
        stderr_offset: u64,
    ) -> NativeExecutionResult<super::model::JobAttachment> {
        self.reconcile().await?;
        let store = self.store.lock().await;
        let mut attachment = store.attachment(job_id, stdout_offset, stderr_offset)?;
        self.attach_program(&store, &mut attachment).await?;
        Ok(attachment)
    }

    pub async fn reconcile(&self) -> NativeExecutionResult<()> {
        let mut store = self.store.lock().await;
        let now = unix_millis();
        let records = store.list()?;
        {
            let mut scheduler = self.scheduler.lock().await;
            let actions = scheduler.tick_if_active()?;
            self.dispatch_program_actions(&mut store, &mut scheduler, actions)
                .await?;
        }
        for mut record in records {
            if store.retained_submission(record.handle.id)?.is_some() {
                self.reconcile_program(&mut store, &mut record).await?;
                continue;
            }
            if record.state.is_terminal() {
                continue;
            }
            if let Some(completion) = store.completion(record.handle.id)? {
                record.state = if record.state == LocalJobState::Cancelling {
                    LocalJobState::Cancelled
                } else if completion.success {
                    LocalJobState::Succeeded
                } else {
                    LocalJobState::Failed
                };
                record.driver_process_id = None;
                record.exit_code = completion.exit_code;
                record.message = completion.message;
                record.updated_unix_millis = now;
                store.write_record(&mut record)?;
                continue;
            }
            match record.state {
                LocalJobState::Queued => {
                    let _ = self.launch(&mut store, record).await?;
                }
                LocalJobState::Starting => {
                    if let Some(marker) = store.driver_marker(record.handle.id)? {
                        if is_process_alive(marker.process_id) {
                            record.state = LocalJobState::Running;
                            record.driver_process_id = Some(marker.process_id);
                            record.updated_unix_millis = now;
                            store.write_record(&mut record)?;
                        } else {
                            mark_indeterminate(
                                &store,
                                &mut record,
                                now,
                                "driver exited during supervisor handoff",
                            )?;
                        }
                    } else if now.saturating_sub(record.updated_unix_millis)
                        >= STARTING_RECOVERY_GRACE_MILLIS
                    {
                        mark_indeterminate(
                            &store,
                            &mut record,
                            now,
                            "supervisor restarted during driver launch",
                        )?;
                    }
                }
                LocalJobState::Running | LocalJobState::Cancelling => {
                    let process_id = record.driver_process_id.expect("validated running record");
                    if !is_process_alive(process_id) {
                        if record.state == LocalJobState::Cancelling {
                            record.state = LocalJobState::Cancelled;
                            record.message = Some("driver stopped after cancellation".into());
                            record.driver_process_id = None;
                            record.updated_unix_millis = now;
                            store.write_record(&mut record)?;
                        } else {
                            mark_indeterminate(
                                &store,
                                &mut record,
                                now,
                                "driver exited without a committed completion",
                            )?;
                        }
                    }
                }
                LocalJobState::Succeeded
                | LocalJobState::Failed
                | LocalJobState::Cancelled
                | LocalJobState::Indeterminate => {}
            }
        }
        let scheduler = self.scheduler.lock().await;
        let _ = store.gc(now, |job| {
            scheduler.job_has_provider(&store, job).map(|owned| !owned)
        })?;
        Ok(())
    }

    pub(super) async fn launch(
        &self,
        store: &mut JobStore,
        mut record: LocalJobRecord,
    ) -> NativeExecutionResult<LocalJobRecord> {
        record.state = LocalJobState::Starting;
        record.driver_process_id = None;
        record.updated_unix_millis = unix_millis();
        store.write_record(&mut record)?;

        let job_dir = store.job_dir(record.handle.id);
        let mut command = HostCommand::new(&self.config.execution.executable);
        command.arguments = vec![HiddenMode::ExecutionDriver.marker().into()];
        command.environment_policy = EnvironmentPolicy::InheritExcept(vec!["RUNMAT_ORG_ID".into()]);
        if let Some(org) =
            crate::config::local_organization_environment(record.local_fea_organization)
        {
            command.environment.insert("RUNMAT_ORG_ID".into(), org);
        }
        command.environment.insert(
            "RUNMAT_EXECUTION_JOB_DIR".into(),
            job_dir.to_string_lossy().into_owned(),
        );
        command.environment.insert(
            crate::NATIVE_OBJECT_STORE_ROOT_ENV.into(),
            self.config
                .paths
                .root
                .join("objects")
                .to_string_lossy()
                .into_owned(),
        );
        if let Some(assignment) = store.program_assignment(record.handle.id)? {
            command.environment.insert(
                super::program_store::ATTEMPT_ID_ENV.into(),
                assignment.request.id.to_string(),
            );
        }
        command.lifetime = ChildLifetime::Detached;
        command.stdio = StdioPolicy::Files {
            stdout: store.stdout_path(record.handle.id),
            stderr: store.stderr_path(record.handle.id),
        };
        command.max_stderr_bytes = self.config.execution.max_stderr_bytes;
        let child = match command.spawn().await {
            Ok(child) => child,
            Err(error) => {
                record.state = LocalJobState::Failed;
                record.updated_unix_millis = unix_millis();
                record.message = Some(format!("failed to launch local batch driver: {error}"));
                store.write_record(&mut record)?;
                return Ok(record);
            }
        };
        let process_id = child.id().ok_or_else(|| {
            NativeExecutionError::Protocol("batch driver did not expose a process id".into())
        })?;
        if store.program_assignment(record.handle.id)?.is_none() {
            // Program markers are published only by the original leased provider.
            // A parent PID observation cannot establish physical attempt ownership.
            write_driver_marker(&job_dir, process_id, None)?;
        }
        drop(child);
        record.state = LocalJobState::Running;
        record.driver_process_id = Some(process_id);
        record.updated_unix_millis = unix_millis();
        store.write_record(&mut record)?;
        Ok(record)
    }

    pub(super) async fn command(
        &self,
        command: SupervisorCommand,
        organization: runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization,
    ) -> NativeExecutionResult<SupervisorResponse> {
        Ok(match command {
            SupervisorCommand::Ping => SupervisorResponse::Pong {
                config_identity: self.config.identity()?,
            },
            SupervisorCommand::ExecutionContext { idempotency_key } => {
                SupervisorResponse::ExecutionContext {
                    context: self.execution_context(idempotency_key.as_deref()).await?,
                }
            }
            SupervisorCommand::Submit { submission } => {
                let (record, duplicate) = self
                    .submit_with_organization(*submission, organization)
                    .await?;
                SupervisorResponse::Submitted { record, duplicate }
            }
            SupervisorCommand::SubmitProgram { submission } => {
                let (record, duplicate) = self
                    .submit_program_with_organization(*submission, organization)
                    .await?;
                SupervisorResponse::Submitted { record, duplicate }
            }
            SupervisorCommand::SubmitGraph { submission } => {
                let (record, duplicate) = self
                    .submit_graph_with_organization(*submission, organization)
                    .await?;
                SupervisorResponse::Submitted { record, duplicate }
            }
            SupervisorCommand::List => SupervisorResponse::Jobs {
                records: self.list().await?,
            },
            SupervisorCommand::Show { job_id } => SupervisorResponse::Job {
                record: self.show(job_id).await?,
            },
            SupervisorCommand::Attach {
                job_id,
                stdout_offset,
                stderr_offset,
            } => SupervisorResponse::Attachment {
                attachment: Box::new(self.attach(job_id, stdout_offset, stderr_offset).await?),
            },
            SupervisorCommand::Cancel { job_id } => SupervisorResponse::Cancelled {
                record: self.cancel(job_id).await?,
            },
            SupervisorCommand::ConsolePage {
                handle,
                attempt_id,
                cursor,
                budget,
            } => SupervisorResponse::ConsolePage {
                attempt_id,
                page: self
                    .console_page(handle, attempt_id, cursor, budget)
                    .await?
                    .map(Box::new),
            },
            SupervisorCommand::RawOutputPage {
                handle,
                attempt_id,
                cursor,
            } => SupervisorResponse::RawOutputPage {
                attempt_id,
                page: self
                    .raw_output_page(handle, attempt_id, cursor)
                    .await?
                    .map(Box::new),
            },
            SupervisorCommand::RawOutputAttempt { handle, ordinal } => {
                SupervisorResponse::RawOutputAttempt {
                    lookup: self.raw_output_attempt(handle, ordinal).await?,
                }
            }
            SupervisorCommand::CancelHandle { handle, reason } => SupervisorResponse::Cancelled {
                record: self.cancel_handle(handle, reason).await?,
            },
        })
    }
}

fn mark_indeterminate(
    store: &JobStore,
    record: &mut LocalJobRecord,
    now: u64,
    message: &str,
) -> NativeExecutionResult<()> {
    record.state = LocalJobState::Indeterminate;
    record.driver_process_id = None;
    record.updated_unix_millis = now;
    record.message = Some(message.into());
    store.write_record(record)
}
