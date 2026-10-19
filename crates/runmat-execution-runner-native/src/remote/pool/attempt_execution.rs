//! Adapt the original remote provider report and preserve consumption across delivery failure.
use super::{FailedRoute, RemotePoolDriver, WorkerRouteIdentity};
use crate::remote::{pool_progress, RemoteAttempt, RemoteAttemptProgram, RemoteWorkerChannel};
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution_artifact::WorkerProgramRequest;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{AttemptFailureKind, AttemptReport, AttemptRequest};
use std::sync::Arc;

impl RemotePoolDriver {
    pub(super) fn launch(self: &Arc<Self>, request: AttemptRequest) {
        let this = Arc::clone(self);
        let cancellation = CancellationSource::new();
        self.attempt_cancellation
            .lock()
            .expect("remote attempt cancellation registry poisoned")
            .insert(request.id, cancellation.clone());
        tokio::spawn(async move {
            let route = this.channels.read().await.get(&request.worker_id).cloned();
            let program = this
                .programs
                .lock()
                .expect("remote program catalog poisoned")
                .get(&request.task_id)
                .map(Arc::clone);
            let progress = this
                .progress
                .lock()
                .expect("remote progress registry poisoned")
                .get(&request.task_id)
                .cloned();
            let mut pending_results = None;
            let mut unsuccessful_resources = None;
            let (report, failed_route) = match (route, program) {
                (Some(route), Some(program)) => {
                    let route_identity = route.identity;
                    let channel = route.channel;
                    // Keep the admitted Graph in the scheduler catalog. A retry
                    // stages its signed chunks and sends only the worker-issued
                    // digest/length in the attempt frame.
                    let collective_context = match program.as_ref() {
                        WorkerProgramRequest::Direct(direct) => match &direct.context {
                            runmat_execution::ProgramInvocationContext::SpmdTask { task } => {
                                Some(pool_progress::RemoteCollectiveAssignment::new(
                                    Arc::clone(&this.collectives),
                                    task.clone(),
                                ))
                            }
                            _ => None,
                        },
                        WorkerProgramRequest::Graph(_) => None,
                    };
                    let transfer = this
                        .attempt_program(
                            channel.as_ref(),
                            route_identity,
                            &request,
                            program.as_ref(),
                            &cancellation,
                        )
                        .await;
                    match transfer {
                        _ if cancellation.token().is_cancelled() => {
                            (AttemptReport::Cancelled, None)
                        }
                        Err(AttemptTransferError::Rejected(error)) => (
                            AttemptReport::Failed {
                                kind: AttemptFailureKind::Rejected,
                                message: error.to_string(),
                            },
                            None,
                        ),
                        Err(AttemptTransferError::Route(error)) => (
                            AttemptReport::Lost {
                                message: error.to_string(),
                            },
                            Some(FailedRoute::Observed(route_identity)),
                        ),
                        Ok(program) => {
                            match pool_progress::execute(
                                channel.as_ref(),
                                RemoteAttempt {
                                    scheduling: request.clone(),
                                    program,
                                },
                                progress.as_deref(),
                                collective_context,
                            )
                            .await
                            {
                                Ok(AttemptReport::Succeeded { result }) => {
                                    match this
                                        .execution_objects
                                        .receive_results(
                                            channel.as_ref(),
                                            result.result_objects(),
                                            &this.value_scope,
                                        )
                                        .await
                                    {
                                        Ok(lease) => {
                                            pending_results = Some(lease);
                                            (AttemptReport::Succeeded { result }, None)
                                        }
                                        Err(error) => {
                                            // Delivery failure does not erase the original
                                            // worker's validated terminal consumption.
                                            // Invalid provider claims remain unknown.
                                            if result.validate_against(&request).is_ok() {
                                                unsuccessful_resources =
                                                    result.resources().cloned().map(Box::new);
                                            }
                                            (
                                                AttemptReport::Lost {
                                                    message: error.to_string(),
                                                },
                                                Some(FailedRoute::Observed(route_identity)),
                                            )
                                        }
                                    }
                                }
                                Ok(report) => (report, None),
                                Err(error) => (
                                    AttemptReport::Lost {
                                        message: error.to_string(),
                                    },
                                    Some(FailedRoute::Observed(route_identity)),
                                ),
                            }
                        }
                    }
                }
                (None, _) => (
                    AttemptReport::Lost {
                        message: "remote worker route is unavailable".into(),
                    },
                    Some(FailedRoute::Missing),
                ),
                (Some(_), None) => (
                    AttemptReport::Failed {
                        kind: AttemptFailureKind::Infrastructure,
                        message: "remote exact program is unavailable".into(),
                    },
                    None,
                ),
            };
            this.attempt_cancellation
                .lock()
                .expect("remote attempt cancellation registry poisoned")
                .remove(&request.id);
            let mut report = BackendReport::for_request(&request, report);
            report.unsuccessful_resources = unsuccessful_resources;
            if let Some(failed_route) = failed_route {
                let identity = match failed_route {
                    FailedRoute::Observed(identity) => Some(identity),
                    FailedRoute::Missing => None,
                };
                if !matches!(this.fence_failed_route(identity, &report).await, Ok(true)) {
                    this.apply_report(report);
                }
            } else {
                this.apply_report(report);
            }
            drop(pending_results);
        });
    }

    /// Bind the admitted program to this exact attempt and move its inputs to
    /// the worker: Direct objects/values, or the signed Graph Program stage.
    async fn attempt_program(
        &self,
        channel: &dyn RemoteWorkerChannel,
        route_identity: WorkerRouteIdentity,
        request: &AttemptRequest,
        program: &WorkerProgramRequest,
        cancellation: &CancellationSource,
    ) -> Result<RemoteAttemptProgram, AttemptTransferError> {
        match program {
            WorkerProgramRequest::Direct(direct) => {
                let mut direct = direct.clone();
                direct.arguments = request.task.inputs.clone();
                direct.assignment = Some(runmat_execution::ProgramExecutionAssignment {
                    scope_id: request.scope_id,
                    pool_id: request.task.pool_id,
                    task_id: request.task_id,
                    attempt_id: request.id,
                    worker_id: request.worker_id,
                    backend: runmat_execution::PoolBackend::Remote,
                    resources: request.resource_assignment.clone(),
                });
                let inventory = self
                    .input_inventories
                    .lock()
                    .expect("input inventory registry poisoned")
                    .get(request.task_id);
                self.execution_objects
                    .transfer(channel, route_identity, inventory.as_deref())
                    .await
                    .map_err(AttemptTransferError::Route)?;
                self.values
                    .transfer(
                        channel,
                        route_identity,
                        &direct.arguments,
                        &self.execution_objects,
                    )
                    .await
                    .map_err(AttemptTransferError::Route)?;
                Ok(RemoteAttemptProgram::Direct(Box::new(direct)))
            }
            WorkerProgramRequest::Graph(_) => {
                let source = self
                    .graph_transfers
                    .lock()
                    .expect("remote Graph transfer registry poisoned")
                    .get(&request.task_id)
                    .cloned()
                    .ok_or_else(|| {
                        AttemptTransferError::Rejected(NativeExecutionError::Protocol(
                            "remote Graph task lost its staged transfer authority".into(),
                        ))
                    })?;
                let token = cancellation.token();
                let receipt: NativeExecutionResult<_> = tokio::select! {
                    result = channel.stage_graph_program(
                        request.id,
                        &source.encrypted_manifest,
                        &source.ordered_downloads,
                        source.maximum_program_bytes,
                    ) => result,
                    _ = token.cancelled() => Err(NativeExecutionError::Protocol(
                        "remote Graph attempt was cancelled during staging".into(),
                    )),
                };
                let receipt = receipt.map_err(AttemptTransferError::Rejected)?;
                if receipt.program_digest != source.program_digest
                    || receipt.program_length != source.program_length
                {
                    return Err(AttemptTransferError::Rejected(
                        NativeExecutionError::Protocol(
                            "remote worker acknowledged a different Graph Program".into(),
                        ),
                    ));
                }
                Ok(RemoteAttemptProgram::GraphStaged(
                    crate::remote::RemoteStagedGraphReference::StagedGraphV10 {
                        program_digest: receipt.program_digest,
                        program_length: receipt.program_length,
                    },
                ))
            }
        }
    }
}

/// Object/value transport failure fences the observed route; a refused
/// Graph stage is a rejected attempt on a still-live route.
enum AttemptTransferError {
    Route(NativeExecutionError),
    Rejected(NativeExecutionError),
}
