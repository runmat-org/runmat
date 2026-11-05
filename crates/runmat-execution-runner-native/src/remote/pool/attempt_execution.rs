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
            let console_budget = match this.reserve_console_attempt(&request) {
                Ok(budget) => budget,
                Err(error) => {
                    this.attempt_cancellation
                        .lock()
                        .expect("remote attempt cancellation registry poisoned")
                        .remove(&request.id);
                    this.apply_report(BackendReport::for_request(
                        &request,
                        AttemptReport::Failed {
                            kind: AttemptFailureKind::Rejected,
                            message: error.to_string(),
                        },
                    ));
                    return;
                }
            };
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
            let mut console = None;
            let mut console_scopes = Vec::new();
            let mut delegated_console_root = None;
            let mut retirement_ack_route: Option<Arc<dyn RemoteWorkerChannel>> = None;
            let mut retirement_evidence: Option<crate::remote::AuthenticatedPhysicalRetirement> =
                None;
            let (mut report, failed_route) = match (route, program) {
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
                            let execution = pool_progress::execute(
                                channel.as_ref(),
                                RemoteAttempt {
                                    scheduling: request.clone(),
                                    program,
                                },
                                progress.as_deref(),
                                collective_context,
                                console_budget,
                                &this.fea_provider,
                            )
                            .await;
                            let execution = execution.and_then(|mut observed| {
                                if observed.attempt_id != request.id
                                    || observed.task_id != request.task_id
                                    || observed.worker_id != request.worker_id
                                    || observed.driver_fence != request.driver_fence
                                {
                                    return Err(NativeExecutionError::Protocol(
                                        "remote observed report differs from its original attempt"
                                            .into(),
                                    ));
                                }
                                console = observed.console.take();
                                console_scopes = std::mem::take(&mut observed.console_scopes);
                                delegated_console_root = observed.delegated_console_root;
                                Ok(observed.report)
                            });
                            let execution = match execution {
                                Ok(AttemptReport::RetirementUnverified { message }) => {
                                    Ok(AttemptReport::RetirementUnverified { message })
                                }
                                Ok(report) => {
                                    let mut certified = false;
                                    for _ in 0..100 {
                                        match channel
                                            .authenticated_physical_retirement(&request)
                                            .await
                                        {
                                            Ok(Some(evidence)) if evidence.matches(&request) => {
                                                certified = true;
                                                retirement_evidence = Some(evidence);
                                                break;
                                            }
                                            Ok(Some(_)) => break,
                                            Ok(None) => {
                                                tokio::time::sleep(
                                                    std::time::Duration::from_millis(10),
                                                )
                                                .await
                                            }
                                            Err(_) => break,
                                        }
                                    }
                                    if certified {
                                        retirement_ack_route = Some(Arc::clone(&channel));
                                        Ok(report)
                                    } else {
                                        Ok(AttemptReport::RetirementUnverified {
                                            message: "remote owner completion lacks exact authenticated physical retirement".into(),
                                        })
                                    }
                                }
                                Err(error) => Err(error),
                            };
                            match execution {
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
            // The worker must release its retained owner claim before the
            // scheduler dispatches replacement work on that same route.
            // The authenticated proof already established physical retirement;
            // a lost ACK remains uncertain at the driver.
            let cancellation_pending = this
                .driver
                .lock()
                .expect("remote driver poisoned")
                .snapshot_ref()
                .attempts
                .get(&request.id)
                .is_some_and(|attempt| attempt.cancellation_requested_at.is_some());
            if let Some(channel) = retirement_ack_route.as_ref() {
                if !cancellation_pending
                    && !matches!(&report, AttemptReport::Retired)
                    && !matches!(
                        &report,
                        AttemptReport::Lost { .. } | AttemptReport::RetirementUnverified { .. }
                    )
                {
                    if let Err(error) = channel.acknowledge_physical_retirement(&request).await {
                        report = AttemptReport::RetirementUnverified {
                            message: format!("remote retirement acknowledgement failed: {error}"),
                        };
                    }
                }
            }
            this.attempt_cancellation
                .lock()
                .expect("remote attempt cancellation registry poisoned")
                .remove(&request.id);
            let mut report = BackendReport::for_request(&request, report);
            report.console = console;
            report.console_scopes = console_scopes;
            report.delegated_console_root = delegated_console_root;
            report.unsuccessful_resources = unsuccessful_resources;
            let mut retired_from_evidence = false;
            if let Some(failed_route) = failed_route {
                let identity = match failed_route {
                    FailedRoute::Observed(identity) => Some(identity),
                    FailedRoute::Missing => None,
                };
                if !matches!(this.fence_failed_route(identity, &report).await, Ok(true)) {
                    if let Some(evidence) = retirement_evidence.take() {
                        retired_from_evidence = this
                            .apply_report_with_authenticated_evidence(report, &request, evidence);
                    } else {
                        this.apply_report(report);
                    }
                }
            } else if let Some(evidence) = retirement_evidence.take() {
                retired_from_evidence =
                    this.apply_report_with_authenticated_evidence(report, &request, evidence);
            } else {
                this.apply_report(report);
            }
            if retired_from_evidence {
                if let Some(channel) = retirement_ack_route.as_ref() {
                    let _ = channel.acknowledge_physical_retirement(&request).await;
                }
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
                Ok(RemoteAttemptProgram::Direct(direct))
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
