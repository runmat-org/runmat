//! Adapt the original remote provider report and preserve consumption across delivery failure.
use super::{FailedRoute, RemotePoolDriver};
use crate::remote::{pool_progress, RemoteAttempt};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{AttemptFailureKind, AttemptReport, AttemptRequest};
use std::sync::Arc;

impl RemotePoolDriver {
    pub(super) fn launch(self: &Arc<Self>, request: AttemptRequest) {
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let route = this.channels.read().await.get(&request.worker_id).cloned();
            let program = this
                .programs
                .lock()
                .expect("remote program catalog poisoned")
                .get(&request.task_id)
                .cloned();
            let progress = this
                .progress
                .lock()
                .expect("remote progress registry poisoned")
                .get(&request.task_id)
                .cloned();
            let mut pending_results = None;
            let mut unsuccessful_resources = None;
            let (report, failed_route) = match (route, program) {
                (Some(route), Some(mut program)) => {
                    let route_identity = route.identity;
                    let channel = route.channel;
                    program.arguments = request.task.inputs.clone();
                    let inventory = this
                        .input_inventories
                        .lock()
                        .expect("input inventory registry poisoned")
                        .get(request.task_id);
                    let transfer = this
                        .execution_objects
                        .transfer(channel.as_ref(), route_identity, inventory.as_deref())
                        .await;
                    let transfer = match transfer {
                        Ok(()) => {
                            this.values
                                .transfer(
                                    channel.as_ref(),
                                    route_identity,
                                    &program.arguments,
                                    &this.execution_objects,
                                )
                                .await
                        }
                        Err(error) => Err(error),
                    };
                    match transfer {
                        Err(error) => (
                            AttemptReport::Lost {
                                message: error.to_string(),
                            },
                            Some(FailedRoute::Observed(route_identity)),
                        ),
                        Ok(()) => {
                            match pool_progress::execute(
                                channel.as_ref(),
                                RemoteAttempt {
                                    scheduling: request.clone(),
                                    program,
                                },
                                progress.as_deref(),
                            )
                            .await
                            {
                                Ok(AttemptReport::Succeeded { result }) => {
                                    match this
                                        .execution_objects
                                        .receive_results(
                                            channel.as_ref(),
                                            &result.result_objects,
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
                                                unsuccessful_resources = result.resources.clone();
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
}
