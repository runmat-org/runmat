//! Fence only the observed route generation and retain the triggering provider report.
use super::{RemotePoolDriver, WorkerRouteIdentity};
use crate::NativeExecutionResult;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::DriverCommand;
use std::sync::Arc;

impl RemotePoolDriver {
    pub(super) async fn fence_failed_route(
        self: &Arc<Self>,
        observed: Option<WorkerRouteIdentity>,
        report: &BackendReport,
    ) -> NativeExecutionResult<bool> {
        let mut channels = self.channels.write().await;
        let current = channels.get(&report.worker_id);
        let owns_failure = match observed {
            Some(identity) => current.is_some_and(|route| route.identity == identity),
            None => current.is_none(), // A missing old route cannot fence a replacement.
        };
        if !owns_failure {
            return Ok(false);
        }
        let old_channel = current.map(|route| Arc::clone(&route.channel));
        let actions = {
            let mut driver = self.driver.lock().expect("remote driver poisoned");
            if !driver.accepts_backend_report(report) {
                return Ok(false);
            }
            let actions = driver.handle(DriverCommand::WorkerLostWithReport {
                worker_id: report.worker_id,
                report: Box::new(report.clone()),
            })?;
            channels.remove(&report.worker_id);
            actions
        };
        drop(channels);
        self.dispatch(actions);
        if let Some(channel) = old_channel {
            let request = self
                .driver
                .lock()
                .expect("remote driver poisoned")
                .snapshot_ref()
                .attempts
                .get(&report.attempt_id)
                .map(|attempt| attempt.request.clone());
            if let Some(request) = request {
                self.reconcile_physical_retirements(channel.as_ref(), &[request])
                    .await;
            }
        }
        self.resolve_non_success_terminals();
        self.refresh_pool_state()?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests;
