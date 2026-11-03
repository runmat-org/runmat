//! Local asynchronous admission at the actual language meshing/solve roots.
use super::{
    FeaRunDenied, FeaRunExecutionError, FeaRunKind, FeaRunOutcome, FeaRunRequest, FeaRunScope,
};
use crate::context::RuntimeServiceFuture;
use std::future::Future;

/// Session-local host transport. It may hold local UI callbacks; unlike the
/// synchronous native provider, neither this port nor its lease requires Send.
pub trait AsyncFeaRunAdmission {
    fn admit(
        &self,
        request: FeaRunRequest,
    ) -> RuntimeServiceFuture<Result<Box<dyn AsyncFeaRunLease>, FeaRunDenied>>;
}
pub trait AsyncFeaRunLease {
    fn scope(&self) -> &FeaRunScope;
    fn settle(
        self: Box<Self>,
        outcome: FeaRunOutcome,
    ) -> RuntimeServiceFuture<Result<(), FeaRunDenied>>;
}

/// No host and already-nested calls preserve the existing native gate. A
/// browser port awaits one lease before polling the first domain effect.
pub(crate) async fn run<T, E>(
    kind: FeaRunKind,
    identity: runmat_execution::Digest,
    work: impl Future<Output = Result<T, E>>,
    failure: impl Fn(FeaRunExecutionError) -> E,
    merge: impl Fn(Result<T, E>, E) -> Result<T, E>,
) -> Result<T, E> {
    let host = crate::context::legacy::active()
        .and_then(|runtime| runtime.service_ports().fea_run_admission().cloned());
    let Some(host) = host.filter(|_| FeaRunScope::current_allocation().is_none()) else {
        return work.await;
    };
    let allocation = (runmat_cpu_execution::current_threads() as u64).saturating_mul(1000);
    let request = FeaRunRequest::local(kind, allocation, identity)
        .map_err(|denied| failure(FeaRunExecutionError::Denied(denied)))?;
    let lease = host
        .admit(request)
        .await
        .map_err(|denied| failure(FeaRunExecutionError::Denied(denied)))?;
    let checked = lease.scope().run_checked_result_future(work).await;
    let outcome = match &checked {
        Err(failure) => failure.outcome(),
        Ok(_) if crate::interrupt::is_cancelled() => FeaRunOutcome::Cancelled,
        Ok(Ok(_)) => FeaRunOutcome::Completed,
        Ok(Err(_)) => FeaRunOutcome::Failed,
    };
    let result = checked.map_err(&failure).and_then(|result| result);
    match lease.settle(outcome).await {
        Ok(()) => result,
        Err(denied) => merge(result, failure(FeaRunExecutionError::Denied(denied))),
    }
}

#[cfg(test)]
#[path = "async_host_tests.rs"]
mod tests;
