//! Local asynchronous admission at the actual language meshing/solve roots.
use super::{
    FeaRunBudget, FeaRunDenied, FeaRunExecutionError, FeaRunKind, FeaRunOutcome, FeaRunRequest,
    FeaRunScope, FeaRunUsage,
};
use crate::context::RuntimeServiceFuture;
use std::future::Future;

/// Session-local host transport. It may hold local UI callbacks; unlike the
/// synchronous native provider, neither this port nor its lease requires Send.
pub trait AsyncFeaRunAdmission {
    /// Acquire absolute cumulative capacity for this already-admitted root or
    /// authenticated numerical child. The host must preserve its fixed cap.
    fn request_grant(
        &self,
        request: FeaRunRequest,
        usage: FeaRunUsage,
        required: runmat_simulation_usage::OperationCount,
    ) -> RuntimeServiceFuture<Result<FeaRunBudget, FeaRunDenied>>;
    fn admit(
        &self,
        request: FeaRunRequest,
    ) -> RuntimeServiceFuture<Result<Box<dyn AsyncFeaRunLease>, FeaRunDenied>>;
}

/// Reserve before polling numerical work. A host wait retries only admission;
/// it never replays a numerical effect. The caller retains the ledger owner
/// while borrowing this reservation across the operation.
pub async fn reserve_work<'a>(
    usage: &'a runmat_simulation_usage::UsageContext,
    category: runmat_simulation_usage::UsageCategory,
    maximum: runmat_simulation_usage::OperationCount,
    runtime: Option<&crate::context::RuntimeContext>,
) -> Result<runmat_simulation_usage::WorkReservation<'a>, FeaRunDenied> {
    use runmat_simulation_usage::LedgerError;
    loop {
        FeaRunScope::check_current()?;
        let (required, granted) = match usage.ledger().reserve(category, maximum) {
            Ok(reservation) => return Ok(reservation),
            Err(LedgerError::MoreGrantRequired { required, granted }) => (required, granted),
            Err(cause) => return Err(cause.into()),
        };
        let unavailable = || LedgerError::MoreGrantRequired { required, granted };
        let state = super::scope::current()
            .filter(|state| std::ptr::eq(state.accounting.ledger(), usage.ledger()))
            .ok_or_else(|| FeaRunDenied::from(unavailable()))?;
        state.check()?;
        let host = match runtime {
            Some(runtime) => runtime.service_ports().fea_run_admission().cloned(),
            None => crate::context::legacy::active()
                .and_then(|runtime| runtime.service_ports().fea_run_admission().cloned()),
        }
        .ok_or_else(|| FeaRunDenied::from(unavailable()))?;
        let budget = host
            .request_grant(state.request.clone(), state.usage()?, required)
            .await
            .inspect_err(|denied| {
                state.deny(denied.clone());
            })?;
        state.renew_bound_budget(budget)?;
        if budget.granted_ops < required {
            return Err(LedgerError::MoreGrantRequired {
                required,
                granted: budget.granted_ops,
            }
            .into());
        }
        // Another parallel reservation may have consumed capacity while the
        // host was awaited. Re-evaluate the same reservation, not the work.
    }
}

/// Grow an exclusive numerical child's available grant before a synchronous kernel block.
/// The block's primitive reservations remain its only completed-work owners. This does not
/// reserve or charge work, and therefore must not be used on a shared root ledger.
pub async fn ensure_available_capacity(
    usage: &runmat_simulation_usage::UsageContext,
    additional: runmat_simulation_usage::OperationCount,
    runtime: Option<&crate::context::RuntimeContext>,
) -> Result<(), FeaRunDenied> {
    use runmat_simulation_usage::{LedgerError, OperationCount};
    loop {
        FeaRunScope::check_current()?;
        let state = super::scope::current()
            .filter(|state| std::ptr::eq(state.accounting.ledger(), usage.ledger()))
            .ok_or(FeaRunDenied::LicenseInvalid)?;
        if !state.is_parent_supervised_child() {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        state.check()?;
        let snapshot = usage.ledger().snapshot()?;
        // A child assignment owns one numerical phase. Outstanding primitive reservations
        // imply overlapping work, which this grant-only path cannot safely pre-admit.
        if snapshot.reserved != OperationCount::ZERO {
            return Err(LedgerError::OutstandingWork.into());
        }
        let required = snapshot
            .completed
            .total()
            .map_err(LedgerError::from)?
            .checked_add(additional)
            .map_err(LedgerError::from)?;
        if required > snapshot.budget {
            return Err(LedgerError::BudgetExceeded {
                required,
                budget: snapshot.budget,
            }
            .into());
        }
        if required <= snapshot.granted {
            return Ok(());
        }
        let host = match runtime {
            Some(runtime) => runtime.service_ports().fea_run_admission().cloned(),
            None => crate::context::legacy::active()
                .and_then(|runtime| runtime.service_ports().fea_run_admission().cloned()),
        }
        .ok_or_else(|| {
            FeaRunDenied::from(LedgerError::MoreGrantRequired {
                required,
                granted: snapshot.granted,
            })
        })?;
        let budget = host
            .request_grant(state.request.clone(), state.usage()?, required)
            .await
            .inspect_err(|denied| {
                state.deny(denied.clone());
            })?;
        state.renew_bound_budget(budget)?;
        if budget.granted_ops < required {
            return Err(LedgerError::MoreGrantRequired {
                required,
                granted: budget.granted_ops,
            }
            .into());
        }
        // Check the actual ledger again after the host wait; no work has executed yet.
    }
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
