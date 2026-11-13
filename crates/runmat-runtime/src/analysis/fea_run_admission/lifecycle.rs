use super::{
    scope, FeaRunAdmission, FeaRunBudget, FeaRunDenied, FeaRunOutcome, FeaRunPermit, FeaRunRequest,
    FeaRunUsage, ScopedFeaRunFuture,
};
use runmat_execution::cancellation::{CancellationSource, CancellationToken};
use runmat_execution::resource::ResourceUsage;
use runmat_simulation_usage::{OperationCount, UsageContext, UsageLedger};
use std::sync::{Arc, Mutex};
mod accounting;

#[cfg(not(target_arch = "wasm32"))]
#[path = "heartbeat.rs"]
mod heartbeat;

#[derive(Clone, Copy)]
enum LeaseAuthority {
    Metered(FeaRunBudget),
    ParentSupervised(FeaRunBudget),
}
#[derive(Clone, Copy, Eq, PartialEq)]
enum LeaseRole {
    Metered,
    ParentSupervised,
}
impl LeaseAuthority {
    fn role(self) -> LeaseRole {
        match self {
            Self::Metered(_) => LeaseRole::Metered,
            Self::ParentSupervised(_) => LeaseRole::ParentSupervised,
        }
    }
    fn budget(self) -> FeaRunBudget {
        match self {
            Self::Metered(budget) | Self::ParentSupervised(budget) => budget,
        }
    }
}
struct LeaseState {
    authority: LeaseAuthority,
    origin: Option<super::FeaRunOrigin>,
    allocation_ceiling_millicores: u64,
    denied: Option<FeaRunDenied>,
    settled: bool,
}
pub(super) struct RunState {
    pub(super) request: FeaRunRequest,
    started: runmat_time::Instant,
    elapsed_wall_millis: u64,
    permit: Mutex<Option<Box<dyn FeaRunPermit>>>,
    lease: Mutex<LeaseState>,
    cancellation: CancellationSource,
    inherited: Option<CancellationToken>,
    pub(super) accounting: UsageContext,
    #[cfg(not(target_arch = "wasm32"))]
    heartbeat: heartbeat::Heartbeat,
}
impl RunState {
    pub(super) fn is_parent_supervised_child(&self) -> bool {
        matches!(
            self.lease
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .authority,
            LeaseAuthority::ParentSupervised(_)
        )
    }
    fn allocation(&self) -> u64 {
        self.lease
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .allocation_ceiling_millicores
    }
    fn check_publication(&self) -> Result<(), FeaRunDenied> {
        let supervised = matches!(
            self.lease
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .authority,
            LeaseAuthority::ParentSupervised(_)
        );
        if supervised {
            self.deny(FeaRunDenied::LicenseInvalid);
            return Err(FeaRunDenied::LicenseInvalid);
        }
        self.check()
    }
    fn resource_usage(&self) -> ResourceUsage {
        let elapsed = self.started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        allocated_usage(
            self.request.allocated_cpu_millicores,
            self.elapsed_wall_millis.saturating_add(elapsed),
        )
    }
    pub(super) fn usage(&self) -> Result<FeaRunUsage, FeaRunDenied> {
        Ok(FeaRunUsage {
            completed_ops: self.accounting.ledger().snapshot()?.completed,
            resources: self.resource_usage(),
        })
    }
    pub(super) fn deny(&self, denied: FeaRunDenied) {
        let mut lease = self.lease.lock().unwrap_or_else(|e| e.into_inner());
        if lease.denied.is_none() {
            lease.denied = Some(denied);
        }
        drop(lease);
        self.cancellation
            .cancel(runmat_execution::CancellationReason::Deadline);
    }
    pub(super) fn check(&self) -> Result<(), FeaRunDenied> {
        if let Err(cause) = self.accounting.ledger().snapshot() {
            let denied = FeaRunDenied::from(cause);
            self.deny(denied.clone());
            return Err(denied);
        }
        let mut lease = self.lease.lock().unwrap_or_else(|e| e.into_inner());
        let denied = lease.denied.clone().or_else(|| {
            if lease.settled {
                Some(FeaRunDenied::LeaseExpired)
            } else {
                (runmat_time::unix_timestamp_ms()
                    >= u128::from(lease.authority.budget().expires_at_unix_millis))
                .then_some(FeaRunDenied::LeaseExpired)
            }
        });
        if let Some(denied) = denied {
            lease.denied = Some(denied.clone());
            drop(lease);
            self.cancellation
                .cancel(runmat_execution::CancellationReason::Deadline);
            return Err(denied);
        }
        Ok(())
    }
    fn renew(&self, budget: FeaRunBudget) -> Result<(), FeaRunDenied> {
        self.renew_for_role(budget, LeaseRole::Metered)
    }
    pub(super) fn renew_bound_budget(&self, budget: FeaRunBudget) -> Result<(), FeaRunDenied> {
        let role = self
            .lease
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .authority
            .role();
        self.renew_for_role(budget, role)
    }
    fn renew_for_role(&self, budget: FeaRunBudget, role: LeaseRole) -> Result<(), FeaRunDenied> {
        // Renewal cannot revive an expired lease or change its spending cap.
        self.check()?;
        if let Err(denied) = budget.validate() {
            self.deny(denied.clone());
            return Err(denied);
        }
        let mut lease = self.lease.lock().unwrap_or_else(|e| e.into_inner());
        if lease.authority.role() != role {
            drop(lease);
            self.deny(FeaRunDenied::LicenseInvalid);
            return Err(FeaRunDenied::LicenseInvalid);
        }
        let previous = lease.authority.budget();
        if budget.budget_ops != previous.budget_ops
            || budget.granted_ops < previous.granted_ops
            || budget.expires_at_unix_millis < previous.expires_at_unix_millis
            || budget.heartbeat_millis != previous.heartbeat_millis
        {
            drop(lease);
            self.deny(FeaRunDenied::LicenseInvalid);
            return Err(FeaRunDenied::LicenseInvalid);
        }
        if let Err(cause) = self.accounting.ledger().update_grant(budget.granted_ops) {
            drop(lease);
            let denied = FeaRunDenied::from(cause);
            self.deny(denied.clone());
            return Err(denied);
        }
        lease.authority = match role {
            LeaseRole::Metered => LeaseAuthority::Metered(budget),
            LeaseRole::ParentSupervised => LeaseAuthority::ParentSupervised(budget),
        };
        drop(lease);
        self.check()
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn heartbeat_millis(&self) -> u64 {
        match self
            .lease
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .authority
        {
            LeaseAuthority::Metered(budget) => budget.heartbeat_millis,
            LeaseAuthority::ParentSupervised(budget) => budget.heartbeat_millis,
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn deadline_wait(&self) -> std::time::Duration {
        let LeaseAuthority::Metered(budget) = self
            .lease
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .authority
        else {
            return std::time::Duration::ZERO;
        };
        let expiry_wait = u128::from(budget.expires_at_unix_millis)
            .saturating_sub(runmat_time::unix_timestamp_ms());
        std::time::Duration::from_millis(expiry_wait.min(u128::from(u64::MAX)).max(1) as u64)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn report(&self) {
        if self.check().is_err() {
            return;
        }
        let renewed = (|| {
            // Serialize both the cumulative snapshot and installing the reply.
            // An earlier report must not overtake a pre-batch grant request.
            let permit = self.permit.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(permit) = permit.as_ref() {
                let budget = permit.heartbeat(&self.usage()?)?;
                self.renew(budget)?;
            }
            Ok::<_, FeaRunDenied>(())
        })();
        if let Err(denied) = renewed {
            self.deny(denied);
        }
    }
    pub(super) fn token(&self) -> CancellationToken {
        let own = self.cancellation.token();
        self.inherited
            .as_ref()
            .map_or(own.clone(), |parent| parent.or(&own))
    }
    fn settle(&self, outcome: FeaRunOutcome) -> Result<(), FeaRunDenied> {
        // End only this lease's descendant generation before waiting for any
        // in-flight host transport or handing final usage to its owner.
        self.cancellation
            .cancel(runmat_execution::CancellationReason::Shutdown);
        #[cfg(not(target_arch = "wasm32"))]
        self.heartbeat.stop();
        let checked = self.check();
        // Outstanding numerical/child owners must retire before final usage
        // can be certified. A failed close retains the original permit/grant.
        let completed_ops = self.accounting.ledger().finish()?;
        let outcome = if checked.is_err() && outcome == FeaRunOutcome::Completed {
            FeaRunOutcome::Failed
        } else {
            outcome
        };
        let permit = self.permit.lock().unwrap_or_else(|e| e.into_inner()).take();
        self.lease.lock().unwrap_or_else(|e| e.into_inner()).settled = true;
        let usage = FeaRunUsage {
            completed_ops,
            resources: self.resource_usage(),
        };
        let settled = permit.map_or(Ok(()), |permit| permit.settle(outcome, &usage));
        checked.and(settled)
    }
}
fn allocated_usage(millicores: u64, wall_millis: u64) -> ResourceUsage {
    let core_millis = (u128::from(wall_millis) * u128::from(millicores)).div_ceil(1000);
    ResourceUsage {
        wall_millis,
        cpu_core_millis: core_millis.min(u128::from(u64::MAX)) as u64,
        ..Default::default()
    }
}
impl Drop for RunState {
    fn drop(&mut self) {
        self.cancellation
            .cancel(runmat_execution::CancellationReason::Shutdown);
        // Only the root state owns a permit. Consumption settlement is exactly once.
        if self
            .permit
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
        {
            let _ = self.settle(FeaRunOutcome::Failed);
        } else {
            #[cfg(not(target_arch = "wasm32"))]
            self.heartbeat.stop();
        }
    }
}

/// One admitted root; nested operations share its lease and numerical ledger.
pub struct FeaRunScope {
    state: Arc<RunState>,
    ownership: ScopeOwnership,
}
#[derive(Clone, Copy)]
enum ScopeOwnership {
    Ledger,
    Borrowed,
}
#[derive(Clone)]
pub struct FeaRunControl {
    state: Arc<RunState>,
}
impl FeaRunControl {
    /// Quote cumulative capacity for an exclusive synchronous phase. Its
    /// primitive reservations remain the only completed-work owners.
    pub fn exclusive_work_requirement(
        &self,
        additional: OperationCount,
    ) -> Result<OperationCount, FeaRunDenied> {
        self.state.check()?;
        self.state
            .accounting
            .ledger()
            .exclusive_work_requirement(additional)
            .map_err(Into::into)
    }
    /// Reserve disjoint child capacity under this root outside its scoped poll.
    /// The host retains the returned owner until authenticated child retirement.
    pub fn delegate(
        &self,
        maximum: OperationCount,
    ) -> Result<super::FeaRunDelegatedGrant, FeaRunDenied> {
        super::FeaRunDelegationSource::delegate(self.state.as_ref(), maximum)
    }
    /// Completed counts from a closed numerical owner. A live or outstanding
    /// reservation cannot be certified as final usage by a host adapter.
    pub fn settled_usage(&self) -> Result<FeaRunUsage, FeaRunDenied> {
        if !self
            .state
            .lease
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .settled
        {
            return Err(runmat_simulation_usage::LedgerError::OutstandingWork.into());
        }
        self.state.usage()
    }
    /// Acquire cumulative capacity before reserving the next numerical batch.
    /// The required total includes completed work and all parallel reservations.
    pub fn request_grant(&self, required: OperationCount) -> Result<(), FeaRunDenied> {
        self.state.request_grant(required)
    }
    pub fn renew(&self, budget: FeaRunBudget) -> Result<(), FeaRunDenied> {
        self.state.renew(budget)
    }
    pub fn deny(&self, denied: FeaRunDenied) {
        self.state.deny(denied);
    }
    pub fn check(&self) -> Result<(), FeaRunDenied> {
        self.state.check()
    }
    pub fn usage(&self) -> Result<FeaRunUsage, FeaRunDenied> {
        self.state.usage()
    }
}
impl FeaRunScope {
    /// Preserve an already-returned domain failure if a lease loss races the
    /// final poll. The owner still settles and attaches its typed lease cause.
    pub async fn run_checked_result_future<T, E, F: std::future::Future<Output = Result<T, E>>>(
        &self,
        future: F,
    ) -> Result<Result<T, E>, super::FeaRunExecutionError> {
        let mut completed = None;
        let execution = self
            .run_checked_future(async {
                completed = Some(future.await);
            })
            .await;
        match execution {
            Ok(()) => Ok(completed.expect("completed future retained its result")),
            Err(failure) => match completed {
                Some(Err(original)) => Ok(Err(original)),
                _ => Err(failure),
            },
        }
    }
    pub fn current_delegation_source(
    ) -> Result<Option<Arc<dyn super::FeaRunDelegationSource>>, FeaRunDenied> {
        if let Some(state) = scope::current() {
            return Ok(Some(state));
        }
        match super::provider::current() {
            Ok(provider) => Ok(provider.delegation_source()),
            Err(FeaRunDenied::SignInRequired) => Ok(None),
            Err(error) => Err(error),
        }
    }
    /// Install only after authenticating the private parent and exact child
    /// assignment; this data is never public JSON authorization.
    pub fn from_private_worker_delegation(
        grant: super::FeaRunDelegation,
        child_cpu_millicores: u64,
    ) -> Result<Self, FeaRunDenied> {
        grant.validate()?;
        if child_cpu_millicores == 0 || child_cpu_millicores > grant.allocation_ceiling_millicores {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        let mut request = grant.request;
        request.allocated_cpu_millicores = child_cpu_millicores;
        // The inherited absolute expiry keeps the parent's deadline. Physical
        // telemetry belongs to this child allocation, not the parent's earlier
        // elapsed interval or CPU count. Work is charged through its ledger.
        let scope = Self::start(
            request,
            LeaseAuthority::Metered(grant.budget),
            None,
            0,
            false,
        )?;
        {
            let mut lease = scope.state.lease.lock().unwrap_or_else(|e| e.into_inner());
            lease.origin = Some(grant.origin);
            lease.allocation_ceiling_millicores = child_cpu_millicores;
        }
        scope.check()?;
        Ok(scope)
    }
    /// Bind the already authenticated child's grant to its private parent
    /// transport. The child reports usage but owns no second Server lease.
    pub fn from_private_worker_permit(
        grant: super::FeaRunDelegation,
        child_cpu_millicores: u64,
        permit: Box<dyn FeaRunPermit>,
    ) -> Result<Self, FeaRunDenied> {
        if permit.budget() != grant.budget {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        let scope = Self::from_private_worker_delegation(grant, child_cpu_millicores)?;
        *scope
            .state
            .permit
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(permit);
        Ok(scope)
    }
    pub fn run_checked_sync<T>(&self, operation: impl FnOnce() -> T) -> Result<T, FeaRunDenied> {
        self.check()?;
        let value = self.run_sync(operation);
        self.check()?;
        Ok(value)
    }
    pub fn run_checked_future<F: std::future::Future>(
        &self,
        future: F,
    ) -> super::CheckedFeaRunFuture<F> {
        super::CheckedFeaRunFuture {
            state: Arc::clone(&self.state),
            future: Box::pin(future),
            cancelled: self.state.token().cancelled(),
        }
    }
    pub fn control(&self) -> FeaRunControl {
        FeaRunControl {
            state: Arc::clone(&self.state),
        }
    }
    pub fn admit(request: FeaRunRequest) -> Result<Self, FeaRunDenied> {
        if let Some(scope) = Self::nested(&request)? {
            return Ok(scope);
        }
        let admission = super::provider::current()?;
        Self::admit_with(request, admission.as_ref())
    }
    pub fn admit_with(
        request: FeaRunRequest,
        admission: &dyn FeaRunAdmission,
    ) -> Result<Self, FeaRunDenied> {
        if let Some(scope) = Self::nested(&request)? {
            return Ok(scope);
        }
        let origin = admission.origin().or_else(|| {
            (request.site == super::FeaRunSite::Local).then(|| super::FeaRunOrigin::LocalLease {
                run_id: request.run_id.clone(),
            })
        });
        if let Some(origin) = &origin {
            origin.validate(request.site)?;
        }
        let permit = admission.admit(&request)?;
        let scope = Self::from_admitted_permit(request, permit)?;
        scope
            .state
            .lease
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .origin = origin;
        Ok(scope)
    }
    /// The host has already completed authenticated admission; no transport occurs here.
    pub fn from_admitted_permit(
        request: FeaRunRequest,
        permit: Box<dyn FeaRunPermit>,
    ) -> Result<Self, FeaRunDenied> {
        request.validate()?;
        let budget = permit.budget();
        let scope = Self::start(
            request,
            LeaseAuthority::Metered(budget),
            Some(permit),
            0,
            true,
        )?;
        Ok(scope)
    }
    /// Trusted owned-worker boundary only. The caller validates the exact original
    /// lease, request, payload and elapsed root clock before delegating this budget.
    /// This child never starts, heartbeats or settles a second server lease.
    pub fn from_delegated_budget(
        request: FeaRunRequest,
        budget: FeaRunBudget,
        elapsed_wall_millis: u64,
    ) -> Result<Self, FeaRunDenied> {
        request.validate()?;
        Self::start(
            request,
            LeaseAuthority::Metered(budget),
            None,
            elapsed_wall_millis,
            false,
        )
    }
    /// Trusted WASM numerical-worker boundary only, after the installed parent
    /// validates the original lease/request/payload. The parent owns live quota,
    /// cancellation and physical worker termination. This role cannot publish,
    /// renew or settle a Server lease and must be revalidated before returning
    /// bytes. Its authenticated parent may renew only the child numerical grant.
    pub fn from_parent_supervised_delegation(
        request: FeaRunRequest,
        budget: FeaRunBudget,
        elapsed_wall_millis: u64,
    ) -> Result<Self, FeaRunDenied> {
        request.validate()?;
        Self::start(
            request,
            LeaseAuthority::ParentSupervised(budget),
            None,
            elapsed_wall_millis,
            false,
        )
    }
    fn start(
        request: FeaRunRequest,
        authority: LeaseAuthority,
        permit: Option<Box<dyn FeaRunPermit>>,
        elapsed_wall_millis: u64,
        report_usage: bool,
    ) -> Result<Self, FeaRunDenied> {
        // Browser reporting is driven by its async host, not the native heartbeat thread.
        #[cfg(target_arch = "wasm32")]
        let _ = report_usage;
        let origin = if request.site == super::FeaRunSite::Local {
            Some(super::FeaRunOrigin::LocalLease {
                run_id: request.run_id.clone(),
            })
        } else {
            super::provider::current()
                .ok()
                .and_then(|provider| provider.origin())
        };
        let allocation_ceiling_millicores = request.allocated_cpu_millicores;
        let budget = authority.budget();
        budget.validate()?;
        let accounting = UsageLedger::new(budget.budget_ops, budget.granted_ops)?;
        let state = Arc::new_cyclic(move |owner: &std::sync::Weak<RunState>| {
            #[cfg(not(target_arch = "wasm32"))]
            let accounting = accounting.with_grant_renewal(owner.clone());
            #[cfg(target_arch = "wasm32")]
            let _ = owner; // Browser grant acquisition belongs to its async host.
            RunState {
                request,
                started: runmat_time::Instant::now(),
                elapsed_wall_millis,
                permit: Mutex::new(permit),
                lease: Mutex::new(LeaseState {
                    authority,
                    origin,
                    allocation_ceiling_millicores,
                    denied: None,
                    settled: false,
                }),
                cancellation: CancellationSource::new(),
                inherited: crate::interrupt::current_interrupt(),
                accounting: UsageContext::new(Arc::new(accounting)),
                #[cfg(not(target_arch = "wasm32"))]
                heartbeat: heartbeat::Heartbeat::default(),
            }
        });
        state.check()?;
        #[cfg(not(target_arch = "wasm32"))]
        if matches!(authority, LeaseAuthority::Metered(_)) {
            state
                .heartbeat
                .start(Arc::downgrade(&state), report_usage)
                .map_err(|_| FeaRunDenied::ServerUnreachable)?;
        }
        Ok(Self {
            state,
            ownership: ScopeOwnership::Ledger,
        })
    }
    fn nested(request: &FeaRunRequest) -> Result<Option<Self>, FeaRunDenied> {
        request.validate()?;
        let Some(state) = scope::current() else {
            return Ok(None);
        };
        if request.site != state.request.site
            || request.allocated_cpu_millicores > state.allocation()
            || request.measurement_version != state.request.measurement_version
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        state.check()?;
        Ok(Some(Self {
            state,
            ownership: ScopeOwnership::Borrowed,
        }))
    }
    pub fn current_allocation() -> Option<u64> {
        scope::current().map(|s| s.allocation())
    }
    pub fn current_site() -> Option<super::FeaRunSite> {
        scope::current().map(|state| state.request.site)
    }
    pub fn admission_site() -> Result<super::FeaRunSite, FeaRunDenied> {
        Self::current_site().map_or_else(|| super::provider::current().map(|host| host.site()), Ok)
    }
    pub fn check_current() -> Result<(), FeaRunDenied> {
        scope::current().map_or(Ok(()), |state| state.check())
    }
    pub fn check_publication() -> Result<(), FeaRunDenied> {
        scope::current().map_or(Ok(()), |state| state.check_publication())
    }
    /// Publication must preserve user cancellation as its own typed cause.
    /// Lease denial takes precedence when this run's own hold was lost.
    pub fn check_publication_checked() -> Result<(), super::FeaRunExecutionError> {
        Self::check_publication().map_err(super::FeaRunExecutionError::Denied)?;
        let token = scope::current()
            .map(|state| state.token())
            .or_else(crate::interrupt::current_interrupt);
        if token.is_some_and(|token| token.is_cancelled()) {
            return Err(super::FeaRunExecutionError::Cancelled);
        }
        Ok(())
    }
    pub fn check_publication_for_operation(
        operation: &str,
        version: &str,
        context: &crate::operations::OperationContext,
    ) -> Result<(), crate::operations::OperationErrorEnvelope> {
        Self::check_publication_checked()
            .map_err(|failure| failure.into_operation_error(operation, version, context))
    }
    pub(super) fn check_execution(&self) -> Result<(), super::FeaRunExecutionError> {
        self.check().map_err(super::FeaRunExecutionError::Denied)?;
        if self.state.token().is_cancelled() {
            return Err(super::FeaRunExecutionError::Cancelled);
        }
        Ok(())
    }
    pub fn check(&self) -> Result<(), FeaRunDenied> {
        self.state.check()
    }
    pub fn usage(&self) -> Result<FeaRunUsage, FeaRunDenied> {
        self.state.usage()
    }
    pub fn renew(&self, budget: FeaRunBudget) -> Result<(), FeaRunDenied> {
        self.state.renew(budget)
    }
    /// Trusted parent transport only. The adapter must bind this reply to its
    /// retained child reservation before allowing more numerical execution.
    /// This cannot change the root cap or promote the child to publication owner.
    pub fn renew_parent_supervised(&self, budget: FeaRunBudget) -> Result<(), FeaRunDenied> {
        self.state
            .renew_for_role(budget, LeaseRole::ParentSupervised)
    }
    pub fn deny(&self, denied: FeaRunDenied) {
        self.state.deny(denied);
    }
    pub fn cancellation_token(&self) -> CancellationToken {
        self.state.token()
    }
    pub fn run_sync<T>(&self, operation: impl FnOnce() -> T) -> T {
        scope::run(Arc::clone(&self.state), operation)
    }
    pub fn run_future<F: std::future::Future>(&self, future: F) -> ScopedFeaRunFuture<F> {
        ScopedFeaRunFuture {
            state: Arc::clone(&self.state),
            future: Box::pin(future),
        }
    }
    pub fn settle(mut self, outcome: FeaRunOutcome) -> Result<(), FeaRunDenied> {
        match std::mem::replace(&mut self.ownership, ScopeOwnership::Borrowed) {
            ScopeOwnership::Ledger => self.state.settle(outcome),
            ScopeOwnership::Borrowed => self.state.check(),
        }
    }
}
impl Drop for FeaRunScope {
    fn drop(&mut self) {
        if matches!(self.ownership, ScopeOwnership::Ledger) {
            let _ = self.state.settle(FeaRunOutcome::Failed);
        }
    }
}
#[cfg(test)]
#[path = "tests.rs"]
mod tests;
