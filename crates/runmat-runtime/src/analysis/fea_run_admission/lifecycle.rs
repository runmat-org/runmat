use super::{
    scope, FeaRunAdmission, FeaRunBudget, FeaRunDenied, FeaRunOutcome, FeaRunPermit, FeaRunRequest,
    ScopedFeaRunFuture,
};
use runmat_execution::cancellation::{CancellationSource, CancellationToken};
use runmat_execution::resource::ResourceUsage;
use std::sync::{Arc, Mutex};

#[cfg(not(target_arch = "wasm32"))]
#[path = "heartbeat.rs"]
mod heartbeat;

#[derive(Clone, Copy)]
enum LeaseAuthority {
    Metered(FeaRunBudget),
    ParentSupervised,
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
    #[cfg(not(target_arch = "wasm32"))]
    heartbeat: heartbeat::Heartbeat,
}
impl RunState {
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
            LeaseAuthority::ParentSupervised
        );
        if supervised {
            self.deny(FeaRunDenied::LicenseInvalid);
            return Err(FeaRunDenied::LicenseInvalid);
        }
        self.check()
    }
    pub(super) fn usage(&self) -> ResourceUsage {
        let elapsed = self.started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        allocated_usage(
            self.request.allocated_cpu_millicores,
            self.elapsed_wall_millis.saturating_add(elapsed),
        )
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
        let usage = self.usage();
        let mut lease = self.lease.lock().unwrap_or_else(|e| e.into_inner());
        let denied = lease.denied.clone().or_else(|| {
            if lease.settled {
                Some(FeaRunDenied::LeaseExpired)
            } else {
                match lease.authority {
                    LeaseAuthority::Metered(budget)
                        if runmat_time::unix_timestamp_ms()
                            >= u128::from(budget.expires_at_unix_millis) =>
                    {
                        Some(FeaRunDenied::LeaseExpired)
                    }
                    LeaseAuthority::Metered(budget)
                        if usage.cpu_core_millis >= budget.held_core_millis =>
                    {
                        Some(FeaRunDenied::AllowanceUsed {
                            refills_at_unix_seconds: None,
                        })
                    }
                    _ => None,
                }
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
        // A renewal cannot revive an exhausted/expired original hold.
        self.check()?;
        if let Err(denied) = budget.validate() {
            self.deny(denied.clone());
            return Err(denied);
        }
        let mut lease = self.lease.lock().unwrap_or_else(|e| e.into_inner());
        let LeaseAuthority::Metered(previous) = lease.authority else {
            drop(lease);
            self.deny(FeaRunDenied::LicenseInvalid);
            return Err(FeaRunDenied::LicenseInvalid);
        };
        if budget.held_core_millis < previous.held_core_millis
            || budget.expires_at_unix_millis < previous.expires_at_unix_millis
            || budget.heartbeat_millis != previous.heartbeat_millis
        {
            drop(lease);
            self.deny(FeaRunDenied::LicenseInvalid);
            return Err(FeaRunDenied::LicenseInvalid);
        }
        lease.authority = LeaseAuthority::Metered(budget);
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
            LeaseAuthority::ParentSupervised => 30_000,
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
        let wall = self.usage().wall_millis;
        // First integral wall millisecond whose rounded CPU charge reaches the hold.
        let stop_wall = (u128::from(budget.held_core_millis.saturating_sub(1)) * 1000)
            / u128::from(self.request.allocated_cpu_millicores)
            + 1;
        let hold_wait = stop_wall.saturating_sub(u128::from(wall));
        let expiry_wait = u128::from(budget.expires_at_unix_millis)
            .saturating_sub(runmat_time::unix_timestamp_ms());
        std::time::Duration::from_millis(
            hold_wait.min(expiry_wait).min(u128::from(u64::MAX)).max(1) as u64,
        )
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn report(&self) {
        if self.check().is_err() {
            return;
        }
        let renewed = {
            let permit = self.permit.lock().unwrap_or_else(|e| e.into_inner());
            permit
                .as_ref()
                .map(|permit| permit.heartbeat(&self.usage()))
        };
        if let Some(result) = renewed {
            match result {
                Ok(budget) => {
                    let _ = self.renew(budget);
                }
                Err(denied) => self.deny(denied),
            }
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
        let outcome = if checked.is_err() && outcome == FeaRunOutcome::Completed {
            FeaRunOutcome::Failed
        } else {
            outcome
        };
        let permit = self.permit.lock().unwrap_or_else(|e| e.into_inner()).take();
        self.lease.lock().unwrap_or_else(|e| e.into_inner()).settled = true;
        let settled = permit.map_or(Ok(()), |permit| permit.settle(outcome, &self.usage()));
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
impl super::FeaRunDelegationSource for RunState {
    fn snapshot(&self) -> Result<super::FeaRunDelegation, FeaRunDenied> {
        self.check()?;
        let lease = self.lease.lock().unwrap_or_else(|e| e.into_inner());
        let LeaseAuthority::Metered(budget) = lease.authority else {
            return Err(FeaRunDenied::LicenseInvalid);
        };
        let grant = super::FeaRunDelegation {
            origin: lease.origin.clone().ok_or(FeaRunDenied::LicenseInvalid)?,
            request: self.request.clone(),
            budget,
            elapsed_wall_millis: self.usage().wall_millis,
            observed_at_unix_millis: runmat_time::unix_timestamp_ms().min(u128::from(u64::MAX))
                as u64,
            allocation_ceiling_millicores: lease.allocation_ceiling_millicores,
        };
        grant.validate()?;
        Ok(grant)
    }
    fn cancellation_token(&self) -> CancellationToken {
        self.token()
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

/// One admitted root; nested operations borrow its lease and cumulative CPU clock.
pub struct FeaRunScope {
    state: Arc<RunState>,
    owns_permit: bool,
}
#[derive(Clone)]
pub struct FeaRunControl {
    state: Arc<RunState>,
}
impl FeaRunControl {
    pub fn renew(&self, budget: FeaRunBudget) -> Result<(), FeaRunDenied> {
        self.state.renew(budget)
    }
    pub fn deny(&self, denied: FeaRunDenied) {
        self.state.deny(denied);
    }
    pub fn check(&self) -> Result<(), FeaRunDenied> {
        self.state.check()
    }
    pub fn usage(&self) -> ResourceUsage {
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
        let now = runmat_time::unix_timestamp_ms().min(u128::from(u64::MAX)) as u64;
        let elapsed = grant
            .elapsed_wall_millis
            .checked_add(now.saturating_sub(grant.observed_at_unix_millis))
            .ok_or(FeaRunDenied::LicenseInvalid)?;
        let scope = Self::start(
            grant.request,
            LeaseAuthority::Metered(grant.budget),
            None,
            elapsed,
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
    /// renew or settle a lease and must be revalidated before returning bytes.
    pub fn from_parent_supervised_delegation(
        request: FeaRunRequest,
        elapsed_wall_millis: u64,
    ) -> Result<Self, FeaRunDenied> {
        request.validate()?;
        Self::start(
            request,
            LeaseAuthority::ParentSupervised,
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
        owns_permit: bool,
    ) -> Result<Self, FeaRunDenied> {
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
        let state = Arc::new(RunState {
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
            #[cfg(not(target_arch = "wasm32"))]
            heartbeat: heartbeat::Heartbeat::default(),
        });
        if let LeaseAuthority::Metered(budget) = authority {
            budget.validate().inspect_err(|denied| {
                state.deny(denied.clone());
            })?;
        }
        state.check()?;
        #[cfg(not(target_arch = "wasm32"))]
        if matches!(authority, LeaseAuthority::Metered(_)) {
            state
                .heartbeat
                .start(Arc::downgrade(&state), owns_permit)
                .map_err(|_| FeaRunDenied::ServerUnreachable)?;
        }
        Ok(Self { state, owns_permit })
    }
    fn nested(request: &FeaRunRequest) -> Result<Option<Self>, FeaRunDenied> {
        request.validate()?;
        let Some(state) = scope::current() else {
            return Ok(None);
        };
        if request.site != state.request.site
            || request.allocated_cpu_millicores > state.allocation()
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        state.check()?;
        Ok(Some(Self {
            state,
            owns_permit: false,
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
    pub fn usage(&self) -> ResourceUsage {
        self.state.usage()
    }
    pub fn renew(&self, budget: FeaRunBudget) -> Result<(), FeaRunDenied> {
        self.state.renew(budget)
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
        let owns = std::mem::replace(&mut self.owns_permit, false);
        if owns {
            self.state.settle(outcome)
        } else {
            self.state.check()
        }
    }
}
impl Drop for FeaRunScope {
    fn drop(&mut self) {
        if self.owns_permit {
            let _ = self.state.settle(FeaRunOutcome::Failed);
        }
    }
}
#[cfg(test)]
#[path = "tests.rs"]
mod tests;
