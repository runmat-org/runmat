//! Explicit per-invocation host selection. Installing this context does not
//! acquire a lease: ordinary scripts remain unmetered until an FEA root starts.
use super::FeaRunAdmission;
use std::{
    cell::RefCell,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

thread_local! {
    static HOST: RefCell<Option<Arc<dyn FeaRunAdmission>>> = const { RefCell::new(None) };
}

pub(super) fn current() -> Option<Arc<dyn FeaRunAdmission>> {
    HOST.with(|value| value.borrow().clone())
}

#[derive(Clone)]
pub struct FeaRunAdmissionContext {
    admission: Arc<dyn FeaRunAdmission>,
}

/// Thread-bound host selection; dropping restores the previous invocation host.
pub struct FeaRunAdmissionHostGuard {
    previous: Option<Arc<dyn FeaRunAdmission>>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for FeaRunAdmissionHostGuard {
    fn drop(&mut self) {
        HOST.with(|value| *value.borrow_mut() = self.previous.take());
    }
}

impl FeaRunAdmissionContext {
    /// Trusted host boundary: constrain independently starting child FEA roots
    /// to the original private worker reservation and its fixed deadline. This
    /// selects an installed provider lazily; it does not grant a lease.
    pub fn with_current_resource_bounds(
        cpu_millicores: u64,
        deadline_unix_millis: u64,
    ) -> Result<Self, super::FeaRunDenied> {
        if cpu_millicores == 0 {
            return Err(super::FeaRunDenied::LicenseInvalid);
        }
        Ok(Self::new(Arc::new(ResourceBoundHost {
            provider: super::provider::current(),
            cpu_millicores,
            deadline_unix_millis,
        })))
    }
    pub fn enter(&self) -> FeaRunAdmissionHostGuard {
        let previous = HOST.with(|value| value.replace(Some(Arc::clone(&self.admission))));
        FeaRunAdmissionHostGuard {
            previous,
            _thread: std::marker::PhantomData,
        }
    }
    pub fn new(admission: Arc<dyn FeaRunAdmission>) -> Self {
        Self { admission }
    }

    pub fn run_sync<T>(&self, operation: impl FnOnce() -> T) -> T {
        let _restore = self.enter();
        operation()
    }

    /// Host identity is present only during this invocation's polls, never
    /// while an unrelated window/task runs on the same executor thread.
    pub fn run_future<F: Future>(&self, future: F) -> FeaRunAdmissionFuture<F> {
        FeaRunAdmissionFuture {
            context: self.clone(),
            future: Box::pin(future),
        }
    }
}

struct ResourceBoundHost {
    provider: Result<Arc<dyn FeaRunAdmission>, super::FeaRunDenied>,
    cpu_millicores: u64,
    deadline_unix_millis: u64,
}
impl FeaRunAdmission for ResourceBoundHost {
    fn admit(
        &self,
        request: &super::FeaRunRequest,
    ) -> Result<Box<dyn super::FeaRunPermit>, super::FeaRunDenied> {
        request.validate()?;
        if request.allocated_cpu_millicores > self.cpu_millicores {
            return Err(super::FeaRunDenied::LicenseInvalid);
        }
        let now = runmat_time::unix_timestamp_ms();
        let remaining = u128::from(self.deadline_unix_millis).saturating_sub(now);
        if remaining == 0 {
            return Err(super::FeaRunDenied::LeaseExpired);
        }
        let permit = self
            .provider
            .as_ref()
            .map_err(Clone::clone)?
            .admit(request)?;
        Ok(Box::new(ResourceBoundPermit {
            permit,
            deadline_unix_millis: self.deadline_unix_millis,
        }))
    }
    fn allocated_cpu_capacity_millicores(&self) -> Option<u64> {
        Some(self.cpu_millicores)
    }
    fn site(&self) -> super::FeaRunSite {
        self.provider
            .as_ref()
            .map_or(super::FeaRunSite::Local, |provider| provider.site())
    }
    fn origin(&self) -> Option<super::FeaRunOrigin> {
        self.provider
            .as_ref()
            .ok()
            .and_then(|provider| provider.origin())
    }
    fn local_organization_hint(&self) -> super::FeaLocalOrganization {
        self.provider
            .as_ref()
            .ok()
            .map_or(super::FeaLocalOrganization::Default, |provider| {
                provider.local_organization_hint()
            })
    }
}
struct ResourceBoundPermit {
    permit: Box<dyn super::FeaRunPermit>,
    deadline_unix_millis: u64,
}
impl ResourceBoundPermit {
    fn clamp(&self, mut budget: super::FeaRunBudget) -> super::FeaRunBudget {
        budget.expires_at_unix_millis =
            budget.expires_at_unix_millis.min(self.deadline_unix_millis);
        budget
    }
}
impl super::FeaRunPermit for ResourceBoundPermit {
    fn budget(&self) -> super::FeaRunBudget {
        self.clamp(self.permit.budget())
    }
    fn heartbeat(
        &self,
        usage: &super::FeaRunUsage,
    ) -> Result<super::FeaRunBudget, super::FeaRunDenied> {
        let budget = self.clamp(self.permit.heartbeat(usage)?);
        budget.validate()?;
        Ok(budget)
    }
    fn settle(
        self: Box<Self>,
        outcome: super::FeaRunOutcome,
        usage: &super::FeaRunUsage,
    ) -> Result<(), super::FeaRunDenied> {
        self.permit.settle(outcome, usage)
    }
    fn request_grant(
        &self,
        usage: &super::FeaRunUsage,
        required: runmat_simulation_usage::OperationCount,
    ) -> Result<super::FeaRunBudget, super::FeaRunDenied> {
        let budget = self.clamp(self.permit.request_grant(usage, required)?);
        budget.validate()?;
        Ok(budget)
    }
}

pub struct FeaRunAdmissionFuture<F> {
    context: FeaRunAdmissionContext,
    future: Pin<Box<F>>,
}

impl<F: Future> Future for FeaRunAdmissionFuture<F> {
    type Output = F::Output;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        this.context.run_sync(|| this.future.as_mut().poll(context))
    }
}
