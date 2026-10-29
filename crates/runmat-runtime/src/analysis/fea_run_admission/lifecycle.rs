use super::{
    scope, FeaRunAdmission, FeaRunDenied, FeaRunOutcome, FeaRunPermit, FeaRunRequest,
    ScopedFeaRunFuture,
};
use runmat_execution::resource::ResourceUsage;
use std::sync::{Arc, Mutex};

#[cfg(not(target_arch = "wasm32"))]
#[path = "heartbeat.rs"]
mod heartbeat;

pub(super) struct RunState {
    pub(super) request: FeaRunRequest,
    started: runmat_time::Instant,
    permit: Mutex<Option<Box<dyn FeaRunPermit>>>,
    #[cfg(not(target_arch = "wasm32"))]
    heartbeat: heartbeat::Heartbeat,
}

impl RunState {
    pub(super) fn usage(&self) -> ResourceUsage {
        let wall_millis = self.started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        allocated_usage(self.request.allocated_cpu_millicores, wall_millis)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn report(&self) {
        let permit = self.permit.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(permit) = permit.as_ref() {
            permit.heartbeat(&self.usage());
        }
    }

    fn settle(&self, outcome: FeaRunOutcome) {
        #[cfg(not(target_arch = "wasm32"))]
        self.heartbeat.stop();
        let permit = self.permit.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(permit) = permit {
            permit.settle(outcome, &self.usage());
        }
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
        self.settle(FeaRunOutcome::Failed);
    }
}

/// One admitted run; nested operations borrow its authorization and accounting.
pub struct FeaRunScope {
    state: Arc<RunState>,
    owns_permit: bool,
}

impl FeaRunScope {
    pub fn admit(request: FeaRunRequest) -> Result<Self, FeaRunDenied> {
        if let Some(scope) = Self::nested(&request)? {
            return Ok(scope);
        }
        let admission = super::provider::current()?;
        Self::admit_with(request, admission.as_ref())
    }

    /// An explicit trusted host port avoids mutating the process default, for
    /// example after a browser session has completed asynchronous admission.
    pub fn admit_with(
        request: FeaRunRequest,
        admission: &dyn FeaRunAdmission,
    ) -> Result<Self, FeaRunDenied> {
        if let Some(scope) = Self::nested(&request)? {
            return Ok(scope);
        }
        let permit = admission.admit(&request)?;
        let state = Arc::new(RunState {
            request,
            started: runmat_time::Instant::now(),
            permit: Mutex::new(Some(permit)),
            #[cfg(not(target_arch = "wasm32"))]
            heartbeat: heartbeat::Heartbeat::default(),
        });
        #[cfg(not(target_arch = "wasm32"))]
        state
            .heartbeat
            .start(Arc::downgrade(&state))
            .map_err(|_| FeaRunDenied::ServerUnreachable)?;
        Ok(Self {
            state,
            owns_permit: true,
        })
    }

    fn nested(request: &FeaRunRequest) -> Result<Option<Self>, FeaRunDenied> {
        request.validate()?;
        let Some(state) = scope::current() else {
            return Ok(None);
        };
        if request.site != state.request.site
            || request.allocated_cpu_millicores > state.request.allocated_cpu_millicores
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        if state
            .permit
            .lock()
            .map_err(|_| FeaRunDenied::LicenseInvalid)?
            .is_none()
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        Ok(Some(Self {
            state,
            owns_permit: false,
        }))
    }

    pub fn current_allocation() -> Option<u64> {
        scope::current().map(|s| s.request.allocated_cpu_millicores)
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

    pub fn settle(self, outcome: FeaRunOutcome) {
        if self.owns_permit {
            self.state.settle(outcome);
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
