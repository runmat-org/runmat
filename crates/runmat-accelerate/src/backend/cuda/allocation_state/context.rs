use super::*;
use runmat_accelerate_api::NativeDeviceContextGuard;

#[derive(Debug, Clone)]
pub(super) struct ContextUnavailable(pub(super) String);

impl std::fmt::Display for ContextUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUDA context unavailable: {}", self.0)
    }
}
impl std::error::Error for ContextUnavailable {}

impl AllocationState {
    pub(super) fn ensure_available(&self) -> Result<()> {
        if self.allocations.is_poisoned() || self.unavailable.is_poisoned() {
            return Err(ContextUnavailable("owner bookkeeping is poisoned".into()).into());
        }
        if let Some(error) = self
            .unavailable
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            return Err(error.clone().into());
        }
        Ok(())
    }

    fn leave_context(&self) -> Result<()> {
        catch_allocation_cleanup_unwind(|| self.driver.pop())
            .unwrap_or_else(|_| Err(anyhow!("context pop unwound")))
            .map_err(|error| {
                let failure = ContextUnavailable(format!("could not leave CUDA context: {error}"));
                *self.unavailable.lock().unwrap_or_else(|e| e.into_inner()) = Some(failure.clone());
                failure.into()
            })
    }

    pub(in crate::backend::cuda) fn with_context<T>(
        &self,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        self.ensure_available()?;
        self.driver.push(self.context)?;
        let scope = ContextScope {
            owner: self,
            active: true,
        };
        let result = operation();
        let leave = scope.finish();
        match (result, leave) {
            (Ok(value), Ok(())) => {
                // A nested operation or another execution lane may have
                // quarantined this shared owner while the operation ran.
                // Check at completion; this is not a lease on future health.
                self.ensure_available()?;
                Ok(value)
            }
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
        }
    }

    pub(in crate::backend::cuda) fn enter_native(
        self: &Arc<Self>,
    ) -> Result<NativeDeviceContextGuard> {
        self.ensure_available()?;
        self.driver.push(self.context)?;
        let owner = self.clone();
        Ok(NativeDeviceContextGuard::new(move || owner.leave_context()))
    }
}

pub(super) struct ContextScope<'a> {
    pub(super) owner: &'a AllocationState,
    pub(super) active: bool,
}
impl ContextScope<'_> {
    pub(super) fn finish(mut self) -> Result<()> {
        self.active = false;
        self.owner.leave_context()
    }
}
impl Drop for ContextScope<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = catch_allocation_cleanup_unwind(|| {
                if let Err(error) = self.owner.leave_context() {
                    log::error!("{error}");
                }
            });
        }
    }
}
