use super::{ActiveInvocation, NativeInvocation};
use crate::NativeExecutorResult;

/// Sole terminal cleanup authority: dropping the active owner releases its
/// arena, pending work, contextual caches and service references together.
pub(super) struct ExecutionLease<'a> {
    owner: &'a mut NativeInvocation,
    active: Option<ActiveInvocation>,
}

impl<'a> ExecutionLease<'a> {
    pub fn take(owner: &'a mut NativeInvocation) -> NativeExecutorResult<Self> {
        let active = owner.active.take().ok_or_else(super::terminal_error)?;
        Ok(Self {
            owner,
            active: Some(active),
        })
    }

    pub fn active(&self) -> &ActiveInvocation {
        self.active.as_ref().expect("live execution lease")
    }

    pub fn active_mut(&mut self) -> &mut ActiveInvocation {
        self.active.as_mut().expect("live execution lease")
    }

    pub fn check_cancellation(&self) -> NativeExecutorResult<()> {
        match self.active().state.runtime.cancellation().stop_reason() {
            Some(reason) => Err(super::super::control::cancellation_error(reason)),
            None => Ok(()),
        }
    }

    pub fn restore(mut self) {
        self.owner.active = self.active.take();
    }
}

impl Drop for ExecutionLease<'_> {
    fn drop(&mut self) {
        if let Some(active) = self.active.take() {
            self.owner.last_resume = active.resume_state();
            // Pending resource destructors run in their originating runtime,
            // with no ambient activation surviving suspension or abandonment.
            let _runtime = active.state.runtime.enter();
            drop(active);
        }
    }
}
