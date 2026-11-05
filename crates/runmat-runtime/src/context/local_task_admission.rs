//! Exact origin selection before session-local work is queued.

use super::{RuntimeContext, RuntimeLocalTaskFuture, RuntimeLocalTaskLease};
use crate::RuntimeError;

#[derive(Clone, Copy)]
pub(crate) enum LocalTaskCommandOrigin {
    /// The currently admitted command, captured before the task is queued.
    CurrentSelected,
    /// Permanent service work and custom ports have no provable command owner.
    Unknown,
}

impl RuntimeContext {
    /// Construct a future under its selected task context, then submit it.
    /// The factory is needed because an Unknown/custom route must mask the
    /// context captured by the future, not just its later poll-local TLS.
    pub(crate) fn spawn_local_with_command_origin(
        &self,
        operation: &'static str,
        origin: LocalTaskCommandOrigin,
        make_future: impl FnOnce(RuntimeContext) -> RuntimeLocalTaskFuture,
    ) -> Result<RuntimeLocalTaskLease, RuntimeError> {
        let tasks = self
            .service_ports()
            .require_local_tasks(operation)
            .map_err(super::RuntimeCapabilityError::into_runtime_error)?;
        let canonical = tasks.canonical_admission();
        let selected = match (origin, canonical.as_ref()) {
            (LocalTaskCommandOrigin::CurrentSelected, Some(_)) => self.command_profiling().cloned(),
            _ => None,
        };
        let deferred = selected
            .as_ref()
            .and_then(|owner| owner.begin_deferred_work());
        let selected = if deferred.is_some() { selected } else { None };
        let task_context = self
            .clone()
            .with_cleanup_command_origin(selected)
            .with_cleanup_completion(None);
        let future = make_future(task_context);
        match canonical {
            Some(canonical) => canonical.spawn_admitted(self.local_identity(), future, deferred),
            None => tasks.spawn_local(self.local_identity(), future),
        }
    }
}
