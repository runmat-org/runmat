//! Own submitted worker tasks until their results have been collected.

use runmat_execution::{CancellationReason, TaskHandle};
use runmat_runtime::context::RuntimeContext;
use runmat_value::{SharedTaskHandle, Value};

/// Reuses the submission vector; no second task list or workspace snapshot.
/// Dropping a pending group also covers a dropped executor future, not only
/// errors that return through its ordinary control flow.
pub(super) struct SubmittedTaskGroup<'a> {
    runtime: &'a RuntimeContext,
    tasks: Vec<SharedTaskHandle>,
    pending: bool,
}

impl<'a> SubmittedTaskGroup<'a> {
    pub(super) fn with_capacity(runtime: &'a RuntimeContext, capacity: usize) -> Self {
        Self {
            runtime,
            tasks: Vec::with_capacity(capacity),
            pending: true,
        }
    }

    pub(super) fn push(&mut self, task: TaskHandle) {
        self.tasks.push(task.into());
    }

    pub(super) fn as_slice(&self) -> &[SharedTaskHandle] {
        &self.tasks
    }

    pub(super) fn complete(&mut self) {
        self.pending = false;
    }

    pub(super) fn cancel(&mut self, reason: CancellationReason) {
        if !self.pending {
            return;
        }
        self.pending = false;
        for task in &self.tasks {
            let _ = self
                .runtime
                .execution()
                .cancel(&Value::Task(task.clone()), reason);
        }
    }
}

impl Drop for SubmittedTaskGroup<'_> {
    fn drop(&mut self) {
        self.cancel(CancellationReason::ParentScope);
    }
}
