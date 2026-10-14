//! Runtime-owned command accounting for work physically executed on a foreign
//! language lane. Payloads and interpreter/JNI state remain with their owners.

use runmat_execution::profiling::{
    select_command_profiling, with_command_profiling, CommandDeferredWork, CommandProfilingReceipt,
    CommandProfilingSelectionGuard, DeferredWorkTerminal,
};

use crate::context::RuntimeContext;

pub(super) struct NativeLaneProfiling {
    receipt: Option<CommandProfilingReceipt>,
    deferred: Option<CommandDeferredWork>,
}

impl NativeLaneProfiling {
    /// Admission precedes queueing. A refused lease remains unknown, never a
    /// fabricated zero-work completion.
    pub(super) fn admit(context: &RuntimeContext) -> Self {
        let receipt = context.command_profiling().cloned();
        let deferred = receipt
            .as_ref()
            .and_then(|owner| owner.begin_deferred_work());
        Self { receipt, deferred }
    }

    pub(super) fn execute<T, E>(self, operation: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
        let result = with_command_profiling(self.receipt, operation);
        settle(self.deferred, result.is_ok());
        result
    }
}

fn settle(deferred: Option<CommandDeferredWork>, succeeded: bool) {
    if let Some(deferred) = deferred {
        deferred.complete(if succeeded {
            DeferredWorkTerminal::Completed
        } else {
            DeferredWorkTerminal::Failed
        });
    }
}

struct ActivePythonObservation {
    selected: CommandProfilingSelectionGuard,
    deferred: Option<CommandDeferredWork>,
}

impl runmat_python::PythonLaneObserver for NativeLaneProfiling {
    fn enter(self: Box<Self>) -> Box<dyn runmat_python::PythonLaneObservation> {
        let Self { receipt, deferred } = *self;
        Box::new(ActivePythonObservation {
            selected: select_command_profiling(receipt),
            deferred,
        })
    }
}

impl runmat_python::PythonLaneObservation for ActivePythonObservation {
    fn complete(self: Box<Self>, succeeded: bool) {
        let Self { selected, deferred } = *self;
        drop(selected);
        settle(deferred, succeeded);
    }
}
