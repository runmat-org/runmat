//! One translation from shared live controls into the existing native ABI.
use runmat_execution::{CancellationReason, CancellationToken};
use runmat_runtime::native::{NativeCancellation, NativeCancellationReason, NativeExit};

pub(super) fn poll(token: &CancellationToken) -> Option<NativeExit> {
    token.stop_reason().map(exit)
}

/// Exit for work that already observed cancellation of `token`, preserving
/// its first published reason (deadline, parent scope, shutdown, ...).
pub(super) fn cancelled(token: &CancellationToken) -> NativeExit {
    exit(token.stop_reason().unwrap_or(CancellationReason::User))
}

/// Host-side error for an observed cancellation, keeping deadlines distinct.
pub(super) fn cancellation_error(reason: CancellationReason) -> crate::NativeExecutorError {
    match reason {
        CancellationReason::Deadline => crate::NativeExecutorError::Deadline,
        _ => crate::NativeExecutorError::Cancelled,
    }
}

fn exit(reason: CancellationReason) -> NativeExit {
    let reason = match reason {
        CancellationReason::Deadline => NativeCancellationReason::DEADLINE,
        CancellationReason::ParentScope => NativeCancellationReason::PARENT_SCOPE,
        CancellationReason::Shutdown => NativeCancellationReason::SHUTDOWN,
        CancellationReason::User
        | CancellationReason::Quota
        | CancellationReason::DependencyFailed => NativeCancellationReason::REQUESTED,
    };
    NativeExit::cancelled(NativeCancellation {
        reason,
        reserved: 0,
        generation: 1,
    })
}
