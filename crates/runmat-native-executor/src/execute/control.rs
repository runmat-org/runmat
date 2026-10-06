//! One translation from shared live controls into the existing native ABI.
use runmat_execution::{CancellationReason, CancellationToken};
use runmat_runtime::native::{NativeCancellation, NativeCancellationReason, NativeExit};

pub(super) fn poll(token: &CancellationToken) -> Option<NativeExit> {
    token.stop_reason().map(exit)
}

pub(super) fn exit(reason: CancellationReason) -> NativeExit {
    let reason = match reason {
        CancellationReason::Deadline => NativeCancellationReason::DEADLINE,
        CancellationReason::ParentScope => NativeCancellationReason::PARENT_SCOPE,
        CancellationReason::Shutdown => NativeCancellationReason::SHUTDOWN,
        CancellationReason::User | CancellationReason::Quota => NativeCancellationReason::REQUESTED,
    };
    NativeExit::cancelled(NativeCancellation {
        reason,
        reserved: 0,
        generation: 1,
    })
}
