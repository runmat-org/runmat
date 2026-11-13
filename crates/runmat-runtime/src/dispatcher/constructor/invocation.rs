//! Exact callback storage and deferred rooted execution for a constructor.

use std::future::Future;

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::execution::rooted_continuation::semantic_control_error;
use crate::execution::{RootedValueSequence, RuntimeContinuationFuture};
use crate::RuntimeError;

type ConstructorFuture = RuntimeContinuationFuture<'static, RootedValueSequence>;
type ConstructorCallback = dyn FnOnce(RuntimeContext) -> Result<ConstructorFuture, RuntimeError>;

/// A selected constructor body whose captured callback and concrete future
/// are admitted separately at their original allocation boundaries. Receiver
/// policy supplies the selected context before the future is constructed.
pub struct PreparedConstructorInvoke {
    // Drop physical callback storage before refunding its external claim.
    callback: Box<ConstructorCallback>,
    claim: Option<HostAllocationReceipt>,
    cancellation: crate::context::RuntimeCancellationContext,
}

/// The original callback allocation, consumed exactly once by constructor policy.
pub(in crate::dispatcher) struct PreparedConstructorBody {
    callback: Box<ConstructorCallback>,
}

/// Funding and diagnostic authority retained by the enclosing constructor call.
/// The transient dispatch target borrows this owner rather than carrying its
/// full receipt in every constructor-target enum value.
pub(in crate::dispatcher) struct PreparedConstructorGuard {
    claim: Option<HostAllocationReceipt>,
    cancellation: crate::context::RuntimeCancellationContext,
}

impl PreparedConstructorInvoke {
    pub fn new<F>(
        context: &RuntimeContext,
        make_future: impl FnOnce(RuntimeContext) -> F + 'static,
    ) -> Result<Self, RuntimeError>
    where
        F: Future<Output = Result<RootedValueSequence, RuntimeError>> + 'static,
    {
        let callback = move |selected: RuntimeContext| {
            // This clone retains the same selected authority, not workspace
            // or argument payloads. The original selected context enters F.
            crate::execution::admit_executable_sequence_with(&selected.clone(), move || {
                make_future(selected)
            })
        };
        let cancellation = context.cancellation_context();
        let claim = u64::try_from(std::mem::size_of_val(&callback))
            .map_err(|_| HostAllocationError::Overflow)
            .and_then(|bytes| HostAllocationPlan::new(bytes, 0))
            .and_then(|plan| HostAllocationReceipt::admit(Some(context), plan))
            .map_err(|cause| {
                semantic_control_error(Some(&cancellation), "constructor callback", cause)
            })?;
        Ok(Self {
            callback: Box::new(callback),
            claim,
            cancellation,
        })
    }

    pub(in crate::dispatcher) fn into_parts(
        self,
    ) -> (PreparedConstructorGuard, PreparedConstructorBody) {
        let Self {
            callback,
            claim,
            cancellation,
        } = self;
        (
            PreparedConstructorGuard {
                claim,
                cancellation,
            },
            PreparedConstructorBody { callback },
        )
    }
}

impl PreparedConstructorBody {
    pub(in crate::dispatcher) fn invoke(
        self,
        selected: RuntimeContext,
        guard: &PreparedConstructorGuard,
    ) -> Result<ConstructorFuture, RuntimeError> {
        if let Some(claim) = &guard.claim {
            claim.checkpoint(&selected).map_err(|cause| {
                semantic_control_error(Some(&guard.cancellation), "constructor callback", cause)
            })?;
        }
        // Consuming FnOnce releases its backing here. The enclosing call keeps
        // guard alive through child completion; the child also owns its own
        // independently admitted allocation.
        (self.callback)(selected)
    }
}
