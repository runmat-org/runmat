//! Allocation-free asynchronous execution boundary for a PCG recurrence.
//!
//! The recurrence owns its admitted numeric workspace and decides when an A,
//! M1, or M2 application is needed. This driver performs each requested
//! operation exactly once, then returns its borrowed native result to the
//! one recurrence also driven by the synchronous matrix-only route.

use super::kernel::{PcgOutcome, PcgScalar};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

/// The recurrence, not the callback adapter, chooses when preconditioning
/// occurs. A solver with no M1/M2 simply never yields those requests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PcgOperation {
    Coefficient,
    FirstPreconditioner,
    SecondPreconditioner,
}

pub(super) enum PcgContinuation<'a, T> {
    Apply {
        operation: PcgOperation,
        input: &'a [T],
    },
    Complete(PcgOutcome),
}

/// Pure recurrence state over the caller's already-admitted numeric slices.
/// `accept` consumes one completed operation result; it may not request that
/// the driver re-execute a callback whose effect has already occurred.
pub(super) trait PcgRecurrence<T: PcgScalar> {
    fn continuation(&self) -> PcgContinuation<'_, T>;
    fn accept(
        &mut self,
        output: &[T],
        checkpoint: &mut dyn FnMut() -> Result<(), RuntimeError>,
    ) -> Result<(), RuntimeError>;
}

/// Generic async operation authority. A concrete callable implementation
/// must retain its input Value and the user-produced output independently;
/// this driver's scratch slice is temporary and cannot fund either owner.
pub(super) trait AsyncPcgOperation<T: PcgScalar> {
    async fn apply(
        &mut self,
        operation: PcgOperation,
        input: &[T],
        output: &mut [T],
        runtime: &RuntimeContext,
    ) -> Result<(), RuntimeError>;
}

fn admission(runtime: &RuntimeContext, error: HostAllocationError) -> RuntimeError {
    crate::builtin::builtin_control_error(Some(&runtime.cancellation_context()), "pcg", error)
}

/// Drive a single owned recurrence to completion. Consuming `recurrence`
/// makes an error terminal: callers cannot retry a completed callback by
/// replaying a partially advanced state. The result slice is an admitted,
/// caller-owned numeric workspace, never a published language Value.
pub(super) async fn drive<T, R, O>(
    mut recurrence: R,
    mut operation: O,
    output_scratch: &mut [T],
    runtime: &RuntimeContext,
    receipt: Option<&HostAllocationReceipt>,
) -> Result<PcgOutcome, RuntimeError>
where
    T: PcgScalar,
    R: PcgRecurrence<T>,
    O: AsyncPcgOperation<T>,
{
    let cancellation = runtime.cancellation();
    let initial_policy =
        HostAllocationReceipt::policy(Some(runtime)).map_err(|error| admission(runtime, error))?;
    if receipt.is_none() && initial_policy == HostAllocationPolicy::Managed {
        return Err(admission(runtime, HostAllocationError::MissingManagedOwner));
    }
    let mut checkpoint = || {
        if !cancellation.same_generation(&runtime.cancellation()) {
            return Err(admission(runtime, HostAllocationError::StaleAuthority));
        }
        match receipt {
            Some(receipt) => receipt
                .checkpoint(runtime)
                .map_err(|error| admission(runtime, error)),
            None => {
                let current = HostAllocationReceipt::policy(Some(runtime))
                    .map_err(|error| admission(runtime, error))?;
                if current != initial_policy {
                    return Err(admission(runtime, HostAllocationError::StaleAuthority));
                }
                Ok(())
            }
        }
    };

    loop {
        checkpoint()?;
        let (requested, input) = match recurrence.continuation() {
            PcgContinuation::Complete(outcome) => return Ok(outcome),
            PcgContinuation::Apply { operation, input } => (operation, input),
        };
        if output_scratch.len() != input.len() {
            return Err(semantic_error(
                "PcgOperationShape",
                "PCG operation scratch extent does not match its input",
            ));
        }
        operation
            .apply(requested, input, output_scratch, runtime)
            .await?;
        // No callback replay occurs if cancellation or rebind wins here. The
        // owned recurrence is discarded with the error before `accept`.
        checkpoint()?;
        recurrence.accept(output_scratch, &mut checkpoint)?;
    }
}
