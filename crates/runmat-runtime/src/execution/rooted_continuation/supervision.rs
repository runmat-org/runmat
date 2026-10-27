//! Allocation ownership for a supervisor that must publish its child's
//! terminal state, including cancellation and already-completed effects.

use super::*;

/// The supervisor owns cancellation handling; its child operations retain
/// their ordinary control checks. Intercepting cancellation around the whole
/// supervisor would drop it before it can publish the child's terminal state.
/// This carrier owns only the concrete allocation and its original funding.
pub struct AdmittedExecutionSupervisor<'a, O, E> {
    // Physical deallocation precedes refund, including abandonment.
    future: Pin<Box<dyn Future<Output = Result<O, E>> + 'a>>,
    _claim: Option<HostAllocationReceipt>,
}

impl<O, E> Future for AdmittedExecutionSupervisor<'_, O, E> {
    type Output = Result<O, E>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut().future.as_mut().poll(context)
    }
}

/// Admit before constructing the command supervisor. Unlike a cancellable
/// hostcall, this boundary leaves terminal handling to the supervised pipeline.
/// It does not grant new authority, reset cancellation, or change child checks.
#[inline(never)]
pub fn admit_execution_supervisor_with<'a, F, O, E>(
    runtime: &RuntimeContext,
    make_future: impl FnOnce() -> F,
) -> Result<AdmittedExecutionSupervisor<'a, O, E>, E>
where
    F: Future<Output = Result<O, E>> + 'a,
    O: 'a,
    E: From<RuntimeError> + 'a,
{
    let (cancellation, admission) = checked_layout::<F, O, E>(Some(runtime));
    let claim = admission.map_err(|error| {
        E::from(semantic_control_error(
            cancellation.as_ref(),
            "execution supervisor",
            error,
        ))
    })?;
    Ok(AdmittedExecutionSupervisor {
        future: Box::pin(make_future()),
        _claim: claim,
    })
}
