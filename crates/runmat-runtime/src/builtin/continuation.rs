use std::future::Future;

use crate::context::host_allocation::HostAllocationError;
use crate::context::RuntimeContext;
use crate::execution::rooted_continuation::{
    admit_rooted_future, admit_rooted_future_with, rooted_future_layout,
};
use crate::execution::RootedValueSequence;
use crate::RuntimeError;

use super::{builtin_control_error, BuiltinContinuationInput, RuntimeBuiltinFuture};

/// Preserve an ingress failure without allocating a continuation or acquiring
/// new execution authority for a builtin body that will never run.
#[doc(hidden)]
pub fn rejected_builtin_future(error: RuntimeError) -> RuntimeBuiltinFuture {
    crate::execution::RuntimeContinuationFuture::rejected(error)
}

/// Erase a concrete builtin continuation once, at the canonical rooted call
/// boundary. Admission precedes the one heap allocation and the first poll.
#[doc(hidden)]
pub fn admitted_builtin_future<F>(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
    future: F,
) -> RuntimeBuiltinFuture
where
    F: Future<Output = Result<RootedValueSequence, RuntimeError>> + 'static,
{
    admit_rooted_future(runtime, builtin, builtin_control_error, future)
}

/// Factory form used by generated canonical bindings. Keeping the concrete
/// body out of the generated wrapper avoids materializing a target-sized
/// async state on an already nested interpreter poll stack. Admission still
/// measures that exact body type and precedes the one erased heap owner.
#[doc(hidden)]
#[inline(never)]
pub fn admitted_builtin_future_with<F>(
    runtime: Option<RuntimeContext>,
    builtin: &'static str,
    make_future: impl FnOnce(Option<RuntimeContext>) -> F,
) -> RuntimeBuiltinFuture
where
    F: Future<Output = Result<RootedValueSequence, RuntimeError>> + 'static,
{
    admit_rooted_future_with(runtime, builtin, builtin_control_error, make_future)
}

/// Measure the very same generated body type used by the wrapper without
/// constructing arguments, a future, or a dummy invocation. This layout is
/// target-specific and must be queried from the selected worker binary.
#[doc(hidden)]
pub fn continuation_layout_for<F>(
    _: fn(BuiltinContinuationInput) -> F,
) -> Result<u64, HostAllocationError>
where
    F: Future<Output = Result<RootedValueSequence, RuntimeError>>,
{
    rooted_future_layout::<F, RootedValueSequence>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_effect::HostEffectAllocationError;
    use crate::execution::ExecutionServiceError;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::execution::RootedValues;
    #[cfg(not(target_arch = "wasm32"))]
    use runmat_value::{HandleRef, Value, ValueSequence};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn control_future_admission_preserves_typed_control_causes() {
        for allocation_error in [
            HostAllocationError::Cancelled,
            HostAllocationError::Resource(HostEffectAllocationError::Cancelled),
        ] {
            let cancelled = builtin_control_error(None, "completion_test", allocation_error);
            let cause = cancelled.source.as_deref().and_then(|source| {
                (source as &dyn std::error::Error).downcast_ref::<ExecutionServiceError>()
            });
            assert!(matches!(cause, Some(ExecutionServiceError::Cancelled)));
        }

        for allocation_error in [
            HostAllocationError::StaleAuthority,
            HostAllocationError::MissingManagedOwner,
            HostAllocationError::Resource(HostEffectAllocationError::Revoked),
        ] {
            let error = builtin_control_error(None, "completion_test", allocation_error);
            let cause = error.source.as_deref().and_then(|source| {
                (source as &dyn std::error::Error).downcast_ref::<ExecutionServiceError>()
            });
            assert!(matches!(
                cause,
                Some(ExecutionServiceError::Infrastructure(_))
            ));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn unpolled_inputs_stay_rooted_until_completed_output_takes_over() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("builtin".into()))
                .expect("allocate rooted payload");
            let handle = initial.handle();
            let args = RootedValues::new(vec![Value::HandleObject(HandleRef {
                class_name: "BuiltinHandle".into(),
                target: handle,
                valid: true,
            })])
            .expect("root builtin input");
            initial.unroot().expect("transfer input root");

            let pending =
                admitted_builtin_future_with(None, "completion_test", move |_| async move {
                    runmat_gc::gc_collect_major().expect("collect while builtin is pending");
                    assert!(runmat_gc::gc_clone_value(&handle).is_ok());
                    let sequence = ValueSequence::single(args.values()[0].clone())
                        .map_err(crate::sequence::sequence_error_to_runtime)?;
                    RootedValueSequence::new(sequence)
                        .map_err(ExecutionServiceError::into_runtime_error)
                });
            runmat_gc::gc_collect_major().expect("collect before first poll");
            assert!(runmat_gc::gc_clone_value(&handle).is_ok());

            let output = futures::executor::block_on(pending).expect("rooted builtin output");
            runmat_gc::gc_collect_major().expect("collect after input root retires");
            assert!(runmat_gc::gc_clone_value(&handle).is_ok());
            drop(output);
            runmat_gc::gc_collect_major().expect("collect after output discard");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }
}
