//! Construct only the selected invocation after exact continuation admission.
//! Ordinary optimization does not embed start/resume transport futures.
use super::{
    checkpoint,
    preparation::{CheckpointRequest, CheckpointStart, Plan},
    publication,
};
use crate::{
    builtins::math::optim::common::ScalarCooperation,
    context::{with_original_scope, RuntimeContext, RuntimeScopeFactory},
    execution::{
        rooted_continuation::{
            admit_optional_hostcall_with_contextual_error, RuntimeContinuationFuture,
        },
        RootedValueSequence,
    },
    BuiltinResult,
};
pub(super) fn launch<'a>(
    plan: Plan,
    c: &'a mut ScalarCooperation,
) -> BuiltinResult<RuntimeContinuationFuture<'a, RootedValueSequence>> {
    let runtime = c.runtime().cloned();
    match plan {
        Plan::Ordinary(ready) => selected(runtime, || ready.into_inner().run(c)),
        Plan::Inconsistent(variables) => {
            selected(runtime, || publication::inconsistent(variables, c))
        }
        Plan::Checkpoint(CheckpointRequest::Start(start)) => selected(runtime, || {
            let CheckpointStart { path, raw, options } = start.into_inner();
            checkpoint::start(path, raw, options, c)
        }),
        Plan::Checkpoint(CheckpointRequest::Resume { path, options }) => {
            selected(runtime, || checkpoint::resume(path, options, c))
        }
    }
}
#[inline(never)]
fn selected<'a, F>(
    runtime: Option<RuntimeContext>,
    make: impl FnOnce() -> F,
) -> BuiltinResult<RuntimeContinuationFuture<'a, RootedValueSequence>>
where
    F: std::future::Future<Output = BuiltinResult<RootedValueSequence>> + 'a,
{
    let scope = RuntimeScopeFactory::new(runtime.clone(), make);
    with_original_scope(runtime.as_ref(), || {
        admit_optional_hostcall_with_contextual_error(runtime.as_ref(), || scope.construct())
    })
}
