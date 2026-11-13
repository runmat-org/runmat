//! One actual auxiliary invocation, not a fitness/poll/element allocation.
//! Classic poll and LHS never enter this boundary.
use super::*;
use crate::{
    context::{with_original_scope, RuntimeContext, RuntimeScopeFactory},
    execution::rooted_continuation::{
        admit_optional_hostcall_with_contextual_error, RuntimeContinuationFuture,
    },
};
#[inline(never)]
pub(in crate::builtins::math::optim::patternsearch) fn admit<'a, F>(
    runtime: Option<RuntimeContext>,
    make: impl FnOnce() -> F,
) -> BuiltinResult<RuntimeContinuationFuture<'a, poll::Outcome>>
where
    F: std::future::Future<Output = BuiltinResult<poll::Outcome>> + 'a,
{
    let scope = RuntimeScopeFactory::new(runtime.clone(), make);
    with_original_scope(runtime.as_ref(), || {
        admit_optional_hostcall_with_contextual_error(runtime.as_ref(), || scope.construct())
    })
}
