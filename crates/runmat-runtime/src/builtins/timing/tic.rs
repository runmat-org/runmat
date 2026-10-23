//! Session-owned stopwatch start and opaque token creation.

use runmat_builtins::{
    BuiltinErrorDescriptor, TIC_ERROR_CONTEXT_UNAVAILABLE, TIC_ERROR_TOKEN_EXHAUSTED,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Value};

use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

fn tic_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin("tic");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "tic",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::timing::tic"
)]
pub async fn tic_builtin(
    runtime: Option<RuntimeContext>,
    requested_outputs: Option<usize>,
) -> BuiltinResult<Value> {
    let runtime = runtime.ok_or_else(|| tic_error(&TIC_ERROR_CONTEXT_UNAVAILABLE))?;
    let now = runtime.service_ports().timing()?.monotonic_elapsed();
    // A missing count is treated as observed for direct API callers and
    // nested `toc(tic)` expressions without an enclosing output scope.
    let publish = requested_outputs != Some(0);
    let token = runtime.state().timing.borrow_mut().start(now, publish);
    match token {
        Some(token) => Ok(Value::Int(IntValue::U64(token.raw()))),
        None if !publish => Ok(Value::Int(IntValue::U64(0))),
        None => Err(tic_error(&TIC_ERROR_TOKEN_EXHAUSTED)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;
    use futures::executor::block_on;
    use std::rc::Rc;

    #[test]
    fn tic_returns_session_owned_uint64_token() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let Value::Int(IntValue::U64(first)) =
            block_on(tic_builtin(Some(runtime.clone()), None)).unwrap()
        else {
            panic!("expected uint64 token")
        };
        let Value::Int(IntValue::U64(second)) =
            block_on(tic_builtin(Some(runtime.clone()), None)).unwrap()
        else {
            panic!("expected uint64 token")
        };
        assert_ne!(first, second);
        assert!(runtime
            .state()
            .timing
            .borrow()
            .start_for(crate::context::StopwatchToken::from_raw(first))
            .is_some());
        assert!(runtime
            .state()
            .timing
            .borrow()
            .start_for(crate::context::StopwatchToken::from_raw(second))
            .is_some());
    }

    #[test]
    fn unobserved_tic_retains_only_latest_start() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        for _ in 0..1000 {
            block_on(tic_builtin(Some(runtime.clone()), Some(0))).unwrap();
        }
        assert!(runtime.state().timing.borrow().latest_start().is_some());
        let foreign = crate::context::StopwatchToken::from_raw(0);
        assert!(runtime.state().timing.borrow().start_for(foreign).is_none());
    }
}
