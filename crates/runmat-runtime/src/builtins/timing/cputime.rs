//! Process CPU accounting through the invocation's timing capability.

use runmat_builtins::{
    BuiltinErrorDescriptor, CPUTIME_ERROR_TOO_MANY_INPUTS, CPUTIME_ERROR_UNAVAILABLE,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

fn cputime_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin("cputime");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "cputime",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::timing::cputime"
)]
fn cputime_builtin(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(cputime_error(&CPUTIME_ERROR_TOO_MANY_INPUTS));
    }
    let seconds = runtime
        .as_ref()
        .map(|runtime| runtime.service_ports().timing())
        .transpose()?
        .and_then(|timing| timing.process_cpu_seconds())
        .ok_or_else(|| cputime_error(&CPUTIME_ERROR_UNAVAILABLE))?;
    Ok(Value::Num(seconds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;
    use std::rc::Rc;

    fn runtime() -> RuntimeContext {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn cputime_returns_finite_nonnegative_scalar() {
        let Value::Num(time) = cputime_builtin(
            Some(runtime()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .expect("cputime") else {
            panic!("expected numeric scalar")
        };
        assert!(time.is_finite() && time >= 0.0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn cputime_is_monotonic_for_current_process() {
        let runtime = runtime();
        let Value::Num(first) = cputime_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .expect("first") else {
            panic!("expected numeric scalar")
        };
        for value in 0_u64..10_000 {
            std::hint::black_box(value.wrapping_mul(value));
        }
        let Value::Num(second) = cputime_builtin(
            Some(runtime),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .expect("second") else {
            panic!("expected numeric scalar")
        };
        assert!(second >= first);
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    fn browser_cpu_time_is_unavailable() {
        let error = cputime_builtin(Some(runtime()), Vec::new()).unwrap_err();
        assert_eq!(error.identifier(), CPUTIME_ERROR_UNAVAILABLE.identifier);
    }

    #[test]
    fn cputime_rejects_inputs_before_clock_access() {
        let error = cputime_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        )
        .unwrap_err();
        assert_eq!(error.identifier(), CPUTIME_ERROR_TOO_MANY_INPUTS.identifier);
    }
}
