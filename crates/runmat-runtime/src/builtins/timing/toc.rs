//! Session-owned elapsed stopwatch reads.

use runmat_builtins::{
    BuiltinErrorDescriptor, TOC_ERROR_INVALID_HANDLE, TOC_ERROR_NO_MATCHING_TIC,
    TOC_ERROR_TOO_MANY_INPUTS, TOC_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Value};

use crate::context::{RuntimeContext, StopwatchToken};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

fn toc_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin("toc");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "toc",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::timing::toc"
)]
pub async fn toc_builtin(
    runtime: Option<RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<f64> {
    if args.len() > 1 {
        return Err(toc_error(&TOC_ERROR_TOO_MANY_INPUTS));
    }
    if requested_outputs.is_some_and(|count| count > 1) {
        return Err(toc_error(&TOC_ERROR_TOO_MANY_OUTPUTS));
    }
    let runtime = runtime.ok_or_else(|| toc_error(&TOC_ERROR_NO_MATCHING_TIC))?;
    let start = if let Some(value) = args.first() {
        let token = parse_token(value).ok_or_else(|| toc_error(&TOC_ERROR_INVALID_HANDLE))?;
        runtime
            .state()
            .timing
            .borrow()
            .start_for(token)
            .ok_or_else(|| toc_error(&TOC_ERROR_INVALID_HANDLE))?
    } else {
        runtime
            .state()
            .timing
            .borrow()
            .latest_start()
            .ok_or_else(|| toc_error(&TOC_ERROR_NO_MATCHING_TIC))?
    };
    let now = runtime.service_ports().timing()?.monotonic_elapsed();
    let elapsed = now
        .checked_sub(start)
        .ok_or_else(|| toc_error(&TOC_ERROR_INVALID_HANDLE))?;
    let seconds = elapsed.as_secs_f64();
    if requested_outputs == Some(0) {
        crate::console::record_console_line(
            crate::console::ConsoleStream::Stdout,
            format!("Elapsed time is {seconds:.6} seconds."),
        )?;
    }
    Ok(seconds)
}

fn parse_token(value: &Value) -> Option<StopwatchToken> {
    match value {
        Value::Int(IntValue::U64(raw)) => Some(StopwatchToken::from_raw(*raw)),
        Value::Tensor(tensor) if tensor.len() == 1 => match tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0))
        {
            Some(IntValue::U64(raw)) => Some(StopwatchToken::from_raw(raw)),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::timing::tic::tic_builtin;
    use crate::execution::RuntimeExecutionService;
    use futures::executor::block_on;
    use std::rc::Rc;

    fn runtime() -> RuntimeContext {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    #[test]
    fn toc_requires_matching_tic() {
        let error = block_on(toc_builtin(
            Some(runtime()),
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), TOC_ERROR_NO_MATCHING_TIC.identifier);
    }

    #[test]
    fn toc_keeps_latest_and_rejects_foreign_token() {
        let first = runtime();
        let second = runtime();
        let token = block_on(tic_builtin(Some(first.clone()), None)).unwrap();
        let elapsed = block_on(toc_builtin(
            Some(first.clone()),
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![token.clone()]),
        ))
        .unwrap();
        let latest = block_on(toc_builtin(
            Some(first),
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        assert!(latest >= elapsed);
        let error = block_on(toc_builtin(
            Some(second),
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![token]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), TOC_ERROR_INVALID_HANDLE.identifier);
    }

    #[test]
    fn toc_rejects_double_and_forged_token() {
        let runtime = runtime();
        let token = block_on(tic_builtin(Some(runtime.clone()), None)).unwrap();
        let Value::Int(IntValue::U64(raw)) = token else {
            panic!("uint64")
        };
        for invalid in [Value::Num(raw as f64), Value::Int(IntValue::U64(0))] {
            let error = block_on(toc_builtin(
                Some(runtime.clone()),
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![invalid]),
            ))
            .unwrap_err();
            assert_eq!(error.identifier(), TOC_ERROR_INVALID_HANDLE.identifier);
        }
    }

    #[test]
    fn toc_rejects_extra_inputs_before_state_access() {
        let error = block_on(toc_builtin(
            None,
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(0.0),
                Value::Num(0.0),
            ]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), TOC_ERROR_TOO_MANY_INPUTS.identifier);
    }

    #[test]
    fn toc_rejects_extra_outputs_before_state_access() {
        let error = block_on(toc_builtin(
            None,
            Some(2),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), TOC_ERROR_TOO_MANY_OUTPUTS.identifier);
    }
}
