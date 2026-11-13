//! Minimal symbolic `piecewise` builtin.

use crate::{call::BuiltinArguments, context::RuntimeContext};
use runmat_builtins::{
    BuiltinErrorDescriptor, SymbolicScalarOperation, PIECEWISE_DESCRIPTOR,
    PIECEWISE_ERROR_INVALID_INPUT, PIECEWISE_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{SymbolicExpr, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::{symbolic_expr_to_value, value_to_symbolic_scalar};

const BUILTIN_NAME: &str = "piecewise";

fn piecewise_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message)
        .with_builtin(BUILTIN_NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "piecewise",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::symbolic::piecewise",
    context = true
)]
async fn piecewise_builtin(
    runtime: Option<RuntimeContext>,
    args: BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &PIECEWISE_DESCRIPTOR,
        &PIECEWISE_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )
    .map_err(super::invocation::terminal)?;
    if args.len() < 2 {
        return Err(piecewise_error(&PIECEWISE_ERROR_INVALID_INPUT));
    }

    for (index, value) in args.iter().enumerate() {
        super::invocation::validate(SymbolicScalarOperation::Piecewise, index, args.len(), value)?;
    }
    super::invocation::admit(SymbolicScalarOperation::Piecewise, runtime.as_ref(), true)?;
    let mut symbolic_args = Vec::with_capacity(args.len());
    for arg in args.iter() {
        let Some(expr) = value_to_symbolic_scalar(arg) else {
            return Err(piecewise_error(&PIECEWISE_ERROR_INVALID_INPUT));
        };
        symbolic_args.push(expr);
    }

    Ok(symbolic_expr_to_value(SymbolicExpr::function_call(
        BUILTIN_NAME,
        symbolic_args,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn builds_symbolic_piecewise_expression() {
        let result = block_on(piecewise_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Symbolic(SymbolicExpr::function_call(
                    "lt",
                    vec![SymbolicExpr::variable("x"), SymbolicExpr::constant(0.0)],
                )),
                Value::Num(-1.0),
                Value::Symbolic(SymbolicExpr::function_call(
                    "gt",
                    vec![SymbolicExpr::variable("x"), SymbolicExpr::constant(0.0)],
                )),
                Value::Num(1.0),
            ]),
        ))
        .expect("piecewise");

        assert_eq!(result.to_string(), "piecewise(lt(x, 0), -1, gt(x, 0), 1)");
    }

    #[test]
    fn rejects_too_few_arguments() {
        let err = block_on(piecewise_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Symbolic(
                SymbolicExpr::variable("x"),
            )]),
        ))
        .expect_err("piecewise arity should fail");

        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:piecewise:InvalidInput")
        );
    }

    #[test]
    fn rejects_nonconvertible_arguments() {
        let err = block_on(piecewise_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Symbolic(SymbolicExpr::variable("x")),
                Value::String("unsupported".to_string().into()),
            ]),
        ))
        .expect_err("piecewise unsupported argument should fail");

        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:piecewise:InvalidInput")
        );
    }
}
