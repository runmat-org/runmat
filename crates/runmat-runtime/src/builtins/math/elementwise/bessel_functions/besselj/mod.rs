//! First-kind Bessel function for real orders and dense real/complex arguments.

mod execution;
mod input;

use runmat_builtins::{
    BuiltinErrorDescriptor, BESSELJ_ERROR_INVALID_ARGUMENT, BESSELJ_ERROR_SIZE_MISMATCH,
    BESSELJ_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "besselj";

fn error(descriptor: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
        .with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "besselj",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::elementwise::bessel_functions::besselj"
)]
async fn besselj_builtin(
    runtime: Option<RuntimeContext>,
    order: Value,
    argument: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(error(
            &BESSELJ_ERROR_TOO_MANY_OUTPUTS,
            "only one output is defined",
        ));
    }
    let scaled = match (rest.len(), rest.first()) {
        (0, _) => false,
        (1, Some(scale)) => input::scale(scale)?,
        _ => {
            return Err(error(
                &BESSELJ_ERROR_INVALID_ARGUMENT,
                "expected two inputs and optional scalar scale",
            ));
        }
    };
    let order = input::Order::parse(&order)?;
    let argument = input::Argument::parse(&argument)?;
    let shape = input::shape(&order, &argument).ok_or_else(|| {
        error(
            &BESSELJ_ERROR_SIZE_MISMATCH,
            "nu and Z must match in size or one must be scalar",
        )
    })?;
    execution::evaluate(runtime.as_ref(), &order, &argument, shape, scaled).await
}

#[cfg(test)]
mod tests;
