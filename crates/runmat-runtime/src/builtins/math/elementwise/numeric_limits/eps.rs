//! Value-dependent IEEE spacing; scalar/class/like forms share numeric limits.

use runmat_macros::runtime_builtin;
use runmat_value::{spacing_f64, Value};

use crate::context::RuntimeContext;
use crate::BuiltinResult;

mod gpu;
mod host;
mod output;
mod sparse;

#[runtime_builtin(
    name = "eps",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::elementwise::numeric_limits::eps"
)]
pub(super) async fn eps_builtin(
    runtime: Option<RuntimeContext>,
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        runmat_builtins::EPS_CATALOG_ENTRY.descriptor,
        &runmat_builtins::NUMERIC_LIMIT_ERROR_INVALID_SYNTAX,
        "eps",
    )?;
    match (arguments.len(), arguments.first()) {
        (1, Some(Value::GpuTensor(handle))) => gpu::value(handle).await,
        (
            1,
            Some(
                value @ (Value::Num(_)
                | Value::Complex(_, _)
                | Value::Tensor(_)
                | Value::ComplexTensor(_)
                | Value::SparseTensor(_)),
            ),
        ) => self::value(value, runtime.as_ref()).await,
        _ => super::execute::run(arguments, super::operation::LimitOperation::EPS),
    }
}

pub(super) async fn value(input: &Value, runtime: Option<&RuntimeContext>) -> BuiltinResult<Value> {
    match input {
        Value::Num(value) => Ok(Value::Num(spacing_f64(*value))),
        Value::Complex(real, imaginary) => Ok(Value::Num(spacing_f64(real.hypot(*imaginary)))),
        Value::Tensor(tensor) => host::real(tensor, runtime).await,
        Value::ComplexTensor(tensor) => host::complex(tensor, runtime).await,
        Value::SparseTensor(tensor) => sparse::value(tensor, runtime).await,
        Value::GpuTensor(_) => Err(super::errors::internal(
            "eps",
            "resident spacing requires async provider execution",
        )),
        _ => Err(invalid_value()),
    }
}

pub(super) fn invalid_value() -> crate::RuntimeError {
    super::errors::class(
        "eps",
        "eps requires a single or double real or complex array",
    )
}
