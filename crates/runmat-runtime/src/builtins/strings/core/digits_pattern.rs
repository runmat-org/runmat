//! Unicode-digit pattern construction with exact, bounded count controls.

use runmat_builtins::{
    BuiltinErrorDescriptor, DIGITS_PATTERN_DESCRIPTOR, DIGITS_PATTERN_ERROR_ARGUMENT_COUNT,
    DIGITS_PATTERN_ERROR_INVALID_COUNT, DIGITS_PATTERN_ERROR_INVALID_RANGE,
    DIGITS_PATTERN_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericScalar, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::core::compat::pattern_object_in_context;
use crate::builtins::strings::core::numeric_length::parse_nonnegative_usize;
use crate::{build_runtime_error, BuiltinResult};

#[runtime_builtin(
    name = "digitsPattern",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::digits_pattern"
)]
pub(crate) async fn digits_pattern_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &DIGITS_PATTERN_DESCRIPTOR,
        &DIGITS_PATTERN_ERROR_TOO_MANY_OUTPUTS,
        "digitsPattern",
    )?;
    if rest.len() > 2 {
        return Err(digits_error(
            &DIGITS_PATTERN_ERROR_ARGUMENT_COUNT,
            DIGITS_PATTERN_ERROR_ARGUMENT_COUNT.message,
        ));
    }
    if rest.iter().any(|value| {
        matches!(
            value,
            Value::Bool(_) | Value::LogicalArray(_) | Value::GpuTensor(_)
        )
    }) {
        return Err(digits_error(
            &DIGITS_PATTERN_ERROR_INVALID_COUNT,
            "digitsPattern: counts must be host nonnegative numeric integer scalars",
        ));
    }
    let regex = match (rest.len(), rest.get(0), rest.get(1)) {
        (0, _, _) => "\\d+".to_string(),
        (1, Some(count), _) => format!("\\d{{{}}}", parse_count(count)?),
        (2, Some(minimum), Some(maximum)) => {
            let minimum = parse_count(minimum)?;
            if is_positive_infinity_scalar(maximum) {
                format!("\\d{{{minimum},}}")
            } else {
                let maximum = parse_count(maximum)?;
                if minimum > maximum {
                    return Err(digits_error(
                        &DIGITS_PATTERN_ERROR_INVALID_RANGE,
                        DIGITS_PATTERN_ERROR_INVALID_RANGE.message,
                    ));
                }
                format!("\\d{{{minimum},{maximum}}}")
            }
        }
        _ => unreachable!("argument count was validated"),
    };
    pattern_object_in_context(runtime.as_ref(), &regex)
}

fn parse_count(value: &Value) -> BuiltinResult<usize> {
    parse_nonnegative_usize(value, "digitsPattern").map_err(|error| {
        digits_error(
            &DIGITS_PATTERN_ERROR_INVALID_COUNT,
            error.message().to_owned(),
        )
    })
}

fn digits_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("digitsPattern");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn is_positive_infinity_scalar(value: &Value) -> bool {
    match value {
        Value::Num(value) => value.is_infinite() && value.is_sign_positive(),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            matches!(tensor.numeric_value_at(0), Some(NumericScalar::F64(value)) if value.is_infinite() && value.is_sign_positive())
                || matches!(tensor.numeric_value_at(0), Some(NumericScalar::F32(value)) if value.is_infinite() && value.is_sign_positive())
        }
        _ => false,
    }
}
