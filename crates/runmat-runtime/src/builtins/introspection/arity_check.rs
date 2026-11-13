use runmat_builtins::{
    BuiltinErrorDescriptor, NARGINCHK_DESCRIPTOR, NARGINCHK_ERROR_ARGUMENT_INVALID,
    NARGINCHK_ERROR_BOUNDS_INVALID, NARGINCHK_ERROR_CONTEXT_UNAVAILABLE,
    NARGINCHK_ERROR_NOT_ENOUGH_INPUTS, NARGINCHK_ERROR_TOO_MANY_INPUTS,
    NARGINCHK_ERROR_TOO_MANY_OUTPUTS, NARGOUTCHK_DESCRIPTOR, NARGOUTCHK_ERROR_ARGUMENT_INVALID,
    NARGOUTCHK_ERROR_BOUNDS_INVALID, NARGOUTCHK_ERROR_CONTEXT_UNAVAILABLE,
    NARGOUTCHK_ERROR_NOT_ENOUGH_OUTPUTS, NARGOUTCHK_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_types::{NARGINCHK_BUILTIN_NAME, NARGOUTCHK_BUILTIN_NAME};
use runmat_value::Value;
use runmat_value::{CharArray, StructValue};

use crate::builtins::common::tensor;

#[derive(Clone, Copy)]
enum ArityBound {
    Finite(usize),
    Unbounded,
}

impl ArityBound {
    fn permits(self, actual: usize) -> bool {
        match self {
            Self::Finite(max) => actual <= max,
            Self::Unbounded => true,
        }
    }
}

fn descriptor_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    crate::runtime_descriptor_error(builtin, error)
}

fn parse_finite_arity_bound(
    value: &Value,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> crate::BuiltinResult<usize> {
    if let Value::Int(value) = value {
        return value
            .try_to_usize()
            .ok_or_else(|| descriptor_error(builtin, error));
    }
    if let Value::Tensor(tensor) = value {
        if !tensor::is_scalar_tensor(tensor) {
            return Err(descriptor_error(builtin, error));
        }
        if let Some(storage) = tensor.integer_storage() {
            return storage
                .value_at(0)
                .and_then(|value| value.try_to_usize())
                .ok_or_else(|| descriptor_error(builtin, error));
        }
    }
    let number = match value {
        Value::Num(value) => *value,
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        _ => return Err(descriptor_error(builtin, error)),
    };

    if !number.is_finite()
        || number < 0.0
        || number.fract() != 0.0
        || number > usize::MAX as f64
        || (usize::BITS == 64 && number == usize::MAX as f64)
    {
        return Err(descriptor_error(builtin, error));
    }
    Ok(number as usize)
}

fn parse_max_arity_bound(
    value: &Value,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> crate::BuiltinResult<ArityBound> {
    match value {
        Value::Num(value) if value.is_infinite() && value.is_sign_positive() => {
            Ok(ArityBound::Unbounded)
        }
        Value::Tensor(tensor)
            if tensor::is_scalar_tensor(tensor) && {
                let value = tensor::tensor_value_f64(tensor, 0);
                value.is_infinite() && value.is_sign_positive()
            } =>
        {
            Ok(ArityBound::Unbounded)
        }
        _ => parse_finite_arity_bound(value, builtin, error).map(ArityBound::Finite),
    }
}

fn validate_bounds(
    args: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
    argument_error: &'static BuiltinErrorDescriptor,
    bounds_error: &'static BuiltinErrorDescriptor,
) -> crate::BuiltinResult<(usize, ArityBound)> {
    let min = parse_finite_arity_bound(&args[0], builtin, argument_error)?;
    let max = parse_max_arity_bound(&args[1], builtin, argument_error)?;
    if let ArityBound::Finite(max_value) = max {
        if min > max_value {
            return Err(descriptor_error(builtin, bounds_error));
        }
    }
    Ok((min, max))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArityViolation {
    TooFew,
    TooMany,
}

fn arity_violation(actual: usize, min: usize, max: ArityBound) -> Option<ArityViolation> {
    if actual < min {
        Some(ArityViolation::TooFew)
    } else if !max.permits(actual) {
        Some(ArityViolation::TooMany)
    } else {
        None
    }
}

fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(value)) => Some(value.as_str().to_owned()),
        Value::StringArray(value) if value.data().len() == 1 && !value.data()[0].is_missing() => {
            Some(
                value.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            )
        }
        Value::CharArray(value) if value.rows == 1 => value.row_string(),
        _ => None,
    }
}

fn legacy_nargout_result(violation: Option<ArityViolation>, structured: bool) -> Value {
    let (message, identifier) = match violation {
        Some(ArityViolation::TooFew) => (
            "Not enough output arguments.",
            "MATLAB:nargoutchk:notEnoughOutputs",
        ),
        Some(ArityViolation::TooMany) => (
            "Too many output arguments.",
            "MATLAB:nargoutchk:tooManyOutputs",
        ),
        None if structured => return Value::Struct(StructValue::new()),
        None => {
            return Value::CharArray(
                CharArray::from_code_units(Vec::new(), 0, 0)
                    .expect("empty legacy nargoutchk result"),
            )
        }
    };
    if structured {
        let mut value = runmat_value::UnboundedStructValueBuilder::new();
        value.insert("message", Value::CharArray(CharArray::new_row(message)));
        value.insert(
            "identifier",
            Value::CharArray(CharArray::new_row(identifier)),
        );
        Value::Struct(value.finish())
    } else {
        Value::CharArray(CharArray::new_row(message))
    }
}

fn validate_arg_count(
    args: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
    too_few: &'static BuiltinErrorDescriptor,
    too_many: &'static BuiltinErrorDescriptor,
) -> crate::BuiltinResult<()> {
    match args.len() {
        0 | 1 => Err(descriptor_error(builtin, too_few)),
        2 => Ok(()),
        _ => Err(descriptor_error(builtin, too_many)),
    }
}

#[cfg(test)]
pub(crate) fn dispatch_narginchk_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    dispatch_narginchk_with_outputs(
        runtime,
        None,
        crate::call::FunctionArgumentView::from_slice(&args),
    )
}

fn dispatch_narginchk_with_outputs(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<Value> {
    let builtin = NARGINCHK_BUILTIN_NAME;
    validate_arg_count(
        args,
        builtin,
        &NARGINCHK_ERROR_NOT_ENOUGH_INPUTS,
        &NARGINCHK_ERROR_TOO_MANY_INPUTS,
    )?;
    crate::output_count::validate_fixed_form_outputs(
        &NARGINCHK_DESCRIPTOR,
        &NARGINCHK_ERROR_TOO_MANY_OUTPUTS,
        builtin,
        args.len(),
        requested_outputs,
    )?;
    let (actual_inputs, _) = runtime
        .and_then(|runtime| runtime.invocation_arity())
        .and_then(|arity| arity.function_counts())
        .ok_or_else(|| descriptor_error(builtin, &NARGINCHK_ERROR_CONTEXT_UNAVAILABLE))?;
    let (min, max) = validate_bounds(
        args,
        builtin,
        &NARGINCHK_ERROR_ARGUMENT_INVALID,
        &NARGINCHK_ERROR_BOUNDS_INVALID,
    )?;
    if actual_inputs < min {
        return Err(descriptor_error(
            builtin,
            &NARGINCHK_ERROR_NOT_ENOUGH_INPUTS,
        ));
    }
    if !max.permits(actual_inputs) {
        return Err(descriptor_error(builtin, &NARGINCHK_ERROR_TOO_MANY_INPUTS));
    }
    Ok(Value::output_list(Vec::new()))
}

pub fn dispatch_nargoutchk(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    dispatch_nargoutchk_in_context(None, args)
}

pub fn dispatch_nargoutchk_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    dispatch_nargoutchk_with_outputs(
        runtime,
        None,
        crate::call::FunctionArgumentView::from_slice(&args),
    )
}

fn dispatch_nargoutchk_with_outputs(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<Value> {
    let builtin = NARGOUTCHK_BUILTIN_NAME;
    if !(2..=4).contains(&args.len()) {
        return Err(descriptor_error(
            builtin,
            if args.len() < 2 {
                &NARGOUTCHK_ERROR_NOT_ENOUGH_OUTPUTS
            } else {
                &NARGOUTCHK_ERROR_TOO_MANY_OUTPUTS
            },
        ));
    }
    crate::output_count::validate_fixed_form_outputs(
        &NARGOUTCHK_DESCRIPTOR,
        &NARGOUTCHK_ERROR_TOO_MANY_OUTPUTS,
        builtin,
        args.len(),
        requested_outputs,
    )?;
    let (min, max) = validate_bounds(
        args,
        builtin,
        &NARGOUTCHK_ERROR_ARGUMENT_INVALID,
        &NARGOUTCHK_ERROR_BOUNDS_INVALID,
    )?;
    if args.len() >= 3 {
        let actual_outputs =
            parse_finite_arity_bound(&args[2], builtin, &NARGOUTCHK_ERROR_ARGUMENT_INVALID)?;
        let structured = if let Some(selector) = args.get(3) {
            scalar_text(selector)
                .is_some_and(|selector| selector.trim().eq_ignore_ascii_case("struct"))
        } else {
            false
        };
        if args.len() == 4 && !structured {
            return Err(descriptor_error(
                builtin,
                &NARGOUTCHK_ERROR_ARGUMENT_INVALID,
            ));
        }
        return Ok(if requested_outputs == Some(0) {
            Value::output_list(Vec::new())
        } else {
            legacy_nargout_result(arity_violation(actual_outputs, min, max), structured)
        });
    }
    let (_, actual_outputs) = runtime
        .and_then(|runtime| runtime.invocation_arity())
        .and_then(|arity| arity.function_counts())
        .ok_or_else(|| descriptor_error(builtin, &NARGOUTCHK_ERROR_CONTEXT_UNAVAILABLE))?;
    match arity_violation(actual_outputs, min, max) {
        Some(ArityViolation::TooFew) => {
            return Err(descriptor_error(
                builtin,
                &NARGOUTCHK_ERROR_NOT_ENOUGH_OUTPUTS,
            ))
        }
        Some(ArityViolation::TooMany) => {
            return Err(descriptor_error(
                builtin,
                &NARGOUTCHK_ERROR_TOO_MANY_OUTPUTS,
            ))
        }
        None => {}
    }
    Ok(Value::output_list(Vec::new()))
}

#[runmat_macros::runtime_builtin(
    name = "narginchk",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::introspection::arity_check"
)]
pub fn narginchk_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    dispatch_narginchk_with_outputs(runtime.as_ref(), requested_outputs, args.view())
}

#[runmat_macros::runtime_builtin(
    name = "nargoutchk",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::introspection::arity_check"
)]
pub fn nargoutchk_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    dispatch_nargoutchk_with_outputs(runtime.as_ref(), requested_outputs, args.view())
}

#[cfg(test)]
mod tests;
