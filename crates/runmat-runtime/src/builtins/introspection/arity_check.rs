use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_hir::{NARGINCHK_BUILTIN_NAME, NARGOUTCHK_BUILTIN_NAME};
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;

const NO_OUTPUTS: [BuiltinParamDescriptor; 0] = [];

const ARITY_CHECK_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "minArgs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minimum allowed argument count.",
    },
    BuiltinParamDescriptor {
        name: "maxArgs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum allowed argument count.",
    },
];

const NARGINCHK_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "narginchk(minArgs, maxArgs)",
    inputs: &ARITY_CHECK_INPUTS,
    outputs: &NO_OUTPUTS,
}];

const NARGOUTCHK_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "nargoutchk(minArgs, maxArgs)",
    inputs: &ARITY_CHECK_INPUTS,
    outputs: &NO_OUTPUTS,
}];

pub const NARGINCHK_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGINCHK.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:NotEnoughInputs"),
    when: "The current function was called with fewer inputs than the lower bound.",
    message: "narginchk: not enough input arguments",
};

pub const NARGINCHK_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGINCHK.TOO_MANY_INPUTS",
    identifier: Some("RunMat:TooManyInputs"),
    when: "The current function was called with more inputs than the upper bound.",
    message: "narginchk: too many input arguments",
};

pub const NARGINCHK_ERROR_ARGUMENT_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGINCHK.ARGUMENT_INVALID",
    identifier: Some("RunMat:NarginchkArgumentInvalid"),
    when: "A bound argument is not a nonnegative integer scalar or valid Inf upper bound.",
    message: "narginchk: bounds must be nonnegative integer scalars",
};

pub const NARGINCHK_ERROR_BOUNDS_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGINCHK.BOUNDS_INVALID",
    identifier: Some("RunMat:NarginchkBoundsInvalid"),
    when: "The minimum bound is greater than the maximum bound.",
    message: "narginchk: minArgs must be less than or equal to maxArgs",
};

pub const NARGINCHK_ERROR_CONTEXT_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGINCHK.CONTEXT_UNAVAILABLE",
    identifier: Some("RunMat:NarginchkContextUnavailable"),
    when: "The runtime dispatcher is invoked without VM function-call context.",
    message: "narginchk: function call context is unavailable",
};

pub const NARGINCHK_ERRORS: [BuiltinErrorDescriptor; 5] = [
    NARGINCHK_ERROR_NOT_ENOUGH_INPUTS,
    NARGINCHK_ERROR_TOO_MANY_INPUTS,
    NARGINCHK_ERROR_ARGUMENT_INVALID,
    NARGINCHK_ERROR_BOUNDS_INVALID,
    NARGINCHK_ERROR_CONTEXT_UNAVAILABLE,
];

pub const NARGINCHK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NARGINCHK_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NARGINCHK_ERRORS,
};

pub const NARGOUTCHK_ERROR_NOT_ENOUGH_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGOUTCHK.NOT_ENOUGH_OUTPUTS",
    identifier: Some("RunMat:NotEnoughOutputs"),
    when: "The current function was called with fewer requested outputs than the lower bound.",
    message: "nargoutchk: not enough output arguments",
};

pub const NARGOUTCHK_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGOUTCHK.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:TooManyOutputs"),
    when: "The current function was called with more requested outputs than the upper bound.",
    message: "nargoutchk: too many output arguments",
};

pub const NARGOUTCHK_ERROR_ARGUMENT_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGOUTCHK.ARGUMENT_INVALID",
    identifier: Some("RunMat:NargoutchkArgumentInvalid"),
    when: "A bound argument is not a nonnegative integer scalar or valid Inf upper bound.",
    message: "nargoutchk: bounds must be nonnegative integer scalars",
};

pub const NARGOUTCHK_ERROR_BOUNDS_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGOUTCHK.BOUNDS_INVALID",
    identifier: Some("RunMat:NargoutchkBoundsInvalid"),
    when: "The minimum bound is greater than the maximum bound.",
    message: "nargoutchk: minArgs must be less than or equal to maxArgs",
};

pub const NARGOUTCHK_ERROR_CONTEXT_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NARGOUTCHK.CONTEXT_UNAVAILABLE",
    identifier: Some("RunMat:NargoutchkContextUnavailable"),
    when: "The runtime dispatcher is invoked without VM function-call context.",
    message: "nargoutchk: function call context is unavailable",
};

pub const NARGOUTCHK_ERRORS: [BuiltinErrorDescriptor; 5] = [
    NARGOUTCHK_ERROR_NOT_ENOUGH_OUTPUTS,
    NARGOUTCHK_ERROR_TOO_MANY_OUTPUTS,
    NARGOUTCHK_ERROR_ARGUMENT_INVALID,
    NARGOUTCHK_ERROR_BOUNDS_INVALID,
    NARGOUTCHK_ERROR_CONTEXT_UNAVAILABLE,
];

pub const NARGOUTCHK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NARGOUTCHK_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NARGOUTCHK_ERRORS,
};

runmat_thread_local! {
    static CALL_COUNTS: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

pub struct ArityCallCountsGuard {
    previous: Vec<(usize, usize)>,
}

impl Drop for ArityCallCountsGuard {
    fn drop(&mut self) {
        let previous = std::mem::take(&mut self.previous);
        CALL_COUNTS.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

pub fn replace_call_counts(call_counts: Vec<(usize, usize)>) -> ArityCallCountsGuard {
    let previous = CALL_COUNTS.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), call_counts));
    ArityCallCountsGuard { previous }
}

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
    let number = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        _ => return Err(descriptor_error(builtin, error)),
    };

    if !number.is_finite() || number < 0.0 || number.fract() != 0.0 || number > usize::MAX as f64 {
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
            if tensor.data.len() == 1
                && tensor.data[0].is_infinite()
                && tensor.data[0].is_sign_positive() =>
        {
            Ok(ArityBound::Unbounded)
        }
        _ => parse_finite_arity_bound(value, builtin, error).map(ArityBound::Finite),
    }
}

fn validate_bounds(
    args: &[Value],
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

fn validate_arg_count(
    args: &[Value],
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

pub(crate) fn dispatch_narginchk(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let builtin = NARGINCHK_BUILTIN_NAME;
    validate_arg_count(
        &args,
        builtin,
        &NARGINCHK_ERROR_NOT_ENOUGH_INPUTS,
        &NARGINCHK_ERROR_TOO_MANY_INPUTS,
    )?;
    let (actual_inputs, _) = CALL_COUNTS
        .with(|slot| slot.borrow().last().copied())
        .ok_or_else(|| descriptor_error(builtin, &NARGINCHK_ERROR_CONTEXT_UNAVAILABLE))?;
    let (min, max) = validate_bounds(
        &args,
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
    Ok(Value::Num(0.0))
}

pub(crate) fn dispatch_nargoutchk(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let builtin = NARGOUTCHK_BUILTIN_NAME;
    validate_arg_count(
        &args,
        builtin,
        &NARGOUTCHK_ERROR_NOT_ENOUGH_OUTPUTS,
        &NARGOUTCHK_ERROR_TOO_MANY_OUTPUTS,
    )?;
    let (_, actual_outputs) = CALL_COUNTS
        .with(|slot| slot.borrow().last().copied())
        .ok_or_else(|| descriptor_error(builtin, &NARGOUTCHK_ERROR_CONTEXT_UNAVAILABLE))?;
    let (min, max) = validate_bounds(
        &args,
        builtin,
        &NARGOUTCHK_ERROR_ARGUMENT_INVALID,
        &NARGOUTCHK_ERROR_BOUNDS_INVALID,
    )?;
    if actual_outputs < min {
        return Err(descriptor_error(
            builtin,
            &NARGOUTCHK_ERROR_NOT_ENOUGH_OUTPUTS,
        ));
    }
    if !max.permits(actual_outputs) {
        return Err(descriptor_error(
            builtin,
            &NARGOUTCHK_ERROR_TOO_MANY_OUTPUTS,
        ));
    }
    Ok(Value::Num(0.0))
}

#[runmat_macros::runtime_builtin(
    name = "narginchk",
    category = "introspection",
    summary = "Validate current function input arity.",
    sink = true,
    suppress_auto_output = true,
    descriptor(self::NARGINCHK_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::arity_check"
)]
pub fn narginchk_builtin_registered(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    dispatch_narginchk(args)
}

#[runmat_macros::runtime_builtin(
    name = "nargoutchk",
    category = "introspection",
    summary = "Validate current function output arity.",
    sink = true,
    suppress_auto_output = true,
    descriptor(self::NARGOUTCHK_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::arity_check"
)]
pub fn nargoutchk_builtin_registered(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    dispatch_nargoutchk(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narginchk_uses_runtime_call_count_context() {
        let _guard = replace_call_counts(vec![(2, 1)]);
        let value =
            dispatch_narginchk(vec![Value::Num(1.0), Value::Num(2.0)]).expect("narginchk succeeds");
        assert_eq!(value, Value::Num(0.0));
    }

    #[test]
    fn nargoutchk_uses_runtime_call_count_context() {
        let _guard = replace_call_counts(vec![(2, 1)]);
        let value = dispatch_nargoutchk(vec![Value::Num(1.0), Value::Num(1.0)])
            .expect("nargoutchk succeeds");
        assert_eq!(value, Value::Num(0.0));
    }

    #[test]
    fn runtime_arity_helpers_report_context_unavailable_without_vm_context() {
        let err = dispatch_narginchk(vec![Value::Num(0.0), Value::Num(1.0)])
            .expect_err("missing context should fail");
        assert_eq!(err.identifier(), Some("RunMat:NarginchkContextUnavailable"));

        let err = dispatch_nargoutchk(vec![Value::Num(0.0), Value::Num(1.0)])
            .expect_err("missing context should fail");
        assert_eq!(
            err.identifier(),
            Some("RunMat:NargoutchkContextUnavailable")
        );
    }
}
