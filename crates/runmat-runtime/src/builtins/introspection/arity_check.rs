use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CharArray, StructValue, Tensor, Value,
};
use runmat_hir::{NARGINCHK_BUILTIN_NAME, NARGOUTCHK_BUILTIN_NAME};
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;

use crate::builtins::common::tensor;

const NO_OUTPUTS: [BuiltinParamDescriptor; 0] = [];

const MESSAGE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "msg",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Legacy message text or structure.",
}];

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

const NARGOUTCHK_LEGACY_INPUTS: [BuiltinParamDescriptor; 3] = [
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
    BuiltinParamDescriptor {
        name: "numArgs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Legacy explicit output count to validate.",
    },
];

const NARGOUTCHK_LEGACY_STRUCT_INPUTS: [BuiltinParamDescriptor; 4] = [
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
    BuiltinParamDescriptor {
        name: "numArgs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Legacy explicit output count to validate.",
    },
    BuiltinParamDescriptor {
        name: "outputType",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("struct"),
        description: "Literal struct output selector.",
    },
];

const NARGOUTCHK_ALL_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "nargoutchk(minArgs, maxArgs)",
        inputs: &ARITY_CHECK_INPUTS,
        outputs: &NO_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "msgText = nargoutchk(minArgs, maxArgs, numArgs)",
        inputs: &NARGOUTCHK_LEGACY_INPUTS,
        outputs: &MESSAGE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "msgStruct = nargoutchk(minArgs, maxArgs, numArgs, 'struct')",
        inputs: &NARGOUTCHK_LEGACY_STRUCT_INPUTS,
        outputs: &MESSAGE_OUTPUT,
    },
];

const NARGINCHK_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "minArgs", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "All eight integer classes are explicitly documented and parsed exactly as nonnegative scalar bounds." },
    BuiltinIntegerInputCapability { name: "maxArgs", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "All eight integer classes are explicitly documented; positive floating infinity remains the documented unbounded maximum." },
];
pub const NARGINCHK_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "narginchk(integer_minArgs, integer_maxArgs)", inputs: &NARGINCHK_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "Bounds are compared exactly with the current host call-context count; interactive resident bounds are not documented and reject without provider access." }];

const NARGOUTCHK_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability { name: "minArgs", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "All eight integer classes are explicitly documented and parsed exactly as nonnegative scalar bounds." },
    BuiltinIntegerInputCapability { name: "maxArgs", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "All eight integer classes are explicitly documented; positive floating infinity remains the documented unbounded maximum." },
    BuiltinIntegerInputCapability { name: "numArgs", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The legacy explicit output count accepts all eight documented integer classes and is evaluated without floating conversion." },
];
pub const NARGOUTCHK_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "nargoutchk(integer_minArgs, integer_maxArgs)", inputs: &NARGINCHK_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "Bounds are compared exactly with the current host call-context output count; interactive resident bounds reject without provider access." },
    BuiltinIntegerCapabilityDescriptor { form: "msg = nargoutchk(integer_minArgs, integer_maxArgs, integer_numArgs, 'struct'?)", inputs: &NARGOUTCHK_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "The discouraged legacy form compares three authoritative integer scalars and returns message text, a message structure, or the documented empty result without changing integer payloads." },
];

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
    signatures: &NARGOUTCHK_ALL_SIGNATURES,
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

pub(crate) fn current_input_count() -> Option<usize> {
    CALL_COUNTS.with(|slot| slot.borrow().last().map(|(inputs, _)| *inputs))
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
        Value::String(value) => Some(value.clone()),
        Value::StringArray(value) if value.data.len() == 1 => Some(value.data[0].clone()),
        Value::CharArray(value) if value.rows == 1 => Some(value.data.iter().collect()),
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
                CharArray::new(Vec::new(), 0, 0).expect("empty legacy nargoutchk result"),
            )
        }
    };
    if structured {
        let mut value = StructValue::new();
        value.insert("message", Value::CharArray(CharArray::new_row(message)));
        value.insert(
            "identifier",
            Value::CharArray(CharArray::new_row(identifier)),
        );
        Value::Struct(value)
    } else {
        Value::CharArray(CharArray::new_row(message))
    }
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

pub fn dispatch_nargoutchk(args: Vec<Value>) -> crate::BuiltinResult<Value> {
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
    let (min, max) = validate_bounds(
        &args,
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
        return Ok(legacy_nargout_result(
            arity_violation(actual_outputs, min, max),
            structured,
        ));
    }
    let (_, actual_outputs) = CALL_COUNTS
        .with(|slot| slot.borrow().last().copied())
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
    Ok(Value::Num(0.0))
}

#[runmat_macros::runtime_builtin(
    name = "narginchk",
    category = "introspection",
    summary = "Validate current function input arity.",
    sink = true,
    suppress_auto_output = true,
    descriptor(self::NARGINCHK_DESCRIPTOR),
    integer_capabilities(self::NARGINCHK_INTEGER_CAPABILITIES),
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
    integer_capabilities(self::NARGOUTCHK_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::introspection::arity_check"
)]
pub fn nargoutchk_builtin_registered(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    dispatch_nargoutchk(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntValue, IntegerStorage, Tensor};

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

    #[test]
    fn arity_bounds_read_typed_integer_tensor_storage_exactly() {
        let _guard = replace_call_counts(vec![(2, 1)]);
        let min =
            Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).expect("integer min");
        let max =
            Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).expect("integer max");

        let value = dispatch_narginchk(vec![Value::Tensor(min), Value::Tensor(max)])
            .expect("narginchk succeeds");
        assert_eq!(value, Value::Num(0.0));

        assert_eq!(
            parse_finite_arity_bound(
                &Value::Int(IntValue::U16(2)),
                "narginchk",
                &NARGINCHK_ERROR_BOUNDS_INVALID
            )
            .unwrap(),
            2
        );
        for value in [
            Value::Int(IntValue::I8(-1)),
            Value::Num(1.5),
            Value::Num(usize::MAX as f64),
            Value::Num(usize::MAX as f64 + 1.0),
        ] {
            assert!(
                parse_finite_arity_bound(&value, "narginchk", &NARGINCHK_ERROR_BOUNDS_INVALID)
                    .is_err()
            );
        }
    }

    #[test]
    fn arity_checks_accept_every_documented_integer_class() {
        let _guard = replace_call_counts(vec![(1, 1)]);
        let cases = [
            (IntValue::I8(0), IntValue::I8(2), IntValue::I8(3)),
            (IntValue::I16(0), IntValue::I16(2), IntValue::I16(3)),
            (IntValue::I32(0), IntValue::I32(2), IntValue::I32(3)),
            (IntValue::I64(0), IntValue::I64(2), IntValue::I64(3)),
            (IntValue::U8(0), IntValue::U8(2), IntValue::U8(3)),
            (IntValue::U16(0), IntValue::U16(2), IntValue::U16(3)),
            (IntValue::U32(0), IntValue::U32(2), IntValue::U32(3)),
            (IntValue::U64(0), IntValue::U64(2), IntValue::U64(3)),
        ];
        for (min, max, invalid_outputs) in cases {
            assert_eq!(
                dispatch_narginchk(vec![Value::Int(min.clone()), Value::Int(max.clone())])
                    .expect("documented integer narginchk bounds"),
                Value::Num(0.0)
            );
            assert_eq!(
                dispatch_nargoutchk(vec![Value::Int(min.clone()), Value::Int(max.clone())])
                    .expect("documented integer nargoutchk bounds"),
                Value::Num(0.0)
            );
            let text = dispatch_nargoutchk(vec![
                Value::Int(min.clone()),
                Value::Int(max.clone()),
                Value::Int(invalid_outputs.clone()),
            ])
            .expect("legacy text nargoutchk");
            assert!(
                matches!(text, Value::CharArray(array) if array.data.iter().collect::<String>().contains("Too many"))
            );

            let structured = dispatch_nargoutchk(vec![
                Value::Int(min),
                Value::Int(max),
                Value::Int(invalid_outputs),
                Value::from("struct"),
            ])
            .expect("legacy structured nargoutchk");
            assert!(
                matches!(structured, Value::Struct(value) if value.fields.contains_key("message") && value.fields.contains_key("identifier"))
            );
        }
    }

    #[test]
    fn legacy_nargoutchk_returns_documented_empty_result_classes() {
        let text = dispatch_nargoutchk(vec![
            Value::Int(IntValue::U8(0)),
            Value::Int(IntValue::U8(2)),
            Value::Int(IntValue::U8(1)),
        ])
        .expect("valid legacy text form");
        assert!(
            matches!(text, Value::CharArray(array) if array.shape == vec![0, 0] && array.data.is_empty())
        );

        let structured = dispatch_nargoutchk(vec![
            Value::Int(IntValue::U8(0)),
            Value::Int(IntValue::U8(2)),
            Value::Int(IntValue::U8(1)),
            Value::from("struct"),
        ])
        .expect("valid legacy structure form");
        assert!(matches!(structured, Value::Struct(value) if value.fields.is_empty()));
    }

    #[test]
    fn arity_checks_reject_resident_bounds_before_provider_access() {
        let _guard = replace_call_counts(vec![(1, 1)]);
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 9_419_007,
            descriptor: Default::default(),
        });
        let nargin = dispatch_narginchk(vec![resident.clone(), Value::Num(2.0)])
            .expect_err("resident nargin bound must reject");
        assert_eq!(nargin.identifier(), Some("RunMat:NarginchkArgumentInvalid"));
        let nargout = dispatch_nargoutchk(vec![resident, Value::Num(2.0)])
            .expect_err("resident nargout bound must reject");
        assert_eq!(
            nargout.identifier(),
            Some("RunMat:NargoutchkArgumentInvalid")
        );
    }
}
