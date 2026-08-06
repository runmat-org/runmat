//! MATLAB-compatible `assert` builtin that mirrors MATLAB diagnostic semantics.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, NumericScalar, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::format::{flatten_arguments, format_variadic};
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::diagnostics::type_resolvers::assert_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "assert";

const ASSERT_OUTPUTS: [BuiltinParamDescriptor; 0] = [];

pub(crate) const ASSERT_COMPLEX_CONDITION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "assert-complex-condition",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "assert with a complex condition is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:AssertComplexConditionExtension"),
    };

pub(crate) const ASSERT_UNQUALIFIED_IDENTIFIER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "assert-unqualified-identifier",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "assert with an unqualified custom error identifier is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:AssertUnqualifiedIdentifierExtension"),
    };

pub const ASSERT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    ASSERT_COMPLEX_CONDITION_EXTENSION,
    ASSERT_UNQUALIFIED_IDENTIFIER_EXTENSION,
];

const ASSERT_INTEGER_CONDITION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "cond",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every real integer scalar or array is convertible to logical; the condition passes only when it is nonempty and every element is nonzero.",
    }];

const ASSERT_COMPLEX_INTEGER_CONDITION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "cond",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode additionally accepts paired complex-integer storage and tests each element for a nonzero real or imaginary component.",
    }];

const ASSERT_INTEGER_FORMAT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Formatting replacement values accept numeric scalars and preserve exact integer values through integer and string conversion specifiers.",
    }];

pub const ASSERT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "assert(integer_cond, ...)",
        inputs: &ASSERT_INTEGER_CONDITION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Truth testing reads exact authoritative integer storage. The public builtin has no output; resident conditions gather to the host because assert accepts gpuArray input but does not execute on the GPU.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "assert(complex_integer_cond, ...)",
        inputs: &ASSERT_COMPLEX_INTEGER_CONDITION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Public logical conversion rejects complex numeric values. RunMat mode retains the pre-existing nonzero-component predicate without floating materialization.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "assert(cond, msg, integer_A...)",
        inputs: &ASSERT_INTEGER_FORMAT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Each documented replacement A is a character vector, string scalar, or numeric scalar. Integer scalars remain exact during host formatting, including after resident gather.",
    },
];

const ASSERT_INPUTS_CONDITION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "condition",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical or logically convertible condition that must evaluate to true.",
}];

const ASSERT_INPUTS_MESSAGE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "condition",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical or logically convertible condition that must evaluate to true.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"Assertion failed.\""),
        description: "Failure message text.",
    },
];

const ASSERT_INPUTS_MESSAGE_VARIADIC: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "condition",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical or logically convertible condition that must evaluate to true.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"Assertion failed.\""),
        description: "Failure message template text.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Formatting values for the message template.",
    },
];

const ASSERT_INPUTS_IDENTIFIER_MESSAGE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "condition",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical or logically convertible condition that must evaluate to true.",
    },
    BuiltinParamDescriptor {
        name: "message_id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"RunMat:assertion:failed\""),
        description: "Message identifier.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"Assertion failed.\""),
        description: "Failure message text.",
    },
];

const ASSERT_INPUTS_IDENTIFIER_MESSAGE_VARIADIC: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "condition",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical or logically convertible condition that must evaluate to true.",
    },
    BuiltinParamDescriptor {
        name: "message_id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"RunMat:assertion:failed\""),
        description: "Message identifier.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"Assertion failed.\""),
        description: "Failure message template text.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Formatting values for the message template.",
    },
];

const ASSERT_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "assert(condition)",
        inputs: &ASSERT_INPUTS_CONDITION,
        outputs: &ASSERT_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "assert(condition, message)",
        inputs: &ASSERT_INPUTS_MESSAGE,
        outputs: &ASSERT_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "assert(condition, message, A...)",
        inputs: &ASSERT_INPUTS_MESSAGE_VARIADIC,
        outputs: &ASSERT_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "assert(condition, message_id, message)",
        inputs: &ASSERT_INPUTS_IDENTIFIER_MESSAGE,
        outputs: &ASSERT_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "assert(condition, message_id, message, A...)",
        inputs: &ASSERT_INPUTS_IDENTIFIER_MESSAGE_VARIADIC,
        outputs: &ASSERT_OUTPUTS,
    },
];

const ASSERT_ERROR_ASSERTION_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASSERT.ASSERTION_FAILED",
    identifier: Some("RunMat:assertion:failed"),
    when: "Condition evaluates to false and no custom identifier/message override is provided.",
    message: "Assertion failed.",
};

const ASSERT_ERROR_INVALID_CONDITION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASSERT.INVALID_CONDITION",
    identifier: Some("RunMat:assertion:invalidCondition"),
    when: "First argument is not logical or convertible to a logical condition.",
    message: "assert: first input must be logical or convertible to logical.",
};

const ASSERT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASSERT.INVALID_INPUT",
    identifier: Some("RunMat:assertion:invalidInput"),
    when: "Message identifier/message text or formatting payload is invalid.",
    message: "assert: invalid input argument",
};

const ASSERT_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASSERT.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:minrhs"),
    when: "No condition argument is provided.",
    message: "Not enough input arguments.",
};

const ASSERT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASSERT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:assertion:TooManyOutputs"),
    when: "One or more public outputs are requested from assert.",
    message: "assert: too many output arguments",
};

const ASSERT_ERRORS: [BuiltinErrorDescriptor; 5] = [
    ASSERT_ERROR_ASSERTION_FAILED,
    ASSERT_ERROR_INVALID_CONDITION,
    ASSERT_ERROR_INVALID_INPUT,
    ASSERT_ERROR_NOT_ENOUGH_INPUTS,
    ASSERT_ERROR_TOO_MANY_OUTPUTS,
];

pub const ASSERT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ASSERT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ASSERT_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::diagnostics::assert")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "assert",
    op_kind: GpuOpKind::Custom("control"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Control-flow builtin; GPU tensors are gathered to host memory before evaluation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::diagnostics::assert")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "assert",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Control-flow builtin with no fusion support.",
};

fn assert_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    assert_error_with_message(error.message, error)
}

fn assert_default_identifier() -> &'static str {
    ASSERT_ERROR_ASSERTION_FAILED
        .identifier
        .expect("assert default identifier must be defined")
}

fn assert_default_message() -> &'static str {
    ASSERT_ERROR_ASSERTION_FAILED.message
}

fn assert_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(normalize_identifier(identifier));
    }
    builder.build()
}

fn assert_flow(identifier: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_identifier(normalize_identifier(identifier))
        .build()
}

fn remap_assert_flow<F>(
    err: RuntimeError,
    error: &'static BuiltinErrorDescriptor,
    message: F,
) -> RuntimeError
where
    F: FnOnce(&crate::RuntimeError) -> String,
{
    let mut builder = build_runtime_error(message(&err))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(normalize_identifier(identifier));
    }
    builder.build()
}

#[runtime_builtin(
    name = "assert",
    category = "diagnostics",
    summary = "Throw an error when a condition is false, matching MATLAB assert semantics.",
    keywords = "assert,diagnostics,validation,error",
    accel = "metadata",
    type_resolver(assert_type),
    descriptor(crate::builtins::diagnostics::assert::ASSERT_DESCRIPTOR),
    extensions(crate::builtins::diagnostics::assert::ASSERT_EXTENSIONS),
    integer_capabilities(crate::builtins::diagnostics::assert::ASSERT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::diagnostics::assert"
)]
async fn assert_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 0) {
        return Err(assert_error(&ASSERT_ERROR_TOO_MANY_OUTPUTS));
    }
    if args.is_empty() {
        return Err(assert_error(&ASSERT_ERROR_NOT_ENOUGH_INPUTS));
    }

    let mut iter = args.into_iter();
    let condition_raw = iter.next().expect("checked length above");
    let rest: Vec<Value> = iter.collect();

    let condition = normalize_condition_value(condition_raw).await?;
    match evaluate_condition(condition)? {
        ConditionOutcome::Pass => Ok(Value::Num(0.0)),
        ConditionOutcome::Fail => {
            let payload = failure_payload(&rest).await?;
            Err(assert_flow(&payload.identifier, payload.message))
        }
    }
}

async fn normalize_condition_value(condition: Value) -> crate::BuiltinResult<Value> {
    match condition {
        Value::GpuTensor(handle) => {
            let gpu_value = Value::GpuTensor(handle);
            gpu_helpers::gather_value_async(&gpu_value)
                .await
                .map_err(|flow| {
                    remap_assert_flow(flow, &ASSERT_ERROR_INVALID_INPUT, |err| {
                        format!("assert: {}", err.message())
                    })
                })
        }
        other => Ok(other),
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ConditionOutcome {
    Pass,
    Fail,
}

fn evaluate_condition(value: Value) -> crate::BuiltinResult<ConditionOutcome> {
    match value {
        Value::Bool(flag) => Ok(if flag {
            ConditionOutcome::Pass
        } else {
            ConditionOutcome::Fail
        }),
        Value::Int(int_value) => {
            if !int_value.is_zero() {
                Ok(ConditionOutcome::Pass)
            } else {
                Ok(ConditionOutcome::Fail)
            }
        }
        Value::Num(num) => {
            if num.is_nan() {
                Err(assert_error(&ASSERT_ERROR_INVALID_CONDITION))
            } else if num == 0.0 {
                Ok(ConditionOutcome::Fail)
            } else {
                Ok(ConditionOutcome::Pass)
            }
        }
        Value::Complex(re, im) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ASSERT_COMPLEX_CONDITION_EXTENSION,
                BUILTIN_NAME,
            )?;
            if complex_element_passes(re, im) {
                Ok(ConditionOutcome::Pass)
            } else {
                Ok(ConditionOutcome::Fail)
            }
        }
        Value::LogicalArray(array) => {
            if !array.data.is_empty() && array.data.iter().all(|&bit| bit != 0) {
                Ok(ConditionOutcome::Pass)
            } else {
                Ok(ConditionOutcome::Fail)
            }
        }
        Value::Tensor(tensor) => evaluate_tensor_condition(&tensor),
        Value::ComplexTensor(tensor) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ASSERT_COMPLEX_CONDITION_EXTENSION,
                BUILTIN_NAME,
            )?;
            evaluate_complex_tensor(&tensor)
        }
        Value::CharArray(chars) => {
            if !chars.data.is_empty() && chars.data.iter().all(|character| *character != '\0') {
                Ok(ConditionOutcome::Pass)
            } else {
                Ok(ConditionOutcome::Fail)
            }
        }
        Value::GpuTensor(_) => {
            unreachable!("gpu tensors are gathered in normalize_condition_value")
        }
        _ => Err(assert_error(&ASSERT_ERROR_INVALID_CONDITION)),
    }
}

fn evaluate_tensor_condition(tensor: &Tensor) -> crate::BuiltinResult<ConditionOutcome> {
    if tensor.is_empty() {
        return Ok(ConditionOutcome::Fail);
    }
    for index in 0..tensor.len() {
        match tensor
            .numeric_value_at(index)
            .ok_or_else(|| assert_error(&ASSERT_ERROR_INVALID_CONDITION))?
        {
            NumericScalar::F64(value) => {
                if value.is_nan() {
                    return Err(assert_error(&ASSERT_ERROR_INVALID_CONDITION));
                }
                if value == 0.0 {
                    return Ok(ConditionOutcome::Fail);
                }
            }
            NumericScalar::F32(value) => {
                if value.is_nan() {
                    return Err(assert_error(&ASSERT_ERROR_INVALID_CONDITION));
                }
                if value == 0.0 {
                    return Ok(ConditionOutcome::Fail);
                }
            }
            value => {
                if value
                    .into_int_value()
                    .is_none_or(|integer| integer.is_zero())
                {
                    return Ok(ConditionOutcome::Fail);
                }
            }
        }
    }
    Ok(ConditionOutcome::Pass)
}

fn evaluate_complex_tensor(tensor: &ComplexTensor) -> crate::BuiltinResult<ConditionOutcome> {
    if let Some(storage) = tensor.integer_storage() {
        if storage.is_empty() {
            return Ok(ConditionOutcome::Fail);
        }
        for idx in 0..storage.len() {
            let real = storage.real.value_at(idx);
            let imag = storage.imag.value_at(idx);
            if real.is_none_or(|value| value.is_zero()) && imag.is_none_or(|value| value.is_zero())
            {
                return Ok(ConditionOutcome::Fail);
            }
        }
        return Ok(ConditionOutcome::Pass);
    }

    if tensor.is_empty() {
        return Ok(ConditionOutcome::Fail);
    }
    for index in 0..tensor.len() {
        let (real, imag) = tensor
            .numeric_value_at(index)
            .ok_or_else(|| assert_error(&ASSERT_ERROR_INVALID_CONDITION))?;
        let (re, im) = match (real, imag) {
            (NumericScalar::F64(re), NumericScalar::F64(im)) => (re, im),
            (NumericScalar::F32(re), NumericScalar::F32(im)) => (f64::from(re), f64::from(im)),
            _ => return Err(assert_error(&ASSERT_ERROR_INVALID_CONDITION)),
        };
        if !complex_element_passes(re, im) {
            return Ok(ConditionOutcome::Fail);
        }
    }
    Ok(ConditionOutcome::Pass)
}

fn complex_element_passes(re: f64, im: f64) -> bool {
    if re.is_nan() || im.is_nan() {
        return false;
    }
    re != 0.0 || im != 0.0
}

struct FailurePayload {
    identifier: String,
    message: String,
}

async fn failure_payload(args: &[Value]) -> crate::BuiltinResult<FailurePayload> {
    if args.is_empty() {
        return Ok(FailurePayload {
            identifier: assert_default_identifier().to_string(),
            message: assert_default_message().to_string(),
        });
    }

    let candidate = &args[0];
    let treat_as_identifier = args.len() >= 2 && value_is_identifier(candidate)?;

    if treat_as_identifier {
        if args.len() < 2 {
            return Err(assert_flow(
                ASSERT_ERROR_INVALID_INPUT
                    .identifier
                    .expect("assert invalid-input identifier must be defined"),
                "assert: message text must follow the message identifier.",
            ));
        }
        let identifier = identifier_from_value(candidate)?;
        let template = message_from_value(&args[1])?;
        let formatting_args = normalize_formatting_arguments(&args[2..]).await?;
        let message = format_message(&template, &formatting_args)?;
        Ok(FailurePayload {
            identifier,
            message,
        })
    } else {
        let template = message_from_value(candidate)?;
        let formatting_args = normalize_formatting_arguments(&args[1..]).await?;
        let message = format_message(&template, &formatting_args)?;
        Ok(FailurePayload {
            identifier: assert_default_identifier().to_string(),
            message,
        })
    }
}

async fn normalize_formatting_arguments(args: &[Value]) -> crate::BuiltinResult<Vec<Value>> {
    let mut normalized = Vec::with_capacity(args.len());
    for value in args {
        let mut flattened = flatten_arguments(std::slice::from_ref(value), BUILTIN_NAME)
            .await
            .map_err(|flow| {
                remap_assert_flow(flow, &ASSERT_ERROR_INVALID_INPUT, |err| {
                    format!("assert: {}", err.message())
                })
            })?;
        if flattened.len() != 1 {
            return Err(assert_error_with_message(
                "assert: each message replacement value must be a character vector, string scalar, or numeric scalar.",
                &ASSERT_ERROR_INVALID_INPUT,
            ));
        }
        normalized.push(flattened.remove(0));
    }
    Ok(normalized)
}

fn value_is_identifier(value: &Value) -> crate::BuiltinResult<bool> {
    if let Some(text) = string_scalar_opt(value) {
        if text.contains(':') {
            return Ok(true);
        }
        if looks_like_unqualified_identifier(&text)
            && crate::compatibility::runmat_extensions_enabled()
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ASSERT_UNQUALIFIED_IDENTIFIER_EXTENSION,
                BUILTIN_NAME,
            )?;
            return Ok(true);
        }
        Ok(false)
    } else {
        Ok(false)
    }
}

fn identifier_from_value(value: &Value) -> crate::BuiltinResult<String> {
    let text = string_scalar_from_value(
        value,
        "assert: message identifier must be a string scalar or character vector.",
    )?;
    if text.trim().is_empty() {
        return Err(assert_flow(
            ASSERT_ERROR_INVALID_INPUT
                .identifier
                .expect("assert invalid-input identifier must be defined"),
            "assert: message identifier must be nonempty.",
        ));
    }
    let trimmed = text.trim();
    if is_message_identifier(trimmed) {
        return Ok(trimmed.to_string());
    }
    if looks_like_unqualified_identifier(trimmed)
        && crate::compatibility::runmat_extensions_enabled()
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ASSERT_UNQUALIFIED_IDENTIFIER_EXTENSION,
            BUILTIN_NAME,
        )?;
        return Ok(normalize_identifier(trimmed));
    }
    Err(assert_error_with_message(
        "assert: error identifier must contain colon-separated fields that each begin with a letter and otherwise contain only letters, digits, or underscores.",
        &ASSERT_ERROR_INVALID_INPUT,
    ))
}

fn message_from_value(value: &Value) -> crate::BuiltinResult<String> {
    string_scalar_from_value(
        value,
        "assert: message text must be a string scalar or character vector.",
    )
}

fn format_message(template: &str, args: &[Value]) -> crate::BuiltinResult<String> {
    format_variadic(template, args).map_err(|flow| {
        remap_assert_flow(flow, &ASSERT_ERROR_INVALID_INPUT, |err| {
            format!("assert: {}", err.message())
        })
    })
}

fn normalize_identifier(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        assert_default_identifier().to_string()
    } else if trimmed.contains(':') {
        trimmed.to_string()
    } else {
        format!("RunMat:{trimmed}")
    }
}

fn is_message_identifier(text: &str) -> bool {
    let trimmed = text.trim();
    let fields: Vec<&str> = trimmed.split(':').collect();
    if fields.len() < 2 {
        return false;
    }
    fields.into_iter().all(is_identifier_field)
}

fn looks_like_unqualified_identifier(text: &str) -> bool {
    let trimmed = text.trim();
    !trimmed.contains(':') && is_identifier_field(trimmed)
}

fn is_identifier_field(field: &str) -> bool {
    let mut chars = field.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic())
        && chars.all(|character| character.is_ascii_alphanumeric() || character == '_')
}

fn string_scalar_from_value(value: &Value, context: &str) -> crate::BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(char_array) if char_array.rows == 1 => {
            Ok(char_array.data.iter().collect::<String>())
        }
        _ => Err(assert_error_with_message(
            context,
            &ASSERT_ERROR_INVALID_INPUT,
        )),
    }
}

fn string_scalar_opt(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(char_array) if char_array.rows == 1 => {
            Some(char_array.data.iter().collect())
        }
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{
        ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray,
        ResolveContext, Tensor, Type,
    };

    fn assert_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::assert_builtin(args))
    }

    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_true_passes() {
        let result = assert_builtin(vec![Value::Bool(true)]).expect("assert should pass");
        assert_eq!(result, Value::Num(0.0));
    }

    #[test]
    fn assert_scalar_wide_uint64_passes() {
        let result =
            assert_builtin(vec![Value::Int(IntValue::U64(u64::MAX))]).expect("assert should pass");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_empty_tensor_fails() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let err =
            assert_builtin(vec![Value::Tensor(tensor)]).expect_err("empty condition should fail");
        assert_eq!(err.identifier(), Some(assert_default_identifier()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_empty_logical_fails() {
        let logical = LogicalArray::new(Vec::new(), vec![0]).unwrap();
        let err = assert_builtin(vec![Value::LogicalArray(logical)])
            .expect_err("empty condition should fail");
        assert_eq!(err.identifier(), Some(assert_default_identifier()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_false_uses_default_message() {
        let err =
            unwrap_error(assert_builtin(vec![Value::Bool(false)]).expect_err("assert should fail"));
        assert_eq!(err.identifier(), Some(assert_default_identifier()));
        assert_eq!(err.message(), assert_default_message());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_handles_numeric_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        assert_builtin(vec![Value::Tensor(tensor)]).expect("assert should pass");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_detects_zero_in_tensor() {
        let tensor = Tensor::new(vec![1.0, 0.0, 3.0], vec![3, 1]).unwrap();
        let err = unwrap_error(
            assert_builtin(vec![Value::Tensor(tensor)]).expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some(assert_default_identifier()));
    }

    #[test]
    fn assert_reads_typed_integer_tensor_storage_exactly() {
        let passing =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1]), vec![2, 1]).unwrap();
        assert_builtin(vec![Value::Tensor(passing)]).expect("assert should pass");

        let failing =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 0]), vec![2, 1]).unwrap();
        let err = unwrap_error(
            assert_builtin(vec![Value::Tensor(failing)]).expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some(assert_default_identifier()));
    }

    #[test]
    fn assert_tests_every_real_integer_class_exactly() {
        for (passing, failing) in [
            (
                IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
                IntegerStorage::I8(vec![i8::MIN, 0]),
            ),
            (
                IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
                IntegerStorage::I16(vec![i16::MIN, 0]),
            ),
            (
                IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
                IntegerStorage::I32(vec![i32::MIN, 0]),
            ),
            (
                IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                IntegerStorage::I64(vec![i64::MIN, 0]),
            ),
            (
                IntegerStorage::U8(vec![1, u8::MAX]),
                IntegerStorage::U8(vec![u8::MAX, 0]),
            ),
            (
                IntegerStorage::U16(vec![1, u16::MAX]),
                IntegerStorage::U16(vec![u16::MAX, 0]),
            ),
            (
                IntegerStorage::U32(vec![1, u32::MAX]),
                IntegerStorage::U32(vec![u32::MAX, 0]),
            ),
            (
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                IntegerStorage::U64(vec![u64::MAX, 0]),
            ),
        ] {
            assert_builtin(vec![Value::Tensor(
                Tensor::new_integer(passing, vec![2, 1]).expect("passing integer condition"),
            )])
            .expect("all nonzero integers pass");
            let err = assert_builtin(vec![Value::Tensor(
                Tensor::new_integer(failing, vec![2, 1]).expect("failing integer condition"),
            )])
            .expect_err("zero integer fails");
            assert_eq!(err.identifier(), Some(assert_default_identifier()));
        }
    }

    #[test]
    fn assert_formats_every_integer_scalar_class_exactly() {
        for (value, format, expected) in [
            (IntValue::I8(i8::MIN), "%d", i8::MIN.to_string()),
            (IntValue::I16(i16::MIN), "%d", i16::MIN.to_string()),
            (IntValue::I32(i32::MIN), "%d", i32::MIN.to_string()),
            (IntValue::I64(i64::MIN), "%d", i64::MIN.to_string()),
            (IntValue::U8(u8::MAX), "%u", u8::MAX.to_string()),
            (IntValue::U16(u16::MAX), "%u", u16::MAX.to_string()),
            (IntValue::U32(u32::MAX), "%u", u32::MAX.to_string()),
            (IntValue::U64(u64::MAX), "%u", u64::MAX.to_string()),
        ] {
            let err = assert_builtin(vec![
                Value::Bool(false),
                Value::String(format.to_string()),
                Value::Int(value),
            ])
            .expect_err("formatted assertion should fail");
            assert_eq!(err.message(), expected);
        }

        let scalar =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("scalar");
        let err = assert_builtin(vec![
            Value::Bool(false),
            Value::String("%u".to_string()),
            Value::Tensor(scalar),
        ])
        .expect_err("formatted assertion should fail");
        assert_eq!(err.message(), u64::MAX.to_string());

        let nonscalar =
            Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![1, 2]).expect("array");
        let err = assert_builtin(vec![
            Value::Bool(false),
            Value::String("%u".to_string()),
            Value::Tensor(nonscalar),
        ])
        .expect_err("format replacement arrays reject");
        assert_eq!(
            err.identifier(),
            Some(ASSERT_ERROR_INVALID_INPUT.identifier.unwrap())
        );
    }

    #[test]
    fn assert_real_condition_conversion_rejects_nan_and_accepts_character_vectors() {
        let chars = runmat_builtins::CharArray::new(vec!['o', 'k'], 1, 2).expect("chars");
        assert_builtin(vec![Value::CharArray(chars)]).expect("nonzero character codes pass");

        for chars in [
            runmat_builtins::CharArray::new(Vec::new(), 1, 0).expect("empty"),
            runmat_builtins::CharArray::new(vec!['o', '\0'], 1, 2).expect("zero character"),
        ] {
            let err =
                assert_builtin(vec![Value::CharArray(chars)]).expect_err("condition should fail");
            assert_eq!(err.identifier(), Some(assert_default_identifier()));
        }

        for value in [
            Value::Num(f64::NAN),
            Value::Tensor(Tensor::new(vec![1.0, f64::NAN], vec![2, 1]).expect("double")),
            Value::Tensor(Tensor::from_f32(vec![1.0, f32::NAN], vec![2, 1]).expect("single")),
        ] {
            let err = assert_builtin(vec![value]).expect_err("NaN cannot convert to logical");
            assert_eq!(
                err.identifier(),
                Some(ASSERT_ERROR_INVALID_CONDITION.identifier.unwrap())
            );
        }
    }

    #[test]
    fn assert_complex_conditions_are_mode_gated() {
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = assert_builtin(vec![Value::Complex(1.0, 0.0)])
                .expect_err("MATLAB mode rejects complex condition");
            assert_eq!(
                err.identifier(),
                ASSERT_COMPLEX_CONDITION_EXTENSION.error_identifier
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert_builtin(vec![Value::Complex(1.0, 0.0)])
                .expect("RunMat mode admits complex condition");
        }
    }

    #[test]
    fn assert_identifier_grammar_and_unqualified_extension_are_explicit() {
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = assert_builtin(vec![
                Value::Bool(false),
                Value::String("plainMessage".to_string()),
                Value::Int(IntValue::I32(7)),
            ])
            .expect_err("plain text is the message form");
            assert_eq!(err.identifier(), Some(assert_default_identifier()));
            assert_eq!(err.message(), "plainMessage");
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let err = assert_builtin(vec![
                Value::Bool(false),
                Value::String("customFailure".to_string()),
                Value::String("failed".to_string()),
            ])
            .expect_err("RunMat mode admits unqualified identifier");
            assert_eq!(err.identifier(), Some("RunMat:customFailure"));
        }
        for identifier in ["bad.segment:mnemonic", "component:9bad", "component::bad"] {
            let err = assert_builtin(vec![
                Value::Bool(false),
                Value::String(identifier.to_string()),
                Value::String("failed".to_string()),
            ])
            .expect_err("invalid qualified identifier rejects");
            assert_eq!(
                err.identifier(),
                Some(ASSERT_ERROR_INVALID_INPUT.identifier.unwrap())
            );
        }
    }

    #[test]
    fn assert_reads_native_single_tensor_storage() {
        let passing = Tensor::from_f32(vec![f32::MIN_POSITIVE, -2.0], vec![2, 1]).unwrap();
        assert_builtin(vec![Value::Tensor(passing)]).expect("assert should pass");

        let zero = Tensor::from_f32(vec![1.0_f32, 0.0], vec![2, 1]).unwrap();
        let err =
            assert_builtin(vec![Value::Tensor(zero)]).expect_err("zero condition should fail");
        assert_eq!(err.identifier(), Some(assert_default_identifier()));

        let nan = Tensor::from_f32(vec![1.0_f32, f32::NAN], vec![2, 1]).unwrap();
        let err =
            assert_builtin(vec![Value::Tensor(nan)]).expect_err("NaN condition should reject");
        assert_eq!(
            err.identifier(),
            Some(ASSERT_ERROR_INVALID_CONDITION.identifier.unwrap())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_detects_nan() {
        let err = unwrap_error(
            assert_builtin(vec![Value::Num(f64::NAN)]).expect_err("assert should reject NaN"),
        );
        assert_eq!(
            err.identifier(),
            Some(ASSERT_ERROR_INVALID_CONDITION.identifier.unwrap())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_complex_scalar_passes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert_builtin(vec![Value::Complex(0.0, 2.0)]).expect("assert should pass");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_complex_scalar_failure() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = unwrap_error(
            assert_builtin(vec![Value::Complex(0.0, 0.0)]).expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some(assert_default_identifier()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_complex_tensor_failure() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = ComplexTensor::new(vec![(1.0, 0.0), (0.0, 0.0)], vec![2, 1]).expect("tensor");
        let err = unwrap_error(
            assert_builtin(vec![Value::ComplexTensor(tensor)]).expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some(assert_default_identifier()));
    }

    #[test]
    fn assert_reads_typed_complex_integer_tensor_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![0, u64::MAX]),
            IntegerStorage::U64(vec![5, 0]),
        )
        .expect("complex integer storage");
        let passing = ComplexTensor::new_integer(storage, vec![2, 1]).unwrap();
        assert_builtin(vec![Value::ComplexTensor(passing)]).expect("assert should pass");

        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX, 0]),
            IntegerStorage::U64(vec![0, 0]),
        )
        .expect("complex integer storage");
        let failing = ComplexTensor::new_integer(storage, vec![2, 1]).unwrap();
        let err = unwrap_error(
            assert_builtin(vec![Value::ComplexTensor(failing)]).expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some(assert_default_identifier()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_accepts_custom_message() {
        let err = unwrap_error(
            assert_builtin(vec![
                Value::Bool(false),
                Value::from("Vector length must be positive."),
            ])
            .expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some(assert_default_identifier()));
        assert!(err.message().contains("Vector length must be positive."));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_supports_message_formatting() {
        let err = unwrap_error(
            assert_builtin(vec![
                Value::Bool(false),
                Value::from("Expected positive value, got %d."),
                Value::Int(IntValue::I32(-4)),
            ])
            .expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some(assert_default_identifier()));
        assert!(err.message().contains("Expected positive value, got -4."));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_supports_custom_identifier() {
        let err = unwrap_error(
            assert_builtin(vec![
                Value::Bool(false),
                Value::from("runmat:tests:failed"),
                Value::from("Failure %d occurred."),
                Value::Int(IntValue::I32(3)),
            ])
            .expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some("runmat:tests:failed"));
        assert!(err.message().contains("Failure 3 occurred."));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_unqualified_identifier_prefixed() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = unwrap_error(
            assert_builtin(vec![
                Value::Bool(false),
                Value::from("customAssertionFailed"),
                Value::from("runtime failure"),
            ])
            .expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some("RunMat:customAssertionFailed"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_rejects_invalid_condition_type() {
        let err = unwrap_error(
            assert_builtin(vec![Value::from("invalid")]).expect_err("assert should error"),
        );
        assert_eq!(
            err.identifier(),
            Some(ASSERT_ERROR_INVALID_CONDITION.identifier.unwrap())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_gpu_tensor_passes() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = assert_builtin(vec![Value::GpuTensor(handle)]).expect("assert");
            assert_eq!(result, Value::Num(0.0));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_invalid_message_type_errors() {
        let err = unwrap_error(
            assert_builtin(vec![Value::Bool(false), Value::Num(5.0)])
                .expect_err("assert should error"),
        );
        assert_eq!(
            err.identifier(),
            Some(ASSERT_ERROR_INVALID_INPUT.identifier.unwrap())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_formatting_error_propagates() {
        let err = unwrap_error(
            assert_builtin(vec![
                Value::Bool(false),
                Value::from("number %d must be > 0"),
            ])
            .expect_err("assert should fail"),
        );
        assert_eq!(
            err.identifier(),
            Some(ASSERT_ERROR_INVALID_INPUT.identifier.unwrap())
        );
        assert!(err.message().contains("sprintf"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_gpu_tensor_failure() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0, 3.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let err =
                unwrap_error(assert_builtin(vec![Value::GpuTensor(handle)]).expect_err("assert"));
            assert_eq!(err.identifier(), Some(assert_default_identifier()));
        });
    }

    #[test]
    fn assert_provider_gather_tests_every_integer_class_and_formats_wide_scalar() {
        test_support::with_test_provider(|provider| {
            for storage in [
                IntegerStorage::I8(vec![i8::MIN, 0]),
                IntegerStorage::I16(vec![i16::MIN, 0]),
                IntegerStorage::I32(vec![i32::MIN, 0]),
                IntegerStorage::I64(vec![i64::MIN, 0]),
                IntegerStorage::U8(vec![u8::MAX, 0]),
                IntegerStorage::U16(vec![u16::MAX, 0]),
                IntegerStorage::U32(vec![u32::MAX, 0]),
                IntegerStorage::U64(vec![u64::MAX, 0]),
            ] {
                let handle = gpu_helpers::upload_tensor(
                    provider,
                    &Tensor::new_integer(storage, vec![2, 1]).expect("condition"),
                )
                .expect("upload");
                let err = assert_builtin(vec![Value::GpuTensor(handle.clone())])
                    .expect_err("resident zero fails");
                assert_eq!(err.identifier(), Some(assert_default_identifier()));
                let _ = provider.free(&handle);
            }

            let handle = gpu_helpers::upload_tensor(
                provider,
                &Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                    .expect("format value"),
            )
            .expect("upload");
            let err = assert_builtin(vec![
                Value::Bool(false),
                Value::String("%u".to_string()),
                Value::GpuTensor(handle.clone()),
            ])
            .expect_err("resident scalar formats");
            assert_eq!(err.message(), u64::MAX.to_string());
            let _ = provider.free(&handle);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_logical_array_failure() {
        let logical = LogicalArray::new(vec![1, 0], vec![2]).unwrap();
        let err = unwrap_error(
            assert_builtin(vec![Value::LogicalArray(logical)]).expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some(assert_default_identifier()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_requires_condition_argument() {
        let err = unwrap_error(assert_builtin(Vec::new()).expect_err("assert should error"));
        assert_eq!(
            err.identifier(),
            Some(ASSERT_ERROR_NOT_ENOUGH_INPUTS.identifier.unwrap())
        );
        assert_eq!(err.message(), ASSERT_ERROR_NOT_ENOUGH_INPUTS.message);
    }

    #[test]
    fn assert_rejects_requested_public_output() {
        let _outputs = crate::output_count::push_output_count(Some(1));
        let err = assert_builtin(vec![Value::Bool(true)]).expect_err("assert has no output");
        assert_eq!(err.identifier(), ASSERT_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn assert_wgpu_tensor_failure_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        if register_wgpu_provider(WgpuProviderOptions::default()).is_err() {
            return;
        }
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };

        let tensor = Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let err = unwrap_error(
            assert_builtin(vec![Value::GpuTensor(handle)]).expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some(assert_default_identifier()));
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn assert_wgpu_integer_conditions_and_formatting_remain_exact() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let _guard = test_support::accel_test_lock();
        if register_wgpu_provider(WgpuProviderOptions::default()).is_err() {
            return;
        }
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        for storage in [
            IntegerStorage::I8(vec![i8::MIN, 0]),
            IntegerStorage::I16(vec![i16::MIN, 0]),
            IntegerStorage::I32(vec![i32::MIN, 0]),
            IntegerStorage::I64(vec![i64::MIN, 0]),
            IntegerStorage::U8(vec![u8::MAX, 0]),
            IntegerStorage::U16(vec![u16::MAX, 0]),
            IntegerStorage::U32(vec![u32::MAX, 0]),
            IntegerStorage::U64(vec![u64::MAX, 0]),
        ] {
            let handle = gpu_helpers::upload_tensor(
                provider,
                &Tensor::new_integer(storage, vec![2, 1]).expect("condition"),
            )
            .expect("upload");
            let err = assert_builtin(vec![Value::GpuTensor(handle.clone())])
                .expect_err("resident zero fails");
            assert_eq!(err.identifier(), Some(assert_default_identifier()));
            let _ = provider.free(&handle);
        }

        let handle = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                .expect("format value"),
        )
        .expect("upload");
        let err = assert_builtin(vec![
            Value::Bool(false),
            Value::String("%u".to_string()),
            Value::GpuTensor(handle.clone()),
        ])
        .expect_err("resident scalar formats");
        assert_eq!(err.message(), u64::MAX.to_string());
        let _ = provider.free(&handle);
    }

    #[test]
    fn assert_has_no_public_output_type() {
        assert_eq!(
            assert_type(&[Type::Bool], &ResolveContext::new(Vec::new())),
            Type::Unknown
        );
    }

    #[test]
    fn assert_metadata_classifies_integer_and_extension_forms() {
        assert_eq!(ASSERT_INTEGER_CAPABILITIES.len(), 3);
        assert_eq!(
            ASSERT_INTEGER_CAPABILITIES[0].output_class,
            BuiltinIntegerOutputClassRule::NotApplicable
        );
        assert_eq!(
            ASSERT_INTEGER_CAPABILITIES[1].inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
        assert_eq!(
            ASSERT_EXTENSIONS,
            [
                ASSERT_COMPLEX_CONDITION_EXTENSION,
                ASSERT_UNQUALIFIED_IDENTIFIER_EXTENSION
            ]
        );
        assert!(ASSERT_SIGNATURES
            .iter()
            .all(|signature| signature.outputs.is_empty()));
    }
}
