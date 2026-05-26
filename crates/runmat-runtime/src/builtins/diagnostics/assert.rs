//! MATLAB-compatible `assert` builtin that mirrors MATLAB diagnostic semantics.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::format::format_variadic;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::diagnostics::type_resolvers::assert_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "assert";
const DEFAULT_IDENTIFIER: &str = "RunMat:assertion:failed";
const DEFAULT_MESSAGE: &str = "Assertion failed.";

const ASSERT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero when the assertion passes.",
}];

const ASSERT_INPUTS_CONDITION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "condition",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical/numeric condition that must evaluate to true.",
}];

const ASSERT_INPUTS_MESSAGE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "condition",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical/numeric condition that must evaluate to true.",
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
        description: "Logical/numeric condition that must evaluate to true.",
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
        description: "Logical/numeric condition that must evaluate to true.",
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
        description: "Logical/numeric condition that must evaluate to true.",
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
        label: "out = assert(condition)",
        inputs: &ASSERT_INPUTS_CONDITION,
        outputs: &ASSERT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = assert(condition, message)",
        inputs: &ASSERT_INPUTS_MESSAGE,
        outputs: &ASSERT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = assert(condition, message, A...)",
        inputs: &ASSERT_INPUTS_MESSAGE_VARIADIC,
        outputs: &ASSERT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = assert(condition, message_id, message)",
        inputs: &ASSERT_INPUTS_IDENTIFIER_MESSAGE,
        outputs: &ASSERT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = assert(condition, message_id, message, A...)",
        inputs: &ASSERT_INPUTS_IDENTIFIER_MESSAGE_VARIADIC,
        outputs: &ASSERT_OUTPUT,
    },
];

const ASSERT_ERROR_ASSERTION_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASSERT.ASSERTION_FAILED",
    identifier: Some(DEFAULT_IDENTIFIER),
    when: "Condition evaluates to false and no custom identifier/message override is provided.",
    message: DEFAULT_MESSAGE,
};

const ASSERT_ERROR_INVALID_CONDITION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASSERT.INVALID_CONDITION",
    identifier: Some("RunMat:assertion:invalidCondition"),
    when: "First argument is not a supported logical or numeric condition input.",
    message: "assert: first input must be logical or numeric.",
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

const ASSERT_ERRORS: [BuiltinErrorDescriptor; 4] = [
    ASSERT_ERROR_ASSERTION_FAILED,
    ASSERT_ERROR_INVALID_CONDITION,
    ASSERT_ERROR_INVALID_INPUT,
    ASSERT_ERROR_NOT_ENOUGH_INPUTS,
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
    summary = "Throw a MATLAB-style error when a logical or numeric condition evaluates to false.",
    keywords = "assert,diagnostics,validation,error",
    accel = "metadata",
    type_resolver(assert_type),
    descriptor(crate::builtins::diagnostics::assert::ASSERT_DESCRIPTOR),
    builtin_path = "crate::builtins::diagnostics::assert"
)]
async fn assert_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
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
            let payload = failure_payload(&rest)?;
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
            if int_value.to_i64() != 0 {
                Ok(ConditionOutcome::Pass)
            } else {
                Ok(ConditionOutcome::Fail)
            }
        }
        Value::Num(num) => {
            if num.is_nan() || num == 0.0 {
                Ok(ConditionOutcome::Fail)
            } else {
                Ok(ConditionOutcome::Pass)
            }
        }
        Value::Complex(re, im) => {
            if complex_element_passes(re, im) {
                Ok(ConditionOutcome::Pass)
            } else {
                Ok(ConditionOutcome::Fail)
            }
        }
        Value::LogicalArray(array) => {
            if array.data.iter().all(|&bit| bit != 0) {
                Ok(ConditionOutcome::Pass)
            } else {
                Ok(ConditionOutcome::Fail)
            }
        }
        Value::Tensor(tensor) => evaluate_tensor_condition(&tensor),
        Value::ComplexTensor(tensor) => evaluate_complex_tensor(&tensor),
        Value::GpuTensor(_) => {
            unreachable!("gpu tensors are gathered in normalize_condition_value")
        }
        _ => Err(assert_error(&ASSERT_ERROR_INVALID_CONDITION)),
    }
}

fn evaluate_tensor_condition(tensor: &Tensor) -> crate::BuiltinResult<ConditionOutcome> {
    if tensor.data.is_empty() {
        return Ok(ConditionOutcome::Pass);
    }
    for value in &tensor.data {
        if value.is_nan() || *value == 0.0 {
            return Ok(ConditionOutcome::Fail);
        }
    }
    Ok(ConditionOutcome::Pass)
}

fn evaluate_complex_tensor(tensor: &ComplexTensor) -> crate::BuiltinResult<ConditionOutcome> {
    if tensor.data.is_empty() {
        return Ok(ConditionOutcome::Pass);
    }
    for &(re, im) in &tensor.data {
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

fn failure_payload(args: &[Value]) -> crate::BuiltinResult<FailurePayload> {
    if args.is_empty() {
        return Ok(FailurePayload {
            identifier: DEFAULT_IDENTIFIER.to_string(),
            message: DEFAULT_MESSAGE.to_string(),
        });
    }

    let candidate = &args[0];
    let treat_as_identifier = args.len() >= 2 && value_is_identifier(candidate);

    if treat_as_identifier {
        if args.len() < 2 {
            return Err(assert_flow(
                ASSERT_ERROR_INVALID_INPUT
                    .identifier
                    .unwrap_or(DEFAULT_IDENTIFIER),
                "assert: message text must follow the message identifier.",
            ));
        }
        let identifier = identifier_from_value(candidate)?;
        let template = message_from_value(&args[1])?;
        let formatting_args: &[Value] = if args.len() > 2 { &args[2..] } else { &[] };
        let message = format_message(&template, formatting_args)?;
        Ok(FailurePayload {
            identifier,
            message,
        })
    } else {
        let template = message_from_value(candidate)?;
        let formatting_args: &[Value] = if args.len() > 1 { &args[1..] } else { &[] };
        let message = format_message(&template, formatting_args)?;
        Ok(FailurePayload {
            identifier: DEFAULT_IDENTIFIER.to_string(),
            message,
        })
    }
}

fn value_is_identifier(value: &Value) -> bool {
    if let Some(text) = string_scalar_opt(value) {
        is_message_identifier(&text) || looks_like_unqualified_identifier(&text)
    } else {
        false
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
                .unwrap_or(DEFAULT_IDENTIFIER),
            "assert: message identifier must be nonempty.",
        ));
    }
    Ok(normalize_identifier(&text))
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
        DEFAULT_IDENTIFIER.to_string()
    } else if trimmed.contains(':') {
        trimmed.to_string()
    } else {
        format!("RunMat:{trimmed}")
    }
}

fn is_message_identifier(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || !trimmed.contains(':') {
        return false;
    }
    trimmed
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, ':' | '_' | '.'))
}

fn looks_like_unqualified_identifier(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.contains(char::is_whitespace) {
        return false;
    }
    trimmed
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.'))
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
    use runmat_builtins::{ComplexTensor, IntValue, LogicalArray, ResolveContext, Tensor, Type};

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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_empty_tensor_passes() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        assert_builtin(vec![Value::Tensor(tensor)]).expect("assert should pass");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_empty_logical_passes() {
        let logical = LogicalArray::new(Vec::new(), vec![0]).unwrap();
        assert_builtin(vec![Value::LogicalArray(logical)]).expect("assert should pass");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_false_uses_default_message() {
        let err =
            unwrap_error(assert_builtin(vec![Value::Bool(false)]).expect_err("assert should fail"));
        assert_eq!(err.identifier(), Some(DEFAULT_IDENTIFIER));
        assert_eq!(err.message(), DEFAULT_MESSAGE);
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
        assert_eq!(err.identifier(), Some(DEFAULT_IDENTIFIER));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_detects_nan() {
        let err = unwrap_error(
            assert_builtin(vec![Value::Num(f64::NAN)]).expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some(DEFAULT_IDENTIFIER));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_complex_scalar_passes() {
        assert_builtin(vec![Value::Complex(0.0, 2.0)]).expect("assert should pass");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_complex_scalar_failure() {
        let err = unwrap_error(
            assert_builtin(vec![Value::Complex(0.0, 0.0)]).expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some(DEFAULT_IDENTIFIER));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_complex_tensor_failure() {
        let tensor = ComplexTensor::new(vec![(1.0, 0.0), (0.0, 0.0)], vec![2, 1]).expect("tensor");
        let err = unwrap_error(
            assert_builtin(vec![Value::ComplexTensor(tensor)]).expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some(DEFAULT_IDENTIFIER));
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
        assert_eq!(err.identifier(), Some(DEFAULT_IDENTIFIER));
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
        assert_eq!(err.identifier(), Some(DEFAULT_IDENTIFIER));
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
                data: &tensor.data,
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
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let err =
                unwrap_error(assert_builtin(vec![Value::GpuTensor(handle)]).expect_err("assert"));
            assert_eq!(err.identifier(), Some(DEFAULT_IDENTIFIER));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_logical_array_failure() {
        let logical = LogicalArray::new(vec![1, 0], vec![2]).unwrap();
        let err = unwrap_error(
            assert_builtin(vec![Value::LogicalArray(logical)]).expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some(DEFAULT_IDENTIFIER));
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
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let err = unwrap_error(
            assert_builtin(vec![Value::GpuTensor(handle)]).expect_err("assert should fail"),
        );
        assert_eq!(err.identifier(), Some(DEFAULT_IDENTIFIER));
    }

    #[test]
    fn assert_type_is_numeric() {
        assert_eq!(
            assert_type(&[Type::Bool], &ResolveContext::new(Vec::new())),
            Type::Num
        );
    }
}
