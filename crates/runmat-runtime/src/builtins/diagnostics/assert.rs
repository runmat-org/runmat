//! MATLAB-compatible `assert` builtin that mirrors MATLAB diagnostic semantics.

use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::format::format_variadic;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

const DEFAULT_IDENTIFIER: &str = "MATLAB:assertion:failed";
const DEFAULT_MESSAGE: &str = "Assertion failed.";
const INVALID_CONDITION_IDENTIFIER: &str = "MATLAB:assertion:invalidCondition";
const INVALID_INPUT_IDENTIFIER: &str = "MATLAB:assertion:invalidInput";
const MIN_INPUT_IDENTIFIER: &str = "MATLAB:minrhs";
const MIN_INPUT_MESSAGE: &str = "Not enough input arguments.";

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

#[runtime_builtin(
    name = "assert",
    category = "diagnostics",
    summary = "Throw a MATLAB-style error when a logical or numeric condition evaluates to false.",
    keywords = "assert,diagnostics,validation,error",
    accel = "metadata",
    builtin_path = "crate::builtins::diagnostics::assert"
)]
fn assert_builtin(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err(build_error(MIN_INPUT_IDENTIFIER, MIN_INPUT_MESSAGE));
    }

    let mut iter = args.into_iter();
    let condition_raw = iter.next().expect("checked length above");
    let rest: Vec<Value> = iter.collect();

    let condition = normalize_condition_value(condition_raw)?;
    match evaluate_condition(condition)? {
        ConditionOutcome::Pass => Ok(Value::Num(0.0)),
        ConditionOutcome::Fail => {
            let payload = failure_payload(&rest)?;
            Err(build_error(&payload.identifier, &payload.message))
        }
    }
}

fn normalize_condition_value(condition: Value) -> Result<Value, String> {
    match condition {
        Value::GpuTensor(handle) => {
            let gpu_value = Value::GpuTensor(handle);
            gpu_helpers::gather_value(&gpu_value)
                .map_err(|e| build_error(INVALID_INPUT_IDENTIFIER, &format!("assert: {e}")))
        }
        other => Ok(other),
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ConditionOutcome {
    Pass,
    Fail,
}

fn evaluate_condition(value: Value) -> Result<ConditionOutcome, String> {
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
        _ => Err(build_error(
            INVALID_CONDITION_IDENTIFIER,
            "assert: first input must be logical or numeric.",
        )),
    }
}

fn evaluate_tensor_condition(tensor: &Tensor) -> Result<ConditionOutcome, String> {
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

fn evaluate_complex_tensor(tensor: &ComplexTensor) -> Result<ConditionOutcome, String> {
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

fn failure_payload(args: &[Value]) -> Result<FailurePayload, String> {
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
            return Err(build_error(
                INVALID_INPUT_IDENTIFIER,
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

fn identifier_from_value(value: &Value) -> Result<String, String> {
    let text = string_scalar_from_value(
        value,
        "assert: message identifier must be a string scalar or character vector.",
    )?;
    if text.trim().is_empty() {
        return Err(build_error(
            INVALID_INPUT_IDENTIFIER,
            "assert: message identifier must be nonempty.",
        ));
    }
    Ok(normalize_identifier(&text))
}

fn message_from_value(value: &Value) -> Result<String, String> {
    string_scalar_from_value(
        value,
        "assert: message text must be a string scalar or character vector.",
    )
}

fn format_message(template: &str, args: &[Value]) -> Result<String, String> {
    format_variadic(template, args)
        .map_err(|err| build_error(INVALID_INPUT_IDENTIFIER, &format!("assert: {err}")))
}

fn build_error(identifier: &str, message: &str) -> String {
    let ident = normalize_identifier(identifier);
    format!("{ident}: {message}")
}

fn normalize_identifier(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        DEFAULT_IDENTIFIER.to_string()
    } else if trimmed.contains(':') {
        trimmed.to_string()
    } else {
        format!("MATLAB:{trimmed}")
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

fn string_scalar_from_value(value: &Value, context: &str) -> Result<String, String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(char_array) if char_array.rows == 1 => {
            Ok(char_array.data.iter().collect::<String>())
        }
        _ => Err(build_error(INVALID_INPUT_IDENTIFIER, context)),
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
    use runmat_builtins::{ComplexTensor, IntValue, LogicalArray, Tensor};

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
        let err = assert_builtin(vec![Value::Bool(false)]).expect_err("assert should fail");
        assert!(err.starts_with(DEFAULT_IDENTIFIER));
        assert!(err.contains(DEFAULT_MESSAGE));
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
        let err = assert_builtin(vec![Value::Tensor(tensor)]).expect_err("assert should fail");
        assert!(err.starts_with(DEFAULT_IDENTIFIER));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_detects_nan() {
        let err = assert_builtin(vec![Value::Num(f64::NAN)]).expect_err("assert should fail");
        assert!(err.starts_with(DEFAULT_IDENTIFIER));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_complex_scalar_passes() {
        assert_builtin(vec![Value::Complex(0.0, 2.0)]).expect("assert should pass");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_complex_scalar_failure() {
        let err = assert_builtin(vec![Value::Complex(0.0, 0.0)]).expect_err("assert should fail");
        assert!(err.starts_with(DEFAULT_IDENTIFIER));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_complex_tensor_failure() {
        let tensor = ComplexTensor::new(vec![(1.0, 0.0), (0.0, 0.0)], vec![2, 1]).expect("tensor");
        let err =
            assert_builtin(vec![Value::ComplexTensor(tensor)]).expect_err("assert should fail");
        assert!(err.starts_with(DEFAULT_IDENTIFIER));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_accepts_custom_message() {
        let err = assert_builtin(vec![
            Value::Bool(false),
            Value::from("Vector length must be positive."),
        ])
        .expect_err("assert should fail");
        assert!(err.contains("Vector length must be positive."));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_supports_message_formatting() {
        let err = assert_builtin(vec![
            Value::Bool(false),
            Value::from("Expected positive value, got %d."),
            Value::Int(IntValue::I32(-4)),
        ])
        .expect_err("assert should fail");
        assert!(err.contains("Expected positive value, got -4."));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_supports_custom_identifier() {
        let err = assert_builtin(vec![
            Value::Bool(false),
            Value::from("runmat:tests:failed"),
            Value::from("Failure %d occurred."),
            Value::Int(IntValue::I32(3)),
        ])
        .expect_err("assert should fail");
        assert!(err.starts_with("runmat:tests:failed"));
        assert!(err.contains("Failure 3 occurred."));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_unqualified_identifier_prefixed() {
        let err = assert_builtin(vec![
            Value::Bool(false),
            Value::from("customAssertionFailed"),
            Value::from("runtime failure"),
        ])
        .expect_err("assert should fail");
        assert!(err.starts_with("MATLAB:customAssertionFailed"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_rejects_invalid_condition_type() {
        let err = assert_builtin(vec![Value::from("invalid")]).expect_err("assert should error");
        assert!(err.starts_with(INVALID_CONDITION_IDENTIFIER));
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
        let err = assert_builtin(vec![Value::Bool(false), Value::Num(5.0)])
            .expect_err("assert should error");
        assert!(err.starts_with(INVALID_INPUT_IDENTIFIER));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_formatting_error_propagates() {
        let err = assert_builtin(vec![
            Value::Bool(false),
            Value::from("number %d must be > 0"),
        ])
        .expect_err("assert should fail");
        assert!(err.starts_with(INVALID_INPUT_IDENTIFIER));
        assert!(err.contains("sprintf"));
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
            let err = assert_builtin(vec![Value::GpuTensor(handle)]).expect_err("assert");
            assert!(err.starts_with(DEFAULT_IDENTIFIER));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_logical_array_failure() {
        let logical = LogicalArray::new(vec![1, 0], vec![2]).unwrap();
        let err =
            assert_builtin(vec![Value::LogicalArray(logical)]).expect_err("assert should fail");
        assert!(err.starts_with(DEFAULT_IDENTIFIER));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assert_requires_condition_argument() {
        let err = assert_builtin(Vec::new()).expect_err("assert should error");
        assert!(err.starts_with(MIN_INPUT_IDENTIFIER));
        assert!(err.contains(MIN_INPUT_MESSAGE));
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
        let err = assert_builtin(vec![Value::GpuTensor(handle)]).expect_err("assert should fail");
        assert!(err.starts_with(DEFAULT_IDENTIFIER));
    }
}
