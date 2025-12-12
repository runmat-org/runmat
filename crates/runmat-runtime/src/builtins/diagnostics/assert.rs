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

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "assert",
        builtin_path = "crate::builtins::diagnostics::assert"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "assert"
category: "diagnostics"
keywords: ["assert", "diagnostics", "validation", "error", "gpu"]
summary: "Throw a MATLAB-style error when a logical or numeric condition evaluates to false."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Conditions are evaluated on the host. GPU tensors are gathered before the logical test, and fall back to the default CPU implementation."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::diagnostics::assert::tests"
  integration: "builtins::diagnostics::assert::tests::assert_gpu_tensor_passes"
---

# What does the `assert` function do in MATLAB / RunMat?
`assert(cond, ...)` aborts execution with a MATLAB-compatible error when `cond` is false or contains any zero/NaN elements. When the condition is true (or an empty array), execution continues with no output. RunMat mirrors MATLAB’s identifier normalisation, message formatting, and argument validation rules.

## How does the `assert` function behave in MATLAB / RunMat?
- The first argument must be logical or numeric (real or complex). Scalars must evaluate to true; arrays must contain only nonzero, non-NaN elements (complex values fail when both real and imaginary parts are zero or contain NaNs). Empty inputs pass automatically.
- `assert(cond)` raises `MATLAB:assertion:failed` with message `Assertion failed.` when `cond` is false.
- `assert(cond, msg, args...)` formats `msg` with `sprintf`-compatible conversions using any additional arguments.
- `assert(cond, id, msg, args...)` uses a custom message identifier (normalised to `MATLAB:*` when missing a namespace) and the formatted message text.
- Arguments are validated strictly: identifiers and message templates must be string scalars or character vectors, and malformed format strings raise `MATLAB:assertion:invalidInput`.
- Conditions supplied as gpuArray values are gathered to host memory prior to evaluation so that MATLAB semantics continue to apply.

## `assert` Function GPU Execution Behaviour
`assert` is a control-flow builtin. RunMat gathers GPU-resident tensors (including logical gpuArrays) to host memory before evaluating the condition. No GPU kernels are launched, and the acceleration provider metadata is marked as a gather-immediately operation so execution always follows the MATLAB-compatible CPU path. Residency metadata is preserved so subsequent statements observe the same values they would have seen without the assertion.

## Examples of using the `assert` function in MATLAB / RunMat

### Checking that all elements are nonzero
```matlab
A = [1 2 3];
assert(all(A));
```
This runs without output because every element of `A` is nonzero.

### Verifying array bounds during development
```matlab
idx = 12;
assert(idx >= 1 && idx <= numel(signal), ...
       "Index %d is outside [1, %d].", idx, numel(signal));
```
If `idx` falls outside the valid range, RunMat throws `MATLAB:assertion:failed` with the formatted bounds message.

### Attaching a custom identifier for tooling
```matlab
assert(det(M) ~= 0, "runmat:demo:singularMatrix", ...
       "Matrix must be nonsingular (determinant is zero).");
```
When the matrix is singular, the assertion fails with identifier `runmat:demo:singularMatrix`, allowing downstream tooling to catch it precisely.

### Guarding GPU computations without manual gathering
```matlab
G = gpuArray(rand(1024, 1));
assert(all(G > 0), "All entries must be positive.");
```
The gpuArray is gathered automatically before evaluation; no manual `gather` call is required.

### Converting NaN checks into assertion failures
```matlab
avg = mean(samples);
assert(~isnan(avg), "Average must be finite.");
```
If `avg` evaluates to `NaN`, RunMat raises an error so the calling code cannot continue with invalid state.

### Ensuring structure fields exist before use
```matlab
assert(isfield(cfg, "rate"), ...
       "runmat:config:missingField", ...
       "Configuration missing required field '%s'.", "rate");
```
Missing fields trigger `runmat:config:missingField`, making it easy to spot configuration mistakes early.

### Detecting invalid enumeration values early
```matlab
valid = ["nearest", "linear", "spline"];
assert(any(mode == valid), ...
       "Invalid interpolation mode '%s'.", mode);
```
Passing an unsupported option raises a descriptive error so callers can correct the mode value.

### Validating dimensions before expensive work
```matlab
assert(size(A, 2) == size(B, 1), ...
       "runmat:demo:dimensionMismatch", ...
       "Inner dimensions must agree (size(A,2)=%d, size(B,1)=%d).", ...
       size(A, 2), size(B, 1));
```
If the dimensions disagree, the assertion stops execution before any costly matrix multiplication is attempted.

## FAQ
1. **What types can I pass as the condition?** Logical scalars/arrays and numeric scalars/arrays are accepted. Character arrays, strings, cells, structs, and complex values raise `MATLAB:assertion:invalidCondition`.
2. **How are NaN values treated?** Any `NaN` element causes the assertion to fail, matching MATLAB’s requirement that all elements are non-NaN and nonzero.
3. **Do empty arrays pass the assertion?** Yes. Empty logical or numeric arrays are treated as true.
4. **Can I omit the namespace in the message identifier?** Yes. RunMat prefixes unqualified identifiers with `MATLAB:` to match MATLAB behaviour.
5. **What happens if my format string is malformed?** The builtin raises `MATLAB:assertion:invalidInput` describing the formatting issue.
6. **Does `assert` run on the GPU?** No. GPU tensors are gathered automatically and evaluated on the CPU to preserve MATLAB semantics.
7. **Can I use strings for messages and identifiers?** Yes. Both character vectors and string scalars are accepted for identifiers and message templates.
8. **What value does `assert` return when the condition is true?** Like MATLAB, `assert` has no meaningful return value. RunMat returns `0.0` internally to satisfy the runtime but nothing is produced in MATLAB code.
9. **How do I disable assertions in production code?** Wrap the condition in an `if` statement controlled by your own flag; MATLAB (and RunMat) always evaluates `assert`.
10. **How do I distinguish assertion failures from other errors?** Provide a custom identifier (for example `runmat:module:assertFailed`) and catch it in a `try`/`catch` block.

## See Also
[error](./error), [warning](./warning), [isnan](./isnan), [sprintf](./sprintf)

## Source & Feedback
- Full source: [`crates/runmat-runtime/src/builtins/diagnostics/assert.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/diagnostics/assert.rs)
- Report issues: https://github.com/runmat-org/runmat/issues/new/choose
"#;

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

    #[test]
    fn assert_true_passes() {
        let result = assert_builtin(vec![Value::Bool(true)]).expect("assert should pass");
        assert_eq!(result, Value::Num(0.0));
    }

    #[test]
    fn assert_empty_tensor_passes() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        assert_builtin(vec![Value::Tensor(tensor)]).expect("assert should pass");
    }

    #[test]
    fn assert_empty_logical_passes() {
        let logical = LogicalArray::new(Vec::new(), vec![0]).unwrap();
        assert_builtin(vec![Value::LogicalArray(logical)]).expect("assert should pass");
    }

    #[test]
    fn assert_false_uses_default_message() {
        let err = assert_builtin(vec![Value::Bool(false)]).expect_err("assert should fail");
        assert!(err.starts_with(DEFAULT_IDENTIFIER));
        assert!(err.contains(DEFAULT_MESSAGE));
    }

    #[test]
    fn assert_handles_numeric_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        assert_builtin(vec![Value::Tensor(tensor)]).expect("assert should pass");
    }

    #[test]
    fn assert_detects_zero_in_tensor() {
        let tensor = Tensor::new(vec![1.0, 0.0, 3.0], vec![3, 1]).unwrap();
        let err = assert_builtin(vec![Value::Tensor(tensor)]).expect_err("assert should fail");
        assert!(err.starts_with(DEFAULT_IDENTIFIER));
    }

    #[test]
    fn assert_detects_nan() {
        let err = assert_builtin(vec![Value::Num(f64::NAN)]).expect_err("assert should fail");
        assert!(err.starts_with(DEFAULT_IDENTIFIER));
    }

    #[test]
    fn assert_complex_scalar_passes() {
        assert_builtin(vec![Value::Complex(0.0, 2.0)]).expect("assert should pass");
    }

    #[test]
    fn assert_complex_scalar_failure() {
        let err = assert_builtin(vec![Value::Complex(0.0, 0.0)]).expect_err("assert should fail");
        assert!(err.starts_with(DEFAULT_IDENTIFIER));
    }

    #[test]
    fn assert_complex_tensor_failure() {
        let tensor = ComplexTensor::new(vec![(1.0, 0.0), (0.0, 0.0)], vec![2, 1]).expect("tensor");
        let err =
            assert_builtin(vec![Value::ComplexTensor(tensor)]).expect_err("assert should fail");
        assert!(err.starts_with(DEFAULT_IDENTIFIER));
    }

    #[test]
    fn assert_accepts_custom_message() {
        let err = assert_builtin(vec![
            Value::Bool(false),
            Value::from("Vector length must be positive."),
        ])
        .expect_err("assert should fail");
        assert!(err.contains("Vector length must be positive."));
    }

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

    #[test]
    fn assert_rejects_invalid_condition_type() {
        let err = assert_builtin(vec![Value::from("invalid")]).expect_err("assert should error");
        assert!(err.starts_with(INVALID_CONDITION_IDENTIFIER));
    }

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

    #[test]
    fn assert_invalid_message_type_errors() {
        let err = assert_builtin(vec![Value::Bool(false), Value::Num(5.0)])
            .expect_err("assert should error");
        assert!(err.starts_with(INVALID_INPUT_IDENTIFIER));
    }

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

    #[test]
    fn assert_logical_array_failure() {
        let logical = LogicalArray::new(vec![1, 0], vec![2]).unwrap();
        let err =
            assert_builtin(vec![Value::LogicalArray(logical)]).expect_err("assert should fail");
        assert!(err.starts_with(DEFAULT_IDENTIFIER));
    }

    #[test]
    fn assert_requires_condition_argument() {
        let err = assert_builtin(Vec::new()).expect_err("assert should error");
        assert!(err.starts_with(MIN_INPUT_IDENTIFIER));
        assert!(err.contains(MIN_INPUT_MESSAGE));
    }

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

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
