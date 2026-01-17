//! MATLAB-compatible `setenv` builtin for RunMat.

use std::any::Any;
use std::env;
use std::panic;

use runmat_builtins::{CharArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{gather_if_needed, build_runtime_error, BuiltinResult, RuntimeControlFlow};

const ERR_TOO_FEW_INPUTS: &str = "setenv: not enough input arguments";
const ERR_TOO_MANY_INPUTS: &str = "setenv: too many input arguments";
const ERR_NAME_TYPE: &str = "setenv: NAME must be a string scalar or character vector";
const ERR_VALUE_TYPE: &str = "setenv: VALUE must be a string scalar or character vector";

const MESSAGE_EMPTY_NAME: &str = "Environment variable name must not be empty.";
const MESSAGE_NAME_HAS_EQUAL: &str = "Environment variable names must not contain '='.";
const MESSAGE_NAME_HAS_NULL: &str = "Environment variable names must not contain null characters.";
const MESSAGE_VALUE_HAS_NULL: &str =
    "Environment variable values must not contain null characters.";
const MESSAGE_OPERATION_FAILED: &str = "Unable to update environment variable: ";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "setenv",
        builtin_path = "crate::builtins::io::repl_fs::setenv"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "setenv"
category: "io/repl_fs"
keywords: ["setenv", "environment variable", "status", "message", "unset"]
summary: "Set or clear environment variables with MATLAB-compatible status outputs."
references:
  - https://www.mathworks.com/help/matlab/ref/setenv.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Host-only operation. RunMat gathers GPU-resident inputs before mutating the process environment; providers do not expose hooks for this builtin."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::setenv::tests"
  integration: "builtins::io::repl_fs::setenv::tests::setenv_reports_failure_for_illegal_name"
---

# What does the `setenv` function do in MATLAB / RunMat?
`setenv` updates the process environment. Provide a variable name and value to create or modify an
entry, or pass an empty value to remove the variable. The builtin mirrors MATLAB by returning a
status code and optional diagnostic message instead of throwing for platform-defined failures.

## How does the `setenv` function behave in MATLAB / RunMat?
- `status = setenv(name, value)` returns `0` when the update succeeds and `1` when the operating
  system rejects the request. The status output is a double scalar, matching MATLAB.
- `[status, message] = setenv(name, value)` returns the status plus a character vector describing
  failures. On success, `message` is an empty `1×0` character array.
- Set `value` to an empty string (`""`) or empty character vector (`''`) to remove the variable from
  the current process environment.
- Names must be string scalars or character vectors containing only valid environment variable
  characters. MATLAB raises an error when `name` is not text; RunMat mirrors this check.
- Character vector inputs trim trailing padding spaces (common with MATLAB character matrices). To
  retain trailing spaces, pass a string scalar instead.
- Environment updates apply to the RunMat process and any child processes it spawns. They do not
  modify the parent shell.

## `setenv` Function GPU Execution Behaviour
`setenv` always runs on the CPU. If a caller stores the arguments on the GPU—for instance via an
accelerated string builtin—RunMat gathers them to host memory automatically before mutating the
environment. Acceleration providers do not implement hooks for this builtin.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `setenv` is a host-side operation. GPU residency offers no benefit, and RunMat gathers
GPU-backed values automatically if they appear as inputs.

## Examples of using the `setenv` function in MATLAB / RunMat

### Set a new environment variable for the current session
```matlab
status = setenv("RUNMAT_MODE", "development")
```
Expected output:
```matlab
status =
     0
```

### Update an existing environment variable
```matlab
status = setenv("PATH", string(getenv("PATH")) + ":~/runmat/bin")
```
Expected output:
```matlab
status =
     0
```

### Remove an environment variable with an empty value
```matlab
[status, message] = setenv("OLD_SETTING", "")
```
Expected output:
```matlab
status =
     0

message =

```

### Capture diagnostic messages when a name is invalid
```matlab
[status, message] = setenv("INVALID=NAME", "value")
```
Expected output:
```matlab
status =
     1

message =
Environment variable names must not contain '='.
```

### Use character vectors from legacy code
```matlab
status = setenv('RUNMAT_LEGACY', 'enabled')
```
Expected output:
```matlab
status =
     0
```

### Combine `setenv` with child process launches
```matlab
setenv("RUNMAT_DATASET", "demo");
status = system("runmat-cli process-data")
```
Expected output:
```matlab
status =
     0
```

## FAQ
- **What status codes does `setenv` return?** `0` means success; `1` means the operating system
  rejected the request (for example, due to an invalid name or an oversized value on Windows).
- **Does `setenv` throw errors?** Only when the inputs are the wrong type (non-text). Platform
  failures are reported through the status and message outputs so scripts can handle them
  programmatically.
- **How do I remove a variable?** Pass an empty string or empty character vector as the value.
- **Are names case-sensitive?** RunMat defers to the operating system: case-sensitive on
  Unix-like systems and case-insensitive on Windows.
- **Can I include trailing spaces in the value?** Use string scalars to preserve trailing spaces.
  Character vector inputs trim trailing padding spaces by design.
- **Does `setenv` affect the parent shell?** No. Changes are limited to the current RunMat process
  and any child processes launched afterwards.
- **What characters are disallowed in names?** `setenv` rejects names containing `=` or null
  characters. Additional platform-specific restrictions are enforced by the operating system and
  reported through the status/message outputs.
- **Can I call `setenv` from GPU-enabled code?** Yes. Arguments are gathered from the GPU before
  updating the environment; the operation itself always runs on the CPU.
- **How can I check whether the update succeeded?** Inspect the returned `status`. When it is `1`,
  read the accompanying message to determine why the operation failed.
- **Will the variable persist after I exit RunMat?** No. Environment modifications are scoped to the
  current process.

## See Also
[getenv](./getenv), [mkdir](./mkdir), [pwd](./pwd)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/setenv.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/setenv.rs)
- Issues: [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::setenv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "setenv",
    op_kind: GpuOpKind::Custom("io"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Host-only environment mutation. GPU-resident arguments are gathered automatically before invoking the OS APIs.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::setenv")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "setenv",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Environment updates terminate fusion; metadata registered for completeness.",
};

const BUILTIN_NAME: &str = "setenv";

fn setenv_error(message: impl Into<String>) -> RuntimeControlFlow {
    RuntimeControlFlow::Error(
        build_runtime_error(message)
            .with_builtin(BUILTIN_NAME)
            .build(),
    )
}

fn map_control_flow(flow: RuntimeControlFlow) -> RuntimeControlFlow {
    match flow {
        RuntimeControlFlow::Error(err) => {
            let identifier = err.identifier().map(str::to_string);
            let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
                .with_builtin(BUILTIN_NAME)
                .with_source(err);
            if let Some(identifier) = identifier {
                builder = builder.with_identifier(identifier);
            }
            RuntimeControlFlow::Error(builder.build())
        }
    }
}

#[runtime_builtin(
    name = "setenv",
    category = "io/repl_fs",
    summary = "Set or clear environment variables with MATLAB-compatible status outputs.",
    keywords = "setenv,environment variable,status,message,unset",
    accel = "cpu",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::io::repl_fs::setenv"
)]
fn setenv_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&args)?;
    Ok(eval.first_output())
}

/// Evaluate `setenv` once and expose both outputs.
pub fn evaluate(args: &[Value]) -> BuiltinResult<SetenvResult> {
    let gathered = gather_arguments(args)?;
    match gathered.len() {
        0 | 1 => Err(setenv_error(ERR_TOO_FEW_INPUTS)),
        2 => apply(&gathered[0], &gathered[1]),
        _ => Err(setenv_error(ERR_TOO_MANY_INPUTS)),
    }
}

#[derive(Debug, Clone)]
pub struct SetenvResult {
    status: f64,
    message: String,
}

impl SetenvResult {
    fn success() -> Self {
        Self {
            status: 0.0,
            message: String::new(),
        }
    }

    fn failure(message: String) -> Self {
        Self {
            status: 1.0,
            message,
        }
    }

    pub fn first_output(&self) -> Value {
        Value::Num(self.status)
    }

    pub fn outputs(&self) -> Vec<Value> {
        vec![Value::Num(self.status), char_array_value(&self.message)]
    }

    #[cfg(test)]
    pub(crate) fn status(&self) -> f64 {
        self.status
    }

    #[cfg(test)]
    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

fn apply(name_value: &Value, value_value: &Value) -> BuiltinResult<SetenvResult> {
    let name = extract_scalar_text(name_value, ERR_NAME_TYPE)?;
    let value = extract_scalar_text(value_value, ERR_VALUE_TYPE)?;

    if name.is_empty() {
        return Ok(SetenvResult::failure(MESSAGE_EMPTY_NAME.to_string()));
    }
    if name.chars().any(|ch| ch == '=') {
        return Ok(SetenvResult::failure(MESSAGE_NAME_HAS_EQUAL.to_string()));
    }
    if name.chars().any(|ch| ch == '\0') {
        return Ok(SetenvResult::failure(MESSAGE_NAME_HAS_NULL.to_string()));
    }
    if value.chars().any(|ch| ch == '\0') {
        return Ok(SetenvResult::failure(MESSAGE_VALUE_HAS_NULL.to_string()));
    }

    Ok(update_environment(&name, &value))
}

fn update_environment(name: &str, value: &str) -> SetenvResult {
    if value.is_empty() {
        match panic::catch_unwind(|| env::remove_var(name)) {
            Ok(()) => SetenvResult::success(),
            Err(payload) => SetenvResult::failure(format!(
                "{}{}",
                MESSAGE_OPERATION_FAILED,
                panic_payload_to_string(payload)
            )),
        }
    } else {
        match panic::catch_unwind(|| env::set_var(name, value)) {
            Ok(()) => SetenvResult::success(),
            Err(payload) => SetenvResult::failure(format!(
                "{}{}",
                MESSAGE_OPERATION_FAILED,
                panic_payload_to_string(payload)
            )),
        }
    }
}

fn gather_arguments(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(gather_if_needed(value).map_err(map_control_flow)?);
    }
    Ok(out)
}

fn extract_scalar_text(value: &Value, error_message: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) => {
            if array.rows != 1 {
                return Err(setenv_error(error_message));
            }
            Ok(char_row_to_string(array))
        }
        Value::StringArray(array) => {
            if array.data.len() == 1 {
                Ok(array.data[0].clone())
            } else {
                Err(setenv_error(error_message))
            }
        }
        _ => Err(setenv_error(error_message)),
    }
}

fn char_row_to_string(array: &CharArray) -> String {
    if array.cols == 0 {
        return String::new();
    }
    let mut text = String::with_capacity(array.cols);
    for col in 0..array.cols {
        text.push(array.data[col]);
    }
    while text.ends_with(' ') {
        text.pop();
    }
    text
}

fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(msg) => *msg,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(msg) => (*msg).to_string(),
            Err(_) => "operation failed".to_string(),
        },
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;
    use crate::{RuntimeControlFlow, RuntimeError};
    use runmat_builtins::{CharArray, StringArray, Value};

    fn unique_name(suffix: &str) -> String {
        format!("RUNMAT_TEST_SETENV_{}", suffix)
    }

    fn unwrap_error(flow: RuntimeControlFlow) -> RuntimeError {
        match flow {
            RuntimeControlFlow::Error(err) => err,
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_sets_variable_and_returns_success() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let name = unique_name("BASIC");
        env::remove_var(&name);

        let result = setenv_builtin(vec![
            Value::String(name.clone()),
            Value::String("value".to_string()),
        ])
        .expect("setenv");

        assert_eq!(result, Value::Num(0.0));
        assert_eq!(env::var(&name).unwrap(), "value");
        env::remove_var(name);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_removes_variable_when_value_is_empty() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let name = unique_name("REMOVE");
        env::set_var(&name, "seed");

        let result = setenv_builtin(vec![
            Value::String(name.clone()),
            Value::CharArray(CharArray::new_row("")),
        ])
        .expect("setenv");

        assert_eq!(result, Value::Num(0.0));
        assert!(env::var(&name).is_err());
        env::remove_var(name);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_reports_failure_for_illegal_name() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let eval = evaluate(&[
            Value::String("INVALID=NAME".to_string()),
            Value::String("value".to_string()),
        ])
        .expect("evaluate");

        assert_eq!(eval.status(), 1.0);
        assert_eq!(eval.message(), MESSAGE_NAME_HAS_EQUAL);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_reports_failure_for_empty_name() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let eval = evaluate(&[
            Value::String(String::new()),
            Value::String("value".to_string()),
        ])
        .expect("evaluate");

        assert_eq!(eval.status(), 1.0);
        assert_eq!(eval.message(), MESSAGE_EMPTY_NAME);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_reports_failure_for_null_in_name() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let eval = evaluate(&[
            Value::String("BAD\0NAME".to_string()),
            Value::String("value".to_string()),
        ])
        .expect("evaluate");

        assert_eq!(eval.status(), 1.0);
        assert_eq!(eval.message(), MESSAGE_NAME_HAS_NULL);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_reports_failure_for_null_in_value() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let eval = evaluate(&[
            Value::String("RUNMAT_NULL_VALUE".to_string()),
            Value::String("abc\0def".to_string()),
        ])
        .expect("evaluate");

        assert_eq!(eval.status(), 1.0);
        assert_eq!(eval.message(), MESSAGE_VALUE_HAS_NULL);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_errors_when_name_is_not_text() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let err = unwrap_error(
            setenv_builtin(vec![Value::Num(5.0), Value::String("value".to_string())])
                .unwrap_err(),
        );
        assert_eq!(err.message(), ERR_NAME_TYPE);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_errors_when_value_is_not_text() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let err = unwrap_error(
            setenv_builtin(vec![
                Value::String("RUNMAT_INVALID_VALUE".to_string()),
                Value::Num(1.0),
            ])
            .unwrap_err(),
        );
        assert_eq!(err.message(), ERR_VALUE_TYPE);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_accepts_scalar_string_array_arguments() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let name = unique_name("STRING_ARRAY");
        env::remove_var(&name);

        let name_array =
            StringArray::new(vec![name.clone()], vec![1]).expect("scalar string array name");
        let value_array =
            StringArray::new(vec!["VALUE".to_string()], vec![1]).expect("scalar string array");

        let status = setenv_builtin(vec![
            Value::StringArray(name_array),
            Value::StringArray(value_array),
        ])
        .expect("setenv");

        assert_eq!(status, Value::Num(0.0));
        assert_eq!(env::var(&name).unwrap(), "VALUE");
        env::remove_var(name);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_errors_for_string_array_with_multiple_elements() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let array =
            StringArray::new(vec!["A".to_string(), "B".to_string()], vec![2]).expect("array");
        let err = unwrap_error(
            setenv_builtin(vec![
                Value::StringArray(array),
                Value::String("value".to_string()),
            ])
            .unwrap_err(),
        );
        assert_eq!(err.message(), ERR_NAME_TYPE);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_errors_for_char_array_with_multiple_rows() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let array = CharArray::new(vec!['R', 'M'], 2, 1).expect("two-row char array");
        let err = unwrap_error(
            setenv_builtin(vec![
                Value::CharArray(array),
                Value::String("value".to_string()),
            ])
            .unwrap_err(),
        );
        assert_eq!(err.message(), ERR_NAME_TYPE);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_char_array_input_trims_padding() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let chars = vec!['F', 'O', 'O', ' '];
        let array = CharArray::new(chars, 1, 4).unwrap();
        let result = extract_scalar_text(&Value::CharArray(array), ERR_NAME_TYPE).unwrap();
        assert_eq!(result, "FOO");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_outputs_success_message_is_empty_char_array() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let name = unique_name("SUCCESS_MSG");
        env::remove_var(&name);

        let eval = evaluate(&[
            Value::String(name.clone()),
            Value::String("value".to_string()),
        ])
        .expect("evaluate");
        let outputs = eval.outputs();
        assert_eq!(outputs.len(), 2);
        match &outputs[1] {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 0);
            }
            other => panic!("expected empty CharArray, got {other:?}"),
        }

        env::remove_var(name);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_outputs_return_status_and_message() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let eval = evaluate(&[
            Value::String("INVALID=NAME".to_string()),
            Value::String("value".to_string()),
        ])
        .expect("evaluate");

        let outputs = eval.outputs();
        assert_eq!(outputs.len(), 2);
        assert!(matches!(outputs[0], Value::Num(1.0)));
        match &outputs[1] {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                let text: String = ca.data.iter().collect();
                assert_eq!(text, MESSAGE_NAME_HAS_EQUAL);
            }
            other => panic!("expected CharArray message, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
