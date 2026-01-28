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
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

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

fn setenv_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
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
async fn setenv_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&args).await?;
    Ok(eval.first_output())
}

/// Evaluate `setenv` once and expose both outputs.
pub async fn evaluate(args: &[Value]) -> BuiltinResult<SetenvResult> {
    let gathered = gather_arguments(args).await?;
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

async fn gather_arguments(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
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
    use runmat_builtins::{CharArray, StringArray, Value};

    fn setenv_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::setenv_builtin(args))
    }

    fn evaluate(args: &[Value]) -> BuiltinResult<SetenvResult> {
        futures::executor::block_on(super::evaluate(args))
    }

    fn unique_name(suffix: &str) -> String {
        format!("RUNMAT_TEST_SETENV_{}", suffix)
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
        let err =
            setenv_builtin(vec![Value::Num(5.0), Value::String("value".to_string())]).unwrap_err();
        assert_eq!(err.message(), ERR_NAME_TYPE);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_errors_when_value_is_not_text() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let err = setenv_builtin(vec![
            Value::String("RUNMAT_INVALID_VALUE".to_string()),
            Value::Num(1.0),
        ])
        .unwrap_err();
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
        let err = setenv_builtin(vec![
            Value::StringArray(array),
            Value::String("value".to_string()),
        ])
        .unwrap_err();
        assert_eq!(err.message(), ERR_NAME_TYPE);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setenv_errors_for_char_array_with_multiple_rows() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let array = CharArray::new(vec!['R', 'M'], 2, 1).expect("two-row char array");
        let err = setenv_builtin(vec![
            Value::CharArray(array),
            Value::String("value".to_string()),
        ])
        .unwrap_err();
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
}
