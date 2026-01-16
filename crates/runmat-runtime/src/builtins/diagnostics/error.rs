//! MATLAB-compatible `error` builtin with structured exception handling semantics.

use std::convert::TryFrom;

use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::format::format_variadic;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

const DEFAULT_IDENTIFIER: &str = "MATLAB:error";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::diagnostics::error")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "error",
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
    notes: "Control-flow builtin; never dispatched to GPU backends.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::diagnostics::error")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "error",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Control-flow builtin; excluded from fusion planning.",
};

#[runtime_builtin(
    name = "error",
    category = "diagnostics",
    summary = "Throw an exception with an identifier and a formatted diagnostic message.",
    keywords = "error,exception,diagnostics,throw",
    accel = "metadata",
    builtin_path = "crate::builtins::diagnostics::error"
)]
fn error_builtin(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err(build_error(
            DEFAULT_IDENTIFIER,
            "error: missing message argument",
        ));
    }

    let mut iter = args.into_iter();
    let first = iter.next().expect("checked above");
    let rest: Vec<Value> = iter.collect();

    match first {
        Value::MException(mex) => {
            if !rest.is_empty() {
                return Err(build_error(
                    DEFAULT_IDENTIFIER,
                    "error: additional arguments are not allowed when passing an MException",
                ));
            }
            Err(build_error(&mex.identifier, &mex.message))
        }
        Value::Struct(ref st) => {
            if !rest.is_empty() {
                return Err(build_error(
                    DEFAULT_IDENTIFIER,
                    "error: additional arguments are not allowed when passing a message struct",
                ));
            }
            let (identifier, message) = extract_struct_error_fields(st)?;
            Err(build_error(&identifier, &message))
        }
        other => handle_message_arguments(other, rest),
    }
}

fn handle_message_arguments(first: Value, rest: Vec<Value>) -> Result<Value, String> {
    let first_string = value_to_string("error", &first)?;

    if rest.is_empty() {
        return Err(build_error(DEFAULT_IDENTIFIER, &first_string));
    }

    let mut identifier = DEFAULT_IDENTIFIER.to_string();
    let mut format_string = first_string;
    let mut format_args: &[Value] = &rest;

    if !rest.is_empty()
        && (is_message_identifier(&format_string)
            || looks_like_unqualified_identifier(&format_string))
    {
        identifier = normalize_identifier(&format_string);
        let (message_value, extra_args) = rest.split_first().expect("rest not empty");
        format_string = value_to_string("error", message_value)?;
        format_args = extra_args;
    }

    let message = if format_args.is_empty() {
        format_string
    } else {
        format_variadic(&format_string, format_args)
            .map_err(|e| build_error(DEFAULT_IDENTIFIER, &e))?
    };

    Err(build_error(&identifier, &message))
}

fn extract_struct_error_fields(struct_value: &StructValue) -> Result<(String, String), String> {
    let identifier_value = struct_value
        .fields
        .get("identifier")
        .or_else(|| struct_value.fields.get("messageid"))
        .ok_or_else(|| {
            build_error(
                DEFAULT_IDENTIFIER,
                "error: message struct must contain an 'identifier' field",
            )
        })?;
    let message_value = struct_value
        .fields
        .get("message")
        .or_else(|| struct_value.fields.get("msg"))
        .ok_or_else(|| {
            build_error(
                DEFAULT_IDENTIFIER,
                "error: message struct must contain a 'message' field",
            )
        })?;

    let identifier = value_to_string("error", identifier_value)?;
    let message = value_to_string("error", message_value)?;
    Ok((identifier, message))
}

fn value_to_string(context: &str, value: &Value) -> Result<String, String> {
    String::try_from(value).map_err(|e| build_error(DEFAULT_IDENTIFIER, &format!("{context}: {e}")))
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{IntValue, MException};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn error_requires_message() {
        let err = error_builtin(Vec::new()).expect_err("should error");
        assert!(err.contains("missing message"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn default_identifier_is_applied() {
        let err = error_builtin(vec![Value::from("Failure!")]).expect_err("should error");
        assert_eq!(err, "MATLAB:error: Failure!");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn custom_identifier_is_preserved() {
        let err = error_builtin(vec![
            Value::from("runmat:tests:badValue"),
            Value::from("Value %d is not allowed."),
            Value::from(5.0),
        ])
        .expect_err("should error");
        assert_eq!(err, "runmat:tests:badValue: Value 5 is not allowed.");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn identifier_is_normalised_when_namespace_missing() {
        let err = error_builtin(vec![
            Value::from("missingNamespace"),
            Value::from("Message"),
        ])
        .expect_err("should error");
        assert_eq!(err, "MATLAB:missingNamespace: Message");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn format_string_with_colon_not_treated_as_identifier() {
        let err = error_builtin(vec![
            Value::from("Value: %d."),
            Value::Int(IntValue::I32(7)),
        ])
        .expect_err("should error");
        assert_eq!(err, "MATLAB:error: Value: 7.");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn error_accepts_mexception() {
        let mex = MException::new("MATLAB:demo:test".to_string(), "broken".to_string());
        let err = error_builtin(vec![Value::MException(mex)]).expect_err("should error");
        assert_eq!(err, "MATLAB:demo:test: broken");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn error_rejects_extra_args_after_mexception() {
        let mex = MException::new("MATLAB:demo:test".to_string(), "broken".to_string());
        let err = error_builtin(vec![Value::MException(mex), Value::from(1.0)])
            .expect_err("should error");
        assert!(err.contains("additional arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn error_accepts_message_struct() {
        let mut st = StructValue::new();
        st.fields
            .insert("identifier".to_string(), Value::from("pkg:demo:failure"));
        st.fields
            .insert("message".to_string(), Value::from("Struct message."));
        let err = error_builtin(vec![Value::Struct(st)]).expect_err("should error");
        assert_eq!(err, "pkg:demo:failure: Struct message.");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn error_struct_requires_message_field() {
        let mut st = StructValue::new();
        st.fields
            .insert("identifier".to_string(), Value::from("pkg:demo:oops"));
        let err = error_builtin(vec![Value::Struct(st)]).expect_err("should error");
        assert!(err.contains("message struct must contain a 'message' field"));
    }
}
