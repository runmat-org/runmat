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

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "error",
        builtin_path = "crate::builtins::diagnostics::error"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "error"
category: "diagnostics"
keywords: ["error", "exception", "throw", "diagnostics", "message"]
summary: "Throw an exception with an identifier and a formatted diagnostic message."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Always executed on the host; GPU tensors are gathered only when they appear in formatted arguments."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::diagnostics::error::tests"
  integration: null
---

# What does the `error` function do in MATLAB / RunMat?
`error` throws an exception immediately, unwinding the current execution frame and transferring control to the nearest `catch` block (or aborting the program if none exists). RunMat mirrors MATLAB's behaviour, including support for message identifiers, formatted messages, `MException` objects, and legacy message structs.

## How does the `error` function behave in MATLAB / RunMat?
- `error(message)` throws using the default identifier `MATLAB:error`.
- `error(id, message)` uses a custom identifier. Identifiers are normalised to `MATLAB:*` when they do not already contain a namespace.
- `error(fmt, arg1, ...)` formats the message with MATLAB's `sprintf` rules before throwing.
- `error(id, fmt, arg1, ...)` combines both a custom identifier and formatted message text.
- `error(MException_obj)` rethrows an existing exception without altering its identifier or message.
- `error(struct('identifier', id, 'message', msg))` honours the legacy structure form.
- Invalid invocations (missing message, extra arguments after an `MException`, malformed structs, etc.) themselves raise `MATLAB:error` diagnostics so the caller can correct usage.

The thrown exception is observed in MATLAB-compatible `try`/`catch` constructs or by the embedding runtime, which converts the string back into an `MException` object.

## GPU execution and residency
`error` is a control-flow builtin and never executes on the GPU. When formatting messages that include GPU-resident arrays (for example, via `%g` or `%s` specifiers), RunMat first gathers those values back to host memory so that the final diagnostic message accurately reflects the data the user passed.

## Examples of using the `error` function in MATLAB / RunMat

### Throwing an error with a simple message
```matlab
try
    error("Computation failed.");
catch err
    fprintf("%s -> %s\n", err.identifier, err.message);
end
```

### Throwing an error with a custom identifier
```matlab
try
    error("runmat:examples:invalidState", "State vector is empty.");
catch err
    fprintf("%s\n", err.identifier);
end
```

### Formatting values inside the error message
```matlab
value = 42;
try
    error("MATLAB:demo:badValue", "Value %d is outside [%d, %d].", value, 0, 10);
catch err
    disp(err.message);
end
```

### Rethrowing an existing MException
```matlab
try
    try
        error("MATLAB:inner:failure", "Inner failure.");
    catch inner
        error(inner); % propagate with original identifier/message
    end
catch err
    fprintf("%s\n", err.identifier);
end
```

### Using a legacy message struct
```matlab
S.identifier = "toolbox:demo:badInput";
S.message = "Inputs must be positive integers.";
try
    error(S);
catch err
    fprintf("%s\n", err.identifier);
end
```

## FAQ

1. **How do I choose a custom identifier?** Use `component:mnemonic` style strings such as `"MATLAB:io:fileNotFound"` or `"runmat:tools:badInput"`. If you omit a namespace (`:`), RunMat prefixes the identifier with `MATLAB:` automatically.
2. **Can I rethrow an existing `MException`?** Yes. Pass the object returned by `catch err` directly to `error(err)` to propagate it unchanged.
3. **What happens if I pass extra arguments after an `MException` or struct?** RunMat treats that as invalid usage and raises `MATLAB:error` explaining that no additional arguments are allowed in those forms.
4. **Does `error` run on the GPU?** No. The builtin executes on the host. If the message references GPU data, RunMat gathers the values before formatting the diagnostic string.
5. **What if I call `error` without arguments?** RunMat raises `MATLAB:error` indicating that a message is required, matching MATLAB's behaviour.
6. **Why was my identifier normalised to `MATLAB:...`?** MATLAB requires message identifiers to contain at least one namespace separator (`:`). RunMat enforces this rule so diagnostics integrate cleanly with tooling that expects fully-qualified identifiers.
7. **Can the message span multiple lines?** Yes. Any newline characters in the formatted message are preserved exactly in the thrown exception.
8. **Does formatting follow MATLAB rules?** Yes. `error` uses the same formatter as `sprintf`, including width/precision specifiers and numeric conversions, and will raise `MATLAB:error` if the format string is invalid or under-specified.
#"#;

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
fn error_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Err(((build_error(
            DEFAULT_IDENTIFIER,
            "error: missing message argument",
        ))).into());
    }

    let mut iter = args.into_iter();
    let first = iter.next().expect("checked above");
    let rest: Vec<Value> = iter.collect();

    match first {
        Value::MException(mex) => {
            if !rest.is_empty() {
                return Err(((build_error(
                    DEFAULT_IDENTIFIER,
                    "error: additional arguments are not allowed when passing an MException",
                ))).into());
            }
            Err(((build_error(&mex.identifier, &mex.message))).into())
        }
        Value::Struct(ref st) => {
            if !rest.is_empty() {
                return Err(((build_error(
                    DEFAULT_IDENTIFIER,
                    "error: additional arguments are not allowed when passing a message struct",
                ))).into());
            }
            let (identifier, message) = extract_struct_error_fields(st)?;
            Err(((build_error(&identifier, &message))).into())
        }
        other => (handle_message_arguments(other, rest)).map_err(Into::into),
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        use crate::builtins::common::test_support;
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
