//! MATLAB-compatible `error` builtin with structured exception handling semantics.

use std::convert::TryFrom;

use runmat_builtins::diagnostic_identifier::{
    is_error_legacy_unqualified as looks_like_unqualified_identifier,
    is_qualified as is_message_identifier,
};
#[cfg(test)]
use runmat_builtins::ERROR_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, ERROR_ERROR_CORRECTION_UNSUPPORTED, ERROR_ERROR_EXTRA_ARGS_MEXCEPTION,
    ERROR_ERROR_EXTRA_ARGS_STRUCT, ERROR_ERROR_INVALID_IDENTIFIER, ERROR_ERROR_INVALID_INPUT,
    ERROR_ERROR_MISSING_MESSAGE, ERROR_ERROR_STRUCT_NO_FIELDS,
    ERROR_ERROR_STRUCT_STACK_UNSUPPORTED, ERROR_MEXCEPTION_EXTENSION, ERROR_STRUCT_ALIAS_EXTENSION,
    ERROR_UNQUALIFIED_IDENTIFIER_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{StructValue, Value};

use crate::builtins::common::format::{
    decode_escape_sequences, flatten_argument_elements, format_variadic,
};
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "error";

fn error_flow(identifier: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_identifier(normalize_identifier(identifier))
        .build()
}

fn error_default_identifier() -> &'static str {
    ERROR_ERROR_MISSING_MESSAGE
        .identifier
        .expect("error default identifier must be defined")
}

fn error_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    error_error_with_message(error.message, error)
}

fn error_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(normalize_identifier(identifier));
    }
    builder.build()
}

fn remap_error_flow(err: RuntimeError, error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(err.message().to_string())
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(normalize_identifier(identifier));
    }
    builder.build()
}

#[runtime_builtin(
    name = "error",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::diagnostics::error"
)]
async fn error_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Err(error_error(&ERROR_ERROR_MISSING_MESSAGE));
    }
    if args.iter().all(value_is_empty_array) {
        return Ok(Value::Num(0.0));
    }

    let first = args[0].clone();
    let rest = args.into_range(1..)?;

    match first {
        Value::MException(mex) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ERROR_MEXCEPTION_EXTENSION,
                BUILTIN_NAME,
            )?;
            if !rest.is_empty() {
                return Err(error_error(&ERROR_ERROR_EXTRA_ARGS_MEXCEPTION));
            }
            Err(crate::runtime_error::error_from_exception(mex))
        }
        Value::Struct(ref st) => {
            if !rest.is_empty() {
                return Err(error_error(&ERROR_ERROR_EXTRA_ARGS_STRUCT));
            }
            let (identifier, message) = extract_struct_error_fields(st)?;
            Err(error_flow(&identifier, &message))
        }
        Value::Object(object)
            if object
                .class_name()
                .is_in_namespace(runmat_types::standard::LANGUAGE_CORRECTION_NAMESPACE) =>
        {
            Err(error_error(&ERROR_ERROR_CORRECTION_UNSUPPORTED))
        }
        other => handle_message_arguments(runtime.as_ref(), other, rest).await,
    }
}

async fn handle_message_arguments(
    runtime: Option<&crate::context::RuntimeContext>,
    first: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let first_string = value_to_string("error", &first)?;

    if rest.is_empty() {
        return Err(error_flow(error_default_identifier(), first_string));
    }

    let mut identifier = error_default_identifier().to_string();
    let mut format_string = first_string;
    let mut format_args = rest.view();

    if !rest.is_empty() && is_message_identifier(&format_string) {
        identifier = normalize_identifier(&format_string);
        let (message_value, extra_args) = rest.view().split_first().expect("rest not empty");
        format_string = value_to_string("error", message_value)?;
        format_args = extra_args;
    } else if !rest.is_empty() && looks_like_unqualified_identifier(&format_string) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ERROR_UNQUALIFIED_IDENTIFIER_EXTENSION,
            BUILTIN_NAME,
        )?;
        identifier = normalize_identifier(&format_string);
        let (message_value, extra_args) = rest.view().split_first().expect("rest not empty");
        format_string = value_to_string("error", message_value)?;
        format_args = extra_args;
    }

    let decoded = decode_escape_sequences(BUILTIN_NAME, &format_string)
        .map_err(|flow| remap_error_flow(flow, &ERROR_ERROR_INVALID_INPUT))?;
    let message = if format_args.is_empty() {
        decoded
    } else {
        let flatten = flatten_argument_elements(format_args, BUILTIN_NAME);
        let flattened = match runtime {
            Some(runtime) => runtime.scope(flatten).await,
            None => flatten.await,
        }
        .map_err(|flow| remap_error_flow(flow, &ERROR_ERROR_INVALID_INPUT))?;
        format_variadic(&decoded, &flattened)
            .map_err(|flow| remap_error_flow(flow, &ERROR_ERROR_INVALID_INPUT))?
    };

    Err(error_flow(&identifier, message))
}

fn extract_struct_error_fields(
    struct_value: &StructValue,
) -> crate::BuiltinResult<(String, String)> {
    if struct_value.fields.contains_key("stack") {
        return Err(error_error(&ERROR_ERROR_STRUCT_STACK_UNSUPPORTED));
    }
    let mut identifier_value = struct_value.fields.get("identifier");
    let mut message_value = struct_value.fields.get("message");
    let uses_alias = identifier_value.is_none() && struct_value.fields.contains_key("messageid")
        || message_value.is_none() && struct_value.fields.contains_key("msg");
    if uses_alias {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ERROR_STRUCT_ALIAS_EXTENSION,
            BUILTIN_NAME,
        )?;
        identifier_value = identifier_value.or_else(|| struct_value.fields.get("messageid"));
        message_value = message_value.or_else(|| struct_value.fields.get("msg"));
    }
    if identifier_value.is_none() && message_value.is_none() {
        return Err(error_error(&ERROR_ERROR_STRUCT_NO_FIELDS));
    }

    let identifier = match identifier_value {
        Some(value) => value_to_string("error", value)?,
        None => error_default_identifier().to_string(),
    };
    if !identifier.is_empty() && !is_message_identifier(&identifier) {
        return Err(error_error(&ERROR_ERROR_INVALID_IDENTIFIER));
    }
    let message = match message_value {
        Some(value) => value_to_string("error", value)?,
        None => String::new(),
    };
    Ok((identifier, message))
}

fn value_is_empty_array(value: &Value) -> bool {
    match value {
        Value::CharArray(array) => array.data().is_empty(),
        Value::StringArray(array) => array.data().is_empty(),
        Value::Tensor(tensor) => tensor.is_empty(),
        Value::ComplexTensor(tensor) => tensor.is_empty(),
        Value::LogicalArray(array) => array.data.is_empty(),
        Value::Cell(cell) => cell.data().is_empty(),
        _ => false,
    }
}

fn value_to_string(context: &str, value: &Value) -> crate::BuiltinResult<String> {
    String::try_from(value).map_err(|e| {
        error_error_with_message(format!("{context}: {e}"), &ERROR_ERROR_INVALID_INPUT)
    })
}

fn normalize_identifier(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        error_default_identifier().to_string()
    } else if trimmed.contains(':') {
        trimmed.to_string()
    } else {
        format!("RunMat:{trimmed}")
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_value::{CharArray, IntValue, IntegerStorage, MException, Tensor};

    fn run_error(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        futures::executor::block_on(super::error_builtin(
            None,
            crate::call::BuiltinArguments::from_borrowed(&args, None)?,
        ))
    }

    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn error_requires_message() {
        let err = unwrap_error(run_error(Vec::new()).expect_err("should error"));
        assert_eq!(err.identifier(), Some(error_default_identifier()));
        assert!(err.message().contains("missing message"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn default_identifier_is_applied() {
        let err = unwrap_error(run_error(vec![Value::from("Failure!")]).expect_err("should error"));
        assert_eq!(err.identifier(), Some(error_default_identifier()));
        assert_eq!(err.message(), "Failure!");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn custom_identifier_is_preserved() {
        let err = unwrap_error(
            run_error(vec![
                Value::from("runmat:tests:badValue"),
                Value::from("Value %d is not allowed."),
                Value::from(5.0),
            ])
            .expect_err("should error"),
        );
        assert_eq!(err.identifier(), Some("runmat:tests:badValue"));
        assert_eq!(err.message(), "Value 5 is not allowed.");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn identifier_is_normalised_when_namespace_missing() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = unwrap_error(
            run_error(vec![
                Value::from("missingNamespace"),
                Value::from("Message"),
            ])
            .expect_err("should error"),
        );
        assert_eq!(err.identifier(), Some("RunMat:missingNamespace"));
        assert_eq!(err.message(), "Message");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn format_string_with_colon_not_treated_as_identifier() {
        let err = unwrap_error(
            run_error(vec![
                Value::from("Value: %d."),
                Value::Int(IntValue::I32(7)),
            ])
            .expect_err("should error"),
        );
        assert_eq!(err.identifier(), Some(error_default_identifier()));
        assert_eq!(err.message(), "Value: 7.");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn error_accepts_mexception() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let mex = MException::new("RunMat:demo:test".to_string(), "broken".to_string());
        let err = unwrap_error(run_error(vec![Value::MException(mex)]).expect_err("should error"));
        assert_eq!(err.identifier(), Some("RunMat:demo:test"));
        assert_eq!(err.message(), "broken");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn error_rejects_extra_args_after_mexception() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let mex = MException::new("RunMat:demo:test".to_string(), "broken".to_string());
        let err = unwrap_error(
            run_error(vec![Value::MException(mex), Value::from(1.0)]).expect_err("should error"),
        );
        assert_eq!(err.identifier(), Some(error_default_identifier()));
        assert!(err.message().contains("additional arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn error_accepts_message_struct() {
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.fields
            .insert("identifier".to_string(), Value::from("pkg:demo:failure"));
        st.fields
            .insert("message".to_string(), Value::from("Struct message."));
        let err =
            unwrap_error(run_error(vec![Value::Struct(st.finish())]).expect_err("should error"));
        assert_eq!(err.identifier(), Some("pkg:demo:failure"));
        assert_eq!(err.message(), "Struct message.");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn error_struct_accepts_identifier_only() {
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.fields
            .insert("identifier".to_string(), Value::from("pkg:demo:oops"));
        let err =
            unwrap_error(run_error(vec![Value::Struct(st.finish())]).expect_err("should error"));
        assert_eq!(err.identifier(), Some("pkg:demo:oops"));
        assert_eq!(err.message(), "");
    }

    #[test]
    fn error_all_empty_inputs_do_not_throw() {
        let empty = CharArray::from_code_units(Vec::new(), 1, 0).expect("empty char vector");
        assert_eq!(
            run_error(vec![Value::CharArray(empty)]).expect("all-empty error is a no-op"),
            Value::Num(0.0)
        );
    }

    #[test]
    fn error_multiargument_form_decodes_escapes() {
        let err = run_error(vec![
            Value::String("value=%d\\nnext".to_string().into()),
            Value::Int(IntValue::I32(7)),
        ])
        .expect_err("error must throw");
        assert_eq!(err.message(), "value=7\nnext");
    }

    #[test]
    fn error_formats_all_integer_classes_without_f64_mirroring() {
        for value in [
            IntValue::I8(-8),
            IntValue::I16(-16),
            IntValue::I32(-32),
            IntValue::I64(i64::MIN),
            IntValue::U8(8),
            IntValue::U16(16),
            IntValue::U32(32),
            IntValue::U64(u64::MAX),
        ] {
            let expected = match &value {
                IntValue::I8(v) => v.to_string(),
                IntValue::I16(v) => v.to_string(),
                IntValue::I32(v) => v.to_string(),
                IntValue::I64(v) => v.to_string(),
                IntValue::U8(v) => v.to_string(),
                IntValue::U16(v) => v.to_string(),
                IntValue::U32(v) => v.to_string(),
                IntValue::U64(v) => v.to_string(),
            };
            let format = if matches!(
                &value,
                IntValue::U8(_) | IntValue::U16(_) | IntValue::U32(_) | IntValue::U64(_)
            ) {
                "value=%u"
            } else {
                "value=%d"
            };
            let err = run_error(vec![
                Value::String(format.to_string().into()),
                Value::Int(value),
            ])
            .expect_err("error must throw");
            assert_eq!(err.message(), format!("value={expected}"));
        }
    }

    #[test]
    fn strict_mode_gates_error_extensions_independently() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let unqualified = run_error(vec![
            Value::String("unqualified".to_string().into()),
            Value::String("message".to_string().into()),
        ])
        .expect_err("unqualified identifier extension");
        assert_eq!(
            unqualified.identifier(),
            ERROR_UNQUALIFIED_IDENTIFIER_EXTENSION.error_identifier
        );

        let mex = MException::new("pkg:test".to_string(), "message".to_string());
        let mex_error =
            run_error(vec![Value::MException(mex)]).expect_err("MException input extension");
        assert_eq!(
            mex_error.identifier(),
            ERROR_MEXCEPTION_EXTENSION.error_identifier
        );
    }

    #[test]
    fn error_descriptor_and_integer_capabilities_cover_settled_forms() {
        assert_eq!(ERROR_DESCRIPTOR.signatures.len(), 5);
        assert!(ERROR_DESCRIPTOR
            .signatures
            .iter()
            .all(|signature| signature.outputs.is_empty()));
        let builtin = runmat_builtins::builtin_catalog_entry_by_name("error").expect("registered");
        assert_eq!(builtin.integer_capabilities.len(), 2);
        assert_eq!(builtin.integer_capabilities[0].inputs[0].classes.len(), 8);
        assert_eq!(builtin.extensions.len(), 3);
    }

    #[test]
    fn resident_message_rejects_without_provider_access() {
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("owned allocation without a registered provider");
        let resident = Value::GpuTensor(allocation.handle);
        let err = run_error(vec![resident]).expect_err("resident msg is not text");
        assert_eq!(err.identifier(), ERROR_ERROR_INVALID_INPUT.identifier);
        assert!(!err.message().contains("provider"));
        drop(err);
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn resident_integer_format_argument_gathers_exactly_from_its_owner() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1])
                    .expect("wide integer tensor");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload wide integer");
            let error = run_error(vec![
                Value::from("integer:resident"),
                Value::from("value %d"),
                Value::GpuTensor(handle),
            ])
            .expect_err("error must throw after formatting");
            assert_eq!(error.identifier(), Some("integer:resident"));
            assert_eq!(error.message(), "value 9007199254740993");
        });
    }
}
