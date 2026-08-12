//! MATLAB-compatible `error` builtin with structured exception handling semantics.

use std::convert::TryFrom;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{StructValue, Value};

use crate::builtins::common::format::{
    decode_escape_sequences, flatten_arguments, format_variadic,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::diagnostics::type_resolvers::error_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "error";

const ERROR_INPUTS_MESSAGE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "message",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Error message text.",
}];

const ERROR_INPUTS_MESSAGE_VARIADIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Error message template text.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Formatting values for the message template.",
    },
];

const ERROR_INPUTS_IDENTIFIER_MESSAGE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "message_id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Message identifier.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Error message text.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Formatting values for the message template.",
    },
];

const ERROR_INPUTS_STRUCT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "msg_struct",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Struct containing identifier/message fields.",
}];

const ERROR_INPUTS_CORRECTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "correction",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "MATLAB correction object.",
    },
    BuiltinParamDescriptor {
        name: "messageArguments",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Identifier, message, and optional formatting values.",
    },
];

const ERROR_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "error(msg)",
        inputs: &ERROR_INPUTS_MESSAGE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "error(msg, A)",
        inputs: &ERROR_INPUTS_MESSAGE_VARIADIC,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "error(errID, ___)",
        inputs: &ERROR_INPUTS_IDENTIFIER_MESSAGE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "error(errorStruct)",
        inputs: &ERROR_INPUTS_STRUCT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "error(correction, ___)",
        inputs: &ERROR_INPUTS_CORRECTION,
        outputs: &[],
    },
];

const ERROR_ERROR_MISSING_MESSAGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.MISSING_MESSAGE",
    identifier: Some("RunMat:error"),
    when: "No arguments are supplied.",
    message: "error: missing message argument",
};

const ERROR_ERROR_EXTRA_ARGS_MEXCEPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.MEXCEPTION_EXTRA_ARGS",
    identifier: Some("RunMat:error"),
    when: "Additional arguments are supplied after an MException input.",
    message: "error: additional arguments are not allowed when passing an MException",
};

const ERROR_ERROR_EXTRA_ARGS_STRUCT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.STRUCT_EXTRA_ARGS",
    identifier: Some("RunMat:error"),
    when: "Additional arguments are supplied after a message-struct input.",
    message: "error: additional arguments are not allowed when passing a message struct",
};

const ERROR_ERROR_STRUCT_NO_FIELDS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.STRUCT_NO_FIELDS",
    identifier: Some("RunMat:error"),
    when: "Message struct contains none of message, identifier, or stack.",
    message: "error: message struct must contain 'message', 'identifier', or 'stack'",
};

const ERROR_ERROR_STRUCT_STACK_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.STRUCT_STACK_UNSUPPORTED",
    identifier: Some("RunMat:error"),
    when: "Message struct requests an explicit MATLAB stack that cannot be represented yet.",
    message: "error: explicit errorStruct stack is not supported yet",
};

const ERROR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.INVALID_INPUT",
    identifier: Some("RunMat:error"),
    when: "Identifier/message inputs or format arguments are not string-compatible.",
    message: "error: invalid input argument",
};

const ERROR_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.INVALID_IDENTIFIER",
    identifier: Some("RunMat:error"),
    when: "Identifier fields do not follow documented colon-separated identifier grammar.",
    message: "error: invalid error identifier",
};

const ERROR_ERROR_CORRECTION_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.CORRECTION_UNSUPPORTED",
    identifier: Some("RunMat:error"),
    when: "A documented matlab.lang.correction object is supplied before correction objects are representable.",
    message: "error: matlab.lang.correction objects are not supported yet",
};

const ERROR_ERRORS: [BuiltinErrorDescriptor; 8] = [
    ERROR_ERROR_MISSING_MESSAGE,
    ERROR_ERROR_EXTRA_ARGS_MEXCEPTION,
    ERROR_ERROR_EXTRA_ARGS_STRUCT,
    ERROR_ERROR_STRUCT_NO_FIELDS,
    ERROR_ERROR_STRUCT_STACK_UNSUPPORTED,
    ERROR_ERROR_INVALID_INPUT,
    ERROR_ERROR_INVALID_IDENTIFIER,
    ERROR_ERROR_CORRECTION_UNSUPPORTED,
];

pub const ERROR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ERROR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERROR_ERRORS,
};

const ERROR_MEXCEPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "error-mexception-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "error(MException) is a RunMat extension; MATLAB uses throw or rethrow",
    error_identifier: Some("RunMat:compatibility:ErrorMExceptionExtension"),
};
const ERROR_UNQUALIFIED_IDENTIFIER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "error-unqualified-identifier",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "Treating an unqualified leading token as an error identifier is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ErrorUnqualifiedIdentifierExtension"),
    };
const ERROR_STRUCT_ALIAS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "error-struct-field-aliases",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "messageid and msg error-structure aliases are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ErrorStructAliasExtension"),
};
pub const ERROR_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    ERROR_MEXCEPTION_EXTENSION,
    ERROR_UNQUALIFIED_IDENTIFIER_EXTENSION,
    ERROR_STRUCT_ALIAS_EXTENSION,
];

const ERROR_INTEGER_FORMAT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight typed-integer classes are documented numeric formatting values and remain exact through integer conversion specifiers.",
    }];
const ERROR_REJECTED_INTEGER_TEXT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "msg",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The message role is a text scalar and never converts integer data to text implicitly.",
    },
    BuiltinIntegerInputCapability {
        name: "errID",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The identifier role is a text scalar and rejects integer values before provider access.",
    },
];
pub const ERROR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "error(msg, integer_A) or error(errID, msg, integer_A)",
        inputs: &ERROR_INTEGER_FORMAT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host formatting is exact for integer conversions; documented GPU-array arguments gather only after host message and identifier validation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "error(integer_msg, ...) or error(integer_errID, ...)",
        inputs: &ERROR_REJECTED_INTEGER_TEXT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer values are valid only in formatting-value roles, never as message or identifier text.",
    },
];

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
    category = "diagnostics",
    summary = "Throw exceptions with identifiers and formatted messages.",
    keywords = "error,exception,diagnostics,throw",
    accel = "metadata",
    type_resolver(error_type),
    descriptor(crate::builtins::diagnostics::error::ERROR_DESCRIPTOR),
    extensions(crate::builtins::diagnostics::error::ERROR_EXTENSIONS),
    integer_capabilities(crate::builtins::diagnostics::error::ERROR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::diagnostics::error"
)]
async fn error_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Err(error_error(&ERROR_ERROR_MISSING_MESSAGE));
    }
    if args.iter().all(value_is_empty_array) {
        return Ok(Value::Num(0.0));
    }

    let mut iter = args.into_iter();
    let first = iter.next().expect("checked above");
    let rest: Vec<Value> = iter.collect();

    match first {
        Value::MException(mex) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ERROR_MEXCEPTION_EXTENSION,
                BUILTIN_NAME,
            )?;
            if !rest.is_empty() {
                return Err(error_error(&ERROR_ERROR_EXTRA_ARGS_MEXCEPTION));
            }
            Err(error_flow(&mex.identifier, &mex.message))
        }
        Value::Struct(ref st) => {
            if !rest.is_empty() {
                return Err(error_error(&ERROR_ERROR_EXTRA_ARGS_STRUCT));
            }
            let (identifier, message) = extract_struct_error_fields(st)?;
            Err(error_flow(&identifier, &message))
        }
        Value::Object(object) if object.class_name.starts_with("matlab.lang.correction.") => {
            Err(error_error(&ERROR_ERROR_CORRECTION_UNSUPPORTED))
        }
        other => handle_message_arguments(other, rest).await,
    }
}

async fn handle_message_arguments(first: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let first_string = value_to_string("error", &first)?;

    if rest.is_empty() {
        return Err(error_flow(error_default_identifier(), first_string));
    }

    let mut identifier = error_default_identifier().to_string();
    let mut format_string = first_string;
    let mut format_args: &[Value] = &rest;

    if !rest.is_empty() && is_message_identifier(&format_string) {
        identifier = normalize_identifier(&format_string);
        let (message_value, extra_args) = rest.split_first().expect("rest not empty");
        format_string = value_to_string("error", message_value)?;
        format_args = extra_args;
    } else if !rest.is_empty() && looks_like_unqualified_identifier(&format_string) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ERROR_UNQUALIFIED_IDENTIFIER_EXTENSION,
            BUILTIN_NAME,
        )?;
        identifier = normalize_identifier(&format_string);
        let (message_value, extra_args) = rest.split_first().expect("rest not empty");
        format_string = value_to_string("error", message_value)?;
        format_args = extra_args;
    }

    let decoded = decode_escape_sequences(BUILTIN_NAME, &format_string)
        .map_err(|flow| remap_error_flow(flow, &ERROR_ERROR_INVALID_INPUT))?;
    let message = if format_args.is_empty() {
        decoded
    } else {
        let flattened = flatten_arguments(format_args, BUILTIN_NAME)
            .await
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
        Value::CharArray(array) => array.data.is_empty(),
        Value::StringArray(array) => array.data.is_empty(),
        Value::Tensor(tensor) => tensor.is_empty(),
        Value::ComplexTensor(tensor) => tensor.is_empty(),
        Value::LogicalArray(array) => array.data.is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
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

fn is_message_identifier(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || !trimmed.contains(':') {
        return false;
    }
    trimmed.split(':').all(|field| {
        let mut chars = field.chars();
        chars.next().is_some_and(|ch| ch.is_ascii_alphabetic())
            && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    })
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
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{CharArray, IntValue, IntegerStorage, MException, Tensor};

    fn run_error(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        futures::executor::block_on(super::error_builtin(args))
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
        let mut st = StructValue::new();
        st.fields
            .insert("identifier".to_string(), Value::from("pkg:demo:failure"));
        st.fields
            .insert("message".to_string(), Value::from("Struct message."));
        let err = unwrap_error(run_error(vec![Value::Struct(st)]).expect_err("should error"));
        assert_eq!(err.identifier(), Some("pkg:demo:failure"));
        assert_eq!(err.message(), "Struct message.");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn error_struct_accepts_identifier_only() {
        let mut st = StructValue::new();
        st.fields
            .insert("identifier".to_string(), Value::from("pkg:demo:oops"));
        let err = unwrap_error(run_error(vec![Value::Struct(st)]).expect_err("should error"));
        assert_eq!(err.identifier(), Some("pkg:demo:oops"));
        assert_eq!(err.message(), "");
    }

    #[test]
    fn error_all_empty_inputs_do_not_throw() {
        let empty = CharArray::new(Vec::new(), 1, 0).expect("empty char vector");
        assert_eq!(
            run_error(vec![Value::CharArray(empty)]).expect("all-empty error is a no-op"),
            Value::Num(0.0)
        );
    }

    #[test]
    fn error_multiargument_form_decodes_escapes() {
        let err = run_error(vec![
            Value::String("value=%d\\nnext".to_string()),
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
            let err = run_error(vec![Value::String(format.to_string()), Value::Int(value)])
                .expect_err("error must throw");
            assert_eq!(err.message(), format!("value={expected}"));
        }
    }

    #[test]
    fn strict_mode_gates_error_extensions_independently() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let unqualified = run_error(vec![
            Value::String("unqualified".to_string()),
            Value::String("message".to_string()),
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
        let builtin = runmat_builtins::builtin_function_by_name("error").expect("registered");
        assert_eq!(builtin.integer_capabilities.len(), 2);
        assert_eq!(builtin.integer_capabilities[0].inputs[0].classes.len(), 8);
        assert_eq!(builtin.extensions.len(), 3);
    }

    #[test]
    fn resident_message_rejects_without_provider_access() {
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let err = run_error(vec![resident]).expect_err("resident msg is not text");
        assert_eq!(err.identifier(), ERROR_ERROR_INVALID_INPUT.identifier);
        assert!(!err.message().contains("provider"));
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

    #[test]
    fn error_type_is_unknown() {
        assert_eq!(
            error_type(&[Type::String], &ResolveContext::new(Vec::new())),
            Type::Unknown
        );
    }
}
