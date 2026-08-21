//! MATLAB-compatible `weboptions` builtin for constructing HTTP client options.

use std::collections::VecDeque;

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

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const DEFAULT_TIMEOUT_SECONDS: f64 = 5.0;
const MAX_TIMEOUT_SECONDS: f64 = 2147.483647;
const BUILTIN_NAME: &str = "weboptions";

const STRUCT_COPY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "weboptions-struct-copy",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "copying and overriding an existing weboptions-compatible struct is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WeboptionsStructCopyExtension"),
};
const QUERY_PARAMETERS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "weboptions-query-parameters",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "the QueryParameters property on weboptions is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WeboptionsQueryParametersExtension"),
};
const EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "weboptions-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "passing explicit gpuArray values to host-only weboptions is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WeboptionsExplicitGpuInputExtension"),
};
pub const WEBOPTIONS_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    STRUCT_COPY_EXTENSION,
    QUERY_PARAMETERS_EXTENSION,
    EXPLICIT_GPU_EXTENSION,
];

const TIMEOUT_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Timeout",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Every integer class is accepted as a positive scalar timeout when its value lies within the documented timeout range.",
    }];
const KEY_VALUE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "KeyValue",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A scalar key value retains its exact signedness, width, and value in the options object until HTTP header serialization.",
    }];
pub const WEBOPTIONS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "options = weboptions('Timeout', integer_seconds)",
        inputs: &TIMEOUT_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The documented positive numeric scalar crosses a bounded duration boundary. Explicit gpuArray values are separately gated before gather; automatic residency gathers transparently.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "options = weboptions('KeyName', name, 'KeyValue', integer_value)",
        inputs: &KEY_VALUE_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The documented numeric key value remains exact in the host options object and is rendered directly from authoritative integer storage by request consumers.",
    },
];

const WEBOPTIONS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HTTP options struct for webread/webwrite.",
}];

const WEBOPTIONS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const WEBOPTIONS_INPUTS_STRUCT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "optionsStruct",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Existing options struct to copy and override.",
}];
const WEBOPTIONS_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option field value.",
    },
];
const WEBOPTIONS_INPUTS_STRUCT_NAME_VALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "optionsStruct",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing options struct to copy and override.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option field value.",
    },
];

const WEBOPTIONS_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "options = weboptions()",
        inputs: &WEBOPTIONS_INPUTS_NONE,
        outputs: &WEBOPTIONS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "options = weboptions(optionsStruct)",
        inputs: &WEBOPTIONS_INPUTS_STRUCT,
        outputs: &WEBOPTIONS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "options = weboptions(name, value, ...)",
        inputs: &WEBOPTIONS_INPUTS_NAME_VALUE,
        outputs: &WEBOPTIONS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "options = weboptions(optionsStruct, name, value, ...)",
        inputs: &WEBOPTIONS_INPUTS_STRUCT_NAME_VALUE,
        outputs: &WEBOPTIONS_OUTPUT,
    },
];

const WEBOPTIONS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBOPTIONS.INVALID_ARGUMENT",
    identifier: Some("RunMat:weboptions:InvalidArgument"),
    when: "Argument shape/type does not match supported weboptions forms.",
    message: "weboptions: invalid argument",
};
const WEBOPTIONS_ERROR_INVALID_OPTION_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBOPTIONS.INVALID_OPTION_NAME",
    identifier: Some("RunMat:weboptions:InvalidOptionName"),
    when: "Name-value option key is missing or not text scalar.",
    message: "weboptions: invalid option name",
};
const WEBOPTIONS_ERROR_MISSING_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBOPTIONS.MISSING_OPTION_VALUE",
    identifier: Some("RunMat:weboptions:MissingOptionValue"),
    when: "A name-value key is not followed by a value.",
    message: "weboptions: missing option value",
};
const WEBOPTIONS_ERROR_INVALID_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBOPTIONS.INVALID_OPTION_VALUE",
    identifier: Some("RunMat:weboptions:InvalidOptionValue"),
    when: "An option value fails type or domain validation.",
    message: "weboptions: invalid option value",
};
const WEBOPTIONS_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBOPTIONS.UNKNOWN_OPTION",
    identifier: Some("RunMat:weboptions:UnknownOption"),
    when: "Name-value key does not map to a supported option.",
    message: "weboptions: unknown option",
};
const WEBOPTIONS_ERROR_INVALID_CREDENTIALS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBOPTIONS.INVALID_CREDENTIALS",
    identifier: Some("RunMat:weboptions:InvalidCredentials"),
    when: "Password is provided without a username.",
    message: "weboptions: invalid credentials",
};
const WEBOPTIONS_ERROR_FLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBOPTIONS.FLOW",
    identifier: Some("RunMat:weboptions:Flow"),
    when: "Nested flow fails while gathering input values.",
    message: "weboptions: flow failure",
};

const WEBOPTIONS_ERRORS: [BuiltinErrorDescriptor; 7] = [
    WEBOPTIONS_ERROR_INVALID_ARGUMENT,
    WEBOPTIONS_ERROR_INVALID_OPTION_NAME,
    WEBOPTIONS_ERROR_MISSING_OPTION_VALUE,
    WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
    WEBOPTIONS_ERROR_UNKNOWN_OPTION,
    WEBOPTIONS_ERROR_INVALID_CREDENTIALS,
    WEBOPTIONS_ERROR_FLOW,
];

pub const WEBOPTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WEBOPTIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WEBOPTIONS_ERRORS,
};

#[allow(clippy::too_many_lines)]
#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::http::weboptions")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "weboptions",
    op_kind: GpuOpKind::Custom("http-options"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "weboptions validates CPU metadata only; gpuArray inputs are gathered eagerly.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::http::weboptions")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "weboptions",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "weboptions constructs option structs and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "weboptions",
    category = "io/http",
    summary = "Create HTTP options structs for `webread` and `webwrite` requests.",
    keywords = "weboptions,http options,timeout,headers,rest client",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::weboptions_type),
    descriptor(crate::builtins::io::http::weboptions::WEBOPTIONS_DESCRIPTOR),
    extensions(crate::builtins::io::http::weboptions::WEBOPTIONS_EXTENSIONS),
    integer_capabilities(crate::builtins::io::http::weboptions::WEBOPTIONS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::io::http::weboptions"
)]
async fn weboptions_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXPLICIT_GPU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(rest.first(), Some(Value::Struct(_))) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STRUCT_COPY_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    for pair in rest.windows(2) {
        if expect_string_scalar(
            &pair[0],
            "weboptions: option names must be character vectors or string scalars",
            &WEBOPTIONS_ERROR_INVALID_OPTION_NAME,
        )
        .is_ok_and(|name| name.eq_ignore_ascii_case("QueryParameters"))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &QUERY_PARAMETERS_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    let mut gathered = Vec::with_capacity(rest.len());
    for value in rest {
        gathered.push(gather_if_needed_async(&value).await.map_err(|flow| {
            remap_weboptions_flow(&WEBOPTIONS_ERROR_FLOW, flow, |err| {
                format!("weboptions: {}", err.message())
            })
        })?);
    }
    let mut queue: VecDeque<Value> = gathered.into();
    let mut options = default_options_struct();

    if matches!(queue.front(), Some(Value::Struct(_))) {
        if let Some(Value::Struct(struct_value)) = queue.pop_front() {
            apply_struct_fields(struct_value, &mut options)?;
        }
    }

    while let Some(name_value) = queue.pop_front() {
        let name = expect_string_scalar(
            &name_value,
            "weboptions: option names must be character vectors or string scalars",
            &WEBOPTIONS_ERROR_INVALID_OPTION_NAME,
        )?;
        let value = queue.pop_front().ok_or_else(|| {
            weboptions_error_with(
                &WEBOPTIONS_ERROR_MISSING_OPTION_VALUE,
                "weboptions: missing value for name-value argument",
            )
        })?;
        set_option_field(&mut options, &name, &value)?;
    }

    validate_credentials(&options)?;

    Ok(Value::Struct(options))
}

fn weboptions_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_weboptions_flow<F>(
    error: &'static BuiltinErrorDescriptor,
    err: RuntimeError,
    message: F,
) -> RuntimeError
where
    F: FnOnce(&RuntimeError) -> String,
{
    let mut builder = build_runtime_error(message(&err))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn default_options_struct() -> StructValue {
    let mut out = StructValue::new();
    out.fields
        .insert("ContentType".to_string(), Value::from("auto"));
    out.fields
        .insert("Timeout".to_string(), Value::Num(DEFAULT_TIMEOUT_SECONDS));
    out.fields.insert(
        "HeaderFields".to_string(),
        Value::Struct(StructValue::new()),
    );
    out.fields.insert("UserAgent".to_string(), Value::from(""));
    out.fields.insert("Username".to_string(), Value::from(""));
    out.fields.insert("Password".to_string(), Value::from(""));
    out.fields.insert("KeyName".to_string(), Value::from(""));
    out.fields.insert("KeyValue".to_string(), Value::from(""));
    out.fields
        .insert("RequestMethod".to_string(), Value::from("auto"));
    out.fields
        .insert("MediaType".to_string(), Value::from("auto"));
    out.fields.insert(
        "QueryParameters".to_string(),
        Value::Struct(StructValue::new()),
    );
    out
}

fn apply_struct_fields(source: StructValue, target: &mut StructValue) -> BuiltinResult<()> {
    for (key, value) in &source.fields {
        set_option_field(target, key, value)?;
    }
    Ok(())
}

fn set_option_field(options: &mut StructValue, name: &str, value: &Value) -> BuiltinResult<()> {
    let lower = name.to_ascii_lowercase();
    match lower.as_str() {
        "contenttype" => {
            let canonical = parse_content_type_option(value)?;
            options
                .fields
                .insert("ContentType".to_string(), Value::from(canonical));
            Ok(())
        }
        "timeout" => {
            let seconds = numeric_scalar(
                value,
                "weboptions: Timeout must be a finite, positive scalar",
                &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            if seconds <= 0.0 || (seconds.is_finite() && seconds > MAX_TIMEOUT_SECONDS) {
                return Err(weboptions_error_with(
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    "weboptions: Timeout must be a positive numeric scalar within the supported range or Inf",
                ));
            }
            options
                .fields
                .insert("Timeout".to_string(), Value::Num(seconds));
            Ok(())
        }
        "keyname" => {
            let key = expect_string_scalar(
                value,
                "weboptions: KeyName must be a character vector or string scalar",
                &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            options
                .fields
                .insert("KeyName".to_string(), Value::from(key));
            Ok(())
        }
        "keyvalue" => {
            if !matches!(
                value,
                Value::String(_)
                    | Value::CharArray(_)
                    | Value::StringArray(_)
                    | Value::Num(_)
                    | Value::Int(_)
                    | Value::Bool(_)
            ) && !matches!(value, Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor))
                && !matches!(value, Value::LogicalArray(array) if array.len() == 1)
            {
                return Err(weboptions_error_with(
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    "weboptions: KeyValue must be a text, numeric, or logical scalar",
                ));
            }
            options.fields.insert("KeyValue".to_string(), value.clone());
            Ok(())
        }
        "headerfields" => {
            let canonical = canonical_header_fields(value)?;
            options.fields.insert("HeaderFields".to_string(), canonical);
            Ok(())
        }
        "useragent" => {
            let ua = expect_string_scalar(
                value,
                "weboptions: UserAgent must be a character vector or string scalar",
                &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            options
                .fields
                .insert("UserAgent".to_string(), Value::from(ua));
            Ok(())
        }
        "username" => {
            let username = expect_string_scalar(
                value,
                "weboptions: Username must be a character vector or string scalar",
                &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            options
                .fields
                .insert("Username".to_string(), Value::from(username));
            Ok(())
        }
        "password" => {
            let password = expect_string_scalar(
                value,
                "weboptions: Password must be a character vector or string scalar",
                &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            options
                .fields
                .insert("Password".to_string(), Value::from(password));
            Ok(())
        }
        "requestmethod" => {
            let method = parse_request_method_option(value)?;
            options
                .fields
                .insert("RequestMethod".to_string(), Value::from(method));
            Ok(())
        }
        "mediatype" => {
            let media = expect_string_scalar(
                value,
                "weboptions: MediaType must be a character vector or string scalar",
                &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            options
                .fields
                .insert("MediaType".to_string(), Value::from(media));
            Ok(())
        }
        "queryparameters" => {
            let qp = canonical_query_parameters(value)?;
            options.fields.insert("QueryParameters".to_string(), qp);
            Ok(())
        }
        _ => Err(weboptions_error_with(
            &WEBOPTIONS_ERROR_UNKNOWN_OPTION,
            format!("weboptions: unknown option '{}'", name),
        )),
    }
}

fn parse_content_type_option(value: &Value) -> BuiltinResult<String> {
    let text = expect_string_scalar(
        value,
        "weboptions: ContentType must be a character vector or string scalar",
        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
    )?;
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok("auto".to_string()),
        "json" => Ok("json".to_string()),
        "text" | "char" | "string" => Ok("text".to_string()),
        "binary" | "raw" | "octet-stream" => Ok("binary".to_string()),
        other => Err(weboptions_error_with(
            &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!(
                "weboptions: unsupported ContentType '{}'; use 'auto', 'json', 'text', or 'binary'",
                other
            ),
        )),
    }
}

fn parse_request_method_option(value: &Value) -> BuiltinResult<String> {
    let text = expect_string_scalar(
        value,
        "weboptions: RequestMethod must be a character vector or string scalar",
        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
    )?;
    let lower = text.trim().to_ascii_lowercase();
    match lower.as_str() {
        "auto" | "get" | "post" | "put" | "patch" | "delete" => Ok(lower),
        _ => Err(weboptions_error_with(
            &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!(
                "weboptions: unsupported RequestMethod '{}'; expected auto, get, post, put, patch, or delete",
                text
            ),
        )),
    }
}

fn canonical_header_fields(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::Struct(struct_value) => {
            let mut out = StructValue::new();
            for (key, val) in &struct_value.fields {
                let header_value = expect_string_scalar(
                    val,
                    "weboptions: HeaderFields values must be character vectors or string scalars",
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                )?;
                if header_value.trim().is_empty() {
                    return Err(weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        "weboptions: header values must not be empty",
                    ));
                }
                if key.trim().is_empty() {
                    return Err(weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        "weboptions: header names must not be empty",
                    ));
                }
                out.fields.insert(key.clone(), Value::from(header_value));
            }
            Ok(Value::Struct(out))
        }
        Value::Cell(cell) => {
            if cell.cols != 2 {
                return Err(weboptions_error_with(
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    "weboptions: HeaderFields cell array must have exactly two columns",
                ));
            }
            let mut out = StructValue::new();
            for row in 0..cell.rows {
                let name_val = cell.get(row, 0).map_err(|err| {
                    weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        format!("weboptions: {err}"),
                    )
                })?;
                let value_val = cell.get(row, 1).map_err(|err| {
                    weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        format!("weboptions: {err}"),
                    )
                })?;

                let name = expect_string_scalar(
                    &name_val,
                    "weboptions: header names must be character vectors or string scalars",
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                )?;
                if name.trim().is_empty() {
                    return Err(weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        "weboptions: header names must not be empty",
                    ));
                }
                let header_value = expect_string_scalar(
                    &value_val,
                    "weboptions: header values must be character vectors or string scalars",
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                )?;
                if header_value.trim().is_empty() {
                    return Err(weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        "weboptions: header values must not be empty",
                    ));
                }
                out.fields.insert(name, Value::from(header_value));
            }
            Ok(Value::Struct(out))
        }
        _ => Err(weboptions_error_with(
            &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            "weboptions: HeaderFields must be a struct or two-column cell array",
        )),
    }
}

fn canonical_query_parameters(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::Struct(struct_value) => {
            let mut out = StructValue::new();
            for (key, val) in &struct_value.fields {
                out.fields.insert(key.clone(), val.clone());
            }
            Ok(Value::Struct(out))
        }
        Value::Cell(cell) => {
            if cell.cols != 2 {
                return Err(weboptions_error_with(
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    "weboptions: QueryParameters cell array must have exactly two columns",
                ));
            }
            let mut out = StructValue::new();
            for row in 0..cell.rows {
                let name_val = cell.get(row, 0).map_err(|err| {
                    weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        format!("weboptions: {err}"),
                    )
                })?;
                let value_val = cell.get(row, 1).map_err(|err| {
                    weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        format!("weboptions: {err}"),
                    )
                })?;
                let name = expect_string_scalar(
                    &name_val,
                    "weboptions: query parameter names must be character vectors or string scalars",
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                )?;
                out.fields.insert(name, value_val);
            }
            Ok(Value::Struct(out))
        }
        _ => Err(weboptions_error_with(
            &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            "weboptions: QueryParameters must be a struct or two-column cell array",
        )),
    }
}

fn validate_credentials(options: &StructValue) -> BuiltinResult<()> {
    let username = string_field(options, "Username").unwrap_or_default();
    let password = string_field(options, "Password").unwrap_or_default();
    if !password.trim().is_empty() && username.trim().is_empty() {
        return Err(weboptions_error_with(
            &WEBOPTIONS_ERROR_INVALID_CREDENTIALS,
            "weboptions: Password requires a Username option",
        ));
    }
    Ok(())
}

fn string_field(options: &StructValue, field: &str) -> Option<String> {
    options.fields.get(field).and_then(|value| match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        _ => None,
    })
}

fn numeric_scalar(
    value: &Value,
    context: &str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(tensor) => {
            if tensor::is_scalar_tensor(tensor) {
                Ok(tensor::tensor_value_f64(tensor, 0))
            } else {
                Err(weboptions_error_with(error, context))
            }
        }
        _ => Err(weboptions_error_with(error, context)),
    }
}

fn expect_string_scalar(
    value: &Value,
    context: &str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(weboptions_error_with(error, context)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::mpsc;
    use std::thread;

    use crate::call_builtin_async;
    use runmat_value::CellArray;

    fn spawn_server<F>(handler: F) -> String
    where
        F: FnOnce(TcpStream) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                handler(stream);
            }
        });
        format!("http://{}", addr)
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_weboptions(rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(weboptions_builtin(rest))
    }

    fn run_call_builtin(name: &str, args: &[Value]) -> BuiltinResult<Value> {
        futures::executor::block_on(call_builtin_async(name, args))
    }

    fn read_request(stream: &mut TcpStream) -> (String, Vec<u8>) {
        let mut buffer = Vec::new();
        let mut tmp = [0u8; 512];
        loop {
            match stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => {
                    buffer.extend_from_slice(&tmp[..n]);
                    if buffer.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        let header_end = buffer
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|idx| idx + 4)
            .unwrap_or(buffer.len());
        let headers = String::from_utf8_lossy(&buffer[..header_end]).to_string();
        let body = buffer[header_end..].to_vec();
        (headers, body)
    }

    fn respond_with(mut stream: TcpStream, content_type: &str, body: &[u8]) {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: {}\r\nConnection: close\r\n\r\n",
            body.len(),
            content_type
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.write_all(body);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = WEBOPTIONS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"options = weboptions()"));
        assert!(labels.contains(&"options = weboptions(name, value, ...)"));
        assert!(labels.contains(&"options = weboptions(optionsStruct, name, value, ...)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_default_struct_matches_expected_fields() {
        let result = run_weboptions(Vec::new()).expect("weboptions");
        let Value::Struct(options) = result else {
            panic!("expected struct result");
        };
        assert_eq!(
            options.fields.get("ContentType").and_then(|v| match v {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            }),
            Some("auto")
        );
        assert_eq!(
            options.fields.get("Timeout").and_then(|v| match v {
                Value::Num(n) => Some(*n),
                _ => None,
            }),
            Some(DEFAULT_TIMEOUT_SECONDS)
        );
        match options.fields.get("HeaderFields") {
            Some(Value::Struct(headers)) => assert!(headers.fields.is_empty()),
            other => panic!("expected empty HeaderFields struct, got {other:?}"),
        }
        assert_eq!(
            options.fields.get("RequestMethod").and_then(|v| match v {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            }),
            Some("auto")
        );
        assert_eq!(
            options.fields.get("MediaType").and_then(|v| match v {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            }),
            Some("auto")
        );
    }

    #[test]
    fn weboptions_timeout_reads_typed_integer_tensor_storage_exactly() {
        let timeout = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U16(vec![2026]),
            vec![1, 1],
        )
        .expect("typed timeout");

        let result = run_weboptions(vec![Value::from("Timeout"), Value::Tensor(timeout)])
            .expect("weboptions timeout");
        let Value::Struct(options) = result else {
            panic!("expected options struct");
        };
        assert_eq!(
            options.fields.get("Timeout").and_then(|v| match v {
                Value::Num(n) => Some(*n),
                _ => None,
            }),
            Some(2026.0)
        );
    }

    #[test]
    fn weboptions_preserves_exact_integer_key_value() {
        let value = Value::Int(runmat_value::IntValue::U64(u64::MAX));
        let result = run_weboptions(vec![
            Value::from("KeyName"),
            Value::from("sequence"),
            Value::from("KeyValue"),
            value.clone(),
        ])
        .expect("integer key value");
        let Value::Struct(options) = result else {
            panic!("expected options struct");
        };
        assert_eq!(options.fields.get("KeyValue"), Some(&value));
    }

    #[test]
    fn weboptions_runmat_only_forms_are_independently_gated() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let query_error = run_weboptions(vec![
            Value::from("QueryParameters"),
            Value::Struct(StructValue::new()),
        ])
        .expect_err("query property gate");
        assert_eq!(
            query_error.identifier(),
            Some("RunMat:compatibility:WeboptionsQueryParametersExtension")
        );

        let copy_error =
            run_weboptions(vec![Value::Struct(StructValue::new())]).expect_err("struct copy gate");
        assert_eq!(
            copy_error.identifier(),
            Some("RunMat:compatibility:WeboptionsStructCopyExtension")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_overrides_timeout_and_headers() {
        let mut headers = StructValue::new();
        headers
            .fields
            .insert("Accept".to_string(), Value::from("application/json"));
        headers
            .fields
            .insert("X-Client".to_string(), Value::from("RunMat"));
        let args = vec![
            Value::from("Timeout"),
            Value::Num(10.0),
            Value::from("HeaderFields"),
            Value::Struct(headers),
        ];
        let result = run_weboptions(args).expect("weboptions overrides");
        let Value::Struct(opts) = result else {
            panic!("expected struct");
        };
        assert_eq!(
            opts.fields.get("Timeout").and_then(|v| match v {
                Value::Num(n) => Some(*n),
                _ => None,
            }),
            Some(10.0)
        );
        match opts.fields.get("HeaderFields") {
            Some(Value::Struct(headers)) => {
                assert_eq!(
                    headers.fields.get("Accept"),
                    Some(&Value::from("application/json"))
                );
                assert_eq!(headers.fields.get("X-Client"), Some(&Value::from("RunMat")));
            }
            other => panic!("expected header struct, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_updates_existing_struct() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let base = run_weboptions(vec![Value::from("ContentType"), Value::from("json")])
            .expect("base weboptions");
        let args = vec![base, Value::from("Timeout"), Value::Num(15.0)];
        let updated = run_weboptions(args).expect("weboptions update");
        let Value::Struct(opts) = updated else {
            panic!("expected struct");
        };
        assert_eq!(
            opts.fields.get("ContentType").and_then(|v| match v {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            }),
            Some("json")
        );
        assert_eq!(
            opts.fields.get("Timeout").and_then(|v| match v {
                Value::Num(n) => Some(*n),
                _ => None,
            }),
            Some(15.0)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_rejects_unknown_option() {
        let err = error_message(
            run_weboptions(vec![Value::from("BogusOption"), Value::Num(1.0)])
                .expect_err("unknown option should fail"),
        );
        assert!(err.contains("unknown option"), "unexpected error: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_requires_username_when_password_provided() {
        let err = error_message(
            run_weboptions(vec![Value::from("Password"), Value::from("secret")])
                .expect_err("password without username"),
        );
        assert!(
            err.contains("Password requires a Username option"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_rejects_timeout_nonpositive() {
        let err = error_message(
            run_weboptions(vec![Value::from("Timeout"), Value::Num(0.0)])
                .expect_err("timeout should reject nonpositive values"),
        );
        assert!(
            err.contains("Timeout must be a positive numeric scalar"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_rejects_headerfields_bad_cell_shape() {
        let cell = CellArray::new(vec![Value::from("Accept")], 1, 1).expect("cell");
        let err = error_message(
            run_weboptions(vec![Value::from("HeaderFields"), Value::Cell(cell)])
                .expect_err("headerfields cell shape"),
        );
        assert!(
            err.contains("HeaderFields cell array must have exactly two columns"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webread_uses_weboptions_without_polluting_query() {
        let options = run_weboptions(Vec::new()).expect("weboptions");
        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let (headers, _) = read_request(&mut stream);
            tx.send(headers).unwrap();
            respond_with(stream, "application/json", br#"{"ok":true}"#);
        });

        let args = vec![Value::from(url.clone()), options];
        let result = run_call_builtin("webread", &args).expect("webread with options");
        match result {
            Value::Struct(reply) => {
                assert!(matches!(reply.fields.get("ok"), Some(Value::Bool(true))));
            }
            other => panic!("expected struct response, got {other:?}"),
        }
        let headers = rx.recv().expect("captured headers");
        assert!(headers.starts_with("GET "));
        assert!(
            !headers.contains("MediaType=auto"),
            "MediaType should not appear in query string"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webwrite_uses_weboptions_auto_request_method() {
        let options = run_weboptions(Vec::new()).expect("weboptions default");
        let payload = Value::from("Hello from RunMat");
        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let (headers, body) = read_request(&mut stream);
            tx.send((headers, body)).unwrap();
            respond_with(stream, "application/json", br#"{"ack":true}"#);
        });

        let args = vec![Value::from(url), payload, options];
        let result = run_call_builtin("webwrite", &args).expect("webwrite with weboptions");
        match result {
            Value::Struct(reply) => {
                assert!(matches!(reply.fields.get("ack"), Some(Value::Bool(true))));
            }
            other => panic!("expected struct response, got {other:?}"),
        }
        let (headers, body) = rx.recv().expect("request captured");
        assert!(
            headers.starts_with("POST "),
            "expected POST request, got headers: {headers}"
        );
        assert!(
            !body.is_empty(),
            "expected request body to be present when posting form data"
        );
    }
}
