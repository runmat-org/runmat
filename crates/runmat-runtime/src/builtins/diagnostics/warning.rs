//! MATLAB-compatible `warning` builtin with state management and formatting support.

#[cfg(test)]
use once_cell::sync::Lazy;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, StructValue, Value};
#[cfg(test)]
use runmat_value::{IntValue, IntegerStorage, Tensor};
use std::convert::TryFrom;
#[cfg(test)]
use std::sync::Mutex;

use crate::builtins::common::format::format_variadic;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::diagnostics::type_resolvers::warning_type;
use crate::warnings::{self, WarningMode, WarningRequest, WarningState};
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "warning";

const INTEGER_ARRAY_FORMATTING_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "warning-integer-array-formatting",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "warning with a nonscalar typed-integer formatting argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WarningIntegerArrayFormattingExtension"),
};
pub const WARNING_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [INTEGER_ARRAY_FORMATTING_EXTENSION];

const INTEGER_SCALAR_FORMAT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric-scalar replacement value includes every built-in integer class; integer conversions format the authoritative value directly.",
    }];
const INTEGER_ARRAY_FORMAT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat can consume a host integer array across successive format conversions; this exceeds the warning page's numeric-scalar A contract and is independently gated.",
    }];
pub const WARNING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "warning(msg, integer_A, ...)",
        inputs: &INTEGER_SCALAR_FORMAT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Signed and unsigned decimal conversions preserve exact values, including uint64 values above flintmax; warning state and the success sentinel remain host metadata.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "warning(msg, integer_array_A, ...)",
        inputs: &INTEGER_ARRAY_FORMAT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The extension gate runs before formatting or warning-state mutation; host values are flattened in column-major order and retain exact integer formatting.",
    },
];

const WARNING_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "state_or_status",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric success sentinel or state/status struct/cell/string result.",
}];

const WARNING_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const WARNING_INPUTS_MESSAGE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "message",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Warning message text or command token.",
}];
const WARNING_INPUTS_MESSAGE_VARIADIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Warning message template text.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Formatting values for the warning message template.",
    },
];
const WARNING_INPUTS_IDENTIFIER_MESSAGE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "message_id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"RunMat:warning\""),
        description: "Warning identifier.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Warning message text.",
    },
];
const WARNING_INPUTS_IDENTIFIER_MESSAGE_VARIADIC: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "message_id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"RunMat:warning\""),
        description: "Warning identifier.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Warning message template text.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Formatting values for the warning message template.",
    },
];
const WARNING_INPUTS_STATE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "state",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "State struct/cell snapshot to restore.",
}];
const WARNING_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Mode token ('on','off','once','error','default','reset','query','status','backtrace').",
}];
const WARNING_INPUTS_MODE_TARGET: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mode token.",
    },
    BuiltinParamDescriptor {
        name: "target",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"all\""),
        description: "Identifier/special target for mode updates or queries.",
    },
];
const WARNING_INPUTS_BACKTRACE_STATE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "command",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"backtrace\""),
        description: "Backtrace command token.",
    },
    BuiltinParamDescriptor {
        name: "state",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"off\""),
        description: "Backtrace state ('on' or 'off').",
    },
];

const WARNING_SIGNATURES: [BuiltinSignatureDescriptor; 17] = [
    BuiltinSignatureDescriptor {
        label: "state = warning()",
        inputs: &WARNING_INPUTS_NONE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(message)",
        inputs: &WARNING_INPUTS_MESSAGE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(message, A...)",
        inputs: &WARNING_INPUTS_MESSAGE_VARIADIC,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(message_id, message)",
        inputs: &WARNING_INPUTS_IDENTIFIER_MESSAGE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(message_id, message, A...)",
        inputs: &WARNING_INPUTS_IDENTIFIER_MESSAGE_VARIADIC,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(state)",
        inputs: &WARNING_INPUTS_STATE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(mode)",
        inputs: &WARNING_INPUTS_MODE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(mode, target)",
        inputs: &WARNING_INPUTS_MODE_TARGET,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"default\")",
        inputs: &WARNING_INPUTS_MODE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"default\", target)",
        inputs: &WARNING_INPUTS_MODE_TARGET,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"reset\")",
        inputs: &WARNING_INPUTS_MODE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"query\")",
        inputs: &WARNING_INPUTS_MODE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"query\", target)",
        inputs: &WARNING_INPUTS_MODE_TARGET,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"status\")",
        inputs: &WARNING_INPUTS_MODE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"backtrace\")",
        inputs: &WARNING_INPUTS_MODE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"backtrace\", state)",
        inputs: &WARNING_INPUTS_BACKTRACE_STATE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(mex)",
        inputs: &WARNING_INPUTS_STATE,
        outputs: &WARNING_OUTPUT,
    },
];

const WARNING_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WARNING.INVALID_INPUT",
    identifier: Some("RunMat:warning"),
    when: "Arguments are invalid for the warning parser branch or command contract.",
    message: "warning: invalid input arguments",
};

const WARNING_ERROR_PROMOTED_TO_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WARNING.PROMOTED_TO_ERROR",
    identifier: Some("RunMat:warning"),
    when: "Warning mode is configured to promote warnings to errors.",
    message: "warning: promoted to error",
};

const WARNING_ERRORS: [BuiltinErrorDescriptor; 2] =
    [WARNING_ERROR_INVALID_INPUT, WARNING_ERROR_PROMOTED_TO_ERROR];

pub const WARNING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WARNING_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WARNING_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::diagnostics::warning")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "warning",
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
    notes: "Control-flow builtin; GPU backends are never invoked.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::diagnostics::warning")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "warning",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Control-flow builtin; excluded from fusion planning.",
};

fn warning_default_identifier() -> &'static str {
    WARNING_ERROR_INVALID_INPUT
        .identifier
        .expect("warning default identifier must be defined")
}

fn warning_default_error(message: impl Into<String>) -> RuntimeError {
    warning_error_with_message(message, &WARNING_ERROR_INVALID_INPUT)
}

fn warning_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(normalize_identifier(identifier));
    }
    builder.build()
}

fn remap_warning_flow<F>(
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
    name = "warning",
    category = "diagnostics",
    summary = "Emit warnings and manage warning states by identifier.",
    keywords = "warning,diagnostics,state,query,backtrace",
    accel = "metadata",
    sink = true,
    suppress_auto_output = true,
    type_resolver(warning_type),
    descriptor(crate::builtins::diagnostics::warning::WARNING_DESCRIPTOR),
    extensions(crate::builtins::diagnostics::warning::WARNING_EXTENSIONS),
    integer_capabilities(crate::builtins::diagnostics::warning::WARNING_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::diagnostics::warning"
)]
fn warning_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return handle_query_default();
    }

    let first = &args[0];
    let rest = &args[1..];

    match first {
        Value::Struct(_) | Value::Cell(_) => {
            if !rest.is_empty() {
                return Err(warning_default_error(
                    "warning: state restoration accepts a single argument",
                ));
            }
            apply_state_value(first)?;
            Ok(Value::Num(0.0))
        }
        Value::MException(mex) => {
            if !rest.is_empty() {
                return Err(warning_default_error(
                    "warning: additional arguments are not allowed when passing an MException",
                ));
            }
            Ok(reissue_exception(mex)?)
        }
        _ => {
            let first_string = value_to_string("warning", first)?;
            if let Some(command) = parse_command(&first_string) {
                return handle_command(command, rest);
            }
            Ok(handle_message_call(None, first_string, rest)?)
        }
    }
}

fn handle_message_call(
    explicit_identifier: Option<String>,
    first_string: String,
    rest: &[Value],
) -> crate::BuiltinResult<Value> {
    if let Some(identifier) = explicit_identifier {
        return emit_warning(&identifier, &first_string, rest);
    }

    if rest.is_empty() {
        emit_warning(warning_default_identifier(), &first_string, rest)
    } else if is_message_identifier(&first_string) {
        let fmt = value_to_string("warning", &rest[0])?;
        let args = &rest[1..];
        emit_warning(&first_string, &fmt, args)
    } else {
        emit_warning(warning_default_identifier(), &first_string, rest)
    }
}

fn emit_warning(identifier_raw: &str, fmt: &str, args: &[Value]) -> crate::BuiltinResult<Value> {
    if args.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some() && tensor.len() != 1)
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_ARRAY_FORMATTING_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let identifier = normalize_identifier(identifier_raw);
    let message = format_variadic(fmt, args).map_err(|flow| {
        remap_warning_flow(flow, &WARNING_ERROR_INVALID_INPUT, |err| {
            err.message().to_string()
        })
    })?;

    warnings::emit(WarningRequest {
        builtin: BUILTIN_NAME,
        identifier: &identifier,
        message: &message,
        show_identifier: identifier != warning_default_identifier(),
    })?;
    Ok(Value::Num(0.0))
}

fn reissue_exception(mex: &runmat_value::MException) -> crate::BuiltinResult<Value> {
    let identifier = normalize_identifier(&mex.identifier);
    emit_warning(&identifier, &mex.message, &[])
}

fn handle_command(command: Command, rest: &[Value]) -> crate::BuiltinResult<Value> {
    match command {
        Command::SetMode(mode) => set_mode_command(mode, rest),
        Command::Default => default_command(rest),
        Command::Reset => {
            if !rest.is_empty() {
                return Err(warning_default_error(
                    "warning: 'reset' does not accept additional arguments",
                ));
            }
            warnings::with_policy(|policy| policy.reset());
            Ok(Value::Num(0.0))
        }
        Command::Query => query_command(rest),
        Command::Status => status_command(rest),
        Command::Backtrace => backtrace_command(rest),
    }
}

fn set_mode_command(mode: WarningMode, rest: &[Value]) -> crate::BuiltinResult<Value> {
    let identifier = if rest.is_empty() {
        "all".to_string()
    } else if rest.len() == 1 {
        value_to_string("warning", &rest[0])?
    } else {
        return Err(warning_default_error(
            "warning: too many input arguments for state change",
        ));
    };

    let trimmed = identifier.trim();
    if trimmed.eq_ignore_ascii_case("all") {
        let previous = warnings::with_policy(|policy| policy.set_global_mode(mode));
        return Ok(state_value("all", previous));
    }

    if trimmed.eq_ignore_ascii_case("last") {
        let last_warning = warnings::with_policy(|policy| policy.last_warning());
        let Some(last_warning) = last_warning else {
            return Err(warning_default_error(
                "warning: there is no last warning identifier to target",
            ));
        };
        return set_mode_for_identifier(mode, &last_warning.identifier);
    }

    if trimmed.eq_ignore_ascii_case("backtrace") || trimmed.eq_ignore_ascii_case("verbose") {
        return set_mode_for_special_mode(mode, trimmed);
    }

    let normalized = normalize_identifier(trimmed);
    set_mode_for_identifier(mode, &normalized)
}

fn set_mode_for_identifier(mode: WarningMode, identifier: &str) -> crate::BuiltinResult<Value> {
    let previous = warnings::with_policy(|policy| policy.set_identifier_mode(identifier, mode));
    Ok(state_value(identifier, previous))
}

fn reset_identifier_to_default(identifier: &str) -> crate::BuiltinResult<Value> {
    let previous = warnings::with_policy(|policy| policy.clear_identifier(identifier));
    Ok(state_value(identifier, previous))
}

fn set_mode_for_special_mode(mode: WarningMode, mode_name: &str) -> crate::BuiltinResult<Value> {
    let mode_lower = mode_name.trim().to_ascii_lowercase();
    if !matches!(mode, WarningMode::On | WarningMode::Off) {
        return Err(warning_default_error(format!(
            "warning: only 'on' or 'off' are valid states for '{mode_lower}'"
        )));
    }

    let enabled = matches!(mode, WarningMode::On);
    let previous_enabled = match mode_lower.as_str() {
        "backtrace" => warnings::with_policy(|policy| policy.set_backtrace(enabled)),
        "verbose" => warnings::with_policy(|policy| policy.set_verbose(enabled)),
        _ => {
            return Err(warning_default_error(format!(
                "warning: unknown mode '{}'; expected 'backtrace' or 'verbose'",
                mode_name
            )))
        }
    };
    Ok(state_struct_value(
        &mode_lower,
        if previous_enabled { "on" } else { "off" },
    ))
}

fn default_command(rest: &[Value]) -> crate::BuiltinResult<Value> {
    match rest.len() {
        0 => {
            let snapshot = warnings::with_policy(|policy| policy.reset_defaults());
            structs_to_cell(snapshot)
        }
        1 => {
            let identifier = value_to_string("warning", &rest[0])?;
            let trimmed = identifier.trim();
            if trimmed.eq_ignore_ascii_case("all") {
                let snapshot = warnings::with_policy(|policy| policy.reset_defaults());
                return structs_to_cell(snapshot);
            }
            if trimmed.eq_ignore_ascii_case("backtrace") {
                let previous = warnings::with_policy(|policy| policy.set_backtrace(false));
                return Ok(state_struct_value(
                    "backtrace",
                    if previous { "on" } else { "off" },
                ));
            }
            if trimmed.eq_ignore_ascii_case("verbose") {
                let previous = warnings::with_policy(|policy| policy.set_verbose(false));
                return Ok(state_struct_value(
                    "verbose",
                    if previous { "on" } else { "off" },
                ));
            }
            if trimmed.eq_ignore_ascii_case("last") {
                let last_warning = warnings::with_policy(|policy| policy.last_warning());
                let Some(last_warning) = last_warning else {
                    return Err(warning_default_error(
                        "warning: there is no last warning identifier to reset to default",
                    ));
                };
                return reset_identifier_to_default(&last_warning.identifier);
            }
            let normalized = normalize_identifier(trimmed);
            reset_identifier_to_default(&normalized)
        }
        _ => Err(warning_default_error(
            "warning: 'default' accepts zero or one identifier argument",
        )),
    }
}

fn query_command(rest: &[Value]) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(warning_default_error(
            "warning: 'query' accepts at most one identifier argument",
        ));
    }

    let target = if rest.is_empty() {
        "all".to_string()
    } else {
        value_to_string("warning", &rest[0])?
    };

    if target.trim().eq_ignore_ascii_case("all") {
        return structs_to_cell(warnings::with_policy(|policy| policy.snapshot()));
    }
    if target.trim().eq_ignore_ascii_case("last") {
        return Ok(
            warnings::with_policy(|policy| policy.last_warning()).map_or_else(
                || {
                    let mut st = StructValue::new();
                    st.fields.insert("identifier".to_string(), Value::from(""));
                    st.fields.insert("message".to_string(), Value::from(""));
                    st.fields.insert("state".to_string(), Value::from("none"));
                    Value::Struct(st)
                },
                |warning| {
                    let mut st = StructValue::new();
                    st.fields
                        .insert("identifier".to_string(), Value::from(warning.identifier));
                    st.fields
                        .insert("message".to_string(), Value::from(warning.message));
                    st.fields.insert("state".to_string(), Value::from("last"));
                    Value::Struct(st)
                },
            ),
        );
    }
    if target.trim().eq_ignore_ascii_case("backtrace") {
        let enabled = warnings::with_policy(|policy| policy.backtrace_enabled());
        return Ok(state_struct_value(
            "backtrace",
            if enabled { "on" } else { "off" },
        ));
    }
    if target.trim().eq_ignore_ascii_case("verbose") {
        let enabled = warnings::with_policy(|policy| policy.verbose_enabled());
        return Ok(state_struct_value(
            "verbose",
            if enabled { "on" } else { "off" },
        ));
    }
    let normalized = normalize_identifier(&target);
    let mode = warnings::with_policy(|policy| policy.lookup_mode(&normalized));
    Ok(state_value(&normalized, mode))
}

fn status_command(rest: &[Value]) -> crate::BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(warning_default_error(
            "warning: 'status' does not accept additional arguments",
        ));
    }
    let value = query_command(&[])?;
    match &value {
        Value::Cell(cell) => {
            emit_status_line("Warning status:".to_string());
            for idx in 0..cell.data.len() {
                let entry = cell.data[idx].clone();
                if let Value::Struct(st) = entry {
                    let identifier = st
                        .fields
                        .get("identifier")
                        .and_then(|v| value_to_string("warning", v).ok())
                        .unwrap_or_default();
                    let state = st
                        .fields
                        .get("state")
                        .and_then(|v| value_to_string("warning", v).ok())
                        .unwrap_or_default();
                    emit_status_line(format!("  {identifier}: {state}"));
                }
            }
        }
        Value::Struct(st) => {
            let identifier = st
                .fields
                .get("identifier")
                .and_then(|v| value_to_string("warning", v).ok())
                .unwrap_or_default();
            let state = st
                .fields
                .get("state")
                .and_then(|v| value_to_string("warning", v).ok())
                .unwrap_or_default();
            emit_status_line(format!("Warning status -> {identifier}: {state}"));
        }
        _ => {}
    }
    Ok(value)
}

fn backtrace_command(rest: &[Value]) -> crate::BuiltinResult<Value> {
    match rest.len() {
        0 => {
            let state = warnings::with_policy(|policy| {
                if policy.backtrace_enabled() {
                    "on"
                } else {
                    "off"
                }
            });
            Ok(Value::from(state))
        }
        1 => {
            let setting = value_to_string("warning", &rest[0])?;
            match setting.trim().to_ascii_lowercase().as_str() {
                "on" => {
                    warnings::with_policy(|policy| policy.set_backtrace(true));
                }
                "off" => {
                    warnings::with_policy(|policy| policy.set_backtrace(false));
                }
                other => {
                    return Err(warning_default_error(format!(
                        "warning: backtrace mode must be 'on' or 'off', got '{other}'"
                    )))
                }
            }
            Ok(Value::Num(0.0))
        }
        _ => Err(warning_default_error(
            "warning: 'backtrace' accepts zero or one argument",
        )),
    }
}

fn handle_query_default() -> crate::BuiltinResult<Value> {
    query_command(&[])
}

fn apply_state_value(value: &Value) -> crate::BuiltinResult<()> {
    match value {
        Value::Struct(st) => apply_state_struct(st),
        Value::Cell(cell) => {
            for idx in 0..cell.data.len() {
                let entry = cell.data[idx].clone();
                apply_state_value(&entry)?;
            }
            Ok(())
        }
        other => Err(warning_default_error(format!(
            "warning: expected a struct or cell array of structs, got {other:?}"
        ))),
    }
}

fn apply_state_struct(st: &StructValue) -> crate::BuiltinResult<()> {
    let identifier_value = st.fields.get("identifier").ok_or_else(|| {
        warning_default_error("warning: state struct must contain an 'identifier' field")
    })?;
    let state_value = st.fields.get("state").ok_or_else(|| {
        warning_default_error("warning: state struct must contain a 'state' field")
    })?;
    let identifier_raw = value_to_string("warning", identifier_value)?;
    let state_raw = value_to_string("warning", state_value)?;
    let identifier_trimmed = identifier_raw.trim();
    if identifier_trimmed.eq_ignore_ascii_case("all") {
        if let Some(mode) = parse_mode_keyword(&state_raw) {
            warnings::with_policy(|policy| policy.set_global_mode(mode));
        } else {
            return Err(warning_default_error(format!(
                "warning: unknown state '{}'",
                state_raw
            )));
        }
    } else if identifier_trimmed.eq_ignore_ascii_case("backtrace") {
        let state = state_raw.trim().to_ascii_lowercase();
        match state.as_str() {
            "on" => {
                warnings::with_policy(|policy| policy.set_backtrace(true));
            }
            "off" | "default" => {
                warnings::with_policy(|policy| policy.set_backtrace(false));
            }
            other => {
                return Err(warning_default_error(format!(
                    "warning: unknown backtrace state '{}'",
                    other
                )))
            }
        }
    } else if identifier_trimmed.eq_ignore_ascii_case("verbose") {
        let state = state_raw.trim().to_ascii_lowercase();
        match state.as_str() {
            "on" => {
                warnings::with_policy(|policy| policy.set_verbose(true));
            }
            "off" | "default" => {
                warnings::with_policy(|policy| policy.set_verbose(false));
            }
            other => {
                return Err(warning_default_error(format!(
                    "warning: unknown verbose state '{}'",
                    other
                )))
            }
        }
    } else if identifier_trimmed.eq_ignore_ascii_case("last") {
        let last_warning = warnings::with_policy(|policy| policy.last_warning());
        let Some(last_warning) = last_warning else {
            return Err(warning_default_error(
                "warning: there is no last warning identifier to apply state",
            ));
        };
        if state_raw.trim().eq_ignore_ascii_case("default") {
            warnings::with_policy(|policy| policy.clear_identifier(&last_warning.identifier));
        } else if let Some(mode) = parse_mode_keyword(&state_raw) {
            warnings::with_policy(|policy| {
                policy.set_identifier_mode(&last_warning.identifier, mode)
            });
        } else {
            return Err(warning_default_error(format!(
                "warning: unknown state '{}'",
                state_raw
            )));
        }
    } else if state_raw.trim().eq_ignore_ascii_case("default") {
        let normalized = normalize_identifier(identifier_trimmed);
        warnings::with_policy(|policy| policy.clear_identifier(&normalized));
    } else if let Some(mode) = parse_mode_keyword(&state_raw) {
        let normalized = normalize_identifier(identifier_trimmed);
        warnings::with_policy(|policy| policy.set_identifier_mode(&normalized, mode));
    } else {
        return Err(warning_default_error(format!(
            "warning: unknown state '{}'",
            state_raw
        )));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Command {
    SetMode(WarningMode),
    Default,
    Reset,
    Query,
    Status,
    Backtrace,
}

fn parse_command(text: &str) -> Option<Command> {
    match text.trim().to_ascii_lowercase().as_str() {
        "on" => Some(Command::SetMode(WarningMode::On)),
        "off" => Some(Command::SetMode(WarningMode::Off)),
        "once" => Some(Command::SetMode(WarningMode::Once)),
        "error" => Some(Command::SetMode(WarningMode::Error)),
        "default" => Some(Command::Default),
        "reset" => Some(Command::Reset),
        "query" => Some(Command::Query),
        "status" => Some(Command::Status),
        "backtrace" => Some(Command::Backtrace),
        _ => None,
    }
}

fn parse_mode_keyword(text: &str) -> Option<WarningMode> {
    match text.trim().to_ascii_lowercase().as_str() {
        "on" => Some(WarningMode::On),
        "off" => Some(WarningMode::Off),
        "once" => Some(WarningMode::Once),
        "error" => Some(WarningMode::Error),
        _ => None,
    }
}

fn value_to_string(context: &str, value: &Value) -> crate::BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::CharArray(_) => Err(warning_default_error(format!(
            "{context}: expected scalar char array"
        ))),
        Value::StringArray(_) => Err(warning_default_error(format!(
            "{context}: expected scalar string"
        ))),
        other => String::try_from(other).map_err(|_| {
            warning_default_error(format!(
                "{context}: expected string-like argument, got {other:?}"
            ))
        }),
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

fn normalize_identifier(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        warning_default_identifier().to_string()
    } else if trimmed.contains(':') {
        trimmed.to_string()
    } else {
        format!("RunMat:{trimmed}")
    }
}

fn state_struct(identifier: &str, state: &str) -> StructValue {
    let mut st = StructValue::new();
    st.fields.insert(
        "identifier".to_string(),
        Value::from(identifier.to_string()),
    );
    st.fields
        .insert("state".to_string(), Value::from(state.to_string()));
    st
}

fn state_struct_value(identifier: &str, state: &str) -> Value {
    Value::Struct(state_struct(identifier, state))
}

fn state_value(identifier: &str, mode: WarningMode) -> Value {
    state_struct_value(identifier, mode.keyword())
}

fn structs_to_cell(states: Vec<WarningState>) -> crate::BuiltinResult<Value> {
    let structs: Vec<StructValue> = states
        .into_iter()
        .map(|state| state_struct(&state.identifier, state.mode.keyword()))
        .collect();
    let rows = structs.len();
    let values: Vec<Value> = structs.into_iter().map(Value::Struct).collect();
    CellArray::new(values, rows, 1)
        .map(Value::Cell)
        .map_err(|e| warning_default_error(format!("warning: failed to assemble state cell: {e}")))
}

fn emit_status_line(line: String) {
    tracing::warn!("{line}");
    crate::console::record_console_line(crate::console::ConsoleStream::Stderr, line);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{ResolveContext, Type};

    static TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    fn reset_manager() {
        warnings::with_policy(|policy| policy.reset());
    }

    fn last_warning() -> Option<crate::warning_store::RuntimeWarning> {
        warnings::with_policy(|policy| policy.last_warning())
    }

    fn assert_state_struct(value: &Value, identifier: &str, state: &str) {
        match value {
            Value::Struct(st) => {
                let id = st
                    .fields
                    .get("identifier")
                    .and_then(|v| String::try_from(v).ok())
                    .unwrap_or_default();
                let st_state = st
                    .fields
                    .get("state")
                    .and_then(|v| String::try_from(v).ok())
                    .unwrap_or_default();
                assert_eq!(id, identifier);
                assert_eq!(st_state, state);
            }
            other => panic!("expected state struct, got {other:?}"),
        }
    }

    fn structs_from_value(value: Value) -> Vec<StructValue> {
        match value {
            Value::Cell(cell) => cell
                .data
                .into_iter()
                .map(|value| match value {
                    Value::Struct(st) => st,
                    other => panic!("expected struct entry, got {other:?}"),
                })
                .collect(),
            Value::Struct(st) => vec![st],
            other => panic!("expected struct array, got {other:?}"),
        }
    }

    fn field_str(struct_value: &StructValue, field: &str) -> Option<String> {
        struct_value
            .fields
            .get(field)
            .and_then(|value| String::try_from(value).ok())
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn emits_basic_warning() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let result = warning_builtin(vec![Value::from("Hello world!")]).expect("warning ok");
        assert!(matches!(result, Value::Num(_)));
        let last = last_warning();
        assert_eq!(
            last.map(|warning| (warning.identifier, warning.message)),
            Some((
                warning_default_identifier().to_string(),
                "Hello world!".to_string()
            ))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn emits_warning_with_identifier_and_format() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let args = vec![
            Value::from("runmat:demo:test"),
            Value::from("value is %d"),
            Value::Int(runmat_value::IntValue::I32(7)),
        ];
        warning_builtin(args).expect("warning ok");
        let last = last_warning();
        assert_eq!(
            last.map(|warning| (warning.identifier, warning.message)),
            Some(("runmat:demo:test".to_string(), "value is 7".to_string()))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn off_suppresses_warning() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let state =
            warning_builtin(vec![Value::from("off"), Value::from("all")]).expect("state change");
        assert_state_struct(&state, "all", "on");
        warning_builtin(vec![Value::from("Should suppress")]).expect("warning ok");
        let last = last_warning();
        assert!(last.is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn once_only_emits_first_warning() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let state =
            warning_builtin(vec![Value::from("once"), Value::from("all")]).expect("state change");
        assert_state_struct(&state, "all", "on");
        warning_builtin(vec![Value::from("First")]).expect("warning ok");
        warning_builtin(vec![Value::from("Second")]).expect("warning ok");
        let last = last_warning();
        assert_eq!(
            last.map(|warning| (warning.identifier, warning.message)),
            Some((
                warning_default_identifier().to_string(),
                "First".to_string()
            ))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn error_mode_promotes_to_error() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let previous =
            warning_builtin(vec![Value::from("error"), Value::from("all")]).expect("state change");
        assert_state_struct(&previous, "all", "on");
        let err =
            unwrap_error(warning_builtin(vec![Value::from("Promoted")]).expect_err("should error"));
        assert_eq!(err.identifier(), Some(warning_default_identifier()));
        assert_eq!(err.message(), "Promoted");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn query_returns_state_struct() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        warning_builtin(vec![Value::from("off"), Value::from("runmat:demo:test")])
            .expect("state change");
        let value = warning_builtin(vec![Value::from("query"), Value::from("runmat:demo:test")])
            .expect("query ok");
        match value {
            Value::Struct(st) => {
                let state = st.fields.get("state").unwrap();
                assert_eq!(String::try_from(state).unwrap(), "off".to_string());
            }
            other => panic!("expected struct, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn state_struct_restores_mode() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let snapshot =
            warning_builtin(vec![Value::from("query"), Value::from("all")]).expect("query all");
        warning_builtin(vec![Value::from("off"), Value::from("all")]).expect("off all");
        warning_builtin(vec![snapshot]).expect("restore");
        let state = warning_builtin(vec![
            Value::from("query"),
            Value::from("runmat:demo:restored"),
        ])
        .expect("query")
        .expect_struct();
        assert_eq!(
            String::try_from(state.fields.get("state").unwrap()).unwrap(),
            "on".to_string()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn set_mode_backtrace_via_state() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let prev = warning_builtin(vec![Value::from("on"), Value::from("backtrace")])
            .expect("enable backtrace");
        assert_state_struct(&prev, "backtrace", "off");
        assert!(warnings::with_policy(|policy| policy.backtrace_enabled()));
        let prev = warning_builtin(vec![Value::from("off"), Value::from("backtrace")])
            .expect("disable backtrace");
        assert_state_struct(&prev, "backtrace", "on");
        assert!(!warnings::with_policy(|policy| policy.backtrace_enabled()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn set_mode_verbose_via_state() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let prev = warning_builtin(vec![Value::from("on"), Value::from("verbose")])
            .expect("enable verbose");
        assert_state_struct(&prev, "verbose", "off");
        assert!(warnings::with_policy(|policy| policy.verbose_enabled()));
        let prev = warning_builtin(vec![Value::from("off"), Value::from("verbose")])
            .expect("disable verbose");
        assert_state_struct(&prev, "verbose", "on");
        assert!(!warnings::with_policy(|policy| policy.verbose_enabled()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn special_mode_rejects_invalid_state() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let err = unwrap_error(
            warning_builtin(vec![Value::from("once"), Value::from("backtrace")])
                .expect_err("invalid state"),
        );
        assert_eq!(err.identifier(), Some(warning_default_identifier()));
        assert!(
            err.message().contains("only 'on' or 'off'"),
            "unexpected error message: {}",
            err.message()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn set_mode_last_requires_identifier() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let err = unwrap_error(
            warning_builtin(vec![Value::from("off"), Value::from("last")])
                .expect_err("missing last"),
        );
        assert_eq!(err.identifier(), Some(warning_default_identifier()));
        assert!(
            err.message().contains("no last warning identifier"),
            "unexpected error: {}",
            err.message()
        );
        warning_builtin(vec![Value::from("Hello!")]).expect("emit warning");
        let previous =
            warning_builtin(vec![Value::from("off"), Value::from("last")]).expect("disable last");
        assert_state_struct(&previous, warning_default_identifier(), "on");
        let last_mode =
            warnings::with_policy(|policy| policy.lookup_mode(warning_default_identifier()));
        assert!(matches!(last_mode, WarningMode::Off));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn default_returns_snapshot() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        warning_builtin(vec![
            Value::from("off"),
            Value::from("runmat:demo:snapshot"),
        ])
        .expect("state change");
        let snapshot = warning_builtin(vec![Value::from("default")]).expect("default");
        let structs = structs_from_value(snapshot);
        assert!(structs.iter().any(|st| {
            field_str(st, "identifier").as_deref() == Some("all")
                && field_str(st, "state").as_deref() == Some("on")
        }));
        assert!(structs.iter().any(|st| {
            field_str(st, "identifier").as_deref() == Some("runmat:demo:snapshot")
                && field_str(st, "state").as_deref() == Some("off")
        }));
        assert!(structs
            .iter()
            .any(|st| { field_str(st, "identifier").as_deref() == Some("backtrace") }));
        assert!(structs
            .iter()
            .any(|st| { field_str(st, "identifier").as_deref() == Some("verbose") }));
        assert!(!warnings::with_policy(
            |policy| policy.has_identifier_rules()
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn default_special_modes_reset() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        warning_builtin(vec![Value::from("on"), Value::from("verbose")]).expect("enable verbose");
        warning_builtin(vec![Value::from("on"), Value::from("backtrace")])
            .expect("enable backtrace");
        let verbose_prev =
            warning_builtin(vec![Value::from("default"), Value::from("verbose")]).expect("default");
        assert_state_struct(&verbose_prev, "verbose", "on");
        assert!(!warnings::with_policy(|policy| policy.verbose_enabled()));
        let backtrace_prev =
            warning_builtin(vec![Value::from("default"), Value::from("backtrace")])
                .expect("default");
        assert_state_struct(&backtrace_prev, "backtrace", "on");
        assert!(!warnings::with_policy(|policy| policy.backtrace_enabled()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn query_backtrace_and_verbose() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        warning_builtin(vec![Value::from("on"), Value::from("verbose")]).expect("enable verbose");
        let verbose = warning_builtin(vec![Value::from("query"), Value::from("verbose")])
            .expect("query verbose");
        assert_state_struct(&verbose, "verbose", "on");
        let backtrace = warning_builtin(vec![Value::from("query"), Value::from("backtrace")])
            .expect("query backtrace");
        assert_state_struct(&backtrace, "backtrace", "off");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn apply_state_struct_special_modes() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let mut backtrace = StructValue::new();
        backtrace
            .fields
            .insert("identifier".to_string(), Value::from("backtrace"));
        backtrace
            .fields
            .insert("state".to_string(), Value::from("on"));
        warning_builtin(vec![Value::Struct(backtrace)]).expect("apply backtrace");
        assert!(warnings::with_policy(|policy| policy.backtrace_enabled()));

        let mut verbose = StructValue::new();
        verbose
            .fields
            .insert("identifier".to_string(), Value::from("verbose"));
        verbose
            .fields
            .insert("state".to_string(), Value::from("default"));
        warning_builtin(vec![Value::Struct(verbose)]).expect("apply verbose");
        assert!(!warnings::with_policy(|policy| policy.verbose_enabled()));
    }

    #[test]
    fn warning_type_is_unknown() {
        assert_eq!(
            warning_type(&[Type::String], &ResolveContext::new(Vec::new())),
            Type::Unknown
        );
    }

    #[test]
    fn warning_formats_wide_integer_scalars_exactly() {
        let formatted =
            format_variadic("%u", &[Value::Int(IntValue::U64(u64::MAX))]).expect("format integer");
        assert_eq!(formatted, u64::MAX.to_string());
    }

    #[test]
    fn warning_gates_nonscalar_integer_format_arrays() {
        let _guard = TEST_LOCK.lock().expect("test lock");
        reset_manager();
        let values = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![1, u64::MAX]), vec![1, 2])
                .expect("integer values"),
        );
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error =
            warning_builtin(vec![Value::from("%u %u"), values]).expect_err("array formatting gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:WarningIntegerArrayFormattingExtension")
        );
    }

    trait ExpectStruct {
        fn expect_struct(self) -> StructValue;
    }

    impl ExpectStruct for Value {
        fn expect_struct(self) -> StructValue {
            match self {
                Value::Struct(st) => st,
                other => panic!("expected struct, got {other:?}"),
            }
        }
    }
}
