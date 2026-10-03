//! MATLAB-compatible `feof` builtin for RunMat.
//!
//! Provides end-of-file detection for file identifiers opened via `fopen`.
//! The implementation mirrors MATLAB semantics for host-side files and
//! integrates with the shared registry used by the other text I/O builtins.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::io::filetext::registry;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const IDENTIFIER_TYPE_ERROR_DETAIL: &str = "file identifier must be a numeric scalar";
const BUILTIN_NAME: &str = "feof";

const FEOF_INTEGER_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feof-integer-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "integer-class feof file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeofIntegerIdExtension"),
};
const FEOF_LOGICAL_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feof-logical-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "logical feof file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeofLogicalIdExtension"),
};
const FEOF_RESIDENT_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feof-resident-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "provider-resident feof file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeofResidentIdExtension"),
};
pub const FEOF_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FEOF_INTEGER_ID_EXTENSION,
    FEOF_LOGICAL_ID_EXTENSION,
    FEOF_RESIDENT_ID_EXTENSION,
];

const FEOF_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "MATLAB-compatible file IDs are integer-valued doubles; all eight typed integer classes are one gated RunMat extension and are checked exactly against i32 before registry access.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tf = feof(integer_fileID)",
        inputs: &FEOF_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The identifier is validated exactly before the registry query; resident and logical identifiers have independent RunMat-only gates, and the result is a host logical scalar.",
    }];

const FEOF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when positioned at end-of-file.",
}];
const FEOF_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier opened by fopen.",
}];
const FEOF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = feof(fid)",
    inputs: &FEOF_INPUTS,
    outputs: &FEOF_OUTPUT,
}];

const FEOF_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEOF.INVALID_INPUT",
    identifier: Some("RunMat:feof:InvalidInput"),
    when: "Input identifier is malformed or out of range.",
    message: "feof: invalid file identifier input",
};
const FEOF_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEOF.INVALID_IDENTIFIER",
    identifier: Some("RunMat:feof:InvalidIdentifier"),
    when: "Identifier does not refer to an open file handle.",
    message: "feof: invalid file identifier. Use fopen to generate a valid file ID.",
};
const FEOF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEOF.INTERNAL",
    identifier: None,
    when: "Internal control-flow conversion failed.",
    message: "feof: internal error",
};
const FEOF_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FEOF_ERROR_INVALID_INPUT,
    FEOF_ERROR_INVALID_IDENTIFIER,
    FEOF_ERROR_INTERNAL,
];
pub const FEOF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FEOF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FEOF_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::feof")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "feof",
    op_kind: GpuOpKind::Custom("file-io"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only file I/O query; providers are not involved.",
};

fn feof_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn feof_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    feof_error_with_message(error.message, error)
}

fn feof_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    feof_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FEOF_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::feof")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "feof",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O queries are not eligible for fusion; metadata registered for completeness.",
};

#[runtime_builtin(
    name = "feof",
    category = "io/filetext",
    summary = "Query end-of-file state for file identifiers.",
    keywords = "feof,end of file,io,file identifier",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::feof_type),
    descriptor(crate::builtins::io::filetext::feof::FEOF_DESCRIPTOR),
    integer_capabilities(crate::builtins::io::filetext::feof::INTEGER_CAPABILITIES),
    extensions(crate::builtins::io::filetext::feof::FEOF_EXTENSIONS),
    builtin_path = "crate::builtins::io::filetext::feof"
)]
async fn feof_builtin(fid: Value) -> crate::BuiltinResult<Value> {
    let at_end = evaluate(&fid).await?;
    Ok(Value::Bool(at_end))
}

/// Evaluate the `feof` builtin without invoking the runtime dispatcher.
pub async fn evaluate(fid_value: &Value) -> BuiltinResult<bool> {
    preflight_fid(fid_value)?;
    let fid_host = gather_if_needed_async(fid_value)
        .await
        .map_err(map_control_flow)?;
    preflight_fid(&fid_host)?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Ok(false);
    }

    registry::eof_encountered(fid).ok_or_else(|| feof_error(&FEOF_ERROR_INVALID_IDENTIFIER))
}

fn preflight_fid(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Num(_) => Ok(()),
        Value::Int(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FEOF_INTEGER_ID_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Bool(_) | Value::LogicalArray(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FEOF_LOGICAL_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FEOF_INTEGER_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64 => Ok(()),
        Value::Tensor(_) => Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier must be an integer-valued double scalar",
        )),
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FEOF_RESIDENT_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FEOF_INTEGER_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_is_logical(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FEOF_LOGICAL_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn parse_fid(value: &Value) -> BuiltinResult<i32> {
    match value {
        Value::Num(n) => parse_scalar_fid(*n),
        Value::Int(int) => int.try_to_i32().ok_or_else(|| {
            feof_error_with_detail(&FEOF_ERROR_INVALID_INPUT, "file identifier is out of range")
        }),
        Value::Tensor(t) => {
            if tensor::is_scalar_tensor(t) {
                if let Some(storage) = t.integer_storage() {
                    let value = storage.value_at(0).expect("one-element integer storage");
                    return value.try_to_i32().ok_or_else(|| {
                        feof_error_with_detail(
                            &FEOF_ERROR_INVALID_INPUT,
                            "file identifier is out of range",
                        )
                    });
                }
                if t.numeric_dtype() != NumericDType::F64 {
                    return Err(feof_error_with_detail(
                        &FEOF_ERROR_INVALID_INPUT,
                        "file identifier must be an integer-valued double scalar",
                    ));
                }
                parse_scalar_fid(tensor::tensor_value_f64(t, 0))
            } else {
                Err(feof_error_with_detail(
                    &FEOF_ERROR_INVALID_INPUT,
                    IDENTIFIER_TYPE_ERROR_DETAIL,
                ))
            }
        }
        Value::LogicalArray(la) if la.data.len() == 1 => {
            let v = if la.data[0] != 0 { 1.0 } else { 0.0 };
            parse_scalar_fid(v)
        }
        Value::LogicalArray(_) => Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            IDENTIFIER_TYPE_ERROR_DETAIL,
        )),
        Value::Bool(b) => parse_scalar_fid(if *b { 1.0 } else { 0.0 }),
        _ => Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            IDENTIFIER_TYPE_ERROR_DETAIL,
        )),
    }
}

fn parse_scalar_fid(value: f64) -> BuiltinResult<i32> {
    if !value.is_finite() {
        return Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier must be finite",
        ));
    }
    if value.fract() != 0.0 {
        return Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier must be an integer",
        ));
    }
    let rounded = value;
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err(feof_error_with_detail(
            &FEOF_ERROR_INVALID_INPUT,
            "file identifier is out of range",
        ));
    }
    Ok(rounded as i32)
}

#[cfg(test)]
pub(crate) mod tests;
