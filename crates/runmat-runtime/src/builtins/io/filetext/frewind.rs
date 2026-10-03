//! MATLAB-compatible `frewind` builtin for RunMat.
//!
//! Rewinds the file position indicator to the beginning of an open file.
//! Equivalent to `fseek(fid, 0, 'bof')`.

use std::io::{Seek, SeekFrom};

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
const BUILTIN_NAME: &str = "frewind";

const FREWIND_INTEGER_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "frewind-integer-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "integer-class frewind file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FrewindIntegerIdExtension"),
};
const FREWIND_SINGLE_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "frewind-single-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "single-precision frewind file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FrewindSingleIdExtension"),
};
const FREWIND_LOGICAL_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "frewind-logical-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "logical frewind file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FrewindLogicalIdExtension"),
};
const FREWIND_RESIDENT_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "frewind-resident-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "provider-resident frewind file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FrewindResidentIdExtension"),
};
pub const FREWIND_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FREWIND_INTEGER_ID_EXTENSION,
    FREWIND_SINGLE_ID_EXTENSION,
    FREWIND_LOGICAL_ID_EXTENSION,
    FREWIND_RESIDENT_ID_EXTENSION,
];
const FREWIND_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents double identifiers; all eight typed integer classes are one exact, independently gated extension.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "frewind(integer_fileID)",
        inputs: &FREWIND_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The identifier is decoded exactly before the host registry seek and the builtin returns no output.",
    }];

const FREWIND_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const FREWIND_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier opened by fopen.",
}];
const FREWIND_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "frewind(fid)",
    inputs: &FREWIND_INPUTS,
    outputs: &FREWIND_OUTPUTS_NONE,
}];

const FREWIND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREWIND.INVALID_INPUT",
    identifier: Some("RunMat:frewind:InvalidInput"),
    when: "Input identifier is malformed or out of range.",
    message: "frewind: invalid input arguments",
};
const FREWIND_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREWIND.INVALID_IDENTIFIER",
    identifier: Some("RunMat:frewind:InvalidIdentifier"),
    when: "Identifier does not refer to an open file handle.",
    message: "frewind: invalid file identifier. Use fopen to generate a valid file ID.",
};
const FREWIND_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREWIND.IO",
    identifier: Some("RunMat:frewind:IoFailure"),
    when: "Underlying seek/rewind operation fails.",
    message: "frewind: file I/O failed",
};
const FREWIND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREWIND.INTERNAL",
    identifier: None,
    when: "Internal runtime control-flow or conversion failed.",
    message: "frewind: internal error",
};
const FREWIND_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FREWIND_ERROR_INVALID_INPUT,
    FREWIND_ERROR_INVALID_IDENTIFIER,
    FREWIND_ERROR_IO,
    FREWIND_ERROR_INTERNAL,
];
pub const FREWIND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FREWIND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FREWIND_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::frewind")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "frewind",
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
    notes: "Host-only operation: rewinds the file position indicator to the beginning of the file.",
};

fn frewind_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    frewind_error_with_message(error.message, error)
}

fn frewind_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    frewind_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn frewind_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn frewind_error_with_source(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn frewind_error_with_source_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    frewind_error_with_source(
        format!("{}: {}", error.message, detail.as_ref()),
        error,
        source,
    )
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let message = err.message().to_string();
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {message}"))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FREWIND_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::frewind")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "frewind",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O operations are not eligible for fusion; metadata registered for completeness.",
};

#[runtime_builtin(
    name = "frewind",
    category = "io/filetext",
    summary = "Rewind file position indicators.",
    keywords = "frewind,file,rewind,seek,io,file identifier",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::frewind_type),
    descriptor(crate::builtins::io::filetext::frewind::FREWIND_DESCRIPTOR),
    extensions(crate::builtins::io::filetext::frewind::FREWIND_EXTENSIONS),
    integer_capabilities(crate::builtins::io::filetext::frewind::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::io::filetext::frewind"
)]
async fn frewind_builtin(fid: Value) -> crate::BuiltinResult<Value> {
    evaluate(&fid).await?;
    Ok(Value::OutputList(Vec::new()))
}

/// Evaluate the `frewind` builtin without invoking the runtime dispatcher.
pub async fn evaluate(fid_value: &Value) -> BuiltinResult<()> {
    preflight_fid(fid_value)?;
    let fid_host = gather_if_needed_async(fid_value)
        .await
        .map_err(map_control_flow)?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err(frewind_error_with_detail(
            &FREWIND_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Ok(());
    }

    let handle = registry::shared_handle(fid)
        .ok_or_else(|| frewind_error(&FREWIND_ERROR_INVALID_IDENTIFIER))?;
    let mut guard = handle.lock().map_err(|_| {
        frewind_error_with_detail(
            &FREWIND_ERROR_INTERNAL,
            "failed to lock file handle (poisoned mutex)",
        )
    })?;
    let file = guard
        .as_mut()
        .ok_or_else(|| frewind_error(&FREWIND_ERROR_INVALID_IDENTIFIER))?;

    file.seek(SeekFrom::Start(0)).map_err(|err| {
        frewind_error_with_source_detail(
            &FREWIND_ERROR_IO,
            format!("failed to rewind file: {err}"),
            err,
        )
    })?;
    drop(guard);
    registry::clear_eof_encountered(fid);

    Ok(())
}

fn preflight_fid(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Int(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FREWIND_INTEGER_ID_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FREWIND_INTEGER_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Bool(_) | Value::LogicalArray(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FREWIND_LOGICAL_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FREWIND_SINGLE_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FREWIND_RESIDENT_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FREWIND_INTEGER_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FREWIND_SINGLE_ID_EXTENSION,
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
            frewind_error_with_detail(
                &FREWIND_ERROR_INVALID_INPUT,
                "file identifier is out of range",
            )
        }),
        Value::Tensor(t) => {
            if tensor::is_scalar_tensor(t) {
                if let Some(storage) = t.integer_storage() {
                    let value = storage.value_at(0).expect("one-element integer storage");
                    return value.try_to_i32().ok_or_else(|| {
                        frewind_error_with_detail(
                            &FREWIND_ERROR_INVALID_INPUT,
                            "file identifier is out of range",
                        )
                    });
                }
                parse_scalar_fid(tensor::tensor_value_f64(t, 0))
            } else {
                Err(frewind_error_with_detail(
                    &FREWIND_ERROR_INVALID_INPUT,
                    IDENTIFIER_TYPE_ERROR_DETAIL,
                ))
            }
        }
        Value::LogicalArray(la) if la.data.len() == 1 => {
            let v = if la.data[0] != 0 { 1.0 } else { 0.0 };
            parse_scalar_fid(v)
        }
        Value::LogicalArray(_) => Err(frewind_error_with_detail(
            &FREWIND_ERROR_INVALID_INPUT,
            IDENTIFIER_TYPE_ERROR_DETAIL,
        )),
        Value::Bool(b) => parse_scalar_fid(if *b { 1.0 } else { 0.0 }),
        _ => Err(frewind_error_with_detail(
            &FREWIND_ERROR_INVALID_INPUT,
            IDENTIFIER_TYPE_ERROR_DETAIL,
        )),
    }
}

fn parse_scalar_fid(value: f64) -> BuiltinResult<i32> {
    if !value.is_finite() {
        return Err(frewind_error_with_detail(
            &FREWIND_ERROR_INVALID_INPUT,
            "file identifier must be finite",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(frewind_error_with_detail(
            &FREWIND_ERROR_INVALID_INPUT,
            "file identifier must be an integer",
        ));
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err(frewind_error_with_detail(
            &FREWIND_ERROR_INVALID_INPUT,
            "file identifier is out of range",
        ));
    }
    Ok(rounded as i32)
}

#[cfg(test)]
pub(crate) mod tests;
