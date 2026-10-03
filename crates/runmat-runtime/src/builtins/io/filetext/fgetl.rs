//! MATLAB-compatible `fgetl` builtin for RunMat.

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
use crate::builtins::io::filetext::{
    helpers::{bytes_to_char_array, read_text_line},
    registry,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "fgetl";

const FGETL_INTEGER_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fgetl-integer-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "integer-class fgetl file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FgetlIntegerIdExtension"),
};
const FGETL_SINGLE_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fgetl-single-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "single-precision fgetl file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FgetlSingleIdExtension"),
};
const FGETL_RESIDENT_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fgetl-resident-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "provider-resident fgetl file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FgetlResidentIdExtension"),
};
pub const FGETL_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FGETL_INTEGER_ID_EXTENSION,
    FGETL_SINGLE_ID_EXTENSION,
    FGETL_RESIDENT_ID_EXTENSION,
];

const FGETL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "MATLAB-compatible file identifiers are integer-valued doubles; all eight typed integer classes are one gated RunMat extension and are checked exactly against the registry identifier domain.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tline = fgetl(integer_fileID)",
        inputs: &FGETL_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The identifier is validated exactly before registry access; a successful read returns a host character row and end-of-file returns the double sentinel -1.",
    }];

const FGETL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tline",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Next line without terminator, or -1 at end-of-file.",
}];
const FGETL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier opened by fopen.",
}];
const FGETL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tline = fgetl(fid)",
    inputs: &FGETL_INPUTS,
    outputs: &FGETL_OUTPUT,
}];

const FGETL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETL.INVALID_INPUT",
    identifier: Some("RunMat:fgetl:InvalidInput"),
    when: "Input argument count or identifier form is invalid.",
    message: "fgetl: invalid input arguments",
};
const FGETL_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETL.INVALID_IDENTIFIER",
    identifier: Some("RunMat:fgetl:InvalidIdentifier"),
    when: "Identifier does not refer to a readable open file.",
    message: "fgetl: invalid file identifier. Use fopen to generate a valid file ID.",
};
const FGETL_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETL.IO",
    identifier: Some("RunMat:fgetl:IoFailure"),
    when: "Line read or decode fails.",
    message: "fgetl: file I/O failed",
};
const FGETL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETL.INTERNAL",
    identifier: None,
    when: "Internal control-flow conversion failed.",
    message: "fgetl: internal error",
};
const FGETL_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FGETL_ERROR_INVALID_INPUT,
    FGETL_ERROR_INVALID_IDENTIFIER,
    FGETL_ERROR_IO,
    FGETL_ERROR_INTERNAL,
];
pub const FGETL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FGETL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FGETL_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fgetl")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fgetl",
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
    notes: "Host-only file I/O; arguments gathered from the GPU when necessary.",
};

fn fgetl_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fgetl_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    fgetl_error_with_message(error.message, error)
}

fn fgetl_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fgetl_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FGETL_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fgetl")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fgetl",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O calls are not eligible for fusion.",
};

#[runtime_builtin(
    name = "fgetl",
    category = "io/filetext",
    summary = "Read the next line without newline characters.",
    keywords = "fgetl,file,io,line,newline",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fgetl_type),
    descriptor(crate::builtins::io::filetext::fgetl::FGETL_DESCRIPTOR),
    integer_capabilities(crate::builtins::io::filetext::fgetl::INTEGER_CAPABILITIES),
    extensions(crate::builtins::io::filetext::fgetl::FGETL_EXTENSIONS),
    builtin_path = "crate::builtins::io::filetext::fgetl"
)]
async fn fgetl_builtin(fid: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&fid, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![eval.first_output()],
        ));
    }
    Ok(eval.first_output())
}

#[derive(Clone, Debug)]
pub struct FgetlEval {
    line: Value,
}

impl FgetlEval {
    fn new(line: Value) -> Self {
        Self { line }
    }

    fn end_of_file() -> Self {
        Self {
            line: Value::Num(-1.0),
        }
    }

    pub fn first_output(&self) -> Value {
        self.line.clone()
    }
}

pub async fn evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FgetlEval> {
    if !rest.is_empty() {
        return Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_INPUT,
            "too many input arguments",
        ));
    }

    preflight_fid(fid_value)?;
    let fid_host = gather_value(fid_value).await?;
    preflight_fid(&fid_host)?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_INPUT,
            "standard input/output identifiers are not supported yet",
        ));
    }

    let info =
        registry::info_for(fid).ok_or_else(|| fgetl_error(&FGETL_ERROR_INVALID_IDENTIFIER))?;
    if !permission_allows_read(&info.permission) {
        return Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_IDENTIFIER,
            "file identifier is not open for reading",
        ));
    }
    let handle =
        registry::shared_handle(fid).ok_or_else(|| fgetl_error(&FGETL_ERROR_INVALID_IDENTIFIER))?;

    let mut guard = handle.lock().map_err(|_| {
        fgetl_error_with_detail(
            &FGETL_ERROR_INTERNAL,
            "failed to lock file handle (poisoned mutex)",
        )
    })?;
    let file = guard
        .as_mut()
        .ok_or_else(|| fgetl_error(&FGETL_ERROR_INVALID_IDENTIFIER))?;
    let read = read_text_line(file, None, BUILTIN_NAME)
        .map_err(|e| fgetl_error_with_detail(&FGETL_ERROR_IO, e.message()))?;
    drop(guard);
    if read.encountered_eof {
        registry::mark_eof_encountered(fid);
    }
    if read.eof_before_any {
        return Ok(FgetlEval::end_of_file());
    }

    let encoding = if info.encoding.trim().is_empty() {
        "UTF-8".to_string()
    } else {
        info.encoding.clone()
    };

    let line_len = read.data.len().saturating_sub(read.terminators.len());
    let line_value = bytes_to_char_array(&read.data[..line_len], &encoding, BUILTIN_NAME)
        .map_err(|e| fgetl_error_with_detail(&FGETL_ERROR_IO, e.message()))?;
    Ok(FgetlEval::new(line_value))
}

fn preflight_fid(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Num(_) => Ok(()),
        Value::Int(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FGETL_INTEGER_ID_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FGETL_INTEGER_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FGETL_SINGLE_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(_) => Ok(()),
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FGETL_RESIDENT_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FGETL_INTEGER_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FGETL_SINGLE_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value)
        .await
        .map_err(map_control_flow)
}

fn parse_fid(value: &Value) -> BuiltinResult<i32> {
    fn checked_f64_to_i32(n: f64) -> BuiltinResult<i32> {
        if n < i32::MIN as f64 || n > i32::MAX as f64 {
            return Err(fgetl_error_with_detail(
                &FGETL_ERROR_INVALID_INPUT,
                "file identifier is out of range",
            ));
        }
        Ok(n as i32)
    }

    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(fgetl_error_with_detail(
                    &FGETL_ERROR_INVALID_INPUT,
                    "file identifier must be finite",
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgetl_error_with_detail(
                    &FGETL_ERROR_INVALID_INPUT,
                    "file identifier must be an integer scalar",
                ));
            }
            checked_f64_to_i32(*n)
        }
        Value::Int(i) => i.try_to_i32().ok_or_else(|| {
            fgetl_error_with_detail(
                &FGETL_ERROR_INVALID_INPUT,
                "file identifier is out of range",
            )
        }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                let value = storage.value_at(0).expect("one-element integer storage");
                return value.try_to_i32().ok_or_else(|| {
                    fgetl_error_with_detail(
                        &FGETL_ERROR_INVALID_INPUT,
                        "file identifier is out of range",
                    )
                });
            }
            let n = tensor::tensor_value_f64(t, 0);
            if !n.is_finite() {
                return Err(fgetl_error_with_detail(
                    &FGETL_ERROR_INVALID_INPUT,
                    "file identifier must be finite",
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgetl_error_with_detail(
                    &FGETL_ERROR_INVALID_INPUT,
                    "file identifier must be an integer scalar",
                ));
            }
            checked_f64_to_i32(n)
        }
        _ => Err(fgetl_error_with_detail(
            &FGETL_ERROR_INVALID_INPUT,
            "file identifier must be a numeric scalar",
        )),
    }
}

fn permission_allows_read(permission: &str) -> bool {
    let lower = permission.to_ascii_lowercase();
    lower.starts_with('r') || lower.contains('+')
}

#[cfg(test)]
pub(crate) mod tests;
