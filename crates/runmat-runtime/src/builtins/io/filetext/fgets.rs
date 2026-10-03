//! MATLAB-compatible `fgets` builtin for RunMat.

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
    helpers::{bytes_to_char_array, empty_numeric_row, numeric_row, read_text_line},
    registry,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "fgets";
const NCHAR_NONNEGATIVE_INTEGER_DETAIL: &str = "nchar must be a non-negative integer scalar";
const NCHAR_OUT_OF_RANGE_DETAIL: &str = "nchar is out of range";

macro_rules! fgets_extension {
    ($name:ident, $id:literal, $description:literal, $error:literal) => {
        const $name: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: $description,
            error_identifier: Some($error),
        };
    };
}
fgets_extension!(
    FGETS_INTEGER_ID_EXTENSION,
    "fgets-integer-fileid",
    "integer-class fgets file identifiers are a RunMat extension",
    "RunMat:compatibility:FgetsIntegerIdExtension"
);
fgets_extension!(
    FGETS_INTEGER_NCHAR_EXTENSION,
    "fgets-integer-nchar",
    "integer-class fgets nchar values are a RunMat extension",
    "RunMat:compatibility:FgetsIntegerNcharExtension"
);
fgets_extension!(
    FGETS_SINGLE_ID_EXTENSION,
    "fgets-single-fileid",
    "single-precision fgets file identifiers are a RunMat extension",
    "RunMat:compatibility:FgetsSingleIdExtension"
);
fgets_extension!(
    FGETS_SINGLE_NCHAR_EXTENSION,
    "fgets-single-nchar",
    "single-precision fgets nchar values are a RunMat extension",
    "RunMat:compatibility:FgetsSingleNcharExtension"
);
fgets_extension!(
    FGETS_RESIDENT_ID_EXTENSION,
    "fgets-resident-fileid",
    "provider-resident fgets file identifiers are a RunMat extension",
    "RunMat:compatibility:FgetsResidentIdExtension"
);
fgets_extension!(
    FGETS_RESIDENT_NCHAR_EXTENSION,
    "fgets-resident-nchar",
    "provider-resident fgets nchar values are a RunMat extension",
    "RunMat:compatibility:FgetsResidentNcharExtension"
);
pub const FGETS_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    FGETS_INTEGER_ID_EXTENSION,
    FGETS_INTEGER_NCHAR_EXTENSION,
    FGETS_SINGLE_ID_EXTENSION,
    FGETS_SINGLE_NCHAR_EXTENSION,
    FGETS_RESIDENT_ID_EXTENSION,
    FGETS_RESIDENT_NCHAR_EXTENSION,
];

const FGETS_INTEGER_ID_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "MATLAB-compatible file identifiers are integer-valued doubles; all eight typed integer classes are one independently gated RunMat extension and are checked exactly against the registry identifier domain.",
    }];
const FGETS_INTEGER_NCHAR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "nchar",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "MATLAB-compatible nchar values are non-negative integer-valued doubles; all eight typed integer classes are one independently gated RunMat extension and are converted exactly to the runtime length domain.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "tline = fgets(integer_fileID, ...)",
        inputs: &FGETS_INTEGER_ID_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The identifier is validated exactly before registry access; a successful read returns a host character row and end-of-file returns the double sentinel -1.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "tline = fgets(fileID, integer_nchar)",
        inputs: &FGETS_INTEGER_NCHAR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The character limit is validated exactly before reading; line and terminator outputs remain host values independent of the nchar class.",
    },
];

const FGETS_OUTPUT_LINE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tline",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Next line including terminators, or -1 at end-of-file.",
}];
const FGETS_OUTPUT_LINE_TERMINATORS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "tline",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Next line including terminators, or -1 at end-of-file.",
    },
    BuiltinParamDescriptor {
        name: "terminators",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Numeric row vector of terminator byte values.",
    },
];
const FGETS_INPUTS_FID: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier opened by fopen.",
}];
const FGETS_INPUTS_FID_NCHAR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "nchar",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Maximum number of characters to read.",
    },
];
const FGETS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "tline = fgets(fid)",
        inputs: &FGETS_INPUTS_FID,
        outputs: &FGETS_OUTPUT_LINE,
    },
    BuiltinSignatureDescriptor {
        label: "tline = fgets(fid, nchar)",
        inputs: &FGETS_INPUTS_FID_NCHAR,
        outputs: &FGETS_OUTPUT_LINE,
    },
    BuiltinSignatureDescriptor {
        label: "[tline, terminators] = fgets(fid, ...)",
        inputs: &FGETS_INPUTS_FID_NCHAR,
        outputs: &FGETS_OUTPUT_LINE_TERMINATORS,
    },
];

const FGETS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETS.INVALID_INPUT",
    identifier: Some("RunMat:fgets:InvalidInput"),
    when: "Input argument count or scalar constraints are invalid.",
    message: "fgets: invalid input arguments",
};
const FGETS_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETS.INVALID_IDENTIFIER",
    identifier: Some("RunMat:fgets:InvalidIdentifier"),
    when: "Identifier does not refer to an open readable file.",
    message: "fgets: invalid file identifier. Use fopen to generate a valid file ID.",
};
const FGETS_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETS.IO",
    identifier: Some("RunMat:fgets:IoFailure"),
    when: "File read or decode operation failed.",
    message: "fgets: file I/O failed",
};
const FGETS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETS.INTERNAL",
    identifier: None,
    when: "Internal control-flow conversion failed.",
    message: "fgets: internal error",
};
const FGETS_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FGETS_ERROR_INVALID_INPUT,
    FGETS_ERROR_INVALID_IDENTIFIER,
    FGETS_ERROR_IO,
    FGETS_ERROR_INTERNAL,
];
pub const FGETS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FGETS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FGETS_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fgets")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fgets",
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

fn fgets_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fgets_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    fgets_error_with_message(error.message, error)
}

fn fgets_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fgets_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FGETS_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fgets")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fgets",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O calls are not eligible for fusion.",
};

#[runtime_builtin(
    name = "fgets",
    category = "io/filetext",
    summary = "Read the next line including newline characters.",
    keywords = "fgets,file,io,line,newline",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fgets_type),
    descriptor(crate::builtins::io::filetext::fgets::FGETS_DESCRIPTOR),
    integer_capabilities(crate::builtins::io::filetext::fgets::INTEGER_CAPABILITIES),
    extensions(crate::builtins::io::filetext::fgets::FGETS_EXTENSIONS),
    builtin_path = "crate::builtins::io::filetext::fgets"
)]
async fn fgets_builtin(fid: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&fid, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            eval.outputs(),
        ));
    }
    Ok(eval.first_output())
}

#[derive(Clone, Debug)]
pub struct FgetsEval {
    line: Value,
    terminators: Value,
}

impl FgetsEval {
    fn new(line: Value, terminators: Value) -> Self {
        Self { line, terminators }
    }

    fn end_of_file() -> Self {
        Self {
            line: Value::Num(-1.0),
            terminators: Value::Num(-1.0),
        }
    }

    pub fn first_output(&self) -> Value {
        self.line.clone()
    }

    pub fn outputs(&self) -> Vec<Value> {
        vec![self.line.clone(), self.terminators.clone()]
    }
}

pub async fn evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FgetsEval> {
    if rest.len() > 1 {
        return Err(fgets_error_with_detail(
            &FGETS_ERROR_INVALID_INPUT,
            "too many input arguments",
        ));
    }

    preflight_fid(fid_value)?;
    if let Some(nchar) = rest.first() {
        preflight_nchar(nchar)?;
    }
    let fid_host = gather_value(fid_value).await?;
    preflight_fid(&fid_host)?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err(fgets_error_with_detail(
            &FGETS_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Err(fgets_error_with_detail(
            &FGETS_ERROR_INVALID_INPUT,
            "standard input/output identifiers are not supported yet",
        ));
    }

    let info =
        registry::info_for(fid).ok_or_else(|| fgets_error(&FGETS_ERROR_INVALID_IDENTIFIER))?;
    if !permission_allows_read(&info.permission) {
        return Err(fgets_error_with_detail(
            &FGETS_ERROR_INVALID_IDENTIFIER,
            "file identifier is not open for reading",
        ));
    }
    let handle =
        registry::shared_handle(fid).ok_or_else(|| fgets_error(&FGETS_ERROR_INVALID_IDENTIFIER))?;

    let nchar_limit = parse_nchar(rest).await?;
    let mut guard = handle.lock().map_err(|_| {
        fgets_error_with_detail(
            &FGETS_ERROR_INTERNAL,
            "failed to lock file handle (poisoned mutex)",
        )
    })?;
    let file = guard
        .as_mut()
        .ok_or_else(|| fgets_error(&FGETS_ERROR_INVALID_IDENTIFIER))?;
    let read = read_text_line(file, nchar_limit, BUILTIN_NAME)
        .map_err(|e| fgets_error_with_detail(&FGETS_ERROR_IO, e.message()))?;
    drop(guard);
    if read.encountered_eof {
        registry::mark_eof_encountered(fid);
    }
    if read.eof_before_any {
        return Ok(FgetsEval::end_of_file());
    }

    let encoding = if info.encoding.trim().is_empty() {
        "UTF-8".to_string()
    } else {
        info.encoding.clone()
    };

    let line_value = bytes_to_char_array(&read.data, &encoding, BUILTIN_NAME)
        .map_err(|e| fgets_error_with_detail(&FGETS_ERROR_IO, e.message()))?;
    let terminators_value = if read.terminators.is_empty() {
        empty_numeric_row()
    } else {
        numeric_row(&read.terminators, BUILTIN_NAME)
            .map_err(|e| fgets_error_with_detail(&FGETS_ERROR_IO, e.message()))?
    };

    Ok(FgetsEval::new(line_value, terminators_value))
}

fn preflight_fid(value: &Value) -> BuiltinResult<()> {
    preflight_numeric_role(
        value,
        &FGETS_INTEGER_ID_EXTENSION,
        &FGETS_SINGLE_ID_EXTENSION,
        &FGETS_RESIDENT_ID_EXTENSION,
    )
}

fn preflight_nchar(value: &Value) -> BuiltinResult<()> {
    preflight_numeric_role(
        value,
        &FGETS_INTEGER_NCHAR_EXTENSION,
        &FGETS_SINGLE_NCHAR_EXTENSION,
        &FGETS_RESIDENT_NCHAR_EXTENSION,
    )
}

fn preflight_numeric_role(
    value: &Value,
    integer_extension: &BuiltinExtensionDescriptor,
    single_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    match value {
        Value::Num(_) => Ok(()),
        Value::Int(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(integer_extension, BUILTIN_NAME)
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(integer_extension, BUILTIN_NAME)
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            crate::compatibility::ensure_builtin_extension_enabled(single_extension, BUILTIN_NAME)
        }
        Value::Tensor(_) => Ok(()),
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                resident_extension,
                BUILTIN_NAME,
            )?;
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    integer_extension,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    single_extension,
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
            return Err(fgets_error_with_detail(
                &FGETS_ERROR_INVALID_INPUT,
                "file identifier is out of range",
            ));
        }
        Ok(n as i32)
    }

    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    "file identifier must be finite",
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    "file identifier must be an integer scalar",
                ));
            }
            checked_f64_to_i32(*n)
        }
        Value::Int(i) => i.try_to_i32().ok_or_else(|| {
            fgets_error_with_detail(
                &FGETS_ERROR_INVALID_INPUT,
                "file identifier is out of range",
            )
        }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                let value = storage.value_at(0).expect("one-element integer storage");
                return value.try_to_i32().ok_or_else(|| {
                    fgets_error_with_detail(
                        &FGETS_ERROR_INVALID_INPUT,
                        "file identifier is out of range",
                    )
                });
            }
            let n = tensor::tensor_value_f64(t, 0);
            if !n.is_finite() {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    "file identifier must be finite",
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    "file identifier must be an integer scalar",
                ));
            }
            checked_f64_to_i32(n)
        }
        _ => Err(fgets_error_with_detail(
            &FGETS_ERROR_INVALID_INPUT,
            "file identifier must be a numeric scalar",
        )),
    }
}

async fn parse_nchar(args: &[Value]) -> BuiltinResult<Option<usize>> {
    if args.is_empty() {
        return Ok(None);
    }
    let value = gather_value(&args[0]).await?;
    preflight_nchar(&value)?;

    fn checked_f64_to_usize(n: f64) -> BuiltinResult<usize> {
        let upper_exclusive = usize::MAX as f64 + 1.0;
        if n >= upper_exclusive {
            return Err(fgets_error_with_detail(
                &FGETS_ERROR_INVALID_INPUT,
                NCHAR_OUT_OF_RANGE_DETAIL,
            ));
        }
        Ok(n as usize)
    }

    fn checked_int_to_usize(value: &runmat_value::IntValue) -> BuiltinResult<usize> {
        if let Some(value) = value.try_to_usize() {
            return Ok(value);
        }
        let detail = if value.try_to_u64().is_some() {
            NCHAR_OUT_OF_RANGE_DETAIL
        } else {
            NCHAR_NONNEGATIVE_INTEGER_DETAIL
        };
        Err(fgets_error_with_detail(&FGETS_ERROR_INVALID_INPUT, detail))
    }

    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    NCHAR_NONNEGATIVE_INTEGER_DETAIL,
                ));
            }
            if n < 0.0 {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    NCHAR_NONNEGATIVE_INTEGER_DETAIL,
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    NCHAR_NONNEGATIVE_INTEGER_DETAIL,
                ));
            }
            checked_f64_to_usize(n).map(Some)
        }
        Value::Int(i) => checked_int_to_usize(&i).map(Some),
        Value::Tensor(t) if tensor::is_scalar_tensor(&t) => {
            if let Some(storage) = t.integer_storage() {
                let value = storage.value_at(0).expect("one-element integer storage");
                return checked_int_to_usize(&value).map(Some);
            }
            let n = tensor::tensor_value_f64(&t, 0);
            if !n.is_finite() {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    NCHAR_NONNEGATIVE_INTEGER_DETAIL,
                ));
            }
            if n < 0.0 {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    NCHAR_NONNEGATIVE_INTEGER_DETAIL,
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgets_error_with_detail(
                    &FGETS_ERROR_INVALID_INPUT,
                    NCHAR_NONNEGATIVE_INTEGER_DETAIL,
                ));
            }
            checked_f64_to_usize(n).map(Some)
        }
        _ => Err(fgets_error_with_detail(
            &FGETS_ERROR_INVALID_INPUT,
            NCHAR_NONNEGATIVE_INTEGER_DETAIL,
        )),
    }
}

fn permission_allows_read(permission: &str) -> bool {
    let lower = permission.to_ascii_lowercase();
    lower.starts_with('r') || lower.contains('+')
}

#[cfg(test)]
pub(crate) mod tests;
