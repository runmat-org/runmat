//! MATLAB-compatible `fclose` builtin for RunMat.
//!
//! MATLAB-compatible mode closes one scalar double file identifier or all open
//! files through the character-vector `all` form. Broader RunMat forms are
//! extension-gated. The implementation integrates with the shared file registry
//! managed by `fopen` and always executes on the host.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericDType, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::io::filetext::{
    helpers::{char_array_value, extract_scalar_string},
    registry,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "fclose";

macro_rules! fclose_extension {
    ($name:ident, $id:literal, $description:literal, $error:literal) => {
        const $name: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: $description,
            error_identifier: Some($error),
        };
    };
}
fclose_extension!(
    FCLOSE_NO_INPUT_EXTENSION,
    "fclose-no-input-close-all",
    "fclose() close-all syntax is a RunMat extension",
    "RunMat:compatibility:FcloseNoInputExtension"
);
fclose_extension!(
    FCLOSE_INTEGER_ID_EXTENSION,
    "fclose-integer-fileid",
    "integer-class fclose file identifiers are a RunMat extension",
    "RunMat:compatibility:FcloseIntegerIdExtension"
);
fclose_extension!(
    FCLOSE_VECTOR_ID_EXTENSION,
    "fclose-fileid-vector",
    "fclose file-identifier arrays are a RunMat extension",
    "RunMat:compatibility:FcloseVectorIdExtension"
);
fclose_extension!(
    FCLOSE_LOGICAL_ID_EXTENSION,
    "fclose-logical-fileid",
    "logical fclose file identifiers are a RunMat extension",
    "RunMat:compatibility:FcloseLogicalIdExtension"
);
fclose_extension!(
    FCLOSE_CELL_ID_EXTENSION,
    "fclose-cell-fileids",
    "cell fclose file-identifier collections are a RunMat extension",
    "RunMat:compatibility:FcloseCellIdExtension"
);
fclose_extension!(
    FCLOSE_STRING_ALL_EXTENSION,
    "fclose-string-all",
    "string-scalar fclose all syntax is a RunMat extension",
    "RunMat:compatibility:FcloseStringAllExtension"
);
fclose_extension!(
    FCLOSE_MESSAGE_OUTPUT_EXTENSION,
    "fclose-message-output",
    "a second fclose diagnostic output is a RunMat extension",
    "RunMat:compatibility:FcloseMessageOutputExtension"
);
fclose_extension!(
    FCLOSE_RESIDENT_ID_EXTENSION,
    "fclose-resident-fileid",
    "resident fclose file identifiers are a RunMat extension",
    "RunMat:compatibility:FcloseResidentIdExtension"
);
fclose_extension!(
    FCLOSE_SINGLE_ID_EXTENSION,
    "fclose-single-fileid",
    "single-precision fclose file identifiers are a RunMat extension",
    "RunMat:compatibility:FcloseSingleIdExtension"
);
pub const FCLOSE_EXTENSIONS: [BuiltinExtensionDescriptor; 9] = [
    FCLOSE_NO_INPUT_EXTENSION,
    FCLOSE_INTEGER_ID_EXTENSION,
    FCLOSE_VECTOR_ID_EXTENSION,
    FCLOSE_LOGICAL_ID_EXTENSION,
    FCLOSE_CELL_ID_EXTENSION,
    FCLOSE_STRING_ALL_EXTENSION,
    FCLOSE_MESSAGE_OUTPUT_EXTENSION,
    FCLOSE_RESIDENT_ID_EXTENSION,
    FCLOSE_SINGLE_ID_EXTENSION,
];

const FCLOSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are parsed exactly and must fit the runtime file-identifier domain before registry access.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "status = fclose(integer_fileID)",
        inputs: &FCLOSE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "This RunMat-only scalar form returns double status and never converts the identifier through binary64.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "status = fclose(integer_fileIDs)",
        inputs: &FCLOSE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The collection and resident forms are independently gated RunMat extensions; every identifier validates before any close side effect.",
    },
];

const FCLOSE_OUTPUT_STATUS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "0 on success, -1 on failure.",
}];

const FCLOSE_OUTPUT_MESSAGE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "status",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "0 on success, -1 on failure.",
    },
    BuiltinParamDescriptor {
        name: "msg",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"\""),
        description: "Failure message when status is -1.",
    },
];

const FCLOSE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const FCLOSE_INPUTS_FID: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier scalar, vector, cell list, or keyword \"all\".",
}];
const FCLOSE_INPUTS_ALL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"all\""),
    description: "Close all open user file identifiers.",
}];

const FCLOSE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "status = fclose()",
        inputs: &FCLOSE_INPUTS_NONE,
        outputs: &FCLOSE_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = fclose(fid)",
        inputs: &FCLOSE_INPUTS_FID,
        outputs: &FCLOSE_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = fclose(\"all\")",
        inputs: &FCLOSE_INPUTS_ALL,
        outputs: &FCLOSE_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "[status, msg] = fclose(...)",
        inputs: &FCLOSE_INPUTS_FID,
        outputs: &FCLOSE_OUTPUT_MESSAGE,
    },
];

const FCLOSE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCLOSE.INVALID_INPUT",
    identifier: Some("RunMat:fclose:InvalidInput"),
    when: "Input argument count or file identifier value/type is invalid.",
    message: "fclose: invalid input arguments",
};

const FCLOSE_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCLOSE.INVALID_IDENTIFIER",
    identifier: Some("RunMat:fclose:InvalidIdentifier"),
    when: "The provided file identifier does not refer to an open file.",
    message: "fclose: invalid file identifier. Use fopen to generate a valid file ID.",
};

const FCLOSE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCLOSE.IO",
    identifier: Some("RunMat:fclose:IoFailure"),
    when: "Closing an open file failed due to I/O error.",
    message: "fclose: failed to close file",
};

const FCLOSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCLOSE.INTERNAL",
    identifier: None,
    when: "Internal runtime control-flow or conversion failed.",
    message: "fclose: internal error",
};

const FCLOSE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FCLOSE_ERROR_INVALID_INPUT,
    FCLOSE_ERROR_INVALID_IDENTIFIER,
    FCLOSE_ERROR_IO,
    FCLOSE_ERROR_INTERNAL,
];

pub const FCLOSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FCLOSE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FCLOSE_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fclose")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fclose",
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
    notes:
        "Host-only operation: closes identifiers stored in the shared file registry; GPU inputs are gathered automatically.",
};

fn fclose_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fclose_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn fclose_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FCLOSE_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fclose")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fclose",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O is not eligible for fusion; metadata is registered for completeness.",
};

#[runtime_builtin(
    name = "fclose",
    category = "io/filetext",
    summary = "Close file identifiers.",
    keywords = "fclose,file,close,io,identifier",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fclose_type),
    descriptor(crate::builtins::io::filetext::fclose::FCLOSE_DESCRIPTOR),
    extensions(FCLOSE_EXTENSIONS),
    integer_capabilities(crate::builtins::io::filetext::fclose::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::io::filetext::fclose"
)]
async fn fclose_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    preflight_fclose_outputs()?;
    let eval = evaluate(&args).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(Value::OutputList(
            eval.outputs().into_iter().take(out_count).collect(),
        ));
    }
    Ok(eval.first_output())
}

#[derive(Debug, Clone)]
pub struct FcloseEval {
    status: f64,
    message: String,
}

impl FcloseEval {
    fn success() -> Self {
        Self {
            status: 0.0,
            message: String::new(),
        }
    }

    fn failure(message: String) -> Self {
        Self {
            status: -1.0,
            message,
        }
    }

    pub fn first_output(&self) -> Value {
        Value::Num(self.status)
    }

    pub fn outputs(&self) -> Vec<Value> {
        vec![Value::Num(self.status), char_array_value(&self.message)]
    }

    #[cfg(test)]
    pub(crate) fn status(&self) -> f64 {
        self.status
    }

    #[cfg(test)]
    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

pub async fn evaluate(args: &[Value]) -> BuiltinResult<FcloseEval> {
    preflight_fclose_inputs(args)?;
    let gathered = gather_args(args).await?;
    match gathered.len() {
        0 => Ok(close_all().await),
        1 => handle_single_argument(&gathered[0]).await,
        _ => Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "too many input arguments",
        )),
    }
}

fn preflight_fclose_outputs() -> BuiltinResult<()> {
    let Some(count) = crate::output_count::current_output_count() else {
        return Ok(());
    };
    if count > 2 {
        return Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "too many output arguments",
        ));
    }
    if count == 2 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FCLOSE_MESSAGE_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn preflight_fclose_inputs(args: &[Value]) -> BuiltinResult<()> {
    match args {
        [] => crate::compatibility::ensure_builtin_extension_enabled(
            &FCLOSE_NO_INPUT_EXTENSION,
            BUILTIN_NAME,
        ),
        [value] => preflight_fclose_value(value),
        _ => Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "too many input arguments",
        )),
    }
}

fn preflight_fclose_value(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Num(_) | Value::CharArray(_) => Ok(()),
        Value::Int(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FCLOSE_INTEGER_ID_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Bool(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FCLOSE_LOGICAL_ID_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Tensor(tensor) => {
            if tensor.len() != 1 {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_VECTOR_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            if tensor.integer_storage().is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_INTEGER_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if tensor.numeric_dtype() == NumericDType::F32 {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_SINGLE_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        Value::LogicalArray(array) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FCLOSE_LOGICAL_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            if array.len() != 1 {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_VECTOR_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        Value::Cell(cell) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FCLOSE_CELL_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            for nested in &cell.data {
                preflight_fclose_value(nested)?;
            }
            Ok(())
        }
        Value::String(_) | Value::StringArray(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FCLOSE_STRING_ALL_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FCLOSE_RESIDENT_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            if tensor::element_count(handle.shape()) != 1 {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_VECTOR_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_INTEGER_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_is_logical(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_LOGICAL_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_SINGLE_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

async fn handle_single_argument(value: &Value) -> BuiltinResult<FcloseEval> {
    if matches_keyword(value, "all") {
        return Ok(close_all().await);
    }
    let fids = collect_file_ids(value)?;
    Ok(close_fids(&fids).await)
}

async fn close_all() -> FcloseEval {
    let infos = registry::list_infos();
    let mut status_ok = true;
    let mut message = String::new();
    for info in infos {
        if info.id >= 3 {
            if let Err(err) = registry::close_async(info.id).await {
                status_ok = false;
                if message.is_empty() {
                    message = err.to_string();
                }
            }
        }
    }
    if status_ok {
        FcloseEval::success()
    } else {
        FcloseEval::failure(message)
    }
}

async fn close_fids(fids: &[i32]) -> FcloseEval {
    if fids.is_empty() {
        return FcloseEval::success();
    }
    let mut status_ok = true;
    let mut message = String::new();
    for &fid in fids {
        if fid < 0 {
            status_ok = false;
            if message.is_empty() {
                message = FCLOSE_ERROR_INVALID_IDENTIFIER.message.to_string();
            }
            continue;
        }
        if fid < 3 {
            continue;
        }
        match registry::close_async(fid).await {
            Ok(Some(_)) => {}
            Ok(None) => {
                status_ok = false;
                if message.is_empty() {
                    message = FCLOSE_ERROR_INVALID_IDENTIFIER.message.to_string();
                }
            }
            Err(err) => {
                status_ok = false;
                if message.is_empty() {
                    message = err.to_string();
                }
            }
        }
    }
    if status_ok {
        FcloseEval::success()
    } else {
        FcloseEval::failure(message)
    }
}

fn collect_file_ids(value: &Value) -> BuiltinResult<Vec<i32>> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(vec![parse_scalar_fid(value)?]),
        Value::Tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                return storage
                    .exact_values()
                    .iter()
                    .map(parse_integer_fid)
                    .collect();
            }
            let mut ids = Vec::with_capacity(t.len());
            for n in tensor::tensor_values_f64(t) {
                ids.push(parse_fid_from_f64(n)?);
            }
            Ok(ids)
        }
        Value::LogicalArray(la) => {
            let mut ids = Vec::with_capacity(la.data.len());
            for &b in &la.data {
                let v = if b != 0 { 1 } else { 0 };
                ids.push(v);
            }
            Ok(ids)
        }
        Value::Cell(ca) => {
            let mut ids = Vec::with_capacity(ca.data.len());
            for ptr in &ca.data {
                let nested = collect_file_ids(ptr)?;
                ids.extend(nested);
            }
            Ok(ids)
        }
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            Err(fclose_error_with_detail(
                &FCLOSE_ERROR_INVALID_INPUT,
                "file identifier must be numeric or 'all'",
            ))
        }
        _ => Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "file identifier must be numeric or 'all'",
        )),
    }
}

fn parse_scalar_fid(value: &Value) -> BuiltinResult<i32> {
    match value {
        Value::Int(i) => parse_integer_fid(i),
        Value::Num(n) => parse_fid_from_f64(*n),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        _ => Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "file identifier must be numeric or 'all'",
        )),
    }
}

fn parse_integer_fid(value: &IntValue) -> BuiltinResult<i32> {
    value.try_to_i32().ok_or_else(|| {
        fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "file identifier is out of range",
        )
    })
}

fn parse_fid_from_f64(value: f64) -> BuiltinResult<i32> {
    if !value.is_finite() {
        return Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "file identifier must be finite",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "file identifier must be an integer",
        ));
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "file identifier is out of range",
        ));
    }
    Ok(rounded as i32)
}

async fn gather_args(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(gathered)
}

fn matches_keyword(value: &Value, keyword: &str) -> bool {
    extract_scalar_string(value)
        .map(|text| text.eq_ignore_ascii_case(keyword))
        .unwrap_or(false)
}

#[cfg(test)]
pub(crate) mod tests;
