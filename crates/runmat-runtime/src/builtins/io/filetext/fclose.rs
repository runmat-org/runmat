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
    IntValue, NumericDType, Value,
};
use runmat_macros::runtime_builtin;

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
            if tensor::element_count(&handle.shape) != 1 {
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
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::io::filetext::{fopen, registry};
    use runmat_builtins::{CellArray, IntValue, IntegerStorage, LogicalArray, StringArray, Tensor};
    use runmat_time::system_time_now;
    use std::future::Future;
    use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
    use std::path::Path;
    use std::path::PathBuf;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, MutexGuard};
    use std::task::{Context, Poll};
    use std::time::UNIX_EPOCH;

    struct FailingFlushProvider {
        fail_flush: Arc<AtomicBool>,
    }

    fn unsupported_provider_op() -> io::Error {
        io::Error::new(io::ErrorKind::Unsupported, "unsupported test provider op")
    }

    #[async_trait::async_trait(?Send)]
    impl runmat_filesystem::FsProvider for FailingFlushProvider {
        fn open(
            &self,
            _path: &Path,
            _flags: &runmat_filesystem::OpenFlags,
        ) -> io::Result<Box<dyn runmat_filesystem::FileHandle>> {
            Ok(Box::new(FailingFlushHandle {
                inner: Cursor::new(Vec::new()),
                fail_flush: self.fail_flush.clone(),
            }))
        }

        async fn read(&self, _path: &Path) -> io::Result<Vec<u8>> {
            Err(unsupported_provider_op())
        }

        async fn write(&self, _path: &Path, _data: &[u8]) -> io::Result<()> {
            Err(unsupported_provider_op())
        }

        async fn remove_file(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported_provider_op())
        }

        async fn metadata(&self, _path: &Path) -> io::Result<runmat_filesystem::FsMetadata> {
            Err(unsupported_provider_op())
        }

        async fn symlink_metadata(
            &self,
            _path: &Path,
        ) -> io::Result<runmat_filesystem::FsMetadata> {
            Err(unsupported_provider_op())
        }

        async fn read_dir(&self, _path: &Path) -> io::Result<Vec<runmat_filesystem::DirEntry>> {
            Err(unsupported_provider_op())
        }

        async fn canonicalize(&self, _path: &Path) -> io::Result<PathBuf> {
            Err(unsupported_provider_op())
        }

        async fn create_dir(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported_provider_op())
        }

        async fn create_dir_all(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported_provider_op())
        }

        async fn remove_dir(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported_provider_op())
        }

        async fn remove_dir_all(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported_provider_op())
        }

        async fn rename(&self, _from: &Path, _to: &Path) -> io::Result<()> {
            Err(unsupported_provider_op())
        }

        async fn set_readonly(&self, _path: &Path, _readonly: bool) -> io::Result<()> {
            Err(unsupported_provider_op())
        }
    }

    struct FailingFlushHandle {
        inner: Cursor<Vec<u8>>,
        fail_flush: Arc<AtomicBool>,
    }

    struct YieldOnce {
        yielded: bool,
    }

    impl Future for YieldOnce {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            if self.yielded {
                Poll::Ready(())
            } else {
                self.yielded = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    impl Read for FailingFlushHandle {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.inner.read(buf)
        }
    }

    impl Write for FailingFlushHandle {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.inner.write(buf)
        }

        fn flush(&mut self) -> io::Result<()> {
            if self.fail_flush.load(Ordering::SeqCst) {
                Err(io::Error::other("flush failed"))
            } else {
                Ok(())
            }
        }
    }

    impl Seek for FailingFlushHandle {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            self.inner.seek(pos)
        }
    }

    #[async_trait::async_trait(?Send)]
    impl runmat_filesystem::FileHandle for FailingFlushHandle {
        async fn flush_async(&mut self) -> io::Result<()> {
            YieldOnce { yielded: false }.await;
            self.flush()
        }
    }

    fn unwrap_error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_evaluate(args: &[Value]) -> BuiltinResult<FcloseEval> {
        futures::executor::block_on(evaluate(args))
    }

    fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
        futures::executor::block_on(fopen::evaluate(args))
    }

    fn registry_guard() -> MutexGuard<'static, ()> {
        registry::test_guard()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FCLOSE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"status = fclose()"));
        assert!(labels.contains(&"status = fclose(fid)"));
        assert!(labels.contains(&"status = fclose(\"all\")"));
        assert!(labels.contains(&"[status, msg] = fclose(...)"));
    }

    #[test]
    fn typed_file_identifier_parser_rejects_unrepresentable_uint64() {
        assert_eq!(parse_scalar_fid(&Value::Int(IntValue::U16(7))).unwrap(), 7);
        assert!(parse_scalar_fid(&Value::Int(IntValue::U64(u64::MAX))).is_err());
    }

    #[test]
    fn fclose_vector_parser_reads_typed_integer_storage_exactly() {
        let ids =
            Tensor::new_integer(IntegerStorage::U16(vec![7, 9]), vec![1, 2]).expect("typed fids");

        assert_eq!(collect_file_ids(&Value::Tensor(ids)).unwrap(), vec![7, 9]);

        let too_large = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("typed fid");
        assert!(collect_file_ids(&Value::Tensor(too_large)).is_err());
    }

    #[test]
    fn fclose_typed_identifier_tensors_ignore_poisoned_f64_mirrors() {
        let classes = [
            IntegerStorage::I8(vec![7]),
            IntegerStorage::I16(vec![7]),
            IntegerStorage::I32(vec![7]),
            IntegerStorage::I64(vec![7]),
            IntegerStorage::U8(vec![7]),
            IntegerStorage::U16(vec![7]),
            IntegerStorage::U32(vec![7]),
            IntegerStorage::U64(vec![7]),
        ];

        for storage in classes {
            let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("typed fid");
            assert_eq!(collect_file_ids(&Value::Tensor(tensor)).unwrap(), vec![7]);
        }
    }

    fn unique_path(prefix: &str) -> PathBuf {
        let now = system_time_now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards");
        let filename = format!("{}_{}_{}.tmp", prefix, now.as_secs(), now.subsec_nanos());
        std::env::temp_dir().join(filename)
    }

    fn open_temp_file(prefix: &str) -> (f64, PathBuf) {
        let path = unique_path(prefix);
        {
            let mut file = runmat_filesystem::File::create(&path).unwrap();
            writeln!(&mut file, "data").unwrap();
        }
        let eval = run_fopen(&[Value::from(path.to_string_lossy().to_string())]).expect("fopen");
        let fid = eval.as_open().unwrap().fid;
        assert!(fid >= 3.0, "expected valid file identifier");
        (fid, path)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_closes_single_file() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid, path) = open_temp_file("fclose_single");
        let eval = run_evaluate(&[Value::Num(fid)]).expect("fclose");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
        assert!(registry::info_for(fid as i32).is_none());
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_invalid_identifier_returns_error() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let eval = run_evaluate(&[Value::Num(9999.0)]).expect("fclose");
        assert_eq!(eval.status(), -1.0);
        assert_eq!(eval.message(), FCLOSE_ERROR_INVALID_IDENTIFIER.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_all_closes_everything() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid, path) = open_temp_file("fclose_all");
        let eval = run_evaluate(&[Value::from("all")]).expect("fclose all");
        assert_eq!(eval.status(), 0.0);
        assert!(registry::info_for(fid as i32).is_none());
        let infos = registry::list_infos();
        assert!(infos.iter().all(|info| info.id < 3));
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_no_args_closes_all() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid, path) = open_temp_file("fclose_no_args");
        let eval = run_evaluate(&[]).expect("fclose");
        assert_eq!(eval.status(), 0.0);
        assert!(registry::info_for(fid as i32).is_none());
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_vector_of_fids_closes_each() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path1 = unique_path("fclose_vec1");
        test_support::fs::write(&path1, "a").unwrap();
        let fid1 = run_fopen(&[Value::from(path1.to_string_lossy().to_string())])
            .expect("open 1")
            .as_open()
            .unwrap()
            .fid;
        let path2 = unique_path("fclose_vec2");
        test_support::fs::write(&path2, "b").unwrap();
        let fid2 = run_fopen(&[Value::from(path2.to_string_lossy().to_string())])
            .expect("open 2")
            .as_open()
            .unwrap()
            .fid;
        let tensor = Tensor::new(vec![fid1, fid2], vec![2, 1]).expect("tensor construction");
        let eval = run_evaluate(&[Value::Tensor(tensor)]).expect("fclose");
        assert_eq!(eval.status(), 0.0);
        assert!(registry::info_for(fid1 as i32).is_none());
        assert!(registry::info_for(fid2 as i32).is_none());
        test_support::fs::remove_file(path1).unwrap();
        test_support::fs::remove_file(path2).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_repeat_returns_error_message() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid, path) = open_temp_file("fclose_repeat");
        let first = run_evaluate(&[Value::Num(fid)]).expect("fclose");
        assert_eq!(first.status(), 0.0);
        let second = run_evaluate(&[Value::Num(fid)]).expect("fclose second");
        assert_eq!(second.status(), -1.0);
        assert_eq!(second.message(), FCLOSE_ERROR_INVALID_IDENTIFIER.message);
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_keeps_file_registered_when_flush_fails() {
        let _guard = registry_guard();
        let _provider_lock = runmat_filesystem::provider_override_lock();
        registry::reset_for_tests();
        let fail_flush = Arc::new(AtomicBool::new(true));
        let provider = Arc::new(FailingFlushProvider {
            fail_flush: fail_flush.clone(),
        });
        let _provider_guard = runmat_filesystem::replace_provider(provider);

        let eval = run_fopen(&[Value::from("flush_fail.txt"), Value::from("w")]).expect("fopen");
        let fid = eval.as_open().expect("open result").fid;
        assert!(fid >= 3.0);

        let first = run_evaluate(&[Value::Num(fid)]).expect("first fclose");
        assert_eq!(first.status(), -1.0);
        assert!(first.message().contains("flush failed"));
        assert!(registry::info_for(fid as i32).is_some());
        let handle = registry::shared_handle(fid as i32).expect("handle remains registered");
        assert!(handle.lock().expect("handle lock").is_some());

        fail_flush.store(false, Ordering::SeqCst);
        let second = run_evaluate(&[Value::Num(fid)]).expect("second fclose");
        assert_eq!(second.status(), 0.0);
        assert!(registry::info_for(fid as i32).is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn concurrent_fclose_only_one_call_closes_file() {
        let _guard = registry_guard();
        let _provider_lock = runmat_filesystem::provider_override_lock();
        registry::reset_for_tests();
        let _provider_guard = runmat_filesystem::replace_provider(Arc::new(FailingFlushProvider {
            fail_flush: Arc::new(AtomicBool::new(false)),
        }));

        let eval =
            run_fopen(&[Value::from("yielding_close.txt"), Value::from("w")]).expect("fopen");
        let fid = eval.as_open().expect("open result").fid as i32;
        assert!(fid >= 3);

        let (first, second) = futures::executor::block_on(futures::future::join(
            registry::close_async(fid),
            registry::close_async(fid),
        ));
        let closed_count = [first.expect("first close"), second.expect("second close")]
            .into_iter()
            .filter(Option::is_some)
            .count();

        assert_eq!(closed_count, 1);
        assert!(registry::info_for(fid).is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_standard_stream_bool_argument() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = registry_guard();
        registry::reset_for_tests();
        let eval = run_evaluate(&[Value::Bool(true)]).expect("fclose stdout");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
        let outputs = eval.outputs();
        assert_eq!(outputs.len(), 2);
        assert!(matches!(outputs[1], Value::CharArray(ref ca) if ca.rows == 1 && ca.cols == 0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_logical_array_converts_to_numeric_ids() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = registry_guard();
        registry::reset_for_tests();
        let logical = LogicalArray::new(vec![1u8, 0u8, 1u8], vec![3]).expect("logical array");
        let eval = run_evaluate(&[Value::LogicalArray(logical)]).expect("fclose logical");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_cell_array_closes_each_entry() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid1, path1) = open_temp_file("fclose_cell1");
        let (fid2, path2) = open_temp_file("fclose_cell2");
        let cell = CellArray::new(vec![Value::Num(fid1), Value::Num(fid2)], 1, 2).expect("cell");
        let eval = run_evaluate(&[Value::Cell(cell)]).expect("fclose cell");
        assert_eq!(eval.status(), 0.0);
        assert!(registry::info_for(fid1 as i32).is_none());
        assert!(registry::info_for(fid2 as i32).is_none());
        test_support::fs::remove_file(path1).unwrap();
        test_support::fs::remove_file(path2).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_tensor_with_non_integer_entries_errors() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = registry_guard();
        registry::reset_for_tests();
        let tensor = Tensor::new(vec![1.5], vec![1, 1]).expect("tensor");
        let err = unwrap_error_message(run_evaluate(&[Value::Tensor(tensor)]).unwrap_err());
        assert_eq!(
            err,
            format!(
                "{}: file identifier must be an integer",
                FCLOSE_ERROR_INVALID_INPUT.message
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_string_array_all_equivalent() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = registry_guard();
        registry::reset_for_tests();
        let strings = StringArray::new(vec!["all".to_string()], vec![1]).expect("string array");
        let eval = run_evaluate(&[Value::StringArray(strings)]).expect("fclose all");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_accepts_empty_tensor() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = registry_guard();
        registry::reset_for_tests();
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).expect("tensor");
        let eval = run_evaluate(&[Value::Tensor(tensor)]).expect("fclose");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_errors_on_non_numeric_input() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = registry_guard();
        registry::reset_for_tests();
        let err = unwrap_error_message(run_evaluate(&[Value::from("not-a-fid")]).unwrap_err());
        assert_eq!(
            err,
            format!(
                "{}: file identifier must be numeric or 'all'",
                FCLOSE_ERROR_INVALID_INPUT.message
            )
        );
    }

    #[test]
    fn fclose_strict_extension_rejection_precedes_close_side_effects() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid, path) = open_temp_file("fclose_preflight");
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

        let integer = run_evaluate(&[Value::Int(IntValue::I64(fid as i64))]).unwrap_err();
        assert_eq!(
            integer.identifier(),
            Some("RunMat:compatibility:FcloseIntegerIdExtension")
        );
        assert!(registry::info_for(fid as i32).is_some());

        let _outputs = crate::output_count::push_output_count(Some(2));
        let message =
            futures::executor::block_on(fclose_builtin(vec![Value::Num(fid)])).unwrap_err();
        assert_eq!(
            message.identifier(),
            Some("RunMat:compatibility:FcloseMessageOutputExtension")
        );
        assert!(registry::info_for(fid as i32).is_some());

        drop(_outputs);
        run_evaluate(&[Value::Num(fid)]).expect("close after rejected forms");
        test_support::fs::remove_file(path).unwrap();
    }

    #[test]
    fn fclose_integer_extension_accepts_exact_scalar_and_documented_char_all() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid, path) = open_temp_file("fclose_integer_extension");
        {
            let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
            let result =
                run_evaluate(&[Value::Int(IntValue::U64(fid as u64))]).expect("integer file id");
            assert_eq!(result.status(), 0.0);
        }
        test_support::fs::remove_file(path).unwrap();

        let all = run_evaluate(&[Value::CharArray(runmat_builtins::CharArray::new_row("all"))])
            .expect("documented char all");
        assert_eq!(all.status(), 0.0);
    }
}
