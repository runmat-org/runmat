//! MATLAB-compatible `fread` builtin for RunMat.

use std::io::{ErrorKind, Read, Seek, SeekFrom};

use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor,
    BuiltinParamType, BuiltinSignatureDescriptor, CharArray, IntValue, IntegerStorage,
    LogicalArray, NumericDType, NumericScalar, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::io::filetext::{helpers::extract_scalar_string, registry};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use runmat_filesystem::File;

const BUILTIN_NAME: &str = "fread";

const FREAD_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fread-like",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "the fread \"like\" prototype selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FreadLikeExtension"),
};

pub const FREAD_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [FREAD_LIKE_EXTENSION];

const FREAD_OUTPUT_DATA: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Read data as numeric tensor or character array depending on precision.",
}];
const FREAD_INPUTS_FID: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier opened by fopen.",
}];
const FREAD_INPUTS_FID_SIZE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "size",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"inf\""),
        description: "Element count or size vector ([m n]); supports \"inf\".",
    },
];
const FREAD_INPUTS_FID_SIZE_PRECISION: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "size",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"inf\""),
        description: "Element count or size vector ([m n]); supports \"inf\".",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label (for example \"double\", \"uint8\", \"*char\").",
    },
];
const FREAD_INPUTS_FID_SIZE_PRECISION_SKIP: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "size",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"inf\""),
        description: "Element count or size vector ([m n]); supports \"inf\".",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label (for example \"double\", \"uint8\", \"*char\").",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element read.",
    },
];
const FREAD_INPUTS_FID_SIZE_PRECISION_SKIP_MACHINEFMT: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "size",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"inf\""),
        description: "Element count or size vector ([m n]); supports \"inf\".",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label (for example \"double\", \"uint8\", \"*char\").",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element read.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine format label (native/little-endian/big-endian aliases).",
    },
];
const FREAD_INPUTS_FID_PRECISION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label when size is omitted.",
    },
];
const FREAD_INPUTS_FID_PRECISION_SKIP: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label when size is omitted.",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element read.",
    },
];
const FREAD_INPUTS_FID_PRECISION_MACHINEFMT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label when size is omitted.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine format label (native/little-endian/big-endian aliases).",
    },
];
const FREAD_INPUTS_FID_PRECISION_SKIP_MACHINEFMT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label when size is omitted.",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element read.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine format label (native/little-endian/big-endian aliases).",
    },
];
const FREAD_INPUTS_WITH_LIKE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "arg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Positional fread arguments before the like clause.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Optional,
        default: Some("\"like\""),
        description: "Prototype keyword; currently only 'like'.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Prototype value controlling output class/residency.",
    },
];
const FREAD_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "data = fread(fid)",
        inputs: &FREAD_INPUTS_FID,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, size)",
        inputs: &FREAD_INPUTS_FID_SIZE,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, size, precision)",
        inputs: &FREAD_INPUTS_FID_SIZE_PRECISION,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, size, precision, skip)",
        inputs: &FREAD_INPUTS_FID_SIZE_PRECISION_SKIP,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, size, precision, skip, machinefmt)",
        inputs: &FREAD_INPUTS_FID_SIZE_PRECISION_SKIP_MACHINEFMT,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, precision)",
        inputs: &FREAD_INPUTS_FID_PRECISION,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, precision, skip)",
        inputs: &FREAD_INPUTS_FID_PRECISION_SKIP,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, precision, machinefmt)",
        inputs: &FREAD_INPUTS_FID_PRECISION_MACHINEFMT,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, precision, skip, machinefmt)",
        inputs: &FREAD_INPUTS_FID_PRECISION_SKIP_MACHINEFMT,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, ..., \"like\", prototype)",
        inputs: &FREAD_INPUTS_WITH_LIKE,
        outputs: &FREAD_OUTPUT_DATA,
    },
];

const FREAD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREAD.INVALID_INPUT",
    identifier: Some("RunMat:fread:InvalidInput"),
    when: "Identifier/argument cardinality/type constraints are violated.",
    message: "fread: invalid input arguments",
};
const FREAD_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREAD.INVALID_IDENTIFIER",
    identifier: Some("RunMat:fread:InvalidIdentifier"),
    when: "Identifier does not refer to a readable open file.",
    message: "fread: invalid file identifier. Use fopen to generate a valid file ID.",
};
const FREAD_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREAD.INVALID_OPTION",
    identifier: Some("RunMat:fread:InvalidOption"),
    when: "Precision, skip, machine format, or like option values are invalid.",
    message: "fread: invalid option configuration",
};
const FREAD_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREAD.IO",
    identifier: Some("RunMat:fread:IoFailure"),
    when: "Read/seek or data-shape materialization fails.",
    message: "fread: file read failed",
};
const FREAD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREAD.INTERNAL",
    identifier: None,
    when: "Internal runtime control-flow conversion failed.",
    message: "fread: internal error",
};
const FREAD_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FREAD_ERROR_INVALID_INPUT,
    FREAD_ERROR_INVALID_IDENTIFIER,
    FREAD_ERROR_INVALID_OPTION,
    FREAD_ERROR_IO,
    FREAD_ERROR_INTERNAL,
];
pub const FREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FREAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FREAD_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fread")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fread",
    op_kind: GpuOpKind::Custom("file-io-read"),
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
        "Host-only operation that reads from the shared file registry; GPU arguments are gathered to the CPU before I/O.",
};

fn fread_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let detail = detail.strip_prefix("fread: ").unwrap_or(detail);
    fread_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn fread_error_with_message(
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
    if let Some(identifier) = FREAD_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_string_result<T>(
    result: Result<T, String>,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<T> {
    result.map_err(|detail| fread_error_with_detail(error, detail))
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fread")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fread",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O cannot participate in fusion; metadata is registered for completeness.",
};

#[runtime_builtin(
    name = "fread",
    category = "io/filetext",
    summary = "Read binary data from file identifiers.",
    keywords = "fread,file,io,binary,precision",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fread_type),
    descriptor(crate::builtins::io::filetext::fread::FREAD_DESCRIPTOR),
    extensions(crate::builtins::io::filetext::fread::FREAD_EXTENSIONS),
    builtin_path = "crate::builtins::io::filetext::fread"
)]
async fn fread_builtin(fid: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&fid, &rest).await?;
    Ok(eval.first_output())
}

#[derive(Debug, Clone)]
pub struct FreadEval {
    data: Value,
    count: usize,
}

impl FreadEval {
    fn new(data: Value, count: usize) -> Self {
        Self { data, count }
    }

    pub fn first_output(&self) -> Value {
        self.data.clone()
    }

    pub fn outputs(&self) -> Vec<Value> {
        vec![self.data.clone(), Value::Num(self.count as f64)]
    }

    fn apply_like(
        &mut self,
        like_proto: Option<&Value>,
        precision: PrecisionSpec,
    ) -> Result<(), String> {
        if let Some(proto) = like_proto {
            let adjusted = adjust_output_for_like(self.data.clone(), proto, precision)?;
            self.data = adjusted;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn data(&self) -> &Value {
        &self.data
    }

    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        self.count
    }
}

pub async fn evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FreadEval> {
    let fid_host = gather_value(fid_value).await?;
    let fid = map_string_result(parse_fid(&fid_host), &FREAD_ERROR_INVALID_INPUT)?;
    if fid < 0 {
        return Err(fread_error_with_detail(
            &FREAD_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Err(fread_error_with_detail(
            &FREAD_ERROR_INVALID_INPUT,
            "standard input/output identifiers are not supported yet",
        ));
    }

    let info = registry::info_for(fid).ok_or_else(|| {
        fread_error_with_message(
            FREAD_ERROR_INVALID_IDENTIFIER.message,
            &FREAD_ERROR_INVALID_IDENTIFIER,
        )
    })?;
    let handle = registry::shared_handle(fid).ok_or_else(|| {
        fread_error_with_message(
            FREAD_ERROR_INVALID_IDENTIFIER.message,
            &FREAD_ERROR_INVALID_IDENTIFIER,
        )
    })?;

    let arg_refs: Vec<&Value> = rest.iter().collect();
    let (size_arg, precision_arg, skip_arg, machine_arg, like_arg) =
        map_string_result(classify_arguments(&arg_refs), &FREAD_ERROR_INVALID_INPUT)?;
    if like_arg.is_some() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FREAD_LIKE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }

    let size_host = match size_arg {
        Some(value) => Some(gather_value(value).await?),
        None => None,
    };
    let precision_host = match precision_arg {
        Some(value) => Some(gather_value(value).await?),
        None => None,
    };
    let skip_host = match skip_arg {
        Some(value) => Some(gather_value(value).await?),
        None => None,
    };
    let machine_host = match machine_arg {
        Some(value) => Some(gather_value(value).await?),
        None => None,
    };

    let size_spec = map_string_result(parse_size(size_host.as_ref()), &FREAD_ERROR_INVALID_INPUT)?;
    let precision = map_string_result(
        parse_precision(precision_host.as_ref()),
        &FREAD_ERROR_INVALID_OPTION,
    )?;
    let skip_bytes =
        map_string_result(parse_skip(skip_host.as_ref()), &FREAD_ERROR_INVALID_OPTION)?;
    let machine_format = map_string_result(
        parse_machine_format(machine_host.as_ref(), &info.machinefmt),
        &FREAD_ERROR_INVALID_OPTION,
    )?;

    let mut guard = handle.lock().map_err(|_| {
        fread_error_with_detail(
            &FREAD_ERROR_INTERNAL,
            "failed to lock file handle (poisoned mutex)",
        )
    })?;
    let file = guard.as_mut().ok_or_else(|| {
        fread_error_with_message(
            FREAD_ERROR_INVALID_IDENTIFIER.message,
            &FREAD_ERROR_INVALID_IDENTIFIER,
        )
    })?;

    let mut eval = map_string_result(
        read_from_handle(file, &size_spec, &precision, skip_bytes, machine_format),
        &FREAD_ERROR_IO,
    )?;
    map_string_result(
        eval.apply_like(like_arg, precision),
        &FREAD_ERROR_INVALID_OPTION,
    )?;
    Ok(eval)
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value)
        .await
        .map_err(map_control_flow)
}

fn parse_fid(value: &Value) -> Result<i32, String> {
    let number = match value {
        Value::Num(n) => *n,
        Value::Int(int) => {
            return int
                .try_to_i32()
                .ok_or_else(|| "file identifier is out of range".to_string());
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                return int
                    .try_to_i32()
                    .ok_or_else(|| "file identifier is out of range".to_string());
            }
            tensor::tensor_value_f64(t, 0)
        }
        _ => {
            return Err("file identifier must be numeric".to_string());
        }
    };
    if !number.is_finite() {
        return Err("file identifier must be finite".to_string());
    }
    let rounded = number.round();
    if (rounded - number).abs() > f64::EPSILON {
        return Err("file identifier must be an integer".to_string());
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err("file identifier is out of range".to_string());
    }
    Ok(rounded as i32)
}

type ClassifiedArgs<'a> = (
    Option<&'a Value>,
    Option<&'a Value>,
    Option<&'a Value>,
    Option<&'a Value>,
    Option<&'a Value>,
);

fn classify_arguments<'a>(args: &'a [&'a Value]) -> Result<ClassifiedArgs<'a>, String> {
    let mut filtered_indices: Vec<usize> = Vec::with_capacity(args.len());
    let mut like_proto: Option<&Value> = None;
    let mut i = 0usize;

    while i < args.len() {
        let value = args[i];
        if matches_keyword(value, "like") {
            if like_proto.is_some() {
                return Err("multiple 'like' prototypes are not supported".to_string());
            }
            i += 1;
            let Some(proto_value) = args.get(i) else {
                return Err("expected prototype after 'like'".to_string());
            };
            like_proto = Some(*proto_value);
            i += 1;
            continue;
        }
        filtered_indices.push(i);
        i += 1;
    }

    if filtered_indices.len() > 4 {
        return Err("too many input arguments".to_string());
    }

    let (size_idx, precision_idx, skip_idx, machine_idx) =
        classify_ordered_indices(args, &filtered_indices)?;

    let size = size_idx.map(|index| args[index]);
    let precision = precision_idx.map(|index| args[index]);
    let skip = skip_idx.map(|index| args[index]);
    let machine = machine_idx.map(|index| args[index]);

    Ok((size, precision, skip, machine, like_proto))
}

type ClassifiedIndices = (Option<usize>, Option<usize>, Option<usize>, Option<usize>);

fn classify_ordered_indices(
    args: &[&Value],
    indices: &[usize],
) -> Result<ClassifiedIndices, String> {
    let mut position = 0usize;
    let mut size_idx: Option<usize> = None;
    let mut precision_idx: Option<usize> = None;
    let mut skip_idx: Option<usize> = None;
    let mut machine_idx: Option<usize> = None;

    if let Some(&first_index) = indices.get(position) {
        let first = args[first_index];
        if is_string_like(first) {
            precision_idx = Some(first_index);
        } else {
            size_idx = Some(first_index);
        }
        position += 1;
    }

    if let Some(&index) = indices.get(position) {
        let candidate = args[index];
        if precision_idx.is_none() && is_string_like(candidate) {
            precision_idx = Some(index);
            position += 1;
        } else if is_numeric_like(candidate) {
            skip_idx = Some(index);
            position += 1;
        } else if is_string_like(candidate) {
            machine_idx = Some(index);
            position += 1;
        } else {
            return Err("invalid argument combination".to_string());
        }
    }

    if let Some(&index) = indices.get(position) {
        let candidate = args[index];
        if skip_idx.is_none() && is_numeric_like(candidate) {
            skip_idx = Some(index);
            position += 1;
        } else if machine_idx.is_none() && is_string_like(candidate) {
            machine_idx = Some(index);
            position += 1;
        } else {
            return Err("invalid argument combination".to_string());
        }
    }

    if let Some(&index) = indices.get(position) {
        let candidate = args[index];
        if machine_idx.is_none() && is_string_like(candidate) {
            machine_idx = Some(index);
            position += 1;
        } else {
            return Err("too many input arguments".to_string());
        }
    }

    if position < indices.len() {
        return Err("too many input arguments".to_string());
    }

    Ok((size_idx, precision_idx, skip_idx, machine_idx))
}

fn is_string_like(value: &Value) -> bool {
    match value {
        Value::String(_) => true,
        Value::CharArray(ca) if ca.rows == 1 => true,
        Value::StringArray(sa) if sa.data.len() == 1 => true,
        _ => false,
    }
}

fn matches_keyword(value: &Value, keyword: &str) -> bool {
    extract_scalar_string(value)
        .map(|text| text.eq_ignore_ascii_case(keyword))
        .unwrap_or(false)
}

fn is_numeric_like(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Tensor(_) | Value::LogicalArray(_)
    )
}

#[derive(Clone, Debug)]
enum SizeSpec {
    All,
    Count(usize),
    Matrix { rows: usize, cols: Option<usize> },
}

impl SizeSpec {
    fn element_limit(&self) -> Option<usize> {
        match self {
            SizeSpec::All => None,
            SizeSpec::Count(n) => Some(*n),
            SizeSpec::Matrix { rows, cols } => {
                if *rows == 0 {
                    return Some(0);
                }
                cols.map(|c| rows.saturating_mul(c))
            }
        }
    }
}

fn parse_size(arg: Option<&Value>) -> Result<SizeSpec, String> {
    match arg {
        None => Ok(SizeSpec::All),
        Some(Value::String(s)) => parse_size_string(s),
        Some(Value::CharArray(ca)) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            parse_size_string(&text)
        }
        Some(Value::StringArray(sa)) if sa.data.len() == 1 => parse_size_string(&sa.data[0]),
        Some(Value::Tensor(t)) => parse_size_tensor(t),
        Some(Value::Int(int)) => Ok(SizeSpec::Count(int_to_usize(
            int,
            "size argument must be a non-negative integer",
        )?)),
        Some(value) => {
            let scalar = value_to_scalar(value, "size argument must be numeric or a size vector")?;
            scalar_to_size(scalar)
        }
    }
}

fn parse_size_string(text: &str) -> Result<SizeSpec, String> {
    if text.trim().is_empty() {
        return Err("size argument must not be empty".to_string());
    }
    let lower = text.trim().to_ascii_lowercase();
    if lower == "inf" {
        Ok(SizeSpec::All)
    } else {
        let number = lower
            .parse::<f64>()
            .map_err(|_| "size argument must be numeric or 'inf'".to_string())?;
        scalar_to_size(number)
    }
}

fn parse_size_tensor(t: &Tensor) -> Result<SizeSpec, String> {
    if let Some(storage) = t.integer_storage() {
        return match storage.len() {
            0 => Ok(SizeSpec::Count(0)),
            1 => {
                let value = storage.value_at(0).expect("one-element integer storage");
                Ok(SizeSpec::Count(value.try_to_usize().ok_or_else(|| {
                    "size argument must be a non-negative integer".to_string()
                })?))
            }
            2 => {
                let rows_value = storage
                    .value_at(0)
                    .expect("integer storage length matches tensor length");
                let cols_value = storage
                    .value_at(1)
                    .expect("integer storage length matches tensor length");
                let rows = rows_value.try_to_usize().ok_or_else(|| {
                    "size vector components must be non-negative integers or Inf".to_string()
                })?;
                let cols = cols_value.try_to_usize().ok_or_else(|| {
                    "size vector components must be non-negative integers or Inf".to_string()
                })?;
                Ok(SizeSpec::Matrix {
                    rows,
                    cols: Some(cols),
                })
            }
            _ => Err("size vector must contain at most two elements".to_string()),
        };
    }

    match t.len() {
        0 => Ok(SizeSpec::Count(0)),
        1 => scalar_to_size(tensor::tensor_value_f64(t, 0)),
        2 => {
            let rows = scalar_to_size_component(
                tensor::tensor_value_f64(t, 0),
                "size vector components must be non-negative integers or Inf",
            )?;
            let cols_raw = tensor::tensor_value_f64(t, 1);
            if cols_raw.is_infinite() && cols_raw.is_sign_positive() {
                Ok(SizeSpec::Matrix { rows, cols: None })
            } else {
                let cols = scalar_to_size_component(
                    cols_raw,
                    "size vector components must be non-negative integers or Inf",
                )?;
                Ok(SizeSpec::Matrix {
                    rows,
                    cols: Some(cols),
                })
            }
        }
        _ => Err("size vector must contain at most two elements".to_string()),
    }
}

fn scalar_to_size(value: f64) -> Result<SizeSpec, String> {
    if value.is_infinite() && value.is_sign_positive() {
        return Ok(SizeSpec::All);
    }
    let count = scalar_to_size_component(value, "size argument must be a non-negative integer")?;
    Ok(SizeSpec::Count(count))
}

fn scalar_to_size_component(value: f64, err: &str) -> Result<usize, String> {
    if !value.is_finite() {
        return Err(err.to_string());
    }
    if value < 0.0 {
        return Err(err.to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(err.to_string());
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err("size argument is too large".to_string());
    }
    Ok(rounded as usize)
}

#[derive(Clone, Copy, Debug)]
enum InputType {
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Float32,
    Float64,
}

impl InputType {
    fn byte_len(&self) -> usize {
        match self {
            InputType::UInt8 | InputType::Int8 => 1,
            InputType::UInt16 | InputType::Int16 => 2,
            InputType::UInt32 | InputType::Int32 | InputType::Float32 => 4,
            InputType::UInt64 | InputType::Int64 | InputType::Float64 => 8,
        }
    }

    fn numeric_dtype(self) -> NumericDType {
        match self {
            Self::UInt8 => NumericDType::U8,
            Self::Int8 => NumericDType::I8,
            Self::UInt16 => NumericDType::U16,
            Self::Int16 => NumericDType::I16,
            Self::UInt32 => NumericDType::U32,
            Self::Int32 => NumericDType::I32,
            Self::UInt64 => NumericDType::U64,
            Self::Int64 => NumericDType::I64,
            Self::Float32 => NumericDType::F32,
            Self::Float64 => NumericDType::F64,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum OutputKind {
    Numeric(NumericDType),
    Char,
}

#[derive(Clone, Copy, Debug)]
struct PrecisionSpec {
    input: InputType,
    output: OutputKind,
    repeat: usize,
}

impl PrecisionSpec {
    fn default() -> Self {
        Self {
            input: InputType::UInt8,
            output: OutputKind::Numeric(NumericDType::F64),
            repeat: 1,
        }
    }
}

fn parse_precision(arg: Option<&Value>) -> Result<PrecisionSpec, String> {
    match arg {
        None => Ok(PrecisionSpec::default()),
        Some(value) => {
            let text = scalar_string(
                value,
                "precision argument must be a string scalar or character vector",
            )?;
            parse_precision_string(&text)
        }
    }
}

fn parse_precision_string(raw: &str) -> Result<PrecisionSpec, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("precision argument must not be empty".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix('*') {
        parse_star_precision(rest.trim())
    } else if let Some((lhs, rhs)) = lower.split_once("=>") {
        let (repeat, input) = parse_repeated_input_label(lhs.trim())?;
        let output = parse_output_label(rhs.trim())?;
        Ok(PrecisionSpec {
            input,
            output,
            repeat,
        })
    } else {
        let (repeat, input) = parse_repeated_input_label(lower.trim())?;
        let wants_char =
            lower == "char" || (matches!(input, InputType::UInt8) && lower.contains("char"));
        let output = if wants_char {
            OutputKind::Char
        } else {
            OutputKind::Numeric(NumericDType::F64)
        };
        Ok(PrecisionSpec {
            input,
            output,
            repeat,
        })
    }
}

fn parse_star_precision(label: &str) -> Result<PrecisionSpec, String> {
    if label == "char" {
        return Ok(PrecisionSpec {
            input: InputType::UInt8,
            output: OutputKind::Char,
            repeat: 1,
        });
    }
    let input = parse_input_label(label)?;
    Ok(PrecisionSpec {
        input,
        output: OutputKind::Numeric(input.numeric_dtype()),
        repeat: 1,
    })
}

fn parse_repeated_input_label(label: &str) -> Result<(usize, InputType), String> {
    if let Some((repeat, source)) = label.split_once('*') {
        if repeat.trim().chars().all(|ch| ch.is_ascii_digit()) {
            let repeat = repeat
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|&repeat| repeat > 0)
                .ok_or_else(|| "precision repeat count must be a positive integer".to_string())?;
            return Ok((repeat, parse_input_label(source.trim())?));
        }
    }
    Ok((1, parse_input_label(label)?))
}

fn parse_input_label(label: &str) -> Result<InputType, String> {
    match label {
        "double" | "float64" | "real*8" => Ok(InputType::Float64),
        "single" | "float" | "float32" | "real*4" => Ok(InputType::Float32),
        "int8" | "schar" | "signedchar" | "signed char" | "integer*1" => Ok(InputType::Int8),
        "uint8" | "uchar" | "unsignedchar" | "unsigned char" | "char" | "byte" => {
            Ok(InputType::UInt8)
        }
        "int16" | "short" | "integer*2" => Ok(InputType::Int16),
        "uint16" | "ushort" | "unsignedshort" | "unsigned short" => Ok(InputType::UInt16),
        "int32" | "int" | "integer*4" | "long" => Ok(InputType::Int32),
        "uint32" | "uint" | "unsignedint" | "unsigned int" | "unsignedlong" | "unsigned long" => {
            Ok(InputType::UInt32)
        }
        "int64" | "integer*8" | "longlong" | "long long" => Ok(InputType::Int64),
        "uint64" | "unsignedlonglong" | "unsigned long long" => Ok(InputType::UInt64),
        other => Err(format!("unsupported precision '{other}'")),
    }
}

fn parse_output_label(label: &str) -> Result<OutputKind, String> {
    match label {
        "double" | "float64" | "real*8" => Ok(OutputKind::Numeric(NumericDType::F64)),
        "single" | "float" | "float32" | "real*4" => Ok(OutputKind::Numeric(NumericDType::F32)),
        "int8" | "schar" | "signedchar" | "signed char" | "integer*1" => {
            Ok(OutputKind::Numeric(NumericDType::I8))
        }
        "int16" | "short" | "integer*2" => Ok(OutputKind::Numeric(NumericDType::I16)),
        "int32" | "int" | "integer*4" | "long" => Ok(OutputKind::Numeric(NumericDType::I32)),
        "int64" | "integer*8" | "longlong" | "long long" => {
            Ok(OutputKind::Numeric(NumericDType::I64))
        }
        "uint8" | "uchar" | "unsignedchar" | "unsigned char" => {
            Ok(OutputKind::Numeric(NumericDType::U8))
        }
        "uint16" | "ushort" | "unsignedshort" | "unsigned short" => {
            Ok(OutputKind::Numeric(NumericDType::U16))
        }
        "uint32" | "uint" | "unsignedint" | "unsigned int" | "unsignedlong" | "unsigned long" => {
            Ok(OutputKind::Numeric(NumericDType::U32))
        }
        "uint64" | "unsignedlonglong" | "unsigned long long" => {
            Ok(OutputKind::Numeric(NumericDType::U64))
        }
        "char" => Ok(OutputKind::Char),
        other => Err(format!("output class '{other}' is not implemented yet")),
    }
}

fn parse_skip(arg: Option<&Value>) -> Result<usize, String> {
    match arg {
        None => Ok(0),
        Some(Value::Int(int)) => int_to_skip(int),
        Some(Value::Tensor(t)) if tensor::is_scalar_tensor(t) => {
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                return int_to_skip(&int);
            }
            parse_skip_scalar(tensor::tensor_value_f64(t, 0))
        }
        Some(value) => {
            let scalar = value_to_scalar(value, "skip value must be numeric")?;
            parse_skip_scalar(scalar)
        }
    }
}

fn parse_skip_scalar(scalar: f64) -> Result<usize, String> {
    if !scalar.is_finite() {
        return Err("skip value must be finite".to_string());
    }
    if scalar < 0.0 {
        return Err("skip value must be non-negative".to_string());
    }
    let rounded = scalar.round();
    if (rounded - scalar).abs() > f64::EPSILON {
        return Err("skip value must be an integer".to_string());
    }
    if rounded >= i64::MAX as f64 {
        return Err("skip value is too large".to_string());
    }
    Ok(rounded as usize)
}

fn int_to_skip(value: &IntValue) -> Result<usize, String> {
    let skip = int_to_usize(value, "skip value must be non-negative")?;
    if skip > i64::MAX as usize {
        return Err("skip value is too large".to_string());
    }
    Ok(skip)
}

fn int_to_usize(value: &IntValue, err: &str) -> Result<usize, String> {
    value.try_to_usize().ok_or_else(|| err.to_string())
}

#[derive(Clone, Copy, Debug)]
enum MachineFormat {
    Native,
    LittleEndian,
    BigEndian,
}

#[derive(Clone, Copy, Debug)]
enum Endianness {
    Little,
    Big,
}

impl MachineFormat {
    fn to_endianness(self) -> Endianness {
        match self {
            MachineFormat::Native => {
                if cfg!(target_endian = "little") {
                    Endianness::Little
                } else {
                    Endianness::Big
                }
            }
            MachineFormat::LittleEndian => Endianness::Little,
            MachineFormat::BigEndian => Endianness::Big,
        }
    }
}

fn parse_machine_format(arg: Option<&Value>, default_label: &str) -> Result<MachineFormat, String> {
    match arg {
        Some(value) => {
            let text = scalar_string(
                value,
                "machine format must be a string scalar or character vector",
            )?;
            machine_format_from_label(&text)
        }
        None => machine_format_from_label(default_label),
    }
}

fn machine_format_from_label(label: &str) -> Result<MachineFormat, String> {
    let trimmed = label.trim();
    if trimmed.is_empty() {
        return Err("machine format must not be empty".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    let collapsed: String = lower
        .chars()
        .filter(|c| !matches!(c, '-' | '_' | ' '))
        .collect();
    if matches!(collapsed.as_str(), "native" | "n" | "system" | "default") {
        return Ok(MachineFormat::Native);
    }
    if matches!(
        collapsed.as_str(),
        "l" | "le" | "littleendian" | "pc" | "intel"
    ) {
        return Ok(MachineFormat::LittleEndian);
    }
    if matches!(
        collapsed.as_str(),
        "b" | "be" | "bigendian" | "mac" | "motorola"
    ) {
        return Ok(MachineFormat::BigEndian);
    }
    if lower.starts_with("ieee-le") {
        return Ok(MachineFormat::LittleEndian);
    }
    if lower.starts_with("ieee-be") {
        return Ok(MachineFormat::BigEndian);
    }
    Err(format!("unsupported machine format '{trimmed}'"))
}

fn scalar_string(value: &Value, err: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(err.to_string()),
    }
}

fn value_to_scalar(value: &Value, err: &str) -> Result<f64, String> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(int) => Ok(int.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Ok(tensor::tensor_value_f64(t, 0)),
        Value::LogicalArray(la) if la.data.len() == 1 => {
            Ok(if la.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(err.to_string()),
    }
}

fn read_from_handle(
    file: &mut File,
    size_spec: &SizeSpec,
    precision: &PrecisionSpec,
    skip: usize,
    machine: MachineFormat,
) -> Result<FreadEval, String> {
    let endianness = machine.to_endianness();
    match precision.output {
        OutputKind::Numeric(output_dtype) => {
            let limit = size_spec.element_limit();
            let (values, count) = read_numeric_values(
                file,
                precision.input,
                precision.repeat,
                limit,
                skip,
                endianness,
            )?;
            let (data, rows, cols) = finalize_numeric(size_spec, count, values);
            let tensor = Tensor::from_numeric_storage(data, vec![rows, cols])
                .map_err(|e| format!("fread: {e}"))?;
            let tensor = if tensor.numeric_dtype() == output_dtype {
                tensor
            } else {
                tensor::coerce_tensor_dtype(tensor, output_dtype)
            };
            Ok(FreadEval::new(Value::Tensor(tensor), count))
        }
        OutputKind::Char => {
            let limit = size_spec.element_limit();
            let (values, count) = read_char_values(
                file,
                precision.input,
                precision.repeat,
                limit,
                skip,
                endianness,
            )?;
            let (row_major, rows, cols) = finalize_char(size_spec, count, values);
            let char_array =
                CharArray::new(row_major, rows, cols).map_err(|e| format!("fread: {e}"))?;
            Ok(FreadEval::new(Value::CharArray(char_array), count))
        }
    }
}

fn adjust_output_for_like(
    data: Value,
    prototype: &Value,
    precision: PrecisionSpec,
) -> Result<Value, String> {
    if matches!(prototype, Value::GpuTensor(_)) {
        return match data {
            Value::Tensor(tensor) => tensor_to_gpu_value(tensor),
            Value::CharArray(_) => {
                Err("fread: character output cannot be returned on the GPU via 'like'".to_string())
            }
            other => Ok(other),
        };
    }

    match prototype {
        Value::LogicalArray(_) | Value::Bool(_) => convert_to_logical_value(data),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            if !matches!(precision.output, OutputKind::Char) {
                return Err(
                    "fread: character prototypes require a character precision such as '*char'"
                        .to_string(),
                );
            }
            ensure_char_result(data)
        }
        Value::Tensor(tensor) => tensor_to_numeric_like(data, tensor.numeric_dtype()),
        Value::Int(value) => tensor_to_numeric_like(
            data,
            IntegerStorage::from_scalar(value.clone()).numeric_dtype(),
        ),
        Value::Num(_) => tensor_to_numeric_like(data, NumericDType::F64),
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            Err("fread: complex prototypes are not supported yet".to_string())
        }
        Value::Cell(_) => Err("fread: cell prototypes are not supported".to_string()),
        _ => Ok(data),
    }
}

fn tensor_to_numeric_like(data: Value, dtype: NumericDType) -> Result<Value, String> {
    match data {
        Value::Tensor(tensor) => Ok(Value::Tensor(tensor::coerce_tensor_dtype(tensor, dtype))),
        Value::CharArray(_) => Err(
            "fread: character output cannot be converted to a numeric 'like' prototype".to_string(),
        ),
        other => Ok(other),
    }
}

fn ensure_char_result(data: Value) -> Result<Value, String> {
    match data {
        Value::CharArray(_) => Ok(data),
        _ => Err("fread: expected character output when using a character prototype".to_string()),
    }
}

fn tensor_to_gpu_value(tensor: Tensor) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let data = tensor.as_f64_slice().ok_or_else(|| {
            "fread: non-double output cannot be returned on the GPU via 'like' yet".to_string()
        })?;
        let view = HostTensorView {
            data,
            shape: &tensor.shape,
        };
        if let Ok(handle) = provider.upload(&view) {
            return Ok(Value::GpuTensor(handle));
        }
    }
    Ok(Value::Tensor(tensor))
}

fn convert_to_logical_value(data: Value) -> Result<Value, String> {
    match data {
        Value::LogicalArray(_) => Ok(data),
        Value::Tensor(tensor) => {
            let bits = (0..tensor.len())
                .map(|index| {
                    let value = tensor
                        .numeric_value_at(index)
                        .expect("validated fread numeric tensor index");
                    match value {
                        NumericScalar::F64(value) => u8::from(value != 0.0),
                        NumericScalar::F32(value) => u8::from(value != 0.0),
                        value => u8::from(
                            !value
                                .into_int_value()
                                .expect("non-floating numeric scalar is integer")
                                .is_zero(),
                        ),
                    }
                })
                .collect();
            LogicalArray::new(bits, tensor.shape.clone())
                .map(Value::LogicalArray)
                .map_err(|e| format!("fread: {e}"))
        }
        Value::CharArray(ca) => {
            let total = ca.rows.saturating_mul(ca.cols);
            let mut bits = Vec::with_capacity(total);
            for c in 0..ca.cols {
                for r in 0..ca.rows {
                    let idx = r * ca.cols + c;
                    let ch = ca.data[idx];
                    bits.push(if ch != '\0' { 1 } else { 0 });
                }
            }
            LogicalArray::new(bits, vec![ca.rows, ca.cols])
                .map(Value::LogicalArray)
                .map_err(|e| format!("fread: {e}"))
        }
        _ => Err(
            "fread: logical prototypes require numeric or character output from the read"
                .to_string(),
        ),
    }
}

fn read_numeric_values<R: Read + Seek>(
    reader: &mut R,
    input: InputType,
    repeat: usize,
    limit: Option<usize>,
    skip: usize,
    endianness: Endianness,
) -> Result<(NumericStorage, usize), String> {
    if let Some(0) = limit {
        return Ok((NumericStorage::zeros(input.numeric_dtype(), 0), 0));
    }
    let element_size = input.byte_len();
    let mut buffer = vec![0u8; element_size];
    let mut values = Vec::new();
    let mut count = 0usize;
    let target = limit.unwrap_or(usize::MAX);

    'outer: loop {
        if count >= target {
            break;
        }
        let mut remaining = element_size;
        while remaining > 0 {
            match reader.read(&mut buffer[element_size - remaining..element_size]) {
                Ok(0) => break 'outer,
                Ok(n) => remaining -= n,
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(err) => {
                    return Err(format!("fread: failed to read from file ({err})"));
                }
            }
        }
        if remaining > 0 {
            break;
        }
        let value = decode_numeric_scalar(&buffer, input, endianness)?;
        values.push(value);
        count += 1;
        if skip > 0 && count % repeat == 0 {
            reader
                .seek(SeekFrom::Current(skip as i64))
                .map_err(|err| format!("fread: failed to skip bytes ({err})"))?;
        }
    }
    Ok((numeric_storage_from_scalars(input, values)?, count))
}

fn read_char_values<R: Read + Seek>(
    reader: &mut R,
    input: InputType,
    repeat: usize,
    limit: Option<usize>,
    skip: usize,
    endianness: Endianness,
) -> Result<(Vec<char>, usize), String> {
    if let Some(0) = limit {
        return Ok((Vec::new(), 0));
    }
    let element_size = input.byte_len();
    let mut buffer = vec![0u8; element_size];
    let mut values = Vec::new();
    let mut count = 0usize;
    let target = limit.unwrap_or(usize::MAX);

    'outer: loop {
        if count >= target {
            break;
        }
        let mut remaining = element_size;
        while remaining > 0 {
            match reader.read(&mut buffer[element_size - remaining..element_size]) {
                Ok(0) => break 'outer,
                Ok(n) => remaining -= n,
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(err) => {
                    return Err(format!("fread: failed to read from file ({err})"));
                }
            }
        }
        if remaining > 0 {
            break;
        }
        let ch = decode_to_char(&buffer, input, endianness)?;
        values.push(ch);
        count += 1;
        if skip > 0 && count % repeat == 0 {
            reader
                .seek(SeekFrom::Current(skip as i64))
                .map_err(|err| format!("fread: failed to skip bytes ({err})"))?;
        }
    }

    Ok((values, count))
}

fn decode_numeric_scalar(
    bytes: &[u8],
    input: InputType,
    endianness: Endianness,
) -> Result<NumericScalar, String> {
    Ok(match input {
        InputType::UInt8 => NumericScalar::U8(bytes[0]),
        InputType::Int8 => NumericScalar::I8(bytes[0] as i8),
        InputType::UInt16 => NumericScalar::U16(read_u16(bytes, endianness)),
        InputType::Int16 => NumericScalar::I16(read_u16(bytes, endianness) as i16),
        InputType::UInt32 => NumericScalar::U32(read_u32(bytes, endianness)),
        InputType::Int32 => NumericScalar::I32(read_u32(bytes, endianness) as i32),
        InputType::UInt64 => NumericScalar::U64(read_u64(bytes, endianness)),
        InputType::Int64 => NumericScalar::I64(read_u64(bytes, endianness) as i64),
        InputType::Float32 => {
            let bits = read_u32(bytes, endianness);
            NumericScalar::F32(f32::from_bits(bits))
        }
        InputType::Float64 => {
            let bits = read_u64(bytes, endianness);
            NumericScalar::F64(f64::from_bits(bits))
        }
    })
}

fn numeric_storage_from_scalars(
    input: InputType,
    values: Vec<NumericScalar>,
) -> Result<NumericStorage, String> {
    macro_rules! collect_variant {
        ($scalar_variant:ident, $storage_variant:ident) => {{
            let mut output = Vec::with_capacity(values.len());
            for value in values {
                let NumericScalar::$scalar_variant(value) = value else {
                    return Err("fread: decoded numeric source class mismatch".to_string());
                };
                output.push(value);
            }
            NumericStorage::$storage_variant(output)
        }};
    }
    Ok(match input {
        InputType::UInt8 => collect_variant!(U8, U8),
        InputType::Int8 => collect_variant!(I8, I8),
        InputType::UInt16 => collect_variant!(U16, U16),
        InputType::Int16 => collect_variant!(I16, I16),
        InputType::UInt32 => collect_variant!(U32, U32),
        InputType::Int32 => collect_variant!(I32, I32),
        InputType::UInt64 => collect_variant!(U64, U64),
        InputType::Int64 => collect_variant!(I64, I64),
        InputType::Float32 => collect_variant!(F32, F32),
        InputType::Float64 => collect_variant!(F64, F64),
    })
}

fn decode_to_char(bytes: &[u8], input: InputType, endianness: Endianness) -> Result<char, String> {
    let scalar = decode_numeric_scalar(bytes, input, endianness)?;
    let code = match scalar {
        NumericScalar::F64(value) => floating_char_code(value)?,
        NumericScalar::F32(value) => floating_char_code(f64::from(value))?,
        value => value
            .into_int_value()
            .and_then(|value| value.try_to_u64())
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| "fread: character value is outside the Unicode range".to_string())?,
    };
    char::from_u32(code).ok_or_else(|| {
        format!("value 0x{code:X} cannot be represented as a Unicode scalar for char output")
    })
}

fn floating_char_code(value: f64) -> Result<u32, String> {
    if !value.is_finite() || value < 0.0 || value > f64::from(u32::MAX) || value.round() != value {
        return Err("fread: character value is outside the Unicode range".to_string());
    }
    Ok(value as u32)
}

fn read_u16(bytes: &[u8], endianness: Endianness) -> u16 {
    match endianness {
        Endianness::Little => u16::from_le_bytes([bytes[0], bytes[1]]),
        Endianness::Big => u16::from_be_bytes([bytes[0], bytes[1]]),
    }
}

fn read_u32(bytes: &[u8], endianness: Endianness) -> u32 {
    match endianness {
        Endianness::Little => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        Endianness::Big => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    }
}

fn read_u64(bytes: &[u8], endianness: Endianness) -> u64 {
    match endianness {
        Endianness::Little => u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        Endianness::Big => u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
    }
}

fn finalize_numeric(
    size_spec: &SizeSpec,
    count_read: usize,
    mut values: NumericStorage,
) -> (NumericStorage, usize, usize) {
    match size_spec {
        SizeSpec::All | SizeSpec::Count(_) => {
            let rows = count_read;
            let cols = if count_read == 0 { 0 } else { 1 };
            (values, rows, cols)
        }
        SizeSpec::Matrix {
            rows,
            cols: Some(c),
        } => {
            let target = rows.saturating_mul(*c);
            resize_numeric_storage(&mut values, target);
            (values, *rows, *c)
        }
        SizeSpec::Matrix { rows, cols: None } => {
            if *rows == 0 {
                resize_numeric_storage(&mut values, 0);
                (values, 0, 0)
            } else {
                let cols = if count_read == 0 {
                    0
                } else {
                    count_read.div_ceil(*rows)
                };
                let target = rows.saturating_mul(cols);
                resize_numeric_storage(&mut values, target);
                (values, *rows, cols)
            }
        }
    }
}

fn resize_numeric_storage(storage: &mut NumericStorage, len: usize) {
    macro_rules! resize {
        ($values:expr, $zero:expr) => {{
            $values.resize(len, $zero);
        }};
    }
    match storage {
        NumericStorage::F64(values) => resize!(values, 0.0),
        NumericStorage::F32(values) => resize!(values, 0.0),
        NumericStorage::I8(values) => resize!(values, 0),
        NumericStorage::I16(values) => resize!(values, 0),
        NumericStorage::I32(values) => resize!(values, 0),
        NumericStorage::I64(values) => resize!(values, 0),
        NumericStorage::U8(values) => resize!(values, 0),
        NumericStorage::U16(values) => resize!(values, 0),
        NumericStorage::U32(values) => resize!(values, 0),
        NumericStorage::U64(values) => resize!(values, 0),
    }
}

fn finalize_char(
    size_spec: &SizeSpec,
    count_read: usize,
    mut column_major: Vec<char>,
) -> (Vec<char>, usize, usize) {
    match size_spec {
        SizeSpec::All | SizeSpec::Count(_) => {
            let rows = count_read;
            let cols = if count_read == 0 { 0 } else { 1 };
            let row_major = column_to_row_major(&column_major, rows, cols);
            (row_major, rows, cols)
        }
        SizeSpec::Matrix {
            rows,
            cols: Some(c),
        } => {
            let target = rows.saturating_mul(*c);
            if column_major.len() < target {
                column_major.resize(target, '\0');
            } else if column_major.len() > target {
                column_major.truncate(target);
            }
            let row_major = column_to_row_major(&column_major, *rows, *c);
            (row_major, *rows, *c)
        }
        SizeSpec::Matrix { rows, cols: None } => {
            if *rows == 0 {
                column_major.clear();
                (Vec::new(), 0, 0)
            } else {
                let cols = if count_read == 0 {
                    0
                } else {
                    count_read.div_ceil(*rows)
                };
                let target = rows.saturating_mul(cols);
                if column_major.len() < target {
                    column_major.resize(target, '\0');
                } else if column_major.len() > target {
                    column_major.truncate(target);
                }
                let row_major = column_to_row_major(&column_major, *rows, cols);
                (row_major, *rows, cols)
            }
        }
    }
}

fn column_to_row_major(data: &[char], rows: usize, cols: usize) -> Vec<char> {
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    let mut output = vec!['\0'; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            let src = c * rows + r;
            let dst = r * cols + c;
            if let Some(ch) = data.get(src) {
                output[dst] = *ch;
            }
        }
    }
    output
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::io::filetext::registry;
    use crate::builtins::io::filetext::{fclose, fopen};
    use crate::RuntimeError;
    use runmat_filesystem::File;
    use runmat_time::system_time_now;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::UNIX_EPOCH;

    fn unwrap_error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FreadEval> {
        futures::executor::block_on(evaluate(fid_value, rest))
    }

    fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
        futures::executor::block_on(fopen::evaluate(args))
    }

    fn run_fclose(args: &[Value]) -> BuiltinResult<fclose::FcloseEval> {
        futures::executor::block_on(fclose::evaluate(args))
    }

    fn double_values(tensor: &Tensor) -> &[f64] {
        tensor.as_f64_slice().expect("expected double tensor")
    }

    #[test]
    fn fread_like_preserves_every_exact_integer_class() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let prototypes = [
            IntegerStorage::I8(vec![0]),
            IntegerStorage::I16(vec![0]),
            IntegerStorage::I32(vec![0]),
            IntegerStorage::I64(vec![0]),
            IntegerStorage::U8(vec![0]),
            IntegerStorage::U16(vec![0]),
            IntegerStorage::U32(vec![0]),
            IntegerStorage::U64(vec![0]),
        ];

        for storage in prototypes {
            let expected = storage
                .from_same_class_values(
                    [1.0, 2.5, -3.0]
                        .into_iter()
                        .map(|value| storage.cast_f64_assignment(value))
                        .collect(),
                )
                .expect("expected storage");
            let data = Value::Tensor(Tensor::new(vec![1.0, 2.5, -3.0], vec![3, 1]).unwrap());
            let prototype_tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
            let prototype = Value::Tensor(prototype_tensor);
            let output = adjust_output_for_like(data, &prototype, PrecisionSpec::default())
                .expect("integer like output");
            let Value::Tensor(output) = output else {
                panic!("expected tensor output");
            };
            assert_eq!(output.integer_storage(), Some(&expected));
        }
    }

    #[cfg(feature = "wgpu")]
    fn run_call_builtin(name: &str, args: &[Value]) -> BuiltinResult<Value> {
        crate::call_builtin(name, args)
    }

    fn registry_guard() -> std::sync::MutexGuard<'static, ()> {
        registry::test_guard()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FREAD_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"data = fread(fid)"));
        assert!(labels.contains(&"data = fread(fid, size, precision, skip, machinefmt)"));
        assert!(labels.contains(&"data = fread(fid, precision, machinefmt)"));
        assert!(labels.contains(&"data = fread(fid, ..., \"like\", prototype)"));
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn fread_size_tensor_parser_preserves_exact_integer_storage() {
        let count = (1_u64 << 53) + 1;
        let count_tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![count]), vec![1, 1]).expect("count");
        match parse_size(Some(&Value::Tensor(count_tensor))).expect("size") {
            SizeSpec::Count(value) => assert_eq!(value, usize::try_from(count).unwrap()),
            other => panic!("expected count size, got {other:?}"),
        }

        let matrix_tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![count, 3]), vec![1, 2]).expect("matrix");
        match parse_size(Some(&Value::Tensor(matrix_tensor))).expect("size") {
            SizeSpec::Matrix {
                rows,
                cols: Some(cols),
            } => {
                assert_eq!(rows, usize::try_from(count).unwrap());
                assert_eq!(cols, 3);
            }
            other => panic!("expected matrix size, got {other:?}"),
        }
    }

    #[test]
    fn fread_size_tensor_parser_rejects_negative_integer_storage() {
        let count_tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("count");
        assert!(parse_size(Some(&Value::Tensor(count_tensor))).is_err());

        let matrix_tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![2, -1]), vec![1, 2]).expect("matrix");
        assert!(parse_size(Some(&Value::Tensor(matrix_tensor))).is_err());
    }

    #[test]
    fn fread_scalar_parser_reads_typed_integer_storage_exactly() {
        let scalar = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("scalar");
        assert_eq!(
            value_to_scalar(&Value::Tensor(scalar), "scalar").expect("scalar"),
            7.0
        );
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn fread_scalar_size_and_skip_parse_integer_values_exactly() {
        let exact = (1_u64 << 53) + 1;
        match parse_size(Some(&Value::Int(IntValue::U64(exact)))).expect("size") {
            SizeSpec::Count(value) => assert_eq!(value, exact as usize),
            other => panic!("expected count size, got {other:?}"),
        }

        assert_eq!(
            parse_skip(Some(&Value::Int(IntValue::U64(exact)))).unwrap(),
            exact as usize
        );
        assert!(parse_skip(Some(&Value::Int(IntValue::U64(u64::MAX)))).is_err());
        assert!(parse_size(Some(&Value::Int(IntValue::I8(-1)))).is_err());

        assert!(parse_size(Some(&Value::Num(usize::MAX as f64))).is_err());
        assert!(parse_size(Some(&Value::Num((usize::MAX as f64) + 1.0))).is_err());
        assert!(parse_skip(Some(&Value::Num(i64::MAX as f64))).is_err());
        assert!(parse_skip(Some(&Value::Num((i64::MAX as f64) + 1.0))).is_err());
    }

    #[test]
    fn fread_fid_and_skip_read_typed_integer_storage_exactly() {
        let fid =
            Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("fid tensor");
        assert_eq!(parse_fid(&Value::Tensor(fid)).unwrap(), 7);
        assert_eq!(parse_fid(&Value::Int(IntValue::U16(7))).unwrap(), 7);
        assert!(parse_fid(&Value::Int(IntValue::U64(u64::MAX))).is_err());

        let skip =
            Tensor::new_integer(IntegerStorage::U16(vec![9]), vec![1, 1]).expect("skip tensor");
        assert_eq!(parse_skip(Some(&Value::Tensor(skip))).unwrap(), 9);

        let too_large =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("skip");
        assert!(parse_skip(Some(&Value::Tensor(too_large))).is_err());
    }

    #[test]
    fn fread_typed_scalar_parameters_cover_every_integer_class() {
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
            let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("typed scalar");
            let value = Value::Tensor(tensor);
            assert_eq!(parse_fid(&value).unwrap(), 7);
            assert!(matches!(
                parse_size(Some(&value)).unwrap(),
                SizeSpec::Count(7)
            ));
            assert_eq!(parse_skip(Some(&value)).unwrap(), 7);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_default_reads_uint8_and_returns_double() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_default_double");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[7_u8]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let eval = run_evaluate(&Value::Num(fid as f64), &Vec::new()).expect("fread");
        assert_eq!(eval.count(), 1);
        match eval.data() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert_eq!(double_values(t), &[7.0]);
            }
            other => panic!("unexpected result {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[test]
    fn fread_precision_parser_supports_every_numeric_output_class() {
        let cases = [
            ("uint8=>double", NumericDType::F64),
            ("uint8=>single", NumericDType::F32),
            ("uint8=>int8", NumericDType::I8),
            ("uint8=>int16", NumericDType::I16),
            ("uint8=>int32", NumericDType::I32),
            ("uint8=>int64", NumericDType::I64),
            ("uint8=>uint8", NumericDType::U8),
            ("uint8=>uint16", NumericDType::U16),
            ("uint8=>uint32", NumericDType::U32),
            ("uint8=>uint64", NumericDType::U64),
        ];
        for (precision, expected) in cases {
            let parsed = parse_precision_string(precision).expect("precision");
            assert!(matches!(parsed.output, OutputKind::Numeric(dtype) if dtype == expected));
        }
        assert!(matches!(
            parse_precision_string("*single").unwrap().output,
            OutputKind::Numeric(NumericDType::F32)
        ));
        assert!(matches!(
            parse_precision_string("*uint64").unwrap().output,
            OutputKind::Numeric(NumericDType::U64)
        ));
        let repeated = parse_precision_string("2*uint16=>uint16").unwrap();
        assert_eq!(repeated.repeat, 2);
        assert!(matches!(
            repeated.output,
            OutputKind::Numeric(NumericDType::U16)
        ));
        assert_eq!(parse_precision_string("real*4").unwrap().repeat, 1);
    }

    #[test]
    fn fread_star_uint64_preserves_values_beyond_binary64_exact_range() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_exact_uint64");
        let expected = [9_007_199_254_740_993_u64, u64::MAX];
        let mut file = File::create(&path).expect("create");
        for value in expected {
            file.write_all(&value.to_le_bytes()).expect("write");
        }
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;
        let eval = run_evaluate(&Value::Num(fid as f64), &[Value::from("*uint64")]).expect("fread");
        assert_eq!(eval.count(), 2);
        let Value::Tensor(output) = eval.data() else {
            panic!("expected tensor output");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(expected.to_vec()))
        );

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[test]
    fn fread_star_single_preserves_native_single_storage() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_native_single");
        let expected = [1.25_f32, -3.5_f32];
        let mut file = File::create(&path).expect("create");
        for value in expected {
            file.write_all(&value.to_le_bytes()).expect("write");
        }
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;
        let eval = run_evaluate(&Value::Num(fid as f64), &[Value::from("*single")]).expect("fread");
        let Value::Tensor(output) = eval.data() else {
            panic!("expected tensor output");
        };
        assert_eq!(
            output.clone().into_numeric_storage().expect("storage"),
            NumericStorage::F32(expected.to_vec())
        );

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[test]
    fn fread_logical_conversion_uses_typed_integer_storage() {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX]),
            vec![3, 1],
        )
        .expect("integer tensor");

        let Value::LogicalArray(logical) =
            convert_to_logical_value(Value::Tensor(tensor)).expect("logical conversion")
        else {
            panic!("expected logical array");
        };
        assert_eq!(logical.shape, vec![3, 1]);
        assert_eq!(logical.data, vec![0, 1, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_uint8_vector_with_count() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_uint8_vector");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[1u8, 2, 3, 4, 5]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let args = vec![Value::Num(4.0), Value::from("uint8")];
        let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
        assert_eq!(eval.count(), 4);
        match eval.data() {
            Value::Tensor(t) => {
                assert_eq!(double_values(t), &[1.0, 2.0, 3.0, 4.0]);
                assert_eq!(t.shape, vec![4, 1]);
            }
            other => panic!("unexpected result {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_uint8_matrix_with_padding() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_uint8_matrix");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[1u8, 2, 3, 4, 5]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let size_tensor = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let args = vec![Value::Tensor(size_tensor), Value::from("uint8")];
        let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
        assert_eq!(eval.count(), 5);
        match eval.data() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(double_values(t), &[1.0, 2.0, 3.0, 4.0, 5.0, 0.0]);
            }
            other => panic!("unexpected result {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_char_output() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_char_output");
        let mut file = File::create(&path).expect("create");
        file.write_all(b"abc").expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let eval = run_evaluate(&Value::Num(fid as f64), &[Value::from("*char")]).expect("fread");
        assert_eq!(eval.count(), 3);
        match eval.data() {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 3);
                assert_eq!(ca.cols, 1);
                let collected: String = ca.data.iter().collect();
                assert_eq!(collected, "abc");
            }
            other => panic!("unexpected result {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_like_logical_output() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_like_logical_output");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[0u8, 3, 0, 4]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let prototype = LogicalArray::zeros(vec![2, 1]);
        let args = vec![
            Value::Num(2.0),
            Value::from("uint8"),
            Value::from("like"),
            Value::LogicalArray(prototype),
        ];
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = run_evaluate(&Value::Num(fid as f64), &args)
                .expect_err("MATLAB mode rejects fread like");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:FreadLikeExtension")
            );
        }
        let eval = {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            run_evaluate(&Value::Num(fid as f64), &args).expect("RunMat mode accepts fread like")
        };
        assert_eq!(eval.count(), 2);
        match eval.data() {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 1]);
                assert_eq!(array.data, vec![0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_like_requires_prototype() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_like_requires_prototype");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[1u8, 2, 3, 4]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let args = vec![Value::from("like")];
        let err = unwrap_error_message(run_evaluate(&Value::Num(fid as f64), &args).unwrap_err());
        assert!(err.contains("expected prototype after 'like'"));

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_like_char_requires_precision() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_like_char_requires_precision");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[65u8]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let args = vec![
            Value::Num(1.0),
            Value::from("uint8"),
            Value::from("like"),
            Value::CharArray(CharArray::new_row("A")),
        ];
        let err = unwrap_error_message(run_evaluate(&Value::Num(fid as f64), &args).unwrap_err());
        assert!(err.contains("character prototypes require"));

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_like_gpu_provider_roundtrip() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_like_gpu_provider_roundtrip");
        let mut file = File::create(&path).expect("create");
        file.write_all(&1.5f64.to_le_bytes()).expect("write");
        drop(file);

        test_support::with_test_provider(|provider| {
            let open = run_fopen(&[
                Value::from(path.to_string_lossy().to_string()),
                Value::from("rb"),
            ])
            .expect("fopen");
            let fid = open.as_open().unwrap().fid as i32;

            let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: double_values(&proto),
                shape: &proto.shape,
            };
            let handle = provider.upload(&view).expect("upload prototype");

            let args = vec![
                Value::Num(1.0),
                Value::from("double"),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
            match eval.data() {
                Value::GpuTensor(result) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(result.clone())).expect("gather");
                    assert_eq!(double_values(&gathered), &[1.5]);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }

            run_fclose(&[Value::Num(fid as f64)]).unwrap();
        });

        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fread_wgpu_like_uploads_gpu() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_wgpu_like_uploads_gpu");
        let mut file = File::create(&path).expect("create");
        file.write_all(&2.25f64.to_le_bytes()).expect("write");
        drop(file);

        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let gpu_proto = run_call_builtin("gpuArray", &[Value::Tensor(proto)]).expect("gpuArray");

        let args = vec![
            Value::Num(1.0),
            Value::from("double"),
            Value::from("like"),
            gpu_proto,
        ];
        let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
        match eval.data() {
            Value::GpuTensor(handle) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(handle.clone())).expect("gather");
                assert_eq!(double_values(&gathered), &[2.25]);
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_skip_bytes() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_skip_bytes");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[1u8, 2, 3, 4, 5, 6]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let args = vec![Value::Num(3.0), Value::from("uint8"), Value::Num(1.0)];
        let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
        assert_eq!(eval.count(), 3);
        match eval.data() {
            Value::Tensor(t) => {
                assert_eq!(double_values(t), &[1.0, 3.0, 5.0]);
            }
            other => panic!("unexpected result {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[test]
    fn fread_repeated_precision_skips_after_each_block() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_repeated_skip");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[1_u8, 2, 99, 3, 4, 99]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;
        let args = vec![Value::Num(4.0), Value::from("2*uint8"), Value::Num(1.0)];
        let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
        let Value::Tensor(output) = eval.data() else {
            panic!("expected tensor output");
        };
        assert_eq!(double_values(output), &[1.0, 2.0, 3.0, 4.0]);

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_big_endian_machine_format() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_big_endian");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[0x01, 0x02, 0x03, 0x04]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
            Value::from("ieee-be"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let args = vec![Value::Num(2.0), Value::from("uint16")];
        let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
        assert_eq!(eval.count(), 2);
        match eval.data() {
            Value::Tensor(t) => {
                assert_eq!(double_values(t), &[258.0, 772.0]);
                assert_eq!(t.shape, vec![2, 1]);
            }
            other => panic!("unexpected result {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_invalid_fid_errors() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let err = unwrap_error_message(run_evaluate(&Value::Num(9999.0), &Vec::new()).unwrap_err());
        assert_eq!(err, FREAD_ERROR_INVALID_IDENTIFIER.message);
    }

    fn unique_path(prefix: &str) -> PathBuf {
        let now = system_time_now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards");
        let filename = format!(
            "runmat_{prefix}_{}_{}.tmp",
            now.as_secs(),
            now.subsec_nanos()
        );
        std::env::temp_dir().join(filename)
    }
}
