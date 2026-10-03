//! MATLAB-compatible `fopen` builtin exposing host file streams.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::io::filetext::{
    helpers::{char_array_value, extract_scalar_string, normalize_encoding_label},
    registry::{self, FileInfo, RegisteredFile},
};
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};
use runmat_filesystem::OpenOptions;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fopen")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fopen",
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
        "Host-only file I/O. Inputs gathered from GPU when necessary; outputs remain on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fopen")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fopen",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O is not eligible for fusion; metadata registered for completeness only.",
};

const BUILTIN_NAME: &str = "fopen";

const FOPEN_INTEGER_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fopen-integer-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "integer-class fopen query identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FopenIntegerIdExtension"),
};
const FOPEN_SINGLE_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fopen-single-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "single-precision fopen query identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FopenSingleIdExtension"),
};
const FOPEN_RESIDENT_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fopen-resident-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "provider-resident fopen query identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FopenResidentIdExtension"),
};
const FOPEN_LEGACY_ALL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fopen-legacy-all",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "legacy fopen no-argument and 'all' listing forms are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FopenLegacyAllExtension"),
};
pub const FOPEN_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FOPEN_INTEGER_ID_EXTENSION,
    FOPEN_SINGLE_ID_EXTENSION,
    FOPEN_RESIDENT_ID_EXTENSION,
    FOPEN_LEGACY_ALL_EXTENSION,
];

const FOPEN_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents double query identifiers; typed integer identifiers are checked exactly as one gated RunMat extension.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "filename = fopen(integer_fileID)",
        inputs: &FOPEN_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "A successful query returns host character metadata; integer identifiers are validated exactly.",
    }];

const FOPEN_OUTPUT_FID: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier on success, or -1 on failure.",
}];
const FOPEN_OUTPUT_FID_MSG: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier on success, or -1 on failure.",
    },
    BuiltinParamDescriptor {
        name: "msg",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"\""),
        description: "Open failure message string (empty on success).",
    },
];
const FOPEN_OUTPUT_FID_MSG_MACHINEFMT_ENCODING: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier on success, or -1 on failure.",
    },
    BuiltinParamDescriptor {
        name: "msg",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"\""),
        description: "Open failure message string (empty on success).",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Resolved machine-format label for the opened stream.",
    },
    BuiltinParamDescriptor {
        name: "encoding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"UTF-8\""),
        description: "Resolved encoding label for the opened stream.",
    },
];
const FOPEN_OUTPUT_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filename associated with queried file identifier.",
}];
const FOPEN_OUTPUT_FILENAME_PERMISSION_MACHINEFMT_ENCODING: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filename associated with queried file identifier.",
    },
    BuiltinParamDescriptor {
        name: "permission",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"\""),
        description: "Canonical fopen permission string.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"\""),
        description: "Machine-format label of queried stream.",
    },
    BuiltinParamDescriptor {
        name: "encoding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"\""),
        description: "Encoding label of queried stream.",
    },
];
const FOPEN_OUTPUT_HANDLES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fids",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Column vector of open user file identifiers.",
}];
const FOPEN_OUTPUT_HANDLES_NAMES_MACHINEFMTS_ENCODINGS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fids",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column vector of open user file identifiers.",
    },
    BuiltinParamDescriptor {
        name: "names",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("{}"),
        description: "Cell column of display names.",
    },
    BuiltinParamDescriptor {
        name: "machinefmts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("{}"),
        description: "Cell column of machine-format labels.",
    },
    BuiltinParamDescriptor {
        name: "encodings",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("{}"),
        description: "Cell column of encoding labels.",
    },
];

const FOPEN_INPUTS_OPEN_FILE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Path to file to open.",
}];
const FOPEN_INPUTS_OPEN_FILE_PERMISSION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to file to open.",
    },
    BuiltinParamDescriptor {
        name: "permission",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"r\""),
        description: "Permission string such as 'r', 'w', 'a', optionally with '+'/'b'/'t'.",
    },
];
const FOPEN_INPUTS_OPEN_FILE_PERMISSION_MACHINEFMT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to file to open.",
    },
    BuiltinParamDescriptor {
        name: "permission",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"r\""),
        description: "Permission string such as 'r', 'w', 'a', optionally with '+'/'b'/'t'.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine-format label.",
    },
];
const FOPEN_INPUTS_OPEN_FILE_PERMISSION_MACHINEFMT_ENCODING: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to file to open.",
    },
    BuiltinParamDescriptor {
        name: "permission",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"r\""),
        description: "Permission string such as 'r', 'w', 'a', optionally with '+'/'b'/'t'.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine-format label.",
    },
    BuiltinParamDescriptor {
        name: "encoding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"UTF-8\""),
        description: "Encoding label.",
    },
];
const FOPEN_INPUTS_QUERY_FID: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric file identifier.",
}];
const FOPEN_INPUTS_LIST_ALL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "selector",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"all\""),
    description: "Literal selector 'all'.",
}];
const FOPEN_INPUTS_LIST_ALL_MACHINEFMT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "selector",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"all\""),
        description: "Literal selector 'all'.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Machine-format filter for listed handles.",
    },
];

const FOPEN_SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "fid = fopen(filename)",
        inputs: &FOPEN_INPUTS_OPEN_FILE,
        outputs: &FOPEN_OUTPUT_FID,
    },
    BuiltinSignatureDescriptor {
        label: "fid = fopen(filename, permission)",
        inputs: &FOPEN_INPUTS_OPEN_FILE_PERMISSION,
        outputs: &FOPEN_OUTPUT_FID,
    },
    BuiltinSignatureDescriptor {
        label: "fid = fopen(filename, permission, machinefmt)",
        inputs: &FOPEN_INPUTS_OPEN_FILE_PERMISSION_MACHINEFMT,
        outputs: &FOPEN_OUTPUT_FID,
    },
    BuiltinSignatureDescriptor {
        label: "fid = fopen(filename, permission, machinefmt, encoding)",
        inputs: &FOPEN_INPUTS_OPEN_FILE_PERMISSION_MACHINEFMT_ENCODING,
        outputs: &FOPEN_OUTPUT_FID,
    },
    BuiltinSignatureDescriptor {
        label: "[fid, msg] = fopen(filename, ...)",
        inputs: &FOPEN_INPUTS_OPEN_FILE_PERMISSION_MACHINEFMT_ENCODING,
        outputs: &FOPEN_OUTPUT_FID_MSG,
    },
    BuiltinSignatureDescriptor {
        label: "[fid, msg, machinefmt, encoding] = fopen(filename, ...)",
        inputs: &FOPEN_INPUTS_OPEN_FILE_PERMISSION_MACHINEFMT_ENCODING,
        outputs: &FOPEN_OUTPUT_FID_MSG_MACHINEFMT_ENCODING,
    },
    BuiltinSignatureDescriptor {
        label: "filename = fopen(fid)",
        inputs: &FOPEN_INPUTS_QUERY_FID,
        outputs: &FOPEN_OUTPUT_FILENAME,
    },
    BuiltinSignatureDescriptor {
        label: "[filename, permission, machinefmt, encoding] = fopen(fid)",
        inputs: &FOPEN_INPUTS_QUERY_FID,
        outputs: &FOPEN_OUTPUT_FILENAME_PERMISSION_MACHINEFMT_ENCODING,
    },
    BuiltinSignatureDescriptor {
        label: "fids = fopen(\"all\")",
        inputs: &FOPEN_INPUTS_LIST_ALL,
        outputs: &FOPEN_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "fids = fopen(\"all\", machinefmt)",
        inputs: &FOPEN_INPUTS_LIST_ALL_MACHINEFMT,
        outputs: &FOPEN_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "[fids, names, machinefmts, encodings] = fopen(\"all\")",
        inputs: &FOPEN_INPUTS_LIST_ALL,
        outputs: &FOPEN_OUTPUT_HANDLES_NAMES_MACHINEFMTS_ENCODINGS,
    },
    BuiltinSignatureDescriptor {
        label: "[fids, names, machinefmts, encodings] = fopen(\"all\", machinefmt)",
        inputs: &FOPEN_INPUTS_LIST_ALL_MACHINEFMT,
        outputs: &FOPEN_OUTPUT_HANDLES_NAMES_MACHINEFMTS_ENCODINGS,
    },
];

const FOPEN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FOPEN.INVALID_INPUT",
    identifier: Some("RunMat:fopen:InvalidInput"),
    when: "Argument count or argument shape/type is invalid.",
    message: "fopen: invalid input arguments",
};
const FOPEN_ERROR_INVALID_PERMISSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FOPEN.INVALID_PERMISSION",
    identifier: Some("RunMat:fopen:InvalidPermission"),
    when: "Permission string is invalid or unsupported.",
    message: "fopen: invalid permission string",
};
const FOPEN_ERROR_INVALID_MACHINEFMT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FOPEN.INVALID_MACHINEFMT",
    identifier: Some("RunMat:fopen:InvalidMachineFormat"),
    when: "Machine-format argument is invalid or unsupported.",
    message: "fopen: invalid machine format",
};
const FOPEN_ERROR_INVALID_ENCODING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FOPEN.INVALID_ENCODING",
    identifier: Some("RunMat:fopen:InvalidEncoding"),
    when: "Encoding argument is invalid.",
    message: "fopen: invalid encoding",
};
const FOPEN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FOPEN.INTERNAL",
    identifier: None,
    when: "Internal runtime conversion/control-flow operation fails.",
    message: "fopen: internal error",
};
const FOPEN_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FOPEN_ERROR_INVALID_INPUT,
    FOPEN_ERROR_INVALID_PERMISSION,
    FOPEN_ERROR_INVALID_MACHINEFMT,
    FOPEN_ERROR_INVALID_ENCODING,
    FOPEN_ERROR_INTERNAL,
];
pub const FOPEN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FOPEN_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FOPEN_ERRORS,
};

fn fopen_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fopen_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn fopen_error_with_message(
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
    if let Some(identifier) = FOPEN_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "fopen",
    category = "io/filetext",
    summary = "Open files and return file identifiers.",
    keywords = "fopen,file,io,permission,encoding",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fopen_type),
    descriptor(crate::builtins::io::filetext::fopen::FOPEN_DESCRIPTOR),
    extensions(crate::builtins::io::filetext::fopen::FOPEN_EXTENSIONS),
    integer_capabilities(crate::builtins::io::filetext::fopen::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::io::filetext::fopen"
)]
async fn fopen_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&args).await?;
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

#[derive(Clone)]
pub struct FopenEval {
    kind: FopenEvalKind,
}

#[derive(Clone)]
enum FopenEvalKind {
    Open(OpenOutputs),
    Query(QueryOutputs),
    List(Box<ListOutputs>),
}

#[derive(Clone)]
pub(crate) struct OpenOutputs {
    pub fid: f64,
    pub message: String,
    pub machinefmt: String,
    pub encoding: String,
}

#[derive(Clone)]
pub(crate) struct QueryOutputs {
    pub filename: String,
    pub permission: String,
    pub machinefmt: String,
    pub encoding: String,
}

#[derive(Clone)]
pub(crate) struct ListOutputs {
    pub handles: Tensor,
    pub names: Value,
    pub machinefmts: Value,
    pub encodings: Value,
}

impl FopenEval {
    fn open(outputs: OpenOutputs) -> Self {
        Self {
            kind: FopenEvalKind::Open(outputs),
        }
    }

    fn query(outputs: QueryOutputs) -> Self {
        Self {
            kind: FopenEvalKind::Query(outputs),
        }
    }

    fn list(outputs: ListOutputs) -> Self {
        Self {
            kind: FopenEvalKind::List(Box::new(outputs)),
        }
    }

    pub fn first_output(&self) -> Value {
        match &self.kind {
            FopenEvalKind::Open(outputs) => Value::Num(outputs.fid),
            FopenEvalKind::Query(outputs) => char_array_value(&outputs.filename),
            FopenEvalKind::List(outputs) => Value::Tensor(outputs.handles.clone()),
        }
    }

    pub fn outputs(&self) -> Vec<Value> {
        match &self.kind {
            FopenEvalKind::Open(outputs) => outputs.outputs(),
            FopenEvalKind::Query(outputs) => outputs.outputs(),
            FopenEvalKind::List(outputs) => outputs.outputs(),
        }
    }

    #[cfg(test)]
    pub(crate) fn as_open(&self) -> Option<&OpenOutputs> {
        match &self.kind {
            FopenEvalKind::Open(outputs) => Some(outputs),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn as_query(&self) -> Option<&QueryOutputs> {
        match &self.kind {
            FopenEvalKind::Query(outputs) => Some(outputs),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn as_list(&self) -> Option<&ListOutputs> {
        match &self.kind {
            FopenEvalKind::List(outputs) => Some(outputs),
            _ => None,
        }
    }
}

impl OpenOutputs {
    fn success(fid: i32, machinefmt: String, encoding: String) -> Self {
        Self {
            fid: fid as f64,
            message: String::new(),
            machinefmt,
            encoding,
        }
    }

    fn failure(message: String) -> Self {
        Self {
            fid: -1.0,
            message,
            machinefmt: String::new(),
            encoding: String::new(),
        }
    }

    fn outputs(&self) -> Vec<Value> {
        vec![
            Value::Num(self.fid),
            char_array_value(&self.message),
            char_array_value(&self.machinefmt),
            char_array_value(&self.encoding),
        ]
    }
}

impl QueryOutputs {
    fn from_info(info: FileInfo) -> Self {
        let filename = match info.path {
            Some(path) => path.to_string_lossy().to_string(),
            None => info.name,
        };
        Self {
            filename,
            permission: info.permission,
            machinefmt: info.machinefmt,
            encoding: info.encoding,
        }
    }

    fn not_found() -> Self {
        Self {
            filename: String::new(),
            permission: String::new(),
            machinefmt: String::new(),
            encoding: String::new(),
        }
    }

    fn outputs(&self) -> Vec<Value> {
        vec![
            char_array_value(&self.filename),
            char_array_value(&self.permission),
            char_array_value(&self.machinefmt),
            char_array_value(&self.encoding),
        ]
    }
}

impl ListOutputs {
    fn from_infos(infos: Vec<FileInfo>) -> BuiltinResult<Self> {
        let mut handles: Vec<f64> = infos.iter().map(|info| info.id as f64).collect();
        let rows = handles.len();
        if rows == 0 {
            handles = Vec::new();
        }
        let shape = if rows == 0 { vec![0, 1] } else { vec![rows, 1] };
        let tensor = Tensor::new(handles, shape)
            .map_err(|e| fopen_error_with_detail(&FOPEN_ERROR_INTERNAL, &e))?;

        let mut name_values = Vec::with_capacity(infos.len());
        let mut machine_values = Vec::with_capacity(infos.len());
        let mut encoding_values = Vec::with_capacity(infos.len());
        for info in &infos {
            let display = match &info.path {
                Some(path) => path.to_string_lossy().to_string(),
                None => info.name.clone(),
            };
            name_values.push(char_array_value(&display));
            machine_values.push(char_array_value(&info.machinefmt));
            encoding_values.push(char_array_value(&info.encoding));
        }

        let names = make_cell_column(name_values)?;
        let machinefmts = make_cell_column(machine_values)?;
        let encodings = make_cell_column(encoding_values)?;

        Ok(Self {
            handles: tensor,
            names,
            machinefmts,
            encodings,
        })
    }

    fn outputs(&self) -> Vec<Value> {
        vec![
            Value::Tensor(self.handles.clone()),
            self.names.clone(),
            self.machinefmts.clone(),
            self.encodings.clone(),
        ]
    }
}

#[derive(Clone)]
struct Permission {
    canonical: String,
    read: bool,
    write: bool,
    append: bool,
    create: bool,
    truncate: bool,
    binary: bool,
}

impl Permission {
    fn parse(value: Option<&Value>) -> BuiltinResult<Self> {
        let raw = match value {
            Some(v) => {
                let text = scalar_string(
                    v,
                    "expected permission as a string scalar or character vector",
                    &FOPEN_ERROR_INVALID_PERMISSION,
                )?;
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    return Err(fopen_error_with_detail(
                        &FOPEN_ERROR_INVALID_PERMISSION,
                        "permission string must not be empty",
                    ));
                }
                trimmed.to_string()
            }
            None => "r".to_string(),
        };

        let mut chars = raw.chars();
        let base = chars
            .next()
            .ok_or_else(|| {
                fopen_error_with_detail(
                    &FOPEN_ERROR_INVALID_PERMISSION,
                    "permission string must not be empty",
                )
            })?
            .to_ascii_lowercase();

        let mut read = false;
        let mut write = false;
        let mut append = false;
        let mut create = false;
        let mut truncate = false;

        match base {
            'r' => {
                read = true;
            }
            'w' => {
                write = true;
                create = true;
                truncate = true;
            }
            'a' => {
                write = true;
                create = true;
                append = true;
            }
            _ => {
                return Err(fopen_error_with_detail(
                    &FOPEN_ERROR_INVALID_PERMISSION,
                    format!("unsupported permission prefix '{base}'"),
                ));
            }
        }

        let mut plus = false;
        let mut binary = false;
        let mut explicit_text = false;

        for c in chars {
            match c {
                '+' => {
                    if plus {
                        return Err(fopen_error_with_detail(
                            &FOPEN_ERROR_INVALID_PERMISSION,
                            "duplicate '+' modifier in permission string",
                        ));
                    }
                    plus = true;
                    read = true;
                    write = true;
                }
                'b' | 'B' => {
                    if binary {
                        return Err(fopen_error_with_detail(
                            &FOPEN_ERROR_INVALID_PERMISSION,
                            "duplicate 'b' modifier in permission string",
                        ));
                    }
                    binary = true;
                }
                't' | 'T' => {
                    if explicit_text {
                        return Err(fopen_error_with_detail(
                            &FOPEN_ERROR_INVALID_PERMISSION,
                            "duplicate 't' modifier in permission string",
                        ));
                    }
                    explicit_text = true;
                }
                other => {
                    return Err(fopen_error_with_detail(
                        &FOPEN_ERROR_INVALID_PERMISSION,
                        format!("unrecognised permission modifier '{other}'"),
                    ));
                }
            }
        }

        if binary && explicit_text {
            return Err(fopen_error_with_detail(
                &FOPEN_ERROR_INVALID_PERMISSION,
                "permission modifiers 'b' and 't' are mutually exclusive",
            ));
        }

        let mut canonical = String::new();
        canonical.push(base);
        if binary {
            canonical.push('b');
        } else if explicit_text {
            canonical.push('t');
        }
        if plus {
            canonical.push('+');
        }

        Ok(Self {
            canonical,
            read,
            write,
            append,
            create,
            truncate,
            binary,
        })
    }
}

pub async fn evaluate(args: &[Value]) -> BuiltinResult<FopenEval> {
    if args.is_empty()
        || args
            .first()
            .is_some_and(|value| matches_keyword(value, "all"))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FOPEN_LEGACY_ALL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if let Some(first) = args.first() {
        preflight_query_fid(first)?;
    }
    let gathered = gather_args(args).await?;
    if gathered.is_empty() {
        return handle_all(&[]);
    }
    let first = &gathered[0];
    if matches_keyword(first, "all") {
        return handle_all(&gathered[1..]);
    }
    if is_numeric_value(first) {
        return handle_query(first, &gathered[1..]);
    }
    handle_open(first, &gathered[1..]).await
}

async fn handle_open(path_value: &Value, rest: &[Value]) -> BuiltinResult<FopenEval> {
    if rest.len() > 3 {
        return Err(fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "too many input arguments",
        ));
    }

    let path = value_to_path(path_value)?;
    let mut args = rest.iter();
    let permission = Permission::parse(args.next())?;
    let machinefmt = parse_machinefmt(args.next())?;
    let encoding = parse_encoding(args.next(), &permission)?;

    let mut options = OpenOptions::new();
    options.read(permission.read);
    options.write(permission.write);
    options.create(permission.create);
    options.append(permission.append);
    options.truncate(permission.truncate);

    match options.open_async(&path).await {
        Ok(file) => {
            let handle = Arc::new(StdMutex::new(Some(file)));
            let registered = RegisteredFile {
                path: path.clone(),
                permission: permission.canonical.clone(),
                machinefmt: machinefmt.clone(),
                encoding: encoding.clone(),
                handle,
            };
            let fid = registry::register_file(registered);
            Ok(FopenEval::open(OpenOutputs::success(
                fid, machinefmt, encoding,
            )))
        }
        Err(err) => Ok(FopenEval::open(OpenOutputs::failure(err.to_string()))),
    }
}

fn handle_query(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FopenEval> {
    if !rest.is_empty() {
        return Err(fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "too many input arguments",
        ));
    }
    let fid = parse_fid(fid_value)?;
    let outputs = match registry::info_for(fid) {
        Some(info) => QueryOutputs::from_info(info),
        None => QueryOutputs::not_found(),
    };
    Ok(FopenEval::query(outputs))
}

fn handle_all(rest: &[Value]) -> BuiltinResult<FopenEval> {
    if rest.len() > 1 {
        return Err(fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "too many input arguments",
        ));
    }
    let machinefmt_filter = if let Some(value) = rest.first() {
        Some(parse_machinefmt(Some(value))?)
    } else {
        None
    };
    let mut infos = registry::list_infos();
    if let Some(filter) = &machinefmt_filter {
        infos.retain(|info| info.machinefmt.eq_ignore_ascii_case(filter));
    }
    let outputs = ListOutputs::from_infos(infos)?;
    Ok(FopenEval::list(outputs))
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

fn is_numeric_value(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.len() == 1)
}

fn preflight_query_fid(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Int(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FOPEN_INTEGER_ID_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FOPEN_INTEGER_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == runmat_value::NumericDType::F32 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FOPEN_SINGLE_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FOPEN_RESIDENT_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FOPEN_INTEGER_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FOPEN_SINGLE_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn parse_fid(value: &Value) -> BuiltinResult<i32> {
    if let Value::Int(value) = value {
        return value.try_to_i32().ok_or_else(|| {
            fopen_error_with_detail(
                &FOPEN_ERROR_INVALID_INPUT,
                "file identifier is out of range",
            )
        });
    }
    if let Value::Tensor(tensor) = value {
        if tensor::is_scalar_tensor(tensor) {
            if let Some(integer) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                return integer.try_to_i32().ok_or_else(|| {
                    fopen_error_with_detail(
                        &FOPEN_ERROR_INVALID_INPUT,
                        "file identifier is out of range",
                    )
                });
            }
        }
    }
    let num: f64 = value.try_into().map_err(|_| {
        fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "file identifier must be numeric",
        )
    })?;
    if !num.is_finite() {
        return Err(fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "file identifier must be finite",
        ));
    }
    let rounded = num.round();
    if (rounded - num).abs() > f64::EPSILON {
        return Err(fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "file identifier must be an integer",
        ));
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err(fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "file identifier is out of range",
        ));
    }
    Ok(rounded as i32)
}

fn value_to_path(value: &Value) -> BuiltinResult<PathBuf> {
    let raw = scalar_string(
        value,
        "expected filename as a string scalar or character vector",
        &FOPEN_ERROR_INVALID_INPUT,
    )?;
    normalize_path(&raw)
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "filename must not be empty",
        ));
    }
    Ok(Path::new(raw).to_path_buf())
}

fn parse_machinefmt(value: Option<&Value>) -> BuiltinResult<String> {
    match value {
        None => Ok("native".to_string()),
        Some(v) => {
            let text = scalar_string(
                v,
                "expected machine format as a string scalar or character vector",
                &FOPEN_ERROR_INVALID_MACHINEFMT,
            )?;
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return Err(fopen_error_with_detail(
                    &FOPEN_ERROR_INVALID_MACHINEFMT,
                    "machine format must not be empty",
                ));
            }
            let lower = trimmed.to_ascii_lowercase();
            let collapsed: String = lower
                .chars()
                .filter(|c| !matches!(c, '-' | '_' | ' '))
                .collect();
            if matches!(collapsed.as_str(), "native" | "n" | "system" | "default") {
                return Ok("native".to_string());
            }
            if matches!(collapsed.as_str(), "l" | "le" | "littleendian" | "pc") {
                return Ok("ieee-le".to_string());
            }
            if matches!(collapsed.as_str(), "b" | "be" | "bigendian" | "mac") {
                return Ok("ieee-be".to_string());
            }
            if matches!(collapsed.as_str(), "vaxd" | "vaxg" | "cray") {
                return Ok(collapsed);
            }
            if let Some(suffix) = lower.strip_prefix("ieee-le") {
                return Ok(format!("ieee-le{suffix}"));
            }
            if let Some(suffix) = lower.strip_prefix("ieee-be") {
                return Ok(format!("ieee-be{suffix}"));
            }
            Err(fopen_error_with_detail(
                &FOPEN_ERROR_INVALID_MACHINEFMT,
                format!("unsupported machine format '{trimmed}'"),
            ))
        }
    }
}

fn parse_encoding(value: Option<&Value>, permission: &Permission) -> BuiltinResult<String> {
    match value {
        None => {
            if permission.binary {
                Ok("binary".to_string())
            } else {
                Ok("UTF-8".to_string())
            }
        }
        Some(v) => {
            let text = scalar_string(
                v,
                "expected encoding as a string scalar or character vector",
                &FOPEN_ERROR_INVALID_ENCODING,
            )?;
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return Err(fopen_error_with_detail(
                    &FOPEN_ERROR_INVALID_ENCODING,
                    "encoding name must not be empty",
                ));
            }
            Ok(normalize_encoding_label(trimmed))
        }
    }
}

fn scalar_string(
    value: &Value,
    err_detail: &str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(fopen_error_with_detail(error, err_detail)),
    }
}

fn make_cell_column(values: Vec<Value>) -> BuiltinResult<Value> {
    let len = values.len();
    if len == 0 {
        make_cell(values, 0, 0).map_err(|err| fopen_error_with_detail(&FOPEN_ERROR_INTERNAL, &err))
    } else {
        make_cell(values, len, 1)
            .map_err(|err| fopen_error_with_detail(&FOPEN_ERROR_INTERNAL, &err))
    }
}

#[cfg(test)]
pub(crate) mod tests;
