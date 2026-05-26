//! MATLAB-compatible `fopen` builtin exposing host file streams.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
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
    summary = "Open a file and obtain a MATLAB-compatible file identifier.",
    keywords = "fopen,file,io,permission,encoding",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fopen_type),
    descriptor(crate::builtins::io::filetext::fopen::FOPEN_DESCRIPTOR),
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
            .map_err(|e| fopen_error_with_detail(&FOPEN_ERROR_INTERNAL, e.to_string()))?;

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
}

fn parse_fid(value: &Value) -> BuiltinResult<i32> {
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
        make_cell(values, 0, 0)
            .map_err(|err| fopen_error_with_detail(&FOPEN_ERROR_INTERNAL, err.to_string()))
    } else {
        make_cell(values, len, 1)
            .map_err(|err| fopen_error_with_detail(&FOPEN_ERROR_INTERNAL, err.to_string()))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::io::filetext::registry;
    use runmat_time::system_time_now;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::UNIX_EPOCH;

    fn unique_path(prefix: &str) -> PathBuf {
        let now = system_time_now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards");
        let filename = format!("{}_{}_{}.tmp", prefix, now.as_secs(), now.subsec_nanos());
        std::env::temp_dir().join(filename)
    }

    fn run_evaluate(args: &[Value]) -> BuiltinResult<FopenEval> {
        futures::executor::block_on(evaluate(args))
    }

    fn registry_guard() -> std::sync::MutexGuard<'static, ()> {
        registry::test_guard()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FOPEN_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"fid = fopen(filename)"));
        assert!(labels.contains(&"fid = fopen(filename, permission)"));
        assert!(labels.contains(&"fid = fopen(filename, permission, machinefmt)"));
        assert!(labels.contains(&"fid = fopen(filename, permission, machinefmt, encoding)"));
        assert!(labels.contains(&"[fid, msg] = fopen(filename, ...)"));
        assert!(labels.contains(&"[fid, msg, machinefmt, encoding] = fopen(filename, ...)"));
        assert!(labels.contains(&"filename = fopen(fid)"));
        assert!(labels.contains(&"[filename, permission, machinefmt, encoding] = fopen(fid)"));
        assert!(labels.contains(&"fids = fopen(\"all\")"));
        assert!(labels.contains(&"fids = fopen(\"all\", machinefmt)"));
        assert!(labels.contains(&"[fids, names, machinefmts, encodings] = fopen(\"all\")"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_read_existing_file_returns_fid() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fopen_read");
        test_support::fs::write(&path, "hello world").unwrap();

        let args = vec![Value::from(path.to_string_lossy().to_string())];
        let eval = run_evaluate(&args).expect("fopen");
        let open = eval.as_open().expect("expected open result");
        assert!(open.fid >= 3.0);
        assert!(open.message.is_empty());
        assert_eq!(open.machinefmt, "native");
        assert_eq!(open.encoding, "UTF-8");

        let _ = registry::close(open.fid as i32);
        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_missing_file_returns_error() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fopen_missing");
        let args = vec![Value::from(path.to_string_lossy().to_string())];
        let eval = run_evaluate(&args).expect("fopen");
        let open = eval.as_open().expect("open output");
        assert_eq!(open.fid, -1.0);
        assert!(!open.message.is_empty());
        assert!(open.machinefmt.is_empty());
        assert!(open.encoding.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_query_returns_metadata() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fopen_query");
        {
            let mut file = runmat_filesystem::File::create(&path).unwrap();
            writeln!(&mut file, "data").unwrap();
        }
        let args = vec![Value::from(path.to_string_lossy().to_string())];
        let eval = run_evaluate(&args).expect("fopen");
        let open = eval.as_open().expect("open result");
        let fid = open.fid;
        assert!(fid >= 3.0);

        let query_eval = run_evaluate(&[Value::from(fid)]).expect("fopen query");
        let query = query_eval.as_query().expect("query result");
        assert!(query.filename.contains("fopen_query"));
        assert_eq!(query.permission, "r");
        assert_eq!(query.machinefmt, "native");

        let _ = registry::close(fid as i32);
        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_all_lists_handles() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fopen_all");
        test_support::fs::write(&path, "abc").unwrap();
        let eval_open =
            run_evaluate(&[Value::from(path.to_string_lossy().to_string())]).expect("fopen");
        let fid = eval_open.as_open().unwrap().fid;

        let list_eval = run_evaluate(&[Value::from("all")]).expect("fopen all");
        let list = list_eval.as_list().expect("list result");
        assert!(!list.handles.data.is_empty());
        assert!(list
            .handles
            .data
            .iter()
            .any(|v| (*v - fid).abs() < f64::EPSILON));

        if let Value::Cell(names) = &list.names {
            assert_eq!(names.data.len(), list.handles.data.len());
            assert_eq!(names.rows, list.handles.data.len());
            assert_eq!(names.cols, 1);
        } else {
            panic!("expected cell array for names");
        }
        if let Value::Cell(machinefmts) = &list.machinefmts {
            assert_eq!(machinefmts.rows, list.handles.data.len());
            assert_eq!(machinefmts.cols, 1);
        } else {
            panic!("expected cell array for machine formats");
        }
        if let Value::Cell(encodings) = &list.encodings {
            assert_eq!(encodings.rows, list.handles.data.len());
            assert_eq!(encodings.cols, 1);
        } else {
            panic!("expected cell array for encodings");
        }

        let _ = registry::close(fid as i32);
        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_all_machinefmt_filters_entries() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let native_path = unique_path("fopen_native");
        let be_path = unique_path("fopen_ieee_be");
        test_support::fs::write(&native_path, "native").unwrap();
        test_support::fs::write(&be_path, "be").unwrap();

        let native_eval = run_evaluate(&[Value::from(native_path.to_string_lossy().to_string())])
            .expect("native");
        let native_fid = native_eval.as_open().unwrap().fid;

        let be_eval = run_evaluate(&[
            Value::from(be_path.to_string_lossy().to_string()),
            Value::from("r"),
            Value::from("ieee-be"),
        ])
        .expect("ieee-be");
        let be_fid = be_eval.as_open().unwrap().fid;

        let list_eval =
            run_evaluate(&[Value::from("all"), Value::from("ieee-be")]).expect("fopen all filter");
        let list = list_eval.as_list().expect("list result");
        assert_eq!(list.handles.data.len(), 1);
        assert!((list.handles.data[0] - be_fid).abs() < f64::EPSILON);

        let _ = registry::close(native_fid as i32);
        let _ = registry::close(be_fid as i32);
        test_support::fs::remove_file(&native_path).unwrap();
        test_support::fs::remove_file(&be_path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_binary_default_encoding_binary() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fopen_binary");
        {
            let _ = runmat_filesystem::File::create(&path).unwrap();
        }
        let eval = run_evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("wb"),
        ])
        .expect("fopen");
        let open = eval.as_open().unwrap();
        assert_eq!(open.encoding, "binary");
        let _ = registry::close(open.fid as i32);
        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_encoding_argument_is_preserved() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fopen_encoding");
        let eval = run_evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w"),
            Value::from("ieee-be"),
            Value::from("latin1"),
        ])
        .expect("fopen");
        let open = eval.as_open().unwrap();
        assert_eq!(open.machinefmt, "ieee-be");
        assert_eq!(open.encoding, "latin1");
        let _ = registry::close(open.fid as i32);
        if path.exists() {
            test_support::fs::remove_file(&path).unwrap();
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_permission_canonicalizes_plus_binary_order() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fopen_perm_order");
        test_support::fs::write(&path, "seed").unwrap();
        let eval = run_evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("r+b"),
        ])
        .expect("fopen");
        let open = eval.as_open().unwrap();
        assert!(open.fid >= 3.0);
        assert_eq!(open.encoding, "binary");
        let query = run_evaluate(&[Value::Num(open.fid)]).expect("query");
        let info = query.as_query().unwrap();
        assert_eq!(info.permission, "rb+");
        let _ = registry::close(open.fid as i32);
        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_machinefmt_preserves_suffix() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fopen_machinefmt_suffix");
        let eval = run_evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w"),
            Value::from("ieee-be.l64"),
        ])
        .expect("fopen");
        let open = eval.as_open().unwrap();
        assert_eq!(open.machinefmt, "ieee-be.l64");
        let query = run_evaluate(&[Value::Num(open.fid)]).expect("query");
        let info = query.as_query().unwrap();
        assert_eq!(info.machinefmt, "ieee-be.l64");
        let _ = registry::close(open.fid as i32);
        if path.exists() {
            test_support::fs::remove_file(&path).unwrap();
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_machinefmt_pc_alias_maps_to_ieee_le() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fopen_machinefmt_pc");
        let eval = run_evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w"),
            Value::from("pc"),
        ])
        .expect("fopen");
        let open = eval.as_open().unwrap();
        assert_eq!(open.machinefmt, "ieee-le");
        let query = run_evaluate(&[Value::Num(open.fid)]).expect("query");
        let info = query.as_query().unwrap();
        assert_eq!(info.machinefmt, "ieee-le");
        let _ = registry::close(open.fid as i32);
        if path.exists() {
            test_support::fs::remove_file(&path).unwrap();
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_outputs_vector_padding() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fopen_outputs");
        test_support::fs::write(&path, "check").unwrap();
        let eval = run_evaluate(&[Value::from(path.to_string_lossy().to_string())]).expect("fopen");
        let outputs = eval.outputs();
        assert_eq!(outputs.len(), 4);
        assert!(matches!(outputs[0], Value::Num(_)));
        assert!(matches!(outputs[1], Value::CharArray(_)));
        let _ = registry::close(eval.as_open().unwrap().fid as i32);
        test_support::fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_invalid_fid_returns_empty() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let eval = run_evaluate(&[Value::from(9999.0)]).expect("fopen");
        let query = eval.as_query().expect("query result");
        assert!(query.filename.is_empty());
        assert!(query.permission.is_empty());
    }
}
