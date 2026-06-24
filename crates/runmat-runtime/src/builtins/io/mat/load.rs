//! MATLAB-compatible `load` builtin for RunMat.

use std::collections::HashMap;
use std::io::{BufReader, Cursor, Read};
use std::path::{Path, PathBuf};

use flate2::read::ZlibDecoder;
use regex::Regex;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, IntValue, LogicalArray, NumericDType, SparseTensor, StringArray,
    StructValue, Tensor, Value,
};
use runmat_filesystem::File;
use runmat_macros::runtime_builtin;

use super::format::{
    MatArray, MatClass, MatData, FLAG_COMPLEX, FLAG_LOGICAL, MAT_HEADER_LEN, MI_COMPRESSED,
    MI_DOUBLE, MI_INT16, MI_INT32, MI_INT64, MI_INT8, MI_MATRIX, MI_SINGLE, MI_UINT16, MI_UINT32,
    MI_UINT64, MI_UINT8, MI_UTF16, MI_UTF32, MI_UTF8,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};

const LOAD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Struct containing the loaded variables.",
}];
const LOAD_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const LOAD_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"matlab.mat\""),
    description: "MAT-file path.",
}];
const LOAD_INPUTS_FILENAME_VARS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"matlab.mat\""),
        description: "MAT-file path.",
    },
    BuiltinParamDescriptor {
        name: "varName",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Variable names to load.",
    },
];
const LOAD_INPUTS_FILENAME_REGEXP: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"matlab.mat\""),
        description: "MAT-file path.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"-regexp\""),
        description: "Regular-expression selection option.",
    },
    BuiltinParamDescriptor {
        name: "pattern",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Regex patterns matched against variable names.",
    },
];
const LOAD_INPUTS_OPTIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Compatibility options such as '-mat' and '-regexp'.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option arguments and variable selectors.",
    },
];
const LOAD_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "S = load()",
        inputs: &LOAD_INPUTS_NONE,
        outputs: &LOAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = load(filename)",
        inputs: &LOAD_INPUTS_FILENAME,
        outputs: &LOAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = load(filename, varName1, varName2, ...)",
        inputs: &LOAD_INPUTS_FILENAME_VARS,
        outputs: &LOAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = load(filename, \"-regexp\", pattern1, ...)",
        inputs: &LOAD_INPUTS_FILENAME_REGEXP,
        outputs: &LOAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = load(option, value, ...)",
        inputs: &LOAD_INPUTS_OPTIONS,
        outputs: &LOAD_OUTPUT,
    },
];
const LOAD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOAD.INVALID_ARGUMENT",
    identifier: Some("RunMat:load:InvalidArgument"),
    when: "Arguments do not match a supported load invocation form.",
    message: "load: invalid argument",
};
const LOAD_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOAD.INVALID_OPTION",
    identifier: Some("RunMat:load:InvalidOption"),
    when: "An option token or option argument is invalid.",
    message: "load: invalid option",
};
const LOAD_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOAD.FILENAME",
    identifier: Some("RunMat:load:Filename"),
    when: "Filename is invalid or cannot be normalized.",
    message: "load: invalid filename",
};
const LOAD_ERROR_SELECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOAD.SELECTION",
    identifier: Some("RunMat:load:Selection"),
    when: "Requested variables are missing or no variables are selected.",
    message: "load: variable selection failed",
};
const LOAD_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOAD.IO",
    identifier: Some("RunMat:load:Io"),
    when: "MAT-file data cannot be read or decoded.",
    message: "load: MAT-file I/O failure",
};
const LOAD_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOAD.OUTPUT_COUNT",
    identifier: Some("RunMat:load:OutputCount"),
    when: "Caller requests more outputs than supported by load.",
    message: "load: unsupported output count",
};
const LOAD_ERROR_WORKSPACE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOAD.WORKSPACE",
    identifier: Some("RunMat:load:Workspace"),
    when: "Statement-form load cannot assign values into workspace.",
    message: "load: workspace assignment failed",
};
const LOAD_ERRORS: [BuiltinErrorDescriptor; 7] = [
    LOAD_ERROR_INVALID_ARGUMENT,
    LOAD_ERROR_INVALID_OPTION,
    LOAD_ERROR_FILENAME,
    LOAD_ERROR_SELECTION,
    LOAD_ERROR_IO,
    LOAD_ERROR_OUTPUT_COUNT,
    LOAD_ERROR_WORKSPACE,
];
pub const LOAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOAD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOAD_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::mat::load")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "load",
    op_kind: GpuOpKind::Custom("io-load"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Reads MAT-files on the host and produces CPU-resident values. Providers are not involved until accelerated code later promotes the results.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::mat::load")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "load",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O is not eligible for fusion. Registration exists for documentation completeness only.",
};

#[runtime_builtin(
    name = "load",
    category = "io/mat",
    summary = "Load variables from a MAT-file.",
    keywords = "load,mat,workspace",
    accel = "cpu",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::load_type),
    descriptor(crate::builtins::io::mat::load::LOAD_DESCRIPTOR),
    builtin_path = "crate::builtins::io::mat::load"
)]
async fn load_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&args).await?;

    // current_output_count() is set by the dispatcher only for multi-output Unpack patterns
    // like `[a, b] = load(...)`. Guard against requesting more than one struct output.
    if let Some(n) = crate::output_count::current_output_count() {
        if n > 1 {
            return Err(load_error_with(
                &LOAD_ERROR_OUTPUT_COUNT,
                "load supports at most one output argument",
            ));
        }
    }

    // The VM sets output_context::requested_output_count() at every call site before
    // dispatching:
    //   Some(0) → statement-level call (result is discarded or printed without capture)
    //             → assign loaded variables directly into the caller's workspace.
    //   Some(1) → single-output assignment `S = load(...)` → return a struct.
    //   None    → called outside the VM (e.g. directly from Rust) → return a struct.
    if crate::output_context::requested_output_count() == Some(0) {
        for (name, value) in eval.variables() {
            crate::workspace::assign(name, value.clone())
                .map_err(|err| load_error_with(&LOAD_ERROR_WORKSPACE, err))?;
        }
        return Ok(Value::OutputList(Vec::new()));
    }

    Ok(eval.first_output())
}

#[derive(Clone, Debug)]
pub struct LoadEval {
    variables: Vec<(String, Value)>,
}

impl LoadEval {
    pub fn first_output(&self) -> Value {
        let mut st = StructValue::new();
        for (name, value) in &self.variables {
            st.fields.insert(name.clone(), value.clone());
        }
        Value::Struct(st)
    }

    pub fn variables(&self) -> &[(String, Value)] {
        &self.variables
    }

    pub fn into_variables(self) -> Vec<(String, Value)> {
        self.variables
    }
}

struct LoadRequest {
    variables: Vec<String>,
    regex_patterns: Vec<Regex>,
}

const BUILTIN_NAME: &str = "load";

fn load_error(message: impl Into<String>) -> RuntimeError {
    load_error_with(&LOAD_ERROR_INVALID_ARGUMENT, message)
}

fn load_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn load_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: impl std::error::Error + Send + Sync + 'static,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub async fn evaluate(args: &[Value]) -> BuiltinResult<LoadEval> {
    let mut host_args = Vec::with_capacity(args.len());
    for arg in args {
        host_args.push(gather_if_needed_async(arg).await?);
    }

    let invocation = parse_invocation(&host_args).await?;

    let mut path_value = if let Some(path) = invocation.path_value {
        path
    } else {
        Value::from("matlab.mat")
    };

    if invocation.path_was_default {
        if let Ok(override_path) = std::env::var("RUNMAT_LOAD_DEFAULT_PATH") {
            path_value = Value::from(override_path);
        }
    }

    let mut regex_patterns = Vec::with_capacity(invocation.regex_tokens.len());
    for pattern in invocation.regex_tokens {
        let regex = Regex::new(&pattern).map_err(|err| {
            load_error_with_source(
                &LOAD_ERROR_INVALID_OPTION,
                format!("load: invalid regular expression '{pattern}': {err}"),
                err,
            )
        })?;
        regex_patterns.push(regex);
    }

    let request = LoadRequest {
        variables: invocation.variables,
        regex_patterns,
    };
    let path = normalise_path(&path_value)?;
    let entries = read_mat_file(&path).await?;

    let selected = select_variables(&entries, &request)?;
    Ok(LoadEval {
        variables: selected,
    })
}

struct ParsedInvocation {
    path_value: Option<Value>,
    path_was_default: bool,
    variables: Vec<String>,
    regex_tokens: Vec<String>,
}

async fn parse_invocation(values: &[Value]) -> BuiltinResult<ParsedInvocation> {
    let mut path_value = None;
    let mut path_was_default = false;
    let mut variables = Vec::new();
    let mut regex_tokens = Vec::new();
    let mut idx = 0usize;
    while idx < values.len() {
        if let Some(flag) = option_token(&values[idx])? {
            match flag.as_str() {
                "-mat" => {
                    idx += 1;
                    continue;
                }
                "-regexp" => {
                    idx += 1;
                    if idx >= values.len() {
                        return Err(load_error_with(
                            &LOAD_ERROR_INVALID_OPTION,
                            "load: '-regexp' requires at least one pattern",
                        ));
                    }
                    while idx < values.len() {
                        if option_token(&values[idx])?.is_some() {
                            break;
                        }
                        let names = extract_names(&values[idx]).await?;
                        if names.is_empty() {
                            return Err(load_error_with(
                                &LOAD_ERROR_INVALID_OPTION,
                                "load: '-regexp' requires non-empty pattern strings",
                            ));
                        }
                        regex_tokens.extend(names);
                        idx += 1;
                    }
                    continue;
                }
                other => {
                    return Err(load_error_with(
                        &LOAD_ERROR_INVALID_OPTION,
                        format!("load: unsupported option '{other}'"),
                    ));
                }
            }
        } else {
            if path_value.is_none() {
                path_value = Some(values[idx].clone());
                idx += 1;
                continue;
            }
            let names = extract_names(&values[idx]).await?;
            variables.extend(names);
            idx += 1;
        }
    }

    if path_value.is_none() {
        path_was_default = true;
    }

    Ok(ParsedInvocation {
        path_value,
        path_was_default,
        variables,
        regex_tokens,
    })
}

fn normalise_path(value: &Value) -> BuiltinResult<PathBuf> {
    let raw = value_to_string_scalar(value).ok_or_else(|| {
        load_error_with(
            &LOAD_ERROR_FILENAME,
            "load: filename must be a character vector or string scalar",
        )
    })?;
    let mut path = PathBuf::from(raw);
    if path.extension().is_none() {
        path.set_extension("mat");
    }
    Ok(path)
}

fn select_variables(
    entries: &[(String, Value)],
    request: &LoadRequest,
) -> BuiltinResult<Vec<(String, Value)>> {
    if request.variables.is_empty() && request.regex_patterns.is_empty() {
        return Ok(entries.to_vec());
    }

    let mut by_name: HashMap<&str, &Value> = HashMap::with_capacity(entries.len());
    for (name, value) in entries {
        by_name.insert(name, value);
    }

    let mut selected = Vec::new();

    for name in &request.variables {
        let value = by_name.get(name.as_str()).ok_or_else(|| {
            load_error_with(
                &LOAD_ERROR_SELECTION,
                format!("load: variable '{name}' was not found in the file"),
            )
        })?;
        insert_or_replace(&mut selected, name, (*value).clone());
    }

    if !request.regex_patterns.is_empty() {
        let mut matched = 0usize;
        for (name, value) in entries {
            if request
                .regex_patterns
                .iter()
                .any(|regex| regex.is_match(name))
            {
                matched += 1;
                insert_or_replace(&mut selected, name, value.clone());
            }
        }
        if matched == 0 && request.variables.is_empty() {
            return Err(load_error_with(
                &LOAD_ERROR_SELECTION,
                "load: no variables matched '-regexp' patterns",
            ));
        }
    }

    if selected.is_empty() {
        return Err(load_error_with(
            &LOAD_ERROR_SELECTION,
            "load: no variables selected",
        ));
    }

    Ok(selected)
}

fn insert_or_replace(selected: &mut Vec<(String, Value)>, name: &str, value: Value) {
    if let Some(entry) = selected.iter_mut().find(|(existing, _)| existing == name) {
        entry.1 = value;
    } else {
        selected.push((name.to_string(), value));
    }
}

pub(crate) async fn read_mat_file_for_builtin(
    path: &Path,
    builtin: &str,
) -> crate::BuiltinResult<Vec<(String, Value)>> {
    match read_mat_file(path).await {
        Ok(entries) => Ok(entries),
        Err(err) => {
            let message = err.message().replacen("load:", &format!("{builtin}:"), 1);
            let mut builder = build_runtime_error(message).with_builtin(builtin);
            if let Some(identifier) = err.identifier() {
                builder = builder.with_identifier(identifier);
            }
            Err(builder.with_source(err).build())
        }
    }
}

pub(crate) async fn read_mat_file(path: &Path) -> BuiltinResult<Vec<(String, Value)>> {
    let file = File::open_async(path).await.map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!("load: failed to open '{}': {err}", path.display()),
            err,
        )
    })?;
    let mut reader = BufReader::new(file);
    read_mat_reader(&mut reader)
}

pub fn decode_workspace_from_mat_bytes(bytes: &[u8]) -> BuiltinResult<Vec<(String, Value)>> {
    let mut cursor = Cursor::new(bytes);
    read_mat_reader(&mut cursor)
}

#[derive(Clone, Copy, Debug)]
enum Endian {
    Little,
    Big,
}

impl Endian {
    fn read_u16(self, bytes: &[u8]) -> u16 {
        match self {
            Endian::Little => u16::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => u16::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_i16(self, bytes: &[u8]) -> i16 {
        match self {
            Endian::Little => i16::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => i16::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_u32(self, bytes: &[u8]) -> u32 {
        match self {
            Endian::Little => u32::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => u32::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_i32(self, bytes: &[u8]) -> i32 {
        match self {
            Endian::Little => i32::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => i32::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_u64(self, bytes: &[u8]) -> u64 {
        match self {
            Endian::Little => u64::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => u64::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_i64(self, bytes: &[u8]) -> i64 {
        match self {
            Endian::Little => i64::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => i64::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_f32(self, bytes: &[u8]) -> f32 {
        match self {
            Endian::Little => f32::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => f32::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_f64(self, bytes: &[u8]) -> f64 {
        match self {
            Endian::Little => f64::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => f64::from_be_bytes(bytes.try_into().unwrap()),
        }
    }
}

fn read_mat_reader<R: Read>(reader: &mut R) -> BuiltinResult<Vec<(String, Value)>> {
    let mut header = [0u8; MAT_HEADER_LEN];
    reader.read_exact(&mut header).map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!("load: failed to read MAT-file header: {err}"),
            err,
        )
    })?;

    let description = String::from_utf8_lossy(&header[..116]);
    if description.contains("MATLAB 7.3") || header.starts_with(b"\x89HDF\r\n\x1a\n") {
        return Err(load_error(
            "load: MATLAB v7.3 MAT-files are HDF5-backed and are not supported yet",
        ));
    }

    let endian = match (header[126], header[127]) {
        (b'I', b'M') => Endian::Little,
        (b'M', b'I') => Endian::Big,
        _ => return Err(load_error("load: file is not a MATLAB Level-5 MAT-file")),
    };

    read_variables_from_elements(reader, endian)
}

fn read_variables_from_elements<R: Read>(
    reader: &mut R,
    endian: Endian,
) -> BuiltinResult<Vec<(String, Value)>> {
    let mut variables = Vec::new();
    while let Some(tagged) = read_tagged(reader, true, endian)? {
        match tagged.data_type {
            MI_MATRIX => {
                let parsed = parse_matrix(&tagged.data, endian)?;
                let value = mat_array_to_value(parsed.array)?;
                variables.push((parsed.name, value));
            }
            MI_COMPRESSED => {
                let mut decoder = ZlibDecoder::new(Cursor::new(tagged.data));
                let mut inflated = Vec::new();
                decoder.read_to_end(&mut inflated).map_err(|err| {
                    load_error_with_source(
                        &LOAD_ERROR_IO,
                        format!("load: failed to decompress MAT element: {err}"),
                        err,
                    )
                })?;
                let mut cursor = Cursor::new(inflated);
                variables.extend(read_variables_from_elements(&mut cursor, endian)?);
            }
            _ => continue,
        }
    }
    Ok(variables)
}

struct ParsedMatrix {
    name: String,
    array: MatArray,
}

fn parse_matrix(buffer: &[u8], endian: Endian) -> BuiltinResult<ParsedMatrix> {
    let mut cursor = Cursor::new(buffer);

    let flags = read_tagged(&mut cursor, false, endian)?
        .ok_or_else(|| load_error("load: matrix element missing array flags"))?;
    if flags.data_type != MI_UINT32 || flags.data.len() < 8 {
        return Err(load_error("load: invalid array flags block"));
    }
    let flags0 = endian.read_u32(&flags.data[0..4]);
    let class_code = flags0 & 0xFF;
    let mut class = MatClass::from_class_code(class_code)
        .ok_or_else(|| load_error("load: unsupported MATLAB class"))?;
    let is_logical = (flags0 & FLAG_LOGICAL) != 0;
    let has_imag = (flags0 & FLAG_COMPLEX) != 0;
    if is_logical {
        class = MatClass::Logical;
    }

    let dims_elem = read_tagged(&mut cursor, false, endian)?
        .ok_or_else(|| load_error("load: matrix element missing dimensions"))?;
    if dims_elem.data_type != MI_INT32 {
        return Err(load_error("load: dimension block must use MI_INT32"));
    }
    if dims_elem.data.is_empty() || dims_elem.data.len() % 4 != 0 {
        return Err(load_error("load: malformed dimension block"));
    }
    let mut dims = Vec::with_capacity(dims_elem.data.len() / 4);
    for chunk in dims_elem.data.chunks_exact(4) {
        let value = endian.read_i32(chunk);
        if value < 0 {
            return Err(load_error("load: negative dimensions are not supported"));
        }
        dims.push(value as usize);
    }
    if dims.is_empty() {
        dims.push(1);
        dims.push(1);
    }

    let name_elem = read_tagged(&mut cursor, false, endian)?
        .ok_or_else(|| load_error("load: matrix element missing name"))?;
    let name = match name_elem.data_type {
        MI_INT8 | MI_UINT8 => bytes_to_string(&name_elem.data),
        MI_UINT16 | MI_UTF16 => {
            let mut bytes = Vec::with_capacity(name_elem.data.len());
            for chunk in name_elem.data.chunks_exact(2) {
                let code = endian.read_u16(chunk);
                if code == 0 {
                    break;
                }
                if let Some(ch) = char::from_u32(code as u32) {
                    bytes.push(ch);
                }
            }
            bytes.into_iter().collect()
        }
        _ => {
            return Err(load_error("load: unsupported array name encoding"));
        }
    };

    let array = match class {
        MatClass::Double => parse_numeric_array(&mut cursor, class, dims, has_imag, endian)?,
        MatClass::Single
        | MatClass::Int8
        | MatClass::UInt8
        | MatClass::Int16
        | MatClass::UInt16
        | MatClass::Int32
        | MatClass::UInt32
        | MatClass::Int64
        | MatClass::UInt64 => parse_numeric_array(&mut cursor, class, dims, has_imag, endian)?,
        MatClass::Logical => parse_logical_array(&mut cursor, dims, endian)?,
        MatClass::Char => parse_char_array(&mut cursor, dims, endian)?,
        MatClass::Cell => parse_cell_array(&mut cursor, dims, endian)?,
        MatClass::Struct => parse_struct(&mut cursor, dims, endian)?,
        MatClass::Sparse => parse_sparse_array(&mut cursor, dims, has_imag, endian)?,
    };

    Ok(ParsedMatrix { name, array })
}

fn parse_numeric_array(
    cursor: &mut Cursor<&[u8]>,
    class: MatClass,
    dims: Vec<usize>,
    has_imag: bool,
    endian: Endian,
) -> BuiltinResult<MatArray> {
    let real_elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: numeric array missing real component"))?;
    let real = decode_numeric_values(&real_elem, endian)?;

    let imag = if has_imag {
        let imag_elem = read_tagged(cursor, false, endian)?
            .ok_or_else(|| load_error("load: numeric array missing imaginary component"))?;
        Some(decode_numeric_values(&imag_elem, endian)?)
    } else {
        None
    };

    let data = if class == MatClass::Double {
        MatData::Double { real, imag }
    } else {
        MatData::Numeric { real, imag }
    };

    Ok(MatArray { class, dims, data })
}

fn parse_logical_array(
    cursor: &mut Cursor<&[u8]>,
    dims: Vec<usize>,
    endian: Endian,
) -> BuiltinResult<MatArray> {
    let elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: logical array missing data block"))?;
    let data = decode_numeric_values(&elem, endian)?
        .into_iter()
        .map(|v| if v != 0.0 { 1 } else { 0 })
        .collect();
    Ok(MatArray {
        class: MatClass::Logical,
        dims,
        data: MatData::Logical { data },
    })
}

fn parse_char_array(
    cursor: &mut Cursor<&[u8]>,
    dims: Vec<usize>,
    endian: Endian,
) -> BuiltinResult<MatArray> {
    let elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: character array missing data block"))?;
    let data = decode_char_codes(&elem, endian)?;
    Ok(MatArray {
        class: MatClass::Char,
        dims,
        data: MatData::Char { data },
    })
}

fn parse_cell_array(
    cursor: &mut Cursor<&[u8]>,
    dims: Vec<usize>,
    endian: Endian,
) -> BuiltinResult<MatArray> {
    let total: usize = dims
        .iter()
        .copied()
        .fold(1usize, |acc, d| acc.saturating_mul(d));
    let mut elements = Vec::with_capacity(total);
    for _ in 0..total {
        let elem = read_tagged(cursor, false, endian)?
            .ok_or_else(|| load_error("load: cell element missing matrix payload"))?;
        if elem.data_type != MI_MATRIX {
            return Err(load_error("load: cell elements must be matrices"));
        }
        let parsed = parse_matrix(&elem.data, endian)?;
        elements.push(parsed.array);
    }
    Ok(MatArray {
        class: MatClass::Cell,
        dims,
        data: MatData::Cell { elements },
    })
}

fn parse_struct(
    cursor: &mut Cursor<&[u8]>,
    dims: Vec<usize>,
    endian: Endian,
) -> BuiltinResult<MatArray> {
    if dims.len() != 2 || dims[0] != 1 || dims[1] != 1 {
        return Err(load_error("load: struct arrays are not supported yet"));
    }
    let len_elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: struct missing maximum field length specifier"))?;
    if len_elem.data_type != MI_INT32 || len_elem.data.len() != 4 {
        return Err(load_error("load: struct field length must be MI_INT32"));
    }
    let max_len = endian.read_i32(&len_elem.data[..4]);
    if max_len <= 0 {
        return Err(load_error("load: struct field length must be positive"));
    }

    let names_elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: struct missing field name table"))?;
    if names_elem.data_type != MI_INT8 && names_elem.data_type != MI_UINT8 {
        return Err(load_error(
            "load: struct field names must be stored as MI_INT8/MI_UINT8",
        ));
    }
    if names_elem.data.len() % (max_len as usize) != 0 {
        return Err(load_error("load: malformed struct field name table"));
    }
    let field_count = names_elem.data.len() / (max_len as usize);
    let mut field_names = Vec::with_capacity(field_count);
    for i in 0..field_count {
        let start = i * (max_len as usize);
        let end = start + (max_len as usize);
        let slice = &names_elem.data[start..end];
        field_names.push(bytes_to_string(slice));
    }

    let mut field_values = Vec::with_capacity(field_count);
    for _ in 0..field_count {
        let elem = read_tagged(cursor, false, endian)?
            .ok_or_else(|| load_error("load: struct field missing matrix payload"))?;
        if elem.data_type != MI_MATRIX {
            return Err(load_error("load: struct fields must be matrices"));
        }
        let parsed = parse_matrix(&elem.data, endian)?;
        field_values.push(parsed.array);
    }

    Ok(MatArray {
        class: MatClass::Struct,
        dims,
        data: MatData::Struct {
            field_names,
            field_values,
        },
    })
}

fn parse_sparse_array(
    cursor: &mut Cursor<&[u8]>,
    dims: Vec<usize>,
    has_imag: bool,
    endian: Endian,
) -> BuiltinResult<MatArray> {
    if has_imag {
        return Err(load_error(
            "load: complex sparse MAT arrays are not supported yet",
        ));
    }
    let rows = dims.first().copied().unwrap_or(0);
    let cols = dims.get(1).copied().unwrap_or(0);

    let ir_elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: sparse array missing row indices"))?;
    let row_indices = decode_index_values(&ir_elem, endian, "sparse row indices")?;

    let jc_elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: sparse array missing column pointers"))?;
    let col_ptrs = decode_index_values(&jc_elem, endian, "sparse column pointers")?;

    let real_elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: sparse array missing values"))?;
    let values = decode_numeric_values(&real_elem, endian)?;

    Ok(MatArray {
        class: MatClass::Sparse,
        dims,
        data: MatData::Sparse {
            rows,
            cols,
            col_ptrs,
            row_indices,
            values,
        },
    })
}

fn decode_numeric_values(elem: &TaggedData, endian: Endian) -> BuiltinResult<Vec<f64>> {
    let data = &elem.data;
    match elem.data_type {
        MI_INT8 => Ok(data.iter().map(|v| (*v as i8) as f64).collect()),
        MI_UINT8 => Ok(data.iter().map(|v| *v as f64).collect()),
        MI_INT16 => {
            ensure_data_width(data, 2, "numeric int16 data")?;
            Ok(data
                .chunks_exact(2)
                .map(|chunk| endian.read_i16(chunk) as f64)
                .collect())
        }
        MI_UINT16 => {
            ensure_data_width(data, 2, "numeric uint16 data")?;
            Ok(data
                .chunks_exact(2)
                .map(|chunk| endian.read_u16(chunk) as f64)
                .collect())
        }
        MI_INT32 => {
            ensure_data_width(data, 4, "numeric int32 data")?;
            Ok(data
                .chunks_exact(4)
                .map(|chunk| endian.read_i32(chunk) as f64)
                .collect())
        }
        MI_UINT32 => {
            ensure_data_width(data, 4, "numeric uint32 data")?;
            Ok(data
                .chunks_exact(4)
                .map(|chunk| endian.read_u32(chunk) as f64)
                .collect())
        }
        MI_SINGLE => {
            ensure_data_width(data, 4, "numeric single data")?;
            Ok(data
                .chunks_exact(4)
                .map(|chunk| endian.read_f32(chunk) as f64)
                .collect())
        }
        MI_DOUBLE => {
            ensure_data_width(data, 8, "numeric double data")?;
            Ok(data
                .chunks_exact(8)
                .map(|chunk| endian.read_f64(chunk))
                .collect())
        }
        MI_INT64 => {
            ensure_data_width(data, 8, "numeric int64 data")?;
            Ok(data
                .chunks_exact(8)
                .map(|chunk| endian.read_i64(chunk) as f64)
                .collect())
        }
        MI_UINT64 => {
            ensure_data_width(data, 8, "numeric uint64 data")?;
            Ok(data
                .chunks_exact(8)
                .map(|chunk| endian.read_u64(chunk) as f64)
                .collect())
        }
        _ => Err(load_error(format!(
            "load: unsupported numeric data type {}",
            elem.data_type
        ))),
    }
}

fn decode_char_codes(elem: &TaggedData, endian: Endian) -> BuiltinResult<Vec<u16>> {
    match elem.data_type {
        MI_INT8 | MI_UINT8 | MI_UTF8 => {
            let text = String::from_utf8_lossy(&elem.data);
            Ok(text
                .chars()
                .map(|ch| {
                    if ch as u32 <= u16::MAX as u32 {
                        ch as u16
                    } else {
                        0xFFFD
                    }
                })
                .collect())
        }
        MI_UINT16 | MI_UTF16 => {
            ensure_data_width(&elem.data, 2, "character UTF-16 data")?;
            Ok(elem
                .data
                .chunks_exact(2)
                .map(|chunk| endian.read_u16(chunk))
                .collect())
        }
        MI_UTF32 => {
            ensure_data_width(&elem.data, 4, "character UTF-32 data")?;
            Ok(elem
                .data
                .chunks_exact(4)
                .map(|chunk| {
                    let code = endian.read_u32(chunk);
                    if code <= u16::MAX as u32 {
                        code as u16
                    } else {
                        0xFFFD
                    }
                })
                .collect())
        }
        _ => Err(load_error("load: unsupported character data encoding")),
    }
}

fn decode_index_values(
    elem: &TaggedData,
    endian: Endian,
    label: &str,
) -> BuiltinResult<Vec<usize>> {
    match elem.data_type {
        MI_INT32 => {
            ensure_data_width(&elem.data, 4, label)?;
            let mut out = Vec::with_capacity(elem.data.len() / 4);
            for chunk in elem.data.chunks_exact(4) {
                let value = endian.read_i32(chunk);
                if value < 0 {
                    return Err(load_error(format!(
                        "load: {label} contain a negative index"
                    )));
                }
                out.push(value as usize);
            }
            Ok(out)
        }
        MI_UINT32 => {
            ensure_data_width(&elem.data, 4, label)?;
            Ok(elem
                .data
                .chunks_exact(4)
                .map(|chunk| endian.read_u32(chunk) as usize)
                .collect())
        }
        _ => Err(load_error(format!(
            "load: {label} must be MI_INT32/MI_UINT32"
        ))),
    }
}

fn ensure_data_width(data: &[u8], width: usize, label: &str) -> BuiltinResult<()> {
    if !data.len().is_multiple_of(width) {
        return Err(load_error(format!("load: malformed {label}")));
    }
    Ok(())
}

fn mat_array_to_value(array: MatArray) -> BuiltinResult<Value> {
    match array.data {
        MatData::Double { real, imag } => {
            let len = real.len();
            if let Some(imag) = imag {
                if imag.len() != len {
                    return Err(load_error(
                        "load: complex data has mismatched real/imag parts",
                    ));
                }
                if len == 1 {
                    Ok(Value::Complex(real[0], imag[0]))
                } else {
                    let mut pairs = Vec::with_capacity(len);
                    for i in 0..len {
                        pairs.push((real[i], imag[i]));
                    }
                    let tensor = ComplexTensor::new(pairs, array.dims.clone())
                        .map_err(|e| load_error(format!("load: {e}")))?;
                    Ok(Value::ComplexTensor(tensor))
                }
            } else if len == 1 {
                Ok(Value::Num(real[0]))
            } else {
                let tensor = Tensor::new(real, array.dims.clone())
                    .map_err(|e| load_error(format!("load: {e}")))?;
                Ok(Value::Tensor(tensor))
            }
        }
        MatData::Numeric { real, imag } => {
            let len = real.len();
            if let Some(imag) = imag {
                if imag.len() != len {
                    return Err(load_error(
                        "load: complex data has mismatched real/imag parts",
                    ));
                }
                if len == 1 {
                    return Ok(Value::Complex(real[0], imag[0]));
                }
                let pairs: Vec<(f64, f64)> = real.into_iter().zip(imag).collect();
                let tensor = ComplexTensor::new(pairs, array.dims.clone())
                    .map_err(|e| load_error(format!("load: {e}")))?;
                return Ok(Value::ComplexTensor(tensor));
            }

            if len == 1 {
                if let Some(value) = numeric_scalar_to_int_value(array.class, real[0]) {
                    return Ok(value);
                }
                return Ok(Value::Num(real[0]));
            }

            let dtype = numeric_tensor_dtype(array.class);
            let tensor = Tensor::new_with_dtype(real, array.dims.clone(), dtype)
                .map_err(|e| load_error(format!("load: {e}")))?;
            Ok(Value::Tensor(tensor))
        }
        MatData::Logical { data } => {
            let total: usize = array
                .dims
                .iter()
                .copied()
                .fold(1usize, |acc, d| acc.saturating_mul(d));
            if data.len() != total {
                return Err(load_error("load: logical data length mismatch"));
            }
            if total == 1 {
                Ok(Value::Bool(data.first().copied().unwrap_or(0) != 0))
            } else {
                let logical = LogicalArray::new(data, array.dims.clone())
                    .map_err(|e| load_error(format!("load: {e}")))?;
                Ok(Value::LogicalArray(logical))
            }
        }
        MatData::Char { data } => {
            let rows = array.dims.first().copied().unwrap_or(1);
            let cols = array.dims.get(1).copied().unwrap_or(1);
            let mut chars = Vec::with_capacity(rows.saturating_mul(cols));
            for code in data {
                let ch = char::from_u32(code as u32).unwrap_or('\u{FFFD}');
                chars.push(ch);
            }
            let char_array =
                CharArray::new(chars, rows, cols).map_err(|e| load_error(format!("load: {e}")))?;
            Ok(Value::CharArray(char_array))
        }
        MatData::Cell { elements } => {
            if let Some(strings) = cell_elements_to_strings(&elements) {
                let string_array = StringArray::new(strings, array.dims.clone())
                    .map_err(|e| load_error(format!("load: {e}")))?;
                return Ok(Value::StringArray(string_array));
            }
            if array.dims.len() != 2 {
                return Err(load_error(
                    "load: cell arrays with more than two dimensions are not supported yet",
                ));
            }
            let rows = array.dims[0];
            let cols = array.dims[1];
            let expected = rows.saturating_mul(cols);
            if elements.len() != expected {
                return Err(load_error("load: cell array element count mismatch"));
            }
            let mut converted = Vec::with_capacity(elements.len());
            for elem in elements {
                converted.push(mat_array_to_value(elem)?);
            }
            let mut row_major = vec![Value::Num(0.0); expected];
            for col in 0..cols {
                for row in 0..rows {
                    let cm_idx = col * rows + row;
                    let rm_idx = row * cols + col;
                    row_major[rm_idx] = converted[cm_idx].clone();
                }
            }
            make_cell(row_major, rows, cols).map_err(|err| load_error(format!("load: {err}")))
        }
        MatData::Struct {
            field_names,
            field_values,
        } => {
            if field_names.len() != field_values.len() {
                return Err(load_error("load: struct field metadata is inconsistent"));
            }
            let mut st = StructValue::new();
            for (name, value) in field_names.into_iter().zip(field_values.into_iter()) {
                let converted = mat_array_to_value(value)?;
                st.fields.insert(name, converted);
            }
            Ok(Value::Struct(st))
        }
        MatData::Sparse {
            rows,
            cols,
            col_ptrs,
            row_indices,
            values,
        } => {
            let sparse = SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
                .map_err(|e| load_error(format!("load: {e}")))?;
            Ok(Value::SparseTensor(sparse))
        }
    }
}

fn numeric_scalar_to_int_value(class: MatClass, value: f64) -> Option<Value> {
    match class {
        MatClass::Int8 => Some(Value::Int(IntValue::I8(value as i8))),
        MatClass::UInt8 => Some(Value::Int(IntValue::U8(value as u8))),
        MatClass::Int16 => Some(Value::Int(IntValue::I16(value as i16))),
        MatClass::UInt16 => Some(Value::Int(IntValue::U16(value as u16))),
        MatClass::Int32 => Some(Value::Int(IntValue::I32(value as i32))),
        MatClass::UInt32 => Some(Value::Int(IntValue::U32(value as u32))),
        MatClass::Int64 => Some(Value::Int(IntValue::I64(value as i64))),
        MatClass::UInt64 => Some(Value::Int(IntValue::U64(value as u64))),
        _ => None,
    }
}

fn numeric_tensor_dtype(class: MatClass) -> NumericDType {
    match class {
        MatClass::Single => NumericDType::F32,
        MatClass::UInt8 => NumericDType::U8,
        MatClass::UInt16 => NumericDType::U16,
        _ => NumericDType::F64,
    }
}

fn cell_elements_to_strings(elements: &[MatArray]) -> Option<Vec<String>> {
    let mut strings = Vec::with_capacity(elements.len());
    for element in elements {
        if element.class != MatClass::Char {
            return None;
        }
        let rows = element.dims.first().copied().unwrap_or(1);
        if rows > 1 {
            return None;
        }
        match &element.data {
            MatData::Char { data } => strings.push(utf16_codes_to_string(data)),
            _ => return None,
        }
    }
    Some(strings)
}

fn utf16_codes_to_string(data: &[u16]) -> String {
    let mut chars: Vec<char> = data
        .iter()
        .map(|code| char::from_u32(*code as u32).unwrap_or('\u{FFFD}'))
        .collect();
    while matches!(chars.last(), Some(&'\0')) {
        chars.pop();
    }
    chars.into_iter().collect()
}

fn option_token(value: &Value) -> BuiltinResult<Option<String>> {
    if let Some(token) = value_to_string_scalar(value) {
        if token.starts_with('-') {
            return Ok(Some(token.to_ascii_lowercase()));
        }
    }
    Ok(None)
}

#[async_recursion::async_recursion(?Send)]
async fn extract_names(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::CharArray(ca) => Ok(char_array_rows_as_strings(ca)),
        Value::StringArray(sa) => Ok(sa.data.clone()),
        Value::Cell(ca) => {
            let mut names = Vec::with_capacity(ca.data.len());
            for handle in &ca.data {
                let inner = unsafe { &*handle.as_raw() };
                let text = value_to_string_scalar(inner).ok_or_else(|| {
                    load_error(
                        "load: cell arrays used for variable selection must contain string scalars",
                    )
                })?;
                names.push(text);
            }
            Ok(names)
        }
        other => {
            let gathered = gather_if_needed_async(other).await?;
            extract_names(&gathered).await
        }
    }
}

fn value_to_string_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        _ => None,
    }
}

fn char_array_rows_as_strings(ca: &CharArray) -> Vec<String> {
    let mut rows = Vec::with_capacity(ca.rows);
    for r in 0..ca.rows {
        let mut row = String::with_capacity(ca.cols);
        for c in 0..ca.cols {
            let idx = r * ca.cols + c;
            row.push(ca.data[idx]);
        }
        let trimmed = row.trim_end_matches([' ', '\0']).to_string();
        rows.push(trimmed);
    }
    rows
}

fn bytes_to_string(bytes: &[u8]) -> String {
    let trimmed = bytes
        .iter()
        .copied()
        .take_while(|b| *b != 0)
        .collect::<Vec<u8>>();
    String::from_utf8(trimmed).unwrap_or_default()
}

struct TaggedData {
    data_type: u32,
    data: Vec<u8>,
}

fn read_tagged<R: Read>(
    reader: &mut R,
    allow_eof: bool,
    endian: Endian,
) -> BuiltinResult<Option<TaggedData>> {
    let mut type_bytes = [0u8; 4];
    match reader.read_exact(&mut type_bytes) {
        Ok(()) => {}
        Err(err) => {
            if allow_eof && err.kind() == std::io::ErrorKind::UnexpectedEof {
                return Ok(None);
            }
            return Err(load_error_with_source(
                &LOAD_ERROR_IO,
                format!("load: failed to read MAT element header: {err}"),
                err,
            ));
        }
    }

    if allow_eof && type_bytes == [0; 4] {
        return Ok(None);
    }

    let type_field = endian.read_u32(&type_bytes);
    let high = (type_field >> 16) & 0xFFFF;
    let low = type_field & 0xFFFF;
    let small = match endian {
        Endian::Little if high != 0 => Some((low, high as usize)),
        Endian::Big if high != 0 && low <= 4 => Some((high, low as usize)),
        _ => None,
    };

    if let Some((data_type, num_bytes)) = small {
        let mut inline = [0u8; 4];
        reader.read_exact(&mut inline).map_err(|err| {
            load_error_with_source(
                &LOAD_ERROR_IO,
                format!("load: failed to read compact MAT element: {err}"),
                err,
            )
        })?;
        let mut data = inline[..num_bytes.min(4)].to_vec();
        data.truncate(num_bytes.min(4));
        Ok(Some(TaggedData { data_type, data }))
    } else {
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes).map_err(|err| {
            load_error_with_source(
                &LOAD_ERROR_IO,
                format!("load: failed to read MAT element length: {err}"),
                err,
            )
        })?;
        let length = endian.read_u32(&len_bytes) as usize;
        let mut data = vec![0u8; length];
        reader.read_exact(&mut data).map_err(|err| {
            load_error_with_source(
                &LOAD_ERROR_IO,
                format!("load: failed to read MAT element body: {err}"),
                err,
            )
        })?;
        let padding = if type_field == MI_COMPRESSED {
            0
        } else {
            (8 - (length % 8)) % 8
        };
        if padding != 0 {
            let mut pad = vec![0u8; padding];
            reader.read_exact(&mut pad).map_err(|err| {
                load_error_with_source(
                    &LOAD_ERROR_IO,
                    format!("load: failed to read MAT padding: {err}"),
                    err,
                )
            })?;
        }
        Ok(Some(TaggedData {
            data_type: type_field,
            data,
        }))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::io::mat::save::encode_workspace_to_mat_bytes;
    use crate::workspace::WorkspaceResolver;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use futures::executor::block_on;
    use runmat_builtins::StringArray;
    use runmat_thread_local::runmat_thread_local;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::io::Write;
    use tempfile::tempdir;

    runmat_thread_local! {
        static TEST_WORKSPACE: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
    }

    fn ensure_test_resolver() {
        crate::workspace::register_workspace_resolver(WorkspaceResolver {
            lookup: |name| TEST_WORKSPACE.with(|slot| slot.borrow().get(name).cloned()),
            snapshot: || {
                let mut entries: Vec<(String, Value)> =
                    TEST_WORKSPACE.with(|slot| slot.borrow().clone().into_iter().collect());
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                entries
            },
            globals: || Vec::new(),
            assign: None,
            clear: None,
            remove: None,
        });
    }

    fn set_workspace(entries: &[(&str, Value)]) {
        TEST_WORKSPACE.with(|slot| {
            let mut map = slot.borrow_mut();
            map.clear();
            for (name, value) in entries {
                map.insert((*name).to_string(), value.clone());
            }
        });
    }

    fn workspace_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::workspace::test_guard()
    }

    fn assert_error_contains<T>(result: crate::BuiltinResult<T>, snippet: &str) {
        match result {
            Err(err) => {
                assert!(
                    err.message().contains(snippet),
                    "expected error to contain '{snippet}', got '{}'",
                    err.message()
                );
            }
            Ok(_) => panic!("expected error containing '{snippet}'"),
        }
    }

    fn wrap_payload_as_compressed(mat_bytes: &[u8]) -> Vec<u8> {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&mat_bytes[MAT_HEADER_LEN..]).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut out = mat_bytes[..MAT_HEADER_LEN].to_vec();
        out.extend_from_slice(&MI_COMPRESSED.to_le_bytes());
        out.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        out.extend_from_slice(&compressed);
        out
    }

    fn load_entries_from_bytes(bytes: Vec<u8>) -> Vec<(String, Value)> {
        let mut cursor = Cursor::new(bytes);
        read_mat_reader(&mut cursor).expect("read MAT bytes")
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn load_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = LOAD_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"S = load()"));
        assert!(labels.contains(&"S = load(filename)"));
        assert!(labels.contains(&"S = load(filename, varName1, varName2, ...)"));
        assert!(labels.contains(&"S = load(filename, \"-regexp\", pattern1, ...)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn load_roundtrip_numeric() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        set_workspace(&[("A", Value::Tensor(tensor))]);

        let dir = tempdir().unwrap();
        let path = dir.path().join("numeric.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        block_on(crate::call_builtin_async(
            "save",
            std::slice::from_ref(&save_arg),
        ))
        .unwrap();

        let eval = block_on(evaluate(&[Value::from(path.to_string_lossy().to_string())]))
            .expect("load numeric");
        let struct_value = eval.first_output();
        match struct_value {
            Value::Struct(sv) => {
                assert!(sv.fields.contains_key("A"));
                match sv.fields.get("A").unwrap() {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![2, 2]);
                        assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0]);
                    }
                    other => panic!("expected tensor, got {other:?}"),
                }
            }
            other => panic!("expected struct, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn load_selected_variables() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[("signal", Value::Num(42.0)), ("noise", Value::Num(5.0))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("selection.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        block_on(crate::call_builtin_async(
            "save",
            std::slice::from_ref(&save_arg),
        ))
        .unwrap();

        let eval = block_on(evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("signal"),
        ]))
        .expect("load selection");
        let vars = eval.variables();
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].0, "signal");
        assert!(matches!(vars[0].1, Value::Num(42.0)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn load_regex_selection() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[
            ("w1", Value::Num(1.0)),
            ("w2", Value::Num(2.0)),
            ("bias", Value::Num(3.0)),
        ]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("regex.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        block_on(crate::call_builtin_async(
            "save",
            std::slice::from_ref(&save_arg),
        ))
        .unwrap();

        let eval = block_on(evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("-regexp"),
            Value::from("^w\\d$"),
        ]))
        .expect("load regex");
        let mut names: Vec<_> = eval.variables().iter().map(|(n, _)| n.clone()).collect();
        names.sort();
        assert_eq!(names, vec!["w1".to_string(), "w2".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn load_missing_variable_errors() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[("existing", Value::Num(7.0))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("missing.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        block_on(crate::call_builtin_async(
            "save",
            std::slice::from_ref(&save_arg),
        ))
        .unwrap();

        assert_error_contains(
            block_on(evaluate(&[
                Value::from(path.to_string_lossy().to_string()),
                Value::from("missing"),
            ])),
            "variable 'missing' was not found",
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn load_string_array_roundtrip() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        let strings = StringArray::new(vec!["foo".into(), "bar".into()], vec![1, 2]).unwrap();
        set_workspace(&[("labels", Value::StringArray(strings))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("strings.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        block_on(crate::call_builtin_async(
            "save",
            std::slice::from_ref(&save_arg),
        ))
        .unwrap();

        let eval = block_on(evaluate(&[Value::from(path.to_string_lossy().to_string())]))
            .expect("load strings");
        let struct_value = eval.first_output();
        match struct_value {
            Value::Struct(sv) => {
                let value = sv
                    .fields
                    .get("labels")
                    .expect("labels field missing in struct");
                match value {
                    Value::StringArray(sa) => {
                        assert_eq!(sa.shape, vec![1, 2]);
                        assert_eq!(sa.data, vec![String::from("foo"), String::from("bar")]);
                    }
                    other => panic!("expected string array, got {other:?}"),
                }
            }
            other => panic!("expected struct, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn load_compressed_level5_payload() {
        let mat_bytes = block_on(encode_workspace_to_mat_bytes(&[
            ("A".to_string(), Value::Num(123.0)),
            ("B".to_string(), Value::from("text")),
        ]))
        .unwrap();
        let compressed = wrap_payload_as_compressed(&mat_bytes);

        let entries = load_entries_from_bytes(compressed);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "A");
        assert!(matches!(entries[0].1, Value::Num(123.0)));
        assert_eq!(entries[1].0, "B");
        match &entries[1].1 {
            Value::CharArray(ca) => assert_eq!(ca.data.iter().collect::<String>(), "text"),
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn load_preserves_supported_numeric_mat_classes() {
        let single = Tensor::new_with_dtype(vec![1.5, 2.5], vec![1, 2], NumericDType::F32)
            .expect("single tensor");
        let uint16 = Tensor::new_with_dtype(vec![10.0, 20.0], vec![1, 2], NumericDType::U16)
            .expect("uint16 tensor");
        let bytes = block_on(encode_workspace_to_mat_bytes(&[
            ("s".to_string(), Value::Tensor(single)),
            ("u16".to_string(), Value::Tensor(uint16)),
            ("i".to_string(), Value::Int(IntValue::I16(-7))),
        ]))
        .unwrap();

        let entries = load_entries_from_bytes(bytes);
        let values: HashMap<_, _> = entries.into_iter().collect();
        match values.get("s").unwrap() {
            Value::Tensor(t) => {
                assert_eq!(t.dtype, NumericDType::F32);
                assert_eq!(t.data, vec![1.5, 2.5]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match values.get("u16").unwrap() {
            Value::Tensor(t) => {
                assert_eq!(t.dtype, NumericDType::U16);
                assert_eq!(t.data, vec![10.0, 20.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        assert_eq!(values.get("i"), Some(&Value::Int(IntValue::I16(-7))));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn load_save_real_sparse_roundtrip() {
        let sparse = SparseTensor::new(3, 3, vec![0, 1, 1, 3], vec![1, 0, 2], vec![4.0, 5.0, 6.0])
            .expect("sparse");
        let bytes = block_on(encode_workspace_to_mat_bytes(&[(
            "S".to_string(),
            Value::SparseTensor(sparse.clone()),
        )]))
        .unwrap();

        let entries = load_entries_from_bytes(bytes);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "S");
        assert_eq!(entries[0].1, Value::SparseTensor(sparse));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn load_option_before_filename() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[("alpha", Value::Num(1.0)), ("beta", Value::Num(2.0))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("option_first.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        block_on(crate::call_builtin_async(
            "save",
            std::slice::from_ref(&save_arg),
        ))
        .unwrap();

        let eval = block_on(evaluate(&[
            Value::from("-mat"),
            Value::from(path.to_string_lossy().to_string()),
            Value::from("beta"),
        ]))
        .expect("load with option first");
        let vars = eval.variables();
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].0, "beta");
        assert!(matches!(vars[0].1, Value::Num(2.0)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn load_char_array_names_trimmed() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[("short", Value::Num(5.0)), ("longer", Value::Num(9.0))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("char_names.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        block_on(crate::call_builtin_async(
            "save",
            std::slice::from_ref(&save_arg),
        ))
        .unwrap();

        let cols = 6;
        let mut data = Vec::new();
        for name in ["short", "longer"] {
            let mut chars: Vec<char> = name.chars().collect();
            while chars.len() < cols {
                chars.push(' ');
            }
            data.extend(chars);
        }
        let name_array = CharArray::new(data, 2, cols).unwrap();

        let eval = block_on(evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::CharArray(name_array),
        ]))
        .expect("load with char array names");
        let vars = eval.variables();
        assert_eq!(vars.len(), 2);
        assert_eq!(vars[0].0, "short");
        assert!(matches!(vars[0].1, Value::Num(5.0)));
        assert_eq!(vars[1].0, "longer");
        assert!(matches!(vars[1].1, Value::Num(9.0)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn load_duplicate_names_last_wins() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[("dup", Value::Num(11.0))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("duplicates.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        block_on(crate::call_builtin_async(
            "save",
            std::slice::from_ref(&save_arg),
        ))
        .unwrap();

        let eval = block_on(evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("dup"),
            Value::from("dup"),
        ]))
        .expect("load with duplicate names");
        let vars = eval.variables();
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].0, "dup");
        assert!(matches!(vars[0].1, Value::Num(11.0)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn load_wgpu_tensor_roundtrip() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };

        use runmat_accelerate_api::HostTensorView;

        let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload tensor");
        set_workspace(&[("gpu_var", Value::GpuTensor(handle))]);

        let dir = tempdir().unwrap();
        let path = dir.path().join("wgpu_load.mat");
        let save_args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("gpu_var"),
        ];
        block_on(crate::call_builtin_async("save", &save_args)).unwrap();

        let eval = block_on(evaluate(&[Value::from(path.to_string_lossy().to_string())]))
            .expect("load wgpu file");
        let struct_value = eval.first_output();
        match struct_value {
            Value::Struct(sv) => match sv.fields.get("gpu_var") {
                Some(Value::Tensor(t)) => {
                    assert_eq!(t.shape, vec![2, 2]);
                    assert_eq!(t.data, tensor.data);
                }
                other => panic!("expected tensor, got {other:?}"),
            },
            other => panic!("expected struct, got {other:?}"),
        }
    }
}
