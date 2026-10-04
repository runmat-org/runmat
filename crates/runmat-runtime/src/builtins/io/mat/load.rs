//! MATLAB-compatible `load` builtin for RunMat.

use std::collections::HashMap;
use std::io::{BufReader, Cursor, Read};
use std::path::{Path, PathBuf};

use flate2::read::ZlibDecoder;
use regex::Regex;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage,
    LogicalArray, NumericStorage, SparseTensor, StringArray, StructArray, StructValue, Tensor,
    Value,
};

use super::authority::MatFsPath;
use super::format::{
    MatArray, MatClass, MatData, FLAG_COMPLEX, FLAG_LOGICAL, MAT_HEADER_LEN, MI_COMPRESSED,
    MI_DOUBLE, MI_INT16, MI_INT32, MI_INT64, MI_INT8, MI_MATRIX, MI_SINGLE, MI_UINT16, MI_UINT32,
    MI_UINT64, MI_UINT8, MI_UTF16, MI_UTF32, MI_UTF8,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::introspection::object_serialization::restore_value_from_mat_load;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use runmat_builtins::{
    LOAD_ERROR_FILENAME, LOAD_ERROR_INVALID_ARGUMENT, LOAD_ERROR_INVALID_OPTION, LOAD_ERROR_IO,
    LOAD_ERROR_OUTPUT_COUNT, LOAD_ERROR_SELECTION, LOAD_ERROR_WORKSPACE,
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
    binding_variant = "default",
    builtin_path = "crate::builtins::io::mat::load"
)]
async fn load_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    // The dispatcher scopes this count to the current call. Presentation
    // output context may belong to an enclosing expression, so it is not an
    // independent upper-bound witness for this invocation.
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(load_error_with(
            &LOAD_ERROR_OUTPUT_COUNT,
            "load supports at most one output argument",
        ));
    }
    let eval = evaluate(&args).await?;

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

pub async fn evaluate(args: &[Value]) -> BuiltinResult<LoadEval> {
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        return Err(load_error_with(
            &LOAD_ERROR_INVALID_ARGUMENT,
            "load: filename, options, and variable selectors must be host text values",
        ));
    }
    preflight_known_options(args)?;
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
    let source = MatFsPath::capture(path).map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!("load: failed to resolve MAT path: {err}"),
            err,
        )
    })?;
    let entries = read_mat_file_on(&source).await?;

    let selected = select_variables(&entries, &request)?;
    Ok(LoadEval {
        variables: selected,
    })
}

fn preflight_known_options(args: &[Value]) -> BuiltinResult<()> {
    for (index, value) in args.iter().enumerate() {
        let Some(token) = value_to_string_scalar(value) else {
            continue;
        };
        if !token.starts_with('-') {
            continue;
        }
        match runmat_builtins::MatLoadOption::parse(&token) {
            Some(runmat_builtins::MatLoadOption::Regexp) if index + 1 == args.len() => {
                return Err(load_error_with(
                    &LOAD_ERROR_INVALID_OPTION,
                    "load: '-regexp' requires at least one pattern",
                ));
            }
            None => {
                return Err(load_error_with(
                    &LOAD_ERROR_INVALID_OPTION,
                    format!("load: unsupported option '{}'", token.to_ascii_lowercase()),
                ));
            }
            _ => {}
        }
    }
    Ok(())
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
            match runmat_builtins::MatLoadOption::parse(&flag) {
                Some(runmat_builtins::MatLoadOption::Mat) => {
                    idx += 1;
                    continue;
                }
                Some(runmat_builtins::MatLoadOption::Regexp) => {
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
                None => {
                    return Err(load_error_with(
                        &LOAD_ERROR_INVALID_OPTION,
                        format!("load: unsupported option '{flag}'"),
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

pub(crate) async fn read_mat_file_for_builtin_on(
    source: &MatFsPath,
    builtin: &str,
) -> crate::BuiltinResult<Vec<(String, Value)>> {
    match read_mat_file_on(source).await {
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
    let source = MatFsPath::capture(path.to_path_buf()).map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!("load: failed to resolve MAT path: {err}"),
            err,
        )
    })?;
    read_mat_file_on(&source).await
}

pub(crate) async fn read_mat_file_on(source: &MatFsPath) -> BuiltinResult<Vec<(String, Value)>> {
    let file = source.open().await.map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!("load: failed to open '{}': {err}", source.path().display()),
            err,
        )
    })?;
    let mut reader = BufReader::new(file);
    let entries = read_mat_reader(&mut reader)?;
    check_source_current(source)?;
    restore_loaded_entries(entries, Some(source)).await
}

pub fn decode_workspace_from_mat_bytes(bytes: &[u8]) -> BuiltinResult<Vec<(String, Value)>> {
    let mut cursor = Cursor::new(bytes);
    let entries = read_mat_reader(&mut cursor)?;
    futures::executor::block_on(restore_loaded_entries(entries, None))
}

async fn restore_loaded_entries(
    entries: Vec<(String, Value)>,
    source: Option<&MatFsPath>,
) -> BuiltinResult<Vec<(String, Value)>> {
    let mut restored = Vec::with_capacity(entries.len());
    for (name, value) in entries {
        if let Some(source) = source {
            check_source_current(source)?;
        }
        restored.push((name, restore_value_from_mat_load(value).await?));
        if let Some(source) = source {
            check_source_current(source)?;
        }
    }
    Ok(restored)
}

fn check_source_current(source: &MatFsPath) -> BuiltinResult<()> {
    source.check_current().map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!(
                "load: MAT source authority for '{}' changed: {err}",
                source.path().display()
            ),
            err,
        )
    })
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
    if is_logical && class != MatClass::Sparse {
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
        MatClass::Sparse => parse_sparse_array(&mut cursor, dims, has_imag, is_logical, endian)?,
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
    if class_is_integer(class) {
        let storage = decode_integer_storage(&real_elem, class, endian)?;
        let imag = if has_imag {
            let imag_elem = read_tagged(cursor, false, endian)?.ok_or_else(|| {
                load_error("load: complex integer array missing imaginary component")
            })?;
            Some(decode_integer_storage(&imag_elem, class, endian)?)
        } else {
            None
        };
        return Ok(MatArray {
            class,
            dims,
            data: MatData::Numeric {
                real: storage.into(),
                imag: imag.map(Into::into),
            },
        });
    }

    let real = decode_numeric_storage(&real_elem, class, endian)?;

    let imag = if has_imag {
        let imag_elem = read_tagged(cursor, false, endian)?
            .ok_or_else(|| load_error("load: numeric array missing imaginary component"))?;
        Some(decode_numeric_storage(&imag_elem, class, endian)?)
    } else {
        None
    };

    Ok(MatArray {
        class,
        dims,
        data: MatData::Numeric { real, imag },
    })
}

fn decode_numeric_storage(
    element: &TaggedData,
    class: MatClass,
    endian: Endian,
) -> BuiltinResult<NumericStorage> {
    let values = decode_numeric_values(element, endian)?;
    match class {
        MatClass::Double => Ok(NumericStorage::F64(values)),
        MatClass::Single => Ok(NumericStorage::F32(
            values.into_iter().map(|value| value as f32).collect(),
        )),
        _ => Err(load_error("load: unsupported floating numeric class")),
    }
}

fn class_is_integer(class: MatClass) -> bool {
    matches!(
        class,
        MatClass::Int8
            | MatClass::UInt8
            | MatClass::Int16
            | MatClass::UInt16
            | MatClass::Int32
            | MatClass::UInt32
            | MatClass::Int64
            | MatClass::UInt64
    )
}

fn decode_integer_storage(
    elem: &TaggedData,
    class: MatClass,
    endian: Endian,
) -> BuiltinResult<IntegerStorage> {
    let expected_type = match class {
        MatClass::Int8 => MI_INT8,
        MatClass::UInt8 => MI_UINT8,
        MatClass::Int16 => MI_INT16,
        MatClass::UInt16 => MI_UINT16,
        MatClass::Int32 => MI_INT32,
        MatClass::UInt32 => MI_UINT32,
        MatClass::Int64 => MI_INT64,
        MatClass::UInt64 => MI_UINT64,
        _ => return Err(load_error("load: expected an integer MAT class")),
    };
    if elem.data_type != expected_type {
        return Err(load_error(format!(
            "load: integer MAT class payload type {} does not match class {:?}",
            elem.data_type, class
        )));
    }

    let data = &elem.data;
    match class {
        MatClass::Int8 => Ok(IntegerStorage::I8(
            data.iter().map(|value| *value as i8).collect(),
        )),
        MatClass::UInt8 => Ok(IntegerStorage::U8(data.clone())),
        MatClass::Int16 => {
            ensure_data_width(data, 2, "integer int16 data")?;
            Ok(IntegerStorage::I16(
                data.chunks_exact(2)
                    .map(|chunk| endian.read_i16(chunk))
                    .collect(),
            ))
        }
        MatClass::UInt16 => {
            ensure_data_width(data, 2, "integer uint16 data")?;
            Ok(IntegerStorage::U16(
                data.chunks_exact(2)
                    .map(|chunk| endian.read_u16(chunk))
                    .collect(),
            ))
        }
        MatClass::Int32 => {
            ensure_data_width(data, 4, "integer int32 data")?;
            Ok(IntegerStorage::I32(
                data.chunks_exact(4)
                    .map(|chunk| endian.read_i32(chunk))
                    .collect(),
            ))
        }
        MatClass::UInt32 => {
            ensure_data_width(data, 4, "integer uint32 data")?;
            Ok(IntegerStorage::U32(
                data.chunks_exact(4)
                    .map(|chunk| endian.read_u32(chunk))
                    .collect(),
            ))
        }
        MatClass::Int64 => {
            ensure_data_width(data, 8, "integer int64 data")?;
            Ok(IntegerStorage::I64(
                data.chunks_exact(8)
                    .map(|chunk| endian.read_i64(chunk))
                    .collect(),
            ))
        }
        MatClass::UInt64 => {
            ensure_data_width(data, 8, "integer uint64 data")?;
            Ok(IntegerStorage::U64(
                data.chunks_exact(8)
                    .map(|chunk| endian.read_u64(chunk))
                    .collect(),
            ))
        }
        _ => Err(load_error("load: expected an integer MAT class")),
    }
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
    let element_count = dims
        .iter()
        .try_fold(1usize, |product, extent| product.checked_mul(*extent))
        .ok_or_else(|| load_error("load: struct array shape exceeds platform limits"))?;
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

    let value_count = field_count
        .checked_mul(element_count)
        .ok_or_else(|| load_error("load: struct array field count exceeds platform limits"))?;
    let mut field_values = Vec::with_capacity(value_count);
    for _ in 0..value_count {
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
    logical: bool,
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
    let values = match decode_sparse_integer_storage(&real_elem, endian)? {
        Some(storage) => NumericStorage::from(storage),
        None if real_elem.data_type == MI_SINGLE => NumericStorage::F32(
            decode_numeric_values(&real_elem, endian)?
                .into_iter()
                .map(|value| value as f32)
                .collect(),
        ),
        None => NumericStorage::F64(decode_numeric_values(&real_elem, endian)?),
    };

    Ok(MatArray {
        class: MatClass::Sparse,
        dims,
        data: MatData::Sparse {
            rows,
            cols,
            col_ptrs,
            row_indices,
            values,
            logical,
        },
    })
}

fn decode_sparse_integer_storage(
    elem: &TaggedData,
    endian: Endian,
) -> BuiltinResult<Option<IntegerStorage>> {
    let class = match elem.data_type {
        MI_INT8 => MatClass::Int8,
        MI_UINT8 => MatClass::UInt8,
        MI_INT16 => MatClass::Int16,
        MI_UINT16 => MatClass::UInt16,
        MI_INT32 => MatClass::Int32,
        MI_UINT32 => MatClass::UInt32,
        MI_INT64 => MatClass::Int64,
        MI_UINT64 => MatClass::UInt64,
        _ => return Ok(None),
    };
    decode_integer_storage(elem, class, endian).map(Some)
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
        MatData::Numeric { real, imag } => {
            let len = real.len();
            if let Some(imag) = imag {
                if imag.len() != len || imag.numeric_dtype() != real.numeric_dtype() {
                    return Err(load_error(
                        "load: complex data has mismatched real/imaginary components",
                    ));
                }
                return match (real, imag) {
                    (NumericStorage::F64(real), NumericStorage::F64(imag)) => {
                        if len == 1 {
                            Ok(Value::Complex(real[0], imag[0]))
                        } else {
                            let pairs = real.into_iter().zip(imag).collect();
                            ComplexTensor::new(pairs, array.dims)
                                .map(Value::ComplexTensor)
                                .map_err(|error| load_error(format!("load: {error}")))
                        }
                    }
                    (NumericStorage::F32(real), NumericStorage::F32(imag)) => {
                        let pairs = real.into_iter().zip(imag).collect();
                        ComplexTensor::from_f32(pairs, array.dims)
                            .map(Value::ComplexTensor)
                            .map_err(|error| load_error(format!("load: {error}")))
                    }
                    (real, imag) => {
                        let real = real
                            .into_integer_storage()
                            .map_err(|_| load_error("load: unsupported complex numeric storage"))?;
                        let imag = imag
                            .into_integer_storage()
                            .map_err(|_| load_error("load: unsupported complex numeric storage"))?;
                        let storage = IntegerComplexStorage::new(real, imag)
                            .map_err(|error| load_error(format!("load: {error}")))?;
                        ComplexTensor::new_integer(storage, array.dims)
                            .map(Value::ComplexTensor)
                            .map_err(|error| load_error(format!("load: {error}")))
                    }
                };
            }
            match real {
                NumericStorage::F64(real) if len == 1 => Ok(Value::Num(real[0])),
                NumericStorage::F64(real) => Tensor::new(real, array.dims)
                    .map(Value::Tensor)
                    .map_err(|error| load_error(format!("load: {error}"))),
                NumericStorage::F32(real) => Tensor::from_f32(real, array.dims)
                    .map(Value::Tensor)
                    .map_err(|error| load_error(format!("load: {error}"))),
                storage => {
                    let storage = storage
                        .into_integer_storage()
                        .map_err(|_| load_error("load: unsupported real numeric storage"))?;
                    if len == 1 {
                        Ok(Value::Int(integer_storage_scalar(&storage, 0)))
                    } else {
                        Tensor::new_integer(storage, array.dims)
                            .map(Value::Tensor)
                            .map_err(|error| load_error(format!("load: {error}")))
                    }
                }
            }
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
            let mut chars = Vec::with_capacity(data.len());
            for code in data {
                let ch = char::from_u32(code as u32).unwrap_or('\u{FFFD}');
                chars.push(ch);
            }
            let char_array = CharArray::from_column_major(chars, array.dims.clone())
                .map_err(|e| load_error(format!("load: {e}")))?;
            Ok(Value::CharArray(char_array))
        }
        MatData::Cell { elements } => {
            if let Some(strings) = cell_elements_to_strings(&elements) {
                let string_array = StringArray::new(strings, array.dims.clone())
                    .map_err(|e| load_error(format!("load: {e}")))?;
                return Ok(Value::StringArray(string_array));
            }
            let expected = array
                .dims
                .iter()
                .try_fold(1usize, |product, &dim| product.checked_mul(dim))
                .ok_or_else(|| load_error("load: cell array shape exceeds platform limits"))?;
            if elements.len() != expected {
                return Err(load_error("load: cell array element count mismatch"));
            }
            let mut converted = Vec::with_capacity(elements.len());
            for elem in elements {
                converted.push(mat_array_to_value(elem)?);
            }
            CellArray::from_column_major(converted, array.dims.clone())
                .map(Value::Cell)
                .map_err(|err| load_error(format!("load: {err}")))
        }
        MatData::Struct {
            field_names,
            field_values,
        } => {
            let element_count = array
                .dims
                .iter()
                .try_fold(1usize, |product, extent| product.checked_mul(*extent))
                .ok_or_else(|| load_error("load: struct array shape exceeds platform limits"))?;
            let expected = field_names
                .len()
                .checked_mul(element_count)
                .ok_or_else(|| load_error("load: struct field storage exceeds platform limits"))?;
            if field_values.len() != expected {
                return Err(load_error("load: struct field metadata is inconsistent"));
            }
            let values = field_values
                .into_iter()
                .map(mat_array_to_value)
                .collect::<BuiltinResult<Vec<_>>>()?;
            StructArray::normalize_field_major(field_names, values, array.dims.clone())
                .map_err(load_error)
        }
        MatData::Sparse {
            rows,
            cols,
            col_ptrs,
            row_indices,
            values,
            logical,
        } => {
            let sparse = if logical {
                let mut filtered_col_ptrs = Vec::with_capacity(cols.saturating_add(1));
                let mut filtered_row_indices = Vec::new();
                filtered_col_ptrs.push(0);
                for col in 0..cols {
                    for index in col_ptrs[col]..col_ptrs[col + 1] {
                        let value = values.value_at(index).ok_or_else(|| {
                            load_error("load: logical sparse value count does not match structure")
                        })?;
                        if !value.is_zero() {
                            filtered_row_indices.push(row_indices[index]);
                        }
                    }
                    filtered_col_ptrs.push(filtered_row_indices.len());
                }
                SparseTensor::new_logical(rows, cols, filtered_col_ptrs, filtered_row_indices)
            } else {
                match values {
                    NumericStorage::F64(values) => {
                        SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
                    }
                    NumericStorage::F32(values) => {
                        SparseTensor::new_f32(rows, cols, col_ptrs, row_indices, values)
                    }
                    storage => SparseTensor::new_integer(
                        rows,
                        cols,
                        col_ptrs,
                        row_indices,
                        storage
                            .into_integer_storage()
                            .expect("non-floating sparse MAT storage is integer"),
                    ),
                }
            }
            .map_err(|e| load_error(format!("load: {e}")))?;
            Ok(Value::SparseTensor(sparse))
        }
    }
}

fn integer_storage_scalar(storage: &IntegerStorage, index: usize) -> IntValue {
    match storage {
        IntegerStorage::I8(values) => IntValue::I8(values[index]),
        IntegerStorage::I16(values) => IntValue::I16(values[index]),
        IntegerStorage::I32(values) => IntValue::I32(values[index]),
        IntegerStorage::I64(values) => IntValue::I64(values[index]),
        IntegerStorage::U8(values) => IntValue::U8(values[index]),
        IntegerStorage::U16(values) => IntValue::U16(values[index]),
        IntegerStorage::U32(values) => IntValue::U32(values[index]),
        IntegerStorage::U64(values) => IntValue::U64(values[index]),
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
                let inner = handle;
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
#[path = "load/tests.rs"]
pub(crate) mod tests;
