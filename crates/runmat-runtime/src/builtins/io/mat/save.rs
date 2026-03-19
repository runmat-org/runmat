//! MATLAB-compatible `save` builtin for RunMat.

use std::collections::HashSet;
use std::io::{BufWriter, Cursor, Write};
use std::path::{Path, PathBuf};

use futures::future::LocalBoxFuture;
use regex::Regex;
use runmat_builtins::{CharArray, StructValue, Value};
use runmat_filesystem::File;
use runmat_macros::runtime_builtin;

use super::format::{
    MatArray, MatClass, MatData, FLAG_COMPLEX, FLAG_LOGICAL, MAT_HEADER_LEN, MI_DOUBLE, MI_INT32,
    MI_INT8, MI_MATRIX, MI_UINT16, MI_UINT32, MI_UINT8,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::workspace;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::mat::save")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "save",
    op_kind: GpuOpKind::Custom("io-save"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs synchronous host I/O; no GPU execution path is involved.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::mat::save")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "save",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Sink operation that terminates fusion graphs before serialisation.",
};

#[runtime_builtin(
    name = "save",
    category = "io/mat",
    summary = "Persist workspace variables to a MAT-file.",
    keywords = "save,mat,workspace",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::save_type),
    builtin_path = "crate::builtins::io::mat::save"
)]
async fn save_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let mut host_args = Vec::with_capacity(args.len());
    for value in &args {
        host_args.push(gather_if_needed_async(value).await?);
    }

    let default_path = Value::from("matlab.mat");
    let (mut path_value, option_start, used_default) = match host_args.first() {
        Some(first) if option_token(first)?.is_some() => (default_path, 0usize, true),
        Some(first) => (first.clone(), 1usize, false),
        None => (default_path, 0usize, true),
    };

    if used_default {
        if let Ok(override_path) = std::env::var("RUNMAT_SAVE_DEFAULT_PATH") {
            path_value = Value::from(override_path);
        }
    }

    let request = parse_arguments(&host_args[option_start..]).await?;
    if request.append {
        return Err(save_error("save: -append is not supported yet"));
    }

    let mut workspace_entries: Option<Vec<(String, Value)>> = None;
    let mut entries: Vec<(String, Value)> = Vec::new();

    if request.variables.is_empty()
        && request.structs.is_empty()
        && request.regex_patterns.is_empty()
    {
        let snapshot = ensure_workspace_entries(&mut workspace_entries).await?;
        entries.extend(snapshot.iter().cloned());
    } else {
        for name in &request.variables {
            if let Some(snapshot) = workspace_entries.as_ref() {
                if let Some(value) = find_in_entries(snapshot, name) {
                    entries.push((name.clone(), value));
                    continue;
                }
            }
            let value = lookup_workspace(name).await?;
            entries.push((name.clone(), value));
        }

        for struct_req in &request.structs {
            let value = if let Some(snapshot) = workspace_entries.as_ref() {
                find_in_entries(snapshot, &struct_req.source)
            } else {
                None
            };
            let value = match value {
                Some(val) => val,
                None => lookup_workspace(&struct_req.source).await?,
            };

            let struct_value = match value {
                Value::Struct(s) => s,
                _ => {
                    return Err(save_error(format!(
                        "save: variable '{}' is not a struct",
                        struct_req.source
                    )))
                }
            };
            append_struct_fields(
                &struct_req.source,
                &struct_value,
                &struct_req.fields,
                &mut entries,
            )
            .await?;
        }

        if !request.regex_patterns.is_empty() {
            let snapshot = ensure_workspace_entries(&mut workspace_entries).await?;
            let mut patterns = Vec::with_capacity(request.regex_patterns.len());
            for pattern in &request.regex_patterns {
                let regex = Regex::new(pattern).map_err(|err| {
                    save_error_with_source(
                        format!("save: invalid regular expression '{pattern}': {err}"),
                        err,
                    )
                })?;
                patterns.push(regex);
            }
            let mut matched = 0usize;
            for (name, value) in snapshot.iter() {
                if patterns.iter().any(|regex| regex.is_match(name)) {
                    entries.push((name.clone(), value.clone()));
                    matched += 1;
                }
            }
            if matched == 0 {
                return Err(save_error("save: no variables matched '-regexp' patterns"));
            }
        }
    }

    if entries.is_empty() {
        return Err(save_error("save: no variables selected"));
    }

    // Deduplicate while preserving the last occurrence for MATLAB compatibility
    let mut seen = HashSet::new();
    let mut unique_entries = Vec::new();
    for (name, value) in entries.into_iter().rev() {
        if seen.insert(name.clone()) {
            unique_entries.push((name, value));
        }
    }
    unique_entries.reverse();

    let mut mat_vars = Vec::with_capacity(unique_entries.len());
    for (name, value) in unique_entries {
        mat_vars.push(MatVar {
            name,
            array: convert_value(value).await?,
        });
    }

    let path = normalise_path(&path_value)?;
    write_mat_file(&path, &mat_vars)?;

    Ok(Value::Num(0.0))
}

struct StructRequest {
    source: String,
    fields: Option<Vec<String>>,
}

struct SaveRequest {
    variables: Vec<String>,
    structs: Vec<StructRequest>,
    regex_patterns: Vec<String>,
    append: bool,
}

const BUILTIN_NAME: &str = "save";

fn save_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn save_error_with_source(
    message: impl Into<String>,
    source: impl std::fmt::Display,
) -> RuntimeError {
    let source = std::io::Error::new(std::io::ErrorKind::Other, source.to_string());
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source)
        .build()
}

async fn parse_arguments(values: &[Value]) -> BuiltinResult<SaveRequest> {
    let mut variables = Vec::new();
    let mut structs = Vec::new();
    let mut regex_patterns = Vec::new();
    let mut append = false;

    let mut idx = 0;
    while idx < values.len() {
        if let Some(flag) = option_token(&values[idx])? {
            match flag.as_str() {
                "-append" => append = true,
                "-struct" => {
                    idx += 1;
                    if idx >= values.len() {
                        return Err(save_error(
                            "save: '-struct' requires a struct variable name",
                        ));
                    }
                    let struct_names = extract_names(&values[idx]).await?;
                    if struct_names.len() != 1 {
                        return Err(save_error(
                            "save: '-struct' requires a single struct variable name",
                        ));
                    }
                    let source = struct_names.into_iter().next().unwrap();
                    idx += 1;
                    let mut field_names = Vec::new();
                    while idx < values.len() {
                        if option_token(&values[idx])?.is_some() {
                            break;
                        }
                        let names = extract_names(&values[idx]).await?;
                        if names.is_empty() {
                            break;
                        }
                        field_names.extend(names);
                        idx += 1;
                    }
                    idx -= 1; // compensate for loop increment
                    let fields = if field_names.is_empty() {
                        None
                    } else {
                        Some(field_names)
                    };
                    structs.push(StructRequest { source, fields });
                }
                "-regexp" => {
                    idx += 1;
                    if idx >= values.len() {
                        return Err(save_error("save: '-regexp' requires at least one pattern"));
                    }
                    let mut patterns = Vec::new();
                    while idx < values.len() {
                        if option_token(&values[idx])?.is_some() {
                            break;
                        }
                        let names = extract_names(&values[idx]).await?;
                        if names.is_empty() {
                            return Err(save_error(
                                "save: '-regexp' requires pattern strings or character rows",
                            ));
                        }
                        patterns.extend(names);
                        idx += 1;
                    }
                    if patterns.is_empty() {
                        return Err(save_error("save: '-regexp' requires at least one pattern"));
                    }
                    idx -= 1;
                    regex_patterns.extend(patterns);
                }
                other => {
                    return Err(save_error(format!("save: unsupported option '{other}'")));
                }
            }
        } else {
            let names = extract_names(&values[idx]).await?;
            variables.extend(names);
        }
        idx += 1;
    }

    Ok(SaveRequest {
        variables,
        structs,
        regex_patterns,
        append,
    })
}

async fn ensure_workspace_entries(
    cache: &mut Option<Vec<(String, Value)>>,
) -> BuiltinResult<&Vec<(String, Value)>> {
    if cache.is_none() {
        let entries = collect_workspace_entries().await?;
        *cache = Some(entries);
    }
    Ok(cache.as_ref().unwrap())
}

async fn collect_workspace_entries() -> BuiltinResult<Vec<(String, Value)>> {
    let snapshot =
        workspace::snapshot().ok_or_else(|| save_error("save: workspace state unavailable"))?;
    let mut entries = Vec::with_capacity(snapshot.len());
    for (name, value) in snapshot {
        let gathered = gather_if_needed_async(&value).await?;
        entries.push((name, gathered));
    }
    Ok(entries)
}

fn find_in_entries(entries: &[(String, Value)], name: &str) -> Option<Value> {
    entries
        .iter()
        .find(|(entry_name, _)| entry_name == name)
        .map(|(_, value)| value.clone())
}

fn option_token(value: &Value) -> BuiltinResult<Option<String>> {
    if let Some(token) = value_to_string_scalar(value) {
        if token.starts_with('-') {
            return Ok(Some(token.to_ascii_lowercase()));
        }
    }
    Ok(None)
}

async fn extract_names(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::CharArray(ca) => {
            let rows = char_array_rows_as_strings(ca);
            if rows.is_empty() && ca.rows > 0 {
                return Err(save_error(
                    "save: character arrays used for variable names must contain non-empty rows",
                ));
            }
            Ok(rows)
        }
        Value::StringArray(sa) => {
            let mut names = Vec::with_capacity(sa.data.len());
            for s in &sa.data {
                names.push(s.clone());
            }
            Ok(names)
        }
        Value::Cell(ca) => {
            let mut names = Vec::with_capacity(ca.data.len());
            for handle in &ca.data {
                let inner = unsafe { &*handle.as_raw() };
                let text = value_to_string_scalar(inner).ok_or_else(|| {
                    save_error(
                        "save: cell arrays must contain string scalars when specifying variable names",
                    )
                })?;
                names.push(text);
            }
            Ok(names)
        }
        other => {
            // Gather once, then require a string-like scalar to avoid infinite recursion.
            let gathered = gather_if_needed_async(other).await?;
            if let Some(text) = value_to_string_scalar(&gathered) {
                return Ok(vec![text]);
            }
            Err(save_error(
                "save: variable names must be strings, character arrays, string arrays, or cell arrays of strings",
            ))
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

async fn append_struct_fields(
    struct_name: &str,
    value: &StructValue,
    fields: &Option<Vec<String>>,
    out: &mut Vec<(String, Value)>,
) -> BuiltinResult<()> {
    if let Some(ref names) = fields {
        for field in names {
            match value.fields.get(field) {
                Some(val) => {
                    let gathered = gather_if_needed_async(val).await?;
                    out.push((field.clone(), gathered));
                }
                None => {
                    return Err(save_error(format!(
                        "save: struct '{}' does not have a field named '{}'",
                        struct_name, field
                    )));
                }
            }
        }
    } else {
        for (field, val) in &value.fields {
            let gathered = gather_if_needed_async(val).await?;
            out.push((field.clone(), gathered));
        }
    }
    Ok(())
}

fn char_array_rows_as_strings(ca: &CharArray) -> Vec<String> {
    let mut rows = Vec::with_capacity(ca.rows);
    for row in 0..ca.rows {
        let mut buffer = String::with_capacity(ca.cols);
        for col in 0..ca.cols {
            let idx = row * ca.cols + col;
            buffer.push(ca.data[idx]);
        }
        let trimmed = buffer.trim_end_matches([' ', '\0']).to_string();
        if !trimmed.is_empty() {
            rows.push(trimmed);
        }
    }
    rows
}

async fn lookup_workspace(name: &str) -> BuiltinResult<Value> {
    let value = workspace::lookup(name).ok_or_else(|| {
        save_error(format!(
            "save: variable '{}' was not found in the workspace",
            name
        ))
    })?;
    gather_if_needed_async(&value).await
}

fn normalise_path(path: &Value) -> BuiltinResult<PathBuf> {
    let raw = value_to_string_scalar(path)
        .ok_or_else(|| save_error("save: filename must be a character vector or string scalar"))?;
    let mut path = PathBuf::from(raw);
    if path.extension().is_none() {
        path.set_extension("mat");
    }
    Ok(path)
}

struct MatVar {
    name: String,
    array: MatArray,
}

fn canonical_dims(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![1, shape[0]],
        _ => shape.to_vec(),
    }
}

fn convert_value(value: Value) -> LocalBoxFuture<'static, BuiltinResult<MatArray>> {
    Box::pin(async move {
        match value {
            Value::Num(n) => Ok(MatArray {
                class: MatClass::Double,
                dims: vec![1, 1],
                data: MatData::Double {
                    real: vec![n],
                    imag: None,
                },
            }),
            Value::Int(i) => Ok(MatArray {
                class: MatClass::Double,
                dims: vec![1, 1],
                data: MatData::Double {
                    real: vec![i.to_f64()],
                    imag: None,
                },
            }),
            Value::Bool(b) => Ok(MatArray {
                class: MatClass::Logical,
                dims: vec![1, 1],
                data: MatData::Logical {
                    data: vec![if b { 1 } else { 0 }],
                },
            }),
            Value::Tensor(t) => Ok(MatArray {
                class: MatClass::Double,
                dims: canonical_dims(&t.shape),
                data: MatData::Double {
                    real: t.data,
                    imag: None,
                },
            }),
            Value::Complex(re, im) => Ok(MatArray {
                class: MatClass::Double,
                dims: vec![1, 1],
                data: MatData::Double {
                    real: vec![re],
                    imag: Some(vec![im]),
                },
            }),
            Value::ComplexTensor(t) => {
                let mut real = Vec::with_capacity(t.data.len());
                let mut imag = Vec::with_capacity(t.data.len());
                for (re, im) in &t.data {
                    real.push(*re);
                    imag.push(*im);
                }
                Ok(MatArray {
                    class: MatClass::Double,
                    dims: canonical_dims(&t.shape),
                    data: MatData::Double {
                        real,
                        imag: Some(imag),
                    },
                })
            }
            Value::LogicalArray(la) => Ok(MatArray {
                class: MatClass::Logical,
                dims: canonical_dims(&la.shape),
                data: MatData::Logical { data: la.data },
            }),
            Value::CharArray(ca) => Ok(MatArray {
                class: MatClass::Char,
                dims: vec![ca.rows, ca.cols],
                data: MatData::Char {
                    data: char_array_to_utf16(&ca),
                },
            }),
            Value::String(s) => Ok(MatArray {
                class: MatClass::Char,
                dims: vec![1, s.chars().count()],
                data: MatData::Char {
                    data: s.encode_utf16().collect(),
                },
            }),
            Value::StringArray(sa) => {
                if sa.data.len() == 1 {
                    return convert_value(Value::String(sa.data[0].clone())).await;
                }
                let mut elements = Vec::with_capacity(sa.data.len());
                for text in &sa.data {
                    elements.push(MatArray {
                        class: MatClass::Char,
                        dims: vec![1, text.chars().count()],
                        data: MatData::Char {
                            data: text.encode_utf16().collect(),
                        },
                    });
                }
                Ok(MatArray {
                    class: MatClass::Cell,
                    dims: canonical_dims(&sa.shape),
                    data: MatData::Cell { elements },
                })
            }
            Value::Cell(cell) => {
                let mut elements = Vec::with_capacity(cell.data.len());
                for col in 0..cell.cols {
                    for row in 0..cell.rows {
                        let idx = row * cell.cols + col;
                        let element = unsafe { &*cell.data[idx].as_raw() };
                        let gathered = gather_if_needed_async(element).await?;
                        elements.push(convert_value(gathered).await?);
                    }
                }
                Ok(MatArray {
                    class: MatClass::Cell,
                    dims: vec![cell.rows, cell.cols],
                    data: MatData::Cell { elements },
                })
            }
            Value::Struct(struct_value) => {
                let mut field_names: Vec<String> = struct_value.fields.keys().cloned().collect();
                field_names.sort();
                let mut field_values = Vec::with_capacity(field_names.len());
                for field in &field_names {
                    let val = struct_value.fields.get(field).ok_or_else(|| {
                        save_error(format!("save: missing struct field '{field}'"))
                    })?;
                    let gathered = gather_if_needed_async(val).await?;
                    field_values.push(convert_value(gathered).await?);
                }
                Ok(MatArray {
                    class: MatClass::Struct,
                    dims: vec![1, 1],
                    data: MatData::Struct {
                        field_names,
                        field_values,
                    },
                })
            }
            Value::GpuTensor(handle) => {
                let gathered = gather_if_needed_async(&Value::GpuTensor(handle)).await?;
                convert_value(gathered).await
            }
            unsupported => Err(save_error(format!(
                "save: value of type '{:?}' is not supported",
                unsupported
            ))),
        }
    })
}

fn char_array_to_utf16(ca: &CharArray) -> Vec<u16> {
    let mut data = Vec::with_capacity(ca.rows * ca.cols);
    for col in 0..ca.cols {
        for row in 0..ca.rows {
            let idx = row * ca.cols + col;
            data.push(ca.data[idx] as u16);
        }
    }
    data
}

fn write_mat_file(path: &Path, vars: &[MatVar]) -> BuiltinResult<()> {
    let file = File::create(path).map_err(|e| {
        save_error_with_source(format!("save: failed to open '{}': {e}", path.display()), e)
    })?;
    let mut writer = BufWriter::new(file);

    let mut header = [0u8; MAT_HEADER_LEN];
    let desc = b"MATLAB 5.0 MAT-file, RunMat save";
    for (i, byte) in desc.iter().enumerate() {
        header[i] = *byte;
    }
    header[124] = 0x00;
    header[125] = 0x01;
    header[126] = b'I';
    header[127] = b'M';
    writer
        .write_all(&header)
        .map_err(|e| save_error_with_source(format!("save: failed to write header: {e}"), e))?;

    for var in vars {
        let matrix_bytes = build_matrix_bytes(&var.array, Some(&var.name))?;
        write_tagged(&mut writer, MI_MATRIX, &matrix_bytes)?;
    }

    writer
        .flush()
        .map_err(|e| save_error_with_source(format!("save: flush failed: {e}"), e))
}

pub async fn encode_workspace_to_mat_bytes(entries: &[(String, Value)]) -> BuiltinResult<Vec<u8>> {
    let mut mat_vars = Vec::with_capacity(entries.len());
    for (name, value) in entries {
        mat_vars.push(MatVar {
            name: name.clone(),
            array: convert_value(value.clone()).await?,
        });
    }
    write_mat_bytes(&mat_vars)
}

fn write_mat_bytes(vars: &[MatVar]) -> BuiltinResult<Vec<u8>> {
    let mut writer = BufWriter::new(Cursor::new(Vec::<u8>::new()));

    let mut header = [0u8; MAT_HEADER_LEN];
    let desc = b"MATLAB 5.0 MAT-file, RunMat save";
    for (i, byte) in desc.iter().enumerate() {
        header[i] = *byte;
    }
    header[124] = 0x00;
    header[125] = 0x01;
    header[126] = b'I';
    header[127] = b'M';
    writer
        .write_all(&header)
        .map_err(|e| save_error_with_source(format!("save: failed to write header: {e}"), e))?;

    for var in vars {
        let matrix_bytes = build_matrix_bytes(&var.array, Some(&var.name))?;
        write_tagged(&mut writer, MI_MATRIX, &matrix_bytes)?;
    }

    writer
        .flush()
        .map_err(|e| save_error_with_source(format!("save: flush failed: {e}"), e))?;
    Ok(writer
        .into_inner()
        .map_err(|e| save_error_with_source("save: failed to finalize MAT bytes", e))?
        .into_inner())
}

fn build_matrix_bytes(array: &MatArray, name: Option<&str>) -> BuiltinResult<Vec<u8>> {
    let mut buf = Vec::new();

    let (flags0, flags1) = match &array.data {
        MatData::Double { imag, .. } => {
            let mut f0 = array.class.class_code();
            if imag.is_some() {
                f0 |= FLAG_COMPLEX;
            }
            (f0, 0u32)
        }
        MatData::Logical { .. } => ((array.class.class_code()) | FLAG_LOGICAL, 0u32),
        _ => (array.class.class_code(), 0u32),
    };

    let mut flags = Vec::with_capacity(8);
    flags.extend_from_slice(&flags0.to_le_bytes());
    flags.extend_from_slice(&flags1.to_le_bytes());
    write_subelement(&mut buf, MI_UINT32, &flags);

    let mut dims_bytes = Vec::with_capacity(array.dims.len() * 4);
    for &dim in &array.dims {
        dims_bytes.extend_from_slice(&((dim as i32).max(0)).to_le_bytes());
    }
    write_subelement(&mut buf, MI_INT32, &dims_bytes);

    let name_bytes = name.unwrap_or("").as_bytes();
    write_subelement(&mut buf, MI_INT8, name_bytes);

    match &array.data {
        MatData::Double { real, imag } => {
            let mut real_bytes = Vec::with_capacity(real.len() * 8);
            for v in real {
                real_bytes.extend_from_slice(&v.to_le_bytes());
            }
            write_subelement(&mut buf, MI_DOUBLE, &real_bytes);
            if let Some(imag) = imag {
                let mut imag_bytes = Vec::with_capacity(imag.len() * 8);
                for v in imag {
                    imag_bytes.extend_from_slice(&v.to_le_bytes());
                }
                write_subelement(&mut buf, MI_DOUBLE, &imag_bytes);
            }
        }
        MatData::Logical { data } => {
            write_subelement(&mut buf, MI_UINT8, data);
        }
        MatData::Char { data } => {
            let mut bytes = Vec::with_capacity(data.len() * 2);
            for code in data {
                bytes.extend_from_slice(&code.to_le_bytes());
            }
            write_subelement(&mut buf, MI_UINT16, &bytes);
        }
        MatData::Cell { elements } => {
            for elem in elements {
                let elem_bytes = build_matrix_bytes(elem, None)?;
                write_subelement(&mut buf, MI_MATRIX, &elem_bytes);
            }
        }
        MatData::Struct {
            field_names,
            field_values,
        } => {
            if array.dims != [1, 1] {
                return Err(save_error("save: struct arrays are not supported"));
            }
            let max_len = field_names
                .iter()
                .map(|n| n.len())
                .max()
                .unwrap_or(0)
                .max(1);
            let len_bytes = (max_len as i32).to_le_bytes();
            write_subelement(&mut buf, MI_INT32, &len_bytes);
            let mut names_bytes = Vec::with_capacity(max_len * field_names.len());
            for name in field_names {
                let bytes = name.as_bytes();
                for i in 0..max_len {
                    let b = if i < bytes.len() { bytes[i] } else { 0 };
                    names_bytes.push(b);
                }
            }
            write_subelement(&mut buf, MI_INT8, &names_bytes);
            for value in field_values {
                let value_bytes = build_matrix_bytes(value, None)?;
                write_subelement(&mut buf, MI_MATRIX, &value_bytes);
            }
        }
    }

    Ok(buf)
}

fn write_tagged<W: Write>(writer: &mut W, data_type: u32, data: &[u8]) -> BuiltinResult<()> {
    if data.len() > u32::MAX as usize {
        return Err(save_error("save: data too large for MAT-file"));
    }
    writer
        .write_all(&data_type.to_le_bytes())
        .map_err(|e| save_error_with_source(format!("save: write failed: {e}"), e))?;
    writer
        .write_all(&(data.len() as u32).to_le_bytes())
        .map_err(|e| save_error_with_source(format!("save: write failed: {e}"), e))?;
    writer
        .write_all(data)
        .map_err(|e| save_error_with_source(format!("save: write failed: {e}"), e))?;
    let padding = (8 - (data.len() % 8)) % 8;
    if padding != 0 {
        let pad = [0u8; 8];
        writer
            .write_all(&pad[..padding])
            .map_err(|e| save_error_with_source(format!("save: write failed: {e}"), e))?;
    }
    Ok(())
}

fn write_subelement(buf: &mut Vec<u8>, data_type: u32, data: &[u8]) {
    buf.extend_from_slice(&data_type.to_le_bytes());
    buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
    buf.extend_from_slice(data);
    let padding = (8 - (data.len() % 8)) % 8;
    if padding != 0 {
        buf.extend(std::iter::repeat_n(0u8, padding));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::workspace::WorkspaceResolver;
    use futures::executor::block_on;
    use once_cell::sync::OnceCell;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{StringArray, Tensor};
    use runmat_thread_local::runmat_thread_local;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use tempfile::tempdir;

    runmat_thread_local! {
        static TEST_WORKSPACE: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
    }

    fn ensure_test_resolver() {
        workspace::register_workspace_resolver(WorkspaceResolver {
            lookup: |name| TEST_WORKSPACE.with(|slot| slot.borrow().get(name).cloned()),
            snapshot: || {
                let mut entries: Vec<(String, Value)> =
                    TEST_WORKSPACE.with(|slot| slot.borrow().clone().into_iter().collect());
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                entries
            },
            globals: || Vec::new(),
            assign: None,
        });
    }

    fn set_workspace(entries: &[(&str, Value)]) {
        TEST_WORKSPACE.with(|slot| {
            let mut map = HashMap::new();
            for (k, v) in entries {
                map.insert(k.to_string(), v.clone());
            }
            *slot.borrow_mut() = map;
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

    fn lock_env_override() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceCell<Mutex<()>> = OnceCell::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    struct EnvOverride {
        key: &'static str,
    }

    impl EnvOverride {
        fn set(key: &'static str, value: &str) -> Self {
            std::env::set_var(key, value);
            EnvOverride { key }
        }
    }

    impl Drop for EnvOverride {
        fn drop(&mut self) {
            std::env::remove_var(self.key);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn save_numeric_variable() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[("A", Value::Num(42.0))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_numeric.mat");
        let args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("A"),
        ];
        block_on(save_builtin(args)).unwrap();

        let file = File::open(&path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        let array = mat.find_by_name("A").unwrap();
        match array.data() {
            matfile::NumericData::Double { real, .. } => {
                assert_eq!(real, &[42.0]);
            }
            _ => panic!("expected double array"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn save_string_array_variable_names() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[
            ("A", Value::Num(1.0)),
            ("B", Value::Num(2.0)),
            ("C", Value::Num(3.0)),
        ]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("string_array.mat");
        let names = StringArray::new(vec!["A".into(), "B".into()], vec![1, 2]).unwrap();
        let args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::StringArray(names),
        ];
        block_on(save_builtin(args)).unwrap();

        let file = File::open(&path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        assert!(mat.find_by_name("A").is_some());
        assert!(mat.find_by_name("B").is_some());
        assert!(mat.find_by_name("C").is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn save_char_matrix_variable_names() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[
            ("foo", Value::Num(10.0)),
            ("bar", Value::Num(20.0)),
            ("baz", Value::Num(30.0)),
        ]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("char_matrix.mat");
        let chars = CharArray::new("foobar".chars().collect(), 2, 3).unwrap();
        let args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::CharArray(chars),
        ];
        block_on(save_builtin(args)).unwrap();

        let file = File::open(&path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        assert!(mat.find_by_name("foo").is_some());
        assert!(mat.find_by_name("bar").is_some());
        assert!(mat.find_by_name("baz").is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn save_struct_fields() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        let mut opts_struct = StructValue::new();
        opts_struct
            .fields
            .insert("foo".to_string(), Value::Num(1.0));
        opts_struct
            .fields
            .insert("bar".to_string(), Value::Num(2.0));
        set_workspace(&[("opts", Value::Struct(opts_struct))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("struct.mat");
        let args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("-struct"),
            Value::from("opts"),
        ];
        block_on(save_builtin(args)).unwrap();

        let file = File::open(&path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        let foo = mat.find_by_name("bar").unwrap();
        match foo.data() {
            matfile::NumericData::Double { real, .. } => assert_eq!(real, &[2.0]),
            _ => panic!("expected double"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn save_struct_field_selection() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        let mut opts_struct = StructValue::new();
        opts_struct
            .fields
            .insert("foo".to_string(), Value::Num(11.0));
        opts_struct
            .fields
            .insert("bar".to_string(), Value::Num(22.0));
        set_workspace(&[("opts", Value::Struct(opts_struct))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("struct_subset.mat");
        let args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("-struct"),
            Value::from("opts"),
            Value::from("bar"),
        ];
        block_on(save_builtin(args)).unwrap();

        let file = File::open(&path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        assert!(mat.find_by_name("foo").is_none());
        let array = mat.find_by_name("bar").unwrap();
        match array.data() {
            matfile::NumericData::Double { real, .. } => assert_eq!(real, &[22.0]),
            _ => panic!("expected double array"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn save_missing_variable_errors() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[]);
        let result = block_on(save_builtin(vec![
            Value::from("missing.mat"),
            Value::from("x"),
        ]));
        assert_error_contains(result, "variable 'x'");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn save_regex_variable_selection() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[
            ("alpha", Value::Num(1.0)),
            ("beta", Value::Num(2.0)),
            ("gamma", Value::Num(3.0)),
        ]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("regex.mat");
        let args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("-regexp"),
            Value::from("^a"),
            Value::from("ma$"),
        ];
        block_on(save_builtin(args)).unwrap();

        let file = File::open(&path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        assert!(mat.find_by_name("alpha").is_some());
        assert!(mat.find_by_name("gamma").is_some());
        assert!(mat.find_by_name("beta").is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn save_regex_requires_pattern() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[("foo", Value::Num(1.0))]);
        let result = block_on(save_builtin(vec![Value::from("-regexp")]));
        assert_error_contains(result, "'-regexp' requires at least one pattern");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn save_unsupported_option_errors() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[("foo", Value::Num(1.0))]);
        let result = block_on(save_builtin(vec![
            Value::from("text.mat"),
            Value::from("-ascii"),
            Value::from("foo"),
        ]));
        assert_error_contains(result, "unsupported option '-ascii'");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn save_defaults_to_matlab_mat() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        let _lock = lock_env_override();
        set_workspace(&[("answer", Value::Num(7.0))]);
        let dir = tempdir().unwrap();
        let target = dir.path().join("matlab_default.mat");
        let target_str = target.to_string_lossy().to_string();
        let _env = EnvOverride::set("RUNMAT_SAVE_DEFAULT_PATH", &target_str);
        block_on(save_builtin(Vec::new())).unwrap();

        assert!(
            target.exists(),
            "expected {} to be created",
            target.display()
        );
        let file = File::open(&target).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        let array = mat.find_by_name("answer").unwrap();
        match array.data() {
            matfile::NumericData::Double { real, .. } => assert_eq!(real, &[7.0]),
            _ => panic!("expected double array"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn save_struct_without_filename_defaults_to_matlab_mat() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        let _lock = lock_env_override();
        let mut payload_struct = StructValue::new();
        payload_struct
            .fields
            .insert("alpha".to_string(), Value::Num(3.0));
        set_workspace(&[("payload", Value::Struct(payload_struct))]);
        let dir = tempdir().unwrap();
        let target = dir.path().join("matlab_struct.mat");
        let target_str = target.to_string_lossy().to_string();
        let _env = EnvOverride::set("RUNMAT_SAVE_DEFAULT_PATH", &target_str);
        block_on(save_builtin(vec![
            Value::from("-struct"),
            Value::from("payload"),
        ]))
        .unwrap();

        assert!(
            target.exists(),
            "expected {} to be created",
            target.display()
        );
        let file = File::open(&target).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        let field = mat.find_by_name("alpha").unwrap();
        match field.data() {
            matfile::NumericData::Double { real, .. } => assert_eq!(real, &[3.0]),
            _ => panic!("expected double array"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn save_gpu_tensor_roundtrip() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload tensor");
            set_workspace(&[("gpu_data", Value::GpuTensor(handle.clone()))]);

            let dir = tempdir().unwrap();
            let path = dir.path().join("gpu_roundtrip.mat");
            block_on(save_builtin(vec![
                Value::from(path.to_string_lossy().to_string()),
                Value::from("gpu_data"),
            ]))
            .unwrap();

            let file = File::open(&path).unwrap();
            let mat = matfile::MatFile::parse(file).unwrap();
            let array = mat.find_by_name("gpu_data").unwrap();
            match array.data() {
                matfile::NumericData::Double { real, imag } => {
                    assert_eq!(real, &tensor.data);
                    assert!(imag.is_none());
                }
                _ => panic!("expected double array"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn save_wgpu_tensor_roundtrip() {
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

        let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload tensor");
        set_workspace(&[("wgpu_tensor", Value::GpuTensor(handle.clone()))]);

        let dir = tempdir().unwrap();
        let path = dir.path().join("wgpu_roundtrip.mat");
        block_on(save_builtin(vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("wgpu_tensor"),
        ]))
        .unwrap();

        let file = File::open(&path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        let array = mat.find_by_name("wgpu_tensor").unwrap();
        match array.data() {
            matfile::NumericData::Double { real, imag } => {
                assert_eq!(real, &tensor.data);
                assert!(imag.is_none());
            }
            _ => panic!("expected double array"),
        }
    }
}
