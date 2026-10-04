//! MATLAB-compatible `save` builtin for RunMat.

use std::collections::HashSet;
use std::io::{BufWriter, Cursor, Write};
use std::path::{Path, PathBuf};

use futures::future::LocalBoxFuture;
use regex::Regex;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexStorage, IntValue, IntegerStorage, NumericDType, NumericStorage, StructValue,
    Value,
};

use super::authority::MatFsPath;
use super::format::{
    MatArray, MatClass, MatData, FLAG_COMPLEX, FLAG_LOGICAL, MAT_HEADER_LEN, MI_DOUBLE, MI_INT16,
    MI_INT32, MI_INT64, MI_INT8, MI_MATRIX, MI_SINGLE, MI_UINT16, MI_UINT32, MI_UINT64, MI_UINT8,
};
use super::load::read_mat_file_on;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::introspection::object_serialization::prepare_value_for_mat_save;
use crate::workspace;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use runmat_builtins::{
    SAVE_ERROR_FILENAME, SAVE_ERROR_INVALID_ARGUMENT, SAVE_ERROR_INVALID_OPTION, SAVE_ERROR_IO,
    SAVE_ERROR_SELECTION, SAVE_ERROR_UNSUPPORTED, SAVE_ERROR_WORKSPACE,
};

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
    binding_variant = "default",
    builtin_path = "crate::builtins::io::mat::save"
)]
async fn save_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if crate::output_count::current_output_count()
        .or_else(crate::output_context::requested_output_count)
        .is_some_and(|count| count > 0)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::SAVE_STATUS_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    preflight_known_options(&args)?;
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
    let path = normalise_path(&path_value)?;
    let destination = MatFsPath::capture(path).map_err(|err| {
        save_error_with_source(
            &SAVE_ERROR_IO,
            format!("save: failed to resolve MAT path: {err}"),
            err,
        )
    })?;

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
                    return Err(save_error_with(
                        &SAVE_ERROR_SELECTION,
                        format!("save: variable '{}' is not a struct", struct_req.source),
                    ))
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
                        &SAVE_ERROR_INVALID_OPTION,
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
                return Err(save_error_with(
                    &SAVE_ERROR_SELECTION,
                    "save: no variables matched '-regexp' patterns",
                ));
            }
        }
    }

    if entries.is_empty() {
        return Err(save_error_with(
            &SAVE_ERROR_SELECTION,
            "save: no variables selected",
        ));
    }

    let mut unique_entries = deduplicate_entries(entries);
    if request.append {
        unique_entries = append_existing_entries(&destination, unique_entries).await?;
    }

    let mut mat_vars = Vec::with_capacity(unique_entries.len());
    for (name, value) in unique_entries {
        let value = prepare_value_for_mat_save(value).await?;
        mat_vars.push(MatVar {
            name,
            array: convert_value(value).await?,
        });
    }

    write_mat_file(&destination, &mat_vars).await?;

    Ok(Value::Num(0.0))
}

fn preflight_known_options(args: &[Value]) -> BuiltinResult<()> {
    for (index, value) in args.iter().enumerate() {
        let Some(token) = value_to_string_scalar(value) else {
            continue;
        };
        if !token.starts_with('-') {
            continue;
        }
        match runmat_builtins::MatSaveOption::parse(&token) {
            Some(
                runmat_builtins::MatSaveOption::Struct | runmat_builtins::MatSaveOption::Regexp,
            ) if index + 1 == args.len() => {
                let message = if token.eq_ignore_ascii_case("-struct") {
                    "save: '-struct' requires a struct variable name"
                } else {
                    "save: '-regexp' requires at least one pattern"
                };
                return Err(save_error_with(&SAVE_ERROR_INVALID_OPTION, message));
            }
            None => {
                return Err(save_error_with(
                    &SAVE_ERROR_INVALID_OPTION,
                    format!("save: unsupported option '{}'", token.to_ascii_lowercase()),
                ));
            }
            _ => {}
        }
    }
    Ok(())
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

fn save_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn save_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: impl std::fmt::Display,
) -> RuntimeError {
    let source = std::io::Error::new(std::io::ErrorKind::Other, source.to_string());
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

async fn parse_arguments(values: &[Value]) -> BuiltinResult<SaveRequest> {
    let mut variables = Vec::new();
    let mut structs = Vec::new();
    let mut regex_patterns = Vec::new();
    let mut append = false;

    let mut idx = 0;
    while idx < values.len() {
        if let Some(flag) = option_token(&values[idx])? {
            match runmat_builtins::MatSaveOption::parse(&flag) {
                Some(runmat_builtins::MatSaveOption::Append) => append = true,
                Some(runmat_builtins::MatSaveOption::Struct) => {
                    idx += 1;
                    if idx >= values.len() {
                        return Err(save_error_with(
                            &SAVE_ERROR_INVALID_OPTION,
                            "save: '-struct' requires a struct variable name",
                        ));
                    }
                    let struct_names = extract_names(&values[idx]).await?;
                    if struct_names.len() != 1 {
                        return Err(save_error_with(
                            &SAVE_ERROR_INVALID_OPTION,
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
                Some(runmat_builtins::MatSaveOption::Regexp) => {
                    idx += 1;
                    if idx >= values.len() {
                        return Err(save_error_with(
                            &SAVE_ERROR_INVALID_OPTION,
                            "save: '-regexp' requires at least one pattern",
                        ));
                    }
                    let mut patterns = Vec::new();
                    while idx < values.len() {
                        if option_token(&values[idx])?.is_some() {
                            break;
                        }
                        let names = extract_names(&values[idx]).await?;
                        if names.is_empty() {
                            return Err(save_error_with(
                                &SAVE_ERROR_INVALID_OPTION,
                                "save: '-regexp' requires pattern strings or character rows",
                            ));
                        }
                        patterns.extend(names);
                        idx += 1;
                    }
                    if patterns.is_empty() {
                        return Err(save_error_with(
                            &SAVE_ERROR_INVALID_OPTION,
                            "save: '-regexp' requires at least one pattern",
                        ));
                    }
                    idx -= 1;
                    regex_patterns.extend(patterns);
                }
                None => {
                    return Err(save_error_with(
                        &SAVE_ERROR_INVALID_OPTION,
                        format!("save: unsupported option '{flag}'"),
                    ));
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
    let snapshot = workspace::snapshot()
        .ok_or_else(|| save_error_with(&SAVE_ERROR_WORKSPACE, SAVE_ERROR_WORKSPACE.message))?;
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

fn deduplicate_entries(entries: Vec<(String, Value)>) -> Vec<(String, Value)> {
    let mut seen = HashSet::new();
    let mut unique_entries = Vec::new();
    for (name, value) in entries.into_iter().rev() {
        if seen.insert(name.clone()) {
            unique_entries.push((name, value));
        }
    }
    unique_entries.reverse();
    unique_entries
}

async fn append_existing_entries(
    path: &MatFsPath,
    new_entries: Vec<(String, Value)>,
) -> BuiltinResult<Vec<(String, Value)>> {
    let mut entries = read_existing_entries_for_append(path).await?;
    entries.extend(new_entries);
    Ok(deduplicate_entries(entries))
}

async fn read_existing_entries_for_append(path: &MatFsPath) -> BuiltinResult<Vec<(String, Value)>> {
    match path.metadata().await {
        Ok(_) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => {
            return Err(save_error_with_source(
                &SAVE_ERROR_IO,
                format!(
                    "save: failed to inspect existing MAT-file '{}': {err}",
                    path.path().display()
                ),
                err,
            ));
        }
    }
    match read_mat_file_on(path).await {
        Ok(entries) => Ok(entries),
        Err(err) => Err(save_error_with_source(
            &SAVE_ERROR_IO,
            format!(
                "save: failed to read existing MAT-file '{}': {}",
                path.path().display(),
                err.message()
            ),
            err,
        )),
    }
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
                return Err(save_error_with(
                    &SAVE_ERROR_INVALID_ARGUMENT,
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
                let inner = handle;
                let text = value_to_string_scalar(inner).ok_or_else(|| {
                    save_error_with(
                        &SAVE_ERROR_INVALID_ARGUMENT,
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
            Err(save_error_with(
                &SAVE_ERROR_INVALID_ARGUMENT,
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
                    return Err(save_error_with(
                        &SAVE_ERROR_SELECTION,
                        format!(
                            "save: struct '{}' does not have a field named '{}'",
                            struct_name, field
                        ),
                    ));
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
        save_error_with(
            &SAVE_ERROR_SELECTION,
            format!("save: variable '{}' was not found in the workspace", name),
        )
    })?;
    gather_if_needed_async(&value).await
}

fn normalise_path(path: &Value) -> BuiltinResult<PathBuf> {
    let raw = value_to_string_scalar(path).ok_or_else(|| {
        save_error_with(
            &SAVE_ERROR_FILENAME,
            "save: filename must be a character vector or string scalar",
        )
    })?;
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
                data: MatData::Numeric {
                    real: NumericStorage::F64(vec![n]),
                    imag: None,
                },
            }),
            Value::Int(i) => {
                let class = int_value_mat_class(&i);
                Ok(MatArray {
                    class,
                    dims: vec![1, 1],
                    data: MatData::Numeric {
                        real: NumericStorage::from(integer_storage_from_scalar(i)),
                        imag: None,
                    },
                })
            }
            Value::Bool(b) => Ok(MatArray {
                class: MatClass::Logical,
                dims: vec![1, 1],
                data: MatData::Logical {
                    data: vec![if b { 1 } else { 0 }],
                },
            }),
            Value::Tensor(t) => {
                let dims = canonical_dims(&t.shape);
                let storage = t
                    .into_numeric_storage()
                    .map_err(|error| save_error_with(&SAVE_ERROR_IO, format!("save: {error}")))?;
                let class = tensor_dtype_mat_class(storage.numeric_dtype());
                let data = MatData::Numeric {
                    real: storage,
                    imag: None,
                };
                Ok(MatArray { class, dims, data })
            }
            Value::Complex(re, im) => Ok(MatArray {
                class: MatClass::Double,
                dims: vec![1, 1],
                data: MatData::Numeric {
                    real: NumericStorage::F64(vec![re]),
                    imag: Some(NumericStorage::F64(vec![im])),
                },
            }),
            Value::ComplexTensor(t) => {
                let dims = canonical_dims(&t.shape);
                let (class, real, imag) = match t.into_complex_storage() {
                    ComplexStorage::F64(values) => {
                        let (real, imag): (Vec<_>, Vec<_>) = values.into_iter().unzip();
                        (
                            MatClass::Double,
                            NumericStorage::F64(real),
                            NumericStorage::F64(imag),
                        )
                    }
                    ComplexStorage::F32(values) => {
                        let (real, imag): (Vec<_>, Vec<_>) = values.into_iter().unzip();
                        (
                            MatClass::Single,
                            NumericStorage::F32(real),
                            NumericStorage::F32(imag),
                        )
                    }
                    ComplexStorage::Integer(storage) => {
                        let class = integer_storage_mat_class(&storage.real);
                        (
                            class,
                            NumericStorage::from(storage.real),
                            NumericStorage::from(storage.imag),
                        )
                    }
                };
                Ok(MatArray {
                    class,
                    dims,
                    data: MatData::Numeric {
                        real,
                        imag: Some(imag),
                    },
                })
            }
            Value::LogicalArray(la) => Ok(MatArray {
                class: MatClass::Logical,
                dims: canonical_dims(&la.shape),
                data: MatData::Logical {
                    data: la.data.into_vec(),
                },
            }),
            Value::SparseTensor(sparse) => {
                let logical = sparse.is_logical();
                let values = if logical {
                    NumericStorage::U8(vec![1; sparse.nnz()])
                } else if let Some(storage) = sparse.integer_storage() {
                    NumericStorage::from(storage.clone())
                } else if let Some(values) = sparse.as_f32_slice() {
                    NumericStorage::F32(values.to_vec())
                } else {
                    NumericStorage::F64(
                        sparse
                            .as_f64_slice()
                            .expect("double sparse storage must expose f64 values")
                            .to_vec(),
                    )
                };
                Ok(MatArray {
                    class: MatClass::Sparse,
                    dims: vec![sparse.rows, sparse.cols],
                    data: MatData::Sparse {
                        rows: sparse.rows,
                        cols: sparse.cols,
                        col_ptrs: sparse.col_ptrs.into_vec(),
                        row_indices: sparse.row_indices.into_vec(),
                        values,
                        logical,
                    },
                })
            }
            Value::CharArray(ca) => Ok(MatArray {
                class: MatClass::Char,
                dims: ca.shape.clone(),
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
                for element in cell.to_column_major() {
                    let gathered = gather_if_needed_async(&element).await?;
                    elements.push(convert_value(gathered).await?);
                }
                Ok(MatArray {
                    class: MatClass::Cell,
                    dims: canonical_dims(&cell.shape),
                    data: MatData::Cell { elements },
                })
            }
            Value::Struct(struct_value) => {
                let field_names: Vec<String> = struct_value.fields.keys().cloned().collect();
                let mut field_values = Vec::with_capacity(field_names.len());
                for field in &field_names {
                    let val = struct_value.fields.get(field).ok_or_else(|| {
                        save_error_with(
                            &SAVE_ERROR_SELECTION,
                            format!("save: missing struct field '{field}'"),
                        )
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
            Value::StructArray(array) => {
                let field_names = array.field_names().cloned().collect::<Vec<_>>();
                let capacity = array.len().checked_mul(field_names.len()).ok_or_else(|| {
                    save_error_with(
                        &SAVE_ERROR_SELECTION,
                        "save: structure array exceeds platform limits",
                    )
                })?;
                let mut field_values = Vec::with_capacity(capacity);
                for field in &field_names {
                    let column = array.field_values(field).ok_or_else(|| {
                        save_error_with(
                            &SAVE_ERROR_SELECTION,
                            format!("save: missing struct field '{field}'"),
                        )
                    })?;
                    for value in column {
                        field_values
                            .push(convert_value(gather_if_needed_async(value).await?).await?);
                    }
                }
                Ok(MatArray {
                    class: MatClass::Struct,
                    dims: canonical_dims(array.shape()),
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
            unsupported => Err(save_error_with(
                &SAVE_ERROR_UNSUPPORTED,
                format!("save: value of type '{:?}' is not supported", unsupported),
            )),
        }
    })
}

fn char_array_to_utf16(ca: &CharArray) -> Vec<u16> {
    ca.to_column_major()
        .into_iter()
        .map(|value| value as u16)
        .collect()
}

fn int_value_mat_class(value: &IntValue) -> MatClass {
    match value {
        IntValue::I8(_) => MatClass::Int8,
        IntValue::I16(_) => MatClass::Int16,
        IntValue::I32(_) => MatClass::Int32,
        IntValue::I64(_) => MatClass::Int64,
        IntValue::U8(_) => MatClass::UInt8,
        IntValue::U16(_) => MatClass::UInt16,
        IntValue::U32(_) => MatClass::UInt32,
        IntValue::U64(_) => MatClass::UInt64,
    }
}

fn integer_storage_from_scalar(value: IntValue) -> IntegerStorage {
    match value {
        IntValue::I8(value) => IntegerStorage::I8(vec![value]),
        IntValue::I16(value) => IntegerStorage::I16(vec![value]),
        IntValue::I32(value) => IntegerStorage::I32(vec![value]),
        IntValue::I64(value) => IntegerStorage::I64(vec![value]),
        IntValue::U8(value) => IntegerStorage::U8(vec![value]),
        IntValue::U16(value) => IntegerStorage::U16(vec![value]),
        IntValue::U32(value) => IntegerStorage::U32(vec![value]),
        IntValue::U64(value) => IntegerStorage::U64(vec![value]),
    }
}

fn integer_storage_mat_class(storage: &IntegerStorage) -> MatClass {
    match storage {
        IntegerStorage::I8(_) => MatClass::Int8,
        IntegerStorage::I16(_) => MatClass::Int16,
        IntegerStorage::I32(_) => MatClass::Int32,
        IntegerStorage::I64(_) => MatClass::Int64,
        IntegerStorage::U8(_) => MatClass::UInt8,
        IntegerStorage::U16(_) => MatClass::UInt16,
        IntegerStorage::U32(_) => MatClass::UInt32,
        IntegerStorage::U64(_) => MatClass::UInt64,
    }
}

fn tensor_dtype_mat_class(dtype: NumericDType) -> MatClass {
    match dtype {
        NumericDType::F64 => MatClass::Double,
        NumericDType::F32 => MatClass::Single,
        NumericDType::I8 => MatClass::Int8,
        NumericDType::I16 => MatClass::Int16,
        NumericDType::I32 => MatClass::Int32,
        NumericDType::I64 => MatClass::Int64,
        NumericDType::U8 => MatClass::UInt8,
        NumericDType::U16 => MatClass::UInt16,
        NumericDType::U32 => MatClass::UInt32,
        NumericDType::U64 => MatClass::UInt64,
    }
}

async fn write_mat_file(path: &MatFsPath, vars: &[MatVar]) -> BuiltinResult<()> {
    let bytes = write_mat_bytes(vars)?;
    path.write(&bytes).await.map_err(|e| {
        save_error_with_source(
            &SAVE_ERROR_IO,
            format!("save: failed to write '{}': {e}", path.path().display()),
            e,
        )
    })
}

pub async fn encode_workspace_to_mat_bytes(entries: &[(String, Value)]) -> BuiltinResult<Vec<u8>> {
    let mut mat_vars = Vec::with_capacity(entries.len());
    for (name, value) in entries {
        let value = prepare_value_for_mat_save(value.clone()).await?;
        mat_vars.push(MatVar {
            name: name.clone(),
            array: convert_value(value).await?,
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
    writer.write_all(&header).map_err(|e| {
        save_error_with_source(
            &SAVE_ERROR_IO,
            format!("save: failed to write header: {e}"),
            e,
        )
    })?;

    for var in vars {
        let matrix_bytes = build_matrix_bytes(&var.array, Some(&var.name))?;
        write_tagged(&mut writer, MI_MATRIX, &matrix_bytes)?;
    }

    writer.flush().map_err(|e| {
        save_error_with_source(&SAVE_ERROR_IO, format!("save: flush failed: {e}"), e)
    })?;
    Ok(writer
        .into_inner()
        .map_err(|e| {
            save_error_with_source(&SAVE_ERROR_IO, "save: failed to finalize MAT bytes", e)
        })?
        .into_inner())
}

fn build_matrix_bytes(array: &MatArray, name: Option<&str>) -> BuiltinResult<Vec<u8>> {
    let mut buf = Vec::new();

    let (flags0, flags1) = match &array.data {
        MatData::Numeric { imag, .. } => {
            let mut f0 = array.class.class_code();
            if imag.is_some() {
                f0 |= FLAG_COMPLEX;
            }
            (f0, 0u32)
        }
        MatData::Logical { .. } => ((array.class.class_code()) | FLAG_LOGICAL, 0u32),
        MatData::Sparse {
            values, logical, ..
        } => (
            array.class.class_code() | if *logical { FLAG_LOGICAL } else { 0 },
            values.len() as u32,
        ),
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
        MatData::Numeric { real, imag } => {
            if tensor_dtype_mat_class(real.numeric_dtype()) != array.class {
                return Err(save_error_with(
                    &SAVE_ERROR_IO,
                    "save: numeric MAT class does not match authoritative storage",
                ));
            }
            let (data_type, real_bytes) = encode_numeric_storage(real)?;
            write_subelement(&mut buf, data_type, &real_bytes);
            if let Some(imag) = imag {
                if imag.numeric_dtype() != real.numeric_dtype() || imag.len() != real.len() {
                    return Err(save_error_with(
                        &SAVE_ERROR_IO,
                        "save: complex numeric components must have matching class and length",
                    ));
                }
                let (imag_type, imag_bytes) = encode_numeric_storage(imag)?;
                write_subelement(&mut buf, imag_type, &imag_bytes);
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
        MatData::Sparse {
            col_ptrs,
            row_indices,
            values,
            ..
        } => {
            let ir_bytes = encode_usize_i32_payload(row_indices, "sparse row index")?;
            write_subelement(&mut buf, MI_INT32, &ir_bytes);
            let jc_bytes = encode_usize_i32_payload(col_ptrs, "sparse column pointer")?;
            write_subelement(&mut buf, MI_INT32, &jc_bytes);
            let (value_type, value_bytes) = encode_numeric_storage(values)?;
            write_subelement(&mut buf, value_type, &value_bytes);
        }
    }

    Ok(buf)
}

fn encode_numeric_storage(storage: &NumericStorage) -> BuiltinResult<(u32, Vec<u8>)> {
    macro_rules! encode {
        ($values:expr, $data_type:expr) => {{
            let mut bytes = Vec::with_capacity(std::mem::size_of_val($values.as_slice()));
            for value in $values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok(($data_type, bytes))
        }};
    }

    match storage {
        NumericStorage::F64(values) => {
            let mut bytes = Vec::with_capacity(values.len() * 8);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok((MI_DOUBLE, bytes))
        }
        NumericStorage::F32(values) => {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok((MI_SINGLE, bytes))
        }
        NumericStorage::I8(values) => {
            Ok((MI_INT8, values.iter().map(|value| *value as u8).collect()))
        }
        NumericStorage::U8(values) => Ok((MI_UINT8, values.clone())),
        NumericStorage::I16(values) => encode!(values, MI_INT16),
        NumericStorage::U16(values) => encode!(values, MI_UINT16),
        NumericStorage::I32(values) => encode!(values, MI_INT32),
        NumericStorage::U32(values) => encode!(values, MI_UINT32),
        NumericStorage::I64(values) => encode!(values, MI_INT64),
        NumericStorage::U64(values) => encode!(values, MI_UINT64),
    }
}

fn encode_usize_i32_payload(values: &[usize], label: &str) -> BuiltinResult<Vec<u8>> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        let converted = i32::try_from(*value).map_err(|_| {
            save_error_with(
                &SAVE_ERROR_IO,
                format!("save: {label} exceeds MAT-file int32 range"),
            )
        })?;
        bytes.extend_from_slice(&converted.to_le_bytes());
    }
    Ok(bytes)
}

fn write_tagged<W: Write>(writer: &mut W, data_type: u32, data: &[u8]) -> BuiltinResult<()> {
    if data.len() > u32::MAX as usize {
        return Err(save_error_with(
            &SAVE_ERROR_IO,
            "save: data too large for MAT-file",
        ));
    }
    writer.write_all(&data_type.to_le_bytes()).map_err(|e| {
        save_error_with_source(&SAVE_ERROR_IO, format!("save: write failed: {e}"), e)
    })?;
    writer
        .write_all(&(data.len() as u32).to_le_bytes())
        .map_err(|e| {
            save_error_with_source(&SAVE_ERROR_IO, format!("save: write failed: {e}"), e)
        })?;
    writer.write_all(data).map_err(|e| {
        save_error_with_source(&SAVE_ERROR_IO, format!("save: write failed: {e}"), e)
    })?;
    let padding = (8 - (data.len() % 8)) % 8;
    if padding != 0 {
        let pad = [0u8; 8];
        writer.write_all(&pad[..padding]).map_err(|e| {
            save_error_with_source(&SAVE_ERROR_IO, format!("save: write failed: {e}"), e)
        })?;
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
#[path = "save/tests.rs"]
pub(crate) mod tests;
