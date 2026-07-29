//! MATLAB-compatible `save` builtin for RunMat.

use std::collections::HashSet;
use std::io::{BufWriter, Cursor, Write};
use std::path::{Path, PathBuf};

use futures::future::LocalBoxFuture;
use regex::Regex;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, IntValue, IntegerStorage, NumericDType, StructValue, Value,
};
use runmat_filesystem::{metadata_async, write_async};
use runmat_macros::runtime_builtin;

use super::format::{
    MatArray, MatClass, MatData, FLAG_COMPLEX, FLAG_LOGICAL, MAT_HEADER_LEN, MI_DOUBLE, MI_INT16,
    MI_INT32, MI_INT64, MI_INT8, MI_MATRIX, MI_SINGLE, MI_UINT16, MI_UINT32, MI_UINT64, MI_UINT8,
};
use super::load::read_mat_file;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::introspection::object_serialization::prepare_value_for_mat_save;
use crate::workspace;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const SAVE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero status code on success.",
}];
const SAVE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const SAVE_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"matlab.mat\""),
    description: "MAT-file output path.",
}];
const SAVE_INPUTS_FILENAME_VARS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"matlab.mat\""),
        description: "MAT-file output path.",
    },
    BuiltinParamDescriptor {
        name: "varName",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Workspace variable names to persist.",
    },
];
const SAVE_INPUTS_STRUCT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"matlab.mat\""),
        description: "MAT-file output path.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"-struct\""),
        description: "Struct field export option.",
    },
    BuiltinParamDescriptor {
        name: "structVar",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Workspace struct variable name.",
    },
];
const SAVE_INPUTS_REGEXP: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"matlab.mat\""),
        description: "MAT-file output path.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"-regexp\""),
        description: "Regex selection option.",
    },
    BuiltinParamDescriptor {
        name: "pattern",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Regex patterns matched against workspace variable names.",
    },
];
const SAVE_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "status = save()",
        inputs: &SAVE_INPUTS_NONE,
        outputs: &SAVE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "status = save(filename)",
        inputs: &SAVE_INPUTS_FILENAME,
        outputs: &SAVE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "status = save(filename, varName1, varName2, ...)",
        inputs: &SAVE_INPUTS_FILENAME_VARS,
        outputs: &SAVE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "status = save(filename, \"-struct\", structVar, field1, ...)",
        inputs: &SAVE_INPUTS_STRUCT,
        outputs: &SAVE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "status = save(filename, \"-regexp\", pattern1, ...)",
        inputs: &SAVE_INPUTS_REGEXP,
        outputs: &SAVE_OUTPUT,
    },
];
const SAVE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAVE.INVALID_ARGUMENT",
    identifier: Some("RunMat:save:InvalidArgument"),
    when: "Arguments do not match supported save invocation forms.",
    message: "save: invalid argument",
};
const SAVE_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAVE.INVALID_OPTION",
    identifier: Some("RunMat:save:InvalidOption"),
    when: "Option token or option value is invalid.",
    message: "save: invalid option",
};
const SAVE_ERROR_SELECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAVE.SELECTION",
    identifier: Some("RunMat:save:Selection"),
    when: "Requested variables or struct fields cannot be resolved.",
    message: "save: variable selection failed",
};
const SAVE_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAVE.FILENAME",
    identifier: Some("RunMat:save:Filename"),
    when: "Filename is invalid or cannot be normalized.",
    message: "save: invalid filename",
};
const SAVE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAVE.IO",
    identifier: Some("RunMat:save:Io"),
    when: "MAT-file cannot be written or finalized.",
    message: "save: MAT-file I/O failure",
};
const SAVE_ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAVE.UNSUPPORTED",
    identifier: Some("RunMat:save:Unsupported"),
    when: "Unsupported save mode or value type is requested.",
    message: "save: unsupported operation",
};
const SAVE_ERROR_WORKSPACE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAVE.WORKSPACE",
    identifier: Some("RunMat:save:Workspace"),
    when: "Workspace state is unavailable.",
    message: "save: workspace state unavailable",
};
const SAVE_ERRORS: [BuiltinErrorDescriptor; 7] = [
    SAVE_ERROR_INVALID_ARGUMENT,
    SAVE_ERROR_INVALID_OPTION,
    SAVE_ERROR_SELECTION,
    SAVE_ERROR_FILENAME,
    SAVE_ERROR_IO,
    SAVE_ERROR_UNSUPPORTED,
    SAVE_ERROR_WORKSPACE,
];
pub const SAVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SAVE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SAVE_ERRORS,
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
    category = "io/mat",
    summary = "Save workspace variables to a MAT-file.",
    keywords = "save,mat,workspace",
    sink = true,
    type_resolver(crate::builtins::io::type_resolvers::save_type),
    descriptor(crate::builtins::io::mat::save::SAVE_DESCRIPTOR),
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

    let path = normalise_path(&path_value)?;
    let mut unique_entries = deduplicate_entries(entries);
    if request.append {
        unique_entries = append_existing_entries(&path, unique_entries).await?;
    }

    let mut mat_vars = Vec::with_capacity(unique_entries.len());
    for (name, value) in unique_entries {
        let value = prepare_value_for_mat_save(value).await?;
        mat_vars.push(MatVar {
            name,
            array: convert_value(value).await?,
        });
    }

    write_mat_file(&path, &mat_vars).await?;

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
            match flag.as_str() {
                "-append" => append = true,
                "-struct" => {
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
                "-regexp" => {
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
                other => {
                    return Err(save_error_with(
                        &SAVE_ERROR_INVALID_OPTION,
                        format!("save: unsupported option '{other}'"),
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
    path: &Path,
    new_entries: Vec<(String, Value)>,
) -> BuiltinResult<Vec<(String, Value)>> {
    let mut entries = read_existing_entries_for_append(path).await?;
    entries.extend(new_entries);
    Ok(deduplicate_entries(entries))
}

async fn read_existing_entries_for_append(path: &Path) -> BuiltinResult<Vec<(String, Value)>> {
    match metadata_async(path).await {
        Ok(_) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => {
            return Err(save_error_with_source(
                &SAVE_ERROR_IO,
                format!(
                    "save: failed to inspect existing MAT-file '{}': {err}",
                    path.display()
                ),
                err,
            ));
        }
    }
    match read_mat_file(path).await {
        Ok(entries) => Ok(entries),
        Err(err) => Err(save_error_with_source(
            &SAVE_ERROR_IO,
            format!(
                "save: failed to read existing MAT-file '{}': {}",
                path.display(),
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
                data: MatData::Double {
                    real: vec![n],
                    imag: None,
                },
            }),
            Value::Int(i) => {
                let class = int_value_mat_class(&i);
                Ok(MatArray {
                    class,
                    dims: vec![1, 1],
                    data: MatData::Integer {
                        storage: integer_storage_from_scalar(i),
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
                let class = if let Some(storage) = t.integer_storage() {
                    integer_storage_mat_class(storage)
                } else {
                    tensor_dtype_mat_class(t.dtype)
                };
                let data = if let Some(storage) = t.integer_data {
                    MatData::Integer {
                        storage,
                        imag: None,
                    }
                } else if class == MatClass::Double {
                    MatData::Double {
                        real: t.data,
                        imag: None,
                    }
                } else {
                    MatData::Numeric {
                        real: t.data,
                        imag: None,
                    }
                };
                Ok(MatArray {
                    class,
                    dims: canonical_dims(&t.shape),
                    data,
                })
            }
            Value::Complex(re, im) => Ok(MatArray {
                class: MatClass::Double,
                dims: vec![1, 1],
                data: MatData::Double {
                    real: vec![re],
                    imag: Some(vec![im]),
                },
            }),
            Value::ComplexTensor(t) => {
                if let Some(storage) = t.integer_data {
                    return Ok(MatArray {
                        class: integer_storage_mat_class(&storage.real),
                        dims: canonical_dims(&t.shape),
                        data: MatData::Integer {
                            storage: storage.real,
                            imag: Some(storage.imag),
                        },
                    });
                }
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
            Value::SparseTensor(sparse) => Ok(MatArray {
                class: MatClass::Sparse,
                dims: vec![sparse.rows, sparse.cols],
                data: MatData::Sparse {
                    rows: sparse.rows,
                    cols: sparse.cols,
                    col_ptrs: sparse.col_ptrs,
                    row_indices: sparse.row_indices,
                    integer_data: sparse.integer_data,
                    values: sparse.values,
                },
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
                        let element = &cell.data[idx];
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
    let mut data = Vec::with_capacity(ca.rows * ca.cols);
    for col in 0..ca.cols {
        for row in 0..ca.rows {
            let idx = row * ca.cols + col;
            data.push(ca.data[idx] as u16);
        }
    }
    data
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

async fn write_mat_file(path: &Path, vars: &[MatVar]) -> BuiltinResult<()> {
    let bytes = write_mat_bytes(vars)?;
    write_async(path, &bytes).await.map_err(|e| {
        save_error_with_source(
            &SAVE_ERROR_IO,
            format!("save: failed to write '{}': {e}", path.display()),
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
        MatData::Double { imag, .. } => {
            let mut f0 = array.class.class_code();
            if imag.is_some() {
                f0 |= FLAG_COMPLEX;
            }
            (f0, 0u32)
        }
        MatData::Numeric { imag, .. } => {
            let mut f0 = array.class.class_code();
            if imag.is_some() {
                f0 |= FLAG_COMPLEX;
            }
            (f0, 0u32)
        }
        MatData::Integer { imag, .. } => {
            let mut f0 = array.class.class_code();
            if imag.is_some() {
                f0 |= FLAG_COMPLEX;
            }
            (f0, 0u32)
        }
        MatData::Logical { .. } => ((array.class.class_code()) | FLAG_LOGICAL, 0u32),
        MatData::Sparse {
            integer_data,
            values,
            ..
        } => (
            array.class.class_code(),
            integer_data
                .as_ref()
                .map_or(values.len(), IntegerStorage::len) as u32,
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
        MatData::Numeric { real, imag } => {
            let (data_type, real_bytes) = encode_numeric_payload(array.class, real)?;
            write_subelement(&mut buf, data_type, &real_bytes);
            if let Some(imag) = imag {
                let (imag_type, imag_bytes) = encode_numeric_payload(array.class, imag)?;
                write_subelement(&mut buf, imag_type, &imag_bytes);
            }
        }
        MatData::Integer { storage, imag } => {
            let (data_type, bytes) = encode_integer_payload(storage);
            write_subelement(&mut buf, data_type, &bytes);
            if let Some(imag) = imag {
                if imag.class_name() != storage.class_name() || imag.len() != storage.len() {
                    return Err(save_error_with(
                        &SAVE_ERROR_IO,
                        "save: complex integer components must have matching class and length",
                    ));
                }
                let (imag_type, imag_bytes) = encode_integer_payload(imag);
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
            if array.dims != [1, 1] {
                return Err(save_error_with(
                    &SAVE_ERROR_UNSUPPORTED,
                    "save: struct arrays are not supported",
                ));
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
        MatData::Sparse {
            col_ptrs,
            row_indices,
            integer_data,
            values,
            ..
        } => {
            let ir_bytes = encode_usize_i32_payload(row_indices, "sparse row index")?;
            write_subelement(&mut buf, MI_INT32, &ir_bytes);
            let jc_bytes = encode_usize_i32_payload(col_ptrs, "sparse column pointer")?;
            write_subelement(&mut buf, MI_INT32, &jc_bytes);
            if let Some(storage) = integer_data {
                let (value_type, value_bytes) = encode_integer_payload(storage);
                write_subelement(&mut buf, value_type, &value_bytes);
            } else {
                let mut value_bytes = Vec::with_capacity(values.len() * 8);
                for value in values {
                    value_bytes.extend_from_slice(&value.to_le_bytes());
                }
                write_subelement(&mut buf, MI_DOUBLE, &value_bytes);
            }
        }
    }

    Ok(buf)
}

fn encode_integer_payload(storage: &IntegerStorage) -> (u32, Vec<u8>) {
    macro_rules! encode {
        ($values:expr, $data_type:expr) => {{
            let mut bytes = Vec::with_capacity(std::mem::size_of_val($values.as_slice()));
            for value in $values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            ($data_type, bytes)
        }};
    }

    match storage {
        IntegerStorage::I8(values) => (MI_INT8, values.iter().map(|value| *value as u8).collect()),
        IntegerStorage::U8(values) => (MI_UINT8, values.clone()),
        IntegerStorage::I16(values) => encode!(values, MI_INT16),
        IntegerStorage::U16(values) => encode!(values, MI_UINT16),
        IntegerStorage::I32(values) => encode!(values, MI_INT32),
        IntegerStorage::U32(values) => encode!(values, MI_UINT32),
        IntegerStorage::I64(values) => encode!(values, MI_INT64),
        IntegerStorage::U64(values) => encode!(values, MI_UINT64),
    }
}

fn encode_numeric_payload(class: MatClass, values: &[f64]) -> BuiltinResult<(u32, Vec<u8>)> {
    let mut bytes = Vec::new();
    let data_type = match class {
        MatClass::Single => {
            bytes.reserve(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&(*value as f32).to_le_bytes());
            }
            MI_SINGLE
        }
        MatClass::Int8 => {
            bytes.reserve(values.len());
            for value in values {
                bytes.push(*value as i8 as u8);
            }
            MI_INT8
        }
        MatClass::UInt8 => {
            bytes.reserve(values.len());
            for value in values {
                bytes.push(*value as u8);
            }
            MI_UINT8
        }
        MatClass::Int16 => {
            bytes.reserve(values.len() * 2);
            for value in values {
                bytes.extend_from_slice(&(*value as i16).to_le_bytes());
            }
            MI_INT16
        }
        MatClass::UInt16 => {
            bytes.reserve(values.len() * 2);
            for value in values {
                bytes.extend_from_slice(&(*value as u16).to_le_bytes());
            }
            MI_UINT16
        }
        MatClass::Int32 => {
            bytes.reserve(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&(*value as i32).to_le_bytes());
            }
            MI_INT32
        }
        MatClass::UInt32 => {
            bytes.reserve(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&(*value as u32).to_le_bytes());
            }
            MI_UINT32
        }
        MatClass::Int64 => {
            bytes.reserve(values.len() * 8);
            for value in values {
                bytes.extend_from_slice(&(*value as i64).to_le_bytes());
            }
            MI_INT64
        }
        MatClass::UInt64 => {
            bytes.reserve(values.len() * 8);
            for value in values {
                bytes.extend_from_slice(&(*value as u64).to_le_bytes());
            }
            MI_UINT64
        }
        MatClass::Double => {
            bytes.reserve(values.len() * 8);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            MI_DOUBLE
        }
        _ => {
            return Err(save_error_with(
                &SAVE_ERROR_UNSUPPORTED,
                "save: unsupported numeric MAT class",
            ))
        }
    };
    Ok((data_type, bytes))
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
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::workspace::WorkspaceResolver;
    use futures::executor::block_on;
    use once_cell::sync::OnceCell;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, IntegerStorage, NumericDType, StringArray, Tensor};
    use runmat_filesystem::File;
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
            clear: None,
            remove: None,
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

    fn assert_saved_double(path: &Path, name: &str, expected: f64) {
        let file = File::open(path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        let array = mat.find_by_name(name).unwrap();
        match array.data() {
            matfile::NumericData::Double { real, .. } => assert_eq!(real, &[expected]),
            other => panic!("expected double array for {name}, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn save_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SAVE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"status = save()"));
        assert!(labels.contains(&"status = save(filename)"));
        assert!(labels.contains(&"status = save(filename, varName1, varName2, ...)"));
        assert!(labels.contains(&"status = save(filename, \"-struct\", structVar, field1, ...)"));
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
    fn save_preserves_supported_numeric_classes() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        let single = Tensor::new_with_dtype(vec![1.25, 2.5], vec![1, 2], NumericDType::F32)
            .expect("single tensor");
        let int8 = Tensor::new_integer(IntegerStorage::I8(vec![i8::MIN, i8::MAX]), vec![1, 2])
            .expect("int8 tensor");
        let uint8 = Tensor::new_integer(IntegerStorage::U8(vec![0, u8::MAX]), vec![1, 2])
            .expect("uint8 tensor");
        let int16 = Tensor::new_integer(IntegerStorage::I16(vec![i16::MIN, i16::MAX]), vec![1, 2])
            .expect("int16 tensor");
        let uint16 = Tensor::new_integer(IntegerStorage::U16(vec![0, u16::MAX]), vec![1, 2])
            .expect("uint16 tensor");
        let int32 = Tensor::new_integer(IntegerStorage::I32(vec![i32::MIN, i32::MAX]), vec![1, 2])
            .expect("int32 tensor");
        let uint32 = Tensor::new_integer(IntegerStorage::U32(vec![0, u32::MAX]), vec![1, 2])
            .expect("uint32 tensor");
        let int64 = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![1, 2])
            .expect("int64 tensor");
        let uint64 =
            Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]), vec![1, 2])
                .expect("uint64 tensor");
        set_workspace(&[
            ("single_data", Value::Tensor(single)),
            ("int8_data", Value::Tensor(int8)),
            ("uint8_data", Value::Tensor(uint8)),
            ("int16_data", Value::Tensor(int16)),
            ("uint16_data", Value::Tensor(uint16)),
            ("int32_data", Value::Tensor(int32)),
            ("uint32_data", Value::Tensor(uint32)),
            ("int64_data", Value::Tensor(int64)),
            ("uint64_data", Value::Tensor(uint64)),
            ("i8_scalar", Value::Int(IntValue::I8(-3))),
        ]);

        let dir = tempdir().unwrap();
        let path = dir.path().join("numeric_classes.mat");
        let args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("single_data"),
            Value::from("int8_data"),
            Value::from("uint8_data"),
            Value::from("int16_data"),
            Value::from("uint16_data"),
            Value::from("int32_data"),
            Value::from("uint32_data"),
            Value::from("int64_data"),
            Value::from("uint64_data"),
            Value::from("i8_scalar"),
        ];
        block_on(save_builtin(args)).unwrap();

        let values: HashMap<_, _> = block_on(crate::builtins::io::mat::load::read_mat_file(&path))
            .unwrap()
            .into_iter()
            .collect();
        match values.get("single_data").unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.dtype, NumericDType::F32);
                assert_eq!(tensor.data, vec![1.25, 2.5]);
            }
            other => panic!("expected single tensor, got {other:?}"),
        }
        for (name, storage) in [
            ("int8_data", IntegerStorage::I8(vec![i8::MIN, i8::MAX])),
            ("uint8_data", IntegerStorage::U8(vec![0, u8::MAX])),
            ("int16_data", IntegerStorage::I16(vec![i16::MIN, i16::MAX])),
            ("uint16_data", IntegerStorage::U16(vec![0, u16::MAX])),
            ("int32_data", IntegerStorage::I32(vec![i32::MIN, i32::MAX])),
            ("uint32_data", IntegerStorage::U32(vec![0, u32::MAX])),
            ("int64_data", IntegerStorage::I64(vec![i64::MIN, i64::MAX])),
            (
                "uint64_data",
                IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
            ),
        ] {
            match values.get(name).unwrap() {
                Value::Tensor(tensor) => assert_eq!(tensor.integer_storage(), Some(&storage)),
                other => panic!("expected integer tensor for {name}, got {other:?}"),
            }
        }
        assert_eq!(values.get("i8_scalar"), Some(&Value::Int(IntValue::I8(-3))));
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
    fn save_append_creates_missing_file() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[("A", Value::Num(1.0))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("append_new.mat");
        let args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("A"),
            Value::from("-append"),
        ];

        block_on(save_builtin(args)).unwrap();

        assert_saved_double(&path, "A", 1.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn save_append_preserves_existing_and_replaces_duplicates() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[("A", Value::Num(1.0))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("append_replace.mat");
        block_on(save_builtin(vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("A"),
        ]))
        .unwrap();

        set_workspace(&[("A", Value::Num(3.0)), ("B", Value::Num(2.0))]);
        block_on(save_builtin(vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("B"),
            Value::from("A"),
            Value::from("-append"),
        ]))
        .unwrap();

        assert_saved_double(&path, "A", 3.0);
        assert_saved_double(&path, "B", 2.0);
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
