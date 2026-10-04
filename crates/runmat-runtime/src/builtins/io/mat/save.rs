//! MATLAB-compatible `save` builtin for RunMat.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use regex::Regex;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, StructValue, Value};

use super::authority::MatFsPath;
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
            Some(runmat_builtins::MatSaveOption::Regexp)
                if args
                    .get(index + 1)
                    .and_then(value_to_string_scalar)
                    .is_some_and(|next| next.starts_with('-')) =>
            {
                return Err(save_error_with(
                    &SAVE_ERROR_INVALID_OPTION,
                    "save: '-regexp' requires at least one pattern",
                ));
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

mod encoding;
pub use encoding::encode_workspace_to_mat_bytes;
use encoding::{convert_value, write_mat_file, MatVar};

#[cfg(test)]
#[path = "save/tests.rs"]
pub(crate) mod tests;
