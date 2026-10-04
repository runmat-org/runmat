//! MATLAB-compatible `load` builtin for RunMat.

use std::collections::HashMap;
use std::io::{BufReader, Cursor};
use std::path::{Path, PathBuf};

use regex::Regex;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, StructValue, Value};

use super::authority::MatFsPath;
use crate::builtins::introspection::object_serialization::restore_value_from_mat_load;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use runmat_builtins::{
    LOAD_ERROR_FILENAME, LOAD_ERROR_INVALID_ARGUMENT, LOAD_ERROR_INVALID_OPTION, LOAD_ERROR_IO,
    LOAD_ERROR_OUTPUT_COUNT, LOAD_ERROR_SELECTION, LOAD_ERROR_WORKSPACE,
};

mod admission;
use crate::context::host_effect::{HostEffectReceipt, MatDecodeOutput};
#[runtime_builtin(
    name = "load",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::mat::load"
)]
async fn load_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let requested_outputs = match crate::output_count::current_output_scope() {
        crate::output_count::InvocationOutputScope::Requested(count) => Some(count),
        crate::output_count::InvocationOutputScope::Implicit => None,
        crate::output_count::InvocationOutputScope::Unscoped => {
            crate::output_context::requested_output_count()
        }
    };
    if requested_outputs.is_some_and(|count| count > 1) {
        return Err(load_error_with(
            &LOAD_ERROR_OUTPUT_COUNT,
            "load supports at most one output argument",
        ));
    }
    let execution = evaluate_with_context(
        &args,
        runtime.as_ref(),
        if requested_outputs == Some(0) {
            MatDecodeOutput::Workspace
        } else {
            MatDecodeOutput::Struct
        },
    )
    .await?;
    let LoadExecution { eval, receipt } = execution;

    // The same invocation-owned output request remains authoritative across
    // the awaited decode/restoration work.
    let result = (|| {
        if requested_outputs == Some(0) {
            for (name, value) in eval.variables() {
                if let Some(receipt) = receipt.as_ref() {
                    receipt.checkpoint()?;
                }
                crate::workspace::assign(name, value.clone())
                    .map_err(|err| load_error_with(&LOAD_ERROR_WORKSPACE, err))?;
            }
            Ok(Value::OutputList(Vec::new()))
        } else {
            if let Some(receipt) = receipt.as_ref() {
                receipt.checkpoint()?;
            }
            Ok(eval.first_output())
        }
    })()
    .and_then(|value| {
        if let Some(receipt) = receipt.as_ref() {
            receipt.checkpoint()?;
        }
        Ok(value)
    });
    admission::finish(receipt, &result);
    result
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
    regex_tokens: Vec<String>,
}

struct LoadExecution {
    eval: LoadEval,
    receipt: Option<HostEffectReceipt>,
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
    evaluate_with_context(args, None, MatDecodeOutput::Struct)
        .await
        .map(|execution| execution.eval)
}

async fn evaluate_with_context(
    args: &[Value],
    runtime: Option<&crate::context::RuntimeContext>,
    output: MatDecodeOutput,
) -> BuiltinResult<LoadExecution> {
    if args.iter().any(crate::dispatcher::value_contains_gpu) {
        return Err(load_error_with(
            &LOAD_ERROR_INVALID_ARGUMENT,
            "load: filename, options, and variable selectors must be host text values",
        ));
    }
    preflight_known_options(args)?;
    // Filename/options/selectors are text controls. A numeric resident handle
    // cannot become an admitted text control by downloading it first.
    let host_args = args.to_vec();

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
    for pattern in &invocation.regex_tokens {
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
        regex_tokens: invocation.regex_tokens,
    };
    let path = normalise_path(&path_value)?;
    let source = MatFsPath::capture(path).map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!("load: failed to resolve MAT path: {err}"),
            err,
        )
    })?;
    let receipt = admission::select(runtime, &source, &request, output)?;
    let result = async {
        let entries = read_mat_file_on_with_context(&source, runtime).await?;
        let selected = select_variables(&entries, &request)?;
        Ok(LoadEval {
            variables: selected,
        })
    }
    .await;
    match result {
        Ok(eval) => Ok(LoadExecution { eval, receipt }),
        Err(error) => {
            admission::finish_error(receipt, &error);
            Err(error)
        }
    }
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
            Some(runmat_builtins::MatLoadOption::Regexp)
                if index + 1 == args.len()
                    || args
                        .get(index + 1)
                        .and_then(value_to_string_scalar)
                        .is_some_and(|next| next.starts_with('-')) =>
            {
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
                    let prior_patterns = regex_tokens.len();
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
                    if regex_tokens.len() == prior_patterns {
                        return Err(load_error_with(
                            &LOAD_ERROR_INVALID_OPTION,
                            "load: '-regexp' requires at least one pattern",
                        ));
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

pub(crate) async fn read_mat_file_for_builtin_on_with_context(
    source: &MatFsPath,
    builtin: &str,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<Vec<(String, Value)>> {
    match read_mat_file_on_with_context(source, runtime).await {
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
    read_mat_file_on_with_context(source, None).await
}

pub(crate) async fn read_mat_file_on_with_context(
    source: &MatFsPath,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Vec<(String, Value)>> {
    let mut cooperation =
        decoder::MatCodecCooperation::new(runtime, Some(source), "load", &LOAD_ERROR_IO);
    let file = cooperation.await_io(source.open()).await?.map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!("load: failed to open '{}': {err}", source.path().display()),
            err,
        )
    })?;
    let mut reader = BufReader::new(file);
    let entries = decoder::read_mat_reader_cooperative(&mut reader, &mut cooperation).await?;
    check_source_current(source)?;
    restore_loaded_entries(entries, Some(source), Some(&mut cooperation)).await
}

pub fn decode_workspace_from_mat_bytes(bytes: &[u8]) -> BuiltinResult<Vec<(String, Value)>> {
    let mut cursor = Cursor::new(bytes);
    let entries = read_mat_reader(&mut cursor)?;
    futures::executor::block_on(restore_loaded_entries(entries, None, None))
}

async fn restore_loaded_entries(
    entries: Vec<(String, Value)>,
    source: Option<&MatFsPath>,
    mut cooperation: Option<&mut decoder::MatCodecCooperation<'_>>,
) -> BuiltinResult<Vec<(String, Value)>> {
    let mut restored = Vec::new();
    restored.try_reserve_exact(entries.len()).map_err(|error| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            "load: restored values cannot be allocated",
            error,
        )
    })?;
    for (name, value) in entries {
        if let Some(cooperation) = cooperation.as_deref_mut() {
            cooperation.completed(0).await?;
        }
        if let Some(source) = source {
            check_source_current(source)?;
        }
        let value = match cooperation.as_deref_mut() {
            Some(cooperation) => {
                cooperation
                    .await_io(restore_value_from_mat_load(value))
                    .await??
            }
            None => restore_value_from_mat_load(value).await?,
        };
        restored.push((name, value));
        if let Some(source) = source {
            check_source_current(source)?;
        }
        if let Some(cooperation) = cooperation.as_deref_mut() {
            cooperation.completed(1).await?;
        }
    }
    if let Some(cooperation) = cooperation.as_deref_mut() {
        cooperation.completed(0).await?;
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

mod decoder;
#[cfg(test)]
use decoder::mat_array_to_value;
use decoder::read_mat_reader;
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

#[cfg(test)]
#[path = "load/tests.rs"]
pub(crate) mod tests;
