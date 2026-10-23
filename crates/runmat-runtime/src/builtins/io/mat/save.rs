//! MATLAB-compatible `save` builtin for RunMat.

use std::collections::HashSet;
use std::path::PathBuf;

use regex::Regex;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, StructValue, Value};

use super::authority::MatFsPath;
use super::cooperation::MatCodecCooperation;
use super::load::read_mat_file_on_with_resource_liveness;
use crate::context::host_effect::{
    HostEffectWorkload, MatEncodeMode, MatEncodeOutput, MatStructSelection,
};

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use workspace_io::WorkspaceReadSet;

use runmat_builtins::{
    SAVE_ERROR_FILENAME, SAVE_ERROR_INVALID_ARGUMENT, SAVE_ERROR_INVALID_OPTION, SAVE_ERROR_IO,
    SAVE_ERROR_SELECTION, SAVE_ERROR_WORKSPACE,
};

#[runtime_builtin(
    name = "save",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::mat::save"
)]
async fn save_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let output_scope = crate::output_count::current_output_scope();
    let requested_outputs = match output_scope {
        crate::output_count::InvocationOutputScope::Requested(count) => Some(count),
        crate::output_count::InvocationOutputScope::Implicit
        | crate::output_count::InvocationOutputScope::Unscoped => None,
    };
    if requested_outputs.is_some_and(|count| count > 0) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::SAVE_STATUS_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    preflight_known_options(args.view())?;
    // Public arguments are filename, option, and workspace-selector text;
    // saved payloads come from the workspace only after these controls pass.
    if args.iter().any(crate::dispatcher::value_contains_gpu) {
        return Err(save_error_with(
            &SAVE_ERROR_INVALID_ARGUMENT,
            "save: filename, options, and variable selectors must be host text values",
        ));
    }
    let default_path = Value::from("matlab.mat");
    let (mut path_value, option_start, used_default) = match args.first() {
        Some(first) if option_token(first)?.is_some() => (default_path, 0usize, true),
        Some(first) => (first.clone(), 1usize, false),
        None => (default_path, 0usize, true),
    };

    if used_default {
        if let Some(override_path) =
            crate::builtins::common::env::optional_var("RUNMAT_SAVE_DEFAULT_PATH")?
        {
            path_value = Value::from(override_path);
        }
    }

    let request = parse_arguments(args.view().range(option_start..)?).await?;
    let regex_patterns: Vec<Regex> = request
        .regex_patterns
        .iter()
        .map(|pattern| {
            Regex::new(pattern).map_err(|error| {
                save_error_with_source(
                    &SAVE_ERROR_INVALID_OPTION,
                    format!("save: invalid regular expression '{pattern}': {error}"),
                    error,
                )
            })
        })
        .collect::<BuiltinResult<_>>()?;
    let path = normalise_path(&path_value)?;
    let destination = MatFsPath::capture(path).map_err(|err| {
        save_error_with_source(
            &SAVE_ERROR_IO,
            format!("save: failed to resolve MAT path: {err}"),
            err,
        )
    })?;
    let struct_selections: Vec<_> = request
        .structs
        .iter()
        .map(|selection| MatStructSelection {
            source: &selection.source,
            fields: selection.fields.as_deref(),
        })
        .collect();
    let workload = HostEffectWorkload::MatEncode {
        destination: destination.path(),
        mode: MatEncodeMode::Save {
            append: request.append,
            names: &request.variables,
            structs: &struct_selections,
            patterns: &request.regex_patterns,
        },
        output: if requested_outputs == Some(0) {
            MatEncodeOutput::EffectOnly
        } else {
            MatEncodeOutput::Status
        },
    };
    let mut workspace_read = WorkspaceReadSet::capture(runtime.as_ref())?;
    let receipt = super::admission::select(runtime.as_ref(), &destination, BUILTIN_NAME, workload)?;
    let result = async {
        let mut cooperation = MatCodecCooperation::new(
            runtime.as_ref(),
            Some(&destination),
            BUILTIN_NAME,
            &SAVE_ERROR_IO,
        )
        .with_resource_liveness(
            receipt
                .as_ref()
                .map(|selected| selected.resource_liveness_lease()),
        )
        .with_workspace_liveness(workspace_read.liveness_service());
        cooperation.completed(0).await?;

        let entries = collect_selected_entries(
            &request,
            &regex_patterns,
            &mut workspace_read,
            &mut cooperation,
        )
        .await?;
        workspace_read.retain_values(entries.iter().map(|(_, value)| value))?;
        let mut unique_entries = deduplicate_entries(entries);
        if request.append {
            unique_entries = append_existing_entries(
                &destination,
                unique_entries,
                runtime.as_ref(),
                &mut cooperation,
                &mut workspace_read,
            )
            .await?;
            workspace_read.retain_values(unique_entries.iter().map(|(_, value)| value))?;
        }

        encode_selected_entries(
            unique_entries,
            &destination,
            &mut workspace_read,
            &mut cooperation,
        )
        .await?;
        Ok(Value::Num(0.0))
    }
    .await;
    super::admission::finish(receipt, &result);
    result
}

// Keep selection and encoding in separate poll frames. Both phases borrow the
// same rooted workspace and cooperation owner; no extra value snapshot is
// introduced by either phase.
async fn collect_selected_entries(
    request: &controls::SaveRequest,
    regex_patterns: &[Regex],
    workspace_read: &mut WorkspaceReadSet<'_>,
    cooperation: &mut MatCodecCooperation<'_>,
) -> BuiltinResult<Vec<(String, Value)>> {
    let mut workspace_entries: Option<Vec<(String, Value)>> = None;
    let mut entries: Vec<(String, Value)> = Vec::new();

    if request.variables.is_empty()
        && request.structs.is_empty()
        && request.regex_patterns.is_empty()
    {
        let snapshot =
            ensure_workspace_entries(&mut workspace_entries, workspace_read, cooperation).await?;
        entries.extend(snapshot.iter().cloned());
    } else {
        for name in &request.variables {
            if let Some(snapshot) = workspace_entries.as_ref() {
                if let Some(value) = find_in_entries(snapshot, name) {
                    entries.push((name.clone(), value));
                    continue;
                }
            }
            let value = cooperation
                .await_io(lookup_workspace(workspace_read, name))
                .await??;
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
                None => {
                    cooperation
                        .await_io(lookup_workspace(workspace_read, &struct_req.source))
                        .await??
                }
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
                cooperation,
            )
            .await?;
        }

        if !request.regex_patterns.is_empty() {
            let snapshot =
                ensure_workspace_entries(&mut workspace_entries, workspace_read, cooperation)
                    .await?;
            let mut matched = 0usize;
            for (name, value) in snapshot.iter() {
                if regex_patterns.iter().any(|regex| regex.is_match(name)) {
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
    Ok(entries)
}

async fn encode_selected_entries(
    entries: Vec<(String, Value)>,
    destination: &MatFsPath,
    workspace_read: &mut WorkspaceReadSet<'_>,
    cooperation: &mut MatCodecCooperation<'_>,
) -> BuiltinResult<()> {
    let mut mat_vars = Vec::with_capacity(entries.len());
    for (name, value) in entries {
        cooperation.completed(0).await?;
        workspace_read.retain_values(std::iter::once(&value))?;
        cooperation.completed(0).await?;
        let array =
            convert_value_with_cooperation(value, MatEncodingOwner::Save, Some(&mut *cooperation))
                .await?;
        mat_vars.push(MatVar { name, array });
        cooperation.completed(0).await?;
    }
    write_mat_file(destination, &mat_vars, MatEncodingOwner::Save, cooperation).await
}

mod controls;
use controls::{normalise_path, option_token, parse_arguments, preflight_known_options};

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

fn save_workspace_flow(mut error: RuntimeError) -> RuntimeError {
    error.context = error.context.with_builtin(BUILTIN_NAME);
    error
}

#[path = "save/workspace.rs"]
mod workspace_io;
use workspace_io::{
    append_existing_entries, append_struct_fields, deduplicate_entries, ensure_workspace_entries,
    find_in_entries, lookup_workspace,
};

pub use super::encoding::encode_workspace_to_mat_bytes;
use super::encoding::{convert_value_with_cooperation, write_mat_file, MatEncodingOwner, MatVar};

#[cfg(test)]
#[path = "save/tests.rs"]
pub(crate) mod tests;
