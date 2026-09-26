//! Script-source resolution for the VM-owned caller-workspace execution boundary.
#[cfg(test)]
mod tests;
use crate::builtins::common::fs::path_to_string;
use crate::builtins::common::path_search::{
    file_candidates, find_file_with_extensions, path_is_file,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, RUN_ERROR_ARG_TYPE, RUN_ERROR_EMPTY_SCRIPT, RUN_ERROR_FILE_NOT_FOUND,
    RUN_ERROR_FILE_READ, RUN_ERROR_PATH_RESOLVE, RUN_ERROR_REQUIRES_VM, RUN_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_types::RUN_BUILTIN_NAME;
use runmat_value::Value;
use std::path::{Path, PathBuf};
const RUN_SCRIPT_EXTENSIONS: &[&str] = &[".p", ".m"];
#[derive(Debug, Clone)]
pub struct RunScriptSource {
    pub path: PathBuf,
    pub display_name: String,
    pub text: String,
}

fn run_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    run_error_with_detail(error, "")
}

fn run_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        error.message.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(RUN_BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn value_to_string_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(array)
            if array.rows == 1
                && crate::builtins::common::shape::effective_rank(&array.shape) <= 2 =>
        {
            Some(array.data.iter().collect())
        }
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        _ => None,
    }
}

pub fn requires_vm_workspace_context() -> crate::BuiltinResult<Value> {
    Err(run_error(&RUN_ERROR_REQUIRES_VM))
}

pub fn too_many_outputs_error() -> RuntimeError {
    run_error(&RUN_ERROR_TOO_MANY_OUTPUTS)
}

fn bare_run_file_stem(script: &str) -> Option<&str> {
    if script.starts_with('~')
        || script.starts_with('@')
        || script.starts_with('+')
        || script.contains('/')
        || script.contains('\\')
    {
        return None;
    }
    let path = Path::new(script);
    if path.components().count() != 1 {
        return None;
    }
    let extension = path.extension()?.to_str()?;
    if !extension.eq_ignore_ascii_case("m") && !extension.eq_ignore_ascii_case("p") {
        return None;
    }
    path.file_stem()?.to_str().filter(|stem| !stem.is_empty())
}

async fn find_run_script(script: &str) -> Result<Option<PathBuf>, String> {
    if let Some(path) =
        find_file_with_extensions(script, RUN_SCRIPT_EXTENSIONS, RUN_BUILTIN_NAME).await?
    {
        let path = crate::builtins::io::repl_fs::pcode::prefer_pcode_source_path(&path).await;
        return Ok(Some(path));
    }

    let Some(stem) = bare_run_file_stem(script) else {
        return Ok(None);
    };
    for candidate in file_candidates(stem, RUN_SCRIPT_EXTENSIONS, RUN_BUILTIN_NAME)? {
        if candidate
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                extension.eq_ignore_ascii_case("p") || extension.eq_ignore_ascii_case("m")
            })
            && path_is_file(&candidate).await
        {
            let candidate =
                crate::builtins::io::repl_fs::pcode::prefer_pcode_source_path(&candidate).await;
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

pub async fn resolve_run_source(value: &Value) -> BuiltinResult<RunScriptSource> {
    let script = value_to_string_scalar(value).ok_or_else(|| run_error(&RUN_ERROR_ARG_TYPE))?;
    if script.is_empty() {
        return Err(run_error(&RUN_ERROR_EMPTY_SCRIPT));
    }

    let path = find_run_script(&script)
        .await
        .map_err(|err| run_error_with_detail(&RUN_ERROR_PATH_RESOLVE, err))?
        .ok_or_else(|| run_error_with_detail(&RUN_ERROR_FILE_NOT_FOUND, format!("'{script}'")))?;

    let text = match crate::builtins::io::repl_fs::pcode::read_source_text_async(&path).await {
        Ok(text) => text,
        Err(crate::builtins::io::repl_fs::pcode::PcodeSourceReadError::InvalidPcode(err)) => {
            return Err(
                crate::builtins::io::repl_fs::pcode::invalid_pcode_runtime_error(format!(
                    "{} ({err})",
                    path.display()
                )),
            );
        }
        Err(crate::builtins::io::repl_fs::pcode::PcodeSourceReadError::Io(err)) => {
            return Err(run_error_with_detail(
                &RUN_ERROR_FILE_READ,
                format!("{} ({err})", path.display()),
            ));
        }
    };

    let display_path = runmat_filesystem::canonicalize_async(&path)
        .await
        .unwrap_or_else(|_| path.clone());
    Ok(RunScriptSource {
        path: display_path.clone(),
        display_name: path_to_string(&display_path),
        text,
    })
}

#[runtime_builtin(name = "run", builtin_path = "crate::builtins::io::repl_fs::run")]
pub fn run_builtin_registered(_args: Vec<Value>) -> crate::BuiltinResult<Value> {
    requires_vm_workspace_context()
}
