//! MATLAB-compatible `unzip` builtin for RunMat.

#[cfg(target_arch = "wasm32")]
use std::io::Cursor;
use std::io::{ErrorKind, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_filesystem::File;
use runmat_filesystem::{self as vfs};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, Value};
use url::Url;
use zip::result::ZipError;
use zip::ZipArchive;

use crate::builtins::common::fs::{expand_user_path, path_to_string};
#[cfg(target_arch = "wasm32")]
use crate::builtins::io::http::transport::{self, HttpMethod, HttpRequest};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "unzip";
const DEFAULT_USER_AGENT: &str = "RunMat unzip/0.0";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_COMPRESSED_ARCHIVE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_TOTAL_EXTRACTED_BYTES: u64 = 10 * 1024 * 1024 * 1024;

pub use runmat_builtins::{
    UNZIP_DESCRIPTOR, UNZIP_ERROR_FILENAME, UNZIP_ERROR_HTTP, UNZIP_ERROR_INVALID_ARGUMENT,
    UNZIP_ERROR_IO, UNZIP_ERROR_OUTPUT, UNZIP_ERROR_OUTPUT_COUNT, UNZIP_ERROR_OUTPUT_FOLDER,
    UNZIP_ERROR_SECURITY, UNZIP_ERROR_ZIP,
};

#[runtime_builtin(
    name = "unzip",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::archive::unzip"
)]
async fn unzip_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count > 1 {
            return Err(unzip_error_with(
                &UNZIP_ERROR_OUTPUT_COUNT,
                format!("unzip: expected at most 1 output, got {out_count}"),
            ));
        }
        let result = evaluate(&args, runtime.as_ref()).await?;
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(Value::OutputList(vec![result.into_value()?]));
    }
    let result = evaluate(&args, runtime.as_ref()).await?;
    result.into_value()
}

pub async fn evaluate(
    args: &[Value],
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<UnzipResult> {
    let request = parse_arguments(args)?;
    check_cancelled(runtime)?;
    let output_folder = resolve_output_folder(request.output_folder.as_deref()).await?;
    vfs::create_dir_all_async(&output_folder)
        .await
        .map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_OUTPUT_FOLDER,
                format!(
                    "unzip: unable to create output folder '{}': {err}",
                    output_folder.display()
                ),
                err,
            )
        })?;
    reject_symlink_path(&output_folder).await?;
    check_cancelled(runtime)?;

    let filenames = match request.archive {
        runmat_builtins::UnzipSourceRoute::Http(url) => {
            extract_remote_archive(&url, &output_folder, runtime).await?
        }
        runmat_builtins::UnzipSourceRoute::LocalPath(path) => {
            let path = resolve_archive_path(&path)?;
            extract_local_archive(&path, &output_folder).await?
        }
    };
    check_cancelled(runtime)?;

    Ok(UnzipResult { filenames })
}

fn check_cancelled(runtime: Option<&crate::context::RuntimeContext>) -> BuiltinResult<()> {
    if runtime.is_some_and(crate::context::RuntimeContext::is_cancelled) {
        Err(crate::runtime_error::semantic_error(
            "RunMat:Cancelled",
            "unzip invocation was cancelled",
        ))
    } else {
        Ok(())
    }
}

mod parse;
use self::parse::parse_arguments;

mod source;
#[cfg(test)]
use self::source::ensure_compressed_size_within_limit;
use self::source::{
    extract_local_archive, extract_remote_archive, resolve_archive_path, resolve_output_folder,
};

mod extraction;
#[cfg(target_arch = "wasm32")]
use self::extraction::extract_archive;
#[cfg(not(target_arch = "wasm32"))]
use self::extraction::extract_archive_file_on_worker;
use self::extraction::reject_symlink_path;

#[derive(Debug, Clone)]
pub struct UnzipResult {
    filenames: Vec<String>,
}

impl UnzipResult {
    fn into_value(self) -> BuiltinResult<Value> {
        let rows = self.filenames.len();
        let values = self
            .filenames
            .into_iter()
            .map(|path| Value::CharArray(CharArray::new_row(&path)))
            .collect::<Vec<_>>();
        crate::make_cell(values, rows, 1)
            .map_err(|err| unzip_error_with(&UNZIP_ERROR_OUTPUT, format!("unzip: {err}")))
    }
}

fn map_zip_error(err: ZipError) -> RuntimeError {
    unzip_error_with_source(
        &UNZIP_ERROR_ZIP,
        format!("unzip: unable to read ZIP archive: {err}"),
        err,
    )
}

fn unzip_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn unzip_error_with_source<E>(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: crate::runtime_error::RuntimeErrorSourceBound,
{
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
#[path = "unzip/tests.rs"]
mod tests;
