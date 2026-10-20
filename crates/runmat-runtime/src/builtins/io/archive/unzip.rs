//! MATLAB-compatible `unzip` builtin for RunMat.

use std::io::{ErrorKind, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_execution::{Digest, WholeWorkObservationOutcome};
use runmat_filesystem::File;
use runmat_filesystem::{self as vfs};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, Value};
use zip::result::ZipError;
use zip::ZipArchive;

use crate::builtins::common::fs::{expand_user_path, path_to_string};
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
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count > 1 {
            return Err(unzip_error_with(
                &UNZIP_ERROR_OUTPUT_COUNT,
                format!("unzip: expected at most 1 output, got {out_count}"),
            ));
        }
        let output = if out_count == 0 {
            crate::context::host_effect::HostEffectOutput::EffectOnly
        } else {
            crate::context::host_effect::HostEffectOutput::ReturnedValue
        };
        let result = evaluate_with_output(args.view(), runtime.as_ref(), output).await?;
        if out_count == 0 {
            return result.finish_effect_only();
        }
        return result.into_value_with(|value| Value::output_list(vec![value]));
    }
    let result = evaluate_view(args.view(), runtime.as_ref()).await?;
    result.into_value_with(std::convert::identity)
}

pub async fn evaluate(
    args: &[Value],
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<UnzipResult> {
    evaluate_view(crate::call::FunctionArgumentView::from_slice(args), runtime).await
}

pub async fn evaluate_view(
    args: crate::call::FunctionArgumentView<'_>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<UnzipResult> {
    evaluate_with_output(
        args,
        runtime,
        crate::context::host_effect::HostEffectOutput::ReturnedValue,
    )
    .await
}

async fn evaluate_with_output(
    args: crate::call::FunctionArgumentView<'_>,
    runtime: Option<&crate::context::RuntimeContext>,
    output: crate::context::host_effect::HostEffectOutput,
) -> BuiltinResult<UnzipResult> {
    use crate::context::host_effect::{
        select_host_effect, HostEffectRequest, HostEffectSource, HostEffectWorkload,
    };
    let request = parse_arguments(args)?;
    check_cancelled(runtime)?;
    let resolved = ResolvedUnzipRequest::from_parsed(request)?;
    let source = match &resolved.archive {
        ResolvedArchiveSource::Http(url) => HostEffectSource::HttpArchive(url.as_str()),
        ResolvedArchiveSource::LocalPath(path) => HostEffectSource::LocalArchive(path.as_path()),
    };
    let workload = HostEffectWorkload::ArchiveExtract {
        source,
        destination: Some(resolved.output_folder.as_path()),
        output,
    };
    let receipt = select_host_effect(
        runtime,
        HostEffectRequest {
            operation: BUILTIN_NAME,
            workload,
            domain_facts: Digest::sha256(workload.identity_bytes()),
            filesystem: &resolved.filesystem,
        },
    )?;
    // Capture the invocation's command owner before any archive-source await.
    // Neither the later worker thread nor a browser rollback uses ambient state.
    let command_origin = runtime
        .and_then(crate::context::RuntimeContext::command_profiling)
        .cloned();
    let resource_lease = receipt
        .as_ref()
        .map(|receipt| receipt.resource_liveness_lease());
    let execution = async {
        check_resource_lease(&resource_lease)?;
        let filenames = match resolved.archive {
            ResolvedArchiveSource::Http(url) => {
                extract_remote_archive(
                    &url,
                    &resolved.output_folder,
                    runtime,
                    command_origin.clone(),
                    resolved.filesystem,
                    &resource_lease,
                )
                .await?
            }
            ResolvedArchiveSource::LocalPath(path) => {
                prepare_output_folder(
                    &resolved.output_folder,
                    runtime,
                    resolved.filesystem.as_ref(),
                    &resource_lease,
                )
                .await?;
                extract_local_archive(
                    &path,
                    &resolved.output_folder,
                    runtime,
                    command_origin.clone(),
                    resolved.filesystem,
                    &resource_lease,
                )
                .await?
            }
        };
        check_cancelled(runtime)?;
        if let Some(receipt) = receipt.as_ref() {
            receipt.checkpoint()?;
        }
        Ok(filenames)
    }
    .await;
    match execution {
        Ok(filenames) => Ok(UnzipResult { filenames, receipt }),
        Err(error) => {
            if let Some(receipt) = receipt {
                receipt.finish(WholeWorkObservationOutcome::OperationalFailure);
            }
            Err(error)
        }
    }
}

enum ResolvedArchiveSource {
    Http(url::Url),
    LocalPath(PathBuf),
}

struct ResolvedUnzipRequest {
    archive: ResolvedArchiveSource,
    output_folder: PathBuf,
    filesystem: std::sync::Arc<dyn vfs::FsProvider>,
}

impl ResolvedUnzipRequest {
    fn from_parsed(request: parse::UnzipRequest) -> BuiltinResult<Self> {
        // Output resolution precedes local-source resolution, retaining the
        // established diagnostic order. A relative/default path captures the
        // provider and effective CWD together; the same owner and absolute
        // paths then serve evidence and execution after suspension.
        let mut captured_filesystem = None;
        let output_folder =
            resolve_output_folder(request.output_folder.as_deref(), &mut captured_filesystem)?;
        let archive = match request.archive {
            runmat_builtins::UnzipSourceRoute::Http(url) => ResolvedArchiveSource::Http(url),
            runmat_builtins::UnzipSourceRoute::LocalPath(path) => ResolvedArchiveSource::LocalPath(
                resolve_archive_path(&path, &mut captured_filesystem)?,
            ),
        };
        let filesystem = captured_filesystem
            .map(|owner| owner.provider)
            .unwrap_or_else(vfs::current_provider);
        Ok(Self {
            archive,
            output_folder,
            filesystem,
        })
    }
}

async fn prepare_output_folder(
    output_folder: &Path,
    runtime: Option<&crate::context::RuntimeContext>,
    provider: &dyn vfs::FsProvider,
    resource_lease: &ArchiveResourceLease,
) -> BuiltinResult<()> {
    check_cancelled(runtime)?;
    check_resource_lease(resource_lease)?;
    provider
        .create_dir_all(output_folder)
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
    check_resource_lease(resource_lease)?;
    reject_symlink_path(provider, output_folder, resource_lease).await?;
    check_cancelled(runtime)
}

type ArchiveResourceLease =
    Option<std::sync::Arc<dyn crate::context::host_effect::HostEffectResourceLivenessLease>>;

fn check_resource_lease(lease: &ArchiveResourceLease) -> BuiltinResult<()> {
    if lease.as_ref().is_some_and(|lease| !lease.is_current()) {
        Err(crate::context::host_effect::stale_resource())
    } else {
        Ok(())
    }
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

pub struct UnzipResult {
    filenames: Vec<String>,
    receipt: Option<crate::context::host_effect::HostEffectReceipt>,
}

impl UnzipResult {
    fn finish_effect_only(self) -> BuiltinResult<Value> {
        let value = Value::output_list(Vec::new());
        if let Some(receipt) = self.receipt {
            receipt.checkpoint()?;
            receipt.finish(WholeWorkObservationOutcome::Success);
        }
        Ok(value)
    }

    fn into_value_with(self, wrap: impl FnOnce(Value) -> Value) -> BuiltinResult<Value> {
        let rows = self.filenames.len();
        let values = self
            .filenames
            .into_iter()
            .map(|path| Value::CharArray(CharArray::new_row(&path)))
            .collect::<Vec<_>>();
        let value = crate::make_cell(values, rows, 1)
            .map(wrap)
            .map_err(|err| unzip_error_with(&UNZIP_ERROR_OUTPUT, format!("unzip: {err}")));
        if let Some(receipt) = self.receipt {
            if value.is_ok() {
                receipt.checkpoint()?;
            }
            receipt.finish(if value.is_ok() {
                WholeWorkObservationOutcome::Success
            } else {
                WholeWorkObservationOutcome::OperationalFailure
            });
        }
        value
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
#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "unzip/path_authority_tests.rs"]
mod path_authority_tests;
