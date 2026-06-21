//! MATLAB-compatible `unzip` builtin for RunMat.

use std::io::Cursor;
use std::io::{ErrorKind, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_filesystem::{self as vfs, File};
use runmat_macros::runtime_builtin;
use url::Url;
use zip::result::ZipError;
use zip::ZipArchive;

use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
#[cfg(target_arch = "wasm32")]
use crate::builtins::io::http::transport::{self, HttpMethod, HttpRequest};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "unzip";
const DEFAULT_USER_AGENT: &str = "RunMat unzip/0.0";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_COMPRESSED_ARCHIVE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_TOTAL_EXTRACTED_BYTES: u64 = 10 * 1024 * 1024 * 1024;

const UNZIP_OUTPUT_FILES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filenames",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array of extracted file paths.",
}];
const UNZIP_INPUT_ARCHIVE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "zipfilename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "ZIP archive path or HTTP/HTTPS URL.",
}];
const UNZIP_INPUT_ARCHIVE_OUTPUT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "zipfilename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "ZIP archive path or HTTP/HTTPS URL.",
    },
    BuiltinParamDescriptor {
        name: "outputfolder",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Destination folder for extracted files.",
    },
];
const UNZIP_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "unzip(zipfilename)",
        inputs: &UNZIP_INPUT_ARCHIVE,
        outputs: &UNZIP_OUTPUT_FILES,
    },
    BuiltinSignatureDescriptor {
        label: "unzip(zipfilename, outputfolder)",
        inputs: &UNZIP_INPUT_ARCHIVE_OUTPUT,
        outputs: &UNZIP_OUTPUT_FILES,
    },
    BuiltinSignatureDescriptor {
        label: "filenames = unzip(___)",
        inputs: &UNZIP_INPUT_ARCHIVE_OUTPUT,
        outputs: &UNZIP_OUTPUT_FILES,
    },
];

const UNZIP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.INVALID_ARGUMENT",
    identifier: Some("RunMat:unzip:InvalidArgument"),
    when: "Arguments do not match supported unzip invocation forms.",
    message: "unzip: invalid argument",
};
const UNZIP_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.FILENAME",
    identifier: Some("RunMat:unzip:Filename"),
    when: "ZIP filename or URL is invalid.",
    message: "unzip: invalid ZIP filename",
};
const UNZIP_ERROR_OUTPUT_FOLDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.OUTPUT_FOLDER",
    identifier: Some("RunMat:unzip:OutputFolder"),
    when: "Output folder is invalid or cannot be created.",
    message: "unzip: invalid output folder",
};
const UNZIP_ERROR_HTTP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.HTTP",
    identifier: Some("RunMat:unzip:Http"),
    when: "Remote ZIP archive cannot be downloaded.",
    message: "unzip: download failed",
};
const UNZIP_ERROR_ZIP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.ZIP",
    identifier: Some("RunMat:unzip:Zip"),
    when: "ZIP archive cannot be decoded or contains unsupported entries.",
    message: "unzip: invalid ZIP archive",
};
const UNZIP_ERROR_SECURITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.SECURITY",
    identifier: Some("RunMat:unzip:UnsafePath"),
    when: "ZIP entry would escape the destination folder.",
    message: "unzip: unsafe archive path",
};
const UNZIP_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.IO",
    identifier: Some("RunMat:unzip:Io"),
    when: "Extraction fails due to filesystem I/O.",
    message: "unzip: extraction failed",
};
const UNZIP_ERROR_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.OUTPUT",
    identifier: Some("RunMat:unzip:Output"),
    when: "Returned filename cell array cannot be materialized.",
    message: "unzip: output materialization failed",
};
const UNZIP_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNZIP.OUTPUT_COUNT",
    identifier: Some("RunMat:unzip:OutputCount"),
    when: "Caller requests more outputs than unzip supports.",
    message: "unzip: unsupported output count",
};
const UNZIP_ERRORS: [BuiltinErrorDescriptor; 9] = [
    UNZIP_ERROR_INVALID_ARGUMENT,
    UNZIP_ERROR_FILENAME,
    UNZIP_ERROR_OUTPUT_FOLDER,
    UNZIP_ERROR_HTTP,
    UNZIP_ERROR_ZIP,
    UNZIP_ERROR_SECURITY,
    UNZIP_ERROR_IO,
    UNZIP_ERROR_OUTPUT,
    UNZIP_ERROR_OUTPUT_COUNT,
];

pub const UNZIP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNZIP_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UNZIP_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::archive::unzip")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "unzip",
    op_kind: GpuOpKind::Custom("io-unzip"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Archive extraction runs through the active filesystem provider and gathers path arguments before I/O.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::archive::unzip")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "unzip",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Archive extraction is host I/O and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "unzip",
    category = "io/archive",
    summary = "Extract files from ZIP archives.",
    keywords = "unzip,zip,archive,extract,url",
    accel = "cpu",
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::unzip_type),
    descriptor(crate::builtins::io::archive::unzip::UNZIP_DESCRIPTOR),
    builtin_path = "crate::builtins::io::archive::unzip"
)]
async fn unzip_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count > 1 {
            return Err(unzip_error_with(
                &UNZIP_ERROR_OUTPUT_COUNT,
                format!("unzip: expected at most 1 output, got {out_count}"),
            ));
        }
        let result = evaluate(&args).await?;
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(Value::OutputList(vec![result.into_value()?]));
    }
    let result = evaluate(&args).await?;
    result.into_value()
}

pub async fn evaluate(args: &[Value]) -> BuiltinResult<UnzipResult> {
    let gathered = gather_arguments(args).await?;
    let request = parse_arguments(&gathered)?;
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

    let filenames = if is_http_url(&request.archive) {
        extract_remote_archive(&request.archive, &output_folder).await?
    } else {
        let path = resolve_archive_path(&request.archive)?;
        let file = File::open(&path).map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_IO,
                format!("unzip: unable to open '{}': {err}", path.display()),
                err,
            )
        })?;
        extract_archive(file, &output_folder).await?
    };

    Ok(UnzipResult { filenames })
}

async fn gather_arguments(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(gathered)
}

struct UnzipRequest {
    archive: String,
    output_folder: Option<String>,
}

fn parse_arguments(args: &[Value]) -> BuiltinResult<UnzipRequest> {
    match args {
        [archive] => Ok(UnzipRequest {
            archive: value_to_string_scalar(archive, &UNZIP_ERROR_FILENAME)?,
            output_folder: None,
        }),
        [archive, output_folder] => Ok(UnzipRequest {
            archive: value_to_string_scalar(archive, &UNZIP_ERROR_FILENAME)?,
            output_folder: Some(value_to_string_scalar(
                output_folder,
                &UNZIP_ERROR_OUTPUT_FOLDER,
            )?),
        }),
        [] => Err(unzip_error_with(
            &UNZIP_ERROR_INVALID_ARGUMENT,
            "unzip: ZIP filename is required",
        )),
        _ => Err(unzip_error_with(
            &UNZIP_ERROR_INVALID_ARGUMENT,
            "unzip: expected ZIP filename and optional output folder",
        )),
    }
}

async fn extract_remote_archive(
    url_text: &str,
    output_folder: &Path,
) -> BuiltinResult<Vec<String>> {
    let bytes = download_archive(url_text)?;
    extract_archive(Cursor::new(bytes), output_folder).await
}

fn value_to_string_scalar(
    value: &Value,
    descriptor: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        Value::StringArray(strings) if strings.data.len() == 1 => Ok(strings.data[0].clone()),
        _ => Err(unzip_error_with(
            descriptor,
            "unzip: expected a string scalar or character vector",
        )),
    }
    .and_then(|text| {
        if text.trim().is_empty() {
            Err(unzip_error_with(
                descriptor,
                "unzip: path arguments must not be empty",
            ))
        } else {
            Ok(text)
        }
    })
}

fn is_http_url(text: &str) -> bool {
    let Ok(url) = Url::parse(text) else {
        return false;
    };
    matches!(url.scheme(), "http" | "https")
}

fn resolve_archive_path(text: &str) -> BuiltinResult<PathBuf> {
    let expanded = expand_user_path(text.trim(), BUILTIN_NAME)
        .map_err(|msg| unzip_error_with(&UNZIP_ERROR_FILENAME, msg))?;
    Ok(PathBuf::from(expanded))
}

async fn resolve_output_folder(text: Option<&str>) -> BuiltinResult<PathBuf> {
    match text {
        Some(raw) => {
            let expanded = expand_user_path(raw.trim(), BUILTIN_NAME)
                .map_err(|msg| unzip_error_with(&UNZIP_ERROR_OUTPUT_FOLDER, msg))?;
            let path = PathBuf::from(expanded);
            if path.is_absolute() {
                Ok(path)
            } else {
                let current = vfs::current_dir().map_err(|err| {
                    unzip_error_with_source(
                        &UNZIP_ERROR_OUTPUT_FOLDER,
                        format!("unzip: unable to resolve current folder: {err}"),
                        err,
                    )
                })?;
                Ok(current.join(path))
            }
        }
        None => vfs::current_dir().map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_OUTPUT_FOLDER,
                format!("unzip: unable to resolve current folder: {err}"),
                err,
            )
        }),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn download_archive(url_text: &str) -> BuiltinResult<Vec<u8>> {
    let url = Url::parse(url_text).map_err(|err| {
        unzip_error_with_source(
            &UNZIP_ERROR_FILENAME,
            format!("unzip: invalid URL '{url_text}': {err}"),
            err,
        )
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(unzip_error_with(
            &UNZIP_ERROR_FILENAME,
            "unzip: URL archives must use HTTP or HTTPS",
        ));
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(DEFAULT_TIMEOUT)
        .user_agent(DEFAULT_USER_AGENT)
        .build()
        .map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_HTTP,
                format!("unzip: unable to create HTTP client: {err}"),
                err,
            )
        })?;
    let mut response = client.get(url.clone()).send().map_err(|err| {
        unzip_error_with_source(
            &UNZIP_ERROR_HTTP,
            format!("unzip: request to {url} failed: {err}"),
            err,
        )
    })?;
    let status = response.status();
    if !status.is_success() {
        return Err(unzip_error_with(
            &UNZIP_ERROR_HTTP,
            format!(
                "unzip: request to {url} failed with HTTP status {}",
                status.as_u16()
            ),
        ));
    }
    if let Some(length) = response.content_length() {
        ensure_compressed_size_within_limit(length)?;
    }

    let mut bytes = Vec::new();
    let mut downloaded = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = response.read(&mut buffer).map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_HTTP,
                format!("unzip: unable to read response from {url}: {err}"),
                err,
            )
        })?;
        if read == 0 {
            break;
        }
        downloaded = downloaded.checked_add(read as u64).ok_or_else(|| {
            unzip_error_with(&UNZIP_ERROR_HTTP, "unzip: downloaded archive is too large")
        })?;
        ensure_compressed_size_within_limit(downloaded)?;
        bytes.extend_from_slice(&buffer[..read]);
    }
    Ok(bytes)
}

#[cfg(target_arch = "wasm32")]
fn download_archive(url_text: &str) -> BuiltinResult<Vec<u8>> {
    let url = Url::parse(url_text).map_err(|err| {
        unzip_error_with_source(
            &UNZIP_ERROR_FILENAME,
            format!("unzip: invalid URL '{url_text}': {err}"),
            err,
        )
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(unzip_error_with(
            &UNZIP_ERROR_FILENAME,
            "unzip: URL archives must use HTTP or HTTPS",
        ));
    }
    let request = HttpRequest {
        url,
        method: HttpMethod::Get,
        headers: Vec::new(),
        body: None,
        timeout: DEFAULT_TIMEOUT,
        user_agent: DEFAULT_USER_AGENT.to_string(),
    };
    let response = transport::send_request(&request).map_err(|err| {
        unzip_error_with_source(&UNZIP_ERROR_HTTP, err.message_with_prefix("unzip"), err)
    })?;
    ensure_compressed_size_within_limit(response.body.len() as u64)?;
    Ok(response.body)
}

fn ensure_compressed_size_within_limit(size: u64) -> BuiltinResult<()> {
    if size > MAX_COMPRESSED_ARCHIVE_BYTES {
        return Err(unzip_error_with(
            &UNZIP_ERROR_HTTP,
            format!(
                "unzip: compressed archive has {size} bytes, exceeding the RunMat limit of {MAX_COMPRESSED_ARCHIVE_BYTES}"
            ),
        ));
    }
    Ok(())
}

async fn extract_archive<R>(reader: R, output_folder: &Path) -> BuiltinResult<Vec<String>>
where
    R: Read + Seek,
{
    let mut archive = ZipArchive::new(reader).map_err(map_zip_error)?;
    let mut filenames = Vec::new();
    let mut total_written = 0u64;

    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(map_zip_error)?;
        let relative = safe_archive_path(entry.name())?;
        if relative.as_os_str().is_empty() {
            continue;
        }
        let output_path = output_folder.join(&relative);
        reject_symlink_components(output_folder, &relative).await?;

        if entry.is_dir() || entry.name().ends_with('/') || entry.name().ends_with('\\') {
            vfs::create_dir_all_async(&output_path)
                .await
                .map_err(|err| {
                    unzip_error_with_source(
                        &UNZIP_ERROR_IO,
                        format!(
                            "unzip: unable to create folder '{}': {err}",
                            output_path.display()
                        ),
                        err,
                    )
                })?;
            continue;
        }

        if let Some(parent) = output_path.parent() {
            vfs::create_dir_all_async(parent).await.map_err(|err| {
                unzip_error_with_source(
                    &UNZIP_ERROR_IO,
                    format!(
                        "unzip: unable to create folder '{}': {err}",
                        parent.display()
                    ),
                    err,
                )
            })?;
        }
        reject_symlink_components(output_folder, &relative).await?;

        let mut out = File::create(&output_path).map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_IO,
                format!("unzip: unable to create '{}': {err}", output_path.display()),
                err,
            )
        })?;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = entry.read(&mut buffer).map_err(|err| {
                unzip_error_with_source(
                    &UNZIP_ERROR_ZIP,
                    format!(
                        "unzip: unable to read archive entry '{}': {err}",
                        entry.name()
                    ),
                    err,
                )
            })?;
            if read == 0 {
                break;
            }
            total_written = total_written.checked_add(read as u64).ok_or_else(|| {
                unzip_error_with(&UNZIP_ERROR_ZIP, "unzip: total extracted size is too large")
            })?;
            if total_written > MAX_TOTAL_EXTRACTED_BYTES {
                return Err(unzip_error_with(
                    &UNZIP_ERROR_ZIP,
                    "unzip: total extracted size exceeds RunMat safety limit",
                ));
            }
            out.write_all(&buffer[..read]).map_err(|err| {
                unzip_error_with_source(
                    &UNZIP_ERROR_IO,
                    format!("unzip: unable to write '{}': {err}", output_path.display()),
                    err,
                )
            })?;
        }
        filenames.push(path_to_string(&output_path));
    }

    Ok(filenames)
}

async fn reject_symlink_path(path: &Path) -> BuiltinResult<()> {
    match vfs::symlink_metadata_async(path).await {
        Ok(metadata) if metadata.is_symlink() => Err(unzip_error_with(
            &UNZIP_ERROR_SECURITY,
            format!("unzip: output path '{}' is a symbolic link", path.display()),
        )),
        Ok(_) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => Err(unzip_error_with_source(
            &UNZIP_ERROR_IO,
            format!("unzip: unable to inspect '{}': {err}", path.display()),
            err,
        )),
    }
}

async fn reject_symlink_components(root: &Path, relative: &Path) -> BuiltinResult<()> {
    let mut current = root.to_path_buf();
    reject_symlink_path(&current).await?;
    for component in relative.components() {
        current.push(component.as_os_str());
        reject_symlink_path(&current).await?;
    }
    Ok(())
}

fn safe_archive_path(name: &str) -> BuiltinResult<PathBuf> {
    if name.is_empty() || name.as_bytes().contains(&0) {
        return Err(unzip_error_with(
            &UNZIP_ERROR_SECURITY,
            "unzip: archive entry name is empty or invalid",
        ));
    }
    let normalized = name.replace('\\', "/");
    if normalized.starts_with('/') || normalized.starts_with('~') || normalized.starts_with("//") {
        return Err(unzip_error_with(
            &UNZIP_ERROR_SECURITY,
            format!("unzip: archive entry '{name}' uses an absolute or user-relative path"),
        ));
    }

    let mut path = PathBuf::new();
    for part in normalized.split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." || part.contains(':') {
            return Err(unzip_error_with(
                &UNZIP_ERROR_SECURITY,
                format!("unzip: archive entry '{name}' would escape the output folder"),
            ));
        }
        path.push(part);
    }
    Ok(path)
}

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
            .map(Value::from)
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
    E: std::error::Error + Send + Sync + 'static,
{
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("unzip: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::CellArray;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use tempfile::tempdir;
    use zip::write::SimpleFileOptions;

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn unique_zip_path(stem: &str) -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("{stem}_{id}.zip"))
    }

    fn write_test_zip(path: &Path, entries: &[(&str, &[u8])]) {
        let file = std::fs::File::create(path).expect("create zip");
        let mut zip = zip::ZipWriter::new(file);
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for (name, bytes) in entries {
            zip.start_file(name, options).expect("start zip entry");
            zip.write_all(bytes).expect("write zip entry");
        }
        zip.finish().expect("finish zip");
    }

    fn write_test_zip_with_dirs(path: &Path, dirs: &[&str], entries: &[(&str, &[u8])]) {
        let file = std::fs::File::create(path).expect("create zip");
        let mut zip = zip::ZipWriter::new(file);
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        for name in dirs {
            zip.add_directory(*name, options).expect("add zip dir");
        }
        for (name, bytes) in entries {
            zip.start_file(name, options).expect("start zip entry");
            zip.write_all(bytes).expect("write zip entry");
        }
        zip.finish().expect("finish zip");
    }

    fn run_unzip(args: Vec<Value>, outputs: Option<usize>) -> BuiltinResult<Value> {
        let _guard = outputs.map(|count| crate::output_count::push_output_count(Some(count)));
        block_on(unzip_builtin(args))
    }

    fn cell_strings(value: Value) -> Vec<String> {
        let Value::Cell(CellArray {
            data, rows, cols, ..
        }) = value
        else {
            panic!("expected cell array");
        };
        assert_eq!(cols, 1);
        assert_eq!(rows, data.len());
        data.into_iter()
            .map(|handle| {
                let value = unsafe { &*handle.as_raw() };
                String::try_from(value).expect("cell string")
            })
            .collect()
    }

    fn spawn_server<F>(handler: F) -> String
    where
        F: FnOnce(TcpStream) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                handler(stream);
            }
        });
        format!("http://{}", addr)
    }

    fn respond_with(mut stream: TcpStream, content_type: &str, body: &[u8]) {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: {}\r\nConnection: close\r\n\r\n",
            body.len(),
            content_type
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.write_all(body);
    }

    fn drain_request(stream: &mut TcpStream) {
        let mut buffer = [0u8; 256];
        while let Ok(read) = stream.read(&mut buffer) {
            if read == 0 || buffer[..read].windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
    }

    #[test]
    fn unzip_registers_public_descriptor() {
        assert!(runmat_builtins::builtin_function_by_name("unzip").is_some());
        let labels = UNZIP_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"unzip(zipfilename)"));
        assert!(labels.contains(&"unzip(zipfilename, outputfolder)"));
        assert!(labels.contains(&"filenames = unzip(___)"));
    }

    #[test]
    fn unzip_extracts_local_archive_to_output_folder() {
        let archive = unique_zip_path("unzip_local");
        write_test_zip(
            &archive,
            &[("data/a.txt", b"alpha"), ("data/nested/b.txt", b"beta")],
        );
        let out = tempdir().expect("output folder");

        let value = run_unzip(
            vec![
                Value::from(archive.to_string_lossy().to_string()),
                Value::from(out.path().to_string_lossy().to_string()),
            ],
            None,
        )
        .expect("unzip");

        let extracted = cell_strings(value);
        let expected_a = out.path().join("data/a.txt").to_string_lossy().to_string();
        let expected_b = out
            .path()
            .join("data/nested/b.txt")
            .to_string_lossy()
            .to_string();
        assert_eq!(extracted, vec![expected_a, expected_b]);
        assert_eq!(
            fs::read_to_string(out.path().join("data/a.txt")).unwrap(),
            "alpha"
        );
        assert_eq!(
            fs::read_to_string(out.path().join("data/nested/b.txt")).unwrap(),
            "beta"
        );
        let _ = fs::remove_file(archive);
    }

    #[test]
    fn unzip_zero_requested_outputs_returns_empty_output_list() {
        let archive = unique_zip_path("unzip_zero_outputs");
        write_test_zip(&archive, &[("a.txt", b"alpha")]);
        let out = tempdir().expect("output folder");

        let value = run_unzip(
            vec![
                Value::from(archive.to_string_lossy().to_string()),
                Value::from(out.path().to_string_lossy().to_string()),
            ],
            Some(0),
        )
        .expect("unzip");

        assert_eq!(value, Value::OutputList(Vec::new()));
        assert_eq!(
            fs::read_to_string(out.path().join("a.txt")).unwrap(),
            "alpha"
        );
        let _ = fs::remove_file(archive);
    }

    #[test]
    fn unzip_rejects_too_many_outputs_before_extracting() {
        let archive = unique_zip_path("unzip_too_many_outputs");
        write_test_zip(&archive, &[("a.txt", b"alpha")]);
        let out = tempdir().expect("output folder");

        let err = run_unzip(
            vec![
                Value::from(archive.to_string_lossy().to_string()),
                Value::from(out.path().to_string_lossy().to_string()),
            ],
            Some(2),
        )
        .expect_err("too many outputs");

        assert_eq!(err.identifier(), Some("RunMat:unzip:OutputCount"));
        assert!(!out.path().join("a.txt").exists());
        let _ = fs::remove_file(archive);
    }

    #[test]
    fn unzip_rejects_parent_traversal_entries() {
        let archive = unique_zip_path("unzip_traversal");
        write_test_zip(&archive, &[("../escape.txt", b"bad")]);
        let out = tempdir().expect("output folder");

        let err = run_unzip(
            vec![
                Value::from(archive.to_string_lossy().to_string()),
                Value::from(out.path().to_string_lossy().to_string()),
            ],
            None,
        )
        .expect_err("unsafe entry");

        assert_eq!(err.identifier(), Some("RunMat:unzip:UnsafePath"));
        assert!(!out.path().join("escape.txt").exists());
        let _ = fs::remove_file(archive);
    }

    #[test]
    fn unzip_rejects_absolute_drive_and_backslash_traversal_entries() {
        for entry_name in ["/abs.txt", "C:/abs.txt", "nested\\..\\escape.txt"] {
            let archive = unique_zip_path("unzip_unsafe_name");
            write_test_zip(&archive, &[(entry_name, b"bad")]);
            let out = tempdir().expect("output folder");

            let err = run_unzip(
                vec![
                    Value::from(archive.to_string_lossy().to_string()),
                    Value::from(out.path().to_string_lossy().to_string()),
                ],
                None,
            )
            .expect_err("unsafe entry");

            assert_eq!(err.identifier(), Some("RunMat:unzip:UnsafePath"));
            let _ = fs::remove_file(archive);
        }
    }

    #[cfg(unix)]
    #[test]
    fn unzip_rejects_existing_symlink_components() {
        let archive = unique_zip_path("unzip_symlink_component");
        write_test_zip(&archive, &[("link/escape.txt", b"bad")]);
        let out = tempdir().expect("output folder");
        let outside = tempdir().expect("outside folder");
        std::os::unix::fs::symlink(outside.path(), out.path().join("link")).expect("symlink");

        let err = run_unzip(
            vec![
                Value::from(archive.to_string_lossy().to_string()),
                Value::from(out.path().to_string_lossy().to_string()),
            ],
            None,
        )
        .expect_err("symlink component");

        assert_eq!(err.identifier(), Some("RunMat:unzip:UnsafePath"));
        assert!(!outside.path().join("escape.txt").exists());
        let _ = fs::remove_file(archive);
    }

    #[test]
    fn unzip_directory_only_archive_returns_empty_cell() {
        let archive = unique_zip_path("unzip_dir_only");
        write_test_zip_with_dirs(&archive, &["folder/"], &[]);
        let out = tempdir().expect("output folder");

        let value = run_unzip(
            vec![
                Value::from(archive.to_string_lossy().to_string()),
                Value::from(out.path().to_string_lossy().to_string()),
            ],
            None,
        )
        .expect("unzip");

        let extracted = cell_strings(value);
        assert!(extracted.is_empty());
        assert!(out.path().join("folder").is_dir());
        let _ = fs::remove_file(archive);
    }

    #[test]
    fn unzip_downloads_url_archive() {
        let archive = unique_zip_path("unzip_url");
        write_test_zip(&archive, &[("remote.txt", b"downloaded")]);
        let bytes = fs::read(&archive).expect("read archive");
        let url = spawn_server(move |mut stream| {
            drain_request(&mut stream);
            respond_with(stream, "application/zip", &bytes);
        });
        let out = tempdir().expect("output folder");

        run_unzip(
            vec![
                Value::from(url),
                Value::from(out.path().to_string_lossy().to_string()),
            ],
            None,
        )
        .expect("unzip URL");

        assert_eq!(
            fs::read_to_string(out.path().join("remote.txt")).unwrap(),
            "downloaded"
        );
        let _ = fs::remove_file(archive);
    }

    #[test]
    fn unzip_http_status_errors() {
        let url = spawn_server(move |mut stream| {
            drain_request(&mut stream);
            let response =
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            let _ = stream.write_all(response.as_bytes());
        });
        let out = tempdir().expect("output folder");

        let err = run_unzip(
            vec![
                Value::from(url),
                Value::from(out.path().to_string_lossy().to_string()),
            ],
            None,
        )
        .expect_err("http status");

        assert_eq!(err.identifier(), Some("RunMat:unzip:Http"));
    }

    #[test]
    fn compressed_size_limit_is_enforced() {
        let err = ensure_compressed_size_within_limit(MAX_COMPRESSED_ARCHIVE_BYTES + 1)
            .expect_err("compressed limit");
        assert_eq!(err.identifier(), Some("RunMat:unzip:Http"));
    }
}
