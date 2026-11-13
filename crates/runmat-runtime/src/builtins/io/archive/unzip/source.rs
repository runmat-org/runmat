use super::*;

mod http;
pub(super) use http::extract_remote_archive;

#[cfg(test)]
#[path = "source/tests.rs"]
mod tests;

#[cfg(not(target_arch = "wasm32"))]
pub(super) async fn extract_local_archive(
    path: &Path,
    output_folder: &Path,
    runtime: Option<&crate::context::RuntimeContext>,
    command_origin: Option<runmat_execution::profiling::CommandProfilingReceipt>,
    provider: std::sync::Arc<dyn vfs::FsProvider>,
    resource_lease: &ArchiveResourceLease,
) -> BuiltinResult<Vec<String>> {
    let temp = local_archive_to_temp_file(path, runtime, provider.as_ref(), resource_lease).await?;
    extract_archive_file_on_worker(
        temp,
        output_folder.to_path_buf(),
        runtime.map(crate::context::RuntimeContext::cancellation),
        command_origin,
        provider,
        resource_lease.clone(),
    )
    .await
}

#[cfg(target_arch = "wasm32")]
pub(super) async fn extract_local_archive(
    path: &Path,
    output_folder: &Path,
    runtime: Option<&crate::context::RuntimeContext>,
    command_origin: Option<runmat_execution::profiling::CommandProfilingReceipt>,
    provider: std::sync::Arc<dyn vfs::FsProvider>,
    resource_lease: &ArchiveResourceLease,
) -> BuiltinResult<Vec<String>> {
    check_resource_lease(resource_lease)?;
    let file = File::open_on(provider.as_ref(), path).map_err(|err| {
        unzip_error_with_source(
            &UNZIP_ERROR_IO,
            format!("unzip: unable to open '{}': {err}", path.display()),
            err,
        )
    })?;
    check_resource_lease(resource_lease)?;
    extract_archive(
        file,
        output_folder,
        runtime,
        command_origin,
        provider,
        resource_lease.clone(),
    )
    .await
}

#[cfg(not(target_arch = "wasm32"))]
async fn local_archive_to_temp_file(
    path: &Path,
    runtime: Option<&crate::context::RuntimeContext>,
    provider: &dyn vfs::FsProvider,
    resource_lease: &ArchiveResourceLease,
) -> BuiltinResult<TempArchive> {
    check_resource_lease(resource_lease)?;
    let mut input = File::open_async_on(provider, path).await.map_err(|err| {
        unzip_error_with_source(
            &UNZIP_ERROR_IO,
            format!("unzip: unable to open '{}': {err}", path.display()),
            err,
        )
    })?;
    check_resource_lease(resource_lease)?;
    if let Ok(metadata) = input.metadata_async().await {
        check_resource_lease(resource_lease)?;
        ensure_compressed_size_within_limit(metadata.len())?;
    }

    check_resource_lease(resource_lease)?;
    let mut temp = TempArchive::create()?;
    let mut copied = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        super::check_cancelled(runtime)?;
        check_resource_lease(resource_lease)?;
        let read = input.read(&mut buffer).map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_IO,
                format!("unzip: unable to read '{}': {err}", path.display()),
                err,
            )
        })?;
        if read == 0 {
            break;
        }
        copied = copied
            .checked_add(read as u64)
            .ok_or_else(|| unzip_error_with(&UNZIP_ERROR_IO, "unzip: archive is too large"))?;
        ensure_compressed_size_within_limit(copied)?;
        check_resource_lease(resource_lease)?;
        temp.write_all(&buffer[..read]).map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_IO,
                format!(
                    "unzip: unable to write temporary archive '{}': {err}",
                    temp.path().display()
                ),
                err,
            )
        })?;
    }
    check_resource_lease(resource_lease)?;
    temp.flush().map_err(|err| {
        unzip_error_with_source(
            &UNZIP_ERROR_IO,
            format!(
                "unzip: unable to flush temporary archive '{}': {err}",
                temp.path().display()
            ),
            err,
        )
    })?;
    check_resource_lease(resource_lease)?;
    temp.close();
    Ok(temp)
}

pub(super) struct CapturedFilesystem {
    pub(super) provider: std::sync::Arc<dyn vfs::FsProvider>,
    cwd: Option<PathBuf>,
}

fn captured_cwd<'a>(
    captured: &'a mut Option<CapturedFilesystem>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<&'a Path> {
    if captured.is_none() {
        let (provider, cwd) = vfs::current_provider_with_current_dir().map_err(|err| {
            unzip_error_with_source(
                descriptor,
                format!("unzip: unable to resolve current folder: {err}"),
                err,
            )
        })?;
        *captured = Some(CapturedFilesystem {
            provider,
            cwd: Some(cwd),
        });
    }
    let captured = captured.as_mut().expect("captured filesystem");
    if captured.cwd.is_none() {
        let (provider, cwd) = vfs::current_provider_with_current_dir().map_err(|err| {
            unzip_error_with_source(
                descriptor,
                format!("unzip: unable to resolve current folder: {err}"),
                err,
            )
        })?;
        if !std::sync::Arc::ptr_eq(&provider, &captured.provider) {
            return Err(unzip_error_with(
                descriptor,
                "unzip: filesystem authority changed during path capture",
            ));
        }
        captured.cwd = Some(cwd);
    }
    Ok(captured
        .cwd
        .as_deref()
        .expect("relative path has captured CWD"))
}

pub(super) fn resolve_archive_path(
    text: &str,
    captured: &mut Option<CapturedFilesystem>,
) -> BuiltinResult<PathBuf> {
    let expanded = expand_user_path(text.trim(), BUILTIN_NAME)
        .map_err(|msg| unzip_error_with(&UNZIP_ERROR_FILENAME, msg))?;
    let path = PathBuf::from(expanded);
    let provider = captured
        .as_ref()
        .map(|owner| owner.provider.clone())
        .unwrap_or_else(vfs::current_provider);
    if provider.path_namespace().is_absolute(&path) {
        captured.get_or_insert(CapturedFilesystem {
            provider,
            cwd: None,
        });
        Ok(path)
    } else {
        Ok(captured_cwd(captured, &UNZIP_ERROR_IO)?.join(path))
    }
}

pub(super) fn resolve_output_folder(
    text: Option<&str>,
    captured: &mut Option<CapturedFilesystem>,
) -> BuiltinResult<PathBuf> {
    match text {
        Some(raw) => {
            let expanded = expand_user_path(raw.trim(), BUILTIN_NAME)
                .map_err(|msg| unzip_error_with(&UNZIP_ERROR_OUTPUT_FOLDER, msg))?;
            let path = PathBuf::from(expanded);
            let provider = captured
                .as_ref()
                .map(|owner| owner.provider.clone())
                .unwrap_or_else(vfs::current_provider);
            if provider.path_namespace().is_absolute(&path) {
                captured.get_or_insert(CapturedFilesystem {
                    provider,
                    cwd: None,
                });
                Ok(path)
            } else {
                Ok(captured_cwd(captured, &UNZIP_ERROR_OUTPUT_FOLDER)?.join(path))
            }
        }
        None => Ok(captured_cwd(captured, &UNZIP_ERROR_OUTPUT_FOLDER)?.to_path_buf()),
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) struct TempArchive {
    path: PathBuf,
    file: Option<std::fs::File>,
}

#[cfg(not(target_arch = "wasm32"))]
impl TempArchive {
    pub(super) fn create() -> BuiltinResult<Self> {
        let (path, file) = create_unique_temp_archive()?;
        Ok(Self {
            path,
            file: Some(file),
        })
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.file
            .as_mut()
            .expect("open temporary archive")
            .write_all(bytes)
    }

    pub(super) fn flush(&mut self) -> std::io::Result<()> {
        self.file.as_mut().expect("open temporary archive").flush()
    }

    pub(super) fn close(&mut self) {
        self.file.take();
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for TempArchive {
    fn drop(&mut self) {
        self.close();
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn unique_temp_archive_path() -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    path.push(format!("runmat_unzip_{}_{}.zip", std::process::id(), nanos));
    path
}

#[cfg(not(target_arch = "wasm32"))]
fn create_unique_temp_archive() -> BuiltinResult<(PathBuf, std::fs::File)> {
    for attempt in 0..32u32 {
        let mut path = unique_temp_archive_path();
        if attempt > 0 {
            path.set_file_name(format!(
                "{}_{}.zip",
                path.file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or("runmat_unzip"),
                attempt
            ));
        }
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => return Ok((path, file)),
            Err(err) if err.kind() == ErrorKind::AlreadyExists => continue,
            Err(err) => {
                return Err(unzip_error_with_source(
                    &UNZIP_ERROR_IO,
                    format!(
                        "unzip: unable to create temporary archive '{}': {err}",
                        path.display()
                    ),
                    err,
                ));
            }
        }
    }
    Err(unzip_error_with(
        &UNZIP_ERROR_IO,
        "unzip: unable to allocate a unique temporary archive path",
    ))
}

pub(super) fn ensure_compressed_size_within_limit(size: u64) -> BuiltinResult<()> {
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
