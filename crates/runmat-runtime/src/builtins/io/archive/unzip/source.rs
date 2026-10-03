use super::*;

pub(super) async fn extract_remote_archive(
    url: &Url,
    output_folder: &Path,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Vec<String>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        super::check_cancelled(runtime)?;
        let temp = download_archive_to_temp_file_async(url.clone()).await?;
        super::check_cancelled(runtime)?;
        extract_archive_file_on_worker(temp.path.clone(), output_folder.to_path_buf()).await
    }

    #[cfg(target_arch = "wasm32")]
    {
        let bytes = download_archive(url, runtime).await?;
        extract_archive(Cursor::new(bytes), output_folder).await
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) async fn extract_local_archive(
    path: &Path,
    output_folder: &Path,
) -> BuiltinResult<Vec<String>> {
    let temp = local_archive_to_temp_file(path).await?;
    extract_archive_file_on_worker(temp.path.clone(), output_folder.to_path_buf()).await
}

#[cfg(target_arch = "wasm32")]
pub(super) async fn extract_local_archive(
    path: &Path,
    output_folder: &Path,
) -> BuiltinResult<Vec<String>> {
    let file = File::open(path).map_err(|err| {
        unzip_error_with_source(
            &UNZIP_ERROR_IO,
            format!("unzip: unable to open '{}': {err}", path.display()),
            err,
        )
    })?;
    extract_archive(file, output_folder).await
}

#[cfg(not(target_arch = "wasm32"))]
async fn download_archive_to_temp_file_async(url: Url) -> BuiltinResult<TempArchive> {
    let (tx, rx) = futures::channel::oneshot::channel();
    std::thread::Builder::new()
        .name("runmat-unzip-download".to_string())
        .spawn(move || {
            let _ = tx.send(download_archive_to_temp_file(&url));
        })
        .map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_HTTP,
                format!("unzip: unable to spawn download worker: {err}"),
                err,
            )
        })?;
    rx.await.map_err(|_| {
        unzip_error_with(
            &UNZIP_ERROR_HTTP,
            "unzip: download worker terminated before returning a result",
        )
    })?
}

#[cfg(not(target_arch = "wasm32"))]
async fn local_archive_to_temp_file(path: &Path) -> BuiltinResult<TempArchive> {
    let mut input = File::open_async(path).await.map_err(|err| {
        unzip_error_with_source(
            &UNZIP_ERROR_IO,
            format!("unzip: unable to open '{}': {err}", path.display()),
            err,
        )
    })?;
    if let Ok(metadata) = input.metadata_async().await {
        ensure_compressed_size_within_limit(metadata.len())?;
    }

    let (temp_path, mut out) = create_unique_temp_archive()?;
    let mut copied = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
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
        out.write_all(&buffer[..read]).map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_IO,
                format!(
                    "unzip: unable to write temporary archive '{}': {err}",
                    temp_path.display()
                ),
                err,
            )
        })?;
    }
    out.flush().map_err(|err| {
        unzip_error_with_source(
            &UNZIP_ERROR_IO,
            format!(
                "unzip: unable to flush temporary archive '{}': {err}",
                temp_path.display()
            ),
            err,
        )
    })?;
    Ok(TempArchive { path: temp_path })
}

pub(super) fn resolve_archive_path(text: &str) -> BuiltinResult<PathBuf> {
    let expanded = expand_user_path(text.trim(), BUILTIN_NAME)
        .map_err(|msg| unzip_error_with(&UNZIP_ERROR_FILENAME, msg))?;
    Ok(PathBuf::from(expanded))
}

pub(super) async fn resolve_output_folder(text: Option<&str>) -> BuiltinResult<PathBuf> {
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
struct TempArchive {
    path: PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for TempArchive {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn download_archive_to_temp_file(url: &Url) -> BuiltinResult<TempArchive> {
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

    let (temp_path, mut out) = create_unique_temp_archive()?;
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
        out.write_all(&buffer[..read]).map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_IO,
                format!(
                    "unzip: unable to write temporary archive '{}': {err}",
                    temp_path.display()
                ),
                err,
            )
        })?;
    }
    out.flush().map_err(|err| {
        unzip_error_with_source(
            &UNZIP_ERROR_IO,
            format!(
                "unzip: unable to flush temporary archive '{}': {err}",
                temp_path.display()
            ),
            err,
        )
    })?;
    Ok(TempArchive { path: temp_path })
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

#[cfg(target_arch = "wasm32")]
async fn download_archive(
    url: &Url,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Vec<u8>> {
    let request = HttpRequest {
        url: url.clone(),
        method: HttpMethod::Get,
        headers: Vec::new(),
        body: None,
        timeout: DEFAULT_TIMEOUT,
        user_agent: DEFAULT_USER_AGENT.to_string(),
    };
    let response = transport::send_request(request, runtime)
        .await
        .map_err(|err| match err {
            transport::TransportCallError::Transport(err) => {
                unzip_error_with_source(&UNZIP_ERROR_HTTP, err.message_with_prefix("unzip"), err)
            }
            transport::TransportCallError::Runtime(err) => err,
        })?;
    ensure_compressed_size_within_limit(response.body.len() as u64)?;
    Ok(response.body)
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
