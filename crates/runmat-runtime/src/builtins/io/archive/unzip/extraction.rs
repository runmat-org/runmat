use super::*;

mod control;
use self::control::ExtractionControl;
mod partial;
use self::partial::PartialEntry;
#[cfg(not(target_arch = "wasm32"))]
mod worker;
#[cfg(not(target_arch = "wasm32"))]
pub(super) use worker::extract_archive_file_on_worker;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
#[cfg(test)]
mod path_tests;

#[cfg(target_arch = "wasm32")]
pub(super) async fn extract_archive<R>(
    reader: R,
    output_folder: &Path,
    runtime: Option<&crate::context::RuntimeContext>,
    command_origin: Option<runmat_execution::profiling::CommandProfilingReceipt>,
    provider: std::sync::Arc<dyn vfs::FsProvider>,
    resource_lease: ArchiveResourceLease,
) -> BuiltinResult<Vec<String>>
where
    R: Read + Seek,
{
    extract_archive_impl(
        reader,
        output_folder,
        ExtractionControl::browser(runtime, resource_lease, command_origin),
        provider,
    )
    .await
}

async fn extract_archive_impl<R>(
    reader: R,
    output_folder: &Path,
    mut control: ExtractionControl,
    provider: std::sync::Arc<dyn vfs::FsProvider>,
) -> BuiltinResult<Vec<String>>
where
    R: Read + Seek,
{
    control.check_resource()?;
    let mut archive = ZipArchive::new(reader).map_err(map_zip_error)?;
    control.check_resource()?;
    let mut filenames = Vec::new();
    let mut total_written = 0u64;

    for index in 0..archive.len() {
        control.completed(1).await?;
        let mut entry = archive.by_index(index).map_err(map_zip_error)?;
        let relative = safe_archive_path(entry.name())?;
        if relative.as_os_str().is_empty() {
            continue;
        }
        let output_path = output_folder.join(&relative);
        reject_symlink_components(
            provider.as_ref(),
            output_folder,
            &relative,
            control.resource_lease(),
        )
        .await?;

        if entry.is_dir() || entry.name().ends_with('/') || entry.name().ends_with('\\') {
            control.check_resource()?;
            provider.create_dir_all(&output_path).await.map_err(|err| {
                unzip_error_with_source(
                    &UNZIP_ERROR_IO,
                    format!(
                        "unzip: unable to create folder '{}': {err}",
                        output_path.display()
                    ),
                    err,
                )
            })?;
            control.check_resource()?;
            continue;
        }

        if let Some(parent) = output_path.parent() {
            control.check_resource()?;
            provider.create_dir_all(parent).await.map_err(|err| {
                unzip_error_with_source(
                    &UNZIP_ERROR_IO,
                    format!(
                        "unzip: unable to create folder '{}': {err}",
                        parent.display()
                    ),
                    err,
                )
            })?;
            control.check_resource()?;
        }
        reject_symlink_components(
            provider.as_ref(),
            output_folder,
            &relative,
            control.resource_lease(),
        )
        .await?;

        control.check_resource()?;
        #[cfg(not(target_arch = "wasm32"))]
        let mut partial =
            PartialEntry::create(temporary_output_path(&output_path), provider.clone()).await?;
        #[cfg(target_arch = "wasm32")]
        let mut partial = PartialEntry::create_with_origin(
            temporary_output_path(&output_path),
            provider.clone(),
            control.command_origin(),
        )
        .await?;
        control.check_resource()?;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            control.completed(0).await?;
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
            control.check_resource()?;
            partial.write_all(&buffer[..read]).map_err(|err| {
                unzip_error_with_source(
                    &UNZIP_ERROR_IO,
                    format!(
                        "unzip: unable to write temporary output '{}': {err}",
                        partial.path().display()
                    ),
                    err,
                )
            })?;
            control.completed(read as u64).await?;
        }
        control.check_resource()?;
        partial.flush().await.map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_IO,
                format!(
                    "unzip: unable to flush temporary output '{}': {err}",
                    partial.path().display()
                ),
                err,
            )
        })?;
        control.check_resource()?;
        partial.sync_all().await.map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_IO,
                format!(
                    "unzip: unable to sync temporary output '{}': {err}",
                    partial.path().display()
                ),
                err,
            )
        })?;
        control.check_resource()?;
        partial.close();
        control.completed(0).await?;
        reject_symlink_components(
            provider.as_ref(),
            output_folder,
            &relative,
            control.resource_lease(),
        )
        .await?;
        control.check_resource()?;
        partial.commit_to(&output_path).await?;
        control.check_resource()?;
        filenames.push(path_to_string(&output_path));
    }

    Ok(filenames)
}

fn temporary_output_path(path: &Path) -> PathBuf {
    static NEXT_OUTPUT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("entry");
    let nanos = runmat_time::unix_timestamp_ns();
    let sequence = NEXT_OUTPUT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(target_arch = "wasm32"))]
    let process = std::process::id();
    #[cfg(target_arch = "wasm32")]
    let process = 0;
    parent.join(format!(
        ".{name}.runmat-unzip-tmp-{process}-{nanos}-{sequence}"
    ))
}

pub(super) async fn reject_symlink_path(
    provider: &dyn vfs::FsProvider,
    path: &Path,
    resource_lease: &ArchiveResourceLease,
) -> BuiltinResult<()> {
    check_resource_lease(resource_lease)?;
    let inspected = provider.symlink_metadata(path).await;
    let metadata = match inspected {
        Ok(metadata) => {
            check_resource_lease(resource_lease)?;
            Ok(metadata)
        }
        Err(err) => Err(err),
    };
    match metadata {
        Ok(metadata) if metadata.is_symlink() => Err(unzip_error_with(
            &UNZIP_ERROR_SECURITY,
            format!("unzip: output path '{}' is a symbolic link", path.display()),
        )),
        Ok(_) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => check_resource_lease(resource_lease),
        Err(err) => Err(unzip_error_with_source(
            &UNZIP_ERROR_IO,
            format!("unzip: unable to inspect '{}': {err}", path.display()),
            err,
        )),
    }
}

async fn reject_symlink_components(
    provider: &dyn vfs::FsProvider,
    root: &Path,
    relative: &Path,
    resource_lease: &ArchiveResourceLease,
) -> BuiltinResult<()> {
    let mut current = root.to_path_buf();
    reject_symlink_path(provider, &current, resource_lease).await?;
    for component in relative.components() {
        current.push(component.as_os_str());
        reject_symlink_path(provider, &current, resource_lease).await?;
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
