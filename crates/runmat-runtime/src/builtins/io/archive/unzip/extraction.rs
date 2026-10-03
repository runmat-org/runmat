use super::*;

#[cfg(not(target_arch = "wasm32"))]
pub(super) async fn extract_archive_file_on_worker(
    path: PathBuf,
    output_folder: PathBuf,
) -> BuiltinResult<Vec<String>> {
    let (tx, rx) = futures::channel::oneshot::channel();
    std::thread::Builder::new()
        .name("runmat-unzip-extract".to_string())
        .spawn(move || {
            let result = (|| {
                let file = std::fs::File::open(&path).map_err(|err| {
                    unzip_error_with_source(
                        &UNZIP_ERROR_IO,
                        format!("unzip: unable to open '{}': {err}", path.display()),
                        err,
                    )
                })?;
                futures::executor::block_on(extract_archive_impl(file, &output_folder))
            })();
            let _ = tx.send(result);
        })
        .map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_IO,
                format!("unzip: unable to spawn extraction worker: {err}"),
                err,
            )
        })?;
    rx.await.map_err(|err| {
        unzip_error_with_source(
            &UNZIP_ERROR_IO,
            format!("unzip: extraction worker did not return a result: {err}"),
            err,
        )
    })?
}

#[cfg(target_arch = "wasm32")]
pub(super) async fn extract_archive<R>(
    reader: R,
    output_folder: &Path,
) -> BuiltinResult<Vec<String>>
where
    R: Read + Seek,
{
    extract_archive_impl(reader, output_folder).await
}

async fn extract_archive_impl<R>(reader: R, output_folder: &Path) -> BuiltinResult<Vec<String>>
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

        let temp_output_path = temporary_output_path(&output_path);
        let mut open_options = vfs::OpenOptions::new();
        open_options.write(true).create_new(true);
        let mut out = open_options
            .open_async(&temp_output_path)
            .await
            .map_err(|err| {
                unzip_error_with_source(
                    &UNZIP_ERROR_IO,
                    format!(
                        "unzip: unable to create temporary output '{}': {err}",
                        temp_output_path.display()
                    ),
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
                    format!(
                        "unzip: unable to write temporary output '{}': {err}",
                        temp_output_path.display()
                    ),
                    err,
                )
            })?;
        }
        out.flush_async().await.map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_IO,
                format!(
                    "unzip: unable to flush temporary output '{}': {err}",
                    temp_output_path.display()
                ),
                err,
            )
        })?;
        out.sync_all_async().await.map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_IO,
                format!(
                    "unzip: unable to sync temporary output '{}': {err}",
                    temp_output_path.display()
                ),
                err,
            )
        })?;
        drop(out);
        reject_symlink_components(output_folder, &relative).await?;
        if let Err(err) = vfs::rename_async(&temp_output_path, &output_path).await {
            let _ = vfs::remove_file_async(&temp_output_path).await;
            return Err(unzip_error_with_source(
                &UNZIP_ERROR_IO,
                format!("unzip: unable to create '{}': {err}", output_path.display()),
                err,
            ));
        }
        filenames.push(path_to_string(&output_path));
    }

    Ok(filenames)
}

fn temporary_output_path(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("entry");
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    parent.join(format!(
        ".{name}.runmat-unzip-tmp-{}-{nanos}",
        std::process::id()
    ))
}

pub(super) async fn reject_symlink_path(path: &Path) -> BuiltinResult<()> {
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
