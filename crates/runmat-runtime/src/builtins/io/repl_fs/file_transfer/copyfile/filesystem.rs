use std::io;
use std::path::Path;

use super::super::authority::CopyOperationError;

pub(super) async fn copy(
    fs: &mut super::TransferFilesystem,
    source: &Path,
    destination: &Path,
    directory: bool,
) -> Result<(), CopyOperationError> {
    if directory {
        copy_directory(fs, source, destination).await
    } else {
        copy_file(fs, source, destination).await
    }
}

#[async_recursion::async_recursion(?Send)]
async fn copy_directory(
    fs: &mut super::TransferFilesystem,
    source: &Path,
    destination: &Path,
) -> Result<(), CopyOperationError> {
    require_existing_parent(fs, destination).await?;
    match fs.metadata(destination).await {
        Ok(metadata) if !metadata.is_dir() => {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "destination exists and is not a directory",
            )
            .into());
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            fs.create_dir_all(destination).await?;
        }
        Err(error) => return Err(error.into()),
    }
    preserve_readonly(fs, source, destination).await;
    for entry in fs.read_dir(source).await? {
        let child_source = entry.path().to_path_buf();
        let child_target = destination.join(entry.file_name());
        let metadata = fs.metadata(&child_source).await?;
        if metadata.is_dir() {
            copy_directory(fs, &child_source, &child_target).await?;
        } else {
            copy_file(fs, &child_source, &child_target).await?;
        }
    }
    Ok(())
}

async fn copy_file(
    fs: &mut super::TransferFilesystem,
    source: &Path,
    destination: &Path,
) -> Result<(), CopyOperationError> {
    require_existing_parent(fs, destination).await?;
    fs.copy_file(source, destination).await?;
    preserve_readonly(fs, source, destination).await;
    Ok(())
}

async fn require_existing_parent(fs: &super::TransferFilesystem, path: &Path) -> io::Result<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() || parent == Path::new(".") {
        return Ok(());
    }
    match fs.metadata(parent).await {
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("destination parent \"{}\" does not exist", parent.display()),
        )),
        Err(error) => Err(error),
    }
}

async fn preserve_readonly(fs: &super::TransferFilesystem, source: &Path, destination: &Path) {
    if let Ok(metadata) = fs.metadata(source).await {
        let _ = fs.set_readonly(destination, metadata.is_readonly()).await;
    }
}
