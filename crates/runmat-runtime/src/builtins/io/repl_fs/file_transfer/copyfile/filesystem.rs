use runmat_filesystem as vfs;
use std::io;
use std::path::Path;

pub(super) async fn copy(source: &Path, destination: &Path, directory: bool) -> io::Result<()> {
    if directory {
        copy_directory(source, destination).await
    } else {
        copy_file(source, destination).await
    }
}

#[async_recursion::async_recursion(?Send)]
async fn copy_directory(source: &Path, destination: &Path) -> io::Result<()> {
    require_existing_parent(destination).await?;
    match vfs::metadata_async(destination).await {
        Ok(metadata) if !metadata.is_dir() => {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "destination exists and is not a directory",
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            vfs::create_dir_all_async(destination).await?;
        }
        Err(error) => return Err(error),
    }
    preserve_readonly(source, destination).await;
    for entry in vfs::read_dir_async(source).await? {
        let child_source = entry.path().to_path_buf();
        let child_target = destination.join(entry.file_name());
        let metadata = vfs::metadata_async(&child_source).await?;
        if metadata.is_dir() {
            copy_directory(&child_source, &child_target).await?;
        } else {
            copy_file(&child_source, &child_target).await?;
        }
    }
    Ok(())
}

async fn copy_file(source: &Path, destination: &Path) -> io::Result<()> {
    require_existing_parent(destination).await?;
    vfs::copy_file(source, destination)?;
    preserve_readonly(source, destination).await;
    Ok(())
}

async fn require_existing_parent(path: &Path) -> io::Result<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() || parent == Path::new(".") {
        return Ok(());
    }
    if vfs::metadata_async(parent).await.is_err() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("destination parent \"{}\" does not exist", parent.display()),
        ));
    }
    Ok(())
}

async fn preserve_readonly(source: &Path, destination: &Path) {
    if let Ok(metadata) = vfs::metadata_async(source).await {
        let _ = vfs::set_readonly_async(destination, metadata.is_readonly()).await;
    }
}
