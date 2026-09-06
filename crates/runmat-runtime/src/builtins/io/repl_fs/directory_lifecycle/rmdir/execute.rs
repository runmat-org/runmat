use std::io;
use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinErrorDescriptor, RMDIR_ERROR_FILESYSTEM, RMDIR_ERROR_NOT_DIRECTORY,
    RMDIR_ERROR_NOT_EMPTY, RMDIR_ERROR_NOT_FOUND,
};
use runmat_filesystem as vfs;

use super::super::result::DirectoryOutcome;
use super::options::RemoveRequest;

pub(super) async fn remove(request: RemoveRequest) -> DirectoryOutcome {
    let selected = match select_target(&request).await {
        Ok(target) => target,
        Err(outcome) => return outcome,
    };
    if selected.remove_link {
        return remove_link(&selected.path).await;
    }
    remove_directory(&selected.path, request.recursive).await
}

struct SelectedTarget {
    path: PathBuf,
    remove_link: bool,
}

async fn select_target(request: &RemoveRequest) -> Result<SelectedTarget, DirectoryOutcome> {
    let metadata = vfs::symlink_metadata_async(&request.path)
        .await
        .map_err(|cause| metadata_failure(&request.path, cause))?;
    if metadata.is_symlink() {
        if !request.resolve_symbolic_links {
            return Ok(SelectedTarget {
                path: request.path.clone(),
                remove_link: true,
            });
        }
        let target = vfs::canonicalize_async(&request.path)
            .await
            .map_err(|cause| metadata_failure(&request.path, cause))?;
        let target_metadata = vfs::metadata_async(&target)
            .await
            .map_err(|cause| metadata_failure(&target, cause))?;
        if !target_metadata.is_dir() {
            return Err(not_directory(&target));
        }
        return Ok(SelectedTarget {
            path: target,
            remove_link: false,
        });
    }
    if !metadata.is_dir() {
        return Err(not_directory(&request.path));
    }
    Ok(SelectedTarget {
        path: request.path.clone(),
        remove_link: false,
    })
}

async fn remove_link(path: &Path) -> DirectoryOutcome {
    match vfs::remove_file_async(path).await {
        Ok(()) => DirectoryOutcome::Success,
        Err(cause) => removal_failure(path, cause),
    }
}

async fn remove_directory(path: &Path, recursive: bool) -> DirectoryOutcome {
    let result = if recursive {
        vfs::remove_dir_all_async(path).await
    } else {
        vfs::remove_dir_async(path).await
    };
    match result {
        Ok(()) => DirectoryOutcome::Success,
        Err(cause) => removal_failure(path, cause),
    }
}

fn metadata_failure(path: &Path, cause: io::Error) -> DirectoryOutcome {
    if cause.kind() == io::ErrorKind::NotFound {
        failure(
            &RMDIR_ERROR_NOT_FOUND,
            format!("Folder \"{}\" does not exist.", path.display()),
        )
    } else {
        filesystem_failure(path, cause)
    }
}

fn removal_failure(path: &Path, cause: io::Error) -> DirectoryOutcome {
    match cause.kind() {
        io::ErrorKind::NotFound => metadata_failure(path, cause),
        io::ErrorKind::DirectoryNotEmpty => failure(
            &RMDIR_ERROR_NOT_EMPTY,
            format!(
                "Cannot remove folder \"{}\": directory is not empty.",
                path.display()
            ),
        ),
        _ => filesystem_failure(path, cause),
    }
}

fn not_directory(path: &Path) -> DirectoryOutcome {
    failure(
        &RMDIR_ERROR_NOT_DIRECTORY,
        format!(
            "Cannot remove \"{}\": target is not a directory.",
            path.display()
        ),
    )
}

fn filesystem_failure(path: &Path, cause: io::Error) -> DirectoryOutcome {
    failure(
        &RMDIR_ERROR_FILESYSTEM,
        format!("Unable to remove folder \"{}\": {cause}", path.display()),
    )
}

fn failure(descriptor: &'static BuiltinErrorDescriptor, message: String) -> DirectoryOutcome {
    DirectoryOutcome::failure(
        message,
        descriptor.identifier.unwrap_or("RunMat:rmdir:Error"),
    )
}
