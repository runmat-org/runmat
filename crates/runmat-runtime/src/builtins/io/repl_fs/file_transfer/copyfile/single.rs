use runmat_builtins::COPYFILE_RESULT_OS_ERROR;
use runmat_filesystem as vfs;
use std::io;
use std::path::PathBuf;

use super::super::outcome::TransferOutcome;
use super::plan::{CopyPlanEntry, TargetKind};

pub(super) async fn copy(source: &str, destination: &str, force: bool) -> TransferOutcome {
    let source_path = PathBuf::from(source);
    let source_metadata = match vfs::metadata_async(&source_path).await {
        Ok(metadata) => metadata,
        Err(_) => return super::result::source_not_found(source),
    };
    let source_display = super::super::target::display(&source_path);
    let destination_path = PathBuf::from(destination);
    if super::super::target::same_path(&source_path, &destination_path).await {
        return super::result::same_path(&source_display);
    }

    let mut target = destination_path.clone();
    let mut replace = None;
    if let Ok(destination_metadata) = vfs::metadata_async(&destination_path).await {
        if destination_metadata.is_dir() {
            let Some(name) = source_path.file_name() else {
                return super::result::filesystem(
                    source,
                    destination,
                    &io::Error::other("cannot determine source file name"),
                );
            };
            target = destination_path.join(name);
            if super::super::target::same_path(&source_path, &target).await {
                return super::result::same_path(&source_display);
            }
            if source_metadata.is_dir()
                && super::super::target::is_descendant(&source_path, &target).await
            {
                return super::result::failure(
                    "Cannot copy a folder into one of its descendants.",
                    &COPYFILE_RESULT_OS_ERROR,
                );
            }
            replace = match vfs::metadata_async(&target).await {
                Ok(_) if !force => {
                    return super::result::destination_exists(&super::super::target::display(
                        &target,
                    ));
                }
                Ok(metadata) => Some(kind(metadata.is_dir())),
                Err(error) if error.kind() == io::ErrorKind::NotFound => None,
                Err(error) => {
                    return super::result::filesystem(
                        source,
                        &super::super::target::display(&target),
                        &error,
                    );
                }
            };
        } else if source_metadata.is_dir() {
            return super::result::destination_not_directory(destination);
        } else if !force {
            return super::result::destination_exists(destination);
        } else {
            replace = Some(TargetKind::File);
        }
    } else if source_metadata.is_dir()
        && super::super::target::is_descendant(&source_path, &destination_path).await
    {
        return super::result::failure(
            "Cannot copy a folder into one of its descendants.",
            &COPYFILE_RESULT_OS_ERROR,
        );
    }

    let target_display = super::super::target::display(&target);
    let entry = CopyPlanEntry {
        source: source_path,
        source_display,
        target,
        target_display,
        source_is_directory: source_metadata.is_dir(),
        replace,
    };
    complete(super::plan::execute(&[entry]).await)
}

fn kind(directory: bool) -> TargetKind {
    if directory {
        TargetKind::Directory
    } else {
        TargetKind::File
    }
}

fn complete(result: Result<(), super::plan::CopyFailure>) -> TransferOutcome {
    match result {
        Ok(()) => super::result::success(),
        Err(failure) => super::result::filesystem(&failure.source, &failure.target, &failure.error),
    }
}
