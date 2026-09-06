use runmat_filesystem as vfs;
use std::io;
use std::path::PathBuf;

use super::super::outcome::TransferOutcome;
use super::plan::MovePlanEntry;

pub(super) async fn move_path(source: &str, destination: &str, force: bool) -> TransferOutcome {
    let source_path = PathBuf::from(source);
    if vfs::metadata_async(&source_path).await.is_err() {
        return super::result::source_not_found(source);
    }
    let destination_path = PathBuf::from(destination);
    if source_path == destination_path {
        return super::result::success();
    }

    let mut target = destination_path.clone();
    let mut replace_directory = None;
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
            if target == source_path {
                return super::result::success();
            }
            replace_directory = match vfs::metadata_async(&target).await {
                Ok(_) if !force => {
                    return super::result::destination_exists(&super::super::target::display(
                        &target,
                    ));
                }
                Ok(metadata) => Some(metadata.is_dir()),
                Err(error) if error.kind() == io::ErrorKind::NotFound => None,
                Err(error) => {
                    return super::result::filesystem(
                        source,
                        &super::super::target::display(&target),
                        &error,
                    );
                }
            };
        } else if !force {
            return super::result::destination_exists(destination);
        } else {
            replace_directory = Some(false);
        }
    }
    let entry = MovePlanEntry {
        source: source_path.clone(),
        source_display: super::super::target::display(&source_path),
        target: target.clone(),
        target_display: super::super::target::display(&target),
        replace_directory,
    };
    complete(super::plan::execute(&[entry]).await)
}

fn complete(result: Result<(), super::plan::MoveFailure>) -> TransferOutcome {
    match result {
        Ok(()) => super::result::success(),
        Err(failure) => super::result::filesystem(&failure.source, &failure.target, &failure.error),
    }
}
