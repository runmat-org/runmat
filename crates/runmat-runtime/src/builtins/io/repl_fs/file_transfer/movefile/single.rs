use std::io;
use std::path::PathBuf;

use super::super::outcome::TransferOutcome;
use super::plan::{DestinationPolicy, MovePlanEntry};

pub(super) async fn move_path(
    fs: &mut super::super::authority::TransferFilesystem,
    source: &str,
    destination: &str,
    force: bool,
) -> crate::BuiltinResult<TransferOutcome> {
    let source_path = PathBuf::from(source);
    match fs.symlink_metadata(&source_path).await {
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(super::result::source_not_found(source))
        }
        Err(error) => return Ok(super::result::filesystem(source, destination, &error)),
    }
    let destination_path = PathBuf::from(destination);
    if fs.same_path(&source_path, &destination_path).await {
        return Ok(super::result::success());
    }

    let mut target = destination_path.clone();
    let mut destination_policy = DestinationPolicy::New;
    let destination_metadata = match fs.metadata(&destination_path).await {
        Ok(metadata) => Some(metadata),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            match fs.symlink_metadata(&destination_path).await {
                Ok(metadata) => Some(metadata),
                Err(error) if error.kind() == io::ErrorKind::NotFound => None,
                Err(error) => return Ok(super::result::filesystem(source, destination, &error)),
            }
        }
        Err(error) => return Ok(super::result::filesystem(source, destination, &error)),
    };
    if let Some(destination_metadata) = destination_metadata {
        if destination_metadata.is_dir() {
            let Some(name) = source_path.file_name() else {
                return Ok(super::result::filesystem(
                    source,
                    destination,
                    &io::Error::other("cannot determine source file name"),
                ));
            };
            target = destination_path.join(name);
            if fs.same_path(&target, &source_path).await {
                return Ok(super::result::success());
            }
            destination_policy = match fs.symlink_metadata(&target).await {
                Ok(_) if !force => {
                    return Ok(super::result::destination_exists(
                        &super::super::target::display(&target),
                    ));
                }
                Ok(_) => DestinationPolicy::ReplaceCompatible,
                Err(error) if error.kind() == io::ErrorKind::NotFound => DestinationPolicy::New,
                Err(error) => {
                    return Ok(super::result::filesystem(
                        source,
                        &super::super::target::display(&target),
                        &error,
                    ));
                }
            };
        } else if !force {
            return Ok(super::result::destination_exists(destination));
        } else {
            destination_policy = DestinationPolicy::ReplaceCompatible;
        }
    }
    let entry = MovePlanEntry {
        source: source_path.clone(),
        source_display: super::super::target::display(&source_path),
        target: target.clone(),
        target_display: super::super::target::display(&target),
        destination_policy,
    };
    Ok(complete(super::plan::execute(fs, &[entry]).await?))
}

fn complete(result: Result<(), super::plan::MoveFailure>) -> TransferOutcome {
    match result {
        Ok(()) => super::result::success(),
        Err(failure) => super::result::filesystem(&failure.source, &failure.target, &failure.error),
    }
}
