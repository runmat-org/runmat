use std::io;
use std::path::PathBuf;

use super::super::outcome::TransferOutcome;
use super::plan::{DestinationPolicy, MovePlanEntry};

pub(super) async fn move_matches(
    fs: &mut super::super::authority::TransferFilesystem,
    pattern: &str,
    destination: &str,
    force: bool,
) -> crate::BuiltinResult<TransferOutcome> {
    let matches = match fs.wildcard_matches(pattern).await {
        Ok(matches) if matches.is_empty() => return Ok(super::result::source_not_found(pattern)),
        Ok(matches) => matches,
        Err(error) if error.kind() == io::ErrorKind::InvalidInput => {
            return Ok(super::result::invalid_pattern(pattern, &error.to_string()));
        }
        Err(error) => return Ok(super::result::filesystem(pattern, destination, &error)),
    };
    let destination_path = PathBuf::from(destination);
    match fs.metadata(&destination_path).await {
        Ok(metadata) if !metadata.is_dir() => {
            return Ok(super::result::destination_not_directory(destination));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(super::result::destination_missing(destination))
        }
        Err(error) => return Ok(super::result::filesystem(pattern, destination, &error)),
    }
    let entries = match plan(fs, matches, &destination_path, force).await? {
        Ok(entries) => entries,
        Err(outcome) => return Ok(outcome),
    };
    Ok(match super::plan::execute(fs, &entries).await? {
        Ok(()) => super::result::success(),
        Err(failure) => super::result::filesystem(&failure.source, &failure.target, &failure.error),
    })
}

async fn plan(
    fs: &mut super::super::authority::TransferFilesystem,
    sources: Vec<PathBuf>,
    destination: &std::path::Path,
    force: bool,
) -> crate::BuiltinResult<Result<Vec<MovePlanEntry>, TransferOutcome>> {
    let mut entries = Vec::with_capacity(sources.len());
    for source in sources {
        fs.checkpoint().await?;
        if let Some(entry) = match plan_entry(fs, source, destination, force).await {
            Ok(entry) => entry,
            Err(outcome) => return Ok(Err(outcome)),
        } {
            entries.push(entry);
        }
    }
    Ok(Ok(entries))
}

async fn plan_entry(
    fs: &super::super::authority::TransferFilesystem,
    source: PathBuf,
    destination: &std::path::Path,
    force: bool,
) -> Result<Option<MovePlanEntry>, TransferOutcome> {
    let source_display = super::super::target::display(&source);
    match fs.symlink_metadata(&source).await {
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Err(super::result::source_not_found(&source_display))
        }
        Err(error) => {
            return Err(super::result::filesystem(
                &source_display,
                &super::super::target::display(destination),
                &error,
            ))
        }
    }
    let name = source.file_name().ok_or_else(|| {
        super::result::filesystem(
            &source_display,
            &super::super::target::display(destination),
            &io::Error::other("cannot determine source name"),
        )
    })?;
    let target = destination.join(name);
    if fs.same_path(&target, &source).await {
        return Ok(None);
    }
    let target_display = super::super::target::display(&target);
    let destination_policy = match fs.symlink_metadata(&target).await {
        Ok(_) if !force => return Err(super::result::destination_exists(&target_display)),
        Ok(_) => DestinationPolicy::ReplaceCompatible,
        Err(error) if error.kind() == io::ErrorKind::NotFound => DestinationPolicy::New,
        Err(error) => {
            return Err(super::result::filesystem(
                &source_display,
                &target_display,
                &error,
            ))
        }
    };
    Ok(Some(MovePlanEntry {
        source,
        source_display,
        target,
        target_display,
        destination_policy,
    }))
}
