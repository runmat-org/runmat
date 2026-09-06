use runmat_filesystem as vfs;
use std::io;
use std::path::PathBuf;

use super::super::outcome::TransferOutcome;
use super::plan::MovePlanEntry;

pub(super) async fn move_matches(pattern: &str, destination: &str, force: bool) -> TransferOutcome {
    let matches = match super::super::target::wildcard_matches(pattern).await {
        Ok(matches) if matches.is_empty() => return super::result::source_not_found(pattern),
        Ok(matches) => matches,
        Err(error) if error.kind() == io::ErrorKind::InvalidInput => {
            return super::result::invalid_pattern(pattern, &error.to_string());
        }
        Err(error) => return super::result::filesystem(pattern, destination, &error),
    };
    let destination_path = PathBuf::from(destination);
    match vfs::metadata_async(&destination_path).await {
        Ok(metadata) if !metadata.is_dir() => {
            return super::result::destination_not_directory(destination);
        }
        Ok(_) => {}
        Err(_) => return super::result::destination_missing(destination),
    }
    let entries = match plan(matches, &destination_path, force).await {
        Ok(entries) => entries,
        Err(outcome) => return outcome,
    };
    match super::plan::execute(&entries).await {
        Ok(()) => super::result::success(),
        Err(failure) => super::result::filesystem(&failure.source, &failure.target, &failure.error),
    }
}

async fn plan(
    sources: Vec<PathBuf>,
    destination: &std::path::Path,
    force: bool,
) -> Result<Vec<MovePlanEntry>, TransferOutcome> {
    let mut entries = Vec::with_capacity(sources.len());
    for source in sources {
        if let Some(entry) = plan_entry(source, destination, force).await? {
            entries.push(entry);
        }
    }
    Ok(entries)
}

async fn plan_entry(
    source: PathBuf,
    destination: &std::path::Path,
    force: bool,
) -> Result<Option<MovePlanEntry>, TransferOutcome> {
    let source_display = super::super::target::display(&source);
    if vfs::metadata_async(&source).await.is_err() {
        return Err(super::result::source_not_found(&source_display));
    }
    let name = source.file_name().ok_or_else(|| {
        super::result::filesystem(
            &source_display,
            &super::super::target::display(destination),
            &io::Error::other("cannot determine source name"),
        )
    })?;
    let target = destination.join(name);
    if target == source {
        return Ok(None);
    }
    let target_display = super::super::target::display(&target);
    let replace_directory = match vfs::metadata_async(&target).await {
        Ok(_) if !force => return Err(super::result::destination_exists(&target_display)),
        Ok(metadata) => Some(metadata.is_dir()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
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
        replace_directory,
    }))
}
