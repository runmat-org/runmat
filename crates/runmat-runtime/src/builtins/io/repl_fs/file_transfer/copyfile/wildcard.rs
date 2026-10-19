use std::io;
use std::path::PathBuf;

use super::super::outcome::TransferOutcome;
use super::plan::{CopyPlanEntry, TargetKind};

pub(super) async fn copy(
    fs: &mut super::TransferFilesystem,
    pattern: &str,
    destination: &str,
    force: bool,
) -> TransferOutcome {
    let matches = match fs.wildcard_matches(pattern).await {
        Ok(matches) if matches.is_empty() => return super::result::source_not_found(pattern),
        Ok(matches) => matches,
        Err(error) if error.kind() == io::ErrorKind::InvalidInput => {
            return super::result::invalid_pattern(pattern, &error.to_string());
        }
        Err(error) => return super::result::filesystem(pattern, destination, &error),
    };
    let destination_path = PathBuf::from(destination);
    match fs.metadata(&destination_path).await {
        Ok(metadata) if !metadata.is_dir() => {
            return super::result::destination_not_directory(destination);
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return super::result::destination_missing(destination);
        }
        Err(error) => return super::result::filesystem(pattern, destination, &error),
    }
    let entries = match plan(fs, matches, &destination_path, force).await {
        Ok(entries) => entries,
        Err(outcome) => return outcome,
    };
    match super::plan::execute(fs, &entries).await {
        Ok(()) => super::result::success(),
        Err(failure) => super::result::filesystem(&failure.source, &failure.target, &failure.error),
    }
}

async fn plan(
    fs: &super::TransferFilesystem,
    sources: Vec<PathBuf>,
    destination: &std::path::Path,
    force: bool,
) -> Result<Vec<CopyPlanEntry>, TransferOutcome> {
    let mut entries = Vec::with_capacity(sources.len());
    for source in sources {
        entries.push(plan_entry(fs, source, destination, force).await?);
    }
    Ok(entries)
}

async fn plan_entry(
    fs: &super::TransferFilesystem,
    source: PathBuf,
    destination: &std::path::Path,
    force: bool,
) -> Result<CopyPlanEntry, TransferOutcome> {
    let source_display = super::super::target::display(&source);
    let metadata = fs.metadata(&source).await.map_err(|error| {
        if error.kind() == io::ErrorKind::NotFound {
            super::result::source_not_found(&source_display)
        } else {
            super::result::filesystem(
                &source_display,
                &super::super::target::display(destination),
                &error,
            )
        }
    })?;
    let name = source.file_name().ok_or_else(|| {
        super::result::filesystem(
            &source_display,
            &super::super::target::display(destination),
            &io::Error::other("cannot determine source name"),
        )
    })?;
    let target = destination.join(name);
    if fs.same_path(&source, &target).await {
        return Err(super::result::same_path(&source_display));
    }
    let target_display = super::super::target::display(&target);
    let replace = match fs.metadata(&target).await {
        Ok(_) if !force => return Err(super::result::destination_exists(&target_display)),
        Ok(existing) if existing.is_dir() => Some(TargetKind::Directory),
        Ok(_) => Some(TargetKind::File),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(super::result::filesystem(
                &source_display,
                &target_display,
                &error,
            ))
        }
    };
    Ok(CopyPlanEntry {
        source,
        source_display,
        target,
        target_display,
        source_is_directory: metadata.is_dir(),
        replace,
    })
}
