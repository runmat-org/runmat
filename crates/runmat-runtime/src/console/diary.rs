//! Session-owned diary sink. The opened handle, provider, and resolved path are
//! one authority; subsequent console writes never revisit ambient filesystem state.

use super::{with_console_state, ConsoleEntry, ConsoleStream};
use runmat_filesystem::{
    self as fs, File, FsProvider, FsResourceContract, FsResourceLease, OpenOptions,
};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

struct DiarySink {
    _provider: Arc<dyn FsProvider>,
    _resolved_path: PathBuf,
    resource_lease: Option<(Arc<dyn FsResourceLease>, FsResourceContract)>,
    file: File,
}

impl DiarySink {
    fn check_resource_lease(&self) -> io::Result<()> {
        if self
            .resource_lease
            .as_ref()
            .is_some_and(|(lease, contract)| !lease.is_valid() || lease.contract() != *contract)
        {
            return Err(io::Error::other(
                "diary filesystem resource lease was invalidated",
            ));
        }
        Ok(())
    }
}

pub(crate) struct DiaryState {
    filename: PathBuf,
    sink: Option<DiarySink>,
    last_error: Option<io::Error>,
}

impl std::fmt::Debug for DiaryState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DiaryState")
            .field("filename", &self.filename)
            .field("enabled", &self.sink.is_some())
            .field("has_error", &self.last_error.is_some())
            .finish()
    }
}

impl Default for DiaryState {
    fn default() -> Self {
        Self {
            filename: PathBuf::from("diary"),
            sink: None,
            last_error: None,
        }
    }
}

fn open_sink(filename: &Path) -> io::Result<DiarySink> {
    let (provider, cwd) = fs::current_provider_with_current_dir()?;
    let resolved_path = if filename.is_absolute() {
        filename.to_path_buf()
    } else {
        cwd.join(filename)
    };
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    // This optional lease is a provider-configuration guard, not a claim that
    // diary's complete call has a finite memory bound or that it grants access.
    let resource_lease = provider.resource_lease().map(|lease| {
        let contract = lease.contract();
        (lease, contract)
    });
    if resource_lease
        .as_ref()
        .is_some_and(|(lease, _)| !lease.is_valid())
    {
        return Err(io::Error::other(
            "diary filesystem resource lease was invalidated",
        ));
    }
    let file = options.open_on(provider.as_ref(), &resolved_path)?;
    let sink = DiarySink {
        _provider: provider,
        _resolved_path: resolved_path,
        resource_lease,
        file,
    };
    sink.check_resource_lease()?;
    Ok(sink)
}

pub fn diary_enabled() -> bool {
    with_console_state(|state| state.diary.sink.is_some())
}

pub fn diary_filename() -> PathBuf {
    with_console_state(|state| state.diary.filename.clone())
}

pub fn take_diary_error() -> Option<io::Error> {
    with_console_state(|state| state.diary.last_error.take())
}

pub fn set_diary_filename_checked(filename: impl Into<PathBuf>) -> io::Result<()> {
    let filename = filename.into();
    let sink = open_sink(&filename)?;
    with_console_state(|state| {
        state.diary.filename = filename;
        state.diary.sink = Some(sink);
        state.diary.last_error = None;
    });
    Ok(())
}

/// Compatibility wrapper for direct console clients. Unlike the checked
/// builtin path it records an open failure for the enclosing execution.
pub fn set_diary_filename(filename: impl Into<PathBuf>) {
    if let Err(error) = set_diary_filename_checked(filename) {
        record_diary_failure(error);
    }
}

pub fn set_diary_enabled_checked(enabled: bool) -> io::Result<()> {
    if !enabled {
        with_console_state(|state| state.diary.sink = None);
        return Ok(());
    }
    if diary_enabled() {
        return Ok(());
    }
    let filename = diary_filename();
    let sink = open_sink(&filename)?;
    with_console_state(|state| {
        state.diary.sink = Some(sink);
        state.diary.last_error = None;
    });
    Ok(())
}

pub fn set_diary_enabled(enabled: bool) {
    if let Err(error) = set_diary_enabled_checked(enabled) {
        record_diary_failure(error);
    }
}

pub fn toggle_diary() -> io::Result<()> {
    set_diary_enabled_checked(!diary_enabled())
}

pub fn ensure_diary_writable() -> io::Result<()> {
    if diary_enabled() {
        return Ok(());
    }
    open_sink(&diary_filename()).map(|_| ())
}

/// Append top-level source without echoing it into command-window output.
pub fn record_diary_command(text: &str) {
    let mut owned = text.to_string();
    if !owned.ends_with('\n') {
        owned.push('\n');
    }
    write_diary_text(&owned);
}

pub(super) fn write_diary_entry(entry: &ConsoleEntry) {
    if matches!(entry.stream, ConsoleStream::Stdout | ConsoleStream::Stderr) {
        write_diary_text(&entry.text);
    }
}

fn write_diary_text(text: &str) {
    with_console_state(|state| {
        let Some(sink) = state.diary.sink.as_mut() else {
            return;
        };
        if let Err(error) = sink
            .check_resource_lease()
            .and_then(|()| sink.file.write_all(text.as_bytes()))
            .and_then(|()| sink.file.flush())
        {
            state.diary.sink = None;
            state.diary.last_error.get_or_insert(error);
        }
    });
}

fn record_diary_failure(error: io::Error) {
    with_console_state(|state| {
        state.diary.sink = None;
        state.diary.last_error.get_or_insert(error);
    });
}

#[cfg(test)]
mod tests;
