//! One selected-provider writer authority shared by cell and spreadsheet sinks.

use std::collections::HashMap;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use super::cooperation::{await_io, TabularCooperation};
use super::path::TabularFsPath;
use crate::context::RuntimeContext;
use crate::RuntimeError;
use futures::lock::Mutex as AsyncMutex;

pub(super) type WriteLock = Arc<AsyncMutex<()>>;
type WeakWriteLock = Weak<AsyncMutex<()>>;
type LockKey = (usize, PathBuf);
static WRITE_LOCKS: OnceLock<Mutex<HashMap<LockKey, WeakWriteLock>>> = OnceLock::new();

pub(super) async fn write_lock_for_path(selected: &TabularFsPath) -> io::Result<WriteLock> {
    let key = (
        selected.provider_identity(),
        write_lock_key(selected).await?,
    );
    let locks = WRITE_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut locks = locks.lock().expect("tabular write lock registry poisoned");
    if let Some(lock) = locks.get(&key).and_then(Weak::upgrade) {
        return Ok(lock);
    }
    locks.retain(|_, lock| lock.strong_count() > 0);
    let lock = Arc::new(AsyncMutex::new(()));
    locks.insert(key, Arc::downgrade(&lock));
    Ok(lock)
}

async fn write_lock_key(selected: &TabularFsPath) -> io::Result<PathBuf> {
    let path = selected.path();
    if let Ok(canonical) = selected.canonicalize_on_owner(path).await {
        return Ok(canonical);
    }
    selected.check()?;
    let mut candidate = path;
    let mut suffix = PathBuf::new();
    loop {
        if let Ok(canonical) = selected.canonicalize_on_owner(candidate).await {
            return Ok(if suffix.as_os_str().is_empty() {
                canonical
            } else {
                canonical.join(suffix)
            });
        }
        selected.check()?;
        let Some(name) = candidate.file_name() else {
            break;
        };
        let mut next_suffix = PathBuf::from(name);
        if !suffix.as_os_str().is_empty() {
            next_suffix.push(&suffix);
        }
        suffix = next_suffix;
        let Some(parent) = candidate.parent() else {
            break;
        };
        if parent == candidate {
            break;
        }
        candidate = parent;
    }
    selected.check()?;
    Ok(lexical_normalize(path))
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

#[derive(Debug)]
pub(super) enum StagedWriteError {
    Io(io::Error),
    Control(RuntimeError),
}

pub(super) async fn safe_replace_file(
    runtime: Option<&RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    builtin: &'static str,
    selected: &TabularFsPath,
    bytes: &[u8],
) -> Result<(), StagedWriteError> {
    let mut stage = selected
        .begin_staged_replace()
        .map_err(StagedWriteError::Io)?
        .ok_or_else(|| {
            StagedWriteError::Io(io::Error::new(
                io::ErrorKind::Unsupported,
                "selected filesystem provider cannot stage a tabular replacement",
            ))
        })?;
    for chunk in bytes.chunks(4096) {
        let mut remaining = chunk;
        while !remaining.is_empty() {
            cooperation
                .completed(remaining.len())
                .await
                .map_err(StagedWriteError::Control)?;
            selected.check().map_err(StagedWriteError::Io)?;
            let written = await_io(runtime, builtin, stage.write_chunk(remaining))
                .await
                .map_err(StagedWriteError::Control)?;
            match written {
                Ok(0) => {
                    return Err(StagedWriteError::Io(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "staged tabular write made no progress",
                    )))
                }
                Ok(n) if n > remaining.len() => {
                    return Err(StagedWriteError::Io(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "staged tabular write reported excess progress",
                    )))
                }
                Ok(n) => remaining = &remaining[n..],
                Err(source) if source.kind() == io::ErrorKind::Interrupted => continue,
                Err(source) => return Err(StagedWriteError::Io(source)),
            }
        }
    }
    cooperation
        .completed(0)
        .await
        .map_err(StagedWriteError::Control)?;
    selected.check().map_err(StagedWriteError::Io)?;
    await_io(runtime, builtin, stage.flush_and_sync())
        .await
        .map_err(StagedWriteError::Control)?
        .map_err(StagedWriteError::Io)?;
    cooperation
        .completed(0)
        .await
        .map_err(StagedWriteError::Control)?;
    selected.check().map_err(StagedWriteError::Io)?;
    await_io(runtime, builtin, stage.publish())
        .await
        .map_err(StagedWriteError::Control)?
        .map_err(StagedWriteError::Io)?;
    selected.check().map_err(StagedWriteError::Io)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::io::tabular::path::TabularFsOwner;
    use runmat_filesystem::{self as fs, MemoryFsProvider};

    fn root() -> &'static str {
        if cfg!(windows) {
            r"C:\tabular"
        } else {
            "/tabular"
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn replacement_and_lock_remain_on_selected_provider_after_ambient_swap() {
        let _lock = fs::provider_override_lock();
        let destination = PathBuf::from(root()).join("report.csv");
        let original = Arc::new(MemoryFsProvider::with_current_dir(root()));
        original
            .write_project_path(&destination, b"old\n")
            .expect("original fixture");
        let _original_guard = fs::replace_provider(original.clone());
        fs::set_current_dir(root()).expect("original cwd");
        let selected = TabularFsOwner::capture(Some(&destination))
            .expect("capture")
            .path(destination.clone())
            .expect("selected path");
        let original_lock =
            futures::executor::block_on(write_lock_for_path(&selected)).expect("original lock");

        let replacement = Arc::new(MemoryFsProvider::with_current_dir(root()));
        replacement
            .write_project_path(&destination, b"other\n")
            .expect("replacement fixture");
        let _replacement_guard = fs::replace_provider(replacement.clone());
        fs::set_current_dir(root()).expect("replacement cwd");
        let replacement_selected = TabularFsOwner::capture(Some(&destination))
            .expect("capture replacement")
            .path(destination.clone())
            .expect("replacement path");
        let replacement_lock =
            futures::executor::block_on(write_lock_for_path(&replacement_selected))
                .expect("replacement lock");
        assert!(!Arc::ptr_eq(&original_lock, &replacement_lock));

        let mut cooperation = TabularCooperation::new(None, "writecell");
        futures::executor::block_on(safe_replace_file(
            None,
            &mut cooperation,
            "writecell",
            &selected,
            b"new\n",
        ))
        .expect("replace with selected owner");
        assert_eq!(
            original.read_project_path(&destination).expect("original"),
            b"new\n"
        );
        assert_eq!(
            replacement
                .read_project_path(&destination)
                .expect("replacement"),
            b"other\n"
        );
    }
}
