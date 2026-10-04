//! One selected-provider writer authority shared by cell and spreadsheet sinks.

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use futures::lock::Mutex as AsyncMutex;
use runmat_filesystem::OpenOptions;

use super::path::TabularFsPath;

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

pub(super) async fn safe_replace_file(selected: &TabularFsPath, bytes: &[u8]) -> io::Result<()> {
    let temp_path = temporary_sibling_path(selected.path());
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    let mut file = selected.open_sibling_with(&temp_path, &options).await?;
    let staged = async {
        for chunk in bytes.chunks(4096) {
            selected.check()?;
            file.write_all(chunk)?;
        }
        selected.check()?;
        file.flush_async().await?;
        selected.check()?;
        file.sync_all_async().await?;
        selected.check()
    }
    .await;
    drop(file);
    if let Err(source) = staged {
        let _ = selected.remove_sibling(&temp_path).await;
        return Err(source);
    }
    if let Err(source) = selected.replace_from_sibling(&temp_path).await {
        let _ = selected.remove_sibling(&temp_path).await;
        return Err(source);
    }
    Ok(())
}

fn temporary_sibling_path(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("tabular");
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    parent.join(format!(".{name}.runmat-tmp-{}-{nanos}", std::process::id()))
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

        futures::executor::block_on(safe_replace_file(&selected, b"new\n"))
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
