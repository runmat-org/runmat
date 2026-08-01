use crate::NativeCacheError;
use fs2::FileExt;
use std::collections::BTreeSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex, OnceLock};

static LOCAL_LOCKS: OnceLock<(Mutex<BTreeSet<PathBuf>>, Condvar)> = OnceLock::new();

#[derive(Debug)]
pub struct ProcessLock {
    file: File,
    path: PathBuf,
}

impl ProcessLock {
    pub fn acquire(path: &Path) -> Result<Self, NativeCacheError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| NativeCacheError::io(parent, error))?;
        }
        acquire_local(path)?;
        let file = match std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)
        {
            Ok(file) => file,
            Err(error) => {
                release_local(path);
                return Err(NativeCacheError::io(path, error));
            }
        };
        if let Err(error) = file.lock_exclusive() {
            release_local(path);
            return Err(NativeCacheError::io(path, error));
        }
        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
    }
}

impl Drop for ProcessLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
        release_local(&self.path);
    }
}

fn acquire_local(path: &Path) -> Result<(), NativeCacheError> {
    let (held, available) =
        LOCAL_LOCKS.get_or_init(|| (Mutex::new(BTreeSet::new()), Condvar::new()));
    let mut held = held
        .lock()
        .map_err(|error| NativeCacheError::Config(format!("local lock poisoned: {error}")))?;
    while held.contains(path) {
        held = available
            .wait(held)
            .map_err(|error| NativeCacheError::Config(format!("local lock poisoned: {error}")))?;
    }
    held.insert(path.to_path_buf());
    Ok(())
}

fn release_local(path: &Path) {
    let (held, available) =
        LOCAL_LOCKS.get_or_init(|| (Mutex::new(BTreeSet::new()), Condvar::new()));
    if let Ok(mut held) = held.lock() {
        held.remove(path);
        available.notify_all();
    }
}
