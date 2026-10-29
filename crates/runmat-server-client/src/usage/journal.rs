//! Write-ahead usage journal. Short OS-file-locked transactions never hold locks across HTTP.
use super::types::*;
use anyhow::{Context, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Entry {
    pub request: CreateLeaseRequest,
    pub lease: LeaseResponse,
    pub usage: UsageReport,
    pub heartbeat: Option<UsageReport>,
    pub settlement: Option<Settlement>,
}

#[derive(Default, Serialize, Deserialize)]
pub(super) struct State {
    pub keys: SigningKeys,
    pub runs: BTreeMap<String, Entry>,
    #[serde(default)]
    pub rejected_creates: BTreeMap<String, super::cleanup::RejectedCreate>,
}

#[derive(Clone)]
pub(super) struct Journal {
    path: PathBuf,
}

impl Journal {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn read(&self) -> Result<State> {
        self.transaction(|state| Ok(std::mem::take(state)), false)
    }

    pub fn update<T>(&self, action: impl FnOnce(&mut State) -> Result<T>) -> Result<T> {
        self.transaction(action, true)
    }

    fn transaction<T>(
        &self,
        action: impl FnOnce(&mut State) -> Result<T>,
        write: bool,
    ) -> Result<T> {
        let parent = self.path.parent().context("Usage journal has no parent")?;
        fs::create_dir_all(parent)?;
        let lock = private_file(&self.path.with_extension("lock"))?;
        lock.lock_exclusive()?;
        let mut state = match fs::read(&self.path) {
            Ok(bytes) => {
                serde_json::from_slice(&bytes).context("Invalid core-hour usage journal")?
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => State::default(),
            Err(error) => return Err(error.into()),
        };
        let result = action(&mut state)?;
        if write {
            let temporary = self
                .path
                .with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
            let save = || -> Result<()> {
                let mut file = private_file(&temporary)?;
                file.write_all(&serde_json::to_vec(&state)?)?;
                file.sync_all()?;
                fs::rename(&temporary, &self.path)?;
                #[cfg(unix)]
                File::open(parent)?.sync_all()?;
                Ok(())
            };
            if let Err(error) = save() {
                let _ = fs::remove_file(&temporary);
                return Err(error);
            }
        }
        Ok(result)
    }
}

fn private_file(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}
