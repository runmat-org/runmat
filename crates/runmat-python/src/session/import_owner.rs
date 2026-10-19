//! Process-wide claims for private wheel import names while a lane is live.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Mutex;

use runmat_process_host::private_bytes::PrivateByteLease;

use crate::{InstalledPythonArtifacts, PythonError};

static CLAIMED_IMPORT_NAMES: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

pub(super) struct ManagedPythonImports {
    pub(super) roots: Vec<PathBuf>,
    pub(super) names: BTreeSet<String>,
    pub(super) byte_lease: PrivateByteLease,
}

pub(super) struct ProcessImportReservation {
    imports: ManagedPythonImports,
    release_on_drop: bool,
}

impl ProcessImportReservation {
    pub(super) fn reserve(installed: &InstalledPythonArtifacts) -> Result<Self, PythonError> {
        let names = installed.top_level_modules();
        let mut claimed = CLAIMED_IMPORT_NAMES.lock().map_err(|_| {
            PythonError::host(
                "PythonImportOwnerError",
                "the process Python import owner lock was poisoned",
            )
        })?;
        if let Some(conflict) = names.iter().find(|name| claimed.contains(*name)) {
            return Err(PythonError::host(
                "PythonImportOwnerError",
                format!("Python module `{conflict}` belongs to another live project lane"),
            ));
        }
        claimed.extend(names.iter().cloned());
        Ok(Self {
            imports: ManagedPythonImports {
                roots: installed.module_paths().to_vec(),
                names: names.clone(),
                byte_lease: installed.byte_lease(),
            },
            release_on_drop: true,
        })
    }

    pub(super) fn imports(&self) -> &ManagedPythonImports {
        &self.imports
    }

    /// Failed CPython cleanup leaves a process-global conflict, never a
    /// reusable claim for a different generation of the same module names.
    pub(super) fn retain_conflict(&mut self) {
        self.release_on_drop = false;
    }
}

impl Drop for ProcessImportReservation {
    fn drop(&mut self) {
        if !self.release_on_drop || std::thread::panicking() {
            return;
        }
        let Ok(mut claimed) = CLAIMED_IMPORT_NAMES.lock() else {
            return;
        };
        for name in &self.imports.names {
            claimed.remove(name);
        }
    }
}
