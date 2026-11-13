//! Process-wide CPython identity and private wheel lifetime authority.

use std::path::PathBuf;
use std::sync::Mutex;

use runmat_process_host::private_bytes::PrivateByteLease;

use super::api::CpythonApi;
use crate::PythonError;

pub(super) struct ProcessInterpreterOwner {
    pub(super) library: Option<PathBuf>,
    exposed_wheels: Vec<PrivateByteLease>,
}

pub(super) static PROCESS_INTERPRETER: Mutex<ProcessInterpreterOwner> =
    Mutex::new(ProcessInterpreterOwner {
        library: None,
        exposed_wheels: Vec::new(),
    });

impl CpythonApi {
    /// This lease comes only from InstalledPythonArtifacts, whose private
    /// wheel root is independently Owned; selected parent-owned roots are not
    /// eligible for this process ownership transition.
    pub(crate) fn retain_exposed_wheel(&self, lease: &PrivateByteLease) -> Result<(), PythonError> {
        let mut process_owner = PROCESS_INTERPRETER.lock().map_err(|_| {
            PythonError::host(
                "PythonImportOwnerError",
                "the process CPython identity lock was poisoned",
            )
        })?;
        if process_owner
            .exposed_wheels
            .iter()
            .any(|existing| existing.path() == lease.path())
        {
            return Ok(());
        }
        // CPython's main interpreter, native extensions, and user threads can
        // retain objects after a lane retires. Statics do not run Rust Drop at
        // process exit, so the exact root is marked for ProcessHost recovery.
        lease
            .persist_unverified(Some(std::process::id()), "in-process CPython wheel owner")
            .map_err(|error| PythonError::host("PythonImportOwnerError", error))?;
        process_owner.exposed_wheels.push(lease.clone());
        Ok(())
    }
}
