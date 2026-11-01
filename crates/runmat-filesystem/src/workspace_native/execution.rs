//! Ownership records share the publication lock and its protected journal.
//! Publication validates these records while holding that same lock.
use super::*;
use crate::execution::*;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    work: ExecutionWorkId,
    state: ExecutionLeaseState,
}

pub(crate) fn transition_execution(
    root: &Path,
    work: &ExecutionWorkId,
    command: ExecutionLeaseCommand,
) -> io::Result<ExecutionLeaseOutcome> {
    if !cfg!(any(unix, windows)) {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "native durable execution authority requires directory synchronization",
        ));
    }
    let key = crate::execution::work_key(work)?;
    let _guard = guard(root, true)?;
    let directory = root.join(JOURNAL).join("executions");
    ensure_no_symlink(root, &directory)?;
    fs::create_dir_all(&directory)?;
    let path = directory.join(format!("{key}.json"));
    ensure_no_symlink(root, &path)?;
    let prior = read_record(root, work)?;
    let (next, outcome) = transition_execution_lease(
        work,
        prior.as_ref().map(|record| &record.state),
        &command,
        crate::execution::authority_time_ms()?,
    )
    .map_err(io::Error::other)?;
    if let Some(state) = next {
        if prior.is_none() {
            let mut count = 0;
            for entry in fs::read_dir(&directory)?.take(MAX_EXECUTION_RECORDS) {
                entry?;
                count += 1;
            }
            if count >= MAX_EXECUTION_RECORDS {
                return Err(invalid("execution ownership record budget exhausted"));
            }
        }
        let bytes = serde_json::to_vec(&Record {
            work: work.clone(),
            state,
        })
        .map_err(io::Error::other)?;
        // One preparation slot per work ID bounds crash leftovers. The prior
        // acknowledged state stays authoritative until the atomic rename.
        let temporary = directory.join(format!("{key}.pending"));
        ensure_no_symlink(root, &temporary)?;
        match fs::remove_file(&temporary) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        let result = (|| {
            let mut file = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temporary)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            fs::rename(&temporary, &path)?;
            sync_ancestors(root, &directory)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result?;
    }
    Ok(outcome)
}

fn read_record(root: &Path, work: &ExecutionWorkId) -> io::Result<Option<Record>> {
    let path = root
        .join(JOURNAL)
        .join("executions")
        .join(format!("{}.json", crate::execution::work_key(work)?));
    ensure_no_symlink(root, &path)?;
    match fs::File::open(&path) {
        Ok(file) => {
            let mut bytes = Vec::new();
            file.take(8193).read_to_end(&mut bytes)?;
            if bytes.len() > 8192 {
                return Err(invalid("execution record exceeds its bounded size"));
            }
            let record: Record = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
            if &record.work != work {
                return Err(invalid(
                    "execution record identity differs from its journal path",
                ));
            }
            Ok(Some(record))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// The caller holds the project publication guard through the effect commit.
pub(super) fn require_current(root: &Path, lease: &ExecutionLease) -> io::Result<()> {
    let record = read_record(root, &lease.work)?;
    crate::execution::require_current(record.as_ref().map(|record| &record.state), lease)
}
