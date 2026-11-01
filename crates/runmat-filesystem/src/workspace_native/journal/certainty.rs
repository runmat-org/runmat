//! Fresh synchronization establishes durability; a decoded receipt alone cannot.
use super::*;

#[cfg(all(test, unix))]
mod tests;

/// Structured cause retained by participating readers and converted only by exact retries.
#[derive(Debug)]
pub(crate) struct UncertainCommit {
    pub(crate) digest: [u8; 32],
    pub(crate) receipt: WorkspacePublicationReceipt,
    cause: io::Error,
}
impl std::fmt::Display for UncertainCommit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "workspace commit durability is indeterminate for {:?}: {}",
            self.receipt, self.cause
        )
    }
}
impl std::error::Error for UncertainCommit {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

pub(in crate::workspace_native) fn confirm(directory: &Path, record: &Record) -> io::Result<()> {
    let confirmed = (|| {
        // Windows FlushFileBuffers requires a writable handle. Opening the
        // existing record for synchronization must never create or truncate it.
        fs::OpenOptions::new()
            .write(true)
            .open(directory.join("commit.json"))?
            .sync_all()?;
        sync_journal_directories(directory)
    })();
    confirmed.map_err(|cause| {
        io::Error::other(UncertainCommit {
            digest: record.digest,
            receipt: record.receipt.clone(),
            cause,
        })
    })
}

pub(in crate::workspace_native) fn uncertain_outcome(
    error: &io::Error,
    request: &WorkspacePublicationRequest,
) -> Option<WorkspacePublicationOutcome> {
    let uncertain = error.get_ref()?.downcast_ref::<UncertainCommit>()?;
    if uncertain.receipt.id != request.id {
        return None;
    }
    Some(if publication_digest(request).ok()? == uncertain.digest {
        WorkspacePublicationOutcome::DurabilityIndeterminate {
            receipt: uncertain.receipt.clone(),
        }
    } else {
        WorkspacePublicationOutcome::IdReused
    })
}

pub(crate) fn uncertain_receipt(error: &io::Error) -> Option<&WorkspacePublicationReceipt> {
    Some(&error.get_ref()?.downcast_ref::<UncertainCommit>()?.receipt)
}
