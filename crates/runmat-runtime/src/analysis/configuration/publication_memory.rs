//! The memory one study publication may stage.
//!
//! A study publication is staged whole in the publishing process before it commits. Its budget
//! comes from the run's admitted memory authority: the memory admitted for the run, which also
//! admits the solve and sizes the thread plan, less the memory the publishing process already
//! holds when the budget is read. A host that admits runs itself installs that authority (the
//! native runner installs each run's pool on Desktop and the CLI; the browser module installs its
//! instance budget). A process where no host has installed one uses this machine's memory: what
//! the machine or container admits for work when the budget is read
//! (`runmat_cpu_execution::machine_memory`, the probe the native runner's pool also reads). A
//! host may also set an explicit cap in `FeaRuntimeConfig::publication_staging_cap_bytes`; the
//! budget is then the smaller of the two.
use std::sync::{Arc, OnceLock, RwLock};

#[cfg(test)]
mod tests;

/// The run's admitted memory and what the publishing process holds of it now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicationMemory {
    /// Memory the host admitted for the run: the same value that admits the solve.
    pub admitted_bytes: u64,
    /// Memory the publishing process holds resident when the budget is read.
    pub resident_bytes: u64,
}

impl PublicationMemory {
    /// The admitted memory left after what the process already holds.
    pub fn staging_bytes(&self) -> Result<u64, String> {
        match self.admitted_bytes.checked_sub(self.resident_bytes) {
            Some(free) if free > 0 => Ok(free),
            _ => Err(format!(
                "No memory is left to save this study's results: the run was admitted {} and \
                 RunMat already holds {}. Close other applications or open projects to free \
                 memory, then run the study again.",
                byte_text(self.admitted_bytes),
                byte_text(self.resident_bytes),
            )),
        }
    }
}

/// The host's resource authority for one run, read each time a publication budget is needed.
pub trait PublicationMemoryAuthority: Send + Sync {
    fn publication_memory(&self) -> Result<PublicationMemory, String>;
}

fn authority_slot() -> &'static RwLock<Option<Arc<dyn PublicationMemoryAuthority>>> {
    static SLOT: OnceLock<RwLock<Option<Arc<dyn PublicationMemoryAuthority>>>> = OnceLock::new();
    SLOT.get_or_init(|| RwLock::new(None))
}

/// Install the run's admitted memory authority and return the one it replaces. The host that
/// admits a run installs it before the run plans or publishes; later runs replace it.
pub fn install_publication_memory_authority(
    authority: Arc<dyn PublicationMemoryAuthority>,
) -> Result<Option<Arc<dyn PublicationMemoryAuthority>>, String> {
    let mut slot = authority_slot()
        .write()
        .map_err(|_| "FEA publication memory authority lock poisoned".to_string())?;
    Ok(slot.replace(authority))
}

/// Remove the installed authority, returning it. Used by hosts and tests that scope one.
pub fn clear_publication_memory_authority(
) -> Result<Option<Arc<dyn PublicationMemoryAuthority>>, String> {
    let mut slot = authority_slot()
        .write()
        .map_err(|_| "FEA publication memory authority lock poisoned".to_string())?;
    Ok(slot.take())
}

/// The staging budget: the installed authority's admitted memory less what the process holds,
/// or this machine's memory when no host installed an authority, lowered to the host cap when
/// one is set.
pub(super) fn staging_budget(cap: Option<u64>) -> Result<u64, String> {
    let authority = authority_slot()
        .read()
        .map_err(|_| "FEA publication memory authority lock poisoned".to_string())?
        .clone();
    let admitted = match authority {
        Some(authority) => authority.publication_memory()?,
        None => machine_publication_memory()?,
    }
    .staging_bytes()?;
    Ok(cap.map_or(admitted, |cap| admitted.min(cap)))
}

/// This machine's memory as the publication authority. The machine's admissible memory is read
/// now, so it already excludes what this process holds: the process is admitted its footprint
/// plus that memory and holds its footprint, which leaves the admissible memory to stage.
#[cfg(not(target_arch = "wasm32"))]
fn machine_publication_memory() -> Result<PublicationMemory, String> {
    use runmat_cpu_execution::machine_memory;
    let unreadable = |error: std::io::Error| {
        format!("RunMat could not read this machine's memory to save study results: {error}")
    };
    let resident_bytes = machine_memory::process_footprint_bytes().map_err(unreadable)?;
    let admissible = machine_memory::admissible_memory_bytes().map_err(unreadable)?;
    Ok(PublicationMemory {
        admitted_bytes: resident_bytes.saturating_add(admissible),
        resident_bytes,
    })
}

/// A browser instance's memory is measured by the module that owns it, which installs the
/// authority when it starts; a wasm process without that module has no memory to stage in.
#[cfg(target_arch = "wasm32")]
fn machine_publication_memory() -> Result<PublicationMemory, String> {
    Err(
        "This browser instance has not admitted any memory for saving study results. \
         Start RunMat through its browser module, which admits the instance's memory."
            .into(),
    )
}

/// The plain-language refusal for a publication that needs more memory than its budget.
/// `facts` are the measured terms, kept after the sentence for diagnosis.
pub(in crate::analysis) fn over_budget_message(required: u64, maximum: u64, facts: &str) -> String {
    let capped = super::current_fea_runtime_config().publication_staging_cap_bytes == Some(maximum);
    let limit = if capped {
        "the publication staging cap this host sets"
    } else {
        "the memory this run may use to save results (its admitted memory less what RunMat \
         already holds)"
    };
    let raise = if capped {
        "raise the host's publication staging cap, "
    } else {
        "close other applications or open projects to free memory, "
    };
    format!(
        "Saving this study's results needs {} of memory, more than the {} available: {limit}. \
         To save it, {raise}or make the result smaller with a coarser mesh (a larger element \
         size), fewer result outputs or fewer steps, then run the study again. Details: {facts}.",
        byte_text(required),
        byte_text(maximum),
    )
}

/// Bytes as decimal megabytes or gigabytes with one decimal place.
fn byte_text(bytes: u64) -> String {
    const MB: f64 = 1e6;
    const GB: f64 = 1e9;
    let value = bytes as f64;
    if value >= GB {
        format!("{:.1} GB", value / GB)
    } else {
        format!("{:.1} MB", value / MB)
    }
}
