//! The memory one study publication may stage.
//!
//! A study publication is staged whole in the publishing process before it commits. Its budget
//! comes from the run's admitted memory authority: the memory the host admitted for the run,
//! which also admits the solve and sizes the thread plan, less the memory the publishing process
//! already holds when the budget is read. The host that owns machine discovery installs that
//! authority (the native runner on Desktop and the CLI, the browser module on the web). A host
//! may also set an explicit cap in `FeaRuntimeConfig::publication_staging_cap_bytes`; the budget
//! is then the smaller of the two. With neither, publication refuses to start.
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

/// The staging budget: the authority's admitted memory less what the process holds, lowered to
/// the host cap when one is set.
pub(super) fn staging_budget(cap: Option<u64>) -> Result<u64, String> {
    let authority = authority_slot()
        .read()
        .map_err(|_| "FEA publication memory authority lock poisoned".to_string())?
        .clone();
    let admitted = authority
        .map(|authority| authority.publication_memory()?.staging_bytes())
        .transpose()?;
    budget_from(admitted, cap)
}

fn budget_from(admitted: Option<u64>, cap: Option<u64>) -> Result<u64, String> {
    match (admitted, cap) {
        (Some(admitted), Some(cap)) => Ok(admitted.min(cap)),
        (Some(bytes), None) | (None, Some(bytes)) => Ok(bytes),
        (None, None) => Err(
            "This host has not admitted any memory for saving study results. \
             Run the study from RunMat Desktop or the RunMat CLI, or set a publication staging \
             cap in the host's FEA configuration."
                .into(),
        ),
    }
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
