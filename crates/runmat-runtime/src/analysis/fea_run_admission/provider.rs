use super::{FeaRunAdmission, FeaRunDenied};
use std::sync::{Arc, OnceLock, RwLock};

fn slot() -> &'static RwLock<Option<Arc<dyn FeaRunAdmission>>> {
    static SLOT: OnceLock<RwLock<Option<Arc<dyn FeaRunAdmission>>>> = OnceLock::new();
    SLOT.get_or_init(|| RwLock::new(None))
}

pub struct FeaRunAdmissionGuard {
    previous: Option<Arc<dyn FeaRunAdmission>>,
}

impl Drop for FeaRunAdmissionGuard {
    fn drop(&mut self) {
        // Even a poisoned host lock must not leave this session's adapter installed.
        let mut current = slot().write().unwrap_or_else(|e| e.into_inner());
        *current = self.previous.take();
    }
}

pub fn replace_fea_run_admission(
    admission: Option<Arc<dyn FeaRunAdmission>>,
) -> Result<FeaRunAdmissionGuard, String> {
    let mut current = slot()
        .write()
        .map_err(|_| "FEA run admission lock poisoned")?;
    Ok(FeaRunAdmissionGuard {
        previous: std::mem::replace(&mut *current, admission),
    })
}

pub(super) fn current() -> Result<Arc<dyn FeaRunAdmission>, FeaRunDenied> {
    if let Some(admission) = super::host::current() {
        return Ok(admission);
    }
    if let Some(admission) = crate::context::legacy::active()
        .and_then(|runtime| runtime.service_ports().sync_fea_run_admission().cloned())
    {
        return Ok(admission);
    }
    slot()
        .read()
        .map_err(|_| FeaRunDenied::LicenseInvalid)?
        .clone()
        .ok_or(FeaRunDenied::SignInRequired)
}

/// Trusted owned-dispatch boundary: retain the originating host adapter, not
/// credentials or a newly granted lease. Plain dispatch performs no admission.
pub fn capture_fea_run_admission() -> Result<Arc<dyn FeaRunAdmission>, FeaRunDenied> {
    current()
}

/// Capture the local lease selector without loading credentials or contacting Server.
/// A dispatched/cluster host cannot request a second local lease through this hint.
pub fn current_local_fea_organization_hint() -> super::FeaLocalOrganization {
    if super::FeaRunScope::current_site().is_some_and(|site| site != super::FeaRunSite::Local) {
        return super::FeaLocalOrganization::Default;
    }
    current()
        .ok()
        .filter(|provider| provider.site() == super::FeaRunSite::Local)
        .map_or(super::FeaLocalOrganization::Default, |provider| {
            provider.local_organization_hint()
        })
}

/// Pure host capacity metadata, never an admission or network request.
pub fn current_fea_cpu_capacity_millicores() -> Option<u64> {
    super::FeaRunScope::current_allocation()
        .or_else(|| current().ok()?.allocated_cpu_capacity_millicores())
}
