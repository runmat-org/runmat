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
    slot()
        .read()
        .map_err(|_| FeaRunDenied::LicenseInvalid)?
        .clone()
        .ok_or(FeaRunDenied::SignInRequired)
}
