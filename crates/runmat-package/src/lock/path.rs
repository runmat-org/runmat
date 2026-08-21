use super::PackageLock;
use crate::LockError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathLockMode {
    Live,
    Locked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PathLockDecision {
    UseExisting,
    WriteGenerated,
}

pub fn reconcile_path_lock(
    generated: &PackageLock,
    existing: Option<&PackageLock>,
    mode: PathLockMode,
) -> Result<PathLockDecision, LockError> {
    generated.validate()?;
    if let Some(existing) = existing {
        existing.validate()?;
        if existing == generated {
            return Ok(PathLockDecision::UseExisting);
        }
    }
    match mode {
        PathLockMode::Live => Ok(PathLockDecision::WriteGenerated),
        PathLockMode::Locked if existing.is_none() => Err(LockError::Incompatible(
            "locked mode requires an existing runmat.lock".to_string(),
        )),
        PathLockMode::Locked => Err(LockError::Incompatible(
            "runmat.lock is stale for the current path dependency contents or selection"
                .to_string(),
        )),
    }
}
