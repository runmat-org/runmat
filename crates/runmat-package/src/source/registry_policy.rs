use super::{SourceAcquisitionIntent, SourceAcquisitionPolicy, SourceLockAction};
use crate::{CanonicalPackageId, RegistryId, RegistrySourceId};
use semver::VersionReq;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use url::Url;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryAcquisitionPlan {
    pub source_registry: RegistryId,
    pub index: String,
    pub package: CanonicalPackageId,
    pub requirement: VersionReq,
    pub allow_network: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<RegistrySourceId>,
    pub lock_action: SourceLockAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryCandidatePlan {
    pub source_registry: RegistryId,
    pub index: String,
    pub package: CanonicalPackageId,
    pub allow_network: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RegistryPolicyError {
    #[error("registry index URL is invalid")]
    InvalidIndex,
    #[error("frozen mode cannot update registry sources")]
    FrozenUpdate,
    #[error("locked mode cannot update registry sources")]
    LockedUpdate,
    #[error("locked or frozen mode requires an exact registry release in runmat.lock")]
    MissingLock,
    #[error("locked registry release does not match the requested package")]
    LockPackageMismatch,
    #[error("locked registry release does not satisfy the manifest version requirement")]
    LockVersionMismatch,
    #[error("registry provider returned {found:?}; expected locked source {expected:?}")]
    LockedSourceMismatch {
        expected: Box<RegistrySourceId>,
        found: Box<RegistrySourceId>,
    },
    #[error("registry provider returned a different package or incompatible version")]
    AcquiredPackageMismatch,
}

pub fn plan_registry_acquisition(
    source_registry: RegistryId,
    index: &str,
    package: CanonicalPackageId,
    requirement: VersionReq,
    locked_source: Option<&RegistrySourceId>,
    intent: SourceAcquisitionIntent,
    policy: SourceAcquisitionPolicy,
) -> Result<RegistryAcquisitionPlan, RegistryPolicyError> {
    let index = normalize_index(index)?;
    if intent == SourceAcquisitionIntent::Update {
        if policy.frozen {
            return Err(RegistryPolicyError::FrozenUpdate);
        }
        if policy.locked {
            return Err(RegistryPolicyError::LockedUpdate);
        }
    }
    if let Some(locked) = locked_source {
        if locked.package != package {
            return Err(RegistryPolicyError::LockPackageMismatch);
        }
        if !requirement.matches(locked.version.as_semver()) {
            return Err(RegistryPolicyError::LockVersionMismatch);
        }
    }
    let use_locked = intent != SourceAcquisitionIntent::Update && locked_source.is_some();
    if !use_locked && (policy.locked || policy.frozen) {
        return Err(RegistryPolicyError::MissingLock);
    }
    Ok(RegistryAcquisitionPlan {
        source_registry,
        index,
        package,
        requirement,
        allow_network: !policy.offline && !policy.frozen,
        expected: use_locked.then(|| locked_source.cloned().expect("checked locked source")),
        lock_action: if use_locked {
            SourceLockAction::Preserve
        } else {
            match intent {
                SourceAcquisitionIntent::Update => SourceLockAction::Replace,
                SourceAcquisitionIntent::Execute | SourceAcquisitionIntent::Fetch => {
                    SourceLockAction::Write
                }
            }
        },
    })
}

pub fn plan_registry_candidates(
    source_registry: RegistryId,
    index: &str,
    package: CanonicalPackageId,
    policy: SourceAcquisitionPolicy,
) -> Result<RegistryCandidatePlan, RegistryPolicyError> {
    Ok(RegistryCandidatePlan {
        source_registry,
        index: normalize_index(index)?,
        package,
        allow_network: !policy.offline && !policy.frozen,
    })
}

pub fn plan_selected_registry_acquisition(
    source_registry: RegistryId,
    index: &str,
    source: RegistrySourceId,
    intent: SourceAcquisitionIntent,
    policy: SourceAcquisitionPolicy,
) -> Result<RegistryAcquisitionPlan, RegistryPolicyError> {
    if policy.frozen && intent == SourceAcquisitionIntent::Update {
        return Err(RegistryPolicyError::FrozenUpdate);
    }
    if policy.locked && intent == SourceAcquisitionIntent::Update {
        return Err(RegistryPolicyError::LockedUpdate);
    }
    let requirement = VersionReq::parse(&format!("={}", source.version))
        .expect("package versions always form exact requirements");
    Ok(RegistryAcquisitionPlan {
        source_registry,
        index: normalize_index(index)?,
        package: source.package.clone(),
        requirement,
        allow_network: !policy.offline && !policy.frozen,
        expected: Some(source),
        lock_action: match intent {
            SourceAcquisitionIntent::Update => SourceLockAction::Replace,
            SourceAcquisitionIntent::Execute | SourceAcquisitionIntent::Fetch => {
                SourceLockAction::Write
            }
        },
    })
}

pub fn validate_registry_acquisition(
    plan: &RegistryAcquisitionPlan,
    acquired: &RegistrySourceId,
) -> Result<(), RegistryPolicyError> {
    if acquired.package != plan.package || !plan.requirement.matches(acquired.version.as_semver()) {
        return Err(RegistryPolicyError::AcquiredPackageMismatch);
    }
    if let Some(expected) = &plan.expected {
        if acquired != expected {
            return Err(RegistryPolicyError::LockedSourceMismatch {
                expected: Box::new(expected.clone()),
                found: Box::new(acquired.clone()),
            });
        }
    }
    Ok(())
}

fn normalize_index(value: &str) -> Result<String, RegistryPolicyError> {
    let mut url = Url::parse(value).map_err(|_| RegistryPolicyError::InvalidIndex)?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(RegistryPolicyError::InvalidIndex);
    }
    url.set_query(None);
    url.set_fragment(None);
    Ok(url.to_string().trim_end_matches('/').to_string())
}
