use super::{SourceAcquisitionIntent, SourceAcquisitionPolicy, SourceLockAction};
use crate::ServerProjectSourceId;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ServerSnapshotSelector {
    Exact { value: String },
    Tag { value: String },
}

impl ServerSnapshotSelector {
    pub fn from_manifest(value: Option<&str>) -> Result<Self, ServerPolicyError> {
        let value = value.unwrap_or("main").trim();
        validate_selector(value)?;
        Ok(if is_exact_snapshot_id(value)? {
            Self::Exact {
                value: value.to_string(),
            }
        } else {
            Self::Tag {
                value: value.to_string(),
            }
        })
    }

    pub fn value(&self) -> &str {
        match self {
            Self::Exact { value } | Self::Tag { value } => value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerProjectAcquisitionPlan {
    pub service: String,
    pub project: String,
    pub selector: ServerSnapshotSelector,
    pub allow_network: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<ServerProjectSourceId>,
    pub lock_action: SourceLockAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ServerPolicyError {
    #[error("Server snapshot selector is invalid")]
    InvalidSelector,
    #[error("frozen mode cannot update mutable Server snapshot sources")]
    FrozenUpdate,
    #[error("locked mode cannot update mutable Server snapshot sources")]
    LockedUpdate,
    #[error("locked or frozen mode requires an exact Server snapshot source in runmat.lock")]
    MissingLock,
    #[error("locked Server snapshot source does not match the manifest service or project")]
    LockLocatorMismatch,
    #[error("Server provider returned {found:?}; expected locked source {expected:?}")]
    LockedSourceMismatch {
        expected: Box<ServerProjectSourceId>,
        found: Box<ServerProjectSourceId>,
    },
    #[error("Server provider returned a different service or project than requested")]
    AcquiredLocatorMismatch,
    #[error("Server service origin is invalid: {0}")]
    InvalidService(String),
}

pub fn plan_server_project_acquisition(
    service: &str,
    project: &str,
    selector: ServerSnapshotSelector,
    locked_source: Option<&ServerProjectSourceId>,
    intent: SourceAcquisitionIntent,
    policy: SourceAcquisitionPolicy,
) -> Result<ServerProjectAcquisitionPlan, ServerPolicyError> {
    let service = ServerProjectSourceId::normalize_service(service)
        .map_err(|error| ServerPolicyError::InvalidService(error.to_string()))?;
    if intent == SourceAcquisitionIntent::Update {
        if policy.frozen {
            return Err(ServerPolicyError::FrozenUpdate);
        }
        if policy.locked {
            return Err(ServerPolicyError::LockedUpdate);
        }
    }
    if let Some(locked) = locked_source {
        if locked.service != service || locked.project != project {
            return Err(ServerPolicyError::LockLocatorMismatch);
        }
    }
    let use_locked = intent != SourceAcquisitionIntent::Update && locked_source.is_some();
    if !use_locked && (policy.locked || policy.frozen) {
        return Err(ServerPolicyError::MissingLock);
    }
    if use_locked {
        let expected = locked_source.cloned().expect("checked locked source");
        return Ok(ServerProjectAcquisitionPlan {
            service,
            project: project.to_string(),
            selector: ServerSnapshotSelector::Exact {
                value: expected.snapshot.clone(),
            },
            allow_network: !policy.offline && !policy.frozen,
            expected: Some(expected),
            lock_action: SourceLockAction::Preserve,
        });
    }
    Ok(ServerProjectAcquisitionPlan {
        service,
        project: project.to_string(),
        selector,
        allow_network: !policy.offline && !policy.frozen,
        expected: None,
        lock_action: match intent {
            SourceAcquisitionIntent::Update => SourceLockAction::Replace,
            SourceAcquisitionIntent::Execute | SourceAcquisitionIntent::Fetch => {
                SourceLockAction::Write
            }
        },
    })
}

pub fn validate_server_project_acquisition(
    plan: &ServerProjectAcquisitionPlan,
    acquired: &ServerProjectSourceId,
) -> Result<(), ServerPolicyError> {
    if acquired.service != plan.service || acquired.project != plan.project {
        return Err(ServerPolicyError::AcquiredLocatorMismatch);
    }
    if let Some(expected) = &plan.expected {
        if expected != acquired {
            return Err(ServerPolicyError::LockedSourceMismatch {
                expected: Box::new(expected.clone()),
                found: Box::new(acquired.clone()),
            });
        }
    } else if let ServerSnapshotSelector::Exact { value } = &plan.selector {
        if acquired.snapshot != *value {
            return Err(ServerPolicyError::AcquiredLocatorMismatch);
        }
    }
    Ok(())
}

fn validate_selector(value: &str) -> Result<(), ServerPolicyError> {
    if value.is_empty()
        || value.len() > 128
        || value
            .chars()
            .any(|character| character.is_control() || matches!(character, '/' | '\\'))
    {
        Err(ServerPolicyError::InvalidSelector)
    } else {
        Ok(())
    }
}

fn is_exact_snapshot_id(value: &str) -> Result<bool, ServerPolicyError> {
    if !value.starts_with("snap_") {
        return Ok(false);
    }
    let suffix = &value["snap_".len()..];
    if suffix.len() != 32
        || !suffix
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ServerPolicyError::InvalidSelector);
    }
    Ok(true)
}
