use crate::{GitRepositoryUrl, GitSelector, GitSourceId, NormalizedRelativePath};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type GitAcquisitionIntent = super::SourceAcquisitionIntent;
pub type GitAcquisitionPolicy = super::SourceAcquisitionPolicy;
pub type GitLockAction = super::SourceLockAction;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GitAcquisitionPlan {
    pub repository: GitRepositoryUrl,
    pub selector: GitSelector,
    pub subdir: NormalizedRelativePath,
    pub allow_network: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<GitSourceId>,
    pub lock_action: GitLockAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GitPolicyError {
    #[error("frozen mode cannot update mutable Git sources")]
    FrozenUpdate,
    #[error("locked mode cannot update mutable Git sources")]
    LockedUpdate,
    #[error("locked or frozen mode requires an exact Git source in runmat.lock")]
    MissingLock,
    #[error("locked Git source does not match the manifest repository or subdirectory")]
    LockLocatorMismatch,
    #[error("Git provider returned {found:?}; expected locked source {expected:?}")]
    LockedSourceMismatch {
        expected: Box<GitSourceId>,
        found: Box<GitSourceId>,
    },
    #[error("Git provider returned a different repository or subdirectory than requested")]
    AcquiredLocatorMismatch,
}

pub fn plan_git_acquisition(
    repository: GitRepositoryUrl,
    selector: GitSelector,
    subdir: NormalizedRelativePath,
    locked_source: Option<&GitSourceId>,
    intent: GitAcquisitionIntent,
    policy: GitAcquisitionPolicy,
) -> Result<GitAcquisitionPlan, GitPolicyError> {
    if intent == GitAcquisitionIntent::Update {
        if policy.frozen {
            return Err(GitPolicyError::FrozenUpdate);
        }
        if policy.locked {
            return Err(GitPolicyError::LockedUpdate);
        }
    }
    if let Some(locked) = locked_source {
        if locked.repository != repository || locked.subdir != subdir {
            return Err(GitPolicyError::LockLocatorMismatch);
        }
    }
    let use_locked = intent != GitAcquisitionIntent::Update && locked_source.is_some();
    if !use_locked && (policy.locked || policy.frozen) {
        return Err(GitPolicyError::MissingLock);
    }
    if use_locked {
        let expected = locked_source.cloned().expect("checked locked source");
        return Ok(GitAcquisitionPlan {
            repository,
            selector: GitSelector::Rev {
                value: expected.commit.hex.clone(),
            },
            subdir,
            allow_network: !policy.offline && !policy.frozen,
            expected: Some(expected),
            lock_action: GitLockAction::Preserve,
        });
    }
    Ok(GitAcquisitionPlan {
        repository,
        selector,
        subdir,
        allow_network: !policy.offline && !policy.frozen,
        expected: None,
        lock_action: match intent {
            GitAcquisitionIntent::Update => GitLockAction::Replace,
            GitAcquisitionIntent::Execute | GitAcquisitionIntent::Fetch => GitLockAction::Write,
        },
    })
}

pub fn validate_git_acquisition(
    plan: &GitAcquisitionPlan,
    acquired: &GitSourceId,
) -> Result<(), GitPolicyError> {
    if acquired.repository != plan.repository || acquired.subdir != plan.subdir {
        return Err(GitPolicyError::AcquiredLocatorMismatch);
    }
    if let Some(expected) = &plan.expected {
        if acquired != expected {
            return Err(GitPolicyError::LockedSourceMismatch {
                expected: Box::new(expected.clone()),
                found: Box::new(acquired.clone()),
            });
        }
    }
    Ok(())
}
