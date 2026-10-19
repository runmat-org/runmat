//! One application-owned user-path setting, distinct from preference groups.

use super::{RuntimeContext, RuntimeServiceFuture};
use serde::{Deserialize, Serialize};

/// Exact persisted text is preserved; filesystem resolution belongs to the
/// caller's captured provider and working-directory authority.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserPathSetting {
    #[default]
    Default,
    Cleared,
    Custom(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UserPathSnapshot {
    pub setting: UserPathSetting,
    pub revision: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UserPathMutation {
    Set(String),
    Reset,
    Clear,
}

impl UserPathMutation {
    pub(crate) fn into_setting(self) -> Result<UserPathSetting, UserPathServiceError> {
        match self {
            Self::Set(text) if text.is_empty() => Err(UserPathServiceError::EmptyCustomPath),
            Self::Set(text) => Ok(UserPathSetting::Custom(text)),
            Self::Reset => Ok(UserPathSetting::Default),
            Self::Clear => Ok(UserPathSetting::Cleared),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum UserPathServiceError {
    #[error("a custom user path cannot be empty; use Clear instead")]
    EmptyCustomPath,
    #[error("application user-path execution failed: {0}")]
    Execution(#[source] crate::execution::ExecutionServiceError),
    #[error("application user-path storage failed: {0}")]
    Storage(#[source] Box<super::PreferenceServiceError>),
}

impl From<super::PreferenceServiceError> for UserPathServiceError {
    fn from(error: super::PreferenceServiceError) -> Self {
        match error {
            super::PreferenceServiceError::Execution(source) => Self::Execution(source),
            other => Self::Storage(Box::new(other)),
        }
    }
}

/// Hosts install one shared application-settings authority. Reads must be
/// awaited before first source resolution; no background hydration or project
/// VFS fallback is permitted. A committed mutation returns a definitive
/// revision, while an uncertain post-authorization outcome is reported through
/// the typed storage error rather than described as rolled back.
pub trait RuntimeUserPathService {
    fn read(
        &self,
        runtime: RuntimeContext,
    ) -> RuntimeServiceFuture<Result<UserPathSnapshot, UserPathServiceError>>;

    fn commit(
        &self,
        runtime: RuntimeContext,
        mutation: UserPathMutation,
    ) -> RuntimeServiceFuture<Result<UserPathSnapshot, UserPathServiceError>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_default_clear_and_exact_custom_text_are_distinct() {
        assert_eq!(
            UserPathMutation::Reset.into_setting().unwrap(),
            UserPathSetting::Default
        );
        assert_eq!(
            UserPathMutation::Clear.into_setting().unwrap(),
            UserPathSetting::Cleared
        );
        assert_eq!(
            UserPathMutation::Set("relative/MiXeD".into())
                .into_setting()
                .unwrap(),
            UserPathSetting::Custom("relative/MiXeD".into())
        );
        assert!(matches!(
            UserPathMutation::Set(String::new()).into_setting(),
            Err(UserPathServiceError::EmptyCustomPath)
        ));
    }
}
