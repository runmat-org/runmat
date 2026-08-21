use git2::CredentialType;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GitCredential {
    UserPassword {
        username: String,
        password: String,
    },
    SshKey {
        username: String,
        public_key: Option<PathBuf>,
        private_key: PathBuf,
        passphrase: Option<String>,
    },
    SshAgent {
        username: String,
    },
    Default,
}

pub trait GitCredentialProvider: Send + Sync {
    fn credential(
        &self,
        repository: &str,
        username_from_url: Option<&str>,
        allowed: CredentialType,
    ) -> Option<GitCredential>;
}

#[derive(Debug, Default)]
pub struct NoGitCredentials;

impl GitCredentialProvider for NoGitCredentials {
    fn credential(
        &self,
        _repository: &str,
        _username_from_url: Option<&str>,
        _allowed: CredentialType,
    ) -> Option<GitCredential> {
        None
    }
}
