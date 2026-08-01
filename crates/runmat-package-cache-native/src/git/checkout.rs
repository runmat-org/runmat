use super::credentials::{GitCredentialProvider, NoGitCredentials};
use super::{fetch, objects, remote};
use crate::concurrency::ProcessLock;
use crate::filesystem::CacheLayout;
use crate::NativeCacheError;
use git2::Oid;
use runmat_package::{GitRepositoryUrl, GitSelector, NormalizedRelativePath};
use runmat_package_cache::{ArchiveLimits, GitSnapshot};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitAcquireRequest {
    pub repository: GitRepositoryUrl,
    pub selector: GitSelector,
    pub subdir: NormalizedRelativePath,
    pub allow_network: bool,
}

pub struct NativeGitClient {
    layout: CacheLayout,
    credentials: Arc<dyn GitCredentialProvider>,
}

impl std::fmt::Debug for NativeGitClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeGitClient")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

impl NativeGitClient {
    pub fn new(layout: CacheLayout) -> Self {
        Self {
            layout,
            credentials: Arc::new(NoGitCredentials),
        }
    }

    pub fn with_credentials(
        layout: CacheLayout,
        credentials: Arc<dyn GitCredentialProvider>,
    ) -> Self {
        Self {
            layout,
            credentials,
        }
    }

    pub fn acquire(&self, request: &GitAcquireRequest) -> Result<GitSnapshot, NativeCacheError> {
        self.layout.create()?;
        let _lock = ProcessLock::acquire(&self.layout.git_repository_lock(&request.repository))?;
        let repository = remote::open_or_initialize(&self.layout, &request.repository)?;
        let commit = fetch::resolve_commit(
            &repository,
            &request.selector,
            request.allow_network,
            self.credentials.as_ref(),
        )?;
        let oid = Oid::from_str(&commit.hex).map_err(remote::git_error)?;
        let inventory = objects::snapshot_tree(&repository, oid, &request.subdir)?;
        debug_assert_eq!(inventory.commit, commit.hex);
        inventory
            .into_snapshot(
                request.repository.as_str(),
                request.subdir.as_str(),
                ArchiveLimits::default(),
            )
            .map_err(NativeCacheError::from)
    }
}
