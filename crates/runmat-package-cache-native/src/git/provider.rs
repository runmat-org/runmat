use super::NativeGitClient;
use crate::filesystem::CacheLayout;
use crate::materialize::materialize_tree;
use crate::{NativeCacheError, NativeCacheLease, SqliteCacheBackend};
use futures::FutureExt;
use runmat_package::{GitAcquisitionPlan, GitPackageMount, GitPackageProvider};
use runmat_package_cache::{
    cache_git_snapshot, load_git_snapshot, CacheBackend, CacheError, CommitOutcome, GitSnapshot,
};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct NativeGitPackageProvider {
    client: NativeGitClient,
    backend: Arc<SqliteCacheBackend>,
    layout: CacheLayout,
    leases: Mutex<Vec<NativeCacheLease>>,
}

impl std::fmt::Debug for NativeGitPackageProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeGitPackageProvider")
            .field("client", &self.client)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

impl NativeGitPackageProvider {
    pub fn new(
        client: NativeGitClient,
        backend: Arc<SqliteCacheBackend>,
        layout: CacheLayout,
    ) -> Self {
        Self {
            client,
            backend,
            layout,
            leases: Mutex::new(Vec::new()),
        }
    }

    async fn acquire_snapshot(
        &self,
        plan: &GitAcquisitionPlan,
    ) -> Result<(GitSnapshot, NativeCacheLease), String> {
        if let Some(expected) = &plan.expected {
            match NativeCacheLease::acquire(
                self.backend.clone(),
                [expected.tree_digest.clone()].into_iter().collect(),
            )
            .await
            {
                Ok(Some(lease)) => match load_git_snapshot(&self.backend, expected.clone()).await {
                    Ok(snapshot) => return Ok((snapshot, lease)),
                    Err(CacheError::Miss(_)) => drop(lease),
                    Err(error) => return Err(error.to_string()),
                },
                Ok(None) => unreachable!("expected Git tree is a lease root"),
                Err(NativeCacheError::Cache(CacheError::Miss(_))) => {}
                Err(error) => return Err(error.to_string()),
            }
        }
        let snapshot = self
            .client
            .acquire_plan(plan)
            .map_err(|error| error.to_string())?;
        loop {
            let current = self
                .backend
                .snapshot()
                .await
                .map_err(|error| error.to_string())?;
            let transaction =
                cache_git_snapshot(current.revision, current.state, &snapshot, now_ms())
                    .map_err(|error| error.to_string())?;
            match self
                .backend
                .commit(transaction)
                .await
                .map_err(|error| error.to_string())?
            {
                CommitOutcome::Committed(_) => {
                    let lease = NativeCacheLease::acquire(
                        self.backend.clone(),
                        [snapshot.tree.digest.clone()].into_iter().collect(),
                    )
                    .await
                    .map_err(|error| error.to_string())?
                    .expect("Git snapshot tree is a lease root");
                    return Ok((snapshot, lease));
                }
                CommitOutcome::Conflict { .. } => continue,
            }
        }
    }
}

impl GitPackageProvider for NativeGitPackageProvider {
    fn acquire<'a>(
        &'a self,
        plan: &'a GitAcquisitionPlan,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<GitPackageMount, String>> + 'a>>
    {
        async move {
            let (snapshot, lease) = self.acquire_snapshot(plan).await?;
            let root = materialize_tree(&self.backend, &self.layout, &snapshot.tree)
                .await
                .map_err(|error| error.to_string())?;
            self.leases
                .lock()
                .map_err(|_| "native package lease lock was poisoned".to_string())?
                .push(lease);
            Ok(GitPackageMount {
                source: snapshot.source,
                root,
            })
        }
        .boxed_local()
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
