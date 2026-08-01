use crate::filesystem::CacheLayout;
use crate::git::NativeGitClient;
use crate::materialize::materialize_tree;
use crate::server::ServerSnapshotTransport;
use crate::{NativeCacheError, NativeCacheLease, SqliteCacheBackend};
use futures::FutureExt;
use runmat_package::{
    GitAcquisitionPlan, GitPackageMount, PackageSourceProvider, ServerProjectAcquisitionPlan,
    ServerProjectPackageMount,
};
use runmat_package_cache::{
    cache_git_snapshot, cache_server_project_snapshot, load_git_snapshot,
    load_server_project_snapshot, ArchiveLimits, CacheBackend, CacheError, CommitOutcome,
    GitSnapshot, ServerProjectSnapshot,
};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct NativePackageSourceProvider {
    client: NativeGitClient,
    backend: Arc<SqliteCacheBackend>,
    layout: CacheLayout,
    leases: Mutex<Vec<NativeCacheLease>>,
    server: Option<Arc<dyn ServerSnapshotTransport>>,
}

impl std::fmt::Debug for NativePackageSourceProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativePackageSourceProvider")
            .field("client", &self.client)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

impl NativePackageSourceProvider {
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
            server: None,
        }
    }

    pub fn with_server_transport(mut self, server: Arc<dyn ServerSnapshotTransport>) -> Self {
        self.server = Some(server);
        self
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

    async fn acquire_server_snapshot(
        &self,
        plan: &ServerProjectAcquisitionPlan,
    ) -> Result<(ServerProjectSnapshot, NativeCacheLease), String> {
        if let Some(expected) = &plan.expected {
            match NativeCacheLease::acquire(
                self.backend.clone(),
                [expected.tree_digest.clone()].into_iter().collect(),
            )
            .await
            {
                Ok(Some(lease)) => {
                    match load_server_project_snapshot(&self.backend, expected.clone()).await {
                        Ok(snapshot) => return Ok((snapshot, lease)),
                        Err(CacheError::Miss(_)) => drop(lease),
                        Err(error) => return Err(error.to_string()),
                    }
                }
                Ok(None) => unreachable!("expected Server tree is a lease root"),
                Err(NativeCacheError::Cache(CacheError::Miss(_))) => {}
                Err(error) => return Err(error.to_string()),
            }
        }
        if !plan.allow_network {
            return Err("Server snapshot is not available in the offline cache".to_string());
        }
        let inventory = self
            .server
            .as_ref()
            .ok_or_else(|| "Server project snapshot acquisition is not configured".to_string())?
            .fetch(plan)
            .await?;
        let snapshot = inventory
            .into_snapshot(&plan.service, ArchiveLimits::default())
            .map_err(|error| error.to_string())?;
        runmat_package::validate_server_project_acquisition(plan, &snapshot.source)
            .map_err(|error| error.to_string())?;
        loop {
            let current = self
                .backend
                .snapshot()
                .await
                .map_err(|error| error.to_string())?;
            let transaction =
                cache_server_project_snapshot(current.revision, current.state, &snapshot, now_ms())
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
                    .expect("Server snapshot tree is a lease root");
                    return Ok((snapshot, lease));
                }
                CommitOutcome::Conflict { .. } => continue,
            }
        }
    }
}

impl PackageSourceProvider for NativePackageSourceProvider {
    fn acquire_git<'a>(
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

    fn acquire_server_project<'a>(
        &'a self,
        plan: &'a ServerProjectAcquisitionPlan,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ServerProjectPackageMount, String>> + 'a>,
    > {
        async move {
            let (snapshot, lease) = self.acquire_server_snapshot(plan).await?;
            let root = materialize_tree(&self.backend, &self.layout, &snapshot.tree)
                .await
                .map_err(|error| error.to_string())?;
            self.leases
                .lock()
                .map_err(|_| "native package lease lock was poisoned".to_string())?
                .push(lease);
            Ok(ServerProjectPackageMount {
                source: snapshot.source,
                root,
            })
        }
        .boxed_local()
    }
}

pub type NativeGitPackageProvider = NativePackageSourceProvider;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
