use crate::filesystem::{make_tree_removable, CacheLayout};
use crate::git::NativeGitClient;
use crate::materialize::materialize_tree;
use crate::registry::{PrivateArtifactDecryptor, RegistryTransport};
use crate::server::ServerSnapshotTransport;
use crate::{NativeCacheError, NativeCacheLease, SqliteCacheBackend};
use futures::FutureExt;
use runmat_package::{
    GitAcquisitionPlan, GitPackageMount, PackageSourceProvider, RegistryAcquisitionPlan,
    RegistryCandidatePlan, RegistryCandidateRecord, RegistryPackageMount,
    ServerProjectAcquisitionPlan, ServerProjectPackageMount,
};
use runmat_package_cache::backend::conformance::MemoryBackend;
use runmat_package_cache::{
    cache_git_snapshot, cache_registry_snapshot, cache_server_project_snapshot, load_git_snapshot,
    load_registry_snapshot, load_server_project_snapshot, ArchiveLimits, CacheBackend, CacheError,
    CommitOutcome, GitSnapshot, RegistryArtifactInventory, RegistrySnapshot, ServerProjectSnapshot,
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
    registry: Option<Arc<dyn RegistryTransport>>,
    private_artifact_decryptor: Option<Arc<dyn PrivateArtifactDecryptor>>,
    ephemeral_private_mounts: Mutex<Vec<EphemeralPrivateMount>>,
}

struct EphemeralPrivateMount {
    directory: Option<tempfile::TempDir>,
}

impl EphemeralPrivateMount {
    fn new(directory: tempfile::TempDir) -> Self {
        Self {
            directory: Some(directory),
        }
    }
}

impl Drop for EphemeralPrivateMount {
    fn drop(&mut self) {
        if let Some(directory) = self.directory.take() {
            let _ = make_tree_removable(directory.path());
            let _ = directory.close();
        }
    }
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
            registry: None,
            private_artifact_decryptor: None,
            ephemeral_private_mounts: Mutex::new(Vec::new()),
        }
    }

    pub fn with_server_transport(mut self, server: Arc<dyn ServerSnapshotTransport>) -> Self {
        self.server = Some(server);
        self
    }

    pub fn with_registry_transport(mut self, registry: Arc<dyn RegistryTransport>) -> Self {
        self.registry = Some(registry);
        self
    }

    pub fn with_private_artifact_decryptor(
        mut self,
        decryptor: Arc<dyn PrivateArtifactDecryptor>,
    ) -> Self {
        self.private_artifact_decryptor = Some(decryptor);
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

    async fn acquire_registry_snapshot(
        &self,
        plan: &RegistryAcquisitionPlan,
    ) -> Result<
        (
            RegistrySnapshot,
            Option<NativeCacheLease>,
            Option<runmat_package::RegistryReleaseMetadata>,
            bool,
        ),
        String,
    > {
        if let Some(expected) = &plan.expected {
            match NativeCacheLease::acquire(
                self.backend.clone(),
                [expected.tree_digest.clone()].into_iter().collect(),
            )
            .await
            {
                Ok(Some(lease)) => {
                    match load_registry_snapshot(&self.backend, expected.clone()).await {
                        Ok(snapshot) => return Ok((snapshot, Some(lease), None, false)),
                        Err(CacheError::Miss(_)) => drop(lease),
                        Err(error) => return Err(error.to_string()),
                    }
                }
                Ok(None) => unreachable!("expected registry tree is a lease root"),
                Err(NativeCacheError::Cache(CacheError::Miss(_))) => {}
                Err(error) => return Err(error.to_string()),
            }
        }
        if !plan.allow_network {
            return Err("registry release is not available in the offline cache".to_string());
        }
        let transfer = self
            .registry
            .as_ref()
            .ok_or_else(|| "registry package acquisition is not configured".to_string())?
            .fetch(plan)
            .await?;
        transfer
            .metadata
            .verify_supply_chain(&transfer.package_id, &transfer.source)
            .map_err(|error| format!("registry metadata is invalid: {error}"))?;
        let metadata = transfer.metadata;
        let encrypted = metadata.encryption.is_some();
        let snapshot = if let Some(encryption) = metadata.encryption.as_ref() {
            let plaintext = self
                .private_artifact_decryptor
                .as_ref()
                .ok_or_else(|| "private registry package decryption is not configured".to_string())?
                .decrypt(
                    &transfer.source,
                    &transfer.artifact_bytes,
                    encryption,
                    &transfer.key_envelopes,
                )?;
            RegistryArtifactInventory::decode_decrypted_snapshot(
                plaintext.as_slice(),
                transfer.source,
                encryption.plaintext_digest.clone(),
                ArchiveLimits::default(),
            )
        } else {
            if !transfer.key_envelopes.is_empty() {
                return Err(
                    "unencrypted registry release unexpectedly included key envelopes".to_string(),
                );
            }
            RegistryArtifactInventory::decode_snapshot(
                &transfer.artifact_bytes,
                transfer.source,
                ArchiveLimits::default(),
            )
        }
        .map_err(|error| error.to_string())?;
        runmat_package::validate_registry_acquisition(plan, &snapshot.source)
            .map_err(|error| error.to_string())?;
        if encrypted {
            return Ok((snapshot, None, Some(metadata), true));
        }
        loop {
            let current = self
                .backend
                .snapshot()
                .await
                .map_err(|error| error.to_string())?;
            let transaction =
                cache_registry_snapshot(current.revision, current.state, &snapshot, now_ms())
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
                    .expect("registry snapshot tree is a lease root");
                    return Ok((snapshot, Some(lease), Some(metadata), false));
                }
                CommitOutcome::Conflict { .. } => continue,
            }
        }
    }

    async fn materialize_ephemeral_registry_snapshot(
        &self,
        snapshot: &RegistrySnapshot,
    ) -> Result<std::path::PathBuf, String> {
        let backend = MemoryBackend::new();
        let current = backend
            .snapshot()
            .await
            .map_err(|error| error.to_string())?;
        let transaction =
            cache_registry_snapshot(current.revision, current.state, snapshot, now_ms())
                .map_err(|error| error.to_string())?;
        match backend
            .commit(transaction)
            .await
            .map_err(|error| error.to_string())?
        {
            CommitOutcome::Committed(_) => {}
            CommitOutcome::Conflict { .. } => {
                return Err("ephemeral private package cache conflicted".to_string())
            }
        }
        let temp = tempfile::Builder::new()
            .prefix("runmat-private-package-")
            .tempdir()
            .map_err(|error| format!("failed to create private package mount: {error}"))?;
        let layout = CacheLayout::new(temp.path().join("cache"));
        let root = materialize_tree(&backend, &layout, &snapshot.tree)
            .await
            .map_err(|error| error.to_string())?;
        self.ephemeral_private_mounts
            .lock()
            .map_err(|_| "private package mount lock was poisoned".to_string())?
            .push(EphemeralPrivateMount::new(temp));
        Ok(root)
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

    fn acquire_registry<'a>(
        &'a self,
        plan: &'a RegistryAcquisitionPlan,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<RegistryPackageMount, String>> + 'a>,
    > {
        async move {
            let (snapshot, lease, metadata, ephemeral) =
                self.acquire_registry_snapshot(plan).await?;
            let root = if ephemeral {
                self.materialize_ephemeral_registry_snapshot(&snapshot)
                    .await?
            } else {
                materialize_tree(&self.backend, &self.layout, &snapshot.tree)
                    .await
                    .map_err(|error| error.to_string())?
            };
            if let Some(lease) = lease {
                self.leases
                    .lock()
                    .map_err(|_| "native package lease lock was poisoned".to_string())?
                    .push(lease);
            }
            Ok(RegistryPackageMount {
                source: snapshot.source,
                root,
                metadata,
            })
        }
        .boxed_local()
    }

    fn registry_candidates<'a>(
        &'a self,
        plan: &'a RegistryCandidatePlan,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<RegistryCandidateRecord>, String>> + 'a>,
    > {
        async move {
            if !plan.allow_network {
                return Err(
                    "registry candidates are not available without network access".to_string(),
                );
            }
            self.registry
                .as_ref()
                .ok_or_else(|| "registry package acquisition is not configured".to_string())?
                .candidates(plan)
                .await
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
