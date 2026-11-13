//! Resolve a run's dependencies from its captured workspace, using the ordinary
//! host source-acquisition port and its immutable cache mounts.
use crate::{
    decode_lock, GitAcquisitionPlan, GitPackageMount, PackageSourceProvider, ProjectResolveOptions,
    RegistryAcquisitionPlan, RegistryCandidatePlan, RegistryCandidateRecord, RegistryPackageMount,
    ResolvedProject, ServerProjectAcquisitionPlan, ServerProjectPackageMount,
};
use runmat_filesystem::{
    pinned::{InputManifest, PinnedFsProvider},
    FsProvider,
};
use std::{
    cell::RefCell,
    collections::BTreeSet,
    future::Future,
    io,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};

type Acquisition<'a, T> = Pin<Box<dyn Future<Output = Result<T, String>> + 'a>>;

#[derive(Debug, thiserror::Error)]
pub enum CapturedProjectError {
    #[error("captured project input is unavailable: {0}")]
    Input(#[from] io::Error),
    #[error("captured package lock is not UTF-8: {0}")]
    Encoding(#[from] std::str::Utf8Error),
    #[error("captured package lock is invalid: {0}")]
    Lock(#[from] crate::LockError),
    #[error("captured package resolution failed: {0}")]
    Resolve(#[from] crate::ProjectResolveError),
}

pub struct CapturedResolvedProject {
    inputs: InputManifest,
    pub project: ResolvedProject,
    pub filesystem: Arc<PinnedFsProvider>,
}

impl CapturedResolvedProject {
    /// The inventory that produced this graph and filesystem authority.
    pub fn inputs(&self) -> &InputManifest {
        &self.inputs
    }
}

/// The host must retain acquired cache leases while the result is in use. Cache
/// mounts come exclusively from `sources`, never from a caller-supplied path
/// allowlist. No generated lock is written back over captured project inputs.
pub async fn resolve_captured_project_async(
    manifest_path: &Path,
    inputs: InputManifest,
    options: ProjectResolveOptions,
    sources: &dyn PackageSourceProvider,
    workspace: Arc<dyn FsProvider>,
    immutable_cache: Arc<dyn FsProvider>,
) -> Result<CapturedResolvedProject, CapturedProjectError> {
    let filesystem = Arc::new(PinnedFsProvider::new(workspace, inputs.clone())?);
    let lock_path = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("/"))
        .join("runmat.lock");
    let lock = match filesystem.metadata(&lock_path).await {
        Ok(_) => {
            let bytes = filesystem.read(&lock_path).await?;
            Some(decode_lock(std::str::from_utf8(&bytes)?)?)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let acquired = CapturedSources {
        sources,
        filesystem: Arc::clone(&filesystem),
        immutable_cache,
        mounted: RefCell::new(BTreeSet::new()),
    };
    let project = super::project_resolver::resolve_project_with_provider_async(
        manifest_path,
        lock.as_ref(),
        options,
        &acquired,
        filesystem.as_ref(),
    )
    .await;
    Ok(CapturedResolvedProject {
        inputs,
        project: project?,
        filesystem,
    })
}

struct CapturedSources<'a> {
    sources: &'a dyn PackageSourceProvider,
    filesystem: Arc<PinnedFsProvider>,
    immutable_cache: Arc<dyn FsProvider>,
    mounted: RefCell<BTreeSet<PathBuf>>,
}
impl CapturedSources<'_> {
    fn mount(&self, root: &Path) -> Result<(), String> {
        if self.mounted.borrow().contains(root) {
            return Ok(());
        }
        self.filesystem
            .mount_readonly_dependency(root, self.immutable_cache.clone())
            .map_err(|e| e.to_string())?;
        self.mounted.borrow_mut().insert(root.to_path_buf());
        Ok(())
    }
}
impl PackageSourceProvider for CapturedSources<'_> {
    fn acquire_git<'a>(&'a self, plan: &'a GitAcquisitionPlan) -> Acquisition<'a, GitPackageMount> {
        Box::pin(async move {
            let mount = self.sources.acquire_git(plan).await?;
            self.mount(&mount.root)?;
            Ok(mount)
        })
    }
    fn acquire_server_project<'a>(
        &'a self,
        plan: &'a ServerProjectAcquisitionPlan,
    ) -> Acquisition<'a, ServerProjectPackageMount> {
        Box::pin(async move {
            let mount = self.sources.acquire_server_project(plan).await?;
            self.mount(&mount.root)?;
            Ok(mount)
        })
    }
    fn acquire_registry<'a>(
        &'a self,
        plan: &'a RegistryAcquisitionPlan,
    ) -> Acquisition<'a, RegistryPackageMount> {
        Box::pin(async move {
            let mount = self.sources.acquire_registry(plan).await?;
            self.mount(&mount.root)?;
            Ok(mount)
        })
    }
    fn registry_candidates<'a>(
        &'a self,
        plan: &'a RegistryCandidatePlan,
    ) -> Acquisition<'a, Vec<RegistryCandidateRecord>> {
        self.sources.registry_candidates(plan)
    }
}
