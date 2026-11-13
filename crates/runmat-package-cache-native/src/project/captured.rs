//! Captured runs reuse the native acquisition and durable cache lifecycle.
use super::{NativeProjectError, NativeProjectResolveRequest, NativeResolvedProject};
use crate::{NativeCacheConfig, NativePackageSourceProvider, SqliteCacheBackend};
use runmat_filesystem::{pinned::InputManifest, FsProvider, NativeFsProvider};
use runmat_package::CapturedResolvedProject;
use std::sync::Arc;

pub struct NativeCapturedProject {
    pub captured: CapturedResolvedProject,
    /// Keep this owner alive until all cells using the capture have finished.
    pub lease: NativeResolvedProject,
}

pub async fn resolve_native_captured_project(
    request: NativeProjectResolveRequest,
    inputs: InputManifest,
    workspace: Arc<dyn FsProvider>,
    backend: Arc<SqliteCacheBackend>,
    cache_config: NativeCacheConfig,
    provider: NativePackageSourceProvider,
) -> Result<NativeCapturedProject, NativeProjectError> {
    let captured = runmat_package::resolve_captured_project_async(
        &request.manifest_path,
        inputs,
        request.options,
        &provider,
        workspace,
        Arc::new(NativeFsProvider),
    )
    .await
    .map_err(|error| NativeProjectError::Resolve(error.to_string()))?;
    let lease = super::service::retain_resolved_project(
        captured.project.clone(),
        backend,
        cache_config,
        provider,
    )
    .await?;
    Ok(NativeCapturedProject { captured, lease })
}
