//! Local-worker adapter for meshing-owned subtask identities and accepted payloads.
use crate::meshing_blob_cache::{BoundedMeshingBlobCache, CacheError, MAXIMUM_ENTRY_BYTES};
use runmat_geometry_core::GeometryEvaluationControl;
use runmat_meshing_core::{MeshingStageKind, StableDigest};
use runmat_meshing_execution::{
    MeshingSubtaskCacheEntry, MeshingSubtaskCacheError, MeshingSubtaskCacheStore,
};
use std::{path::PathBuf, sync::Arc};

#[derive(Debug)]
struct NativeMeshingSubtaskCache {
    backend: BoundedMeshingBlobCache,
    implementation: StableDigest,
    volume_implementation: Option<StableDigest>,
    refresh: bool,
}

pub(crate) fn native_subtask_cache() -> Option<Arc<dyn MeshingSubtaskCacheStore>> {
    let mode = std::env::var("RUNMAT_MESHING_CACHE_MODE").unwrap_or_else(|_| "reuse".into());
    if mode == "off" {
        return None;
    }
    if mode != "reuse" && mode != "refresh" {
        return None;
    }
    // The execution host grants this directory explicitly. Direct/remote callers
    // without that capability compute normally and do not write a user-global cache.
    let root = PathBuf::from(std::env::var_os(
        crate::process_provider::NATIVE_MESHING_SUBTASK_CACHE_ROOT_ENV,
    )?);
    if !root.is_absolute() {
        return None;
    }
    let implementation =
        crate::meshing_cache_identity::implementation(MeshingStageKind::SurfaceMesh)?;
    Some(Arc::new(NativeMeshingSubtaskCache {
        backend: BoundedMeshingBlobCache::new(root.join("cache.sqlite")),
        implementation: StableDigest::from_bytes(*implementation.bytes()),
        volume_implementation: crate::meshing_cache_identity::implementation(
            MeshingStageKind::Tetrahedralization,
        )
        .map(|identity| StableDigest::from_bytes(*identity.bytes())),
        refresh: mode == "refresh",
    }))
}

impl MeshingSubtaskCacheStore for NativeMeshingSubtaskCache {
    fn volume_implementation_identity(&self) -> Option<StableDigest> {
        self.volume_implementation
    }
    fn implementation_identity(&self) -> StableDigest {
        self.implementation
    }
    fn maximum_entry_bytes(&self) -> u64 {
        MAXIMUM_ENTRY_BYTES
    }
    fn admit(
        &self,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<bool, MeshingSubtaskCacheError> {
        self.backend.admit(control).map_err(map_error)
    }
    fn read<'a>(
        &self,
        key: StableDigest,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Result<Option<MeshingSubtaskCacheEntry<'a>>, MeshingSubtaskCacheError> {
        if self.refresh {
            return Ok(None);
        }
        self.backend
            .read(key.bytes(), control)
            .map(|entry| {
                entry.map(|entry| MeshingSubtaskCacheEntry {
                    bytes: entry.bytes,
                    storage: entry.storage,
                    materialization_resident_bytes: entry.materialization_resident_bytes,
                })
            })
            .map_err(map_error)
    }
    fn write(
        &self,
        key: StableDigest,
        bytes: &[u8],
        workspace: u64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<bool, MeshingSubtaskCacheError> {
        self.backend
            .write(key.bytes(), bytes, workspace, control)
            .map_err(map_error)
    }
}

fn map_error(error: CacheError) -> MeshingSubtaskCacheError {
    match error {
        CacheError::Control(error) => MeshingSubtaskCacheError::Control(error),
        CacheError::Storage(reason) => MeshingSubtaskCacheError::Storage(reason),
    }
}
