//! Native persistence for meshing-owned accepted stage receipts. Numerical
//! contracts and session rebinding remain in runmat-meshing-execution.
use crate::{meshing_cache_identity::implementation, FilesystemObjectStore};
use runmat_execution::Digest;
use runmat_execution_artifact::{
    cache::{CacheExport, CacheImport},
    object::ObjectInventoryLimits,
    ArtifactResult, LogicalObject,
};
use runmat_execution_runner::AttemptSuccess;
use runmat_filesystem::{AtomicCreateResult, NativeFsProvider};
use runmat_meshing_core::MeshingStageKind;
use runmat_meshing_execution::{
    meshing_stage_reuse_key, AcceptedMeshingStageReceipt, MeshingStageExecutionError,
    PlannedMeshingStage,
};
use std::path::{Path, PathBuf};

mod control;
mod maintenance;
#[cfg(test)]
mod tests;
const RECEIPT_BYTES: u64 = 8 * 1024 * 1024;
pub(super) struct NativeMeshingStageCache {
    index: PathBuf,
    objects: DurableCacheStore,
    refresh: bool,
    lifecycle: maintenance::Lifecycle,
}
impl NativeMeshingStageCache {
    pub(super) fn open(root: &Path) -> Result<Option<Self>, MeshingStageExecutionError> {
        let mode = std::env::var("RUNMAT_MESHING_CACHE_MODE").unwrap_or_else(|_| "reuse".into());
        match mode.as_str() {
            "off" => return Ok(None),
            "reuse" | "refresh" => {}
            _ => {
                return Err(error(
                    "RUNMAT_MESHING_CACHE_MODE must be reuse, refresh or off",
                ))
            }
        }
        if implementation(MeshingStageKind::CurveMesh).is_none()
            || implementation(MeshingStageKind::Tetrahedralization).is_none()
        {
            tracing::info!(
                event_code = "meshing.cache.disabled",
                "complete implementation inventory unavailable"
            );
            return Ok(None);
        }
        // This lifecycle protocol has exclusive ownership of its format. Older
        // executables cannot mutate it without publishing the invalidation epoch.
        let root = root.join("indexed-v2");
        let index = root.join("receipts");
        let objects = root.join("objects");
        NativeFsProvider
            .create_private_directory_durable(&index)
            .map_err(error)?;
        NativeFsProvider
            .create_private_directory_durable(&objects)
            .map_err(error)?;
        Ok(Some(Self {
            index,
            objects: DurableCacheStore(
                FilesystemObjectStore::open(
                    objects,
                    ObjectInventoryLimits::default().max_object_bytes,
                )
                .map_err(error)?,
            ),
            refresh: mode == "refresh",
            lifecycle: maintenance::Lifecycle::default(),
        }))
    }
    pub(super) fn restore(
        &self,
        stage: &PlannedMeshingStage,
        destination: &mut FilesystemObjectStore,
        session: &crate::NativeProgramSession,
        deadline: Option<u64>,
        started: std::time::Instant,
    ) -> Result<Option<AttemptSuccess>, MeshingStageExecutionError> {
        let control = control::CacheControl::new(session, stage, deadline, started);
        control.check()?;
        if self.refresh || stage.host().workload.stage == MeshingStageKind::Publication {
            return Ok(None);
        }
        let Some(_lease) = runmat_filesystem::ExclusiveFileLock::try_acquire(
            &self.index.parent().unwrap().join("cache.lock"),
        )
        .map_err(error)?
        else {
            tracing::info!(
                event_code = "meshing.cache.busy",
                "cache lifecycle is busy; execute the stage freshly"
            );
            return Ok(None);
        };
        let Some(implementation) = implementation(stage.host().workload.stage) else {
            return Ok(None);
        };
        let key = meshing_stage_reuse_key(stage.host(), implementation).map_err(error)?;
        let path = self.index.join(filename(key));
        let bytes = match NativeFsProvider.read_bounded_sync(&path, RECEIPT_BYTES) {
            Ok(bytes) => bytes,
            Err(failure) if failure.kind() == std::io::ErrorKind::NotFound => {
                tracing::info!(event_code="meshing.cache.miss", stage=?stage.host().workload.stage, key=%key);
                return Ok(None);
            }
            Err(failure) => return Err(error(failure)),
        };
        let receipt: AcceptedMeshingStageReceipt = serde_json::from_slice(&bytes).map_err(error)?;
        let source = control::ControlledSource {
            store: &self.objects,
            control: &control,
        };
        let mut destination = control::ControlledDestination {
            store: destination,
            control: &control,
        };
        let restored = receipt.restore(
            &source,
            &mut destination,
            stage.host(),
            implementation,
            limits(stage),
        );
        control.check()?;
        let success = restored.map_err(error)?;
        tracing::info!(event_code="meshing.cache.hit", stage=?stage.host().workload.stage, key=%key, "verified original stage artifacts reused; observations retain original execution values");
        Ok(Some(success))
    }
    pub(super) fn retain(
        &mut self,
        stage: &PlannedMeshingStage,
        source: &FilesystemObjectStore,
        success: &AttemptSuccess,
        session: &crate::NativeProgramSession,
        deadline: Option<u64>,
        started: std::time::Instant,
    ) -> Result<(), MeshingStageExecutionError> {
        let control = control::CacheControl::new(session, stage, deadline, started);
        control.check()?;
        if stage.host().workload.stage == MeshingStageKind::Publication {
            return Ok(());
        }
        let Some(implementation) = implementation(stage.host().workload.stage) else {
            return Ok(());
        };
        let Some(_lease) = runmat_filesystem::ExclusiveFileLock::try_acquire(
            &self.index.parent().unwrap().join("cache.lock"),
        )
        .map_err(error)?
        else {
            tracing::info!(
                event_code = "meshing.cache.busy",
                "accepted stage remains valid; cache retention skipped under contention"
            );
            return Ok(());
        };
        let reserve = success
            .result_objects()
            .iter()
            .try_fold(RECEIPT_BYTES, |total, reference| {
                total.checked_add(reference.encoded_length)
            })
            .ok_or_else(|| error("cache reservation overflow"))?;
        if reserve > limits(stage).max_total_bytes {
            tracing::info!(
                event_code = "meshing.cache.capacity",
                "accepted stage exceeds cache import memory admission; retention skipped"
            );
            return Ok(());
        }
        let Some(reservation) = self.lifecycle.reserve(
            &self.index,
            self.objects.0.root(),
            reserve,
            success.result_objects().len(),
            &|| control.check(),
        )?
        else {
            tracing::info!(
                event_code = "meshing.cache.capacity",
                "accepted stage exceeds bounded cache capacity; retention skipped"
            );
            return Ok(());
        };
        let key = meshing_stage_reuse_key(stage.host(), implementation).map_err(error)?;
        let controlled_source = control::ControlledSource {
            store: source,
            control: &control,
        };
        let mut controlled_destination = control::ControlledDestination {
            store: &mut self.objects,
            control: &control,
        };
        let retained = AcceptedMeshingStageReceipt::retain(
            &controlled_source,
            &mut controlled_destination,
            stage.host(),
            implementation,
            success,
            limits(stage),
        );
        control.check()?;
        let mut receipt = retained.map_err(error)?;
        let written = receipt.object_digests().collect::<Vec<_>>();
        let bytes = serde_json::to_vec(&receipt).map_err(error)?;
        if bytes.len() as u64 > RECEIPT_BYTES {
            return Err(error("stage receipt exceeds its admitted index bound"));
        }
        // Every object is durably created before this complete receipt becomes visible.
        if let AtomicCreateResult::Existing(existing) = NativeFsProvider
            .atomic_create_private_new(&self.index.join(filename(key)), &bytes, RECEIPT_BYTES)
            .map_err(error)?
        {
            let winner: AcceptedMeshingStageReceipt =
                serde_json::from_slice(&existing).map_err(error)?;
            let accepted = winner
                .restore(
                    &self.objects,
                    &mut source.clone(),
                    stage.host(),
                    implementation,
                    limits(stage),
                )
                .map_err(error)?;
            if accepted.values().map(|(outputs, _)| outputs)
                != success.values().map(|(outputs, _)| outputs)
                || accepted
                    .result_objects()
                    .iter()
                    .filter(|r| r.media_type != runmat_meshing_execution::STAGE_EVIDENCE_MEDIA_TYPE)
                    .collect::<Vec<_>>()
                    != success
                        .result_objects()
                        .iter()
                        .filter(|r| {
                            r.media_type != runmat_meshing_execution::STAGE_EVIDENCE_MEDIA_TYPE
                        })
                        .collect::<Vec<_>>()
            {
                return Err(error(
                    "identical stage dependencies produced different accepted artifacts",
                ));
            }
            receipt = winner;
        }
        self.lifecycle.published(
            reservation,
            &self.index.join(filename(key)),
            self.objects.0.root(),
            &receipt,
            written.into_iter(),
            &|| control.check(),
        )?;
        control.check()?;
        tracing::info!(event_code="meshing.cache.retained", stage=?stage.host().workload.stage, key=%key);
        Ok(())
    }
}
fn limits(stage: &PlannedMeshingStage) -> ObjectInventoryLimits {
    let mut limits = ObjectInventoryLimits::default();
    // Encoded closure, verification/import work and destination IO can coexist.
    // Cache reuse must fit its own envelope; fresh worker acceptance is separate.
    limits.max_total_bytes = limits
        .max_total_bytes
        .min(stage.host().ceilings.maximum_memory_bytes / 3);
    limits.max_total_bytes = limits
        .max_total_bytes
        .min(stage.host().ceilings.maximum_artifact_bytes);
    limits.max_object_bytes = limits.max_object_bytes.min(limits.max_total_bytes);
    limits
}

struct DurableCacheStore(FilesystemObjectStore);
impl CacheImport for DurableCacheStore {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.0.read_verified(digest)
    }
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> runmat_execution_artifact::ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.0
            .read_verified_bounded_with_admission(digest, maximum_bytes, admit)
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.0.read_verified_bounded(digest, maximum_bytes)
    }
}
impl CacheExport for DurableCacheStore {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        object.validate()?;
        let maximum = ObjectInventoryLimits::default().max_object_bytes;
        if object.bytes.len() as u64 > maximum {
            return Err(runmat_execution_artifact::ArtifactError::Limit(
                "cache object exceeds admitted bound".into(),
            ));
        }
        if let AtomicCreateResult::Existing(bytes) = NativeFsProvider.atomic_create_private_new(
            &self.0.root().join(filename(object.descriptor.digest)),
            &object.bytes,
            maximum,
        )? {
            if bytes != object.bytes {
                return Err(runmat_execution_artifact::ArtifactError::Identity(
                    "cache object identity collision".into(),
                ));
            }
        }
        Ok(())
    }
}
fn error(value: impl std::fmt::Display) -> MeshingStageExecutionError {
    MeshingStageExecutionError::new(value.to_string())
}

fn filename(digest: Digest) -> String {
    digest.to_string().trim_start_matches("sha256:").to_owned()
}
