//! Stream encrypted scheduler checkpoint objects through a bounded disk staging store.
use std::collections::BTreeMap;
use std::sync::Arc;

use runmat_execution::identity::ValueId;
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::{CacheExport, CacheImport, FilesystemObjectStore};
use runmat_execution_artifact::encryption::RunKeyMaterial;
use runmat_execution_artifact::{
    ArtifactAccess, LogicalObject, ObjectNamespace, ProgramExecutionObjectLocation,
    MAX_PROGRAM_EXECUTION_OBJECT_BYTES,
};
use runmat_execution_transport_native::control::{
    ClockDriverHeartbeatRequestV1, ClockObservationReleaseManifestV1, DriverAuthority,
    DriverControlPlane, MAX_CLOCK_HEARTBEAT_ITEMS,
};

use super::{RemotePoolCheckpoint, RemotePoolDriver, RemotePoolRestoreAuthority};
use crate::{NativeExecutionError, NativeExecutionResult};

const MEDIA_TYPE: &str = "application/vnd.runmat.remote-pool-checkpoint+json";
const VALUE_SCHEMA: &str = "runmat.remote-pool-checkpoint.v1";

pub(super) struct PublishedPoolCheckpoint {
    pub(super) catalog: ProgramExecutionObjectLocation,
    pub(super) locations: BTreeMap<ValueId, ProgramExecutionObjectLocation>,
    pub(super) event_watermark: u64,
    pub(super) checkpointed_attempts: Vec<(
        runmat_execution::identity::WorkerId,
        super::CheckpointedAttempt,
    )>,
}

pub(super) async fn publish(
    control: &dyn DriverControlPlane,
    key: &RunKeyMaterial,
    authority: &DriverAuthority,
    pool: &RemotePoolDriver,
    generation: u64,
    retained: &BTreeMap<ValueId, ProgramExecutionObjectLocation>,
    required_event_watermark: u64,
) -> NativeExecutionResult<PublishedPoolCheckpoint> {
    if pool.value_scope() != authority.run_id {
        return Err(invalid(
            "pool checkpoint is outside the driver run authority",
        ));
    }
    let checkpoint = pool.capture_checkpoint(generation)?;
    let event_watermark = checkpoint.scheduler_event_watermark();
    let checkpointed_attempts = checkpoint.checkpointed_attempts()?;
    if event_watermark < required_event_watermark {
        return Err(invalid(
            "remote checkpoint did not include its committed scheduler event",
        ));
    }
    let references = checkpoint.object_references()?;
    let root_object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        "execution/remote-pool-checkpoint",
        MEDIA_TYPE,
        checkpoint.encode()?,
    )
    .map_err(invalid)?;
    let access = ArtifactAccess {
        authorization_scope: authority.run_id.clone(),
        encryption_context: Digest::sha256(VALUE_SCHEMA.as_bytes()),
    };
    let root = access
        .reference(&root_object, ValueRefKind::ResultObject, VALUE_SCHEMA)
        .map_err(invalid)?;
    let root_location = super::recovery::store_object(
        control,
        key,
        authority,
        &root,
        &root_object.bytes,
        retained.get(&root.id),
    )
    .await?;
    drop(root_object);
    let mut locations = BTreeMap::from([(root.id, root_location)]);
    for reference in references {
        if locations.contains_key(&reference.id) {
            return Err(invalid(
                "pool checkpoint root collides with an execution object",
            ));
        }
        let bytes = pool.checkpoint_object(&reference)?;
        let location = super::recovery::store_object(
            control,
            key,
            authority,
            &reference,
            &bytes,
            retained.get(&reference.id),
        )
        .await?;
        locations.insert(reference.id, location);
    }
    let catalog = super::recovery::store_catalog(
        control,
        key,
        authority,
        access,
        root,
        locations.values().cloned().collect(),
    )
    .await?;
    Ok(PublishedPoolCheckpoint {
        catalog,
        locations,
        event_watermark,
        checkpointed_attempts,
    })
}

pub(super) struct RestoredPoolCheckpoint {
    pub(super) catalog: ProgramExecutionObjectLocation,
    pub(super) locations: BTreeMap<ValueId, ProgramExecutionObjectLocation>,
    checkpoint: RemotePoolCheckpoint,
    store: FilesystemObjectStore,
    physical_digests: BTreeMap<ValueId, Digest>,
    _directory: tempfile::TempDir,
}

impl RestoredPoolCheckpoint {
    pub(super) fn authoritative_unstarted_inventory(
        &self,
    ) -> NativeExecutionResult<super::pool::AuthoritativeUnstartedInventory> {
        // `restore` is reached only from the pool catalog named by the
        // authenticated latest accepted driver bootstrap checkpoint.
        self.checkpoint.authoritative_unstarted_inventory()
    }

    pub(super) async fn reconcile_closed_clocks(
        &mut self,
        control: &dyn DriverControlPlane,
        authority: &DriverAuthority,
        original_sequence: u64,
        original_artifact_id: &str,
        original_ciphertext_digest: &str,
        original_driver_fence: u64,
    ) -> NativeExecutionResult<()> {
        let entries = self.checkpoint.closed_clock_release_entries()?;
        if entries.is_empty() {
            return Ok(());
        }
        if entries.len() > MAX_CLOCK_HEARTBEAT_ITEMS {
            return Err(invalid(
                "restored closed clock inventory exceeds one bounded release manifest",
            ));
        }
        let prior = control
            .clock_heartbeat(authority, 30, ClockDriverHeartbeatRequestV1::default())
            .await
            .map_err(invalid)?;
        if let Some(manifest) = prior.latest_accepted_release_manifest {
            let mut saved = manifest.entries.clone();
            saved.sort_by_key(|entry| entry.observation_id);
            let mut expected = entries.clone();
            expected.sort_by_key(|entry| entry.observation_id);
            if saved == expected {
                if manifest.checkpoint_sequence > original_sequence
                    || manifest.driver_fence > original_driver_fence
                    || (manifest.checkpoint_sequence == original_sequence
                        && (manifest.driver_fence != original_driver_fence
                            || manifest.checkpoint_artifact_id != original_artifact_id
                            || manifest.checkpoint_ciphertext_digest != original_ciphertext_digest))
                {
                    return Err(invalid(
                        "accepted clock release is outside original checkpoint lineage",
                    ));
                }
                self.checkpoint.prune_released_clocks(&entries)?;
                return Ok(());
            }
            if manifest.entries.iter().any(|saved| {
                entries
                    .iter()
                    .any(|entry| entry.observation_id == saved.observation_id)
            }) {
                return Err(invalid(
                    "accepted clock release partially overlaps restored ownership",
                ));
            }
        }
        let response = control
            .clock_heartbeat(
                authority,
                30,
                ClockDriverHeartbeatRequestV1 {
                    release_manifest: Some(ClockObservationReleaseManifestV1 {
                        checkpoint_sequence: original_sequence,
                        checkpoint_artifact_id: original_artifact_id.into(),
                        checkpoint_ciphertext_digest: original_ciphertext_digest.into(),
                        driver_fence: original_driver_fence,
                        entries: entries.clone(),
                    }),
                    ..Default::default()
                },
            )
            .await
            .map_err(invalid)?;
        if !response.release_accepted {
            return Err(invalid(
                "Server did not reconcile original closed node clock ownership",
            ));
        }
        self.checkpoint.prune_released_clocks(&entries)?;
        Ok(())
    }

    pub(super) fn into_pool(
        self,
        authority: RemotePoolRestoreAuthority,
        bundle: Arc<[u8]>,
    ) -> NativeExecutionResult<Arc<RemotePoolDriver>> {
        let Self {
            checkpoint,
            store,
            physical_digests,
            _directory,
            ..
        } = self;
        let pool =
            RemotePoolDriver::restore_checkpoint(checkpoint, authority, bundle, |reference| {
                let digest = physical_digests
                    .get(&reference.id)
                    .ok_or_else(|| invalid("pool checkpoint object was not staged"))?;
                store
                    .read_verified(*digest)
                    .map_err(invalid)?
                    .map(Arc::<[u8]>::from)
                    .ok_or_else(|| invalid("staged pool checkpoint object is missing"))
            })?;
        // The restored pool now owns verified copies in its own catalogs.
        drop(_directory);
        Ok(pool)
    }
}

pub(super) async fn restore(
    control: &dyn DriverControlPlane,
    key: &RunKeyMaterial,
    authority: &DriverAuthority,
    catalog: &ProgramExecutionObjectLocation,
) -> NativeExecutionResult<RestoredPoolCheckpoint> {
    let closure = super::recovery::read_catalog(control, key, authority, catalog).await?;
    let access = ArtifactAccess {
        authorization_scope: authority.run_id.clone(),
        encryption_context: Digest::sha256(VALUE_SCHEMA.as_bytes()),
    };
    access
        .validate_reference(
            &closure.root,
            ValueRefKind::ResultObject,
            MEDIA_TYPE,
            VALUE_SCHEMA,
        )
        .map_err(invalid)?;
    let locations: BTreeMap<_, _> = closure
        .objects
        .into_iter()
        .map(|location| (location.reference.id, location))
        .collect();
    let root_location = locations
        .get(&closure.root.id)
        .ok_or_else(|| invalid("pool checkpoint catalog omits its root"))?;
    let root_object = super::recovery::read_object(control, key, authority, root_location).await?;
    let checkpoint = RemotePoolCheckpoint::decode(&root_object.bytes)?;
    drop(root_object);
    let references = checkpoint.object_references()?;
    if locations.len() != references.len() + 1 {
        return Err(invalid(
            "pool checkpoint catalog differs from its exact object inventory",
        ));
    }
    for reference in &references {
        let location = locations
            .get(&reference.id)
            .filter(|location| {
                reference.id != closure.root.id && same_object(&location.reference, reference)
            })
            .ok_or_else(|| {
                invalid("pool checkpoint catalog omits or substitutes an admitted object")
            })?;
        if location.reference.authorization_scope != authority.run_id {
            return Err(invalid(
                "pool checkpoint object is outside the run authority",
            ));
        }
    }
    let directory = tempfile::Builder::new()
        .prefix("runmat-pool-checkpoint-")
        .tempdir()
        .map_err(invalid)?;
    let mut store = FilesystemObjectStore::open(
        directory.path().join("objects"),
        MAX_PROGRAM_EXECUTION_OBJECT_BYTES,
    )
    .map_err(invalid)?;
    let mut physical_digests = BTreeMap::new();
    for reference in references {
        let location = locations
            .get(&reference.id)
            .expect("exact inventory validated");
        let object = super::recovery::read_object(control, key, authority, location).await?;
        // Externalized values use a semantic logical digest, not SHA256(encoded bytes).
        // Stage by physical digest and retain the immutable manifest reference separately.
        let physical = LogicalObject::new(
            ObjectNamespace::Checkpoint,
            "execution/checkpoint-staging",
            object.reference.media_type,
            object.bytes,
        )
        .map_err(invalid)?;
        store.write_verified(&physical).map_err(invalid)?;
        physical_digests.insert(reference.id, physical.descriptor.digest);
    }
    Ok(RestoredPoolCheckpoint {
        catalog: catalog.clone(),
        locations,
        checkpoint,
        store,
        physical_digests,
        _directory: directory,
    })
}

fn same_object(left: &ValueRef, right: &ValueRef) -> bool {
    left == right || super::object_transfer::references_share_content_identity(left, right)
}

fn invalid(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
