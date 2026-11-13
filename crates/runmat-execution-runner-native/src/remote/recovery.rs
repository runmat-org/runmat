use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use runmat_analysis_execution::StructuralTransientRecoveryClosure;
use runmat_execution::identity::ValueId;
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution_artifact::encryption::{EncryptionPurpose, RunKeyMaterial};
use runmat_execution_artifact::{
    ArtifactAccess, LogicalObject, ObjectNamespace, ProgramExecutionObject,
    ProgramExecutionObjectClosure, ProgramExecutionObjectLocation, ProgramExecutionRequest,
    PROGRAM_EXECUTION_OBJECT_CLOSURE_MEDIA_TYPE, PROGRAM_EXECUTION_OBJECT_CLOSURE_VALUE_SCHEMA,
};
use runmat_execution_transport_native::control::{
    ClockDriverHeartbeatRequestV1, ClockObservationReleaseManifestV1, DriverArtifactKind,
    DriverAuthority, DriverControlPlane,
};
use tokio::sync::Mutex;

use super::checkpoint::{RemoteDriverCheckpoint, RemoteDriverCheckpointPhase};
use super::crypto::open_object;
use super::RemotePoolDriver;
use crate::{NativeExecutionError, NativeExecutionResult, StructuralTransientRecoveryPublisher};

#[derive(Clone)]
pub(super) struct RestoredObjectClosure {
    pub(super) root: ValueRef,
    pub(super) objects: Vec<ProgramExecutionObject>,
    pub(super) locations: Vec<ProgramExecutionObjectLocation>,
    pub(super) catalog: ProgramExecutionObjectLocation,
}

struct PublicationState {
    sequence: u64,
    checkpoint_write_uncertain: bool,
    physical_catalog: Option<ProgramExecutionObjectLocation>,
    pool_catalog: Option<ProgramExecutionObjectLocation>,
    physical_locations: BTreeMap<ValueId, ProgramExecutionObjectLocation>,
    pool_locations: BTreeMap<ValueId, ProgramExecutionObjectLocation>,
    last_checkpoint: Option<(u64, String, String)>,
}

impl PublicationState {
    fn ensure_writable(&self) -> NativeExecutionResult<()> {
        if self.checkpoint_write_uncertain {
            return Err(protocol(
                "remote checkpoint acknowledgement is uncertain; replace the driver before publishing again",
            ));
        }
        Ok(())
    }
}

/// One transaction owner for periodic scheduler and physical solver publication.
#[derive(Clone)]
pub(super) struct RemoteRecoveryPublisher<'a> {
    control: &'a dyn DriverControlPlane,
    run_key: &'a RunKeyMaterial,
    authority: &'a DriverAuthority,
    project_revision: String,
    request: ProgramExecutionRequest,
    pool: Option<Arc<RemotePoolDriver>>,
    state: Arc<Mutex<PublicationState>>,
}

impl<'a> RemoteRecoveryPublisher<'a> {
    pub(super) fn new(
        control: &'a dyn DriverControlPlane,
        run_key: &'a RunKeyMaterial,
        authority: &'a DriverAuthority,
        project_revision: &str,
        request: &ProgramExecutionRequest,
        checkpoint_sequence: u64,
        restored: Option<&RestoredObjectClosure>,
    ) -> NativeExecutionResult<Self> {
        if !matches!(
            super::checkpoint::execution_start_phase(request),
            RemoteDriverCheckpointPhase::ReplayableExecutionStarted
        ) {
            return Err(protocol(
                "non-replayable execution cannot publish a resume-safe pool checkpoint",
            ));
        }
        let physical_locations: BTreeMap<_, _> = restored
            .into_iter()
            .flat_map(|closure| closure.locations.iter().cloned())
            .map(|location| (location.reference.id, location))
            .collect();
        if physical_locations
            .values()
            .any(|location| location.reference.authorization_scope != authority.run_id)
        {
            return Err(protocol(
                "restored recovery locations are outside the driver run authority",
            ));
        }
        Ok(Self {
            control,
            run_key,
            authority,
            project_revision: project_revision.into(),
            request: request.clone(),
            pool: None,
            state: Arc::new(Mutex::new(PublicationState {
                sequence: checkpoint_sequence,
                checkpoint_write_uncertain: false,
                physical_catalog: restored.map(|closure| closure.catalog.clone()),
                pool_catalog: None,
                physical_locations,
                pool_locations: BTreeMap::new(),
                last_checkpoint: None,
            })),
        })
    }

    /// Bind once before cloning this publisher into execution and periodic-save owners.
    pub(super) fn bind_pool(
        &mut self,
        pool: Arc<RemotePoolDriver>,
        catalog: Option<ProgramExecutionObjectLocation>,
        locations: BTreeMap<ValueId, ProgramExecutionObjectLocation>,
    ) -> NativeExecutionResult<()> {
        if self.pool.is_some() || Arc::strong_count(&self.state) != 1 {
            return Err(protocol(
                "recovery pool must be bound before publisher sharing",
            ));
        }
        let mut state = self.state.try_lock().map_err(protocol)?;
        if locations
            .values()
            .any(|location| location.reference.authorization_scope != self.authority.run_id)
        {
            return Err(protocol(
                "restored pool locations are outside the driver run authority",
            ));
        }
        state.pool_catalog = catalog;
        state.pool_locations = locations;
        self.pool = Some(pool);
        Ok(())
    }

    pub(super) async fn publish_objects(
        &mut self,
        root: ValueRef,
        objects: Vec<ProgramExecutionObject>,
    ) -> NativeExecutionResult<()> {
        let access = ArtifactAccess {
            authorization_scope: self.authority.run_id.clone(),
            encryption_context: root.encryption_context,
        };
        access.validate().map_err(protocol)?;
        if root.authorization_scope != self.authority.run_id
            || root.kind != ValueRefKind::ResultObject
        {
            return Err(protocol(
                "physical recovery root is outside the driver result authority",
            ));
        }
        let mut state = self.state.lock().await;
        state.ensure_writable()?;
        let mut locations = Vec::with_capacity(objects.len());
        for object in objects {
            if object.reference.encryption_context != access.encryption_context {
                return Err(protocol(
                    "physical recovery object has a different encryption context",
                ));
            }
            let location = store_object(
                self.control,
                self.run_key,
                self.authority,
                &object.reference,
                &object.bytes,
                state.physical_locations.get(&object.reference.id),
            )
            .await?;
            locations.push(location);
        }
        let catalog = store_catalog(
            self.control,
            self.run_key,
            self.authority,
            access,
            root,
            locations.clone(),
        )
        .await?;
        self.commit(&mut state, Some((catalog, locations)), 0)
            .await
            .map(|_| ())
    }

    pub(super) async fn publish_pool(&self) -> NativeExecutionResult<u64> {
        self.publish_pool_through(0).await
    }

    /// The acknowledged post-end checkpoint is the sole release authority.
    /// A lost manifest ACK leaves the bounded signed records in place; the
    /// replacement driver may retry against the latest durable checkpoint.
    pub(super) async fn publish_closed_wave(&self) -> NativeExecutionResult<()> {
        let pool = self
            .pool
            .as_ref()
            .ok_or_else(|| protocol("closed wave has no bound pool"))?;
        let entries = pool.closed_clock_release_entries()?;
        if entries.is_empty() {
            return Err(protocol("closed wave has no signed node clock end"));
        }
        let mut state = self.state.lock().await;
        self.commit(&mut state, None, 0).await?;
        let (sequence, artifact_id, digest) = state
            .last_checkpoint
            .clone()
            .ok_or_else(|| protocol("post-end checkpoint has no acknowledged artifact"))?;
        let response = self
            .control
            .clock_heartbeat(
                self.authority,
                30,
                ClockDriverHeartbeatRequestV1 {
                    release_manifest: Some(ClockObservationReleaseManifestV1 {
                        checkpoint_sequence: sequence,
                        checkpoint_artifact_id: artifact_id,
                        checkpoint_ciphertext_digest: digest,
                        driver_fence: self.authority.fencing_token,
                        entries: entries.clone(),
                    }),
                    ..Default::default()
                },
            )
            .await
            .map_err(protocol)?;
        if !response.release_accepted {
            return Err(protocol(
                "Server did not acknowledge the post-end clock release manifest",
            ));
        }
        pool.prune_released_clock_entries(&entries)?;
        Ok(())
    }

    pub(super) async fn publish_pool_through(
        &self,
        required_event_watermark: u64,
    ) -> NativeExecutionResult<u64> {
        if self.pool.is_none() {
            return Err(protocol("periodic recovery has no bound pool"));
        }
        let mut state = self.state.lock().await;
        self.commit(&mut state, None, required_event_watermark)
            .await
    }

    async fn commit(
        &self,
        state: &mut PublicationState,
        physical: Option<(
            ProgramExecutionObjectLocation,
            Vec<ProgramExecutionObjectLocation>,
        )>,
        required_event_watermark: u64,
    ) -> NativeExecutionResult<u64> {
        state.ensure_writable()?;
        let sequence = state
            .sequence
            .checked_add(1)
            .ok_or_else(|| protocol("remote driver checkpoint sequence overflow"))?;
        let pool = match &self.pool {
            Some(pool) => Some(
                super::pool_checkpoint::publish(
                    self.control,
                    self.run_key,
                    self.authority,
                    pool,
                    sequence,
                    &state.pool_locations,
                    required_event_watermark,
                )
                .await?,
            ),
            None => None,
        };
        let physical_catalog = physical
            .as_ref()
            .map(|(catalog, _)| catalog.clone())
            .or_else(|| state.physical_catalog.clone());
        let pool_catalog = pool
            .as_ref()
            .map(|published| published.catalog.clone())
            .or_else(|| state.pool_catalog.clone());
        let checkpoint = RemoteDriverCheckpoint::new(
            sequence,
            self.authority,
            &self.project_revision,
            &self.request,
            RemoteDriverCheckpointPhase::ReplayableExecutionStarted,
            physical_catalog.clone(),
        )?
        .with_pool_catalog(pool_catalog.clone())?;
        // Set this before the await: a timeout can drop this future while the
        // control plane is still persisting the checkpoint. No clone of this
        // publisher may reuse the sequence without a replacement bootstrap.
        state.checkpoint_write_uncertain = true;
        let stored = super::driver::store_driver_checkpoint_with_identity(
            self.control,
            self.authority,
            self.run_key,
            &checkpoint,
        )
        .await?;
        state.checkpoint_write_uncertain = false;
        state.last_checkpoint = Some((sequence, stored.artifact_id, stored.ciphertext_digest));
        if let (Some(driver), Some(published)) = (&self.pool, &pool) {
            // Only the server's acknowledged latest checkpoint can reclaim
            // worker-sandbox accounting. A lost store ACK never enters here.
            driver
                .prune_durable_attempt_journals(sequence, &published.checkpointed_attempts)
                .await;
        }
        // Neither root nor its reuse cache advances on upload failure or lost acknowledgment.
        state.sequence = sequence;
        state.physical_catalog = physical_catalog;
        state.pool_catalog = pool_catalog;
        if let Some((_, locations)) = physical {
            state.physical_locations = locations
                .into_iter()
                .map(|location| (location.reference.id, location))
                .collect();
        }
        let event_watermark = if let Some(pool) = pool {
            state.pool_locations = pool.locations;
            pool.event_watermark
        } else {
            0
        };
        Ok(event_watermark)
    }

    #[cfg(test)]
    pub(super) fn checkpoint_sequence(&self) -> u64 {
        self.state
            .try_lock()
            .expect("publication transaction is idle")
            .sequence
    }
}

#[async_trait]
impl StructuralTransientRecoveryPublisher for RemoteRecoveryPublisher<'_> {
    async fn publish(
        &mut self,
        closure: StructuralTransientRecoveryClosure,
    ) -> NativeExecutionResult<()> {
        self.publish_objects(closure.root, closure.objects).await
    }
}

/// Preserve typed encryption purpose even when a valid alias shares identical bytes.
pub(super) async fn store_object(
    control: &dyn DriverControlPlane,
    key: &RunKeyMaterial,
    authority: &DriverAuthority,
    reference: &ValueRef,
    bytes: &[u8],
    retained: Option<&ProgramExecutionObjectLocation>,
) -> NativeExecutionResult<ProgramExecutionObjectLocation> {
    let object = ProgramExecutionObject {
        reference: reference.clone(),
        bytes: bytes.to_vec(),
    };
    object.validate().map_err(protocol)?;
    if reference.authorization_scope != authority.run_id {
        return Err(protocol(
            "recovery object is outside the driver run authority",
        ));
    }
    let purpose = object_purpose(reference)?;
    if let Some(location) = retained {
        if location.reference == *reference {
            return Ok(location.clone());
        }
        if !super::object_transfer::references_share_content_identity(
            &location.reference,
            reference,
        ) {
            return Err(protocol("recovery object id changed immutable content"));
        }
    }
    Ok(ProgramExecutionObjectLocation {
        reference: reference.clone(),
        artifact_id: super::driver::store_encrypted(
            control,
            authority,
            key,
            DriverArtifactKind::Result,
            purpose,
            bytes,
            None,
        )
        .await?,
    })
}

pub(super) async fn store_catalog(
    control: &dyn DriverControlPlane,
    key: &RunKeyMaterial,
    authority: &DriverAuthority,
    access: ArtifactAccess,
    root: ValueRef,
    locations: Vec<ProgramExecutionObjectLocation>,
) -> NativeExecutionResult<ProgramExecutionObjectLocation> {
    let catalog = ProgramExecutionObjectClosure::new(root, locations).map_err(protocol)?;
    let object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        "execution/recovery-closure-catalog",
        PROGRAM_EXECUTION_OBJECT_CLOSURE_MEDIA_TYPE,
        catalog.canonical_bytes().map_err(protocol)?,
    )
    .map_err(protocol)?;
    let reference = access
        .reference(
            &object,
            ValueRefKind::ResultObject,
            PROGRAM_EXECUTION_OBJECT_CLOSURE_VALUE_SCHEMA,
        )
        .map_err(protocol)?;
    store_object(control, key, authority, &reference, &object.bytes, None).await
}

pub(super) async fn read_object(
    control: &dyn DriverControlPlane,
    key: &RunKeyMaterial,
    authority: &DriverAuthority,
    location: &ProgramExecutionObjectLocation,
) -> NativeExecutionResult<ProgramExecutionObject> {
    location.validate().map_err(protocol)?;
    if location.reference.authorization_scope != authority.run_id {
        return Err(protocol("recovery object is outside the run authority"));
    }
    let purpose = object_purpose(&location.reference)?;
    let artifact = control
        .execution_artifact(authority, &location.artifact_id)
        .await
        .map_err(protocol)?;
    if artifact.artifact_id != location.artifact_id || artifact.kind != "result" {
        return Err(protocol(
            "Server resolved the wrong recovery object artifact",
        ));
    }
    let object = ProgramExecutionObject {
        reference: location.reference.clone(),
        bytes: open_object(
            key,
            &control.download(&artifact).await.map_err(protocol)?,
            &authority.run_id,
            purpose,
        )?,
    };
    object.validate().map_err(protocol)?;
    Ok(object)
}

pub(super) async fn read_catalog(
    control: &dyn DriverControlPlane,
    key: &RunKeyMaterial,
    authority: &DriverAuthority,
    location: &ProgramExecutionObjectLocation,
) -> NativeExecutionResult<ProgramExecutionObjectClosure> {
    if location.reference.kind != ValueRefKind::ResultObject
        || location.reference.media_type != PROGRAM_EXECUTION_OBJECT_CLOSURE_MEDIA_TYPE
        || location.reference.value_schema != PROGRAM_EXECUTION_OBJECT_CLOSURE_VALUE_SCHEMA
    {
        return Err(protocol("recovery catalog has the wrong type"));
    }
    let object = read_object(control, key, authority, location).await?;
    ProgramExecutionObjectClosure::from_canonical_bytes(&object.bytes).map_err(protocol)
}

pub(super) async fn restore(
    control: &dyn DriverControlPlane,
    run_key: &RunKeyMaterial,
    authority: &DriverAuthority,
    catalog_location: &ProgramExecutionObjectLocation,
) -> NativeExecutionResult<RestoredObjectClosure> {
    let closure = read_catalog(control, run_key, authority, catalog_location).await?;
    let mut objects = Vec::with_capacity(closure.objects.len());
    for location in &closure.objects {
        objects.push(read_object(control, run_key, authority, location).await?);
    }
    Ok(RestoredObjectClosure {
        root: closure.root,
        objects,
        locations: closure.objects,
        catalog: catalog_location.clone(),
    })
}

fn object_purpose(reference: &ValueRef) -> NativeExecutionResult<EncryptionPurpose> {
    match reference.kind {
        ValueRefKind::DriverObject => Ok(EncryptionPurpose::Input),
        ValueRefKind::ResultObject => Ok(EncryptionPurpose::Result),
        _ => Err(protocol(
            "recovery object kind is not portable input or result",
        )),
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
