use std::collections::BTreeMap;

use async_trait::async_trait;
use runmat_analysis_execution::StructuralTransientRecoveryClosure;
use runmat_execution::value::ValueRef;
use runmat_execution_artifact::encryption::{EncryptionPurpose, RunKeyMaterial};
use runmat_execution_artifact::{
    ArtifactAccess, LogicalObject, ObjectNamespace, ProgramExecutionObject,
    ProgramExecutionObjectClosure, ProgramExecutionObjectLocation, ProgramExecutionRequest,
    PROGRAM_EXECUTION_OBJECT_CLOSURE_MEDIA_TYPE, PROGRAM_EXECUTION_OBJECT_CLOSURE_VALUE_SCHEMA,
};
use runmat_execution_transport_native::control::{
    DriverArtifactKind, DriverAuthority, DriverControlPlane,
};

use super::checkpoint::{RemoteDriverCheckpoint, RemoteDriverCheckpointPhase};
use super::crypto::open_object;
use crate::{NativeExecutionError, NativeExecutionResult, StructuralTransientRecoveryPublisher};

#[derive(Clone)]
pub(super) struct RestoredObjectClosure {
    pub(super) root: ValueRef,
    pub(super) objects: Vec<ProgramExecutionObject>,
    pub(super) locations: Vec<ProgramExecutionObjectLocation>,
}

pub(super) struct RemoteRecoveryPublisher<'a> {
    control: &'a dyn DriverControlPlane,
    run_key: &'a RunKeyMaterial,
    authority: &'a DriverAuthority,
    project_revision: String,
    request: ProgramExecutionRequest,
    checkpoint_sequence: u64,
    retained_locations:
        BTreeMap<runmat_execution::identity::ValueId, ProgramExecutionObjectLocation>,
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
        let retained_locations = restored
            .into_iter()
            .flat_map(|closure| closure.locations.iter().cloned())
            .map(|location| (location.reference.id, location))
            .collect::<BTreeMap<_, _>>();
        if retained_locations
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
            checkpoint_sequence,
            retained_locations,
        })
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
            || root.kind != runmat_execution::value::ValueRefKind::ResultObject
        {
            return Err(protocol(
                "transient recovery root is outside the driver result authority",
            ));
        }
        let mut locations = Vec::with_capacity(objects.len());
        let mut next_retained = BTreeMap::new();
        for object in objects {
            object.validate().map_err(protocol)?;
            if object.reference.authorization_scope != self.authority.run_id
                || object.reference.encryption_context != access.encryption_context
                || object.reference.kind != runmat_execution::value::ValueRefKind::ResultObject
            {
                return Err(protocol(
                    "transient recovery object is outside the driver result authority",
                ));
            }
            let location = match self.retained_locations.get(&object.reference.id) {
                Some(location) if location.reference == object.reference => location.clone(),
                Some(_) => {
                    return Err(protocol(
                        "transient recovery object id changed immutable content",
                    ))
                }
                None => ProgramExecutionObjectLocation {
                    reference: object.reference.clone(),
                    artifact_id: super::driver::store_encrypted(
                        self.control,
                        self.authority,
                        self.run_key,
                        DriverArtifactKind::Result,
                        EncryptionPurpose::Result,
                        &object.bytes,
                        None,
                    )
                    .await?,
                },
            };
            next_retained.insert(location.reference.id, location.clone());
            locations.push(location);
        }
        let catalog = ProgramExecutionObjectClosure::new(root, locations).map_err(protocol)?;
        let catalog_object = LogicalObject::new(
            ObjectNamespace::ResultValue,
            "execution/recovery-closure-catalog",
            PROGRAM_EXECUTION_OBJECT_CLOSURE_MEDIA_TYPE,
            catalog.canonical_bytes().map_err(protocol)?,
        )
        .map_err(protocol)?;
        let catalog_reference = access
            .reference(
                &catalog_object,
                runmat_execution::value::ValueRefKind::ResultObject,
                PROGRAM_EXECUTION_OBJECT_CLOSURE_VALUE_SCHEMA,
            )
            .map_err(protocol)?;
        let catalog_location = ProgramExecutionObjectLocation {
            reference: catalog_reference,
            artifact_id: super::driver::store_encrypted(
                self.control,
                self.authority,
                self.run_key,
                DriverArtifactKind::Result,
                EncryptionPurpose::Result,
                &catalog_object.bytes,
                None,
            )
            .await?,
        };
        let next_sequence = self
            .checkpoint_sequence
            .checked_add(1)
            .ok_or_else(|| protocol("remote driver checkpoint sequence overflow"))?;
        let checkpoint = RemoteDriverCheckpoint::new(
            next_sequence,
            self.authority,
            &self.project_revision,
            &self.request,
            RemoteDriverCheckpointPhase::ReplayableExecutionStarted,
            Some(catalog_location),
        )?;
        super::driver::store_driver_checkpoint(
            self.control,
            self.authority,
            self.run_key,
            &checkpoint,
        )
        .await?;
        self.checkpoint_sequence = next_sequence;
        self.retained_locations = next_retained;
        Ok(())
    }

    #[cfg(test)]
    pub(super) const fn checkpoint_sequence(&self) -> u64 {
        self.checkpoint_sequence
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

pub(super) async fn restore(
    control: &dyn DriverControlPlane,
    run_key: &RunKeyMaterial,
    authority: &DriverAuthority,
    catalog_location: &ProgramExecutionObjectLocation,
) -> NativeExecutionResult<RestoredObjectClosure> {
    let catalog_download = control
        .execution_artifact(authority, &catalog_location.artifact_id)
        .await
        .map_err(protocol)?;
    if catalog_download.artifact_id != catalog_location.artifact_id
        || catalog_download.kind != "result"
    {
        return Err(protocol(
            "Server resolved the wrong recovery catalog artifact",
        ));
    }
    let catalog_object = ProgramExecutionObject {
        reference: catalog_location.reference.clone(),
        bytes: open_object(
            run_key,
            &control
                .download(&catalog_download)
                .await
                .map_err(protocol)?,
            &authority.run_id,
            EncryptionPurpose::Result,
        )?,
    };
    catalog_object.validate().map_err(protocol)?;
    let closure = ProgramExecutionObjectClosure::from_canonical_bytes(&catalog_object.bytes)
        .map_err(protocol)?;
    if closure.root.authorization_scope != authority.run_id {
        return Err(protocol("recovery closure is outside the run authority"));
    }

    let mut objects = Vec::with_capacity(closure.objects.len());
    for location in &closure.objects {
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
                run_key,
                &control.download(&artifact).await.map_err(protocol)?,
                &authority.run_id,
                EncryptionPurpose::Result,
            )?,
        };
        object.validate().map_err(protocol)?;
        if object.reference.authorization_scope != authority.run_id {
            return Err(protocol("recovery object is outside the run authority"));
        }
        objects.push(object);
    }
    Ok(RestoredObjectClosure {
        root: closure.root,
        objects,
        locations: closure.objects,
    })
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
