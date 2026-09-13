use runmat_execution::value::ValueRef;
use runmat_execution_artifact::encryption::{EncryptionPurpose, RunKeyMaterial};
use runmat_execution_artifact::{
    ProgramExecutionObject, ProgramExecutionObjectClosure, ProgramExecutionObjectLocation,
};
use runmat_execution_transport_native::control::{DriverAuthority, DriverControlPlane};

use super::crypto::open_object;
use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) struct RestoredObjectClosure {
    pub(super) root: ValueRef,
    pub(super) objects: Vec<ProgramExecutionObject>,
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
    })
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
