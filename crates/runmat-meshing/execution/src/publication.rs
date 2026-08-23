use runmat_execution::identity::ValueId;
use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1;
use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{LogicalObject, ObjectDescriptor};
use runmat_execution_runner::AttemptSuccess;

use crate::{
    import_stage_objects, MeshingExecutionError, MeshingExecutionResult, MeshingStageObjectRoot,
    PreparedMeshingStageObjects, MESHING_STAGE_MANIFEST_MEDIA_TYPE,
};

const RESULT_IDENTITY_SCHEMA: &str = "runmat.meshing.stage-result-identity.v2";
const STAGE_MANIFEST_SCHEMA: &str = "runmat.meshing.stage-manifest.v2";
const CHUNK_SCHEMA: &str = "runmat.meshing.logical-record-chunk.v2";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshingArtifactAccess {
    pub authorization_scope: String,
    pub encryption_context: Digest,
}

impl MeshingArtifactAccess {
    pub fn validate(&self) -> MeshingExecutionResult<()> {
        if self.authorization_scope.is_empty()
            || self.authorization_scope.len() > 1024
            || self.authorization_scope.chars().any(char::is_control)
            || self
                .encryption_context
                .bytes()
                .iter()
                .all(|byte| *byte == 0)
        {
            return Err(MeshingExecutionError::Identity(
                "artifact authorization or encryption context is invalid",
            ));
        }
        Ok(())
    }

    pub fn value_id(&self, logical_digest: Digest) -> ValueId {
        ValueId::derive(&[
            b"runmat-meshing-artifact-object-v2",
            logical_digest.bytes(),
            self.encryption_context.bytes(),
            self.authorization_scope.as_bytes(),
        ])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedMeshingResultPublication {
    stage_objects: PreparedMeshingStageObjects,
    root_output: ValuePayload,
    result_objects: Vec<ValueRef>,
}

impl PreparedMeshingResultPublication {
    pub const fn stage_objects(&self) -> &PreparedMeshingStageObjects {
        &self.stage_objects
    }

    pub const fn root_output(&self) -> &ValuePayload {
        &self.root_output
    }

    pub fn result_objects(&self) -> &[ValueRef] {
        &self.result_objects
    }

    pub fn attempt_success(&self) -> AttemptSuccess {
        AttemptSuccess {
            outputs: vec![self.root_output.clone()],
            result_objects: self.result_objects.clone(),
        }
    }
}

pub fn prepare_result_publication(
    stage_objects: PreparedMeshingStageObjects,
    access: MeshingArtifactAccess,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedMeshingResultPublication> {
    access.validate()?;
    stage_objects.revalidate(limits)?;
    if !stage_objects.manifest.is_dependency_eligible() {
        return Err(MeshingExecutionError::Identity(
            "diagnostic-only meshing artifacts cannot become successful task results",
        ));
    }
    let result_objects = stage_objects
        .objects
        .iter()
        .map(|object| object_reference(object, &stage_objects.root, &access))
        .collect::<MeshingExecutionResult<Vec<_>>>()?;
    let root = result_objects
        .iter()
        .find(|reference| reference.logical_digest == stage_objects.root.digest)
        .cloned()
        .ok_or(MeshingExecutionError::Identity(
            "prepared publication has no root manifest reference",
        ))?;
    let root_output = ValuePayload::Object(Box::new(root));
    root_output.validate(ValueLimits::default()).map_err(|_| {
        MeshingExecutionError::Identity("root manifest is not a valid execution value reference")
    })?;
    Ok(PreparedMeshingResultPublication {
        stage_objects,
        root_output,
        result_objects,
    })
}

pub fn import_result_publication(
    source: &impl CacheImport,
    root: &ValueRef,
    access: MeshingArtifactAccess,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedMeshingResultPublication> {
    access.validate()?;
    ValuePayload::Object(Box::new(root.clone()))
        .validate(ValueLimits::default())
        .map_err(|_| MeshingExecutionError::Identity("invalid root manifest reference"))?;
    if root.kind != ValueRefKind::ResultObject
        || root.authorization_scope != access.authorization_scope
        || root.encryption_context != access.encryption_context
        || root.media_type != MESHING_STAGE_MANIFEST_MEDIA_TYPE
        || root.value_schema != STAGE_MANIFEST_SCHEMA
    {
        return Err(MeshingExecutionError::Identity(
            "root manifest reference is outside meshing artifact authority",
        ));
    }
    let stage_objects = import_stage_objects(
        source,
        MeshingStageObjectRoot {
            digest: root.logical_digest,
            encoded_length: root.encoded_length,
        },
        limits,
    )?;
    let publication = prepare_result_publication(stage_objects, access, limits)?;
    let ValuePayload::Object(imported_root) = &publication.root_output else {
        return Err(MeshingExecutionError::Identity(
            "imported publication root is not an object reference",
        ));
    };
    if imported_root.as_ref() != root {
        return Err(MeshingExecutionError::Identity(
            "imported publication root differs from its execution reference",
        ));
    }
    Ok(publication)
}

fn object_reference(
    object: &LogicalObject,
    root: &ObjectDescriptor,
    access: &MeshingArtifactAccess,
) -> MeshingExecutionResult<ValueRef> {
    object.validate()?;
    let value_schema = if object.descriptor.digest == root.digest {
        STAGE_MANIFEST_SCHEMA
    } else if object.descriptor.logical_name.contains("/identities/") {
        RESULT_IDENTITY_SCHEMA
    } else {
        CHUNK_SCHEMA
    };
    result_object_reference(object, access, value_schema)
}

pub(crate) fn result_object_reference(
    object: &LogicalObject,
    access: &MeshingArtifactAccess,
    value_schema: &str,
) -> MeshingExecutionResult<ValueRef> {
    object.validate()?;
    let reference = ValueRef {
        schema_version: VALUE_PAYLOAD_SCHEMA_V1,
        id: access.value_id(object.descriptor.digest),
        logical_digest: object.descriptor.digest,
        encoded_length: object.descriptor.encoded_length,
        media_type: object.descriptor.media_type.clone(),
        value_schema: value_schema.into(),
        encryption_context: access.encryption_context,
        kind: ValueRefKind::ResultObject,
        authorization_scope: access.authorization_scope.clone(),
        resident_fence: None,
    };
    ValuePayload::Object(Box::new(reference.clone()))
        .validate(ValueLimits::default())
        .map_err(|_| MeshingExecutionError::Identity("invalid meshing result object reference"))?;
    Ok(reference)
}
