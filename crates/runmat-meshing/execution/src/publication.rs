mod storage;
use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef, ValueRefKind};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, LogicalObject, ObjectDescriptor};
use runmat_execution_runner::AttemptSuccess;

use crate::{
    MeshingExecutionError, MeshingExecutionResult, MeshingStageObjectRoot,
    PreparedMeshingStageObjects, MESHING_STAGE_MANIFEST_MEDIA_TYPE,
};

const RESULT_IDENTITY_SCHEMA: &str = "runmat.meshing.stage-result-identity.v2";
const STAGE_MANIFEST_SCHEMA: &str = "runmat.meshing.stage-manifest.v2";
const CHUNK_SCHEMA: &str = "runmat.meshing.logical-record-chunk.v2";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedMeshingResultPublication {
    stage_objects: PreparedMeshingStageObjects,
    root_output: ValuePayload,
    result_objects: Vec<ValueRef>,
}

impl PreparedMeshingResultPublication {
    pub(crate) fn into_stage_objects(self) -> PreparedMeshingStageObjects {
        self.stage_objects
    }

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
        AttemptSuccess::Values {
            outputs: vec![self.root_output.clone()],
            result_objects: self.result_objects.clone(),
            resources: None,
        }
    }
}

pub fn prepare_result_publication(
    stage_objects: PreparedMeshingStageObjects,
    access: ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedMeshingResultPublication> {
    access.validate()?;
    let validated = crate::objects::revalidate_stage_objects(stage_objects, limits)?;
    prepare_validated_result_publication(validated, &access, &mut |_| Ok(()))
}

pub(crate) fn prepare_validated_result_publication(
    stage_objects: crate::objects::ValidatedStageObjects,
    access: &ArtifactAccess,
    admit: &mut impl FnMut(u64) -> MeshingExecutionResult<()>,
) -> MeshingExecutionResult<PreparedMeshingResultPublication> {
    let stage_objects = stage_objects.into_raw();
    access.validate()?;
    if !stage_objects.manifest.is_dependency_eligible() {
        return Err(MeshingExecutionError::Identity(
            "diagnostic-only meshing artifacts cannot become successful task results",
        ));
    }
    let slots = stage_objects
        .objects
        .len()
        .checked_mul(std::mem::size_of::<ValueRef>())
        .ok_or_else(|| {
            MeshingExecutionError::Invalid("result reference storage overflow".into())
        })?;
    admit(slots as u64)?;
    let mut result_objects = Vec::with_capacity(stage_objects.objects.len());
    for object in &stage_objects.objects {
        let schema = object_schema(object, &stage_objects.root);
        admit(access.reference_text_bytes(&object.descriptor.media_type, schema)?)?;
        result_objects.push(access.reference_descriptor(
            &object.descriptor,
            ValueRefKind::ResultObject,
            schema,
        )?);
    }
    let root = result_objects
        .iter()
        .find(|reference| reference.logical_digest == stage_objects.root.digest)
        .ok_or(MeshingExecutionError::Identity(
            "prepared publication has no root manifest reference",
        ))?;
    admit(
        std::mem::size_of::<ValueRef>() as u64
            + access.reference_text_bytes(&root.media_type, &root.value_schema)?,
    )?;
    let root_output = ValuePayload::Object(Box::new(root.clone()));
    Ok(PreparedMeshingResultPublication {
        stage_objects,
        root_output,
        result_objects,
    })
}

pub fn import_result_publication(
    source: &impl CacheImport,
    root: &ValueRef,
    access: ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedMeshingResultPublication> {
    import_result_publication_with_admission(
        source,
        root,
        &access,
        limits,
        &runmat_meshing_core::NeverCancelled,
        &mut |_| Ok(()),
    )
}
pub(crate) fn import_result_publication_with_admission(
    source: &impl CacheImport,
    root: &ValueRef,
    access: &ArtifactAccess,
    limits: ObjectInventoryLimits,
    cancellation: &dyn runmat_meshing_core::MeshingCancellationSignal,
    admit: &mut impl FnMut(u64) -> MeshingExecutionResult<()>,
) -> MeshingExecutionResult<PreparedMeshingResultPublication> {
    access.validate()?;
    root.validate(ValueLimits::default())
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
    let stage_objects = crate::objects::import_stage_objects_with_admission(
        source,
        MeshingStageObjectRoot {
            digest: root.logical_digest,
            encoded_length: root.encoded_length,
        },
        limits,
        cancellation,
        admit,
    )?;
    let publication = prepare_validated_result_publication(stage_objects, access, admit)?;
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

fn object_schema(object: &LogicalObject, root: &ObjectDescriptor) -> &'static str {
    if object.descriptor.digest == root.digest {
        STAGE_MANIFEST_SCHEMA
    } else if object.descriptor.logical_name.contains("/identities/") {
        RESULT_IDENTITY_SCHEMA
    } else {
        CHUNK_SCHEMA
    }
}
pub(crate) fn result_object_reference(
    object: &LogicalObject,
    access: &ArtifactAccess,
    value_schema: &str,
) -> MeshingExecutionResult<ValueRef> {
    Ok(access.reference(object, ValueRefKind::ResultObject, value_schema)?)
}
