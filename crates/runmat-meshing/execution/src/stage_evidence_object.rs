//! Externalized factual stage observations carried through generic execution result inventories.
//!
//! Stage evidence is deliberately outside the stage-manifest closure: measured resource use and
//! elapsed time must not affect the scheduling-independent geometric result identity.

use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};
use runmat_execution_artifact::{LogicalObject, ObjectNamespace};
use runmat_meshing_core::{CanonicalMeshingContract, MeshingStageEvidence};

use crate::object_support::{enforce_object_length, logical_object, read_exact};
use crate::publication::result_object_reference;
use crate::{
    MeshingExecutionError, MeshingExecutionResult, MeshingHostWorkload,
    MESHING_STAGE_MANIFEST_MEDIA_TYPE,
};

pub const STAGE_EVIDENCE_MEDIA_TYPE: &str = "application/vnd.runmat.meshing-stage-evidence.v2+cbor";
pub const STAGE_EVIDENCE_VALUE_SCHEMA: &str = "runmat.meshing.stage-evidence.v2";

pub(crate) struct PreparedStageEvidenceObservation {
    pub object: LogicalObject,
    pub reference: ValueRef,
}

pub(crate) fn prepare_stage_evidence_observation(
    host: &MeshingHostWorkload,
    evidence: MeshingStageEvidence,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedStageEvidenceObservation> {
    validate_evidence(host, &evidence)?;
    let object = logical_object(
        "meshing/stage-evidence",
        ObjectNamespace::ResultValue,
        STAGE_EVIDENCE_MEDIA_TYPE,
        evidence.canonical_encode()?,
    )?;
    validate_inventory(std::slice::from_ref(&object), limits)?;
    let reference =
        result_object_reference(&object, &host.artifact_access, STAGE_EVIDENCE_VALUE_SCHEMA)?;
    Ok(PreparedStageEvidenceObservation { object, reference })
}

pub(crate) fn validate_complete_result_inventory(
    stage_objects: &[LogicalObject],
    evidence_object: &LogicalObject,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<()> {
    let object_count = stage_objects
        .len()
        .checked_add(1)
        .ok_or_else(|| limit("meshing result object count overflowed"))?;
    if object_count > limits.max_objects
        || stage_objects
            .iter()
            .any(|object| object.descriptor.digest == evidence_object.descriptor.digest)
    {
        return Err(limit(
            "meshing result object inventory exceeds its count or uniqueness bound",
        ));
    }
    let total_bytes = stage_objects
        .iter()
        .chain(std::iter::once(evidence_object))
        .try_fold(0_u64, |total, object| {
            total
                .checked_add(object.descriptor.encoded_length)
                .ok_or_else(|| limit("meshing result byte inventory overflowed"))
        })?;
    if total_bytes > limits.max_total_bytes {
        return Err(limit(
            "meshing result object inventory exceeds its total-byte bound",
        ));
    }
    Ok(())
}

/// Loads and independently validates the one factual observation attached to a completed stage.
pub fn import_stage_evidence_observation(
    source: &impl CacheImport,
    host: &MeshingHostWorkload,
    result_objects: &[ValueRef],
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<MeshingStageEvidence> {
    let matches = result_objects
        .iter()
        .filter(|reference| {
            reference.media_type == STAGE_EVIDENCE_MEDIA_TYPE
                || reference.value_schema == STAGE_EVIDENCE_VALUE_SCHEMA
        })
        .collect::<Vec<_>>();
    let [reference] = matches.as_slice() else {
        return Err(MeshingExecutionError::Invalid(
            "completed meshing stage requires exactly one externalized stage observation".into(),
        ));
    };
    validate_reference(host, reference)?;
    enforce_object_length("meshing stage evidence", reference.encoded_length, limits)?;
    if limits.max_objects == 0 || reference.encoded_length > limits.max_total_bytes {
        return Err(runmat_execution_artifact::ArtifactError::Limit(
            "meshing stage evidence exceeds its inventory limit".into(),
        )
        .into());
    }
    let bytes = read_exact(source, reference.logical_digest, reference.encoded_length)?;
    let evidence = MeshingStageEvidence::canonical_decode(&bytes)?;
    validate_evidence(host, &evidence)?;
    validate_result_binding(&evidence, result_objects)?;
    Ok(evidence)
}

pub(crate) fn validate_stage_evidence_inventory(
    host: &MeshingHostWorkload,
    evidence: &MeshingStageEvidence,
    result_objects: &[ValueRef],
) -> MeshingExecutionResult<()> {
    validate_evidence(host, evidence)?;
    validate_stage_evidence_result(&host.artifact_access, evidence, result_objects)
}

pub(crate) fn validate_stage_evidence_result(
    access: &crate::MeshingArtifactAccess,
    evidence: &MeshingStageEvidence,
    result_objects: &[ValueRef],
) -> MeshingExecutionResult<()> {
    let matches = result_objects
        .iter()
        .filter(|reference| {
            reference.media_type == STAGE_EVIDENCE_MEDIA_TYPE
                || reference.value_schema == STAGE_EVIDENCE_VALUE_SCHEMA
        })
        .collect::<Vec<_>>();
    let [reference] = matches.as_slice() else {
        return Err(MeshingExecutionError::Invalid(
            "meshing host response requires exactly one stage-evidence result object".into(),
        ));
    };
    validate_reference_for_access(access, reference)?;
    validate_result_binding(evidence, result_objects)?;
    let bytes = evidence.canonical_encode()?;
    if reference.logical_digest != runmat_execution::Digest::sha256(&bytes)
        || reference.encoded_length != bytes.len() as u64
    {
        return Err(MeshingExecutionError::Identity(
            "stage-evidence result object does not bind the host observation",
        ));
    }
    Ok(())
}

fn validate_reference(
    host: &MeshingHostWorkload,
    reference: &ValueRef,
) -> MeshingExecutionResult<()> {
    validate_reference_for_access(&host.artifact_access, reference)
}

fn validate_reference_for_access(
    access: &crate::MeshingArtifactAccess,
    reference: &ValueRef,
) -> MeshingExecutionResult<()> {
    ValuePayload::Object(Box::new(reference.clone()))
        .validate(runmat_execution::value::ValueLimits::default())
        .map_err(|_| MeshingExecutionError::Identity("invalid stage-evidence reference"))?;
    if reference.kind != ValueRefKind::ResultObject
        || reference.media_type != STAGE_EVIDENCE_MEDIA_TYPE
        || reference.value_schema != STAGE_EVIDENCE_VALUE_SCHEMA
        || reference.authorization_scope != access.authorization_scope
        || reference.encryption_context != access.encryption_context
        || reference.id != access.value_id(reference.logical_digest)
        || reference.resident_fence.is_some()
    {
        return Err(MeshingExecutionError::Identity(
            "stage-evidence reference is outside the workload artifact authority",
        ));
    }
    Ok(())
}

fn validate_evidence(
    host: &MeshingHostWorkload,
    evidence: &MeshingStageEvidence,
) -> MeshingExecutionResult<()> {
    evidence.validate()?;
    if evidence.stage != host.workload.stage || evidence.partition != host.workload.partition {
        return Err(MeshingExecutionError::Identity(
            "stage evidence does not bind its meshing workload",
        ));
    }
    Ok(())
}

fn validate_result_binding(
    evidence: &MeshingStageEvidence,
    result_objects: &[ValueRef],
) -> MeshingExecutionResult<()> {
    let roots = result_objects
        .iter()
        .filter(|reference| reference.media_type == MESHING_STAGE_MANIFEST_MEDIA_TYPE)
        .collect::<Vec<_>>();
    let [root] = roots.as_slice() else {
        return Err(MeshingExecutionError::Invalid(
            "meshing result inventory requires exactly one stage-manifest root".into(),
        ));
    };
    if root.logical_digest.bytes() != evidence.stage_result_digest.bytes() {
        return Err(MeshingExecutionError::Identity(
            "stage observation does not bind the result manifest",
        ));
    }
    Ok(())
}

fn limit(reason: &'static str) -> MeshingExecutionError {
    runmat_execution_artifact::ArtifactError::Limit(reason.into()).into()
}
