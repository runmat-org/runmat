use runmat_execution::value::ValueRef;
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};
use runmat_execution_artifact::{LogicalObject, ObjectDescriptor, ObjectNamespace};
use runmat_meshing_core::{CanonicalMeshingContract, MeshingChunkMediaType, MeshingEvidence};

use crate::object_support::{
    enforce_object_length, input_object_reference, logical_object, read_exact, validate_input_root,
};
use crate::{MeshingArtifactAccess, MeshingExecutionError, MeshingExecutionResult};

pub(crate) const EVIDENCE_VALUE_SCHEMA: &str = "runmat.meshing.evidence.v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceObjectRoot {
    pub digest: Digest,
    pub encoded_length: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedEvidenceObjects {
    pub evidence: MeshingEvidence,
    pub root: ObjectDescriptor,
    pub objects: Vec<LogicalObject>,
}

impl PreparedEvidenceObjects {
    pub fn root_reference(&self) -> EvidenceObjectRoot {
        EvidenceObjectRoot {
            digest: self.root.digest,
            encoded_length: self.root.encoded_length,
        }
    }

    pub fn revalidate(&self, limits: ObjectInventoryLimits) -> MeshingExecutionResult<()> {
        if prepare_evidence_objects(self.evidence.clone(), limits)? != *self {
            return Err(MeshingExecutionError::Identity(
                "prepared meshing evidence is not its canonical object closure",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedEvidenceInput {
    evidence_objects: PreparedEvidenceObjects,
    root_input: ValueRef,
    input_objects: Vec<ValueRef>,
}

impl PreparedEvidenceInput {
    pub const fn evidence_objects(&self) -> &PreparedEvidenceObjects {
        &self.evidence_objects
    }

    pub const fn root_input(&self) -> &ValueRef {
        &self.root_input
    }

    pub fn input_objects(&self) -> &[ValueRef] {
        &self.input_objects
    }
}

pub fn prepare_evidence_input(
    evidence_objects: PreparedEvidenceObjects,
    access: MeshingArtifactAccess,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedEvidenceInput> {
    access.validate()?;
    evidence_objects.revalidate(limits)?;
    let object = evidence_objects
        .objects
        .first()
        .ok_or(MeshingExecutionError::Identity(
            "prepared meshing evidence has no root object",
        ))?;
    let root_input = input_object_reference(
        object,
        &access,
        EVIDENCE_VALUE_SCHEMA,
        "invalid meshing evidence input reference",
    )?;
    Ok(PreparedEvidenceInput {
        evidence_objects,
        input_objects: vec![root_input.clone()],
        root_input,
    })
}

pub fn import_evidence_input(
    source: &impl CacheImport,
    root: &ValueRef,
    access: MeshingArtifactAccess,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedEvidenceInput> {
    access.validate()?;
    validate_input_root(
        root,
        &access,
        MeshingChunkMediaType::MeshingEvidence.media_type(),
        EVIDENCE_VALUE_SCHEMA,
        "meshing evidence root is outside input artifact authority",
    )?;
    let objects = import_evidence_objects(
        source,
        EvidenceObjectRoot {
            digest: root.logical_digest,
            encoded_length: root.encoded_length,
        },
        limits,
    )?;
    let prepared = prepare_evidence_input(objects, access, limits)?;
    if prepared.root_input != *root {
        return Err(MeshingExecutionError::Identity(
            "imported meshing evidence root differs from its execution reference",
        ));
    }
    Ok(prepared)
}

pub fn prepare_evidence_objects(
    evidence: MeshingEvidence,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedEvidenceObjects> {
    let bytes = evidence.canonical_encode()?;
    let object = logical_object(
        "meshing/evidence",
        ObjectNamespace::InputValue,
        MeshingChunkMediaType::MeshingEvidence.media_type(),
        bytes,
    )?;
    let root = object.descriptor.clone();
    let objects = vec![object];
    validate_inventory(&objects, limits)?;
    Ok(PreparedEvidenceObjects {
        evidence,
        root,
        objects,
    })
}

pub fn import_evidence_objects(
    source: &impl CacheImport,
    root: EvidenceObjectRoot,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedEvidenceObjects> {
    enforce_object_length("meshing evidence", root.encoded_length, limits)?;
    if limits.max_objects == 0 || root.encoded_length > limits.max_total_bytes {
        return Err(runmat_execution_artifact::ArtifactError::Limit(
            "meshing evidence object inventory exceeds its limit".into(),
        )
        .into());
    }
    let bytes = read_exact(source, root.digest, root.encoded_length)?;
    let evidence = MeshingEvidence::canonical_decode(&bytes)?;
    let prepared = prepare_evidence_objects(evidence, limits)?;
    if prepared.root.digest != root.digest || prepared.root.encoded_length != root.encoded_length {
        return Err(MeshingExecutionError::Identity(
            "imported meshing evidence differs from requested root",
        ));
    }
    Ok(prepared)
}
