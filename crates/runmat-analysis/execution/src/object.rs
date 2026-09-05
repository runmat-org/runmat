use runmat_analysis_fea::execution::{
    chunk_numerical_payload, AnalysisNumericalObjectKind, AnalysisObjectManifest,
    ANALYSIS_CHUNK_MEDIA_TYPE, ANALYSIS_CHUNK_VALUE_SCHEMA, ANALYSIS_MANIFEST_MEDIA_TYPE,
    ANALYSIS_MANIFEST_VALUE_SCHEMA,
};
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};
use runmat_execution_artifact::{ArtifactAccess, LogicalObject, ObjectDescriptor, ObjectNamespace};

use crate::{AnalysisBridgeError, AnalysisBridgeResult};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedAnalysisObjects {
    pub manifest: AnalysisObjectManifest,
    pub root: ObjectDescriptor,
    pub objects: Vec<LogicalObject>,
}

impl PreparedAnalysisObjects {
    pub fn payload(&self) -> Vec<u8> {
        self.objects
            .iter()
            .skip(1)
            .flat_map(|object| object.bytes.iter().copied())
            .collect()
    }

    pub fn references(
        &self,
        access: &ArtifactAccess,
        kind: ValueRefKind,
    ) -> AnalysisBridgeResult<Vec<ValueRef>> {
        self.objects
            .iter()
            .enumerate()
            .map(|(index, object)| {
                let schema = if index == 0 {
                    ANALYSIS_MANIFEST_VALUE_SCHEMA
                } else {
                    ANALYSIS_CHUNK_VALUE_SCHEMA
                };
                access.reference(object, kind, schema).map_err(Into::into)
            })
            .collect()
    }

    pub fn root_reference(
        &self,
        access: &ArtifactAccess,
        kind: ValueRefKind,
    ) -> AnalysisBridgeResult<ValueRef> {
        Ok(access.reference(
            self.objects.first().ok_or(AnalysisBridgeError::Identity(
                "analysis object has no manifest",
            ))?,
            kind,
            ANALYSIS_MANIFEST_VALUE_SCHEMA,
        )?)
    }
}

pub fn import_analysis_objects(
    source: &impl CacheImport,
    root: &ValueRef,
    expected_kind: AnalysisNumericalObjectKind,
    expected_logical_identity: Digest,
    expected_payload_digest: Digest,
    access: &ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    access.validate_reference(
        root,
        root.kind,
        ANALYSIS_MANIFEST_MEDIA_TYPE,
        ANALYSIS_MANIFEST_VALUE_SCHEMA,
    )?;
    if matches!(
        root.kind,
        runmat_execution::value::ValueRefKind::ResidentObject
            | runmat_execution::value::ValueRefKind::SlicedObject
    ) || root.encoded_length == 0
        || root.encoded_length > limits.max_object_bytes
    {
        return Err(AnalysisBridgeError::Invalid(
            "analysis input root is not a portable bounded manifest".into(),
        ));
    }
    let manifest_bytes = source.read_verified(root.logical_digest)?.ok_or_else(|| {
        AnalysisBridgeError::Invalid("analysis manifest is absent from the object store".into())
    })?;
    if manifest_bytes.len() as u64 != root.encoded_length
        || Digest::sha256(&manifest_bytes) != root.logical_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "analysis manifest bytes differ from their reference",
        ));
    }
    let manifest = AnalysisObjectManifest::from_canonical_bytes(&manifest_bytes)?;
    if manifest.kind != expected_kind
        || manifest.logical_identity != expected_logical_identity
        || manifest.payload_digest != expected_payload_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "analysis manifest semantics differ from the workload input",
        ));
    }
    if manifest.chunks.len().saturating_add(1) > limits.max_objects
        || manifest.encoded_length > limits.max_total_bytes
        || root
            .encoded_length
            .checked_add(manifest.encoded_length)
            .is_none_or(|total| total > limits.max_total_bytes)
    {
        return Err(AnalysisBridgeError::Invalid(
            "analysis object closure exceeds its inventory limits".into(),
        ));
    }
    let mut objects = Vec::with_capacity(manifest.chunks.len() + 1);
    objects.push(logical_object(
        ObjectNamespace::InputValue,
        "analysis/manifests",
        ANALYSIS_MANIFEST_MEDIA_TYPE,
        manifest_bytes,
    )?);
    let mut payload =
        Vec::with_capacity(usize::try_from(manifest.encoded_length).map_err(|_| {
            AnalysisBridgeError::Invalid("analysis payload length is not representable".into())
        })?);
    for chunk in &manifest.chunks {
        if chunk.encoded_length > limits.max_object_bytes {
            return Err(AnalysisBridgeError::Invalid(
                "analysis chunk exceeds its object limit".into(),
            ));
        }
        let bytes = source.read_verified(chunk.digest)?.ok_or_else(|| {
            AnalysisBridgeError::Invalid("analysis chunk is absent from the object store".into())
        })?;
        if bytes.len() as u64 != chunk.encoded_length || Digest::sha256(&bytes) != chunk.digest {
            return Err(AnalysisBridgeError::Identity(
                "analysis chunk bytes differ from their descriptor",
            ));
        }
        payload.extend_from_slice(&bytes);
        objects.push(logical_object(
            ObjectNamespace::InputValue,
            &format!("analysis/chunks/{:08}", chunk.ordinal),
            ANALYSIS_CHUNK_MEDIA_TYPE,
            bytes,
        )?);
    }
    if payload.len() as u64 != manifest.encoded_length
        || Digest::sha256(&payload) != manifest.payload_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "analysis object closure differs from its payload identity",
        ));
    }
    validate_inventory(&objects, limits)?;
    Ok(PreparedAnalysisObjects {
        manifest,
        root: objects[0].descriptor.clone(),
        objects,
    })
}

pub fn prepare_analysis_objects(
    kind: AnalysisNumericalObjectKind,
    logical_identity: Digest,
    payload: &[u8],
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    let chunked = chunk_numerical_payload(kind, logical_identity, payload, chunk_bytes)?;
    let mut objects = Vec::with_capacity(chunked.chunks.len() + 1);
    let manifest_object = logical_object(
        namespace,
        "analysis/manifests",
        ANALYSIS_MANIFEST_MEDIA_TYPE,
        chunked.manifest.canonical_bytes()?,
    )?;
    let root = manifest_object.descriptor.clone();
    objects.push(manifest_object);
    for (descriptor, bytes) in chunked.manifest.chunks.iter().zip(chunked.chunks) {
        if descriptor.digest != Digest::sha256(&bytes)
            || descriptor.encoded_length != bytes.len() as u64
        {
            return Err(AnalysisBridgeError::Identity(
                "analysis chunk differs from its manifest descriptor",
            ));
        }
        objects.push(logical_object(
            namespace,
            &format!("analysis/chunks/{:08}", descriptor.ordinal),
            ANALYSIS_CHUNK_MEDIA_TYPE,
            bytes,
        )?);
    }
    validate_inventory(&objects, limits)?;
    Ok(PreparedAnalysisObjects {
        manifest: chunked.manifest,
        root,
        objects,
    })
}

fn logical_object(
    namespace: ObjectNamespace,
    prefix: &str,
    media_type: &str,
    bytes: Vec<u8>,
) -> AnalysisBridgeResult<LogicalObject> {
    let digest = Digest::sha256(&bytes);
    let suffix = digest
        .bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(LogicalObject::new(
        namespace,
        format!("{prefix}/{suffix}"),
        media_type,
        bytes,
    )?)
}
