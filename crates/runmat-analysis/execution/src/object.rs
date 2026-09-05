use runmat_analysis_fea::execution::{
    chunk_numerical_payload, AnalysisNumericalObjectKind, AnalysisObjectManifest,
    ANALYSIS_CHUNK_MEDIA_TYPE, ANALYSIS_CHUNK_VALUE_SCHEMA, ANALYSIS_MANIFEST_MEDIA_TYPE,
    ANALYSIS_MANIFEST_VALUE_SCHEMA,
};
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution::Digest;
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
