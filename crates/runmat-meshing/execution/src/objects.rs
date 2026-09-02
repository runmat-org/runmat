use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};
use runmat_execution_artifact::{ArtifactError, LogicalObject, ObjectDescriptor, ObjectNamespace};
use runmat_meshing_core::{
    decode_stage_manifest_streams_with_cancellation, verify_stage_manifest_closure,
    CanonicalMeshingContract, EncodedMeshingChunk, MeshingCancellationSignal, MeshingStageManifest,
    MeshingStageResultIdentity, StableDigest,
};

use crate::object_support::{
    add_inventory_bytes, enforce_object_length, logical_object, read_exact, read_verified,
};
use crate::{MeshingExecutionError, MeshingExecutionResult};

pub const MESHING_STAGE_MANIFEST_MEDIA_TYPE: &str =
    "application/vnd.runmat.meshing-stage-manifest.v2+cbor";
pub const MESHING_RESULT_IDENTITY_MEDIA_TYPE: &str =
    "application/vnd.runmat.meshing-stage-result-identity.v2+cbor";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshingStageObjectRoot {
    pub digest: Digest,
    pub encoded_length: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedMeshingStageObjects {
    pub result_identity: MeshingStageResultIdentity,
    pub manifest: MeshingStageManifest,
    pub root: ObjectDescriptor,
    pub objects: Vec<LogicalObject>,
}

impl PreparedMeshingStageObjects {
    pub fn root_reference(&self) -> MeshingStageObjectRoot {
        MeshingStageObjectRoot {
            digest: self.root.digest,
            encoded_length: self.root.encoded_length,
        }
    }

    pub fn revalidate(&self, limits: ObjectInventoryLimits) -> MeshingExecutionResult<()> {
        let chunks = self.encoded_chunks()?;
        let rebuilt = prepare_stage_objects(
            self.result_identity.clone(),
            self.manifest.clone(),
            chunks,
            limits,
        )?;
        if rebuilt != *self {
            return Err(MeshingExecutionError::Identity(
                "prepared meshing object set is not its canonical closure",
            ));
        }
        Ok(())
    }

    pub fn decoded_streams(
        &self,
    ) -> MeshingExecutionResult<Vec<runmat_meshing_core::MeshingChunkStream>> {
        Ok(runmat_meshing_core::decode_stage_manifest_streams(
            &self.manifest,
            &self.result_identity,
            &self.encoded_chunks()?,
        )?)
    }

    pub fn decoded_streams_with_cancellation(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> MeshingExecutionResult<Vec<runmat_meshing_core::MeshingChunkStream>> {
        let chunks = self.encoded_chunks_with_cancellation(cancellation)?;
        Ok(decode_stage_manifest_streams_with_cancellation(
            &self.manifest,
            &self.result_identity,
            &chunks,
            cancellation,
        )?)
    }

    fn encoded_chunks(&self) -> MeshingExecutionResult<Vec<EncodedMeshingChunk>> {
        let mut chunks = Vec::with_capacity(self.manifest.chunks.len());
        for descriptor in &self.manifest.chunks {
            let digest = execution_digest(descriptor.digest);
            let object = self
                .objects
                .iter()
                .find(|object| object.descriptor.digest == digest)
                .ok_or(MeshingExecutionError::MissingObject(digest))?;
            chunks.push(EncodedMeshingChunk {
                descriptor: descriptor.clone(),
                bytes: object.bytes.clone(),
            });
        }
        Ok(chunks)
    }

    fn encoded_chunks_with_cancellation(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> MeshingExecutionResult<Vec<EncodedMeshingChunk>> {
        let mut chunks = Vec::with_capacity(self.manifest.chunks.len());
        for descriptor in &self.manifest.chunks {
            ensure_active(cancellation)?;
            let digest = execution_digest(descriptor.digest);
            let object = self
                .objects
                .iter()
                .find(|object| object.descriptor.digest == digest)
                .ok_or(MeshingExecutionError::MissingObject(digest))?;
            let mut bytes = Vec::with_capacity(object.bytes.len());
            for block in object.bytes.chunks(1024 * 1024) {
                ensure_active(cancellation)?;
                bytes.extend_from_slice(block);
            }
            chunks.push(EncodedMeshingChunk {
                descriptor: descriptor.clone(),
                bytes,
            });
        }
        Ok(chunks)
    }
}

fn ensure_active(cancellation: &dyn MeshingCancellationSignal) -> MeshingExecutionResult<()> {
    if cancellation.is_cancelled() {
        return Err(MeshingExecutionError::Invalid(
            "meshing stream reconstruction was cancelled".into(),
        ));
    }
    Ok(())
}

pub fn prepare_stage_objects(
    result_identity: MeshingStageResultIdentity,
    manifest: MeshingStageManifest,
    chunks: Vec<EncodedMeshingChunk>,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedMeshingStageObjects> {
    verify_stage_manifest_closure(&manifest, &result_identity, &chunks)?;

    let mut objects = Vec::with_capacity(chunks.len() + 2);
    let identity_bytes = result_identity.canonical_encode()?;
    objects.push(logical_object(
        "meshing/v2/identities",
        ObjectNamespace::ResultValue,
        MESHING_RESULT_IDENTITY_MEDIA_TYPE,
        identity_bytes,
    )?);
    for chunk in chunks {
        let object = logical_object(
            "meshing/v2/chunks",
            ObjectNamespace::ResultValue,
            chunk.descriptor.media_type.media_type(),
            chunk.bytes,
        )?;
        if stable_digest(object.descriptor.digest) != chunk.descriptor.digest
            || object.descriptor.encoded_length != chunk.descriptor.encoded_length
        {
            return Err(MeshingExecutionError::Identity(
                "chunk descriptor differs from shared object descriptor",
            ));
        }
        objects.push(object);
    }
    let manifest_bytes = manifest.canonical_encode()?;
    let manifest_object = logical_object(
        "meshing/v2/manifests",
        ObjectNamespace::ResultValue,
        MESHING_STAGE_MANIFEST_MEDIA_TYPE,
        manifest_bytes,
    )?;
    let root = manifest_object.descriptor.clone();
    objects.push(manifest_object);
    validate_inventory(&objects, limits)?;
    Ok(PreparedMeshingStageObjects {
        result_identity,
        manifest,
        root,
        objects,
    })
}

pub fn import_stage_objects(
    source: &impl CacheImport,
    root: MeshingStageObjectRoot,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedMeshingStageObjects> {
    enforce_object_length("manifest", root.encoded_length, limits)?;
    let manifest_bytes = read_exact(source, root.digest, root.encoded_length)?;
    let manifest = MeshingStageManifest::canonical_decode(&manifest_bytes)?;
    let object_count = manifest.chunks.len().checked_add(2).ok_or_else(|| {
        ArtifactError::Limit("meshing stage object inventory count overflow".into())
    })?;
    if object_count > limits.max_objects {
        return Err(ArtifactError::Limit("too many meshing stage objects".into()).into());
    }
    let mut total_bytes = root.encoded_length;
    let identity_digest = execution_digest(manifest.logical_result_identity);
    let identity_bytes = read_verified(source, identity_digest)?;
    enforce_object_length(
        "meshing result identity",
        identity_bytes.len() as u64,
        limits,
    )?;
    add_inventory_bytes(
        "meshing stage",
        &mut total_bytes,
        identity_bytes.len() as u64,
        limits,
    )?;
    let result_identity = MeshingStageResultIdentity::canonical_decode(&identity_bytes)?;
    if result_identity.canonical_digest()? != manifest.logical_result_identity {
        return Err(MeshingExecutionError::Identity(
            "result identity bytes differ from manifest identity",
        ));
    }

    let mut chunks = Vec::with_capacity(manifest.chunks.len());
    for descriptor in &manifest.chunks {
        enforce_object_length("meshing chunk", descriptor.encoded_length, limits)?;
        add_inventory_bytes(
            "meshing stage",
            &mut total_bytes,
            descriptor.encoded_length,
            limits,
        )?;
        let digest = execution_digest(descriptor.digest);
        let bytes = read_exact(source, digest, descriptor.encoded_length)?;
        chunks.push(EncodedMeshingChunk {
            descriptor: descriptor.clone(),
            bytes,
        });
    }
    let prepared = prepare_stage_objects(result_identity, manifest, chunks, limits)?;
    if prepared.root.digest != root.digest || prepared.root.encoded_length != root.encoded_length {
        return Err(MeshingExecutionError::Identity(
            "imported manifest differs from requested root",
        ));
    }
    Ok(prepared)
}

fn execution_digest(digest: StableDigest) -> Digest {
    Digest::from_bytes(*digest.bytes())
}

fn stable_digest(digest: Digest) -> StableDigest {
    StableDigest::from_bytes(*digest.bytes())
}
