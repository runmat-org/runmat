use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};
use runmat_execution_artifact::{ArtifactError, LogicalObject, ObjectDescriptor, ObjectNamespace};
use runmat_meshing_core::{
    verify_stage_manifest_closure, CanonicalMeshingContract, EncodedMeshingChunkV2,
    MeshingStageManifestV2, MeshingStageResultIdentityV2, StableDigest,
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
    pub result_identity: MeshingStageResultIdentityV2,
    pub manifest: MeshingStageManifestV2,
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
        let mut chunks = Vec::with_capacity(self.manifest.chunks.len());
        for descriptor in &self.manifest.chunks {
            let digest = execution_digest(descriptor.digest);
            let object = self
                .objects
                .iter()
                .find(|object| object.descriptor.digest == digest)
                .ok_or(MeshingExecutionError::MissingObject(digest))?;
            chunks.push(EncodedMeshingChunkV2 {
                descriptor: descriptor.clone(),
                bytes: object.bytes.clone(),
            });
        }
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
}

pub fn prepare_stage_objects(
    result_identity: MeshingStageResultIdentityV2,
    manifest: MeshingStageManifestV2,
    chunks: Vec<EncodedMeshingChunkV2>,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedMeshingStageObjects> {
    verify_stage_manifest_closure(&manifest, &result_identity, &chunks)?;

    let mut objects = Vec::with_capacity(chunks.len() + 2);
    let identity_bytes = result_identity.canonical_encode()?;
    objects.push(logical_object(
        "identities",
        ObjectNamespace::ResultValue,
        MESHING_RESULT_IDENTITY_MEDIA_TYPE,
        identity_bytes,
    )?);
    for chunk in chunks {
        let object = logical_object(
            "chunks",
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
        "manifests",
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
    let manifest = MeshingStageManifestV2::canonical_decode(&manifest_bytes)?;
    let object_count = manifest.chunks.len().checked_add(2).ok_or_else(|| {
        ArtifactError::Limit("meshing stage object inventory count overflow".into())
    })?;
    if object_count > limits.max_objects {
        return Err(ArtifactError::Limit("too many meshing stage objects".into()).into());
    }
    let mut total_bytes = root.encoded_length;
    let identity_digest = execution_digest(manifest.logical_result_identity);
    let identity_bytes = read_unbounded(source, identity_digest)?;
    enforce_object_length("result identity", identity_bytes.len() as u64, limits)?;
    add_inventory_bytes(&mut total_bytes, identity_bytes.len() as u64, limits)?;
    let result_identity = MeshingStageResultIdentityV2::canonical_decode(&identity_bytes)?;
    if result_identity.canonical_digest()? != manifest.logical_result_identity {
        return Err(MeshingExecutionError::Identity(
            "result identity bytes differ from manifest identity",
        ));
    }

    let mut chunks = Vec::with_capacity(manifest.chunks.len());
    for descriptor in &manifest.chunks {
        enforce_object_length("chunk", descriptor.encoded_length, limits)?;
        add_inventory_bytes(&mut total_bytes, descriptor.encoded_length, limits)?;
        let digest = execution_digest(descriptor.digest);
        let bytes = read_exact(source, digest, descriptor.encoded_length)?;
        chunks.push(EncodedMeshingChunkV2 {
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

fn logical_object(
    class: &str,
    namespace: ObjectNamespace,
    media_type: &str,
    bytes: Vec<u8>,
) -> MeshingExecutionResult<LogicalObject> {
    let digest = Digest::sha256(&bytes);
    LogicalObject::new(
        namespace,
        format!("meshing/v2/{class}/{}", digest_hex(digest)),
        media_type,
        bytes,
    )
    .map_err(Into::into)
}

fn read_exact(
    source: &impl CacheImport,
    digest: Digest,
    encoded_length: u64,
) -> MeshingExecutionResult<Vec<u8>> {
    let bytes = read_unbounded(source, digest)?;
    if bytes.len() as u64 != encoded_length {
        return Err(MeshingExecutionError::Identity(
            "object length differs from its descriptor",
        ));
    }
    Ok(bytes)
}

fn read_unbounded(source: &impl CacheImport, digest: Digest) -> MeshingExecutionResult<Vec<u8>> {
    let bytes = source
        .read_verified(digest)?
        .ok_or(MeshingExecutionError::MissingObject(digest))?;
    if Digest::sha256(&bytes) != digest {
        return Err(MeshingExecutionError::Identity(
            "cache returned bytes under the wrong digest",
        ));
    }
    Ok(bytes)
}

fn enforce_object_length(
    class: &str,
    encoded_length: u64,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<()> {
    if encoded_length > limits.max_object_bytes {
        return Err(ArtifactError::Limit(format!("meshing {class} object is too large")).into());
    }
    Ok(())
}

fn add_inventory_bytes(
    total: &mut u64,
    encoded_length: u64,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<()> {
    *total = total.checked_add(encoded_length).ok_or_else(|| {
        ArtifactError::Limit("meshing stage object inventory size overflow".into())
    })?;
    if *total > limits.max_total_bytes {
        return Err(
            ArtifactError::Limit("meshing stage object inventory is too large".into()).into(),
        );
    }
    Ok(())
}

fn execution_digest(digest: StableDigest) -> Digest {
    Digest::from_bytes(*digest.bytes())
}

fn stable_digest(digest: Digest) -> StableDigest {
    StableDigest::from_bytes(*digest.bytes())
}

fn digest_hex(digest: Digest) -> String {
    digest
        .bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
