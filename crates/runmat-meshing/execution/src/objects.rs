mod storage;
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};
use runmat_execution_artifact::{ArtifactError, LogicalObject, ObjectDescriptor, ObjectNamespace};
use runmat_meshing_core::{
    verify_stage_manifest_closure, CanonicalMeshingContract, EncodedMeshingChunk,
    MeshingCancellationSignal, MeshingStageManifest, MeshingStageResultIdentity, StableDigest,
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
        validate_inventory(&self.objects, limits)?;
        let chunks = self.borrowed_chunks()?;
        runmat_meshing_core::verify_stage_manifest_closure_borrowed(
            &self.manifest,
            &self.result_identity,
            &chunks,
            &runmat_meshing_core::NeverCancelled,
        )?;
        // Canonical names, ordering and bytes remain part of the closure. Compare
        // descriptors against borrowed payloads rather than rebuilding owned objects.
        let identity = self.result_identity.canonical_encode()?;
        self.check_object(
            0,
            "meshing/v2/identities",
            MESHING_RESULT_IDENTITY_MEDIA_TYPE,
            &identity,
        )?;
        for (index, chunk) in chunks.iter().enumerate() {
            self.check_object(
                index + 1,
                "meshing/v2/chunks",
                chunk.descriptor.media_type.media_type(),
                chunk.bytes,
            )?;
        }
        let manifest = self.manifest.canonical_encode()?;
        self.check_object(
            self.objects.len() - 1,
            "meshing/v2/manifests",
            MESHING_STAGE_MANIFEST_MEDIA_TYPE,
            &manifest,
        )?;
        if self.root
            != self
                .objects
                .last()
                .expect("validated closure is nonempty")
                .descriptor
        {
            return Err(MeshingExecutionError::Identity(
                "prepared manifest root differs from its closure",
            ));
        }
        Ok(())
    }

    fn check_object(
        &self,
        index: usize,
        prefix: &str,
        media: &str,
        bytes: &[u8],
    ) -> MeshingExecutionResult<()> {
        let digest = Digest::sha256(bytes);
        let expected = ObjectDescriptor::new(
            ObjectNamespace::ResultValue,
            format!("{prefix}/{}", crate::object_support::digest_hex(digest)),
            media,
            bytes,
        )?;
        let actual = &self.objects[index];
        if actual.descriptor != expected || actual.bytes != bytes {
            return Err(MeshingExecutionError::Identity(
                "prepared meshing object set is not its canonical closure",
            ));
        }
        Ok(())
    }

    pub fn decoded_streams(
        &self,
    ) -> MeshingExecutionResult<Vec<runmat_meshing_core::MeshingChunkStream>> {
        self.decoded_streams_with_cancellation(&runmat_meshing_core::NeverCancelled)
    }

    pub fn decoded_streams_with_cancellation(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> MeshingExecutionResult<Vec<runmat_meshing_core::MeshingChunkStream>> {
        self.decoded_streams_with_admission(cancellation, |_| Ok(()))
    }

    /// Admission covers borrowed lookup workspace as well as reconstructed data.
    /// The caller retains its charges until the decoded records are released.
    pub fn decoded_streams_with_admission(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
        mut admit: impl FnMut(u64) -> MeshingExecutionResult<()>,
    ) -> MeshingExecutionResult<Vec<runmat_meshing_core::MeshingChunkStream>> {
        ensure_active(cancellation)?;
        let lookup = self
            .objects
            .len()
            .checked_mul(std::mem::size_of::<(&Digest, &LogicalObject)>())
            .and_then(|bytes| {
                self.manifest
                    .chunks
                    .len()
                    .checked_mul(std::mem::size_of::<
                        runmat_meshing_core::EncodedMeshingChunkRef<'_>,
                    >())
                    .and_then(|chunks| bytes.checked_add(chunks))
            })
            .ok_or_else(|| ArtifactError::Limit("chunk lookup storage overflow".into()))?;
        admit(lookup as u64)?;
        let chunks = self.borrowed_chunks()?;
        let mut failure = None;
        let result = runmat_meshing_core::decode_stage_manifest_streams_borrowed_with_admission(
            &self.manifest,
            &self.result_identity,
            &chunks,
            cancellation,
            |profile| {
                let bytes = profile.requested_bytes()?;
                admit(bytes).map_err(|error| {
                    let detail = error.to_string();
                    failure = Some(error);
                    runmat_meshing_core::MeshingContractError::invalid("stream admission", detail)
                })
            },
        );
        if let Some(error) = failure {
            return Err(error);
        }
        Ok(result?)
    }

    fn borrowed_chunks(
        &self,
    ) -> MeshingExecutionResult<Vec<runmat_meshing_core::EncodedMeshingChunkRef<'_>>> {
        let mut indexed = Vec::with_capacity(self.objects.len());
        indexed.extend(
            self.objects
                .iter()
                .map(|object| (&object.descriptor.digest, object)),
        );
        indexed.sort_unstable_by_key(|(digest, _)| **digest);
        if self.objects.len()
            != self
                .manifest
                .chunks
                .len()
                .checked_add(2)
                .ok_or_else(|| ArtifactError::Limit("meshing object count overflow".into()))?
        {
            for descriptor in &self.manifest.chunks {
                let digest = execution_digest(descriptor.digest);
                if indexed
                    .binary_search_by_key(&digest, |(digest, _)| **digest)
                    .is_err()
                {
                    return Err(MeshingExecutionError::MissingObject(digest));
                }
            }
            return Err(MeshingExecutionError::Identity(
                "prepared object inventory differs from its manifest",
            ));
        }
        if indexed.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(MeshingExecutionError::Identity(
                "prepared object inventory contains duplicate identities",
            ));
        }
        let mut chunks = Vec::with_capacity(self.manifest.chunks.len());
        for descriptor in &self.manifest.chunks {
            let digest = execution_digest(descriptor.digest);
            let index = indexed
                .binary_search_by_key(&digest, |(digest, _)| **digest)
                .map_err(|_| MeshingExecutionError::MissingObject(digest))?;
            chunks.push(runmat_meshing_core::EncodedMeshingChunkRef {
                descriptor,
                bytes: &indexed[index].1.bytes,
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

    Ok(validated::assemble(result_identity, manifest, chunks, limits, &mut |_| Ok(()))?.into_raw())
}

pub fn import_stage_objects(
    source: &impl CacheImport,
    root: MeshingStageObjectRoot,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedMeshingStageObjects> {
    Ok(import_stage_objects_with_admission(
        source,
        root,
        limits,
        &runmat_meshing_core::NeverCancelled,
        &mut |_| Ok(()),
    )?
    .into_raw())
}

fn execution_digest(digest: StableDigest) -> Digest {
    Digest::from_bytes(*digest.bytes())
}

mod validated;
pub(crate) use validated::{prepare_closed_stage_objects, ValidatedStageObjects};

pub(crate) fn revalidate_stage_objects(
    objects: PreparedMeshingStageObjects,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<ValidatedStageObjects> {
    objects.revalidate(limits)?;
    Ok(ValidatedStageObjects::from_revalidated(objects))
}

mod import;
pub(crate) use import::import_stage_objects_with_admission;
