use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{AnalysisExecutionError, AnalysisExecutionResult};

pub const DEFAULT_ANALYSIS_CHUNK_BYTES: usize = 4 * 1024 * 1024;
pub const ANALYSIS_MANIFEST_MEDIA_TYPE: &str =
    "application/vnd.runmat.analysis-object-manifest+cbor";
pub const ANALYSIS_CHUNK_MEDIA_TYPE: &str = "application/vnd.runmat.analysis-object-chunk";
pub const ANALYSIS_MANIFEST_VALUE_SCHEMA: &str = "runmat.analysis.object-manifest.v1";
pub const ANALYSIS_CHUNK_VALUE_SCHEMA: &str = "runmat.analysis.object-chunk.v1";
const MANIFEST_PREFIX: &[u8] = b"runmat-analysis-object-manifest/v1\0";
const MANIFEST_DOMAIN: &str = "numerical-object-manifest";
const MANIFEST_LIMITS: CanonicalLimits = CanonicalLimits::new(8 * 1024 * 1024, 1_000_000, 4096, 32);
const MAX_CHUNKS: usize = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisNumericalObjectKind {
    Mesh,
    Signal,
    ResponseSpectrum,
    RealizedExcitation,
    BoundaryResolution,
    SolverInvocation,
    SolverResult,
    NumericalReceipt,
    FieldRecoveryRequest,
    ObservationEvaluationRequest,
    ObservationPartitionEvaluationRequest,
    ObservationPartitionReductionRequest,
    ObservationAssemblyRequest,
    AccuracyEvaluationRequest,
    PublicationRequest,
    StatePublicationRequest,
    ObservationHistoryCheckpointRequest,
    StructuralTransientBlockRequest,
    StructuralTransientResultAssemblyRequest,
    StructuralTransientWorkload,
    StructuralTransientSegment,
    Field,
    FieldSelection,
    FieldPartition,
    SemanticState,
    History,
    Observation,
    ObservationCatalog,
    ObservationPartitionPlan,
    ObservationPartitionContribution,
    ObservationResult,
    AccuracyEvidence,
    AccuracyReport,
    PublishedResult,
    Operator,
    ObservationHistoryCheckpoint,
    StructuralTransientCheckpoint,
    ConstraintPlan,
    ResourcePlan,
    ResourceUsage,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisChunkDescriptor {
    pub ordinal: u32,
    pub digest: Digest,
    pub encoded_length: u64,
}

/// One independently addressable semantic payload within a numerical object.
///
/// Chunk boundaries remain a transport choice. This descriptor binds an analysis-owned logical
/// member to the contiguous chunk interval that carries it, allowing a reader to verify and load
/// that member without gathering unrelated members.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisPayloadPartitionDescriptor {
    pub ordinal: u32,
    pub logical_identity: Digest,
    pub payload_digest: Digest,
    pub encoded_length: u64,
    pub first_chunk_ordinal: u32,
    pub chunk_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AnalysisPayloadPartition<'a> {
    pub logical_identity: Digest,
    pub payload: &'a [u8],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisObjectManifest {
    pub schema_version: u16,
    pub kind: AnalysisNumericalObjectKind,
    pub logical_identity: Digest,
    pub payload_digest: Digest,
    pub encoded_length: u64,
    pub chunks: Vec<AnalysisChunkDescriptor>,
    pub partitions: Vec<AnalysisPayloadPartitionDescriptor>,
}

impl AnalysisObjectManifest {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1
            || self.chunks.is_empty()
            || self.chunks.len() > MAX_CHUNKS
            || self.partitions.is_empty()
            || self.partitions.len() > MAX_CHUNKS
        {
            return Err(AnalysisExecutionError::Invalid(
                "unsupported schema or invalid chunk count".into(),
            ));
        }
        let mut expected_ordinal = 0_u32;
        let mut total = 0_u64;
        for chunk in &self.chunks {
            if chunk.ordinal != expected_ordinal || chunk.encoded_length == 0 {
                return Err(AnalysisExecutionError::Invalid(
                    "chunks must be non-empty and have contiguous ordinals".into(),
                ));
            }
            expected_ordinal = expected_ordinal
                .checked_add(1)
                .ok_or(AnalysisExecutionError::Limit("chunk ordinal overflow"))?;
            total = total
                .checked_add(chunk.encoded_length)
                .ok_or(AnalysisExecutionError::Limit("object byte length overflow"))?;
        }
        if total != self.encoded_length {
            return Err(AnalysisExecutionError::Identity(
                "manifest byte length differs from its chunks",
            ));
        }
        let mut expected_partition_ordinal = 0_u32;
        let mut expected_first_chunk = 0_u32;
        let mut partition_bytes = 0_u64;
        let mut partition_identities = std::collections::BTreeSet::new();
        for partition in &self.partitions {
            if partition.ordinal != expected_partition_ordinal
                || partition.first_chunk_ordinal != expected_first_chunk
                || partition.encoded_length == 0
                || partition.chunk_count == 0
                || !partition_identities.insert(partition.logical_identity)
            {
                return Err(AnalysisExecutionError::Invalid(
                    "payload partitions must be non-empty, unique, contiguous, and ordered".into(),
                ));
            }
            let end = partition
                .first_chunk_ordinal
                .checked_add(partition.chunk_count)
                .ok_or(AnalysisExecutionError::Limit("partition chunk interval"))?;
            if end as usize > self.chunks.len() {
                return Err(AnalysisExecutionError::Invalid(
                    "payload partition exceeds the numerical object chunk inventory".into(),
                ));
            }
            let described_bytes = self.chunks[partition.first_chunk_ordinal as usize..end as usize]
                .iter()
                .try_fold(0_u64, |sum, chunk| sum.checked_add(chunk.encoded_length))
                .ok_or(AnalysisExecutionError::Limit("partition byte length"))?;
            if described_bytes != partition.encoded_length {
                return Err(AnalysisExecutionError::Identity(
                    "payload partition byte length differs from its chunks",
                ));
            }
            partition_bytes = partition_bytes
                .checked_add(partition.encoded_length)
                .ok_or(AnalysisExecutionError::Limit(
                    "partitioned object byte length",
                ))?;
            expected_partition_ordinal = expected_partition_ordinal
                .checked_add(1)
                .ok_or(AnalysisExecutionError::Limit("partition ordinal overflow"))?;
            expected_first_chunk = end;
        }
        if expected_first_chunk as usize != self.chunks.len()
            || partition_bytes != self.encoded_length
        {
            return Err(AnalysisExecutionError::Identity(
                "payload partitions do not cover the complete numerical object",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            MANIFEST_PREFIX,
            MANIFEST_DOMAIN,
            self,
            MANIFEST_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let manifest: Self =
            decode_contract(MANIFEST_PREFIX, MANIFEST_DOMAIN, bytes, MANIFEST_LIMITS)?;
        manifest.validate()?;
        Ok(manifest)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkedAnalysisPayload {
    pub manifest: AnalysisObjectManifest,
    pub chunks: Vec<Vec<u8>>,
}

impl ChunkedAnalysisPayload {
    pub fn revalidate(&self, chunk_bytes: usize) -> AnalysisExecutionResult<()> {
        let payloads = self
            .manifest
            .partitions
            .iter()
            .map(|partition| {
                let start = partition.first_chunk_ordinal as usize;
                let end = start + partition.chunk_count as usize;
                self.chunks[start..end]
                    .iter()
                    .flat_map(|chunk| chunk.iter().copied())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let partitions = self
            .manifest
            .partitions
            .iter()
            .zip(&payloads)
            .map(|(partition, payload)| AnalysisPayloadPartition {
                logical_identity: partition.logical_identity,
                payload,
            })
            .collect::<Vec<_>>();
        if chunk_partitioned_numerical_payload(
            self.manifest.kind,
            self.manifest.logical_identity,
            &partitions,
            chunk_bytes,
        )? != *self
        {
            return Err(AnalysisExecutionError::Identity(
                "prepared numerical object is not its canonical closure",
            ));
        }
        Ok(())
    }
}

pub fn chunk_numerical_payload(
    kind: AnalysisNumericalObjectKind,
    logical_identity: Digest,
    payload: &[u8],
    chunk_bytes: usize,
) -> AnalysisExecutionResult<ChunkedAnalysisPayload> {
    chunk_partitioned_numerical_payload(
        kind,
        logical_identity,
        &[AnalysisPayloadPartition {
            logical_identity,
            payload,
        }],
        chunk_bytes,
    )
}

pub fn chunk_partitioned_numerical_payload(
    kind: AnalysisNumericalObjectKind,
    logical_identity: Digest,
    partitions: &[AnalysisPayloadPartition<'_>],
    chunk_bytes: usize,
) -> AnalysisExecutionResult<ChunkedAnalysisPayload> {
    if partitions.is_empty()
        || partitions
            .iter()
            .any(|partition| partition.payload.is_empty())
    {
        return Err(AnalysisExecutionError::Invalid(
            "numerical object partitions must be non-empty".into(),
        ));
    }
    if chunk_bytes == 0 {
        return Err(AnalysisExecutionError::Invalid(
            "chunk byte limit must be non-zero".into(),
        ));
    }
    let chunk_count = partitions
        .iter()
        .try_fold(0_usize, |count, partition| {
            count.checked_add(partition.payload.len().div_ceil(chunk_bytes))
        })
        .ok_or(AnalysisExecutionError::Limit(
            "numerical object chunk count",
        ))?;
    if chunk_count > MAX_CHUNKS || chunk_count > u32::MAX as usize {
        return Err(AnalysisExecutionError::Limit(
            "numerical object chunk count",
        ));
    }
    let mut payload_chunks = Vec::with_capacity(chunk_count);
    let mut chunks = Vec::with_capacity(chunk_count);
    let mut descriptors = Vec::with_capacity(partitions.len());
    let mut complete_payload = Vec::new();
    let mut identities = std::collections::BTreeSet::new();
    for (partition_ordinal, partition) in partitions.iter().enumerate() {
        if !identities.insert(partition.logical_identity) {
            return Err(AnalysisExecutionError::Invalid(
                "numerical object partition identities must be unique".into(),
            ));
        }
        let first_chunk_ordinal = chunks.len() as u32;
        for bytes in partition.payload.chunks(chunk_bytes) {
            chunks.push(AnalysisChunkDescriptor {
                ordinal: chunks.len() as u32,
                digest: Digest::sha256(bytes),
                encoded_length: bytes.len() as u64,
            });
            payload_chunks.push(bytes.to_vec());
        }
        descriptors.push(AnalysisPayloadPartitionDescriptor {
            ordinal: partition_ordinal as u32,
            logical_identity: partition.logical_identity,
            payload_digest: Digest::sha256(partition.payload),
            encoded_length: partition.payload.len() as u64,
            first_chunk_ordinal,
            chunk_count: chunks.len() as u32 - first_chunk_ordinal,
        });
        complete_payload.extend_from_slice(partition.payload);
    }
    let manifest = AnalysisObjectManifest {
        schema_version: 1,
        kind,
        logical_identity,
        payload_digest: Digest::sha256(&complete_payload),
        encoded_length: complete_payload.len() as u64,
        chunks,
        partitions: descriptors,
    };
    Ok(ChunkedAnalysisPayload {
        manifest,
        chunks: payload_chunks,
    })
}
