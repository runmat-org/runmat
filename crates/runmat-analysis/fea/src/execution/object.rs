use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use super::{AnalysisExecutionError, AnalysisExecutionResult};

pub const DEFAULT_ANALYSIS_CHUNK_BYTES: usize = 4 * 1024 * 1024;
pub const ANALYSIS_MANIFEST_MEDIA_TYPE: &str =
    "application/vnd.runmat.analysis-object-manifest+cbor";
pub const ANALYSIS_CHUNK_MEDIA_TYPE: &str = "application/vnd.runmat.analysis-object-chunk";
pub const ANALYSIS_MANIFEST_VALUE_SCHEMA: &str = "runmat.analysis.object-manifest.v1";
pub const ANALYSIS_CHUNK_VALUE_SCHEMA: &str = "runmat.analysis.object-chunk.v1";
const MANIFEST_PREFIX: &[u8] = b"runmat-analysis-object-manifest/v1\0";
const MANIFEST_DOMAIN: &str = "numerical-object-manifest";
const MANIFEST_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(4096, 32);
pub const ANALYSIS_MANIFEST_MAX_ENCODED_BYTES: u64 = 8 * 1024 * 1024;
/// Schema bound of the one output-set root a workload returns. The root lists output roots only;
/// numerical payloads are bounded separately by each phase's prepared output facts.
pub const ANALYSIS_OUTPUT_SET_MAX_ENCODED_BYTES: u64 = 8 * 1024 * 1024;
const MAX_CHUNKS: usize = 1_000_000;

mod incremental;
pub use incremental::AnalysisPayloadBuilder;
mod transport_profile;
pub use transport_profile::{AnalysisTransportProfile, AnalysisTransportProfiler};
mod plan;
mod serialization;
pub use plan::{AnalysisPayloadPartitionPlan, AnalysisPayloadPlan};

#[cfg(test)]
#[path = "object_tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisNumericalObjectKind {
    NonlinearSolidWorkload,
    NonlinearSolidBlockRequest,
    NonlinearSolidBlockArtifact,
    NonlinearSolidStudyCompletion,
    FluidSteadyWorkload,
    FluidTransientBlockWorkload,
    FluidTransientCheckpoint,
    FluidStudyCompletion,
    AcousticStudyCompletion,
    AcousticResponseStudyCompletion,
    ElectricalStudyCompletion,
    MagneticStudyCompletion,
    MaxwellStudyCompletion,
    ThermalStudyCompletion,
    ThermalWorkload,
    FieldSourceBinding,
    FrequencySamplePublication,
    AcousticHarmonicWorkload,
    AcousticFrequencyWorkload,
    AcousticResponseWorkload,
    AcousticFrequencyAssemblyRequest,
    ElectricalStaticWorkload,
    MagnetostaticWorkload,
    MaxwellHarmonicWorkload,
    Mesh,
    Signal,
    ResponseSpectrum,
    RandomSpectrum,
    RealizedExcitation,
    TopologyResolution,
    SolverInvocation,
    SolverResult,
    NumericalReceipt,
    StatePublicationProfiles,
    FieldRecoveryRequest,
    ObservationEvaluationRequest,
    ObservationFieldCatalog,
    ObservationPartitionEvaluationRequest,
    ObservationPartitionReductionRequest,
    ObservationAssemblyRequest,
    AccuracyEvaluationRequest,
    AccuracySourceContext,
    PublicationRequest,
    StatePublicationRequest,
    ObservationHistoryCheckpointRequest,
    ObservationHistoryFinalizationRequest,
    SeriesReductionResult,
    StructuralTransientCheckpointScope,
    StructuralTransientBlockRequest,
    StructuralTransientResultAssemblyRequest,
    StructuralTransientWorkload,
    StructuralTransientSegment,
    StructuralFrequencyWorkload,
    StructuralFrequencyEntryRequest,
    StructuralFrequencyEntryResult,
    StructuralFrequencyResultAssemblyRequest,
    StructuralModalWorkload,
    StructuralModalResponseSpectrumWorkload,
    StructuralModalResponseSpectrumDirectionRequest,
    StructuralModalResponseSpectrumDirectionResult,
    StructuralModalResponseSpectrumAssemblyRequest,
    StructuralModalResponseSpectrumFieldRecovery,
    StructuralRandomResponseWorkload,
    StructuralRandomResponseIntervalRequest,
    StructuralRandomResponseIntervalResult,
    StructuralRandomResponseAssemblyRequest,
    StructuralBucklingWorkload,
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

#[derive(Clone, Debug, Eq, PartialEq, Deserialize)]
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
    /// Admit the canonical single partition used for a whole numerical payload. A whole
    /// payload still has a partition descriptor; its identity must match the object root.
    pub fn whole_payload_partition(
        &self,
    ) -> AnalysisExecutionResult<&AnalysisPayloadPartitionDescriptor> {
        self.validate()?;
        let [partition] = self.partitions.as_slice() else {
            return Err(AnalysisExecutionError::Invalid(
                "whole payload requires exactly one partition".into(),
            ));
        };
        if partition.logical_identity != self.logical_identity
            || partition.payload_digest != self.payload_digest
        {
            return Err(AnalysisExecutionError::Identity(
                "whole payload partition differs from its object root",
            ));
        }
        // validate() establishes complete contiguous chunk coverage and equal byte lengths.
        Ok(partition)
    }

    /// Requested Rust storage for the borrowed identity sort buffer, excluding allocator overhead.
    pub fn validation_scratch_bytes(partition_count: usize) -> AnalysisExecutionResult<u64> {
        partition_count
            .checked_mul(std::mem::size_of::<&Digest>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(AnalysisExecutionError::Limit(
                "manifest validation scratch overflow",
            ))
    }

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
        Self::validation_scratch_bytes(self.partitions.len())?;
        let mut partition_identities = Vec::new();
        partition_identities
            .try_reserve_exact(self.partitions.len())
            .map_err(|_| {
                AnalysisExecutionError::Limit("cannot reserve manifest validation scratch")
            })?;
        for partition in &self.partitions {
            if partition.ordinal != expected_partition_ordinal
                || partition.first_chunk_ordinal != expected_first_chunk
                || partition.encoded_length == 0
                || partition.chunk_count == 0
            {
                return Err(AnalysisExecutionError::Invalid(
                    "payload partitions must be non-empty, unique, contiguous, and ordered".into(),
                ));
            }
            partition_identities.push(&partition.logical_identity);
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
        partition_identities.sort_unstable();
        if partition_identities
            .windows(2)
            .any(|pair| pair[0] == pair[1])
        {
            return Err(AnalysisExecutionError::Invalid(
                "payload partitions must be non-empty, unique, contiguous, and ordered".into(),
            ));
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
        self.manifest.validate()?;
        if chunk_bytes == 0 || self.chunks.len() != self.manifest.chunks.len() {
            return Err(AnalysisExecutionError::Invalid(
                "invalid chunk byte limit or payload chunk inventory".into(),
            ));
        }
        let mut payload_hash = Sha256::new();
        for partition in &self.manifest.partitions {
            // Manifest validation proves these intervals cover the inventory exactly.
            let start = partition.first_chunk_ordinal as usize;
            let end = start + partition.chunk_count as usize;
            let mut partition_hash = Sha256::new();
            for index in start..end {
                let bytes = &self.chunks[index];
                let descriptor = &self.manifest.chunks[index];
                if bytes.len() > chunk_bytes
                    || (index + 1 < end && bytes.len() != chunk_bytes)
                    || bytes.len() as u64 != descriptor.encoded_length
                    || Digest::sha256(bytes) != descriptor.digest
                {
                    return Err(AnalysisExecutionError::Identity(
                        "prepared numerical chunk differs from its canonical descriptor",
                    ));
                }
                partition_hash.update(bytes);
                payload_hash.update(bytes);
            }
            if Digest::from_bytes(partition_hash.finalize().into()) != partition.payload_digest {
                return Err(AnalysisExecutionError::Identity(
                    "prepared numerical partition differs from its payload identity",
                ));
            }
        }
        if Digest::from_bytes(payload_hash.finalize().into()) != self.manifest.payload_digest {
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
    AnalysisTransportProfiler::new(chunk_bytes)?;
    let mut entries = AnalysisPayloadPlan::reserve_partitions(partitions.len())?;
    for partition in partitions {
        entries.push(AnalysisPayloadPartitionPlan {
            logical_identity: partition.logical_identity,
            encoded_bytes_upper_bound: u64::try_from(partition.payload.len())
                .map_err(|_| AnalysisExecutionError::Limit("partition byte length"))?,
        });
    }
    let plan = AnalysisPayloadPlan::new(kind, chunk_bytes, entries)?;
    let mut builder = AnalysisPayloadBuilder::new(logical_identity, plan)?;
    for partition in partitions {
        builder.push(*partition)?;
    }
    builder.finish()
}
