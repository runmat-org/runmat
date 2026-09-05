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
    Field,
    SemanticState,
    History,
    Observation,
    AccuracyEvidence,
    Operator,
    Checkpoint,
    ConstraintPlan,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisChunkDescriptor {
    pub ordinal: u32,
    pub digest: Digest,
    pub encoded_length: u64,
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
}

impl AnalysisObjectManifest {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 || self.chunks.is_empty() || self.chunks.len() > MAX_CHUNKS {
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
        let payload = self
            .chunks
            .iter()
            .flat_map(|chunk| chunk.iter().copied())
            .collect::<Vec<_>>();
        if chunk_numerical_payload(
            self.manifest.kind,
            self.manifest.logical_identity,
            &payload,
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
    if payload.is_empty() {
        return Err(AnalysisExecutionError::Invalid(
            "numerical object payload is empty".into(),
        ));
    }
    if chunk_bytes == 0 {
        return Err(AnalysisExecutionError::Invalid(
            "chunk byte limit must be non-zero".into(),
        ));
    }
    let chunk_count = payload.len().div_ceil(chunk_bytes);
    if chunk_count > MAX_CHUNKS || chunk_count > u32::MAX as usize {
        return Err(AnalysisExecutionError::Limit(
            "numerical object chunk count",
        ));
    }
    let mut payload_chunks = Vec::with_capacity(chunk_count);
    let mut chunks = Vec::with_capacity(chunk_count);
    for (ordinal, bytes) in payload.chunks(chunk_bytes).enumerate() {
        chunks.push(AnalysisChunkDescriptor {
            ordinal: ordinal as u32,
            digest: Digest::sha256(bytes),
            encoded_length: bytes.len() as u64,
        });
        payload_chunks.push(bytes.to_vec());
    }
    let manifest = AnalysisObjectManifest {
        schema_version: 1,
        kind,
        logical_identity,
        payload_digest: Digest::sha256(payload),
        encoded_length: payload.len() as u64,
        chunks,
    };
    Ok(ChunkedAnalysisPayload {
        manifest,
        chunks: payload_chunks,
    })
}
