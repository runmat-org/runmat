//! Original input reads and canonical materialization within a carried task reservation.
//! Conservative requested-storage admission; allocator overhead and RSS are separate.
use super::*;
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};
use runmat_execution_artifact::ProgramExecutionRequest;
use std::cell::Cell;

pub(crate) struct AnalysisInputAdmission {
    maximum: u64,
    retained: Cell<u64>,
}
impl AnalysisInputAdmission {
    pub(crate) fn new(
        request: &ProgramExecutionRequest,
        maximum: u64,
    ) -> AnalysisBridgeResult<Self> {
        let admission = Self {
            maximum,
            retained: Cell::new(0),
        };
        let profile = profile_serialized(request, CanonicalLimits::platform_content(65_536, 64))?;
        admission.charge(profile.materialization_resident_bytes_estimate)?;
        Ok(admission)
    }
    pub(crate) fn require_additional(&self, bytes: u64) -> AnalysisBridgeResult<()> {
        if runmat_analysis_fea::is_fea_cancelled() {
            return Err(runmat_analysis_fea::contracts::FeaRunError::Cancelled.into());
        }
        let required = self.retained.get().checked_add(bytes).ok_or(
            runmat_analysis_fea::planning::resource::ResourcePlanningError::Overflow(
                "original analysis input reservation",
            ),
        )?;
        if required > self.maximum {
            return Err(
                runmat_analysis_fea::planning::resource::ResourcePlanningError::AccuracyCeiling {
                    resource: "original analysis input execution reservation",
                    required,
                    maximum: self.maximum,
                }
                .into(),
            );
        }
        Ok(())
    }
    pub(crate) fn charge(&self, bytes: u64) -> AnalysisBridgeResult<()> {
        self.require_additional(bytes)?;
        self.retained.set(self.retained.get() + bytes);
        Ok(())
    }
    pub(crate) fn read(
        &self,
        source: &impl CacheImport,
        digest: Digest,
        length: u64,
    ) -> AnalysisBridgeResult<Option<Vec<u8>>> {
        self.require_additional(length)?;
        let bytes = source.read_verified_bounded(digest, length)?;
        if let Some(bytes) = &bytes {
            if bytes.len() as u64 != length {
                return Err(AnalysisBridgeError::Identity(
                    "bounded analysis input length differs from its reference",
                ));
            }
            self.charge(bytes.capacity() as u64)?;
        }
        Ok(bytes)
    }
    /// Profile without constructing a value tree, then admit canonical verification/typed decode.
    pub(crate) fn admit_canonical_decode(&self, bytes: &[u8]) -> AnalysisBridgeResult<()> {
        let bytes =
            runmat_canonical_codec::canonical_chunked_payload_typed_resident_bytes_estimate(
                std::iter::once(bytes),
                &ResourceProfileControl,
            )?;
        self.charge(bytes)
    }
    pub(crate) fn admit_payload_decoding(
        &self,
        inputs: &[PreparedAnalysisObjects],
    ) -> AnalysisBridgeResult<()> {
        // The structural scanner authenticates each original partition without joining chunks.
        // Encoded bytes are already charged per read occurrence, including repeated digests.
        let facts = prepared_input_resource_facts(inputs)?;
        self.require_additional(facts.decoded_resident_bytes_admission_budget.0)
    }
}

pub(crate) fn read_input(
    source: &impl CacheImport,
    digest: Digest,
    length: u64,
    admission: Option<&AnalysisInputAdmission>,
) -> AnalysisBridgeResult<Option<Vec<u8>>> {
    match admission {
        Some(admission) => admission.read(source, digest, length),
        None => Ok(source.read_verified(digest)?),
    }
}
