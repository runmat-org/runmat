//! Static producer catalog reservation shared by portable and numerical preflight.
use super::*;
use crate::execution::{
    structural_state_publication_profiles_encoding_profile, SemanticStatePublicationSource,
    ANALYSIS_MANIFEST_MAX_ENCODED_BYTES,
};
use runmat_analysis_core::{AnalysisModel, ProcedureStep};
use runmat_execution::Digest;

#[cfg(test)]
mod tests;

/// Shared chunk-policy-independent envelope. Manifest limits bound every admitted chunk
/// inventory; canonical materialization accounting is owned by the common codec.
pub fn projected_canonical_output_resource_facts(
    encoded_payload_bytes_upper_bound: u64,
) -> Result<PreparedOutputResourceFacts, ResourcePlanningError> {
    if encoded_payload_bytes_upper_bound == 0 {
        return Err(ResourcePlanningError::Invalid(
            "projected output payload must be nonzero".into(),
        ));
    }
    let closure = encoded_payload_bytes_upper_bound
        .checked_add(ANALYSIS_MANIFEST_MAX_ENCODED_BYTES)
        .ok_or(ResourcePlanningError::Overflow("canonical output closure"))?;
    // Outputs are encoded directly from their owned typed values, never through a value tree.
    let construction =
        runmat_canonical_codec::projected_canonical_encoding_resident_bytes_estimate(
            encoded_payload_bytes_upper_bound,
            1,
        )
        .and_then(|bytes| bytes.checked_add(closure))
        .ok_or(ResourcePlanningError::Overflow(
            "canonical output construction",
        ))?;
    Ok(PreparedOutputResourceFacts {
        encoded_payload_bytes_upper_bound: ByteCount(encoded_payload_bytes_upper_bound),
        transport_closure_bytes_upper_bound: ByteCount(closure),
        construction_resident_bytes_admission_budget: ByteCount(construction),
    })
}

/// A domain-owned structural profile replaces the unknown-shape multiplier.
/// Transport and manifest allowances stay identical to ordinary output planning. Construction
/// holds the owned typed state (the profile's materialization estimate) and must also cover
/// direct canonical encoding of that state, whichever is larger.
pub fn profiled_canonical_output_resource_facts(
    profile: runmat_canonical_codec::SerializedCanonicalProfile,
) -> Result<PreparedOutputResourceFacts, ResourcePlanningError> {
    let encoded = profile.encoded_bytes_upper_bound;
    if encoded == 0 {
        return Err(ResourcePlanningError::Invalid(
            "profiled output must be nonempty".into(),
        ));
    }
    let closure = encoded
        .checked_add(ANALYSIS_MANIFEST_MAX_ENCODED_BYTES)
        .ok_or(ResourcePlanningError::Overflow("profiled output closure"))?;
    let encoding =
        runmat_canonical_codec::projected_canonical_encoding_resident_bytes_estimate(encoded, 1)
            .ok_or(ResourcePlanningError::Overflow("profiled output encoding"))?;
    let construction = profile
        .materialization_resident_bytes_estimate
        .max(encoding)
        .checked_add(closure)
        .ok_or(ResourcePlanningError::Overflow(
            "profiled output construction",
        ))?;
    Ok(PreparedOutputResourceFacts {
        encoded_payload_bytes_upper_bound: ByteCount(encoded),
        transport_closure_bytes_upper_bound: ByteCount(closure),
        construction_resident_bytes_admission_budget: ByteCount(construction),
    })
}

pub fn structural_state_profiles_output_resource_facts(
    model: &AnalysisModel,
    step: &ProcedureStep,
    topology: Digest,
) -> Result<PreparedOutputResourceFacts, ResourcePlanningError> {
    let profile = structural_state_publication_profiles_encoding_profile(
        model,
        step,
        &[
            SemanticStatePublicationSource::StructuralPrimaryFields,
            SemanticStatePublicationSource::StructuralLinearization,
        ],
        topology,
    )
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    projected_canonical_output_resource_facts(profile.encoded_bytes_upper_bound)
}
