//! Canonical metadata and counted declaration coverage, excluding original-source reconstruction.
use super::*;

impl ObservationPublicationCatalog {
    /// Additional metadata validation workspace. The catalog is already resident (borrowed
    /// storage is excluded); validation directly encodes it for canonical identities and builds
    /// counted coverage. Physical source projection/schedules/target reconstruction and RSS are
    /// excluded.
    pub fn metadata_validation_resident_bytes(&self) -> AnalysisExecutionResult<u64> {
        // A worst-case wire estimate is a memory fact, not the wire contract ceiling.
        let profile =
            runmat_canonical_codec::profile_serialized(self, PROFILE_LIMITS).map_err(invalid)?;
        let encoding =
            runmat_canonical_codec::projected_canonical_encoding_resident_bytes_estimate(
                profile.encoded_bytes_upper_bound,
                1,
            )
            .ok_or(AnalysisExecutionError::Limit(
                "catalog metadata encoding workspace",
            ))?;
        encoding
            .checked_add(coverage::PublicationCoverage::resident_bytes(
                self.declarations.len(),
            )?)
            .ok_or(AnalysisExecutionError::Limit(
                "catalog metadata validation workspace",
            ))
    }
    /// Admit the metadata stage before materializing canonical identities or coverage.
    /// Original source reconstruction requires its own combined reservation.
    pub fn validate_metadata_admitted(&self, maximum: u64) -> AnalysisExecutionResult<()> {
        if self.metadata_validation_resident_bytes()? > maximum {
            return Err(AnalysisExecutionError::Limit(
                "catalog metadata validation workspace",
            ));
        }
        self.validate_metadata()
    }
}
