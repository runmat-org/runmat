//! Requested cumulative evaluation workspace through captured storage authority.
use super::{AssessmentEvaluationError as Error, ScalarEvidenceSource};

pub(super) struct Workspace {
    maximum: Option<u64>,
}
impl Workspace {
    pub(super) fn capture(source: &(impl ScalarEvidenceSource + ?Sized)) -> Result<Self, Error> {
        Ok(Self {
            maximum: source.workspace_limit().map_err(store_error)?,
        })
    }
    /// Bytes one run's evidence may occupy: the captured run allowance when the publication
    /// supplies one, otherwise the standalone reassessment's explicit limit.
    pub(super) fn run_artifact_bytes(&self, standalone: u64) -> u64 {
        self.maximum.unwrap_or(standalone)
    }
    pub(super) fn admit(&self, inventory: &impl serde::Serialize) -> Result<u64, Error> {
        let Some(maximum) = self.maximum else {
            return Ok(0);
        };
        let bytes = crate::analysis::storage::publication::resident(inventory, maximum).map_err(
            |error| match error {
                crate::analysis::storage::AnalysisPublicationError::ResourceLimit { .. } => {
                    Error::ResourceLimit
                }
                other => Error::InvalidEvidence(other.to_string()),
            },
        )?;
        if bytes > maximum {
            return Err(Error::ResourceLimit);
        }
        Ok(bytes)
    }
}
fn store_error(error: crate::analysis::storage::AssessmentStoreError) -> Error {
    match error {
        crate::analysis::storage::AssessmentStoreError::ResourceLimit => Error::ResourceLimit,
        crate::analysis::storage::AssessmentStoreError::Invalid(message) => {
            Error::InvalidEvidence(message)
        }
    }
}
