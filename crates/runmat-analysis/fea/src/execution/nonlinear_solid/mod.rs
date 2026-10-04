//! Original load-controller blocks; transport integrity is not production qualification.
use super::{AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::Serialize;

mod artifact;
mod fields;
pub use fields::nonlinear_native_field_definitions;
mod receipt;
mod request;
mod workload;
pub use artifact::NonlinearSolidBlockArtifact;
pub use receipt::{
    NonlinearSolidNumericalReceipt, NonlinearSolidNumericalReceiptArtifact,
    NONLINEAR_SOLID_RECEIPT_MAX_ENCODED_BYTES,
};
pub use request::{NonlinearSolidBlockReference, NonlinearSolidBlockRequest};
pub use workload::NonlinearSolidWorkload;

pub(super) const LIMITS: CanonicalLimits =
    CanonicalLimits::new(1024 * 1024 * 1024, 100_000_000, 65_536, 64);
fn invalid(e: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(e.to_string())
}
fn digest(domain: &'static str, value: &impl Serialize) -> AnalysisExecutionResult<Digest> {
    Ok(Digest::sha256(encode_contract(
        b"runmat-nonlinear-solid/v1\0",
        domain,
        value,
        LIMITS,
    )?))
}

impl super::NumericalPlanIdentity {
    pub fn from_nonlinear_solid_block(
        request: &NonlinearSolidBlockRequest,
    ) -> AnalysisExecutionResult<Self> {
        request.validate()?;
        Self::derive("nonlinear-solid-block-plan", request)
    }
}

#[cfg(test)]
mod tests;
