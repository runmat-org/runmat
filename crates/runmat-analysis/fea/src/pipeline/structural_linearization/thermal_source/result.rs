//! Non-circular physical result identity: no serialized prestress operator appears in its source.

use super::*;
use runmat_analysis_core::AnalysisModel;

const PREFIX: &[u8] = b"runmat-analysis-thermoelastic-result/v2\0";
const LIMITS: CanonicalLimits = crate::contracts::canonical_digest::wire_bounded(
    CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64),
);

pub(super) fn model_digest(model: &AnalysisModel) -> Result<Digest, StructuralLinearizationError> {
    crate::contracts::canonical_digest::canonical_contract_digest(
        PREFIX,
        "solver-model",
        &model.solver_identity_contract(),
        LIMITS,
    )
    .map_err(|error| StructuralLinearizationError::Identity(error.to_string()))
}

pub(super) fn content_digest(
    model: Digest,
    transition: &CanonicalStateTransition,
    run: &crate::FeaRunResult,
) -> Result<Digest, StructuralLinearizationError> {
    crate::contracts::canonical_digest::canonical_contract_digest(
        PREFIX,
        "solved-thermal-static-result",
        &(model, transition, &run.identity, &run.fields),
        LIMITS,
    )
    .map_err(|error| StructuralLinearizationError::Identity(error.to_string()))
}
