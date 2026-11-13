use runmat_canonical_codec::CanonicalLimits;
use runmat_execution::Digest;

use super::AssemblySummary;
use crate::contracts::FeaRunError;

const STRUCTURAL_OPERATOR_PREFIX: &[u8] = b"runmat-fea-structural-operator/v1\0";
const STRUCTURAL_OPERATOR_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(
        512 * 1024 * 1024,
        20_000_000,
        4096,
        16,
    ));

/// Exact formulation, constraint, operator, layout, and recovery compatibility identity.
pub fn structural_operator_digest(summary: &AssemblySummary) -> Result<Digest, FeaRunError> {
    crate::contracts::canonical_digest::canonical_contract_digest(
        STRUCTURAL_OPERATOR_PREFIX,
        "structural-operator",
        &(
            summary.operator.operators(),
            summary.operator.constrained_dofs(),
            &summary.constraint_matrix,
            &summary.structural_dof_layout,
            &summary.structural_solid_recovery,
            &summary.structural_beam_recovery,
            &summary.structural_shell_recovery,
        ),
        STRUCTURAL_OPERATOR_LIMITS,
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))
}
