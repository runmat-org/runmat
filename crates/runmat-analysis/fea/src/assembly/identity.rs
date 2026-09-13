use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;

use super::AssemblySummary;
use crate::contracts::FeaRunError;

const STRUCTURAL_OPERATOR_PREFIX: &[u8] = b"runmat-fea-structural-operator/v1\0";
const STRUCTURAL_OPERATOR_LIMITS: CanonicalLimits =
    CanonicalLimits::new(512 * 1024 * 1024, 20_000_000, 4096, 16);

/// Exact formulation, constraint, operator, layout, and recovery compatibility identity.
pub fn structural_operator_digest(summary: &AssemblySummary) -> Result<Digest, FeaRunError> {
    encode_contract(
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
    .map(Digest::sha256)
    .map_err(|error| FeaRunError::Assembly(error.to_string()))
}
