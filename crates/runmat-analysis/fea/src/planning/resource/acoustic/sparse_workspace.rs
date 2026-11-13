//! Sampling/assembly-free sparse real-block congruence and certified direct-factor envelope.
use super::*;

pub(crate) fn sparse_workspace_bound(
    support: u64,
    block: u64,
    recovery_row_support: u64,
) -> Result<u64, ResourcePlanningError> {
    if !matches!(recovery_row_support, 1 | 2) {
        return Err(invalid("unsupported acoustic affine row support"));
    }
    // K/M entries occur in two real diagonal blocks; B in two off-diagonal blocks.
    // Identity/prescribed recovery rows have <=1 coefficient; active unit-phase
    // relations require <=2. T^T A T emits at most row_support^2 contributions
    // per full entry. Cancellations can only reduce nnz.
    let full = mul(support, 2)?;
    let reduced = mul(full, mul(recovery_row_support, recovery_row_support)?)?;
    // Conservatively sum full and reduced discovered BTree assembly + CSR,
    // including finalization overlap. Do not substitute K/M/B's support for the
    // potentially denser phase-congruent reduced matrix.
    let matrices = mul(add(full, reduced)?, 160)?;
    // One certified sparse LU of the reduced operator (factor model plus estimator and
    // refinement vectors), quoted by the shared certificate layer.
    let factor = crate::solve::certificate::nonsingularity_reservation_bytes(
        usize::try_from(block).map_err(|_| invalid("acoustic block dimension"))?,
        usize::try_from(reduced).map_err(|_| invalid("acoustic reduced nonzeros"))?,
    );
    if factor == u64::MAX {
        return Err(ResourcePlanningError::Overflow(
            "acoustic factor reservation",
        ));
    }
    // CSR row/header allowances, reduction clone and solution/residual vectors coexist.
    add(
        add(matrices, factor)?,
        add(mul(add(block, 1)?, 256)?, 4096)?,
    )
}
