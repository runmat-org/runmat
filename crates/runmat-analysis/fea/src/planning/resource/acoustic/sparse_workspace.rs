//! Sampling/assembly-free sparse real-block congruence and right-factor envelope.
use super::*;
use crate::contracts::SolverPreconditioner;

pub(crate) fn sparse_workspace_bound(
    support: u64,
    block: u64,
    recovery_row_support: u64,
    preconditioner: SolverPreconditioner,
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
    let factors = match preconditioner {
        SolverPreconditioner::None => 0,
        // Fixed-support LU: two nnz buffers and two row-sized index buffers.
        // Twice the exact payload allows allocation capacity; no row maps are
        // constructed. Source reduced CSR remains retained alongside factors.
        SolverPreconditioner::Ilu0 => add(mul(reduced, 32)?, mul(add(block, 1)?, 32)?)?,
        _ => return Err(invalid("unsupported acoustic sparse preconditioner")),
    };
    // Both CSR row/header allowances, reduction clone and extra physical
    // correction/application vectors coexist. Existing Krylov buffers remain
    // separately reserved, rather than counted here as a replacement.
    add(
        add(matrices, factors)?,
        add(mul(add(block, 1)?, 256)?, 4096)?,
    )
}
