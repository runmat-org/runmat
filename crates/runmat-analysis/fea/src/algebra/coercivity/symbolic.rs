//! Resource envelopes for coercivity proofs by supernodal sparse Cholesky.
//!
//! The proof and the direct solve are one factorization from the shared solver layer
//! (`runmat-sparse-solvers`): nested-dissection ordering and an exact symbolic analysis give the
//! factor size and work before any numeric allocation.

use runmat_sparse_solvers::{
    control::SolveControl,
    direct::{DirectOptions, FactorEstimate, FactorKind, SymbolicFactorization},
};

use crate::algebra::SparsePattern;

use super::CoercivityError;

/// Headroom (numerator/denominator) between the planned support pattern and the exact
/// numerical pattern the proof factors. The ordering is recomputed on the numerical support,
/// which may differ from the planned superset; nested dissection fill is stable to within a few
/// percent under such changes, and the 25% margin is checked against every certificate.
const HEADROOM_NUMERATOR: u64 = 5;
const HEADROOM_DENOMINATOR: u64 = 4;

fn with_headroom(value: u64) -> Option<u64> {
    value
        .checked_mul(HEADROOM_NUMERATOR)
        .map(|value| value / HEADROOM_DENOMINATOR)
}

/// Bounds the work of a sparse Cholesky proof on any numerical support of `pattern`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CoercivityWorkEnvelope {
    dimension: u64,
    factor_nonzero_limit: u64,
    update_limit: u64,
    resident_bytes_limit: u64,
    /// Resident peak of a sparse Cholesky factorization of the pattern (with headroom).
    factor_resident_limit: u64,
    /// Graph-count budgets of the anchored local-energy proof (independent of factorization).
    energy_operation_limit: u64,
    energy_resident_limit: u64,
}

/// Local-energy proof budgets from the undirected graph size: work ≤ 1024 (n + |E|) + n and
/// storage ≤ 192 B · (16 (n + |E|) + n) + 256 B · n.
fn energy_limits(dimension: u64, edges: u64) -> Option<(u64, u64)> {
    let input = dimension.checked_add(edges)?;
    let operations = input.checked_mul(1024)?.checked_add(dimension)?;
    let entries = input.checked_mul(16)?.checked_add(dimension)?;
    let resident = entries.checked_mul(192)?.checked_add(dimension.checked_mul(256)?)?;
    Some((operations, resident))
}

impl CoercivityWorkEnvelope {
    pub(crate) fn derive(pattern: &SparsePattern) -> Result<Self, CoercivityError> {
        let dimension = pattern.rows();
        if dimension == 0 || pattern.columns() != dimension {
            return Err(CoercivityError::IncompatibleOperator);
        }
        let estimate = analyze(pattern)?;
        let mut edges = std::collections::BTreeSet::new();
        for row in 0..pattern.rows() {
            for entry in pattern.row_range(row) {
                let column = pattern.column(entry);
                if row != column {
                    edges.insert((row.min(column), row.max(column)));
                }
            }
        }
        Self::from_estimate(&estimate, edges.len() as u64)
    }

    pub(super) fn from_estimate(estimate: &FactorEstimate, edges: u64) -> Result<Self, CoercivityError> {
        let dimension = u64::try_from(estimate.dimension).map_err(|_| CoercivityError::WorkOverflow)?;
        let (energy_operation_limit, energy_resident_limit) =
            energy_limits(dimension, edges).ok_or(CoercivityError::WorkOverflow)?;
        Ok(Self {
            energy_operation_limit,
            energy_resident_limit,
            dimension,
            factor_nonzero_limit: with_headroom(estimate.factor_entries).ok_or(CoercivityError::WorkOverflow)?,
            // Either proof method may run: the limits cover the Cholesky factorization and the
            // anchored local-energy proof.
            update_limit: with_headroom(flop_count(estimate))
                .ok_or(CoercivityError::WorkOverflow)?
                .max(energy_operation_limit),
            resident_bytes_limit: with_headroom(estimate.peak_bytes())
                .ok_or(CoercivityError::WorkOverflow)?
                .max(energy_resident_limit),
            factor_resident_limit: with_headroom(estimate.peak_bytes())
                .ok_or(CoercivityError::WorkOverflow)?,
        })
    }

    pub(crate) fn dimension(self) -> u64 {
        self.dimension
    }
    pub(crate) fn factor_nonzero_limit(self) -> u64 {
        self.factor_nonzero_limit
    }
    pub(crate) fn update_limit(self) -> u64 {
        self.update_limit
    }
    pub(crate) fn resident_bytes_limit(self) -> u64 {
        self.resident_bytes_limit
    }
    pub(crate) fn factor_resident_limit(self) -> u64 {
        self.factor_resident_limit
    }
    pub(crate) fn energy_operation_limit(self) -> u64 {
        self.energy_operation_limit
    }
    pub(crate) fn energy_resident_limit(self) -> u64 {
        self.energy_resident_limit
    }
}

/// Floating point operations of the numeric factorization, rounded up.
pub(super) fn flop_count(estimate: &FactorEstimate) -> u64 {
    if estimate.flops.is_finite() && estimate.flops > 0.0 {
        estimate.flops.ceil().min(u64::MAX as f64) as u64
    } else {
        0
    }
}

/// Symbolic Cholesky analysis of a pattern (values are not needed).
pub(super) fn analyze(pattern: &SparsePattern) -> Result<FactorEstimate, CoercivityError> {
    let offsets = pattern
        .row_offsets()
        .iter()
        .map(|&value| usize::try_from(value).map_err(|_| CoercivityError::WorkOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    let columns = pattern
        .column_indices()
        .iter()
        .map(|&value| usize::try_from(value).map_err(|_| CoercivityError::WorkOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    let symbolic = SymbolicFactorization::analyze_pattern::<f64>(
        pattern.rows(),
        &offsets,
        &columns,
        &DirectOptions::new(FactorKind::Cholesky),
        SolveControl::NONE,
    )
    .map_err(|_| CoercivityError::SymbolicOrdering)?;
    Ok(symbolic.estimate())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_quotes_exact_symbolic_counts_with_headroom() {
        // 8 × 8 grid Laplacian pattern.
        const WIDTH: usize = 8;
        let n = WIDTH * WIDTH;
        let mut offsets = vec![0u64];
        let mut columns = Vec::new();
        for row in 0..n {
            let (x, y) = (row % WIDTH, row / WIDTH);
            let mut neighbors = vec![row];
            if x > 0 {
                neighbors.push(row - 1);
            }
            if x + 1 < WIDTH {
                neighbors.push(row + 1);
            }
            if y > 0 {
                neighbors.push(row - WIDTH);
            }
            if y + 1 < WIDTH {
                neighbors.push(row + WIDTH);
            }
            neighbors.sort_unstable();
            columns.extend(neighbors.into_iter().map(|c| c as u64));
            offsets.push(columns.len() as u64);
        }
        let pattern = SparsePattern::try_new(n as u64, n as u64, offsets, columns).unwrap();
        let estimate = analyze(&pattern).unwrap();
        let envelope = CoercivityWorkEnvelope::derive(&pattern).unwrap();
        assert_eq!(envelope.dimension(), n as u64);
        assert!(envelope.factor_nonzero_limit() >= estimate.factor_entries);
        assert!(estimate.factor_entries as usize >= n);
        assert!(envelope.resident_bytes_limit() >= estimate.peak_bytes());
    }
}
