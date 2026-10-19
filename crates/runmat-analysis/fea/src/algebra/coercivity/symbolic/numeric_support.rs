//! Exact numerical support for proof construction; no tolerance-based coefficient dropping.
use super::*;
use crate::algebra::RealSparseMatrix;

impl CoercivitySymbolicPlan {
    pub(in crate::algebra::coercivity) fn derive_for_matrix(
        matrix: &RealSparseMatrix,
    ) -> Result<Self, CoercivityError> {
        let pattern = matrix.pattern();
        if pattern.rows() == 0 || pattern.rows() != pattern.columns() {
            return Err(CoercivityError::IncompatibleOperator);
        }
        let mut graph = vec![BTreeSet::new(); pattern.rows()];
        for row in 0..pattern.rows() {
            for entry in pattern.row_range(row) {
                let column = pattern.column(entry);
                if row != column && matrix.values()[entry] != 0.0 {
                    graph[row].insert(column);
                    graph[column].insert(row);
                }
            }
        }
        Self::derive_graph(pattern, graph)
    }
}

#[cfg(test)]
mod tests;
