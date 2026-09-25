use std::collections::{BTreeSet, VecDeque};

use runmat_execution::Digest;

use crate::algebra::SparsePattern;

use super::CoercivityError;

const MAX_FACTOR_FILL_MULTIPLIER: u64 = 16;
// Connected physical patches include 4,096 Tet4 and 1,024 Tet10 elements. The latter
// crossed the former 768x update envelope by 38,559 updates at pivot 4,402, while its
// factor fill remained below the separate 16x cap. Bound the work above both independently
// verified affine patches without treating a failed symbolic proof as a physics result.
const MAX_FACTOR_UPDATE_MULTIPLIER: u64 = 1024;
const BYTES_PER_SYMBOLIC_FACTOR_ENTRY: u64 = 192;
const BYTES_PER_SYMBOLIC_ROW: u64 = 256;

pub(super) fn symbolic_resident_bytes(dimension: u64, factor_nonzeros: u64) -> Option<u64> {
    factor_nonzeros
        .checked_mul(BYTES_PER_SYMBOLIC_FACTOR_ENTRY)
        .and_then(|bytes| {
            dimension
                .checked_mul(BYTES_PER_SYMBOLIC_ROW)
                .and_then(|rows| bytes.checked_add(rows))
        })
}

/// Bounds the work of any successful exact-pattern proof whose graph is a subset of `pattern`.
/// This is a resource reservation, not proof that such a factorization exists or fits its own
/// pattern-relative work limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CoercivityWorkEnvelope {
    dimension: u64,
    factor_nonzero_limit: u64,
    update_limit: u64,
    resident_bytes_limit: u64,
}

impl CoercivityWorkEnvelope {
    pub(crate) fn derive(pattern: &SparsePattern) -> Result<Self, CoercivityError> {
        let dimension = pattern.rows();
        if dimension == 0 || pattern.columns() != dimension {
            return Err(CoercivityError::IncompatibleOperator);
        }
        let dimension = u64::try_from(dimension).map_err(|_| CoercivityError::WorkOverflow)?;
        let mut edges = BTreeSet::new();
        for row in 0..pattern.rows() {
            for entry in pattern.row_range(row) {
                let column = pattern.column(entry);
                if row != column {
                    edges.insert((row.min(column), row.max(column)));
                }
            }
        }
        Self::from_graph_counts(dimension, edges.len())
    }

    fn from_graph_counts(dimension: u64, edge_count: usize) -> Result<Self, CoercivityError> {
        let input_lower_nonzeros = dimension
            .checked_add(u64::try_from(edge_count).map_err(|_| CoercivityError::WorkOverflow)?)
            .ok_or(CoercivityError::WorkOverflow)?;
        let factor_nonzero_limit = input_lower_nonzeros
            .checked_mul(MAX_FACTOR_FILL_MULTIPLIER)
            .and_then(|limit| limit.checked_add(dimension))
            .ok_or(CoercivityError::WorkOverflow)?;
        let update_limit = input_lower_nonzeros
            .checked_mul(MAX_FACTOR_UPDATE_MULTIPLIER)
            .and_then(|limit| limit.checked_add(dimension))
            .ok_or(CoercivityError::WorkOverflow)?;
        let resident_bytes_limit = symbolic_resident_bytes(dimension, factor_nonzero_limit)
            .ok_or(CoercivityError::WorkOverflow)?;
        Ok(Self {
            dimension,
            factor_nonzero_limit,
            update_limit,
            resident_bytes_limit,
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
}

/// Deterministic fill-reducing symbolic preparation shared by resource planning and numerical
/// coercivity validation. Indices in `pivot_neighbors` use elimination-order coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CoercivitySymbolicPlan {
    pattern_identity: Digest,
    dimension: u64,
    elimination_order: Vec<usize>,
    inverse_order: Vec<usize>,
    pivot_neighbors: Vec<Vec<usize>>,
    factor_nonzero_count: u64,
    update_count: u64,
    resident_bytes_upper_bound: u64,
}

impl CoercivitySymbolicPlan {
    pub(crate) fn derive(pattern: &SparsePattern) -> Result<Self, CoercivityError> {
        let dimension = pattern.rows();
        if dimension == 0 || pattern.columns() != dimension {
            return Err(CoercivityError::IncompatibleOperator);
        }
        let mut original_graph = vec![BTreeSet::<usize>::new(); dimension];
        for row in 0..dimension {
            for entry in pattern.row_range(row) {
                let column = pattern.column(entry);
                if row != column {
                    original_graph[row].insert(column);
                    original_graph[column].insert(row);
                }
            }
        }
        let elimination_order = reverse_cuthill_mckee(&original_graph)?;
        let mut inverse_order = vec![0; dimension];
        for (position, original) in elimination_order.iter().copied().enumerate() {
            inverse_order[original] = position;
        }
        let mut graph = vec![BTreeSet::<usize>::new(); dimension];
        for original_row in 0..dimension {
            let row = inverse_order[original_row];
            for original_column in &original_graph[original_row] {
                graph[row].insert(inverse_order[*original_column]);
            }
        }

        let dimension_u64 = u64::try_from(dimension).map_err(|_| CoercivityError::WorkOverflow)?;
        let edge_count = original_graph
            .iter()
            .try_fold(0_usize, |total, neighbors| {
                total.checked_add(neighbors.len())
            })
            .ok_or(CoercivityError::WorkOverflow)?
            / 2;
        let envelope = CoercivityWorkEnvelope::from_graph_counts(dimension_u64, edge_count)?;
        let factor_limit = envelope.factor_nonzero_limit();
        let update_limit = envelope.update_limit();

        let mut pivot_neighbors = Vec::with_capacity(dimension);
        let mut factor_nonzero_count = dimension_u64;
        let mut update_count = 0_u64;
        for pivot in 0..dimension {
            let neighbors = graph[pivot]
                .range((pivot + 1)..)
                .copied()
                .collect::<Vec<_>>();
            factor_nonzero_count = factor_nonzero_count
                .checked_add(
                    u64::try_from(neighbors.len()).map_err(|_| CoercivityError::WorkOverflow)?,
                )
                .ok_or(CoercivityError::WorkOverflow)?;
            let neighbor_count =
                u64::try_from(neighbors.len()).map_err(|_| CoercivityError::WorkOverflow)?;
            update_count = update_count
                .checked_add(
                    neighbor_count
                        .checked_mul(neighbor_count.saturating_add(1))
                        .and_then(|value| value.checked_div(2))
                        .ok_or(CoercivityError::WorkOverflow)?,
                )
                .ok_or(CoercivityError::WorkOverflow)?;
            if factor_nonzero_count > factor_limit || update_count > update_limit {
                return Err(CoercivityError::ResourceLimit {
                    pivot,
                    factor_nonzero_count,
                    factor_nonzero_limit: factor_limit,
                    update_count,
                    update_limit,
                });
            }
            for left in 0..neighbors.len() {
                for right in 0..left {
                    graph[neighbors[left]].insert(neighbors[right]);
                    graph[neighbors[right]].insert(neighbors[left]);
                }
            }
            for neighbor in &neighbors {
                graph[*neighbor].remove(&pivot);
            }
            graph[pivot].clear();
            pivot_neighbors.push(neighbors);
        }
        let resident_bytes_upper_bound =
            symbolic_resident_bytes(dimension_u64, factor_nonzero_count)
                .ok_or(CoercivityError::WorkOverflow)?;
        Ok(Self {
            pattern_identity: pattern
                .identity()
                .map_err(|error| CoercivityError::Encoding(error.to_string()))?,
            dimension: dimension_u64,
            elimination_order,
            inverse_order,
            pivot_neighbors,
            factor_nonzero_count,
            update_count,
            resident_bytes_upper_bound,
        })
    }

    pub(crate) fn validate_for(&self, pattern: &SparsePattern) -> Result<(), CoercivityError> {
        if self.pattern_identity
            != pattern
                .identity()
                .map_err(|error| CoercivityError::Encoding(error.to_string()))?
            || usize::try_from(self.dimension) != Ok(pattern.rows())
            || self.elimination_order.len() != pattern.rows()
            || self.inverse_order.len() != pattern.rows()
            || self.pivot_neighbors.len() != pattern.rows()
        {
            return Err(CoercivityError::BindingMismatch);
        }
        let mut seen = vec![false; pattern.rows()];
        for (position, original) in self.elimination_order.iter().copied().enumerate() {
            if original >= pattern.rows()
                || seen[original]
                || self.inverse_order[original] != position
            {
                return Err(CoercivityError::BindingMismatch);
            }
            seen[original] = true;
        }
        Ok(())
    }

    pub(crate) fn factor_rows(&self) -> Vec<Vec<(usize, f64)>> {
        let mut rows = vec![Vec::new(); self.elimination_order.len()];
        for (pivot, neighbors) in self.pivot_neighbors.iter().enumerate() {
            rows[pivot].push((pivot, 0.0));
            for neighbor in neighbors {
                rows[*neighbor].push((pivot, 0.0));
            }
        }
        rows
    }

    pub(crate) fn elimination_order(&self) -> &[usize] {
        &self.elimination_order
    }

    pub(crate) fn inverse_order(&self) -> &[usize] {
        &self.inverse_order
    }

    pub(crate) fn pivot_neighbors(&self) -> &[Vec<usize>] {
        &self.pivot_neighbors
    }

    pub(crate) const fn factor_nonzero_count(&self) -> u64 {
        self.factor_nonzero_count
    }

    pub(crate) const fn update_count(&self) -> u64 {
        self.update_count
    }

    pub(crate) const fn resident_bytes_upper_bound(&self) -> u64 {
        self.resident_bytes_upper_bound
    }
}

fn reverse_cuthill_mckee(graph: &[BTreeSet<usize>]) -> Result<Vec<usize>, CoercivityError> {
    let mut visited = vec![false; graph.len()];
    let mut order = Vec::with_capacity(graph.len());
    while order.len() < graph.len() {
        let Some(root) = (0..graph.len())
            .filter(|node| !visited[*node])
            .min_by_key(|node| (graph[*node].len(), *node))
        else {
            return Err(CoercivityError::SymbolicOrdering);
        };
        let component_start = order.len();
        visited[root] = true;
        let mut queue = VecDeque::from([root]);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            let mut neighbors = graph[node]
                .iter()
                .copied()
                .filter(|neighbor| !visited[*neighbor])
                .collect::<Vec<_>>();
            neighbors.sort_unstable_by_key(|neighbor| (graph[*neighbor].len(), *neighbor));
            for neighbor in neighbors {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        order[component_start..].reverse();
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(rows: &[&[usize]]) -> SparsePattern {
        let mut offsets = vec![0_u64];
        let mut columns = Vec::new();
        for row in rows {
            columns.extend(row.iter().map(|column| *column as u64));
            offsets.push(columns.len() as u64);
        }
        SparsePattern::try_new(rows.len() as u64, rows.len() as u64, offsets, columns).unwrap()
    }

    #[test]
    fn ordering_and_fill_counts_are_deterministic_and_pattern_bound() {
        let source = pattern(&[&[0, 1, 3], &[0, 1, 2], &[1, 2, 3], &[0, 2, 3]]);
        let first = CoercivitySymbolicPlan::derive(&source).unwrap();
        let second = CoercivitySymbolicPlan::derive(&source).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.factor_nonzero_count(), 9);
        assert_eq!(first.update_count(), 7);
        first.validate_for(&source).unwrap();
    }

    #[test]
    fn upper_graph_work_envelope_bounds_a_successful_sparse_subgraph_proof() {
        let upper = pattern(&[&[0, 1, 2, 3], &[0, 1, 2, 3], &[0, 1, 2, 3], &[0, 1, 2, 3]]);
        let exact = pattern(&[&[0, 1], &[0, 1, 2], &[1, 2, 3], &[2, 3]]);
        let envelope = CoercivityWorkEnvelope::derive(&upper).unwrap();
        let proof = CoercivitySymbolicPlan::derive(&exact).unwrap();
        assert_eq!(envelope.dimension(), proof.dimension);
        assert!(envelope.factor_nonzero_limit() >= proof.factor_nonzero_count());
        assert!(envelope.update_limit() >= proof.update_count());
        assert!(envelope.resident_bytes_limit() >= proof.resident_bytes_upper_bound());
    }

    #[test]
    fn successful_upper_graph_symbolic_work_is_not_a_subgraph_work_bound() {
        let upper = pattern(&[
            &[0, 1, 2, 3, 4],
            &[0, 1, 2, 3, 4],
            &[0, 1, 2],
            &[0, 1, 3],
            &[0, 1, 4],
        ]);
        let exact = pattern(&[
            &[0, 1, 3, 4],
            &[0, 1, 2, 3, 4],
            &[1, 2],
            &[0, 1, 3],
            &[0, 1, 4],
        ]);
        let upper_proof = CoercivitySymbolicPlan::derive(&upper).unwrap();
        let exact_proof = CoercivitySymbolicPlan::derive(&exact).unwrap();
        assert_eq!(upper_proof.update_count(), 10);
        assert_eq!(exact_proof.update_count(), 11);
        let envelope = CoercivityWorkEnvelope::derive(&upper).unwrap();
        assert!(envelope.update_limit() >= exact_proof.update_count());
        assert!(envelope.factor_nonzero_limit() >= exact_proof.factor_nonzero_count());
        assert!(envelope.resident_bytes_limit() >= exact_proof.resident_bytes_upper_bound());
    }

    #[test]
    fn structured_three_dimensional_pattern_has_bounded_repeatable_fill() {
        const WIDTH: usize = 6;
        let dimension = WIDTH * WIDTH * WIDTH;
        let mut rows = vec![BTreeSet::new(); dimension];
        let index = |x: usize, y: usize, z: usize| x + WIDTH * (y + WIDTH * z);
        for z in 0..WIDTH {
            for y in 0..WIDTH {
                for x in 0..WIDTH {
                    let row = index(x, y, z);
                    rows[row].insert(row);
                    for (dx, dy, dz) in [
                        (-1_isize, 0_isize, 0_isize),
                        (1, 0, 0),
                        (0, -1, 0),
                        (0, 1, 0),
                        (0, 0, -1),
                        (0, 0, 1),
                    ] {
                        let (Some(nx), Some(ny), Some(nz)) = (
                            x.checked_add_signed(dx),
                            y.checked_add_signed(dy),
                            z.checked_add_signed(dz),
                        ) else {
                            continue;
                        };
                        if nx < WIDTH && ny < WIDTH && nz < WIDTH {
                            rows[row].insert(index(nx, ny, nz));
                        }
                    }
                }
            }
        }
        let rows = rows
            .iter()
            .map(|row| row.iter().copied().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let row_refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let source = pattern(&row_refs);
        let first = CoercivitySymbolicPlan::derive(&source).unwrap();
        let second = CoercivitySymbolicPlan::derive(&source).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.dimension, dimension as u64);
        assert!(first.factor_nonzero_count() > dimension as u64);
        assert!(first.resident_bytes_upper_bound() < 64 * 1024 * 1024);
    }
}
