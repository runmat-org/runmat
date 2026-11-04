//! A sufficient coercivity check for a streamed assembly of constant-null local energies.
//! Local LDL candidates are checked against outward product bounds. A grounded graph
//! inequality supplies the global lower bound; the exact CSR is compared against the
//! reconstructed local energies, including assembly roundoff. Labels and hashes prove nothing.
use super::{CoercivityWorkEnvelope, MatrixSymmetry, RealSparseMatrix};
use runmat_execution::Digest;
use std::collections::{BTreeMap, VecDeque};
mod admission;
mod local;
pub(crate) use admission::SourceEnergyDemand;
/// Resident bytes the check holds per stored operator entry: its reconstructed interval.
pub(crate) const ENTRY_RESIDENT_BYTES: u64 = std::mem::size_of::<outward::Interval>() as u64;
/// Resident bytes per projected vertex: graph adjacency header, BFS distance and queue slot,
/// and the row/column error accumulators of `finish`.
pub(crate) const VERTEX_RESIDENT_BYTES: u64 = 80;
/// Resident bytes per distinct missing pair: the B-tree entry (16 B key, 16 B interval) at
/// minimum node occupancy including its share of internal-node edges.
pub(crate) const MISSING_PAIR_RESIDENT_BYTES: u64 = 96;
use admission::{EnergyAdmission, LOCAL_PROOF_CACHE_CAPACITY};
mod projection;
use projection::EnergyProjection;
mod source;
pub(crate) use source::AnchoredEnergySource;
mod outward;
use outward::Interval;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EnergyError {
    #[error("incompatible anchored-energy operator or inventory")]
    Shape,
    #[error("invalid or incomplete local-energy source")]
    InvalidBlock,
    #[error("local-energy graph has an unanchored degree of freedom")]
    Unanchored,
    #[error("anchored-energy proof exceeded its admitted work or storage")]
    Admission {
        operations: u64,
        operation_limit: u64,
        resident: u64,
        resident_limit: u64,
    },
    #[error("anchored-energy arithmetic overflowed")]
    Arithmetic,
    #[error("anchored-energy operator has asymmetric values")]
    Asymmetric,
    #[error("anchored-energy verification cancelled")]
    Cancelled,
}

#[derive(Debug)]
pub(crate) struct AnchoredEnergyCertificate {
    operator_identity: Digest,
    lower_bound: f64,
    assembly_error_upper: f64,
    scalar_operations: u64,
    resident_bytes: u64,
}

impl AnchoredEnergyCertificate {
    pub fn operator_identity(&self) -> Digest {
        self.operator_identity
    }
    pub fn lower_bound(&self) -> f64 {
        self.lower_bound
    }
    pub fn assembly_error_upper(&self) -> f64 {
        self.assembly_error_upper
    }
    pub fn scalar_operations(&self) -> u64 {
        self.scalar_operations
    }
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

#[derive(Debug)]
pub(crate) enum EnergyVerdict {
    Verified(AnchoredEnergyCertificate),
    Indeterminate { graph_lower: f64, error_upper: f64 },
}

pub(crate) struct AnchoredEnergyCheck<'a> {
    matrix: &'a RealSparseMatrix,
    projection: EnergyProjection<'a>,
    entries: Vec<Interval>,
    graph: Vec<Vec<usize>>,
    missing: BTreeMap<(usize, usize), Interval>,
    minimum_local: f64,
    operations: u64,
    resident: u64,
    admitted: EnergyAdmission,
    failed: bool,
    /// A local block could not be certified positive: the verdict is indeterminate.
    uncertified: bool,
    peak_resident: u64,
    local_proofs: VecDeque<(Vec<f64>, f64)>,
}

impl<'a> AnchoredEnergyCheck<'a> {
    #[cfg(test)]
    pub fn new(
        matrix: &'a RealSparseMatrix,
        prescribed: &'a [bool],
        admitted: CoercivityWorkEnvelope,
    ) -> Result<Self, EnergyError> {
        Self::new_projected(
            matrix,
            EnergyProjection::Identity(prescribed),
            admitted.into(),
            0,
            0,
        )
    }

    fn new_projected(
        matrix: &'a RealSparseMatrix,
        projection: EnergyProjection<'a>,
        admitted: EnergyAdmission,
        retained_bytes: u64,
        source_operations: u64,
    ) -> Result<Self, EnergyError> {
        check_cancelled()?;
        let n = matrix.rows();
        if n == 0
            || matrix.columns() != n
            || matrix.properties().symmetry != MatrixSymmetry::Symmetric
            || admitted.dimension() != n as u64
        {
            return Err(EnergyError::Shape);
        }
        let operations = (matrix.values().len() as u64)
            .checked_mul(8)
            .and_then(|v| v.checked_add(source_operations))
            .ok_or(EnergyError::Arithmetic)?;
        if operations > admitted.update_limit() {
            return Err(EnergyError::Admission {
                operations,
                operation_limit: admitted.update_limit(),
                resident: 0,
                resident_limit: admitted.resident_bytes_limit(),
            });
        }
        for row in 0..n {
            check_cancelled()?;
            for k in matrix.pattern().row_range(row) {
                let value = matrix.values()[k];
                let other = matrix.get(matrix.pattern().column(k), row);
                if (value - other).abs() > value.abs().max(other.abs()).max(1.0) * 1e-12 {
                    return Err(EnergyError::Asymmetric);
                }
            }
        }
        let resident = (matrix.values().len() as u64)
            .checked_mul(ENTRY_RESIDENT_BYTES)
            .and_then(|v| {
                v.checked_add((projection.len() as u64).checked_mul(VERTEX_RESIDENT_BYTES)?)
            })
            .and_then(|v| v.checked_add(retained_bytes))
            .ok_or(EnergyError::Arithmetic)?;
        if resident > admitted.resident_bytes_limit() {
            return Err(EnergyError::Admission {
                operations,
                operation_limit: admitted.update_limit(),
                resident,
                resident_limit: admitted.resident_bytes_limit(),
            });
        }
        projection.validate(n)?;
        let entries = projection.seed_entries(matrix)?;
        Ok(Self {
            matrix,
            projection,
            entries,
            graph: vec![Vec::new(); projection.len()],
            missing: BTreeMap::new(),
            minimum_local: f64::INFINITY,
            operations,
            resident,
            admitted,
            failed: false,
            uncertified: false,
            peak_resident: resident,
            local_proofs: VecDeque::new(),
        })
    }

    fn admission_error(&self) -> EnergyError {
        EnergyError::Admission {
            operations: self.operations,
            operation_limit: self.admitted.update_limit(),
            resident: self.resident.max(self.peak_resident),
            resident_limit: self.admitted.resident_bytes_limit(),
        }
    }

    fn charge(&mut self, operations: u64) -> Result<(), EnergyError> {
        self.operations = self
            .operations
            .checked_add(operations)
            .ok_or(EnergyError::Arithmetic)?;
        if self.operations > self.admitted.update_limit() {
            return Err(self.admission_error());
        }
        Ok(())
    }

    /// Local matrices must use the same global indices and coefficients as assembly.
    /// Their rank is checked, rather than inferred from connectivity or a material label.
    pub fn push_block(&mut self, nodes: &[usize], values: &[f64]) -> Result<(), EnergyError> {
        if self.failed {
            return Err(EnergyError::InvalidBlock);
        }
        self.failed = true;
        let result = self.push_block_inner(nodes, values);
        if result.is_ok() {
            self.failed = false;
        }
        result
    }

    fn push_block_inner(&mut self, nodes: &[usize], values: &[f64]) -> Result<(), EnergyError> {
        check_cancelled()?;
        let n = nodes.len();
        if !(2..=32).contains(&n)
            || values.len() != n * n
            || values.iter().any(|v| !v.is_finite())
            || nodes.iter().any(|&i| i >= self.projection.len())
            || nodes
                .iter()
                .enumerate()
                .any(|(i, node)| nodes[..i].contains(node))
        {
            return Err(EnergyError::InvalidBlock);
        }
        if self.uncertified
            || nodes
                .iter()
                .all(|&node| self.projection.row(node).is_none())
        {
            return Ok(());
        }
        // Only Q enters the local proof. Reserve subtraction/absolute/max work
        // for every candidate coefficient; unused row-zero entries are not compared.
        let quotient_width = (n - 1) as u64;
        self.charge(3 * quotient_width.pow(2) * self.local_proofs.len() as u64 + 1)?;
        let scratch = (n as u64).pow(2) * 64 + (n as u64) * 128;
        self.peak_resident = self.peak_resident.max(
            self.resident
                .checked_add(scratch)
                .ok_or(EnergyError::Arithmetic)?,
        );
        if self.peak_resident > self.admitted.resident_bytes_limit() {
            return Err(self.admission_error());
        }
        // Closeness selects a candidate only. Every changed coefficient is enclosed
        // by the perturbation proof; this is never tolerance-based cache acceptance.
        let existing = self
            .local_proofs
            .iter()
            .filter(|(original, _)| original.len() == values.len())
            .map(|(original, lower)| {
                let distance = (1..n)
                    .flat_map(|i| (1..n).map(move |j| i * n + j))
                    .map(|index| (original[index] - values[index]).abs())
                    .fold(0.0, f64::max);
                (distance, original, lower)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, original, lower)| (original, *lower));
        // Release the cache borrow while charging the conservative numerical demand.
        let perturbation_work = if existing.is_some() {
            12 * quotient_width.pow(2) + 16 * quotient_width + 32
        } else {
            0
        };
        let pending_operations = self
            .operations
            .checked_add(perturbation_work)
            .ok_or(EnergyError::Arithmetic)?;
        if pending_operations > self.admitted.update_limit() {
            self.operations = pending_operations;
            return Err(self.admission_error());
        }
        let existing = existing
            .and_then(|(original, lower)| local::perturbation_lower(original, lower, values, n));
        self.operations = pending_operations;
        let lower = match existing {
            Some(lower) => lower,
            None => {
                // Candidate construction and outward verification both fit this arithmetic bound.
                self.charge(
                    (n as u64)
                        .pow(3)
                        .checked_mul(6)
                        .ok_or(EnergyError::Arithmetic)?,
                )?;
                let Some(lower) = local::lower_bound(values, n)? else {
                    // An element energy this sufficient check cannot certify leaves the
                    // global bound undecided; the remaining stream is not needed.
                    self.uncertified = true;
                    return Ok(());
                };
                if self.local_proofs.len() == LOCAL_PROOF_CACHE_CAPACITY {
                    let (removed, _) = self
                        .local_proofs
                        .pop_front()
                        .expect("bounded cache is nonempty");
                    self.resident -= removed.len() as u64 * 8 + 256;
                }
                let bytes = values.len() as u64 * 8 + 256;
                if self
                    .resident
                    .checked_add(bytes)
                    .and_then(|v| v.checked_add(scratch))
                    .is_some_and(|v| v <= self.admitted.resident_bytes_limit())
                {
                    self.local_proofs.push_back((values.to_vec(), lower));
                    self.resident += bytes;
                }
                lower
            }
        };
        self.minimum_local = self.minimum_local.min(lower);
        let added = ((n - 1) as u64)
            .checked_mul(2 * 8 * 4)
            .ok_or(EnergyError::Arithmetic)?;
        self.resident = self
            .resident
            .checked_add(added)
            .ok_or(EnergyError::Arithmetic)?;
        if self.resident > self.admitted.resident_bytes_limit() {
            return Err(self.admission_error());
        }
        for &node in &nodes[1..] {
            self.graph[nodes[0]].push(node);
            self.graph[node].push(nodes[0]);
        }
        // Compare the constrained free block of C' Q C directly. Identity rows
        // were seeded once in new(). Entries eliminated by prescribed rows need
        // neither reconstruction nor a second dense local block.
        for (i, &row) in nodes.iter().enumerate() {
            let Some(row) = self.projection.row(row) else {
                continue;
            };
            for (j, &column) in nodes.iter().enumerate() {
                let Some(column) = self.projection.row(column) else {
                    continue;
                };
                let mut contribution = Interval::exact(0.0);
                match (i, j) {
                    (0, 0) => {
                        self.charge(4 * quotient_width.pow(2))?;
                        for a in 1..n {
                            for b in 1..n {
                                contribution = contribution.add(Interval::exact(values[a * n + b]));
                            }
                        }
                    }
                    (0, _) => {
                        self.charge(4 * quotient_width)?;
                        for a in 1..n {
                            contribution = contribution.sub(Interval::exact(values[a * n + j]));
                        }
                    }
                    (_, 0) => {
                        self.charge(4 * quotient_width)?;
                        for b in 1..n {
                            contribution = contribution.sub(Interval::exact(values[i * n + b]));
                        }
                    }
                    _ => contribution = Interval::exact(values[i * n + j]),
                }
                // Two directed sums and their outward roundings.
                self.charge(4)?;
                if let Some(index) = self
                    .matrix
                    .pattern()
                    .row_range(row)
                    .find(|&k| self.matrix.pattern().column(k) == column)
                {
                    self.entries[index] = self.entries[index].add(contribution);
                } else {
                    if !self.missing.contains_key(&(row, column)) {
                        self.resident = self
                            .resident
                            .checked_add(MISSING_PAIR_RESIDENT_BYTES)
                            .ok_or(EnergyError::Arithmetic)?;
                        if self.resident > self.admitted.resident_bytes_limit() {
                            return Err(self.admission_error());
                        }
                    }
                    let entry = self
                        .missing
                        .entry((row, column))
                        .or_insert(Interval::exact(0.0));
                    *entry = entry.add(contribution);
                }
            }
        }
        self.peak_resident = self.peak_resident.max(
            self.resident
                .checked_add(scratch)
                .ok_or(EnergyError::Arithmetic)?,
        );
        if self.peak_resident > self.admitted.resident_bytes_limit() {
            return Err(self.admission_error());
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<EnergyVerdict, EnergyError> {
        if self.failed {
            return Err(EnergyError::InvalidBlock);
        }
        if self.uncertified {
            return Ok(EnergyVerdict::Indeterminate {
                graph_lower: 0.0,
                error_upper: 0.0,
            });
        }
        let n = self.matrix.rows();
        self.charge(
            ((self.matrix.values().len() + self.missing.len()) as u64)
                .checked_mul(16)
                .and_then(|v| v.checked_add((self.projection.len() as u64) * 16))
                .ok_or(EnergyError::Arithmetic)?,
        )?;
        let mut distances = vec![usize::MAX; self.projection.len()];
        let mut queue = VecDeque::new();
        for (i, distance) in distances.iter_mut().enumerate() {
            if self.projection.row(i).is_none() {
                *distance = 0;
                queue.push_back(i);
            }
        }
        while let Some(node) = queue.pop_front() {
            check_cancelled()?;
            for &neighbor in &self.graph[node] {
                if distances[neighbor] == usize::MAX {
                    distances[neighbor] = distances[node]
                        .checked_add(1)
                        .ok_or(EnergyError::Arithmetic)?;
                    queue.push_back(neighbor);
                }
            }
        }
        if distances.contains(&usize::MAX) {
            return Err(EnergyError::Unanchored);
        }
        let distance_upper = distances
            .iter()
            .fold(0.0_f64, |a, &d| (a + (d as f64).next_up()).next_up());
        let graph_lower = if distance_upper > 0.0 {
            (self.minimum_local / distance_upper).next_down().min(1.0)
        } else {
            1.0
        };
        let mut rows = vec![0.0_f64; n];
        let mut columns = vec![0.0_f64; n];
        for (row, bound) in rows.iter_mut().enumerate() {
            check_cancelled()?;
            for k in self.matrix.pattern().row_range(row) {
                let error = self.entries[k]
                    .sub(Interval::exact(self.matrix.values()[k]))
                    .absolute_upper();
                *bound = (*bound + error).next_up();
                let column = self.matrix.pattern().column(k);
                columns[column] = (columns[column] + error).next_up();
            }
        }
        for ((row, column), value) in self.missing {
            let error = value.absolute_upper();
            rows[row] = (rows[row] + error).next_up();
            columns[column] = (columns[column] + error).next_up();
        }
        let max_row = rows.into_iter().fold(0.0, f64::max);
        let max_column = columns.into_iter().fold(0.0, f64::max);
        let error_upper = ((max_row * max_column).next_up().sqrt()).next_up();
        let lower_bound = (graph_lower - error_upper).next_down();
        if !lower_bound.is_finite() || lower_bound <= 0.0 {
            return Ok(EnergyVerdict::Indeterminate {
                graph_lower,
                error_upper,
            });
        }
        Ok(EnergyVerdict::Verified(AnchoredEnergyCertificate {
            operator_identity: self
                .matrix
                .numerical_identity()
                .map_err(|_| EnergyError::Arithmetic)?,
            lower_bound,
            assembly_error_upper: error_upper,
            scalar_operations: self.operations,
            resident_bytes: self.peak_resident,
        }))
    }
}

fn check_cancelled() -> Result<(), EnergyError> {
    if crate::progress::is_cancelled() {
        Err(EnergyError::Cancelled)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
