use std::collections::BTreeMap;

use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{SparsePattern, SparsePatternError};

const VALUES_IDENTITY_PREFIX: &[u8] = b"runmat-fea-sparse-values/v1\0";
const VALUES_IDENTITY_DOMAIN: &str = "analysis.fea.sparse-values/v1";
const NUMERICAL_IDENTITY_PREFIX: &[u8] = b"runmat-fea-sparse-numerics\0";
const NUMERICAL_IDENTITY_DOMAIN: &str = "analysis.fea.sparse-numerics";
const VALUES_IDENTITY_LIMITS: CanonicalLimits =
    CanonicalLimits::new(512 * 1024 * 1024, 64_000_000, 4096, 16);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarDomain {
    Real,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarPrecision {
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorRole {
    Stiffness,
    GeometricStiffness,
    Mass,
    Damping,
    Conductivity,
    Capacity,
    EffectiveTangent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContributionDomain {
    Structural,
    Thermal,
    Electromagnetic,
    Acoustic,
    Fluid,
    Constraint,
    TimeIntegration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContributionId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatrixSymmetry {
    General,
    Symmetric,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatrixDefiniteness {
    Unknown,
    PositiveSemidefinite,
    PositiveDefinite,
    Indefinite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatrixProperties {
    pub symmetry: MatrixSymmetry,
    pub definiteness: MatrixDefiniteness,
}

impl MatrixProperties {
    pub const SYMMETRIC_POSITIVE_DEFINITE: Self = Self {
        symmetry: MatrixSymmetry::Symmetric,
        definiteness: MatrixDefiniteness::PositiveDefinite,
    };
    pub const SYMMETRIC_POSITIVE_SEMIDEFINITE: Self = Self {
        symmetry: MatrixSymmetry::Symmetric,
        definiteness: MatrixDefiniteness::PositiveSemidefinite,
    };
    pub const SYMMETRIC_INDEFINITE: Self = Self {
        symmetry: MatrixSymmetry::Symmetric,
        definiteness: MatrixDefiniteness::Indefinite,
    };
    pub const GENERAL: Self = Self {
        symmetry: MatrixSymmetry::General,
        definiteness: MatrixDefiniteness::Unknown,
    };
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContributionProvenance {
    pub domain: ContributionDomain,
    pub contribution_id: ContributionId,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RealSparseMatrixError {
    Pattern(SparsePatternError),
    ValueCountMismatch { expected: usize, actual: usize },
    DenseShapeMismatch { expected: usize, actual: usize },
    NonFiniteValue { entry: usize },
    StorageAllocation { elements: usize },
    OutsidePlannedPattern { row: usize, column: usize },
    ShapeMismatch,
}

impl std::fmt::Display for RealSparseMatrixError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid canonical real sparse matrix: {self:?}")
    }
}

impl std::error::Error for RealSparseMatrixError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RealSparseMatrixData")]
pub struct RealSparseMatrix {
    pattern: SparsePattern,
    values: std::sync::Arc<Vec<f64>>,
    scalar_domain: ScalarDomain,
    precision: ScalarPrecision,
    role: OperatorRole,
    properties: MatrixProperties,
    contributions: std::sync::Arc<Vec<ContributionProvenance>>,
}

#[derive(Deserialize)]
struct RealSparseMatrixData {
    pattern: SparsePattern,
    values: Vec<f64>,
    scalar_domain: ScalarDomain,
    precision: ScalarPrecision,
    role: OperatorRole,
    properties: MatrixProperties,
    contributions: Vec<ContributionProvenance>,
}

impl TryFrom<RealSparseMatrixData> for RealSparseMatrix {
    type Error = RealSparseMatrixError;

    fn try_from(data: RealSparseMatrixData) -> Result<Self, Self::Error> {
        if data.scalar_domain != ScalarDomain::Real || data.precision != ScalarPrecision::F64 {
            return Err(RealSparseMatrixError::ShapeMismatch);
        }
        Self::try_new(
            data.pattern,
            data.values,
            data.role,
            data.properties,
            data.contributions,
        )
    }
}

impl RealSparseMatrix {
    pub fn try_new(
        pattern: SparsePattern,
        values: Vec<f64>,
        role: OperatorRole,
        properties: MatrixProperties,
        contributions: Vec<ContributionProvenance>,
    ) -> Result<Self, RealSparseMatrixError> {
        if values.len() != pattern.column_indices().len() {
            return Err(RealSparseMatrixError::ValueCountMismatch {
                expected: pattern.column_indices().len(),
                actual: values.len(),
            });
        }
        if let Some(entry) = values.iter().position(|value| !value.is_finite()) {
            return Err(RealSparseMatrixError::NonFiniteValue { entry });
        }
        Ok(Self {
            pattern,
            values: std::sync::Arc::new(values),
            scalar_domain: ScalarDomain::Real,
            precision: ScalarPrecision::F64,
            role,
            properties,
            contributions: std::sync::Arc::new(contributions),
        })
    }

    pub fn diagonal(
        diagonal: Vec<f64>,
        role: OperatorRole,
        properties: MatrixProperties,
    ) -> Result<Self, RealSparseMatrixError> {
        let n = diagonal.len();
        let row_offsets = (0..=n)
            .map(|value| u64::try_from(value).map_err(|_| SparsePatternError::DimensionOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        let columns = (0..n)
            .map(|value| u64::try_from(value).map_err(|_| SparsePatternError::DimensionOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        Self::try_new(
            SparsePattern::try_new(
                u64::try_from(n).map_err(|_| SparsePatternError::DimensionOverflow)?,
                u64::try_from(n).map_err(|_| SparsePatternError::DimensionOverflow)?,
                row_offsets,
                columns,
            )?,
            diagonal,
            role,
            properties,
            Vec::new(),
        )
    }

    #[cfg(test)]
    pub fn from_dense(
        dense: &[f64],
        n: usize,
        role: OperatorRole,
        properties: MatrixProperties,
    ) -> Result<Self, RealSparseMatrixError> {
        if dense.len()
            != n.checked_mul(n)
                .ok_or(SparsePatternError::DimensionOverflow)?
        {
            return Err(RealSparseMatrixError::DenseShapeMismatch {
                expected: n.saturating_mul(n),
                actual: dense.len(),
            });
        }
        let mut assembler = RealSparseAssembler::new(n, n, role, properties);
        for row in 0..n {
            for column in 0..n {
                assembler.add(row, column, dense[row * n + column]);
            }
        }
        assembler.finish()
    }

    pub fn pattern(&self) -> &SparsePattern {
        &self.pattern
    }
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    pub fn scalar_domain(&self) -> ScalarDomain {
        self.scalar_domain
    }
    pub fn precision(&self) -> ScalarPrecision {
        self.precision
    }
    pub fn role(&self) -> OperatorRole {
        self.role
    }
    pub fn properties(&self) -> MatrixProperties {
        self.properties
    }
    pub fn contributions(&self) -> &[ContributionProvenance] {
        &self.contributions
    }
    pub fn with_contribution(mut self, provenance: ContributionProvenance) -> Self {
        std::sync::Arc::make_mut(&mut self.contributions).push(provenance);
        self
    }
    pub fn pattern_identity(&self) -> Result<Digest, CanonicalCodecError> {
        self.pattern.identity()
    }
    /// Stable revision identity for values and typed numerical metadata on this pattern.
    pub fn values_identity(&self) -> Result<Digest, CanonicalCodecError> {
        // Streamed: a mesh-sized operator is hashed without materializing its encoding.
        crate::contracts::canonical_digest::canonical_contract_digest(
            VALUES_IDENTITY_PREFIX,
            VALUES_IDENTITY_DOMAIN,
            &(
                self.pattern.identity()?,
                &self.values,
                self.scalar_domain,
                self.precision,
                self.role,
                self.properties,
                &self.contributions,
            ),
            VALUES_IDENTITY_LIMITS,
        )
    }
    /// Stable identity for the executable numerical matrix, excluding contribution provenance.
    ///
    /// Prepared-system caches may reuse this identity across assemblies with identical numerical
    /// coefficients but distinct load provenance. Changes to topology, coefficients, scalar
    /// representation, role, or declared properties still invalidate the identity.
    pub fn numerical_identity(&self) -> Result<Digest, CanonicalCodecError> {
        // Streamed: a mesh-sized operator is hashed without materializing its encoding.
        crate::contracts::canonical_digest::canonical_contract_digest(
            NUMERICAL_IDENTITY_PREFIX,
            NUMERICAL_IDENTITY_DOMAIN,
            &(
                self.pattern.identity()?,
                &self.values,
                self.scalar_domain,
                self.precision,
                self.role,
                self.properties,
            ),
            VALUES_IDENTITY_LIMITS,
        )
    }
    pub fn rows(&self) -> usize {
        self.pattern.rows()
    }
    pub fn columns(&self) -> usize {
        self.pattern.columns()
    }

    pub fn get(&self, row: usize, column: usize) -> f64 {
        for entry in self.pattern.row_range(row) {
            match self.pattern.column(entry).cmp(&column) {
                std::cmp::Ordering::Less => {}
                std::cmp::Ordering::Equal => return self.values[entry],
                std::cmp::Ordering::Greater => break,
            }
        }
        0.0
    }

    pub fn diagonal_values(&self) -> Vec<f64> {
        (0..self.rows()).map(|row| self.get(row, row)).collect()
    }

    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0; self.rows().saturating_mul(self.columns())];
        for row in 0..self.rows() {
            for entry in self.pattern.row_range(row) {
                dense[row * self.columns() + self.pattern.column(entry)] = self.values[entry];
            }
        }
        dense
    }

    /// [`Self::expanded_square`] that consumes the matrix: an unchanged size with no stored zero
    /// reuses the value storage instead of copying it. The result is identical.
    pub fn into_expanded_square(self, size: usize) -> Result<Self, RealSparseMatrixError> {
        if size == self.rows()
            && self.rows() == self.columns()
            && self.values.iter().all(|value| *value != 0.0)
        {
            return Ok(Self {
                pattern: self.pattern,
                values: self.values,
                scalar_domain: ScalarDomain::Real,
                precision: ScalarPrecision::F64,
                role: self.role,
                properties: self.properties,
                contributions: std::sync::Arc::new(Vec::new()),
            });
        }
        self.expanded_square(size)
    }

    pub fn expanded_square(&self, size: usize) -> Result<Self, RealSparseMatrixError> {
        if size < self.rows() || self.rows() != self.columns() {
            return Err(RealSparseMatrixError::ShapeMismatch);
        }
        if self.values.iter().all(|value| *value != 0.0) {
            // No stored zero to compact: reuse the canonical support (shared when the size is
            // unchanged) and append empty rows. Provenance is not carried, as before.
            let pattern = if size == self.rows() {
                self.pattern.clone()
            } else {
                let mut offsets = reserve_payload(size + 1)?;
                offsets.extend_from_slice(self.pattern.row_offsets());
                let last = *offsets.last().unwrap_or(&0);
                offsets.resize(size + 1, last);
                let mut columns = reserve_payload(self.pattern.column_indices().len())?;
                columns.extend_from_slice(self.pattern.column_indices());
                SparsePattern::try_new(size as u64, size as u64, offsets, columns)?
            };
            return Ok(Self {
                pattern,
                values: self.values.clone(),
                scalar_domain: ScalarDomain::Real,
                precision: ScalarPrecision::F64,
                role: self.role,
                properties: self.properties,
                contributions: std::sync::Arc::new(Vec::new()),
            });
        }
        let mut assembler = RealSparseAssembler::square(size, self.role, self.properties);
        for row in 0..self.rows() {
            for entry in self.pattern.row_range(row) {
                assembler.add(row, self.pattern.column(entry), self.values[entry]);
            }
        }
        assembler.finish()
    }

    pub fn principal_submatrix(&self, retained: &[usize]) -> Result<Self, RealSparseMatrixError> {
        if self.rows() != self.columns()
            || retained.windows(2).any(|pair| pair[0] >= pair[1])
            || retained.iter().any(|index| *index >= self.rows())
        {
            return Err(RealSparseMatrixError::ShapeMismatch);
        }
        let reverse = retained
            .iter()
            .enumerate()
            .map(|(reduced, original)| (*original, reduced))
            .collect::<BTreeMap<_, _>>();
        let mut assembler = RealSparseAssembler::square(retained.len(), self.role, self.properties);
        for contribution in self.contributions.iter() {
            assembler.add_contribution(contribution.clone());
        }
        for (reduced_row, original_row) in retained.iter().copied().enumerate() {
            for entry in self.pattern.row_range(original_row) {
                if let Some(reduced_column) = reverse.get(&self.pattern.column(entry)) {
                    assembler.add(reduced_row, *reduced_column, self.values[entry]);
                }
            }
        }
        assembler.finish()
    }

    pub fn with_identity_constraints(
        &self,
        constrained: &[bool],
    ) -> Result<Self, RealSparseMatrixError> {
        if self.rows() != self.columns() || constrained.len() != self.rows() {
            return Err(RealSparseMatrixError::ShapeMismatch);
        }
        // Select rows/columns directly from canonical CSR. Essential rows may introduce a
        // diagonal missing from the original pattern; no sparse-map discovery is needed.
        let entries = |row: usize| {
            constrained[row]
                .then_some((row as u64, 1.0))
                .into_iter()
                .chain(self.pattern.row_range(row).filter_map(move |entry| {
                    let column = self.pattern.column(entry);
                    (!constrained[row] && !constrained[column] && self.values[entry] != 0.0)
                        .then_some((column as u64, self.values[entry]))
                }))
        };
        let count = (0..self.rows()).try_fold(0_usize, |total, row| {
            total
                .checked_add(entries(row).count())
                .ok_or(SparsePatternError::DimensionOverflow)
        })?;
        let mut offsets = reserve_payload(
            self.rows()
                .checked_add(1)
                .ok_or(SparsePatternError::DimensionOverflow)?,
        )?;
        let mut columns = reserve_payload(count)?;
        let mut values = reserve_payload(count)?;
        offsets.push(0);
        for row in 0..self.rows() {
            for (column, value) in entries(row) {
                columns.push(column);
                values.push(value);
            }
            offsets.push(values.len() as u64);
        }
        let mut contributions = self.contributions.as_ref().clone();
        contributions.push(ContributionProvenance {
            domain: ContributionDomain::Constraint,
            contribution_id: ContributionId("essential_elimination".to_string()),
        });
        Self::try_new(
            SparsePattern::try_new(self.rows() as u64, self.columns() as u64, offsets, columns)?,
            values,
            self.role,
            self.properties,
            contributions,
        )
    }

    pub fn scaled_sum(
        terms: &[(f64, &Self)],
        role: OperatorRole,
        properties: MatrixProperties,
    ) -> Result<Self, RealSparseMatrixError> {
        let Some((_, first)) = terms.first() else {
            return Self::diagonal(Vec::new(), role, properties);
        };
        let (rows, columns) = (first.rows(), first.columns());
        if terms
            .iter()
            .any(|(_, matrix)| matrix.rows() != rows || matrix.columns() != columns)
        {
            return Err(RealSparseMatrixError::ShapeMismatch);
        }
        let mut contributions = Vec::new();
        for (_, matrix) in terms {
            for contribution in matrix.contributions.iter() {
                if !contributions.contains(contribution) {
                    contributions.push(contribution.clone());
                }
            }
        }
        // Row-wise merge of the sorted term rows. Every entry accumulates its nonzero scaled
        // contributions from zero in term order, and entries that sum to exactly zero are
        // dropped: the same values and support as accumulating term by term into a discovered
        // assembler, independent of the admitted thread count (rows are independent).
        const ROWS_PER_TASK: usize = 2048;
        let block_ids = (0..rows.div_ceil(ROWS_PER_TASK)).collect::<Vec<_>>();
        let blocks = runmat_cpu_execution::ordered_map(
            &block_ids,
            |&block| -> Result<(Vec<u64>, Vec<u64>, Vec<f64>), RealSparseMatrixError> {
                let range = block * ROWS_PER_TASK..((block + 1) * ROWS_PER_TASK).min(rows);
                let mut lengths = Vec::with_capacity(range.len());
                let mut row_columns = Vec::new();
                let mut row_values = Vec::new();
                let mut cursors = vec![0usize; terms.len()];
                for row in range {
                    let start = row_values.len();
                    for (cursor, (_, matrix)) in cursors.iter_mut().zip(terms) {
                        *cursor = matrix.pattern.row_range(row).start;
                    }
                    loop {
                        // Smallest pending column across terms.
                        let mut next = None::<usize>;
                        for (cursor, (_, matrix)) in cursors.iter().zip(terms) {
                            if *cursor < matrix.pattern.row_range(row).end {
                                let column = matrix.pattern.column(*cursor);
                                next = Some(next.map_or(column, |c: usize| c.min(column)));
                            }
                        }
                        let Some(column) = next else { break };
                        let mut sum = 0.0_f64;
                        let mut touched = false;
                        for (cursor, (scale, matrix)) in cursors.iter_mut().zip(terms) {
                            let end = matrix.pattern.row_range(row).end;
                            if *cursor < end && matrix.pattern.column(*cursor) == column {
                                let value = *scale * matrix.values[*cursor];
                                *cursor += 1;
                                if !value.is_finite() {
                                    return Err(RealSparseMatrixError::NonFiniteValue { entry: 0 });
                                }
                                if value != 0.0 {
                                    sum += value;
                                    touched = true;
                                }
                            }
                        }
                        if touched && sum != 0.0 {
                            row_columns.push(
                                u64::try_from(column)
                                    .map_err(|_| SparsePatternError::DimensionOverflow)?,
                            );
                            row_values.push(sum);
                        }
                    }
                    lengths.push((row_values.len() - start) as u64);
                }
                Ok((lengths, row_columns, row_values))
            },
        );
        let mut offsets = reserve_payload(rows + 1)?;
        offsets.push(0_u64);
        let mut column_indices = Vec::new();
        let mut values = Vec::new();
        for block in blocks {
            let (lengths, block_columns, block_values) = block?;
            for length in lengths {
                offsets.push(offsets.last().copied().unwrap_or(0) + length);
            }
            column_indices.extend_from_slice(&block_columns);
            values.extend_from_slice(&block_values);
        }
        Self::try_new(
            SparsePattern::try_new(rows as u64, columns as u64, offsets, column_indices)?,
            values,
            role,
            properties,
            contributions,
        )
    }
}
mod assembler;
pub use assembler::RealSparseAssembler;

fn reserve_payload<T>(elements: usize) -> Result<Vec<T>, RealSparseMatrixError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| RealSparseMatrixError::StorageAllocation { elements })?;
    Ok(values)
}

impl From<SparsePatternError> for RealSparseMatrixError {
    fn from(value: SparsePatternError) -> Self {
        Self::Pattern(value)
    }
}

#[cfg(test)]
mod tests;
