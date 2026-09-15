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
    values: Vec<f64>,
    scalar_domain: ScalarDomain,
    precision: ScalarPrecision,
    role: OperatorRole,
    properties: MatrixProperties,
    contributions: Vec<ContributionProvenance>,
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
            values,
            scalar_domain: ScalarDomain::Real,
            precision: ScalarPrecision::F64,
            role,
            properties,
            contributions,
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
        self.contributions.push(provenance);
        self
    }
    pub fn pattern_identity(&self) -> Result<Digest, CanonicalCodecError> {
        self.pattern.identity()
    }
    /// Stable revision identity for values and typed numerical metadata on this pattern.
    pub fn values_identity(&self) -> Result<Digest, CanonicalCodecError> {
        runmat_canonical_codec::encode_contract(
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
        .map(Digest::sha256)
    }
    /// Stable identity for the executable numerical matrix, excluding contribution provenance.
    ///
    /// Prepared-system caches may reuse this identity across assemblies with identical numerical
    /// coefficients but distinct load provenance. Changes to topology, coefficients, scalar
    /// representation, role, or declared properties still invalidate the identity.
    pub fn numerical_identity(&self) -> Result<Digest, CanonicalCodecError> {
        runmat_canonical_codec::encode_contract(
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
        .map(Digest::sha256)
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

    pub fn expanded_square(&self, size: usize) -> Result<Self, RealSparseMatrixError> {
        if size < self.rows() || self.rows() != self.columns() {
            return Err(RealSparseMatrixError::ShapeMismatch);
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
        for contribution in &self.contributions {
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
        let mut assembler = RealSparseAssembler::square(self.rows(), self.role, self.properties);
        for contribution in &self.contributions {
            assembler.add_contribution(contribution.clone());
        }
        assembler.add_contribution(ContributionProvenance {
            domain: ContributionDomain::Constraint,
            contribution_id: ContributionId("essential_elimination".to_string()),
        });
        for row in 0..self.rows() {
            if constrained[row] {
                assembler.add(row, row, 1.0);
                continue;
            }
            for entry in self.pattern.row_range(row) {
                let column = self.pattern.column(entry);
                if !constrained[column] {
                    assembler.add(row, column, self.values[entry]);
                }
            }
        }
        assembler.finish()
    }

    pub fn scaled_sum(
        terms: &[(f64, &Self)],
        role: OperatorRole,
        properties: MatrixProperties,
    ) -> Result<Self, RealSparseMatrixError> {
        let Some((_, first)) = terms.first() else {
            return Self::diagonal(Vec::new(), role, properties);
        };
        let mut assembler =
            RealSparseAssembler::new(first.rows(), first.columns(), role, properties);
        for (scale, matrix) in terms {
            if matrix.rows() != first.rows() || matrix.columns() != first.columns() {
                return Err(RealSparseMatrixError::ShapeMismatch);
            }
            for contribution in &matrix.contributions {
                if !assembler.contributions.contains(contribution) {
                    assembler.add_contribution(contribution.clone());
                }
            }
            for row in 0..matrix.rows() {
                for entry in matrix.pattern.row_range(row) {
                    assembler.add(
                        row,
                        matrix.pattern.column(entry),
                        *scale * matrix.values[entry],
                    );
                }
            }
        }
        assembler.finish()
    }
}

pub struct RealSparseAssembler {
    columns: usize,
    rows: Vec<BTreeMap<usize, f64>>,
    role: OperatorRole,
    properties: MatrixProperties,
    contributions: Vec<ContributionProvenance>,
    error: Option<RealSparseMatrixError>,
}

impl RealSparseAssembler {
    pub fn new(
        rows: usize,
        columns: usize,
        role: OperatorRole,
        properties: MatrixProperties,
    ) -> Self {
        Self {
            columns,
            rows: (0..rows).map(|_| BTreeMap::new()).collect(),
            role,
            properties,
            contributions: Vec::new(),
            error: None,
        }
    }

    pub fn square(size: usize, role: OperatorRole, properties: MatrixProperties) -> Self {
        Self::new(size, size, role, properties)
    }

    pub fn add(&mut self, row: usize, column: usize, value: f64) {
        if row >= self.rows.len() || column >= self.columns {
            self.error
                .get_or_insert(RealSparseMatrixError::ShapeMismatch);
            return;
        }
        if !value.is_finite() {
            self.error
                .get_or_insert(RealSparseMatrixError::NonFiniteValue { entry: 0 });
            return;
        }
        if value != 0.0 {
            *self.rows[row].entry(column).or_insert(0.0) += value;
        }
    }

    pub fn add_contribution(&mut self, provenance: ContributionProvenance) {
        self.contributions.push(provenance);
    }

    pub fn finish(self) -> Result<RealSparseMatrix, RealSparseMatrixError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        let row_count = self.rows.len();
        let mut offsets = Vec::with_capacity(self.rows.len() + 1);
        let mut columns = Vec::new();
        let mut values = Vec::new();
        offsets.push(0_u64);
        for row in self.rows {
            for (column, value) in row {
                if value != 0.0 {
                    columns.push(
                        u64::try_from(column).map_err(|_| SparsePatternError::DimensionOverflow)?,
                    );
                    values.push(value);
                }
            }
            offsets.push(
                u64::try_from(values.len()).map_err(|_| SparsePatternError::DimensionOverflow)?,
            );
        }
        let pattern = SparsePattern::try_new(
            u64::try_from(row_count).map_err(|_| SparsePatternError::DimensionOverflow)?,
            u64::try_from(self.columns).map_err(|_| SparsePatternError::DimensionOverflow)?,
            offsets,
            columns,
        )?;
        RealSparseMatrix::try_new(
            pattern,
            values,
            self.role,
            self.properties,
            self.contributions,
        )
    }
}

impl From<SparsePatternError> for RealSparseMatrixError {
    fn from(value: SparsePatternError) -> Self {
        Self::Pattern(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_identity_is_separate_from_pattern_identity() {
        let first = RealSparseMatrix::diagonal(
            vec![1.0, 2.0],
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let changed_values = RealSparseMatrix::diagonal(
            vec![1.0, 3.0],
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let changed_metadata = RealSparseMatrix::diagonal(
            vec![1.0, 2.0],
            OperatorRole::Mass,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();

        assert_eq!(
            first.pattern_identity().unwrap(),
            changed_values.pattern_identity().unwrap()
        );
        assert_ne!(
            first.values_identity().unwrap(),
            changed_values.values_identity().unwrap()
        );
        assert_ne!(
            first.values_identity().unwrap(),
            changed_metadata.values_identity().unwrap()
        );

        let changed_provenance = first.clone().with_contribution(ContributionProvenance {
            domain: ContributionDomain::TimeIntegration,
            contribution_id: ContributionId("sample-1".to_string()),
        });
        assert_ne!(
            first.values_identity().unwrap(),
            changed_provenance.values_identity().unwrap()
        );
        assert_eq!(
            first.numerical_identity().unwrap(),
            changed_provenance.numerical_identity().unwrap()
        );
    }

    #[test]
    fn assembler_reports_invalid_admission_without_panicking() {
        let mut assembler =
            RealSparseAssembler::square(1, OperatorRole::Stiffness, MatrixProperties::GENERAL);
        assembler.add(1, 0, 1.0);
        assert_eq!(
            assembler.finish(),
            Err(RealSparseMatrixError::ShapeMismatch)
        );
    }
}
