use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::Serialize;
use thiserror::Error;

use super::{
    MatrixSymmetry, NullspaceEliminationEvidence, PreparedNullspaceEvidence, RealSparseMatrix,
};

const CERTIFICATE_PREFIX: &[u8] = b"runmat-fea-coercivity-certificate/v1\0";
const CERTIFICATE_DOMAIN: &str = "fea-coercivity-certificate";
const PREPARED_DOMAIN: &str = "fea-prepared-solver-admissibility";
const RHS_DOMAIN: &str = "fea-coercivity-rhs";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);
const SYMMETRY_RELATIVE_TOLERANCE: f64 = 1.0e-12;

mod symbolic;

pub(crate) use symbolic::{CoercivitySymbolicPlan, CoercivityWorkEnvelope};

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CoercivityError {
    #[error("coercivity requires a nonempty square symmetric operator")]
    IncompatibleOperator,
    #[error("operator values do not satisfy the declared symmetry")]
    AsymmetricValues,
    #[error("operator is not numerically coercive at pivot {pivot}")]
    NonPositivePivot { pivot: usize },
    #[error("coercivity proof arithmetic overflowed")]
    WorkOverflow,
    #[error("coercivity proof exceeded its bounded sparse envelope at pivot {pivot}: factor entries {factor_nonzero_count}/{factor_nonzero_limit}, updates {update_count}/{update_limit}")]
    ResourceLimit {
        pivot: usize,
        factor_nonzero_count: u64,
        factor_nonzero_limit: u64,
        update_count: u64,
        update_limit: u64,
    },
    #[error("coercivity symbolic ordering could not cover the complete operator graph")]
    SymbolicOrdering,
    #[error("coercivity evidence does not match its prepared operator or right-hand side")]
    BindingMismatch,
    #[error("coercivity evidence identity encoding failed: {0}")]
    Encoding(String),
}

/// A process-local numerical certificate that the exact sparse operator admitted to an SPD
/// solver has strictly positive inertia. The factor itself is intentionally discarded; the
/// certificate retains enough immutable evidence to bind the completed proof to the operator.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CoercivityCertificate {
    pattern_identity: Digest,
    values_identity: Digest,
    dimension: usize,
    minimum_scaled_pivot_bits: u64,
    factor_nonzero_count: u64,
    update_count: u64,
    identity: Digest,
}

/// Binds algebraic coercivity, the exact prepared right-hand side, and optional physics-owned
/// null-space attribution. This evidence is never accepted from serialized callers.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PreparedSolverAdmissibility {
    coercivity: CoercivityCertificate,
    source_nullspace_identity: Option<Digest>,
    prepared_rhs_identity: Digest,
    identity: Digest,
}

impl CoercivityCertificate {
    pub(crate) fn work_counts(&self) -> (u64, u64, u64, u64) {
        let dimension = u64::try_from(self.dimension).expect("proved dimension fits u64");
        let resident_bytes =
            symbolic::symbolic_resident_bytes(dimension, self.factor_nonzero_count)
                .expect("proved symbolic residency fits u64");
        (
            dimension,
            self.factor_nonzero_count,
            self.update_count,
            resident_bytes,
        )
    }

    pub(super) fn prove(matrix: &RealSparseMatrix) -> Result<Self, CoercivityError> {
        let dimension = matrix.rows();
        if dimension == 0
            || matrix.columns() != dimension
            || matrix.properties().symmetry != MatrixSymmetry::Symmetric
        {
            return Err(CoercivityError::IncompatibleOperator);
        }
        let diagonal = matrix.diagonal_values();
        if let Some(pivot) = diagonal
            .iter()
            .position(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(CoercivityError::NonPositivePivot { pivot });
        }
        let scales = diagonal
            .iter()
            .map(|value| value.sqrt())
            .collect::<Vec<_>>();
        let symbolic = CoercivitySymbolicPlan::derive(matrix.pattern())?;
        symbolic.validate_for(matrix.pattern())?;
        let mut rows = symbolic.factor_rows();
        for row in 0..dimension {
            for entry in matrix.pattern().row_range(row) {
                let column = matrix.pattern().column(entry);
                let value = matrix.values()[entry];
                let transpose = matrix.get(column, row);
                let symmetry_scale = value.abs().max(transpose.abs()).max(1.0);
                if (value - transpose).abs() > symmetry_scale * SYMMETRY_RELATIVE_TOLERANCE {
                    return Err(CoercivityError::AsymmetricValues);
                }
                let ordered_row = symbolic.inverse_order()[row];
                let ordered_column = symbolic.inverse_order()[column];
                if ordered_row >= ordered_column {
                    let scaled = value / (scales[row] * scales[column]);
                    let entry = factor_entry_mut(&mut rows, ordered_row, ordered_column)?;
                    *entry = scaled;
                }
            }
        }

        let tolerance = f64::EPSILON * (dimension.max(1) as f64) * 64.0;
        let mut minimum_scaled_pivot = f64::INFINITY;
        for pivot_index in 0..dimension {
            let pivot = *factor_entry_mut(&mut rows, pivot_index, pivot_index)?;
            if !pivot.is_finite() || pivot <= tolerance {
                return Err(CoercivityError::NonPositivePivot {
                    pivot: symbolic.elimination_order()[pivot_index],
                });
            }
            minimum_scaled_pivot = minimum_scaled_pivot.min(pivot);
            let neighbors = &symbolic.pivot_neighbors()[pivot_index];
            let mut factors = Vec::with_capacity(neighbors.len());
            for row in neighbors.iter().copied() {
                let value = *factor_entry_mut(&mut rows, row, pivot_index)?;
                let factor = value / pivot;
                if !factor.is_finite() {
                    return Err(CoercivityError::NonPositivePivot {
                        pivot: symbolic.elimination_order()[pivot_index],
                    });
                }
                *factor_entry_mut(&mut rows, row, pivot_index)? = factor;
                factors.push((row, factor));
            }
            for left in 0..factors.len() {
                for right in 0..=left {
                    let (left_row, left_factor) = factors[left];
                    let (right_row, right_factor) = factors[right];
                    let (row, column) = if left_row >= right_row {
                        (left_row, right_row)
                    } else {
                        (right_row, left_row)
                    };
                    let update = left_factor * pivot * right_factor;
                    let entry = factor_entry_mut(&mut rows, row, column)?;
                    *entry -= update;
                    if !entry.is_finite() {
                        return Err(CoercivityError::NonPositivePivot {
                            pivot: symbolic.elimination_order()[pivot_index],
                        });
                    }
                }
            }
        }
        let pattern_identity = matrix
            .pattern_identity()
            .map_err(|error| CoercivityError::Encoding(error.to_string()))?;
        let values_identity = matrix
            .values_identity()
            .map_err(|error| CoercivityError::Encoding(error.to_string()))?;
        let mut certificate = Self {
            pattern_identity,
            values_identity,
            dimension,
            minimum_scaled_pivot_bits: minimum_scaled_pivot.to_bits(),
            factor_nonzero_count: symbolic.factor_nonzero_count(),
            update_count: symbolic.update_count(),
            identity: Digest::sha256([]),
        };
        certificate.identity = certificate.expected_identity()?;
        Ok(certificate)
    }

    pub(crate) fn validate_for(&self, matrix: &RealSparseMatrix) -> Result<(), CoercivityError> {
        if self.dimension != matrix.rows()
            || self.pattern_identity
                != matrix
                    .pattern_identity()
                    .map_err(|error| CoercivityError::Encoding(error.to_string()))?
            || self.values_identity
                != matrix
                    .values_identity()
                    .map_err(|error| CoercivityError::Encoding(error.to_string()))?
            || self.minimum_scaled_pivot_bits == 0
            || !f64::from_bits(self.minimum_scaled_pivot_bits).is_finite()
            || self.identity != self.expected_identity()?
        {
            return Err(CoercivityError::BindingMismatch);
        }
        Ok(())
    }

    fn expected_identity(&self) -> Result<Digest, CoercivityError> {
        digest(
            CERTIFICATE_DOMAIN,
            &(
                self.pattern_identity,
                self.values_identity,
                self.dimension,
                self.minimum_scaled_pivot_bits,
                self.factor_nonzero_count,
                self.update_count,
            ),
        )
    }
}

fn factor_entry_mut(
    rows: &mut [Vec<(usize, f64)>],
    row: usize,
    column: usize,
) -> Result<&mut f64, CoercivityError> {
    let entries = rows.get_mut(row).ok_or(CoercivityError::SymbolicOrdering)?;
    let entry = entries
        .binary_search_by_key(&column, |(candidate, _)| *candidate)
        .map_err(|_| CoercivityError::SymbolicOrdering)?;
    Ok(&mut entries[entry].1)
}

impl PreparedSolverAdmissibility {
    pub(super) fn prove(
        matrix: &RealSparseMatrix,
        rhs: &[f64],
        source: Option<&NullspaceEliminationEvidence>,
    ) -> Result<Self, CoercivityError> {
        let source_prepared = source
            .map(|source| source.bind_prepared(matrix, rhs))
            .transpose()
            .map_err(|error| CoercivityError::Encoding(error.to_string()))?;
        let coercivity = CoercivityCertificate::prove(matrix)?;
        let prepared_rhs_identity = digest(RHS_DOMAIN, &rhs)?;
        let mut evidence = Self {
            coercivity,
            source_nullspace_identity: source_prepared
                .as_ref()
                .map(PreparedNullspaceEvidence::identity),
            prepared_rhs_identity,
            identity: Digest::sha256([]),
        };
        evidence.identity = evidence.expected_identity()?;
        Ok(evidence)
    }

    pub(super) fn validate_for(
        &self,
        matrix: &RealSparseMatrix,
        rhs: &[f64],
    ) -> Result<(), CoercivityError> {
        self.coercivity.validate_for(matrix)?;
        if self.prepared_rhs_identity != digest(RHS_DOMAIN, &rhs)?
            || self.identity != self.expected_identity()?
        {
            return Err(CoercivityError::BindingMismatch);
        }
        Ok(())
    }

    pub(crate) fn coercivity(&self) -> &CoercivityCertificate {
        &self.coercivity
    }

    pub(crate) fn bind_existing(
        coercivity: CoercivityCertificate,
        matrix: &RealSparseMatrix,
        rhs: &[f64],
    ) -> Result<Self, CoercivityError> {
        coercivity.validate_for(matrix)?;
        let prepared_rhs_identity = digest(RHS_DOMAIN, &rhs)?;
        let mut evidence = Self {
            coercivity,
            source_nullspace_identity: None,
            prepared_rhs_identity,
            identity: Digest::sha256([]),
        };
        evidence.identity = evidence.expected_identity()?;
        evidence.validate_for(matrix, rhs)?;
        Ok(evidence)
    }

    fn expected_identity(&self) -> Result<Digest, CoercivityError> {
        digest(
            PREPARED_DOMAIN,
            &(
                self.coercivity.identity,
                self.source_nullspace_identity,
                self.prepared_rhs_identity,
            ),
        )
    }
}

fn digest<T: Serialize>(domain: &'static str, value: &T) -> Result<Digest, CoercivityError> {
    encode_contract(CERTIFICATE_PREFIX, domain, value, LIMITS)
        .map(Digest::sha256)
        .map_err(|error| CoercivityError::Encoding(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler};

    fn matrix(values: [[f64; 3]; 3]) -> RealSparseMatrix {
        let mut assembler = RealSparseAssembler::square(
            3,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        );
        for (row, values) in values.into_iter().enumerate() {
            for (column, value) in values.into_iter().enumerate() {
                assembler.add(row, column, value);
            }
        }
        assembler.finish().unwrap()
    }

    #[test]
    fn sparse_inertia_proves_fill_requiring_spd_operator_and_binds_rhs() {
        let operator = matrix([[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]]);
        let evidence =
            PreparedSolverAdmissibility::prove(&operator, &[1.0, 2.0, 3.0], None).unwrap();
        let symbolic = CoercivitySymbolicPlan::derive(operator.pattern()).unwrap();
        assert_eq!(
            evidence.coercivity.factor_nonzero_count,
            symbolic.factor_nonzero_count()
        );
        assert_eq!(evidence.coercivity.update_count, symbolic.update_count());
        evidence.validate_for(&operator, &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            evidence.validate_for(&operator, &[1.0, 2.0, 4.0]),
            Err(CoercivityError::BindingMismatch)
        );
    }

    #[test]
    fn positive_definite_metadata_cannot_hide_an_extra_mechanism_or_negative_mode() {
        let mechanism = matrix([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        let indefinite = matrix([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert!(matches!(
            CoercivityCertificate::prove(&mechanism),
            Err(CoercivityError::NonPositivePivot { .. })
        ));
        assert!(matches!(
            CoercivityCertificate::prove(&indefinite),
            Err(CoercivityError::NonPositivePivot { .. })
        ));
    }

    #[test]
    fn symmetric_metadata_cannot_hide_asymmetric_values() {
        let asymmetric = matrix([[2.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(
            CoercivityCertificate::prove(&asymmetric),
            Err(CoercivityError::AsymmetricValues)
        );
    }
}
