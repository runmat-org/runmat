use std::fmt;

use super::matrix::{MatrixDefiniteness, MatrixSymmetry};
use super::{PreparedRealLinearSystem, RealSparseMatrix};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealSolver {
    Direct,
    ConjugateGradient,
    Gmres,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealPreconditioner {
    None,
    Jacobi,
    Ilu0,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompatibilityError {
    NonSquare,
    ConjugateGradientRequiresSymmetry,
    ConjugateGradientRequiresPositiveDefiniteness,
    ConjugateGradientRequiresNullspaceCompatibility,
    Ilu0RequiresSquareOperator,
    UnsupportedOperatorRole,
}

impl fmt::Display for CompatibilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "incompatible real sparse solve: {self:?}")
    }
}

impl std::error::Error for CompatibilityError {}

pub fn validate_solver_compatibility(
    matrix: &RealSparseMatrix,
    solver: RealSolver,
    preconditioner: RealPreconditioner,
) -> Result<(), CompatibilityError> {
    if matrix.rows() != matrix.columns() {
        return if preconditioner == RealPreconditioner::Ilu0 {
            Err(CompatibilityError::Ilu0RequiresSquareOperator)
        } else {
            Err(CompatibilityError::NonSquare)
        };
    }
    if !matches!(
        matrix.role(),
        super::OperatorRole::Stiffness
            | super::OperatorRole::Conductivity
            | super::OperatorRole::EffectiveTangent
    ) {
        return Err(CompatibilityError::UnsupportedOperatorRole);
    }
    if solver == RealSolver::ConjugateGradient {
        if matrix.properties().symmetry != MatrixSymmetry::Symmetric {
            return Err(CompatibilityError::ConjugateGradientRequiresSymmetry);
        }
        match matrix.properties().definiteness {
            MatrixDefiniteness::PositiveDefinite => {}
            MatrixDefiniteness::PositiveSemidefinite => {
                return Err(CompatibilityError::ConjugateGradientRequiresNullspaceCompatibility)
            }
            MatrixDefiniteness::Indefinite | MatrixDefiniteness::Unknown => {
                return Err(CompatibilityError::ConjugateGradientRequiresPositiveDefiniteness)
            }
        }
    }
    Ok(())
}

pub fn validate_prepared_solver_compatibility(
    system: &PreparedRealLinearSystem,
    solver: RealSolver,
    preconditioner: RealPreconditioner,
) -> Result<(), CompatibilityError> {
    match validate_solver_compatibility(system.matrix(), solver, preconditioner) {
        Err(CompatibilityError::ConjugateGradientRequiresNullspaceCompatibility)
            if solver == RealSolver::ConjugateGradient && system.nullspace_evidence().is_some() =>
        {
            Ok(())
        }
        result => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{MatrixProperties, OperatorRole, RealSparseMatrix};

    #[test]
    fn incompatible_cg_operator_is_rejected() {
        let matrix = RealSparseMatrix::diagonal(
            vec![-1.0],
            OperatorRole::Stiffness,
            MatrixProperties::GENERAL,
        )
        .unwrap();
        assert_eq!(
            validate_solver_compatibility(
                &matrix,
                RealSolver::ConjugateGradient,
                RealPreconditioner::Jacobi
            ),
            Err(CompatibilityError::ConjugateGradientRequiresSymmetry)
        );
    }

    #[test]
    fn semidefinite_cg_requires_explicit_nullspace_compatibility() {
        let matrix = RealSparseMatrix::diagonal(
            vec![1.0, 0.0],
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .unwrap();
        assert_eq!(
            validate_solver_compatibility(
                &matrix,
                RealSolver::ConjugateGradient,
                RealPreconditioner::Jacobi,
            ),
            Err(CompatibilityError::ConjugateGradientRequiresNullspaceCompatibility)
        );
    }
}
