mod compatibility;
mod matrix;
mod operator;
mod pattern;
mod reference;

pub use compatibility::{
    validate_solver_compatibility, CompatibilityError, RealPreconditioner, RealSolver,
};
pub use matrix::{
    ContributionDomain, ContributionId, ContributionProvenance, MatrixDefiniteness,
    MatrixProperties, MatrixSymmetry, OperatorRole, RealSparseAssembler, RealSparseMatrix,
    RealSparseMatrixError, ScalarDomain, ScalarPrecision,
};
pub use operator::{
    apply_c, apply_k, apply_m, apply_prepared, AssembledRealSystem, CanonicalRealOperators,
    PreparedModalSystem, PreparedRealLinearSystem, RealSystemError,
};
pub use pattern::{SparseIndexWidth, SparsePattern, SparsePatternError};
pub use reference::{residual, spmv};
