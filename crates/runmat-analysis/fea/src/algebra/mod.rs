mod coercivity;
mod compatibility;
mod matrix;
mod nullspace;
mod operator;
mod pattern;
mod reference;

pub use coercivity::{CoercivityCertificate, CoercivityError, PreparedSolverAdmissibility};
pub(crate) use coercivity::{CoercivitySymbolicPlan, CoercivityWorkEnvelope};
pub use compatibility::{
    validate_prepared_solver_compatibility, validate_solver_compatibility, CompatibilityError,
    RealPreconditioner, RealSolver,
};
pub use matrix::{
    ContributionDomain, ContributionId, ContributionProvenance, MatrixDefiniteness,
    MatrixProperties, MatrixSymmetry, OperatorRole, RealSparseAssembler, RealSparseMatrix,
    RealSparseMatrixError, ScalarDomain, ScalarPrecision,
};
pub(crate) use nullspace::{
    connected_components, ConnectorAnchoredSolidInput, ConnectorAnchoredStructuralFrameInput,
    MixedStructuralNullspaceInput,
};
pub use nullspace::{
    NullspaceEliminationEvidence, NullspaceEvidenceError, NullspaceKind, PreparedNullspaceEvidence,
};
pub use operator::{
    apply_c, apply_k, apply_m, apply_prepared, AssembledRealSystem, CanonicalRealOperators,
    PreparedModalSystem, PreparedRealLinearSystem, RealSystemError,
};
pub use pattern::{SparseIndexWidth, SparsePattern, SparsePatternError};
pub(crate) use reference::dot_canonical;
pub use reference::{
    axpy_scoped, direction_update_scoped, dot_scoped, normalized_copy_scoped, residual,
    scale_scoped, spmv, spmv_scoped, subtract_into_scoped, vector_sub_scoped, ScopedSpmvError,
    SpmvWorkError,
};
