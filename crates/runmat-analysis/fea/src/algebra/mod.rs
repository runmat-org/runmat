mod compensated;
pub(crate) mod capacity;
pub(crate) use compensated::CompensatedSum;
pub(crate) mod anchored_energy;
pub use anchored_energy::EnergyError;
pub(crate) mod coercivity;
pub(crate) mod elastic_rigidity;
pub use elastic_rigidity::ElasticRigidityError;
pub mod element_assembly;
mod compatibility;
mod matrix;
mod nullspace;
mod operator;
mod physical_operator;
pub(crate) use physical_operator::PhysicalScalarSystem;
mod pattern;
mod reference;
mod right_inverse;
pub(crate) mod storage;

pub(crate) use coercivity::CoercivityWorkEnvelope;
pub use coercivity::NumericalCapacityPhase;
pub use coercivity::{CoercivityCertificate, CoercivityError, PreparedSolverAdmissibility};
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
    connected_components, prepare_scalar_component_gauges, ConnectorAnchoredSolidInput,
    ConnectorAnchoredStructuralFrameInput, MixedStructuralNullspaceInput,
};
pub use nullspace::{
    NullspaceEliminationEvidence, NullspaceEvidenceError, NullspaceKind, PreparedNullspaceEvidence,
};
pub use operator::{
    apply_c, apply_k, apply_m, apply_prepared, AssembledRealSystem, CanonicalRealOperators,
    PreparedModalSystem, PreparedRealLinearSystem, RealSystemError,
};
pub use pattern::{SparseIndexWidth, SparsePattern, SparsePatternError, SparsePatternStatistics};
pub(crate) use reference::dot_canonical;
pub use reference::{
    axpy_scoped, direction_update_scoped, dot_scoped, normalized_copy_scoped, residual,
    scale_scoped, spmv, spmv_scoped, subtract_into_scoped, vector_sub_scoped, ScopedSpmvError,
    SpmvWorkError,
};
pub use right_inverse::{
    RightInverseCertificate, RightInverseCheck, RightInverseError, RightInverseVerdict,
    RightInverseWorkEnvelope,
};

pub(crate) use operator::LinearPreparationReuse;

mod normalized_vector;
pub use normalized_vector::{
    normalized_vector, NormalizedVectorDifferential, NormalizedVectorError,
};
