//! Storage and numerical workspace policy for admitted solves.

use serde::{Deserialize, Serialize};

use crate::algebra::CoercivityCertificate;

use super::contract::{ByteCount, Count};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreconditionerStorage {
    None,
    Diagonal,
    SparseCsr {
        nonzero_upper_bound: Count,
        resident_bytes_upper_bound: ByteCount,
    },
}

/// How an SPD static solve proves coercivity and solves; recorded in the plan so every
/// re-derivation and the numerical kernel follow the planned path.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StaticSolvePath {
    /// The sparse Cholesky factor proves coercivity and solves.
    ProvingFactorization,
    /// Element rigidity, face gluing and anchors prove the elastic operator positive definite
    /// (no global factor); AMG-preconditioned CG solves, accepted by its true residual.
    ElasticRigidityMultigrid,
}

/// Selects the static solve path while planning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StaticSolveSelection {
    /// Choose the factorization when its plan fits the effective memory ceiling,
    /// min(authored ceiling, this machine's admitted memory when known), and otherwise the
    /// elastic rigidity path when the model admits it.
    WithinCeiling { machine_memory_bytes: Option<u64> },
    /// Reproduce the path an admitted plan recorded.
    Recorded(StaticSolvePath),
}

/// How a symmetric eigen-search finds its modes and proves that no mode below the reported
/// ones was missed; recorded in the plan so every re-derivation and the solver follow it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubspaceSolvePath {
    /// The Cholesky factor that proves `K` coercive is the shift-invert operator of block
    /// Lanczos. The factor is released before the Sturm inertia count.
    ShiftInvertFactorization,
    /// Block LOBPCG with an AMG preconditioner of `K`; no factor is retained. Two factor-free
    /// inertia counts prove `K` positive definite and count the eigenvalues below the Sturm
    /// probe.
    PreconditionedLobpcg,
}

/// Selects the modal solve path while planning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModalSolveSelection {
    /// The shift-invert factorization when its plan fits the effective memory ceiling,
    /// min(authored ceiling, this machine's admitted memory when known); otherwise LOBPCG when
    /// its plan fits that ceiling.
    WithinCeiling { machine_memory_bytes: Option<u64> },
    /// Plan this path alone, as an authored `solver = factorization | iterative` requires.
    /// When its plan does not fit the effective ceiling, planning is refused with
    /// `ResourcePlanningError::MemoryBudget`; no other path is tried.
    Forced {
        path: SubspaceSolvePath,
        machine_memory_bytes: Option<u64>,
    },
    /// Reproduce the path an admitted plan recorded.
    Recorded(SubspaceSolvePath),
}

impl ModalSolveSelection {
    /// The selection the authored solver method asks for: `auto` keeps the ceiling choice, a
    /// named method forces its path. A recorded or already forced selection is kept.
    pub fn with_authored_method(self, method: runmat_analysis_core::ModalSolverMethod) -> Self {
        let path = match method {
            runmat_analysis_core::ModalSolverMethod::Auto => return self,
            runmat_analysis_core::ModalSolverMethod::Factorization => {
                SubspaceSolvePath::ShiftInvertFactorization
            }
            runmat_analysis_core::ModalSolverMethod::Iterative => {
                SubspaceSolvePath::PreconditionedLobpcg
            }
        };
        match self {
            Self::WithinCeiling {
                machine_memory_bytes,
            } => Self::Forced {
                path,
                machine_memory_bytes,
            },
            other => other,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericalScalar {
    RealF64,
    ComplexF64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverAdmissionResourcePolicy {
    pub dimension: Count,
    pub factor_nonzero_upper_bound: Count,
    pub update_count_upper_bound: Count,
    pub resident_bytes_upper_bound: ByteCount,
}

impl SolverAdmissionResourcePolicy {
    pub(crate) fn admits_certificate(&self, certificate: &CoercivityCertificate) -> bool {
        let (dimension, factor_nonzeros, updates, resident_bytes) = certificate.work_counts();
        dimension == self.dimension.0
            && factor_nonzeros <= self.factor_nonzero_upper_bound.0
            && updates <= self.update_count_upper_bound.0
            && resident_bytes <= self.resident_bytes_upper_bound.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveResourcePolicy {
    /// Serial real GENERAL solve with its complete independent uniqueness work.
    GeneralSparse(super::GeneralSparseSolveResourceDemand),
    NoSolve {
        workspace_bytes_upper_bound: ByteCount,
    },
    Iterative {
        vector_count: Count,
        scalar: NumericalScalar,
        preconditioner: PreconditionerStorage,
        admission: SolverAdmissionResourcePolicy,
        /// Peak retained index plus construction cursors for scoped affine Tᵀ application.
        transpose_index_bytes_upper_bound: ByteCount,
        path: StaticSolvePath,
    },
    DirectFactorization {
        vector_count: Count,
        scalar: NumericalScalar,
        factor_nonzero_upper_bound: Count,
        factor_bytes_upper_bound: ByteCount,
    },
    RestartedGmres {
        block_dimension: Count,
        restart: Count,
        maximum_iterations: Count,
        workspace_bytes_upper_bound: ByteCount,
    },
    /// Structural saddle-point GMRES, including the temporary reduced SPD proof.
    SaddlePointGmres {
        block_dimension: Count,
        restart: Count,
        maximum_iterations: Count,
        workspace_bytes_upper_bound: ByteCount,
        admission: SolverAdmissionResourcePolicy,
    },
    ModalDenseHarmonic {
        full_dimension: Count,
        mode_count: Count,
        workspace_bytes_upper_bound: ByteCount,
    },
    TransientBlock {
        physical_dimension: Count,
        coordinate_dimension: Count,
        workspace_bytes_upper_bound: ByteCount,
    },
    SymmetricSubspaceIteration {
        full_dimension: Count,
        reduced_dimension: Count,
        requested_mode_count: Count,
        initial_subspace_dimension: Count,
        maximum_subspace_dimension: Count,
        boundary_probe_width: Count,
        subspace_expansion_count_upper_bound: Count,
        maximum_iterations_per_expansion: Count,
        orthogonalization_passes_per_block: Count,
        dense_projected_value_count_upper_bound: Count,
        workspace_bytes_upper_bound: ByteCount,
        path: SubspaceSolvePath,
    },
}

impl SolveResourcePolicy {
    /// The recorded eigen-search path (`None` for policies without one).
    pub fn subspace_solve_path(&self) -> Option<SubspaceSolvePath> {
        match self {
            Self::SymmetricSubspaceIteration { path, .. } => Some(*path),
            _ => None,
        }
    }

    /// The recorded static solve path (`None` for policies without one).
    pub fn static_solve_path(&self) -> Option<StaticSolvePath> {
        match self {
            Self::Iterative { path, .. } => Some(*path),
            _ => None,
        }
    }

    pub(crate) fn transpose_index_bytes_upper_bound(&self) -> ByteCount {
        match self {
            Self::Iterative {
                transpose_index_bytes_upper_bound,
                ..
            } => *transpose_index_bytes_upper_bound,
            _ => ByteCount(0),
        }
    }

    pub(crate) fn vector_count(&self) -> Option<u64> {
        match self {
            Self::Iterative { vector_count, .. }
            | Self::DirectFactorization { vector_count, .. } => Some(vector_count.0),
            Self::GeneralSparse(_)
            | Self::NoSolve { .. }
            | Self::RestartedGmres { .. }
            | Self::SaddlePointGmres { .. }
            | Self::ModalDenseHarmonic { .. }
            | Self::TransientBlock { .. }
            | Self::SymmetricSubspaceIteration { .. } => None,
        }
    }

    pub(crate) fn scalar_bytes(&self) -> u64 {
        let scalar = match self {
            Self::Iterative { scalar, .. } | Self::DirectFactorization { scalar, .. } => scalar,
            Self::GeneralSparse(_)
            | Self::NoSolve { .. }
            | Self::RestartedGmres { .. }
            | Self::SaddlePointGmres { .. }
            | Self::ModalDenseHarmonic { .. }
            | Self::TransientBlock { .. }
            | Self::SymmetricSubspaceIteration { .. } => return 8,
        };
        match scalar {
            NumericalScalar::RealF64 => 8,
            NumericalScalar::ComplexF64 => 16,
        }
    }

    pub(crate) const fn explicit_workspace_bytes(&self) -> Option<ByteCount> {
        match self {
            Self::GeneralSparse(demand) => Some(demand.workspace_bytes()),
            Self::NoSolve {
                workspace_bytes_upper_bound,
            }
            | Self::RestartedGmres {
                workspace_bytes_upper_bound,
                ..
            }
            | Self::SaddlePointGmres {
                workspace_bytes_upper_bound,
                ..
            }
            | Self::ModalDenseHarmonic {
                workspace_bytes_upper_bound,
                ..
            }
            | Self::TransientBlock {
                workspace_bytes_upper_bound,
                ..
            }
            | Self::SymmetricSubspaceIteration {
                workspace_bytes_upper_bound,
                ..
            } => Some(*workspace_bytes_upper_bound),
            Self::Iterative { .. } | Self::DirectFactorization { .. } => None,
        }
    }
}
