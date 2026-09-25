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
    },
}

impl SolveResourcePolicy {
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
            Self::NoSolve { .. }
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
            Self::NoSolve { .. }
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
