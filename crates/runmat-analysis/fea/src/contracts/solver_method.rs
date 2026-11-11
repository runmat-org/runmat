//! Closed algebraic method evidence, independent of formulation and placement.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverMethod {
    DenseDirect,
    AssembledPcg,
    AssembledSaddlePointGmres,
    GeneralGmres,
    MatrixFreePcg,
    MatrixFreeSaddlePointGmres,
    /// Supernodal sparse Cholesky (nested dissection) with iterative refinement.
    SparseDirectCholesky,
    /// Sparse symmetric indefinite factorization (Bunch–Kaufman LBLᵀ, signed-regularized LDLᵀ
    /// fallback for saddle points) with refinement.
    SparseDirectLdlt,
    /// Sparse LU with partial pivoting (general and complex-symmetric systems).
    SparseDirectLu,
    /// Shift-invert thick-restart block Lanczos on `(K − σM)⁻¹M` with a Sturm check.
    ShiftInvertBlockLanczos,
    /// Block LOBPCG with an AMG preconditioner of `K`, no retained factorization, and
    /// factor-free inertia counts for coercivity and the Sturm check.
    PreconditionedLobpcg,
    /// Block Lanczos on `K⁻¹(−K_G)` in the `K` inner product (linear buckling).
    BucklingBlockLanczos,
    NewmarkAverageAccelerationPcg,
    StructuralModalTransientNewmarkDenseFactorization,
    StructuralDirectHarmonicGmres,
    StructuralModalHarmonicDenseDirect,
    StructuralModalResponseSpectrumCombination,
    StructuralRandomResponseIntegration,
    IncrementalNewtonRaphson,
    AcousticDomainGraphHelmholtzHarmonic,
    AcousticCanonicalHelmholtzCertifiedLu,
    AcousticCanonicalShiftInvertBlockKrylov,
    AcousticCanonicalNewmarkPcg,
    CfdVelocityPressureFiniteVolume,
}

impl SolverMethod {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DenseDirect => "dense_direct",
            Self::AssembledPcg => "assembled_pcg",
            Self::AssembledSaddlePointGmres => "assembled_saddle_point_gmres",
            Self::GeneralGmres => "general_gmres",
            Self::MatrixFreePcg => "matrix_free_pcg",
            Self::MatrixFreeSaddlePointGmres => "matrix_free_saddle_point_gmres",
            Self::SparseDirectCholesky => "sparse_direct_cholesky",
            Self::SparseDirectLdlt => "sparse_direct_ldlt",
            Self::SparseDirectLu => "sparse_direct_lu",
            Self::ShiftInvertBlockLanczos => "shift_invert_block_lanczos",
            Self::PreconditionedLobpcg => "preconditioned_lobpcg",
            Self::BucklingBlockLanczos => "buckling_block_lanczos",
            Self::NewmarkAverageAccelerationPcg => "newmark_average_acceleration_pcg",
            Self::StructuralModalTransientNewmarkDenseFactorization => {
                "structural_modal_transient_newmark_dense_factorization"
            }
            Self::StructuralDirectHarmonicGmres => "structural_direct_harmonic_gmres",
            Self::StructuralModalHarmonicDenseDirect => "structural_modal_harmonic_dense_direct",
            Self::StructuralModalResponseSpectrumCombination => {
                "structural_modal_response_spectrum_combination"
            }
            Self::StructuralRandomResponseIntegration => "structural_random_response_integration",
            Self::IncrementalNewtonRaphson => "incremental_newton_raphson",
            Self::AcousticDomainGraphHelmholtzHarmonic => {
                "acoustic_domain_graph_helmholtz_harmonic"
            }
            Self::AcousticCanonicalHelmholtzCertifiedLu => {
                "acoustic_canonical_helmholtz_certified_lu"
            }
            Self::AcousticCanonicalShiftInvertBlockKrylov => {
                "acoustic_canonical_shift_invert_block_krylov"
            }
            Self::AcousticCanonicalNewmarkPcg => "acoustic_canonical_newmark_pcg",
            Self::CfdVelocityPressureFiniteVolume => "cfd_velocity_pressure_finite_volume",
        }
    }
}

impl std::fmt::Display for SolverMethod {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}
