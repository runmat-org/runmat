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
    MatrixFreeSubspaceIteration,
    MatrixFreeBucklingSubspaceIteration,
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
    ChtConjugateProjection,
    FsiPartitionedProjection,
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
            Self::MatrixFreeSubspaceIteration => "matrix_free_subspace_iteration",
            Self::MatrixFreeBucklingSubspaceIteration => "matrix_free_buckling_subspace_iteration",
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
            Self::AcousticCanonicalHelmholtzCertifiedLu => "acoustic_canonical_helmholtz_certified_lu",
            Self::AcousticCanonicalShiftInvertBlockKrylov => {
                "acoustic_canonical_shift_invert_block_krylov"
            }
            Self::AcousticCanonicalNewmarkPcg => "acoustic_canonical_newmark_pcg",
            Self::CfdVelocityPressureFiniteVolume => "cfd_velocity_pressure_finite_volume",
            Self::ChtConjugateProjection => "cht_conjugate_projection",
            Self::FsiPartitionedProjection => "fsi_partitioned_projection",
        }
    }
}

impl std::fmt::Display for SolverMethod {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}
