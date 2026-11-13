use runmat_analysis_core::{CouplingImplementation, SolutionSpace, TemporalImplementation};
use thiserror::Error;

use crate::admission::AnalysisRunIdentity;

use super::{SolverMethod, SolverPreconditioner};

mod run_result;

pub(super) use run_result::FeaRunResultWire;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SolverFormulationCompatibilityError {
    #[error("structural static algebraic convergence is invalid")]
    InvalidStructuralStaticConvergence,
    #[error("structural static result lacks original typed convergence")]
    MissingStructuralStaticConvergence,
    #[error("structural static convergence is attached to another procedure")]
    UnexpectedStructuralStaticConvergence,
    #[error("structural static balance evidence is invalid")]
    InvalidStructuralStaticBalance,
    #[error("a structural static result is missing mandatory physical balance evidence")]
    MissingStructuralStaticBalance,
    #[error("structural static balance evidence is attached to a non-static result")]
    UnexpectedStructuralStaticBalance,
    #[error("structural constraint result is invalid")]
    InvalidStructuralConstraintResult,
    #[error("a structural equilibrium result is missing its canonical constraint result")]
    MissingStructuralConstraintResult,
    #[error("a structural constraint result is attached to an unsupported formulation")]
    UnexpectedStructuralConstraintResult,
    #[error("structural connector result is invalid")]
    InvalidStructuralConnectorResult,
    #[error("a structural result is missing its canonical connector result")]
    MissingStructuralConnectorResult,
    #[error("a structural connector result is attached to a formulation without structural connector state")]
    UnexpectedStructuralConnectorResult,
    #[error("the admitted formulation has no solver-owning temporal implementation")]
    MissingSolverTemporal,
    #[error(
        "the admitted formulation has multiple solver-owning temporal implementations: {first:?} and {second:?}"
    )]
    MultipleSolverTemporals {
        first: TemporalImplementation,
        second: TemporalImplementation,
    },
    #[error(
        "solver method {method} is incompatible with temporal implementation {temporal:?}, solution space {solution_space:?}, and coupling {coupling:?}"
    )]
    UnsupportedMethod {
        temporal: TemporalImplementation,
        solution_space: SolutionSpace,
        coupling: CouplingImplementation,
        method: SolverMethod,
    },
    #[error("preconditioner {preconditioner} is incompatible with solver method {method}")]
    UnsupportedPreconditioner {
        method: SolverMethod,
        preconditioner: SolverPreconditioner,
    },
}

/// Validates the solver actually reported by an execution against the exact admitted formulation.
///
/// Formulation identity owns the physical and temporal algorithms. Solver evidence owns the
/// algebraic method and preconditioner actually used. This boundary keeps those independently
/// useful contracts consistent without deriving either one from serialized caller assertions.
pub fn validate_solver_formulation_compatibility(
    identity: &AnalysisRunIdentity,
    method: SolverMethod,
    preconditioner: SolverPreconditioner,
) -> Result<(), SolverFormulationCompatibilityError> {
    let realization = identity.realization();
    let mut solver_temporal = None;
    for temporal in realization.temporals.iter().copied() {
        if is_auxiliary_temporal(temporal) {
            continue;
        }
        if let Some(first) = solver_temporal {
            return Err(
                SolverFormulationCompatibilityError::MultipleSolverTemporals {
                    first,
                    second: temporal,
                },
            );
        }
        solver_temporal = Some(temporal);
    }
    let temporal =
        solver_temporal.ok_or(SolverFormulationCompatibilityError::MissingSolverTemporal)?;
    if !method_is_compatible(temporal, identity.solution_space(), method) {
        return Err(SolverFormulationCompatibilityError::UnsupportedMethod {
            temporal,
            solution_space: identity.solution_space(),
            coupling: realization.coupling,
            method,
        });
    }
    if !preconditioner_is_compatible(temporal, method, preconditioner) {
        return Err(
            SolverFormulationCompatibilityError::UnsupportedPreconditioner {
                method,
                preconditioner,
            },
        );
    }
    Ok(())
}

const fn is_auxiliary_temporal(temporal: TemporalImplementation) -> bool {
    matches!(
        temporal,
        TemporalImplementation::PrescribedThermomechanicalProfileApplication
            | TemporalImplementation::SolvedThermalStateApplication
    )
}

const fn method_is_compatible(
    temporal: TemporalImplementation,
    solution_space: SolutionSpace,
    method: SolverMethod,
) -> bool {
    match temporal {
        TemporalImplementation::StructuralStaticLinearSolve => matches!(
            method,
            SolverMethod::AssembledPcg | SolverMethod::SparseDirectCholesky
        ),
        TemporalImplementation::StructuralSubspaceIteration
        | TemporalImplementation::StructuralPrestressedSubspaceIteration => {
            matches!(
                method,
                SolverMethod::ShiftInvertBlockLanczos | SolverMethod::PreconditionedLobpcg
            )
        }
        TemporalImplementation::StructuralEigenvalueBuckling => {
            matches!(method, SolverMethod::BucklingBlockLanczos)
        }
        TemporalImplementation::StructuralHarmonicSolve
        | TemporalImplementation::StructuralPrestressedHarmonicSolve => match solution_space {
            SolutionSpace::Direct => {
                matches!(method, SolverMethod::StructuralDirectHarmonicGmres)
            }
            SolutionSpace::ModalReduced => {
                matches!(method, SolverMethod::StructuralModalHarmonicDenseDirect)
            }
        },
        TemporalImplementation::StructuralModalResponseSpectrumCombination => matches!(
            method,
            SolverMethod::StructuralModalResponseSpectrumCombination
        ),
        TemporalImplementation::StructuralRandomResponseIntegration => {
            matches!(method, SolverMethod::StructuralRandomResponseIntegration)
        }
        TemporalImplementation::StructuralNewmarkAverageAcceleration => match solution_space {
            SolutionSpace::Direct => {
                matches!(method, SolverMethod::NewmarkAverageAccelerationPcg)
            }
            SolutionSpace::ModalReduced => matches!(
                method,
                SolverMethod::StructuralModalTransientNewmarkDenseFactorization
            ),
        },
        TemporalImplementation::StructuralQuasistaticNewtonEquilibrium => {
            matches!(method, SolverMethod::IncrementalNewtonRaphson)
        }
        TemporalImplementation::ThermalSteadyLinearSolve
        | TemporalImplementation::ThermalBackwardEuler
        | TemporalImplementation::ThermalSteadyDampedNewton
        | TemporalImplementation::ThermalBackwardEulerDampedNewton
        | TemporalImplementation::ElectrostaticPotentialLinearSolve
        | TemporalImplementation::SteadyConductionPotentialLinearSolve => {
            matches!(
                method,
                SolverMethod::AssembledPcg | SolverMethod::SparseDirectCholesky
            )
        }
        TemporalImplementation::AcousticDomainGraphHarmonicSolve => {
            matches!(method, SolverMethod::AcousticDomainGraphHelmholtzHarmonic)
        }
        TemporalImplementation::AcousticCanonicalHarmonicSolve => {
            matches!(method, SolverMethod::AcousticCanonicalHelmholtzCertifiedLu)
        }
        TemporalImplementation::AcousticCanonicalModalSolve => {
            matches!(
                method,
                SolverMethod::AcousticCanonicalShiftInvertBlockKrylov
            )
        }
        TemporalImplementation::AcousticCanonicalTransientSolve => {
            matches!(method, SolverMethod::AcousticCanonicalNewmarkPcg)
        }
        TemporalImplementation::PrescribedThermomechanicalProfileApplication
        | TemporalImplementation::SolvedThermalStateApplication => false,
        TemporalImplementation::MagnetostaticWeakCoulombSaddleSolve => {
            matches!(method, SolverMethod::SparseDirectLdlt)
        }
        TemporalImplementation::MaxwellComplexElectricFieldGeneralSolve => {
            matches!(method, SolverMethod::SparseDirectLu)
        }
        TemporalImplementation::FluidSteadyNonorthogonalPressureCorrection
        | TemporalImplementation::FluidBackwardEulerConservativeMomentumPressure
        | TemporalImplementation::ConjugateSteadyFlowMonolithicEnergy
        | TemporalImplementation::ConjugateSteadyFlowBackwardEulerEnergy
        | TemporalImplementation::FluidStructureSteadyAitkenPartitioned
        | TemporalImplementation::FluidStructureTransientAitkenPartitioned => {
            matches!(method, SolverMethod::GeneralGmres)
        }
    }
}

const fn preconditioner_is_compatible(
    temporal: TemporalImplementation,
    method: SolverMethod,
    preconditioner: SolverPreconditioner,
) -> bool {
    // The method name alone cannot authorize another formulation's execution claim.
    // Canonical Maxwell applies its physical block directly; CFD uses right scaling.
    if matches!(
        temporal,
        TemporalImplementation::MaxwellComplexElectricFieldGeneralSolve
    ) {
        return matches!(method, SolverMethod::SparseDirectLu)
            && matches!(preconditioner, SolverPreconditioner::None);
    }
    match method {
        SolverMethod::GeneralGmres => matches!(
            preconditioner,
            SolverPreconditioner::None | SolverPreconditioner::Jacobi
        ),
        SolverMethod::AssembledPcg
        | SolverMethod::MatrixFreePcg
        | SolverMethod::NewmarkAverageAccelerationPcg
        | SolverMethod::IncrementalNewtonRaphson => matches!(
            preconditioner,
            SolverPreconditioner::None
                | SolverPreconditioner::Jacobi
                | SolverPreconditioner::Ilu0
                | SolverPreconditioner::Amg
                | SolverPreconditioner::SparseCholesky
        ),
        SolverMethod::ShiftInvertBlockLanczos | SolverMethod::BucklingBlockLanczos => {
            matches!(preconditioner, SolverPreconditioner::SparseCholesky)
        }
        SolverMethod::PreconditionedLobpcg => matches!(preconditioner, SolverPreconditioner::Amg),
        SolverMethod::SparseDirectCholesky
        | SolverMethod::SparseDirectLdlt
        | SolverMethod::SparseDirectLu => matches!(preconditioner, SolverPreconditioner::None),
        SolverMethod::AssembledSaddlePointGmres
        | SolverMethod::MatrixFreeSaddlePointGmres
        | SolverMethod::DenseDirect
        | SolverMethod::StructuralModalTransientNewmarkDenseFactorization
        | SolverMethod::StructuralModalHarmonicDenseDirect
        | SolverMethod::StructuralModalResponseSpectrumCombination
        | SolverMethod::StructuralRandomResponseIntegration
        | SolverMethod::AcousticDomainGraphHelmholtzHarmonic
        | SolverMethod::AcousticCanonicalHelmholtzCertifiedLu => {
            matches!(preconditioner, SolverPreconditioner::None)
        }
        SolverMethod::AcousticCanonicalShiftInvertBlockKrylov
        | SolverMethod::AcousticCanonicalNewmarkPcg => {
            matches!(preconditioner, SolverPreconditioner::Ilu0)
        }
        SolverMethod::StructuralDirectHarmonicGmres => {
            matches!(preconditioner, SolverPreconditioner::Jacobi)
        }
        SolverMethod::CfdVelocityPressureFiniteVolume => matches!(
            preconditioner,
            SolverPreconditioner::FiniteVolumePressureBalance
        ),
    }
}

#[cfg(test)]
mod tests;
