use runmat_analysis_core::{CouplingImplementation, SolutionSpace, TemporalImplementation};
use thiserror::Error;

use crate::admission::AnalysisRunIdentity;

use super::{SolverMethod, SolverPreconditioner};

mod run_result;

pub(super) use run_result::FeaRunResultWire;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SolverFormulationCompatibilityError {
    #[error("structural static balance evidence is invalid")]
    InvalidStructuralStaticBalance,
    #[error("a structural static result is missing mandatory physical balance evidence")]
    MissingStructuralStaticBalance,
    #[error("structural static balance evidence is attached to a non-static result")]
    UnexpectedStructuralStaticBalance,
    #[error("structural static constraint result is invalid")]
    InvalidStructuralConstraintResult,
    #[error("a structural static result is missing its canonical constraint result")]
    MissingStructuralConstraintResult,
    #[error("a structural static constraint result is attached to a non-static result")]
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
    if !method_is_compatible(
        temporal,
        identity.solution_space(),
        realization.coupling,
        method,
    ) {
        return Err(SolverFormulationCompatibilityError::UnsupportedMethod {
            temporal,
            solution_space: identity.solution_space(),
            coupling: realization.coupling,
            method,
        });
    }
    if !preconditioner_is_compatible(method, preconditioner) {
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
    )
}

const fn method_is_compatible(
    temporal: TemporalImplementation,
    solution_space: SolutionSpace,
    coupling: CouplingImplementation,
    method: SolverMethod,
) -> bool {
    match temporal {
        TemporalImplementation::StructuralStaticLinearSolve => matches!(
            method,
            SolverMethod::AssembledPcg
                | SolverMethod::MatrixFreePcg
                | SolverMethod::MatrixFreeSaddlePointGmres
        ),
        TemporalImplementation::StructuralSubspaceIteration
        | TemporalImplementation::StructuralPrestressedSubspaceIteration => {
            matches!(method, SolverMethod::MatrixFreeSubspaceIteration)
        }
        TemporalImplementation::StructuralEigenvalueBuckling => {
            matches!(method, SolverMethod::MatrixFreeBucklingSubspaceIteration)
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
        TemporalImplementation::StructuralScalarResidualIteration => {
            matches!(method, SolverMethod::IncrementalNewtonRaphson)
        }
        TemporalImplementation::ThermalSteadyLinearSolve
        | TemporalImplementation::ThermalBackwardEuler => {
            matches!(
                method,
                SolverMethod::AssembledPcg | SolverMethod::MatrixFreePcg
            )
        }
        TemporalImplementation::ElectromagneticReducedEdgeGraphHarmonicSolve => matches!(
            method,
            SolverMethod::ElectromagneticEdgeCurlCurlHarmonicBlockBicgstab
        ),
        TemporalImplementation::AcousticDomainGraphHarmonicSolve => {
            matches!(method, SolverMethod::AcousticDomainGraphHelmholtzHarmonic)
        }
        TemporalImplementation::CfdReducedChannelSteadyIteration
        | TemporalImplementation::CfdReducedChannelTimeStepping => {
            matches!(method, SolverMethod::CfdVelocityPressureFiniteVolume)
        }
        TemporalImplementation::CoupledSinglePassProfileTransfer
        | TemporalImplementation::CoupledFixedPointRelaxation
        | TemporalImplementation::CoupledScalarResidualIteration => match coupling {
            CouplingImplementation::PartitionedConjugateProjection => {
                matches!(method, SolverMethod::ChtConjugateProjection)
            }
            CouplingImplementation::PartitionedFluidStructureRelaxation => {
                matches!(method, SolverMethod::FsiPartitionedProjection)
            }
            _ => false,
        },
        TemporalImplementation::PrescribedThermomechanicalProfileApplication => false,
    }
}

const fn preconditioner_is_compatible(
    method: SolverMethod,
    preconditioner: SolverPreconditioner,
) -> bool {
    match method {
        SolverMethod::AssembledPcg
        | SolverMethod::MatrixFreePcg
        | SolverMethod::MatrixFreeSubspaceIteration
        | SolverMethod::MatrixFreeBucklingSubspaceIteration
        | SolverMethod::NewmarkAverageAccelerationPcg
        | SolverMethod::IncrementalNewtonRaphson => matches!(
            preconditioner,
            SolverPreconditioner::None | SolverPreconditioner::Jacobi | SolverPreconditioner::Ilu0
        ),
        SolverMethod::MatrixFreeSaddlePointGmres
        | SolverMethod::DenseDirect
        | SolverMethod::StructuralModalTransientNewmarkDenseFactorization
        | SolverMethod::StructuralDirectHarmonicGmres
        | SolverMethod::StructuralModalHarmonicDenseDirect
        | SolverMethod::StructuralModalResponseSpectrumCombination
        | SolverMethod::StructuralRandomResponseIntegration
        | SolverMethod::AcousticDomainGraphHelmholtzHarmonic => {
            matches!(preconditioner, SolverPreconditioner::None)
        }
        SolverMethod::ElectromagneticEdgeCurlCurlHarmonicBlockBicgstab => {
            matches!(preconditioner, SolverPreconditioner::BlockJacobi)
        }
        SolverMethod::CfdVelocityPressureFiniteVolume => matches!(
            preconditioner,
            SolverPreconditioner::FiniteVolumePressureBalance
        ),
        SolverMethod::ChtConjugateProjection => {
            matches!(preconditioner, SolverPreconditioner::ThermalCfdProjection)
        }
        SolverMethod::FsiPartitionedProjection => {
            matches!(preconditioner, SolverPreconditioner::InterfaceRelaxation)
        }
    }
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        AnalysisProcedure, BoundaryTransferImplementation, CoupledPhysicsFamily,
        FormulationRequirement, PhysicsFamily, SourceImplementation,
        StructuralFormulationRealization, StructuralOperatorImplementation,
        StructuralRecoveryImplementation,
    };

    use crate::{
        admission::{
            admit_formulation, registered_formulations, structural_exact_realization,
            FormulationAdmissionRequest, StructuralRealizationInput,
        },
        contracts::{ComputeBackend, ExecutionBackend, FeaRunResult},
        solve::backend::kind::LinearAlgebraBackendKind,
    };

    use super::*;

    fn structural_identity(
        procedure: AnalysisProcedure,
        solution_space: SolutionSpace,
    ) -> AnalysisRunIdentity {
        let realization = structural_exact_realization(StructuralRealizationInput {
            assemblies: vec![StructuralFormulationRealization::new(
                StructuralOperatorImplementation::Tet4LinearElasticity,
                Some(StructuralRecoveryImplementation::Tet4ConstantStrain),
            )],
            procedure,
            coupling: CouplingImplementation::Uncoupled,
            sources: vec![SourceImplementation::CanonicalBoundaryProjection],
            boundary_transfers: vec![BoundaryTransferImplementation::CanonicalBoundaryProjection],
        })
        .expect("structural compatibility fixture must have an exact realization");
        admit_formulation(FormulationAdmissionRequest {
            family: PhysicsFamily::Structural,
            procedure,
            solution_space,
            requirement: FormulationRequirement::Experimental,
            realization,
        })
        .expect("structural compatibility fixture must be admitted")
        .into_identity()
    }

    fn nonstructural_identity(
        family: PhysicsFamily,
        procedure: AnalysisProcedure,
    ) -> AnalysisRunIdentity {
        let realization = registered_formulations(family, procedure)
            .into_iter()
            .next()
            .expect("nonstructural compatibility fixture must be registered")
            .realization()
            .clone();
        admit_formulation(FormulationAdmissionRequest {
            family,
            procedure,
            solution_space: SolutionSpace::Direct,
            requirement: FormulationRequirement::Experimental,
            realization,
        })
        .expect("nonstructural compatibility fixture must be admitted")
        .into_identity()
    }

    #[test]
    fn structural_temporals_admit_only_their_solver_families() {
        let cases = [
            (
                AnalysisProcedure::Static,
                SolutionSpace::Direct,
                SolverMethod::MatrixFreePcg,
                SolverPreconditioner::Jacobi,
            ),
            (
                AnalysisProcedure::ModalExtraction,
                SolutionSpace::Direct,
                SolverMethod::MatrixFreeSubspaceIteration,
                SolverPreconditioner::Ilu0,
            ),
            (
                AnalysisProcedure::PrestressedModalExtraction,
                SolutionSpace::Direct,
                SolverMethod::MatrixFreeSubspaceIteration,
                SolverPreconditioner::Jacobi,
            ),
            (
                AnalysisProcedure::EigenvalueBuckling,
                SolutionSpace::Direct,
                SolverMethod::MatrixFreeBucklingSubspaceIteration,
                SolverPreconditioner::Ilu0,
            ),
            (
                AnalysisProcedure::HarmonicResponse,
                SolutionSpace::Direct,
                SolverMethod::StructuralDirectHarmonicGmres,
                SolverPreconditioner::None,
            ),
            (
                AnalysisProcedure::HarmonicResponse,
                SolutionSpace::ModalReduced,
                SolverMethod::StructuralModalHarmonicDenseDirect,
                SolverPreconditioner::None,
            ),
            (
                AnalysisProcedure::TransientResponse,
                SolutionSpace::Direct,
                SolverMethod::NewmarkAverageAccelerationPcg,
                SolverPreconditioner::Jacobi,
            ),
            (
                AnalysisProcedure::TransientResponse,
                SolutionSpace::ModalReduced,
                SolverMethod::StructuralModalTransientNewmarkDenseFactorization,
                SolverPreconditioner::None,
            ),
            (
                AnalysisProcedure::Nonlinear,
                SolutionSpace::Direct,
                SolverMethod::IncrementalNewtonRaphson,
                SolverPreconditioner::Jacobi,
            ),
            (
                AnalysisProcedure::ModalResponseSpectrum,
                SolutionSpace::ModalReduced,
                SolverMethod::StructuralModalResponseSpectrumCombination,
                SolverPreconditioner::None,
            ),
            (
                AnalysisProcedure::RandomResponse,
                SolutionSpace::ModalReduced,
                SolverMethod::StructuralRandomResponseIntegration,
                SolverPreconditioner::None,
            ),
        ];
        for (procedure, solution_space, method, preconditioner) in cases {
            let identity = structural_identity(procedure, solution_space);
            validate_solver_formulation_compatibility(&identity, method, preconditioner)
                .expect("the exact structural solver matrix must admit its implemented pair");
            assert!(validate_solver_formulation_compatibility(
                &identity,
                SolverMethod::AcousticDomainGraphHelmholtzHarmonic,
                SolverPreconditioner::None,
            )
            .is_err());
        }
    }

    #[test]
    fn nonstructural_temporals_admit_only_their_solver_families() {
        let cases = [
            (
                PhysicsFamily::Thermal,
                AnalysisProcedure::Static,
                SolverMethod::MatrixFreePcg,
                SolverPreconditioner::Jacobi,
            ),
            (
                PhysicsFamily::Thermal,
                AnalysisProcedure::TransientResponse,
                SolverMethod::MatrixFreePcg,
                SolverPreconditioner::Jacobi,
            ),
            (
                PhysicsFamily::Electromagnetic,
                AnalysisProcedure::HarmonicResponse,
                SolverMethod::ElectromagneticEdgeCurlCurlHarmonicBlockBicgstab,
                SolverPreconditioner::BlockJacobi,
            ),
            (
                PhysicsFamily::Acoustic,
                AnalysisProcedure::HarmonicResponse,
                SolverMethod::AcousticDomainGraphHelmholtzHarmonic,
                SolverPreconditioner::None,
            ),
            (
                PhysicsFamily::Fluid,
                AnalysisProcedure::Static,
                SolverMethod::CfdVelocityPressureFiniteVolume,
                SolverPreconditioner::FiniteVolumePressureBalance,
            ),
            (
                PhysicsFamily::Fluid,
                AnalysisProcedure::TransientResponse,
                SolverMethod::CfdVelocityPressureFiniteVolume,
                SolverPreconditioner::FiniteVolumePressureBalance,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
                AnalysisProcedure::Static,
                SolverMethod::ChtConjugateProjection,
                SolverPreconditioner::ThermalCfdProjection,
            ),
            (
                PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction),
                AnalysisProcedure::TransientResponse,
                SolverMethod::FsiPartitionedProjection,
                SolverPreconditioner::InterfaceRelaxation,
            ),
        ];
        for (family, procedure, method, preconditioner) in cases {
            let identity = nonstructural_identity(family, procedure);
            validate_solver_formulation_compatibility(&identity, method, preconditioner)
                .expect("the exact nonstructural solver matrix must admit its implemented pair");
            assert!(validate_solver_formulation_compatibility(
                &identity,
                SolverMethod::MatrixFreeSubspaceIteration,
                SolverPreconditioner::None,
            )
            .is_err());
        }
    }

    #[test]
    fn persisted_run_rejects_solver_or_preconditioner_substitution() {
        let identity = structural_identity(AnalysisProcedure::Static, SolutionSpace::Direct);
        let connector_plan = crate::connectors::CanonicalStructuralConnectorPlan::empty(
            crate::physics::structural::topology::CanonicalStructuralTopologyIdentity::Solid {
                mesh_digest: runmat_meshing_core::StableDigest::from_bytes([7; 32]),
            },
            0,
        )
        .unwrap();
        let run = FeaRunResult {
            identity,
            requested_backend: ComputeBackend::Cpu,
            execution_backend: ExecutionBackend::Cpu,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            solver_device_apply_k_ratio: 0.0,
            solver_method: SolverMethod::MatrixFreePcg,
            preconditioner: SolverPreconditioner::Jacobi,
            solver_host_sync_count: 0,
            accelerator_evidence: Vec::new(),
            diagnostics: Vec::new(),
            spatial_projection_evidence: Vec::new(),
            structural_static_balance: Some(Box::new(crate::solve::StructuralStaticBalanceEvidence {
                internal_work_joules: 0.0,
                applied_load_work_joules: 0.0,
                constraint_work_joules: 0.0,
                work_balance_residual_joules: 0.0,
                work_balance_relative_error: 0.0,
                equilibrium_residual_norm_newtons: 0.0,
                equilibrium_scale_newtons: 1.0,
                equilibrium_relative_error: 0.0,
                relative_tolerance: 1.0e-8,
                accepted: true,
            })),
            structural_constraint_result: Some(Box::new(
                crate::constraints::CanonicalStructuralConstraintResult {
                    schema_version: 1,
                    constraint_plan_digest: runmat_execution::Digest::sha256(b"plan"),
                    topology:
                        crate::physics::structural::topology::CanonicalStructuralTopologyIdentity::Solid {
                            mesh_digest: runmat_meshing_core::StableDigest::from_bytes([7; 32]),
                        },
                    reference_points: Vec::new(),
                    groups: Vec::new(),
                    resultant_force_newtons: [0.0; 3],
                    resultant_moment_about_global_origin_newton_meters: [0.0; 3],
                    constraint_work_joules: 0.0,
                    maximum_displacement_error_meters: 0.0,
                    maximum_rotation_error_radians: 0.0,
                },
            )),
            structural_connector_result: Some(Box::new(
                crate::connectors::CanonicalStructuralConnectorResult::recover_static(
                    connector_plan,
                    &crate::assembly::dofs::StructuralDofLayout::default(),
                    &[],
                )
                .unwrap(),
            )),
            observations: Vec::new(),
            observation_histories: Vec::new(),
            series_reductions: Vec::new(),
            fields: Vec::new(),
        };
        run.validate_solver_formulation()
            .expect("fixture solver evidence must match its formulation");

        let mut substituted_method = serde_json::to_value(&run).unwrap();
        substituted_method["solver_method"] =
            serde_json::json!(SolverMethod::ElectromagneticEdgeCurlCurlHarmonicBlockBicgstab);
        assert!(serde_json::from_value::<FeaRunResult>(substituted_method).is_err());

        let mut substituted_preconditioner = serde_json::to_value(&run).unwrap();
        substituted_preconditioner["preconditioner"] =
            serde_json::json!(SolverPreconditioner::BlockJacobi);
        assert!(serde_json::from_value::<FeaRunResult>(substituted_preconditioner).is_err());

        let mut missing_balance = serde_json::to_value(&run).unwrap();
        missing_balance["structural_static_balance"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<FeaRunResult>(missing_balance).is_err());

        let mut missing_constraints = serde_json::to_value(&run).unwrap();
        missing_constraints["structural_constraint_result"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<FeaRunResult>(missing_constraints).is_err());

        let mut unexpected_static_evidence = run.clone();
        unexpected_static_evidence.identity =
            structural_identity(AnalysisProcedure::ModalExtraction, SolutionSpace::Direct);
        unexpected_static_evidence.solver_method = SolverMethod::MatrixFreeSubspaceIteration;
        unexpected_static_evidence.preconditioner = SolverPreconditioner::Ilu0;
        assert_eq!(
            unexpected_static_evidence.validate_solver_formulation(),
            Err(SolverFormulationCompatibilityError::UnexpectedStructuralStaticBalance)
        );

        unexpected_static_evidence.structural_static_balance = None;
        assert_eq!(
            unexpected_static_evidence.validate_solver_formulation(),
            Err(SolverFormulationCompatibilityError::UnexpectedStructuralConstraintResult)
        );
    }
}
