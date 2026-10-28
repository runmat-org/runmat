#[test]
fn canonical_fluid_general_gmres_does_not_report_reduced_channel_method() {
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
    let identity = crate::admission::admit_canonical_fluid_steady(
        &canonical_tetrahedron_solver_mesh(ElementOrder::Tet4),
    )
    .unwrap();
    assert!(validate_solver_formulation_compatibility(
        &identity,
        SolverMethod::GeneralGmres,
        SolverPreconditioner::Jacobi
    )
    .is_ok());
    assert!(validate_solver_formulation_compatibility(
        &identity,
        SolverMethod::CfdVelocityPressureFiniteVolume,
        SolverPreconditioner::Jacobi
    )
    .is_err());
}

#[test]
fn magnetic_saddle_method_cannot_claim_electrical_or_harmonic_algebra() {
    use super::*;
    assert!(method_is_compatible(
        TemporalImplementation::MagnetostaticWeakCoulombSaddleSolve,
        SolutionSpace::Direct,
        SolverMethod::SparseDirectLdlt,
    ));
    for method in [
        SolverMethod::AssembledPcg,
        SolverMethod::SparseDirectLu,
        SolverMethod::StructuralDirectHarmonicGmres,
    ] {
        assert!(!method_is_compatible(
            TemporalImplementation::MagnetostaticWeakCoulombSaddleSolve,
            SolutionSpace::Direct,
            method,
        ));
    }
    assert!(!method_is_compatible(
        TemporalImplementation::ElectrostaticPotentialLinearSolve,
        SolutionSpace::Direct,
        SolverMethod::SparseDirectLdlt,
    ));
    assert!(preconditioner_is_compatible(
        TemporalImplementation::MagnetostaticWeakCoulombSaddleSolve,
        SolverMethod::SparseDirectLdlt,
        SolverPreconditioner::None,
    ));
    assert!(!preconditioner_is_compatible(
        TemporalImplementation::MagnetostaticWeakCoulombSaddleSolve,
        SolverMethod::SparseDirectLdlt,
        SolverPreconditioner::Jacobi,
    ));
}

use runmat_analysis_core::{
    AnalysisProcedure, BoundaryTransferImplementation, CoupledPhysicsFamily, PhysicsFamily,
    SourceImplementation, StructuralFormulationRealization, StructuralOperatorImplementation,
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
        kinematics: runmat_analysis_core::StructuralKinematics::SmallStrain,
        coupling: CouplingImplementation::Uncoupled,
        sources: vec![SourceImplementation::CanonicalBoundaryProjection],
        boundary_transfers: vec![BoundaryTransferImplementation::CanonicalBoundaryProjection],
    })
    .expect("structural compatibility fixture must have an exact realization");
    admit_formulation(FormulationAdmissionRequest {
        family: PhysicsFamily::Structural,
        procedure,
        solution_space,
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
        .find(|entry| {
            family != PhysicsFamily::Electromagnetic
                || procedure != AnalysisProcedure::HarmonicResponse
                || entry.realization().temporals
                    == vec![TemporalImplementation::MaxwellComplexElectricFieldGeneralSolve]
        })
        .expect("nonstructural compatibility fixture must be registered")
        .realization()
        .clone();
    admit_formulation(FormulationAdmissionRequest {
        family,
        procedure,
        solution_space: SolutionSpace::Direct,
        realization,
    })
    .expect("nonstructural compatibility fixture must be admitted")
    .into_identity()
}

#[test]
fn canonical_maxwell_reports_only_genuine_general_gmres_without_spd_or_gauge_claims() {
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let identity = crate::admission::admit_canonical_maxwell(&mesh).unwrap();
        validate_solver_formulation_compatibility(
            &identity,
            SolverMethod::SparseDirectLu,
            SolverPreconditioner::None,
        )
        .unwrap();
        for method in [
            SolverMethod::AssembledPcg,
            SolverMethod::GeneralGmres,
            SolverMethod::SparseDirectCholesky,
            SolverMethod::SparseDirectLdlt,
            SolverMethod::StructuralDirectHarmonicGmres,
            SolverMethod::StructuralDirectHarmonicGmres,
            SolverMethod::DenseDirect,
        ] {
            assert!(validate_solver_formulation_compatibility(
                &identity,
                method,
                SolverPreconditioner::None
            )
            .is_err());
        }
        for preconditioner in [
            SolverPreconditioner::Jacobi,
            SolverPreconditioner::Ilu0,
            SolverPreconditioner::BlockJacobi,
        ] {
            assert!(validate_solver_formulation_compatibility(
                &identity,
                SolverMethod::SparseDirectLu,
                preconditioner
            )
            .is_err());
        }
        let magnetic = crate::admission::admit_canonical_magnetostatic(&mesh).unwrap();
        assert!(validate_solver_formulation_compatibility(
            &magnetic,
            SolverMethod::SparseDirectLu,
            SolverPreconditioner::None
        )
        .is_err());
    }
    assert_eq!(SolverMethod::GeneralGmres.as_str(), "general_gmres");
    assert!(!method_is_compatible(
        TemporalImplementation::SteadyConductionPotentialLinearSolve,
        SolutionSpace::Direct,
        SolverMethod::GeneralGmres
    ));
}

#[test]
fn structural_temporals_admit_only_their_solver_families() {
    let cases = [
        (
            AnalysisProcedure::Static,
            SolutionSpace::Direct,
            SolverMethod::AssembledPcg,
            SolverPreconditioner::Jacobi,
        ),
        (
            AnalysisProcedure::ModalExtraction,
            SolutionSpace::Direct,
            SolverMethod::ShiftInvertBlockLanczos,
            SolverPreconditioner::SparseCholesky,
        ),
        (
            AnalysisProcedure::PrestressedModalExtraction,
            SolutionSpace::Direct,
            SolverMethod::ShiftInvertBlockLanczos,
            SolverPreconditioner::SparseCholesky,
        ),
        (
            AnalysisProcedure::EigenvalueBuckling,
            SolutionSpace::Direct,
            SolverMethod::BucklingBlockLanczos,
            SolverPreconditioner::SparseCholesky,
        ),
        (
            AnalysisProcedure::HarmonicResponse,
            SolutionSpace::Direct,
            SolverMethod::StructuralDirectHarmonicGmres,
            SolverPreconditioner::Jacobi,
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
            SolverMethod::SparseDirectCholesky,
            SolverPreconditioner::None,
        ),
        (
            PhysicsFamily::Thermal,
            AnalysisProcedure::TransientResponse,
            SolverMethod::AssembledPcg,
            SolverPreconditioner::Jacobi,
        ),
        (
            PhysicsFamily::Electromagnetic,
            AnalysisProcedure::HarmonicResponse,
            SolverMethod::SparseDirectLu,
            SolverPreconditioner::None,
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
            SolverMethod::GeneralGmres,
            SolverPreconditioner::Jacobi,
        ),
        // Canonical CHT is the monolithic flow-advected conjugate energy realization; its
        // publication reports the flow lifecycle's GMRES/Jacobi pair. The partitioned
        // projection pair belonged to the removed reduced realization.
        (
            PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
            AnalysisProcedure::Static,
            SolverMethod::GeneralGmres,
            SolverPreconditioner::Jacobi,
        ),
    ];
    for (family, procedure, method, preconditioner) in cases {
        let identity = nonstructural_identity(family, procedure);
        validate_solver_formulation_compatibility(&identity, method, preconditioner)
            .expect("the exact nonstructural solver matrix must admit its implemented pair");
        assert!(validate_solver_formulation_compatibility(
            &identity,
            SolverMethod::ShiftInvertBlockLanczos,
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
            solver_method: SolverMethod::AssembledPcg,
            preconditioner: SolverPreconditioner::Jacobi,
            solver_host_sync_count: 0,
            accelerator_evidence: Vec::new(),
            diagnostics: Vec::new(),
            spatial_projection_evidence: Vec::new(),
            structural_static_convergence: Some(crate::contracts::StructuralStaticConvergence { converged: true, iterations: 0, residual_norm: 0.0, relative_residual: Some(0.0), operator_proof: crate::contracts::StaticOperatorProof::SparseCholesky, spectral_estimate: None }),
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
        serde_json::json!("electromagnetic_edge_curl_curl_harmonic_block_bicgstab");
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
    unexpected_static_evidence.solver_method = SolverMethod::ShiftInvertBlockLanczos;
    unexpected_static_evidence.preconditioner = SolverPreconditioner::SparseCholesky;
    assert_eq!(
        unexpected_static_evidence.validate_solver_formulation(),
        Err(SolverFormulationCompatibilityError::UnexpectedStructuralStaticConvergence)
    );

    unexpected_static_evidence.structural_static_balance = None;
    unexpected_static_evidence.structural_static_convergence = None;
    assert_eq!(
        unexpected_static_evidence.validate_solver_formulation(),
        Err(SolverFormulationCompatibilityError::UnexpectedStructuralConstraintResult)
    );
    // An absent static-only carrier must not change immutable nonstatic wire records.
    let mut thermal = run;
    thermal.identity = nonstructural_identity(PhysicsFamily::Thermal, AnalysisProcedure::Static);
    thermal.structural_static_convergence = None;
    thermal.structural_static_balance = None;
    thermal.structural_constraint_result = None;
    thermal.structural_connector_result = None;
    thermal.validate_solver_formulation().unwrap();
    let original_wire = serde_json::to_value(&thermal).unwrap();
    assert!(original_wire.get("structural_static_convergence").is_none());
    let cold: FeaRunResult = serde_json::from_value(original_wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(cold).unwrap(), original_wire);
}
