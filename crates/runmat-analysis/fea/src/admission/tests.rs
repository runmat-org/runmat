use runmat_analysis_core::{
    AnalysisProcedure, CoupledPhysicsFamily, CouplingImplementation,
    FormulationAdmissionRejectionReason, PhysicsFamily, RecoveryImplementation, SolutionSpace,
    SourceImplementation, StructuralFormulationRealization, StructuralOperatorImplementation,
    StructuralRecoveryImplementation, TopologyImplementation,
};

use super::*;

fn exposed_identities() -> Vec<(PhysicsFamily, AnalysisProcedure)> {
    vec![
        (PhysicsFamily::Structural, AnalysisProcedure::Static),
        (
            PhysicsFamily::Structural,
            AnalysisProcedure::ModalExtraction,
        ),
        (
            PhysicsFamily::Structural,
            AnalysisProcedure::TransientResponse,
        ),
        (
            PhysicsFamily::Structural,
            AnalysisProcedure::HarmonicResponse,
        ),
        (PhysicsFamily::Structural, AnalysisProcedure::Nonlinear),
        (PhysicsFamily::Thermal, AnalysisProcedure::Static),
        (PhysicsFamily::Thermal, AnalysisProcedure::TransientResponse),
        (PhysicsFamily::Electromagnetic, AnalysisProcedure::Static),
        (
            PhysicsFamily::Electromagnetic,
            AnalysisProcedure::HarmonicResponse,
        ),
        (PhysicsFamily::Acoustic, AnalysisProcedure::HarmonicResponse),
        (PhysicsFamily::Fluid, AnalysisProcedure::Static),
        (
            PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
            AnalysisProcedure::Static,
        ),
        (
            PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
            AnalysisProcedure::TransientResponse,
        ),
        (
            PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
            AnalysisProcedure::Static,
        ),
    ]
}

#[test]
fn procedure_labels_cannot_wrap_incompatible_thermal_or_harmonic_em_math() {
    let cases = [
        (
            PhysicsFamily::Thermal,
            AnalysisProcedure::Static,
            registered_formulations(PhysicsFamily::Thermal, AnalysisProcedure::TransientResponse)
                .remove(0)
                .realization()
                .clone(),
            FormulationAdmissionRejectionReason::IncompatibleAxis(
                runmat_analysis_core::FormulationAxis::Temporal,
            ),
        ),
        (
            PhysicsFamily::Electromagnetic,
            AnalysisProcedure::Static,
            registered_formulations(
                PhysicsFamily::Electromagnetic,
                AnalysisProcedure::HarmonicResponse,
            )
            .remove(0)
            .realization()
            .clone(),
            FormulationAdmissionRejectionReason::IncompatibleAxis(
                runmat_analysis_core::FormulationAxis::SpatialOperator,
            ),
        ),
    ];

    for (family, procedure, realization, expected_reason) in cases {
        let rejection = admit_formulation(FormulationAdmissionRequest {
            family,
            procedure,
            solution_space: SolutionSpace::Direct,
            realization,
        })
        .expect_err("a static identity must not relabel non-static mathematics");
        assert_eq!(rejection.reason, expected_reason);
    }
}

#[test]
fn acoustic_registry_names_the_executed_hashed_target_projection() {
    let realizations =
        registered_formulations(PhysicsFamily::Acoustic, AnalysisProcedure::HarmonicResponse);
    assert!(realizations.iter().any(|entry| {
        entry.realization().components[0]
            .sources
            .contains(&SourceImplementation::AcousticHashedTargetGraphProjection)
    }));
    assert!(realizations.iter().all(|entry| {
        !entry.realization().components[0]
            .sources
            .contains(&SourceImplementation::UnmappedFallback)
    }));
}

fn structural_assembly_realizations() -> Vec<StructuralFormulationRealization> {
    vec![
        StructuralFormulationRealization::new(
            StructuralOperatorImplementation::Tet4LinearElasticity,
            Some(StructuralRecoveryImplementation::Tet4ConstantStrain),
        ),
        StructuralFormulationRealization::new(
            StructuralOperatorImplementation::Tet10LinearElasticity,
            Some(StructuralRecoveryImplementation::Tet10IntegrationPointProjection),
        ),
        StructuralFormulationRealization::new(
            StructuralOperatorImplementation::EulerBernoulliBeam3d,
            Some(StructuralRecoveryImplementation::BeamSectionResultants),
        ),
        StructuralFormulationRealization::new(
            StructuralOperatorImplementation::Mitc3TriangularShell,
            Some(StructuralRecoveryImplementation::Mitc3ShellStressResultants),
        ),
        StructuralFormulationRealization::new(
            StructuralOperatorImplementation::SyntheticDiagonalChain,
            None,
        ),
    ]
}

fn registered_realizations(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
) -> Vec<runmat_analysis_core::ExactFormulationRealization> {
    if family == PhysicsFamily::Structural {
        structural_assembly_realizations()
            .into_iter()
            .filter(|assembly| {
                procedure != AnalysisProcedure::Nonlinear
                    || matches!(
                        assembly.operator,
                        StructuralOperatorImplementation::Tet4LinearElasticity
                            | StructuralOperatorImplementation::Tet10LinearElasticity
                    )
            })
            .map(|assembly| {
                structural_exact_realization(StructuralRealizationInput {
                    assemblies: vec![assembly],
                    procedure,
                    kinematics: runmat_analysis_core::StructuralKinematics::SmallStrain,
                    coupling: CouplingImplementation::Uncoupled,
                    sources: vec![runmat_analysis_core::SourceImplementation::NotApplicable],
                    boundary_transfers: vec![
                        runmat_analysis_core::BoundaryTransferImplementation::NotApplicable,
                    ],
                })
                .expect("exposed structural procedure must map every implemented operator")
            })
            .collect()
    } else {
        registered_formulations(family, procedure)
            .into_iter()
            .map(|entry| entry.realization().clone())
            .collect()
    }
}

#[test]
fn every_registered_branch_admits_its_exact_realization() {
    for (family, procedure) in exposed_identities() {
        let realizations = registered_realizations(family, procedure);
        assert!(
            !realizations.is_empty(),
            "registry has no realization for family={family:?} procedure={procedure:?}"
        );
        for realization in realizations {
            let request = FormulationAdmissionRequest {
                family,
                procedure,
                solution_space: SolutionSpace::Direct,
                realization: realization.clone(),
            };
            let receipt =
                admit_formulation(request).expect("registered current formulation should admit");
            assert_eq!(receipt.family(), family);
            assert_eq!(receipt.procedure(), procedure);
            assert_eq!(receipt.solution_space(), SolutionSpace::Direct);
            assert_eq!(receipt.realization(), &realization);
            let identity = receipt.clone().into_identity();
            assert_eq!(identity.family(), family);
            assert_eq!(identity.procedure(), procedure);
            assert_eq!(identity.solution_space(), SolutionSpace::Direct);
        }
    }
}

#[test]
fn availability_gate_admits_every_registered_procedure() {
    for (family, procedure) in exposed_identities() {
        require_available_formulation(family, procedure, SolutionSpace::Direct)
            .expect("every registered procedure should be available");
    }
}

#[test]
fn transient_fluid_requires_its_distinct_realization() {
    let family = PhysicsFamily::Fluid;
    let procedure = AnalysisProcedure::TransientResponse;
    let registered = registered_formulations(family, procedure);
    assert_eq!(registered.len(), 1);
    let transient = registered[0].realization().clone();
    let steady = canonical_fluid_steady_realization();
    assert_ne!(transient, steady);
    admit_formulation(FormulationAdmissionRequest {
        family,
        procedure,
        solution_space: SolutionSpace::Direct,
        realization: transient,
    })
    .expect("canonical transient realization is available");
    assert!(
        admit_formulation(FormulationAdmissionRequest {
            family,
            procedure,
            solution_space: SolutionSpace::Direct,
            realization: steady,
        })
        .is_err(),
        "steady realization cannot be relabeled transient"
    );
    require_available_formulation(family, procedure, SolutionSpace::Direct)
        .expect("transient fluid is available");
}

#[test]
fn thermomechanical_static_procedure_is_available() {
    require_available_formulation(
        PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
        AnalysisProcedure::Static,
        SolutionSpace::Direct,
    )
    .expect("the implemented static thermomechanical procedure is available");
    for procedure in [
        AnalysisProcedure::TransientResponse,
        AnalysisProcedure::Nonlinear,
    ] {
        let rejection = require_available_formulation(
            PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
            procedure,
            SolutionSpace::Direct,
        )
        .expect_err("unimplemented coupled evolution must not borrow the J2 identity");
        assert_eq!(
            rejection.reason,
            FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure
        );
    }
}

#[test]
fn structural_exact_identity_distinguishes_operator_topology_and_recovery() {
    let realizations =
        registered_realizations(PhysicsFamily::Structural, AnalysisProcedure::Static);
    let expected = [
        (
            StructuralOperatorImplementation::Tet4LinearElasticity,
            TopologyImplementation::CanonicalSolverMeshTet4,
            RecoveryImplementation::Structural(
                StructuralRecoveryImplementation::Tet4ConstantStrain,
            ),
        ),
        (
            StructuralOperatorImplementation::Tet10LinearElasticity,
            TopologyImplementation::CanonicalSolverMeshTet10,
            RecoveryImplementation::Structural(
                StructuralRecoveryImplementation::Tet10IntegrationPointProjection,
            ),
        ),
        (
            StructuralOperatorImplementation::EulerBernoulliBeam3d,
            TopologyImplementation::ExplicitStructuralConnectivityBeam,
            RecoveryImplementation::Structural(
                StructuralRecoveryImplementation::BeamSectionResultants,
            ),
        ),
        (
            StructuralOperatorImplementation::Mitc3TriangularShell,
            TopologyImplementation::ExplicitStructuralConnectivityTriangularShell,
            RecoveryImplementation::Structural(
                StructuralRecoveryImplementation::Mitc3ShellStressResultants,
            ),
        ),
        (
            StructuralOperatorImplementation::SyntheticDiagonalChain,
            TopologyImplementation::DofAdjacency,
            RecoveryImplementation::Unavailable,
        ),
    ];

    for (realization, (operator, topology, recovery)) in realizations.iter().zip(expected) {
        let [component] = realization.components.as_slice() else {
            panic!("structural realization must contain exactly one component");
        };
        assert_eq!(
            component.spatial_operator,
            runmat_analysis_core::SpatialOperatorImplementation::Structural(operator)
        );
        assert_eq!(component.topologies, vec![topology]);
        assert_eq!(component.recovery, recovery);
    }
}

// witness: FEA-SHARED-BASELINE-001/decode_revalidation_and_proxy_rejection @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: An incompatible exact axis is rejected before admission.
#[test]
fn incompatible_exact_axis_is_rejected_before_admission() {
    let family = PhysicsFamily::Thermal;
    let procedure = AnalysisProcedure::TransientResponse;
    let mut realization = registered_realizations(family, procedure)
        .into_iter()
        .next()
        .expect("thermal realization must be registered");
    realization.coupling = CouplingImplementation::StaggeredElectrothermalConductanceLine;
    let rejection = admit_formulation(FormulationAdmissionRequest {
        family,
        procedure,
        solution_space: SolutionSpace::Direct,
        realization,
    })
    .expect_err("an unregistered coupling axis must fail closed");

    assert_eq!(
        rejection.reason,
        FormulationAdmissionRejectionReason::IncompatibleAxis(
            runmat_analysis_core::FormulationAxis::Coupling
        )
    );
}

// witness: FEA-SHARED-BASELINE-001/decode_revalidation_and_proxy_rejection @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Unsupported identity and solution space are typed rejections.
#[test]
fn unsupported_identity_and_solution_space_are_typed_rejections() {
    let realization =
        registered_realizations(PhysicsFamily::Thermal, AnalysisProcedure::TransientResponse)
            .into_iter()
            .next()
            .expect("thermal realization must be registered");
    let unsupported_family = admit_formulation(FormulationAdmissionRequest {
        family: PhysicsFamily::Fluid,
        procedure: AnalysisProcedure::ModalExtraction,
        solution_space: SolutionSpace::Direct,
        realization: realization.clone(),
    })
    .expect_err("fluid modal extraction is not exposed");
    assert_eq!(
        unsupported_family.reason,
        FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure
    );

    let unsupported_solution = admit_formulation(FormulationAdmissionRequest {
        family: PhysicsFamily::Thermal,
        procedure: AnalysisProcedure::Static,
        solution_space: SolutionSpace::ModalReduced,
        realization,
    })
    .expect_err("current thermal formulation is not modal-reduced");
    assert_eq!(
        unsupported_solution.reason,
        FormulationAdmissionRejectionReason::UnsupportedSolutionSpace
    );
}

// witness: FEA-SHARED-BASELINE-001/decode_revalidation_and_proxy_rejection @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Thermal-transient identity cold decode revalidates and rejects fields outside the
//   registry identity; no end-to-end publication proof.
#[test]
fn persisted_identity_is_revalidated_and_cannot_forge_production() {
    let family = PhysicsFamily::Thermal;
    let procedure = AnalysisProcedure::TransientResponse;
    let realization = registered_realizations(family, procedure).remove(0);
    let identity = admit_formulation(FormulationAdmissionRequest {
        family,
        procedure,
        solution_space: SolutionSpace::Direct,
        realization,
    })
    .expect("registered formulation should admit")
    .into_identity();
    let mut encoded = serde_json::to_value(&identity).expect("identity should serialize");
    let object = encoded
        .as_object()
        .expect("identity wire representation must be an object");
    assert_eq!(object.len(), 4);
    assert!(!object.contains_key("fallback_policy"));
    assert!(!object.contains_key("formulations"));
    let decoded = serde_json::from_value::<AnalysisRunIdentity>(encoded.clone())
        .expect("registered identity should decode after revalidation");
    assert_eq!(decoded, identity);

    encoded["qualification"] = serde_json::json!("production");
    serde_json::from_value::<AnalysisRunIdentity>(encoded)
        .expect_err("persisted identity rejects a field outside the registry identity");
}

#[test]
fn persisted_identity_cannot_forge_family_or_exact_realization() {
    let family = PhysicsFamily::Thermal;
    let procedure = AnalysisProcedure::TransientResponse;
    let realization = registered_realizations(family, procedure).remove(0);
    let identity = admit_formulation(FormulationAdmissionRequest {
        family,
        procedure,
        solution_space: SolutionSpace::Direct,
        realization,
    })
    .expect("registered formulation should admit")
    .into_identity();
    let encoded = serde_json::to_value(&identity).expect("identity should serialize");
    let mut forged_family = encoded.clone();
    forged_family["family"] = serde_json::json!("electromagnetic");
    let error = serde_json::from_value::<AnalysisRunIdentity>(forged_family)
        .expect_err("family cannot disagree with the exact registered realization");
    assert!(error.to_string().contains("formulation admission rejected"));

    let mut forged_realization = encoded;
    forged_realization["realization"]["components"][0]["sources"] =
        serde_json::json!(["canonical_point_projection"]);
    let error = serde_json::from_value::<AnalysisRunIdentity>(forged_realization)
        .expect_err("unregistered exact realization cannot be persisted");
    assert!(error.to_string().contains("formulation admission rejected"));
}
