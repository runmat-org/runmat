use runmat_analysis_core::{
    AnalysisProcedure, CoupledPhysicsFamily, CouplingImplementation,
    FormulationAdmissionRejectionReason, FormulationMaturity, FormulationRequirement,
    PhysicsFamily, RecoveryImplementation, SolutionSpace, SourceImplementation,
    StructuralFormulationRealization, StructuralOperatorImplementation,
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
        (
            PhysicsFamily::Electromagnetic,
            AnalysisProcedure::HarmonicResponse,
        ),
        (PhysicsFamily::Acoustic, AnalysisProcedure::HarmonicResponse),
        (PhysicsFamily::Fluid, AnalysisProcedure::Static),
        (PhysicsFamily::Fluid, AnalysisProcedure::TransientResponse),
        (
            PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
            AnalysisProcedure::Static,
        ),
        (
            PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
            AnalysisProcedure::TransientResponse,
        ),
        (
            PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction),
            AnalysisProcedure::TransientResponse,
        ),
        (
            PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
            AnalysisProcedure::Static,
        ),
        (
            PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
            AnalysisProcedure::Nonlinear,
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
            FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure,
        ),
    ];

    for (family, procedure, realization, expected_reason) in cases {
        let rejection = admit_formulation(FormulationAdmissionRequest {
            family,
            procedure,
            solution_space: SolutionSpace::Direct,
            requirement: FormulationRequirement::Experimental,
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
            None,
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
            .map(|assembly| {
                structural_exact_realization(StructuralRealizationInput {
                    assemblies: vec![assembly],
                    procedure,
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
fn every_registered_branch_requires_explicit_experimental_admission() {
    for (family, procedure) in exposed_identities() {
        let realizations = registered_realizations(family, procedure);
        assert!(
            !realizations.is_empty(),
            "registry has no realization for family={family:?} procedure={procedure:?}"
        );
        for realization in realizations {
            let experimental = FormulationAdmissionRequest {
                family,
                procedure,
                solution_space: SolutionSpace::Direct,
                requirement: FormulationRequirement::Experimental,
                realization: realization.clone(),
            };
            let receipt = admit_formulation(experimental.clone())
                .expect("registered current formulation should admit experimentally");
            assert_eq!(receipt.family(), family);
            assert_eq!(receipt.procedure(), procedure);
            assert_eq!(receipt.solution_space(), SolutionSpace::Direct);
            assert_eq!(receipt.realization(), &realization);
            assert_eq!(receipt.qualification(), FormulationMaturity::Experimental);
            let identity = receipt.clone().into_identity();
            assert_eq!(identity.family(), family);
            assert_eq!(identity.procedure(), procedure);
            assert_eq!(identity.solution_space(), SolutionSpace::Direct);
            assert_eq!(identity.qualification(), FormulationMaturity::Experimental);
            assert!(identity.permits_fallback());
            let production = FormulationAdmissionRequest {
                requirement: FormulationRequirement::Production,
                ..experimental
            };
            let rejection = admit_formulation(production)
                .expect_err("no current formulation has completed its production gate");
            assert_eq!(
                rejection.reason,
                FormulationAdmissionRejectionReason::ProductionQualificationMissing,
                "family={family:?} procedure={procedure:?} realization={realization:?}"
            );
        }
    }
}

#[test]
fn availability_gate_rejects_production_for_every_registered_procedure() {
    for (family, procedure) in exposed_identities() {
        require_qualified_formulation(
            family,
            procedure,
            SolutionSpace::Direct,
            FormulationRequirement::Experimental,
        )
        .expect("every registered procedure should be available experimentally");
        let rejection = require_qualified_formulation(
            family,
            procedure,
            SolutionSpace::Direct,
            FormulationRequirement::Production,
        )
        .expect_err("no current procedure has completed production qualification");
        assert_eq!(
            rejection.reason,
            FormulationAdmissionRejectionReason::ProductionQualificationMissing
        );
    }
}

#[test]
fn thermomechanical_structural_procedures_are_available_experimentally() {
    for procedure in [AnalysisProcedure::Static, AnalysisProcedure::Nonlinear] {
        require_qualified_formulation(
            PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
            procedure,
            SolutionSpace::Direct,
            FormulationRequirement::Experimental,
        )
        .expect("thermomechanical structural procedures should pass pre-availability");
    }

    let rejection = require_qualified_formulation(
        PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
        AnalysisProcedure::TransientResponse,
        SolutionSpace::Direct,
        FormulationRequirement::Experimental,
    )
    .expect_err("thermomechanical transient requires a solver-owned coupled thermal state");
    assert_eq!(
        rejection.reason,
        FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure
    );
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
            RecoveryImplementation::Unavailable,
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

#[test]
fn incompatible_exact_axis_is_rejected_before_qualification() {
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
        requirement: FormulationRequirement::Experimental,
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

#[test]
fn unsupported_identity_and_solution_space_are_typed_rejections() {
    let realization =
        registered_realizations(PhysicsFamily::Thermal, AnalysisProcedure::TransientResponse)
            .into_iter()
            .next()
            .expect("thermal realization must be registered");
    let unsupported_family = admit_formulation(FormulationAdmissionRequest {
        family: PhysicsFamily::Acoustic,
        procedure: AnalysisProcedure::TransientResponse,
        solution_space: SolutionSpace::Direct,
        requirement: FormulationRequirement::Experimental,
        realization: realization.clone(),
    })
    .expect_err("transient acoustic is not exposed");
    assert_eq!(
        unsupported_family.reason,
        FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure
    );

    let unsupported_solution = admit_formulation(FormulationAdmissionRequest {
        family: PhysicsFamily::Thermal,
        procedure: AnalysisProcedure::Static,
        solution_space: SolutionSpace::ModalReduced,
        requirement: FormulationRequirement::Experimental,
        realization,
    })
    .expect_err("current thermal formulation is not modal-reduced");
    assert_eq!(
        unsupported_solution.reason,
        FormulationAdmissionRejectionReason::UnsupportedSolutionSpace
    );
}

#[test]
fn persisted_identity_is_revalidated_and_cannot_forge_production() {
    let family = PhysicsFamily::Thermal;
    let procedure = AnalysisProcedure::TransientResponse;
    let realization = registered_realizations(family, procedure).remove(0);
    let identity = admit_formulation(FormulationAdmissionRequest {
        family,
        procedure,
        solution_space: SolutionSpace::Direct,
        requirement: FormulationRequirement::Experimental,
        realization,
    })
    .expect("registered experimental formulation should admit")
    .into_identity();
    let mut encoded = serde_json::to_value(&identity).expect("identity should serialize");
    let object = encoded
        .as_object()
        .expect("identity wire representation must be an object");
    assert_eq!(object.len(), 5);
    assert!(!object.contains_key("fallback_policy"));
    assert!(!object.contains_key("formulations"));
    let decoded = serde_json::from_value::<AnalysisRunIdentity>(encoded.clone())
        .expect("registered identity should decode after revalidation");
    assert_eq!(decoded, identity);

    encoded["qualification"] = serde_json::json!("production");
    let error = serde_json::from_value::<AnalysisRunIdentity>(encoded)
        .expect_err("persisted identity cannot forge production qualification");
    assert!(error.to_string().contains("ProductionQualificationMissing"));
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
        requirement: FormulationRequirement::Experimental,
        realization,
    })
    .expect("registered experimental formulation should admit")
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
