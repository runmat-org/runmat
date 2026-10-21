use super::*;
use crate::admission::{admit_formulation, FormulationAdmissionRequest};
use runmat_analysis_core::{PhysicsFamily, SolutionSpace};

fn input(operator: StructuralOperatorImplementation) -> StructuralRealizationInput {
    let recovery = match operator {
        StructuralOperatorImplementation::Tet4LinearElasticity
        | StructuralOperatorImplementation::Tet4SmallStrainContinuum
        | StructuralOperatorImplementation::Tet4TotalLagrangianContinuum => {
            Some(StructuralRecoveryImplementation::Tet4ConstantStrain)
        }
        StructuralOperatorImplementation::Tet10LinearElasticity
        | StructuralOperatorImplementation::Tet10SmallStrainContinuum
        | StructuralOperatorImplementation::Tet10TotalLagrangianContinuum => {
            Some(StructuralRecoveryImplementation::Tet10IntegrationPointProjection)
        }
        StructuralOperatorImplementation::EulerBernoulliBeam3d => {
            Some(StructuralRecoveryImplementation::BeamSectionResultants)
        }
        StructuralOperatorImplementation::Mitc3TriangularShell => {
            Some(StructuralRecoveryImplementation::Mitc3ShellStressResultants)
        }
        StructuralOperatorImplementation::SyntheticDiagonalChain => None,
    };
    StructuralRealizationInput {
        assemblies: vec![StructuralFormulationRealization::new(operator, recovery)],
        procedure: AnalysisProcedure::Nonlinear,
        kinematics: runmat_analysis_core::StructuralKinematics::SmallStrain,
        coupling: CouplingImplementation::Uncoupled,
        sources: vec![SourceImplementation::CanonicalBoundaryPressureIntegration],
        boundary_transfers: vec![BoundaryTransferImplementation::CanonicalBoundaryProjection],
    }
}

#[test]
fn canonical_geometry_translates_to_exact_j2_equilibrium_axes() {
    for (geometry, solved) in [
        (
            StructuralOperatorImplementation::Tet4LinearElasticity,
            StructuralOperatorImplementation::Tet4SmallStrainContinuum,
        ),
        (
            StructuralOperatorImplementation::Tet10LinearElasticity,
            StructuralOperatorImplementation::Tet10SmallStrainContinuum,
        ),
    ] {
        let realization = structural_exact_realization(input(geometry)).unwrap();
        assert_eq!(
            realization.components[0].spatial_operator,
            SpatialOperatorImplementation::Structural(solved)
        );
        assert_eq!(
            realization.components[0].constitutive,
            ConstitutiveImplementation::SmallStrainJ2IsotropicHardening
        );
        assert_eq!(
            realization.temporals,
            [TemporalImplementation::StructuralQuasistaticNewtonEquilibrium]
        );
        assert_eq!(
            realization,
            structural_exact_realization(input(solved)).unwrap()
        );
        admit_formulation(FormulationAdmissionRequest {
            family: PhysicsFamily::Structural,
            procedure: AnalysisProcedure::Nonlinear,
            solution_space: SolutionSpace::Direct,
            realization,
        })
        .expect("J2 nonlinear realization is admitted");
    }
}

#[test]
fn nonlinear_identity_rejects_unsupported_geometry_and_sources() {
    for operator in [
        StructuralOperatorImplementation::EulerBernoulliBeam3d,
        StructuralOperatorImplementation::Mitc3TriangularShell,
        StructuralOperatorImplementation::SyntheticDiagonalChain,
    ] {
        assert!(structural_exact_realization(input(operator)).is_err());
    }
    let mut mixed = input(StructuralOperatorImplementation::Tet4LinearElasticity);
    mixed
        .assemblies
        .extend(input(StructuralOperatorImplementation::Tet10LinearElasticity).assemblies);
    assert!(structural_exact_realization(mixed).is_err());
    let mut reference_point = input(StructuralOperatorImplementation::Tet4LinearElasticity);
    reference_point.sources = vec![SourceImplementation::ReferencePointGeneralizedLoad];
    assert!(structural_exact_realization(reference_point).is_err());
    let mut coupled = input(StructuralOperatorImplementation::Tet4LinearElasticity);
    coupled.coupling = CouplingImplementation::OnePassPrescribedThermomechanicalProfile;
    assert!(structural_exact_realization(coupled).is_err());
}

#[test]
fn nonlinear_axes_cannot_be_relabelled_as_linear_or_reverse() {
    let mut nonlinear = input(StructuralOperatorImplementation::Tet4SmallStrainContinuum);
    nonlinear.procedure = AnalysisProcedure::Static;
    assert!(structural_exact_realization(nonlinear).is_err());
    let mut wrong_recovery = input(StructuralOperatorImplementation::Tet4LinearElasticity);
    wrong_recovery.assemblies[0].recovery =
        Some(StructuralRecoveryImplementation::BeamSectionResultants);
    assert!(structural_exact_realization(wrong_recovery).is_err());
    let mut forged = structural_exact_realization(input(
        StructuralOperatorImplementation::Tet4LinearElasticity,
    ))
    .unwrap();
    forged.components[0].spatial_operator = SpatialOperatorImplementation::Structural(
        StructuralOperatorImplementation::Tet4LinearElasticity,
    );
    assert!(admit_formulation(FormulationAdmissionRequest {
        family: PhysicsFamily::Structural,
        procedure: AnalysisProcedure::Nonlinear,
        solution_space: SolutionSpace::Direct,
        realization: forged,
    })
    .is_err());
}

#[test]
fn retired_scalar_axes_are_not_wire_compatibility_aliases() {
    assert!(serde_json::from_str::<ConstitutiveImplementation>(
        "\"reduced_scalar_structural_nonlinearity\""
    )
    .is_err());
    assert!(serde_json::from_str::<TemporalImplementation>(
        "\"structural_scalar_residual_iteration\""
    )
    .is_err());
}

#[test]
fn coupled_profiles_cannot_borrow_the_uncoupled_j2_identity() {
    for coupling in [
        CouplingImplementation::OnePassPrescribedThermomechanicalProfile,
        CouplingImplementation::OneWaySolvedThermoelasticState,
    ] {
        let mut structural = input(StructuralOperatorImplementation::Tet4LinearElasticity);
        structural.coupling = coupling;
        assert_eq!(
            thermomechanical_exact_realization(ThermomechanicalRealizationInput {
                structural,
                profile_sources: vec![SourceImplementation::PrescribedTemperatureDeltaProfile],
            })
            .unwrap_err(),
            FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure
        );
    }
    for registered in crate::admission::registered_formulations(
        PhysicsFamily::Coupled(
            runmat_analysis_core::CoupledPhysicsFamily::FluidStructureInteraction,
        ),
        AnalysisProcedure::TransientResponse,
    ) {
        assert!(registered
            .realization()
            .components
            .iter()
            .all(|component| component.constitutive
                != ConstitutiveImplementation::SmallStrainJ2IsotropicHardening));
    }
}
