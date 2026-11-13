use super::*;

fn input(
    operator: StructuralOperatorImplementation,
    recovery: Option<StructuralRecoveryImplementation>,
    source: SourceImplementation,
) -> StructuralRealizationInput {
    StructuralRealizationInput {
        assemblies: vec![StructuralFormulationRealization::new(operator, recovery)],
        procedure: AnalysisProcedure::TransientResponse,
        kinematics: runmat_analysis_core::StructuralKinematics::SmallStrain,
        coupling: CouplingImplementation::Uncoupled,
        sources: vec![source],
        boundary_transfers: vec![valid_boundary(operator)],
    }
}

#[test]
fn canonical_volume_sources_are_exact_solid_evidence_not_equal_node_distribution() {
    for (operator, recovery) in [
        (
            StructuralOperatorImplementation::Tet4LinearElasticity,
            Some(StructuralRecoveryImplementation::Tet4ConstantStrain),
        ),
        (
            StructuralOperatorImplementation::Tet10LinearElasticity,
            Some(StructuralRecoveryImplementation::Tet10IntegrationPointProjection),
        ),
    ] {
        let realized = structural_exact_realization(input(
            operator,
            recovery,
            SourceImplementation::CanonicalVolumeProjection,
        ))
        .unwrap();
        assert_eq!(
            realized.components[0].sources,
            vec![SourceImplementation::CanonicalVolumeProjection]
        );
        assert_eq!(
            structural_exact_realization(input(
                operator,
                recovery,
                SourceImplementation::UniformNodeDistribution
            ))
            .unwrap_err(),
            FormulationAdmissionRejectionReason::IncompatibleAxis(FormulationAxis::Source)
        );
    }
}

#[test]
fn volume_source_evidence_does_not_authorize_explicit_beam_or_synthetic_projection() {
    for (operator, recovery) in [
        (
            StructuralOperatorImplementation::EulerBernoulliBeam3d,
            Some(StructuralRecoveryImplementation::BeamSectionResultants),
        ),
        (
            StructuralOperatorImplementation::SyntheticDiagonalChain,
            None,
        ),
    ] {
        assert_eq!(
            structural_exact_realization(input(
                operator,
                recovery,
                SourceImplementation::CanonicalVolumeProjection
            ))
            .unwrap_err(),
            FormulationAdmissionRejectionReason::IncompatibleAxis(FormulationAxis::Source)
        );
    }
}

#[test]
fn nonlinear_solids_retain_canonical_volume_source_evidence() {
    for (operator, recovery) in [
        (
            StructuralOperatorImplementation::Tet4LinearElasticity,
            StructuralRecoveryImplementation::Tet4ConstantStrain,
        ),
        (
            StructuralOperatorImplementation::Tet10LinearElasticity,
            StructuralRecoveryImplementation::Tet10IntegrationPointProjection,
        ),
    ] {
        let mut request = input(
            operator,
            Some(recovery),
            SourceImplementation::CanonicalVolumeProjection,
        );
        request.procedure = AnalysisProcedure::Nonlinear;
        let realized = structural_exact_realization(request.clone()).unwrap();
        assert_eq!(
            realized.components[0].sources,
            vec![SourceImplementation::CanonicalVolumeProjection]
        );
        request.sources = vec![SourceImplementation::UniformNodeDistribution];
        assert_eq!(
            structural_exact_realization(request).unwrap_err(),
            FormulationAdmissionRejectionReason::IncompatibleAxis(FormulationAxis::Source)
        );
    }
}
