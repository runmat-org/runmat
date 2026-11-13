use runmat_analysis_core::{
    AnalysisProcedure, BoundaryTransferImplementation, ConstitutiveImplementation,
    CouplingImplementation, ExactFormulationComponent, ExactFormulationRealization,
    FormulationAdmissionRejectionReason, FormulationAxis, RecoveryImplementation,
    SourceImplementation, SpatialOperatorImplementation, StructuralFormulationRealization,
    StructuralKinematics, StructuralOperatorImplementation, StructuralRecoveryImplementation,
    TemporalImplementation, TopologyImplementation,
};

#[cfg(test)]
mod nonlinear_tests;
mod thermo_mechanical;
#[cfg(test)]
mod volume_source_tests;
pub use thermo_mechanical::thermomechanical_exact_realization;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralRealizationInput {
    pub assemblies: Vec<StructuralFormulationRealization>,
    pub procedure: AnalysisProcedure,
    /// Kinematics of a nonlinear step; ignored (and required small strain) for every other procedure.
    pub kinematics: StructuralKinematics,
    pub coupling: CouplingImplementation,
    pub sources: Vec<SourceImplementation>,
    pub boundary_transfers: Vec<BoundaryTransferImplementation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThermomechanicalRealizationInput {
    pub structural: StructuralRealizationInput,
    pub profile_sources: Vec<SourceImplementation>,
}

/// Translates an assembled structural operator into the only compatible exact realization.
pub fn structural_exact_realization(
    mut input: StructuralRealizationInput,
) -> Result<ExactFormulationRealization, FormulationAdmissionRejectionReason> {
    if input.coupling != CouplingImplementation::Uncoupled {
        return Err(FormulationAdmissionRejectionReason::IncompatibleAxis(
            FormulationAxis::Coupling,
        ));
    }

    let temporal = structural_temporal(input.procedure)
        .ok_or(FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure)?;
    if input.assemblies.is_empty() {
        return Err(FormulationAdmissionRejectionReason::InvalidRealization);
    }
    if input.procedure == AnalysisProcedure::Nonlinear {
        // The genuine constitutive equilibrium path admits one canonical solid basis only.
        // Initial geometry authentication is linear; the solved operator is B^T K_alg B.
        if input.assemblies.len() != 1 {
            return Err(FormulationAdmissionRejectionReason::IncompatibleAxis(
                FormulationAxis::SpatialOperator,
            ));
        }
        let assembly = &mut input.assemblies[0];
        let large = input.kinematics == StructuralKinematics::LargeDeformation;
        assembly.operator = match (assembly.operator, large) {
            (
                StructuralOperatorImplementation::Tet4LinearElasticity
                | StructuralOperatorImplementation::Tet4SmallStrainContinuum,
                false,
            ) => StructuralOperatorImplementation::Tet4SmallStrainContinuum,
            (
                StructuralOperatorImplementation::Tet10LinearElasticity
                | StructuralOperatorImplementation::Tet10SmallStrainContinuum,
                false,
            ) => StructuralOperatorImplementation::Tet10SmallStrainContinuum,
            (
                StructuralOperatorImplementation::Tet4LinearElasticity
                | StructuralOperatorImplementation::Tet4TotalLagrangianContinuum,
                true,
            ) => StructuralOperatorImplementation::Tet4TotalLagrangianContinuum,
            (
                StructuralOperatorImplementation::Tet10LinearElasticity
                | StructuralOperatorImplementation::Tet10TotalLagrangianContinuum,
                true,
            ) => StructuralOperatorImplementation::Tet10TotalLagrangianContinuum,
            _ => {
                return Err(FormulationAdmissionRejectionReason::IncompatibleAxis(
                    FormulationAxis::SpatialOperator,
                ))
            }
        };
    } else if input.kinematics != StructuralKinematics::SmallStrain
        || input.assemblies.iter().any(|assembly| {
            matches!(
                assembly.operator,
                StructuralOperatorImplementation::Tet4SmallStrainContinuum
                    | StructuralOperatorImplementation::Tet10SmallStrainContinuum
                    | StructuralOperatorImplementation::Tet4TotalLagrangianContinuum
                    | StructuralOperatorImplementation::Tet10TotalLagrangianContinuum
            )
        })
    {
        return Err(FormulationAdmissionRejectionReason::IncompatibleAxis(
            FormulationAxis::SpatialOperator,
        ));
    }
    input
        .assemblies
        .sort_by_key(|assembly| structural_operator_order(assembly.operator));
    if input
        .assemblies
        .windows(2)
        .any(|pair| pair[0].operator == pair[1].operator)
    {
        return Err(FormulationAdmissionRejectionReason::InvalidRealization);
    }

    canonicalize_sources(&mut input.sources);
    canonicalize_boundary_transfers(&mut input.boundary_transfers);
    validate_projection_inventory(&input.assemblies, &input.sources, &input.boundary_transfers)?;
    let components = input
        .assemblies
        .into_iter()
        .map(|assembly| {
            let sources = component_sources(assembly.operator, &input.sources);
            structural_component(assembly, input.procedure, sources)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ExactFormulationRealization {
        components,
        temporals: vec![temporal],
        coupling: input.coupling,
        boundary_transfers: input.boundary_transfers,
    })
}

fn structural_component(
    assembly: StructuralFormulationRealization,
    procedure: AnalysisProcedure,
    sources: Vec<SourceImplementation>,
) -> Result<ExactFormulationComponent, FormulationAdmissionRejectionReason> {
    let (topologies, recovery) = match assembly.operator {
        StructuralOperatorImplementation::Tet4LinearElasticity
        | StructuralOperatorImplementation::Tet4SmallStrainContinuum
        | StructuralOperatorImplementation::Tet4TotalLagrangianContinuum => (
            vec![TopologyImplementation::CanonicalSolverMeshTet4],
            RecoveryImplementation::Structural(
                StructuralRecoveryImplementation::Tet4ConstantStrain,
            ),
        ),
        StructuralOperatorImplementation::Tet10LinearElasticity
        | StructuralOperatorImplementation::Tet10SmallStrainContinuum
        | StructuralOperatorImplementation::Tet10TotalLagrangianContinuum => (
            vec![TopologyImplementation::CanonicalSolverMeshTet10],
            RecoveryImplementation::Structural(
                StructuralRecoveryImplementation::Tet10IntegrationPointProjection,
            ),
        ),
        StructuralOperatorImplementation::EulerBernoulliBeam3d => (
            vec![TopologyImplementation::ExplicitStructuralConnectivityBeam],
            RecoveryImplementation::Structural(
                StructuralRecoveryImplementation::BeamSectionResultants,
            ),
        ),
        StructuralOperatorImplementation::Mitc3TriangularShell => (
            vec![TopologyImplementation::ExplicitStructuralConnectivityTriangularShell],
            RecoveryImplementation::Structural(
                StructuralRecoveryImplementation::Mitc3ShellStressResultants,
            ),
        ),
        StructuralOperatorImplementation::SyntheticDiagonalChain => (
            vec![TopologyImplementation::DofAdjacency],
            RecoveryImplementation::Unavailable,
        ),
    };

    let expected_recovery = match recovery {
        RecoveryImplementation::Structural(value) => Some(value),
        RecoveryImplementation::Unavailable => None,
        _ => return Err(FormulationAdmissionRejectionReason::InvalidRealization),
    };
    if assembly.recovery != expected_recovery {
        return Err(FormulationAdmissionRejectionReason::IncompatibleAxis(
            FormulationAxis::Recovery,
        ));
    }

    Ok(ExactFormulationComponent {
        spatial_operator: SpatialOperatorImplementation::Structural(assembly.operator),
        constitutive: if matches!(
            assembly.operator,
            StructuralOperatorImplementation::Tet4TotalLagrangianContinuum
                | StructuralOperatorImplementation::Tet10TotalLagrangianContinuum
        ) {
            ConstitutiveImplementation::GreenLagrangeJ2IsotropicHardening
        } else if procedure == AnalysisProcedure::Nonlinear {
            ConstitutiveImplementation::SmallStrainJ2IsotropicHardening
        } else {
            ConstitutiveImplementation::IsotropicLinearElasticity
        },
        recovery,
        topologies,
        sources,
    })
}

const fn structural_operator_order(operator: StructuralOperatorImplementation) -> u8 {
    match operator {
        StructuralOperatorImplementation::Tet4LinearElasticity => 0,
        StructuralOperatorImplementation::Tet10LinearElasticity => 1,
        StructuralOperatorImplementation::EulerBernoulliBeam3d => 2,
        StructuralOperatorImplementation::Mitc3TriangularShell => 3,
        StructuralOperatorImplementation::SyntheticDiagonalChain => 4,
        StructuralOperatorImplementation::Tet4SmallStrainContinuum => 5,
        StructuralOperatorImplementation::Tet10SmallStrainContinuum => 6,
        StructuralOperatorImplementation::Tet4TotalLagrangianContinuum => 7,
        StructuralOperatorImplementation::Tet10TotalLagrangianContinuum => 8,
    }
}

fn canonicalize_sources(sources: &mut Vec<SourceImplementation>) {
    sources.sort_by_key(|source| source_order(*source));
    sources.dedup();
}

const fn source_order(source: SourceImplementation) -> u8 {
    match source {
        SourceImplementation::NotApplicable => 0,
        SourceImplementation::CanonicalVolumeProjection => 1,
        SourceImplementation::CanonicalBoundaryProjection => 2,
        SourceImplementation::CanonicalBoundaryPressureIntegration => 3,
        SourceImplementation::CanonicalFollowerBoundaryIntegration => 4,
        SourceImplementation::CanonicalPointProjection => 5,
        SourceImplementation::AcousticHashedTargetGraphProjection => 6,
        SourceImplementation::BoundaryResultantMomentProjection => 7,
        SourceImplementation::ReferencePointGeneralizedLoad => 8,
        SourceImplementation::ExplicitRegionNodalDistribution => 9,
        SourceImplementation::UniformNodeDistribution => 10,
        SourceImplementation::PrescribedTemperatureDeltaProfile => 11,
        SourceImplementation::OrdinalRegionTemperatureDistribution => 12,
        SourceImplementation::PersistentRegionElementTemperatureMap => 13,
        SourceImplementation::OrdinalDistribution => 14,
        SourceImplementation::SyntheticProfileFallback => 15,
        SourceImplementation::UnmappedFallback => 16,
        SourceImplementation::SolvedThermalStateNodalField => 17,
    }
}

fn canonicalize_boundary_transfers(transfers: &mut Vec<BoundaryTransferImplementation>) {
    transfers.sort_by_key(|transfer| boundary_transfer_order(*transfer));
    transfers.dedup();
}

const fn boundary_transfer_order(transfer: BoundaryTransferImplementation) -> u8 {
    match transfer {
        BoundaryTransferImplementation::NotApplicable => 0,
        BoundaryTransferImplementation::CanonicalBoundaryProjection => 1,
        BoundaryTransferImplementation::ExplicitRegionConnectivity => 3,
        BoundaryTransferImplementation::GlobalCountProfile => 4,
        BoundaryTransferImplementation::ImplicitNaturalBoundaryAndGaugeAnchor => 5,
        BoundaryTransferImplementation::HeuristicInterfaceTransfer => 6,
        BoundaryTransferImplementation::UnmappedFallback => 7,
        BoundaryTransferImplementation::FullExteriorTangentialPotentialGroundWithWeakCoulombGauge => 8,
        BoundaryTransferImplementation::FullExteriorElectricPerfectConductor => 9,
        BoundaryTransferImplementation::DeclaredConformalInterfaceLaw => 10,
    }
}

fn validate_projection_inventory(
    assemblies: &[StructuralFormulationRealization],
    sources: &[SourceImplementation],
    boundary_transfers: &[BoundaryTransferImplementation],
) -> Result<(), FormulationAdmissionRejectionReason> {
    if sources.is_empty()
        || boundary_transfers.is_empty()
        || (sources.len() > 1 && sources.contains(&SourceImplementation::NotApplicable))
        || (boundary_transfers.len() > 1
            && boundary_transfers.contains(&BoundaryTransferImplementation::NotApplicable))
    {
        return Err(FormulationAdmissionRejectionReason::InvalidRealization);
    }
    if sources.iter().any(|source| {
        *source != SourceImplementation::NotApplicable
            && !assemblies
                .iter()
                .any(|assembly| valid_sources(assembly.operator).contains(source))
    }) {
        return Err(FormulationAdmissionRejectionReason::IncompatibleAxis(
            FormulationAxis::Source,
        ));
    }
    if boundary_transfers.iter().any(|boundary| {
        *boundary != BoundaryTransferImplementation::NotApplicable
            && !assemblies
                .iter()
                .any(|assembly| valid_boundary(assembly.operator) == *boundary)
    }) {
        return Err(FormulationAdmissionRejectionReason::IncompatibleAxis(
            FormulationAxis::BoundaryTransfer,
        ));
    }
    Ok(())
}

fn component_sources(
    operator: StructuralOperatorImplementation,
    sources: &[SourceImplementation],
) -> Vec<SourceImplementation> {
    let mut component = sources
        .iter()
        .copied()
        .filter(|source| {
            *source != SourceImplementation::NotApplicable
                && valid_sources(operator).contains(source)
        })
        .collect::<Vec<_>>();
    if component.is_empty() {
        component.push(SourceImplementation::NotApplicable);
    }
    component
}

fn valid_sources(operator: StructuralOperatorImplementation) -> &'static [SourceImplementation] {
    match operator {
        StructuralOperatorImplementation::Tet4SmallStrainContinuum
        | StructuralOperatorImplementation::Tet10SmallStrainContinuum => &[
            SourceImplementation::CanonicalVolumeProjection,
            SourceImplementation::CanonicalBoundaryProjection,
            SourceImplementation::CanonicalBoundaryPressureIntegration,
            SourceImplementation::BoundaryResultantMomentProjection,
        ],
        StructuralOperatorImplementation::Tet4TotalLagrangianContinuum
        | StructuralOperatorImplementation::Tet10TotalLagrangianContinuum => &[
            SourceImplementation::CanonicalVolumeProjection,
            SourceImplementation::CanonicalBoundaryProjection,
            SourceImplementation::CanonicalBoundaryPressureIntegration,
            SourceImplementation::CanonicalFollowerBoundaryIntegration,
            SourceImplementation::BoundaryResultantMomentProjection,
        ],
        StructuralOperatorImplementation::Tet4LinearElasticity
        | StructuralOperatorImplementation::Tet10LinearElasticity => &[
            SourceImplementation::CanonicalVolumeProjection,
            SourceImplementation::CanonicalBoundaryProjection,
            SourceImplementation::CanonicalBoundaryPressureIntegration,
            SourceImplementation::BoundaryResultantMomentProjection,
            SourceImplementation::ReferencePointGeneralizedLoad,
        ],
        StructuralOperatorImplementation::EulerBernoulliBeam3d
        | StructuralOperatorImplementation::Mitc3TriangularShell => &[
            SourceImplementation::ExplicitRegionNodalDistribution,
            SourceImplementation::BoundaryResultantMomentProjection,
        ],
        StructuralOperatorImplementation::SyntheticDiagonalChain => {
            &[SourceImplementation::OrdinalDistribution]
        }
    }
}

fn valid_boundary(operator: StructuralOperatorImplementation) -> BoundaryTransferImplementation {
    match operator {
        StructuralOperatorImplementation::Tet4LinearElasticity
        | StructuralOperatorImplementation::Tet10LinearElasticity
        | StructuralOperatorImplementation::Tet4SmallStrainContinuum
        | StructuralOperatorImplementation::Tet10SmallStrainContinuum
        | StructuralOperatorImplementation::Tet4TotalLagrangianContinuum
        | StructuralOperatorImplementation::Tet10TotalLagrangianContinuum => {
            BoundaryTransferImplementation::CanonicalBoundaryProjection
        }
        StructuralOperatorImplementation::EulerBernoulliBeam3d
        | StructuralOperatorImplementation::Mitc3TriangularShell => {
            BoundaryTransferImplementation::ExplicitRegionConnectivity
        }
        StructuralOperatorImplementation::SyntheticDiagonalChain => {
            BoundaryTransferImplementation::GlobalCountProfile
        }
    }
}

pub(super) fn structural_temporal(procedure: AnalysisProcedure) -> Option<TemporalImplementation> {
    match procedure {
        AnalysisProcedure::Static => Some(TemporalImplementation::StructuralStaticLinearSolve),
        AnalysisProcedure::ModalExtraction => {
            Some(TemporalImplementation::StructuralSubspaceIteration)
        }
        AnalysisProcedure::PrestressedModalExtraction => {
            Some(TemporalImplementation::StructuralPrestressedSubspaceIteration)
        }
        AnalysisProcedure::EigenvalueBuckling => {
            Some(TemporalImplementation::StructuralEigenvalueBuckling)
        }
        AnalysisProcedure::HarmonicResponse => {
            Some(TemporalImplementation::StructuralHarmonicSolve)
        }
        AnalysisProcedure::PrestressedHarmonicResponse => {
            Some(TemporalImplementation::StructuralPrestressedHarmonicSolve)
        }
        AnalysisProcedure::ModalResponseSpectrum => {
            Some(TemporalImplementation::StructuralModalResponseSpectrumCombination)
        }
        AnalysisProcedure::RandomResponse => {
            Some(TemporalImplementation::StructuralRandomResponseIntegration)
        }
        AnalysisProcedure::TransientResponse => {
            Some(TemporalImplementation::StructuralNewmarkAverageAcceleration)
        }
        AnalysisProcedure::Nonlinear => {
            Some(TemporalImplementation::StructuralQuasistaticNewtonEquilibrium)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        FormulationAdmissionRejectionReason, FormulationAxis, PhysicsFamily, SolutionSpace,
    };

    use super::*;
    use crate::admission::{admit_formulation, FormulationAdmissionRequest};

    fn structural_input(sources: Vec<SourceImplementation>) -> StructuralRealizationInput {
        StructuralRealizationInput {
            assemblies: vec![StructuralFormulationRealization::new(
                StructuralOperatorImplementation::Tet4LinearElasticity,
                Some(StructuralRecoveryImplementation::Tet4ConstantStrain),
            )],
            procedure: AnalysisProcedure::Static,
            kinematics: runmat_analysis_core::StructuralKinematics::SmallStrain,
            coupling: CouplingImplementation::Uncoupled,
            sources,
            boundary_transfers: vec![BoundaryTransferImplementation::CanonicalBoundaryProjection],
        }
    }

    #[test]
    fn structural_source_sets_are_invariant_to_authored_order() {
        let first = structural_exact_realization(structural_input(vec![
            SourceImplementation::CanonicalVolumeProjection,
            SourceImplementation::CanonicalBoundaryProjection,
        ]))
        .expect("valid structural realization");
        let reordered = structural_exact_realization(structural_input(vec![
            SourceImplementation::CanonicalBoundaryProjection,
            SourceImplementation::CanonicalVolumeProjection,
        ]))
        .expect("valid reordered structural realization");

        assert_eq!(first, reordered);
    }

    // witness: FEA-SHARED-BASELINE-001/exact_realization_and_typed_contracts @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
    //   boundary: Modal response spectrum has a distinct admitted temporal realization.
    #[test]
    fn modal_response_spectrum_has_a_distinct_admitted_temporal_realization() {
        let mut input = structural_input(vec![SourceImplementation::CanonicalBoundaryProjection]);
        input.procedure = AnalysisProcedure::ModalResponseSpectrum;
        let realization = structural_exact_realization(input)
            .expect("response-spectrum combination should have an exact realization");
        assert_eq!(
            realization.temporals,
            vec![TemporalImplementation::StructuralModalResponseSpectrumCombination]
        );
        admit_formulation(FormulationAdmissionRequest {
            family: PhysicsFamily::Structural,
            procedure: AnalysisProcedure::ModalResponseSpectrum,
            solution_space: SolutionSpace::ModalReduced,
            realization,
        })
        .expect("modal response-spectrum formulation should be admitted");
    }

    #[test]
    fn random_response_has_a_distinct_admitted_temporal_realization() {
        let mut input = structural_input(vec![SourceImplementation::CanonicalBoundaryProjection]);
        input.procedure = AnalysisProcedure::RandomResponse;
        let realization = structural_exact_realization(input)
            .expect("random-response integration should have an exact realization");
        assert_eq!(
            realization.temporals,
            vec![TemporalImplementation::StructuralRandomResponseIntegration]
        );
        admit_formulation(FormulationAdmissionRequest {
            family: PhysicsFamily::Structural,
            procedure: AnalysisProcedure::RandomResponse,
            solution_space: SolutionSpace::ModalReduced,
            realization,
        })
        .expect("random-response formulation should be admitted");
    }

    #[test]
    fn admission_rejects_noncanonical_or_duplicate_structural_source_evidence() {
        let canonical = structural_exact_realization(structural_input(vec![
            SourceImplementation::CanonicalBoundaryProjection,
            SourceImplementation::CanonicalVolumeProjection,
        ]))
        .expect("valid structural realization");
        admit_formulation(FormulationAdmissionRequest {
            family: PhysicsFamily::Structural,
            procedure: AnalysisProcedure::Static,
            solution_space: SolutionSpace::Direct,
            realization: canonical.clone(),
        })
        .expect("canonical exact evidence should be admitted");

        let mut noncanonical = canonical.clone();
        noncanonical.components[0].sources.reverse();
        let rejection = admit_formulation(FormulationAdmissionRequest {
            family: PhysicsFamily::Structural,
            procedure: AnalysisProcedure::Static,
            solution_space: SolutionSpace::Direct,
            realization: noncanonical,
        })
        .expect_err("noncanonical source order must not produce a second identity");
        assert_eq!(
            rejection.reason,
            FormulationAdmissionRejectionReason::IncompatibleAxis(FormulationAxis::Source)
        );

        let mut duplicated = canonical;
        duplicated.components[0]
            .sources
            .push(SourceImplementation::CanonicalVolumeProjection);
        let rejection = admit_formulation(FormulationAdmissionRequest {
            family: PhysicsFamily::Structural,
            procedure: AnalysisProcedure::Static,
            solution_space: SolutionSpace::Direct,
            realization: duplicated,
        })
        .expect_err("duplicate source evidence must be rejected");
        assert_eq!(
            rejection.reason,
            FormulationAdmissionRejectionReason::IncompatibleAxis(FormulationAxis::Source)
        );
    }

    #[test]
    fn mixed_beam_shell_realization_is_canonical_and_admissible() {
        let input = StructuralRealizationInput {
            assemblies: vec![
                StructuralFormulationRealization::new(
                    StructuralOperatorImplementation::Mitc3TriangularShell,
                    Some(StructuralRecoveryImplementation::Mitc3ShellStressResultants),
                ),
                StructuralFormulationRealization::new(
                    StructuralOperatorImplementation::EulerBernoulliBeam3d,
                    Some(StructuralRecoveryImplementation::BeamSectionResultants),
                ),
            ],
            procedure: AnalysisProcedure::Static,
            kinematics: runmat_analysis_core::StructuralKinematics::SmallStrain,
            coupling: CouplingImplementation::Uncoupled,
            sources: vec![SourceImplementation::ExplicitRegionNodalDistribution],
            boundary_transfers: vec![BoundaryTransferImplementation::ExplicitRegionConnectivity],
        };
        let realization = structural_exact_realization(input).expect("valid mixed realization");
        assert_eq!(realization.components.len(), 2);
        assert!(matches!(
            realization.components[0].spatial_operator,
            SpatialOperatorImplementation::Structural(
                StructuralOperatorImplementation::EulerBernoulliBeam3d
            )
        ));
        assert!(matches!(
            realization.components[1].spatial_operator,
            SpatialOperatorImplementation::Structural(
                StructuralOperatorImplementation::Mitc3TriangularShell
            )
        ));
        admit_formulation(FormulationAdmissionRequest {
            family: PhysicsFamily::Structural,
            procedure: AnalysisProcedure::Static,
            solution_space: SolutionSpace::Direct,
            realization,
        })
        .expect("the exact mixed formulation must pass admission");
    }

    #[test]
    fn mixed_solid_beam_realization_assigns_each_source_to_its_owner() {
        let realization = structural_exact_realization(StructuralRealizationInput {
            assemblies: vec![
                StructuralFormulationRealization::new(
                    StructuralOperatorImplementation::EulerBernoulliBeam3d,
                    Some(StructuralRecoveryImplementation::BeamSectionResultants),
                ),
                StructuralFormulationRealization::new(
                    StructuralOperatorImplementation::Tet4LinearElasticity,
                    Some(StructuralRecoveryImplementation::Tet4ConstantStrain),
                ),
            ],
            procedure: AnalysisProcedure::Static,
            kinematics: runmat_analysis_core::StructuralKinematics::SmallStrain,
            coupling: CouplingImplementation::Uncoupled,
            sources: vec![
                SourceImplementation::ExplicitRegionNodalDistribution,
                SourceImplementation::CanonicalBoundaryProjection,
            ],
            boundary_transfers: vec![
                BoundaryTransferImplementation::ExplicitRegionConnectivity,
                BoundaryTransferImplementation::CanonicalBoundaryProjection,
            ],
        })
        .expect("valid mixed solid-beam realization");

        assert_eq!(
            realization.components[0].sources,
            vec![SourceImplementation::CanonicalBoundaryProjection]
        );
        assert_eq!(
            realization.components[1].sources,
            vec![SourceImplementation::ExplicitRegionNodalDistribution]
        );
        admit_formulation(FormulationAdmissionRequest {
            family: PhysicsFamily::Structural,
            procedure: AnalysisProcedure::Static,
            solution_space: SolutionSpace::Direct,
            realization,
        })
        .expect("component-owned mixed sources must round-trip through admission");
    }
}
