//! Exact identity for prescribed solid thermoelasticity and reduced profiles.

use runmat_analysis_core::{
    AnalysisProcedure, ConstitutiveImplementation, CouplingImplementation,
    ExactFormulationComponent, ExactFormulationRealization, FormulationAdmissionRejectionReason,
    FormulationAxis, RecoveryImplementation, SourceImplementation, SpatialOperatorImplementation,
    StructuralOperatorImplementation, TemporalImplementation, TopologyImplementation,
};

use super::{
    canonicalize_sources, structural_exact_realization, structural_temporal,
    StructuralRealizationInput, ThermomechanicalRealizationInput,
};

pub fn thermomechanical_exact_realization(
    mut input: ThermomechanicalRealizationInput,
) -> Result<ExactFormulationRealization, FormulationAdmissionRejectionReason> {
    if input.structural.procedure == AnalysisProcedure::Nonlinear {
        return Err(FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure);
    }
    if input.structural.coupling == CouplingImplementation::OneWaySolvedThermoelasticState {
        return solved_thermal_realization(input);
    }
    if input.structural.coupling != CouplingImplementation::OnePassPrescribedThermomechanicalProfile
    {
        return Err(FormulationAdmissionRejectionReason::IncompatibleAxis(
            FormulationAxis::Coupling,
        ));
    }
    canonicalize_sources(&mut input.profile_sources);
    if input.profile_sources.is_empty()
        || input.profile_sources.iter().any(|source| {
            !matches!(
                source,
                SourceImplementation::PrescribedTemperatureDeltaProfile
                    | SourceImplementation::OrdinalRegionTemperatureDistribution
            )
        })
        || !input
            .profile_sources
            .contains(&SourceImplementation::PrescribedTemperatureDeltaProfile)
    {
        return Err(FormulationAdmissionRejectionReason::IncompatibleAxis(
            FormulationAxis::Source,
        ));
    }
    let procedure = input.structural.procedure;
    let physical_solid_topology =
        if procedure == AnalysisProcedure::Static && input.structural.assemblies.len() == 1 {
            match input.structural.assemblies[0].operator {
                StructuralOperatorImplementation::Tet4LinearElasticity => {
                    Some(TopologyImplementation::CanonicalSolverMeshTet4)
                }
                StructuralOperatorImplementation::Tet10LinearElasticity => {
                    Some(TopologyImplementation::CanonicalSolverMeshTet10)
                }
                _ => None,
            }
        } else {
            None
        };
    let structural_boundaries = input.structural.boundary_transfers.clone();
    let mut realization = structural_exact_realization(StructuralRealizationInput {
        coupling: CouplingImplementation::Uncoupled,
        ..input.structural
    })?;
    realization
        .components
        .push(if let Some(topology) = physical_solid_topology {
            let physical_sources = input
                .profile_sources
                .into_iter()
                .map(|source| match source {
                    SourceImplementation::OrdinalRegionTemperatureDistribution => {
                        SourceImplementation::PersistentRegionElementTemperatureMap
                    }
                    other => other,
                })
                .collect();
            ExactFormulationComponent {
                spatial_operator:
                    SpatialOperatorImplementation::IntegratedPrescribedSolidThermoelasticLoad,
                constitutive: ConstitutiveImplementation::ElementOwnedIsotropicThermalExpansion,
                recovery: RecoveryImplementation::ElementOwnedThermoelasticStressEnergy,
                topologies: vec![topology],
                sources: physical_sources,
            }
        } else {
            ExactFormulationComponent {
                spatial_operator: SpatialOperatorImplementation::PrescribedThermalCouplingProfile,
                constitutive: ConstitutiveImplementation::ReducedThermalCoefficientProfile,
                recovery: RecoveryImplementation::PrescribedThermomechanicalProfileDerivedFields,
                topologies: vec![TopologyImplementation::DofAdjacency],
                sources: input.profile_sources,
            }
        });
    let structural_temporal = structural_temporal(procedure)
        .ok_or(FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure)?;
    realization.temporals = vec![
        structural_temporal,
        TemporalImplementation::PrescribedThermomechanicalProfileApplication,
    ];
    realization.coupling = CouplingImplementation::OnePassPrescribedThermomechanicalProfile;
    realization.boundary_transfers = structural_boundaries;
    Ok(realization)
}

fn solved_thermal_realization(
    input: ThermomechanicalRealizationInput,
) -> Result<ExactFormulationRealization, FormulationAdmissionRejectionReason> {
    if input.profile_sources != [SourceImplementation::SolvedThermalStateNodalField] {
        return Err(FormulationAdmissionRejectionReason::IncompatibleAxis(
            FormulationAxis::Source,
        ));
    }
    if input.structural.procedure != AnalysisProcedure::Static
        || input.structural.assemblies.len() != 1
    {
        return Err(FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure);
    }
    let topology = match input.structural.assemblies[0].operator {
        StructuralOperatorImplementation::Tet4LinearElasticity => {
            TopologyImplementation::CanonicalSolverMeshTet4
        }
        StructuralOperatorImplementation::Tet10LinearElasticity => {
            TopologyImplementation::CanonicalSolverMeshTet10
        }
        _ => {
            return Err(FormulationAdmissionRejectionReason::IncompatibleAxis(
                FormulationAxis::Topology,
            ))
        }
    };
    let mut realization = structural_exact_realization(StructuralRealizationInput {
        coupling: CouplingImplementation::Uncoupled,
        ..input.structural
    })?;
    realization.components.push(ExactFormulationComponent {
        spatial_operator: SpatialOperatorImplementation::IntegratedSolvedSolidThermoelasticLoad,
        constitutive: ConstitutiveImplementation::ElementOwnedIsotropicThermalExpansion,
        recovery: RecoveryImplementation::ElementOwnedThermoelasticStressEnergy,
        topologies: vec![topology],
        sources: input.profile_sources,
    });
    realization
        .temporals
        .push(TemporalImplementation::SolvedThermalStateApplication);
    realization.coupling = CouplingImplementation::OneWaySolvedThermoelasticState;
    Ok(realization)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_core::{
        BoundaryTransferImplementation, StructuralFormulationRealization,
        StructuralRecoveryImplementation,
    };

    #[test]
    fn solved_thermal_identity_requires_static_solid_and_exclusive_retained_field() {
        let input = ThermomechanicalRealizationInput {
            structural: StructuralRealizationInput {
                assemblies: vec![StructuralFormulationRealization::new(
                    StructuralOperatorImplementation::Tet4LinearElasticity,
                    Some(StructuralRecoveryImplementation::Tet4ConstantStrain),
                )],
                procedure: AnalysisProcedure::Static,
                kinematics: runmat_analysis_core::StructuralKinematics::SmallStrain,
                coupling: CouplingImplementation::OneWaySolvedThermoelasticState,
                sources: vec![SourceImplementation::NotApplicable],
                boundary_transfers: vec![BoundaryTransferImplementation::NotApplicable],
            },
            profile_sources: vec![SourceImplementation::SolvedThermalStateNodalField],
        };
        let identity = thermomechanical_exact_realization(input.clone()).unwrap();
        assert_eq!(
            identity.components[1].spatial_operator,
            SpatialOperatorImplementation::IntegratedSolvedSolidThermoelasticLoad
        );
        assert_eq!(
            identity.temporals,
            vec![
                TemporalImplementation::StructuralStaticLinearSolve,
                TemporalImplementation::SolvedThermalStateApplication
            ]
        );
        for procedure in [
            AnalysisProcedure::TransientResponse,
            AnalysisProcedure::Nonlinear,
        ] {
            let mut invalid = input.clone();
            invalid.structural.procedure = procedure;
            assert!(thermomechanical_exact_realization(invalid).is_err());
        }
        let mut mixed = input.clone();
        mixed
            .profile_sources
            .push(SourceImplementation::PrescribedTemperatureDeltaProfile);
        assert!(thermomechanical_exact_realization(mixed).is_err());
        let mut synthetic = input;
        synthetic.structural.assemblies[0] = StructuralFormulationRealization::new(
            StructuralOperatorImplementation::SyntheticDiagonalChain,
            None,
        );
        assert!(thermomechanical_exact_realization(synthetic).is_err());
    }
}
