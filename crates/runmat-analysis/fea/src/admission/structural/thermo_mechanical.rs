//! Exact experimental identity for prescribed solid thermoelasticity and reduced profiles.

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
            ExactFormulationComponent {
                spatial_operator:
                    SpatialOperatorImplementation::IntegratedPrescribedSolidThermoelasticLoad,
                constitutive: ConstitutiveImplementation::ElementOwnedIsotropicThermalExpansion,
                recovery: RecoveryImplementation::ElementOwnedThermoelasticStressEnergy,
                topologies: vec![topology],
                sources: input.profile_sources,
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
