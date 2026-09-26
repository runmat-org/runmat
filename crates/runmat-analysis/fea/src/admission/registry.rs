use runmat_analysis_core::{
    is_exposed_analysis_identity, CouplingImplementation, ExactFormulationRealization,
    FormulationAdmissionRejectionReason, FormulationAxis, FormulationMaturity,
    FormulationRequirement, PhysicsFamily, RecoveryImplementation, SolutionSpace,
    SpatialOperatorImplementation, StructuralFormulationRealization,
};

pub(super) fn require_available(
    family: PhysicsFamily,
    procedure: runmat_analysis_core::AnalysisProcedure,
    solution_space: SolutionSpace,
    requirement: FormulationRequirement,
) -> Result<(), FormulationAdmissionRejectionReason> {
    if solution_space != SolutionSpace::Direct
        && !(family == PhysicsFamily::Structural
            && is_exposed_analysis_identity(family, procedure, solution_space))
    {
        return Err(FormulationAdmissionRejectionReason::UnsupportedSolutionSpace);
    }
    if !is_exposed_analysis_identity(family, procedure, solution_space) {
        return Err(FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure);
    }
    let qualifications = available_qualifications(family, procedure);
    if qualifications.is_empty() {
        return Err(FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure);
    }
    if requirement == FormulationRequirement::Production
        && !qualifications.contains(&FormulationMaturity::Production)
    {
        return Err(FormulationAdmissionRejectionReason::ProductionQualificationMissing);
    }
    Ok(())
}

use super::{
    realizations::{registered_formulations, RegisteredFormulation},
    structural::{
        structural_exact_realization, thermomechanical_exact_realization,
        StructuralRealizationInput, ThermomechanicalRealizationInput,
    },
    FormulationAdmissionRequest,
};

pub(super) fn qualification(
    request: &FormulationAdmissionRequest,
) -> Result<FormulationMaturity, FormulationAdmissionRejectionReason> {
    if request.solution_space != SolutionSpace::Direct
        && !(request.family == PhysicsFamily::Structural
            && is_exposed_analysis_identity(
                request.family,
                request.procedure,
                request.solution_space,
            ))
    {
        return Err(FormulationAdmissionRejectionReason::UnsupportedSolutionSpace);
    }
    if !is_exposed_analysis_identity(request.family, request.procedure, request.solution_space) {
        return Err(FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure);
    }

    let qualification = if request.family == PhysicsFamily::Structural
        || request.family
            == PhysicsFamily::Coupled(runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical)
    {
        let expected =
            validate_structural_identity(request, request.family != PhysicsFamily::Structural)?;
        if request.realization != expected {
            return Err(first_incompatible_axis(&request.realization, &expected));
        }
        registered_structural_formulation(expected).qualification()
    } else {
        let registered = registered_formulations(request.family, request.procedure);
        if registered.is_empty() {
            return Err(FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure);
        }
        let Some(entry) = registered
            .iter()
            .find(|entry| entry.realization() == &request.realization)
        else {
            let reason = registered
                .first()
                .map(|expected| {
                    first_incompatible_axis(&request.realization, expected.realization())
                })
                .unwrap_or(FormulationAdmissionRejectionReason::UnregisteredRealization);
            return Err(reason);
        };
        entry.qualification()
    };

    Ok(qualification)
}

fn available_qualifications(
    family: PhysicsFamily,
    procedure: runmat_analysis_core::AnalysisProcedure,
) -> Vec<FormulationMaturity> {
    if matches!(
        family,
        PhysicsFamily::Structural
            | PhysicsFamily::Coupled(runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical)
    ) {
        if is_exposed_analysis_identity(family, procedure, SolutionSpace::Direct)
            || is_exposed_analysis_identity(family, procedure, SolutionSpace::ModalReduced)
        {
            structural_available_qualifications()
        } else {
            Vec::new()
        }
    } else {
        let mut qualifications = Vec::new();
        for entry in registered_formulations(family, procedure) {
            if !qualifications.contains(&entry.qualification()) {
                qualifications.push(entry.qualification());
            }
        }
        qualifications
    }
}

fn registered_structural_formulation(
    exact_realization: ExactFormulationRealization,
) -> RegisteredFormulation {
    let qualification = qualification_from_exact_whitelist(
        &exact_realization,
        &production_structural_realizations(),
    );
    RegisteredFormulation::new(exact_realization, qualification)
}

fn qualification_from_exact_whitelist(
    exact_realization: &ExactFormulationRealization,
    production_realizations: &[ExactFormulationRealization],
) -> FormulationMaturity {
    if production_realizations.contains(exact_realization) {
        FormulationMaturity::Production
    } else {
        FormulationMaturity::Experimental
    }
}

fn production_structural_realizations() -> Vec<ExactFormulationRealization> {
    // Empty until a complete exact realization passes its production verification gate.
    Vec::new()
}

fn structural_available_qualifications() -> Vec<FormulationMaturity> {
    let mut qualifications = vec![FormulationMaturity::Experimental];
    if !production_structural_realizations().is_empty() {
        qualifications.push(FormulationMaturity::Production);
    }
    qualifications
}

fn validate_structural_identity(
    request: &FormulationAdmissionRequest,
    thermomechanical: bool,
) -> Result<ExactFormulationRealization, FormulationAdmissionRejectionReason> {
    let structural_component_count = request
        .realization
        .components
        .len()
        .checked_sub(usize::from(thermomechanical))
        .ok_or(FormulationAdmissionRejectionReason::InvalidRealization)?;
    if structural_component_count == 0 {
        return Err(FormulationAdmissionRejectionReason::IncompatibleAxis(
            FormulationAxis::SpatialOperator,
        ));
    }
    let structural_components = &request.realization.components[..structural_component_count];
    let mut sources = Vec::new();
    for source in structural_components
        .iter()
        .flat_map(|component| component.sources.iter().copied())
    {
        if source != runmat_analysis_core::SourceImplementation::NotApplicable
            && !sources.contains(&source)
        {
            sources.push(source);
        }
    }
    if sources.is_empty() {
        sources.push(runmat_analysis_core::SourceImplementation::NotApplicable);
    }
    let assemblies = structural_components
        .iter()
        .map(|component| {
            let SpatialOperatorImplementation::Structural(operator) = component.spatial_operator
            else {
                return Err(FormulationAdmissionRejectionReason::IncompatibleAxis(
                    FormulationAxis::SpatialOperator,
                ));
            };
            let recovery = match component.recovery {
                RecoveryImplementation::Structural(recovery) => Some(recovery),
                RecoveryImplementation::Unavailable => None,
                _ => {
                    return Err(FormulationAdmissionRejectionReason::IncompatibleAxis(
                        FormulationAxis::Recovery,
                    ))
                }
            };
            Ok(StructuralFormulationRealization::new(operator, recovery))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected_coupling = if thermomechanical {
        CouplingImplementation::OnePassPrescribedThermomechanicalProfile
    } else {
        CouplingImplementation::Uncoupled
    };
    if request.realization.coupling != expected_coupling {
        return Err(FormulationAdmissionRejectionReason::IncompatibleAxis(
            FormulationAxis::Coupling,
        ));
    }

    let structural_boundaries = request.realization.boundary_transfers.clone();
    let input = StructuralRealizationInput {
        assemblies,
        procedure: request.procedure,
        coupling: expected_coupling,
        sources,
        boundary_transfers: structural_boundaries,
    };
    if thermomechanical {
        let profile_sources = request
            .realization
            .components
            .get(structural_component_count)
            .map(|component| {
                component
                    .sources
                    .iter()
                    .map(|source| match source {
                        runmat_analysis_core::SourceImplementation::PersistentRegionElementTemperatureMap => {
                            runmat_analysis_core::SourceImplementation::OrdinalRegionTemperatureDistribution
                        }
                        other => *other,
                    })
                    .collect()
            })
            .ok_or(FormulationAdmissionRejectionReason::InvalidRealization)?;
        thermomechanical_exact_realization(ThermomechanicalRealizationInput {
            structural: input,
            profile_sources,
        })
    } else {
        structural_exact_realization(input)
    }
}

fn first_incompatible_axis(
    actual: &ExactFormulationRealization,
    expected: &ExactFormulationRealization,
) -> FormulationAdmissionRejectionReason {
    let axis = if actual.components.len() != expected.components.len()
        || actual
            .components
            .iter()
            .zip(&expected.components)
            .any(|(actual, expected)| actual.spatial_operator != expected.spatial_operator)
    {
        FormulationAxis::SpatialOperator
    } else if actual.temporals != expected.temporals {
        FormulationAxis::Temporal
    } else if actual
        .components
        .iter()
        .zip(&expected.components)
        .any(|(actual, expected)| actual.constitutive != expected.constitutive)
    {
        FormulationAxis::Constitutive
    } else if actual.coupling != expected.coupling {
        FormulationAxis::Coupling
    } else if actual
        .components
        .iter()
        .zip(&expected.components)
        .any(|(actual, expected)| actual.recovery != expected.recovery)
    {
        FormulationAxis::Recovery
    } else if actual
        .components
        .iter()
        .zip(&expected.components)
        .any(|(actual, expected)| actual.topologies != expected.topologies)
    {
        FormulationAxis::Topology
    } else if actual
        .components
        .iter()
        .zip(&expected.components)
        .any(|(actual, expected)| actual.sources != expected.sources)
    {
        FormulationAxis::Source
    } else if actual.boundary_transfers != expected.boundary_transfers {
        FormulationAxis::BoundaryTransfer
    } else {
        return FormulationAdmissionRejectionReason::UnregisteredRealization;
    };
    FormulationAdmissionRejectionReason::IncompatibleAxis(axis)
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        AnalysisProcedure, BoundaryTransferImplementation, CouplingImplementation,
        SourceImplementation, StructuralOperatorImplementation, StructuralRecoveryImplementation,
    };

    use super::*;

    #[test]
    fn qualification_whitelist_matches_the_complete_exact_realization() {
        let canonical = structural_exact_realization(StructuralRealizationInput {
            assemblies: vec![StructuralFormulationRealization::new(
                StructuralOperatorImplementation::Tet4LinearElasticity,
                Some(StructuralRecoveryImplementation::Tet4ConstantStrain),
            )],
            procedure: AnalysisProcedure::Static,
            coupling: CouplingImplementation::Uncoupled,
            sources: vec![SourceImplementation::CanonicalBoundaryProjection],
            boundary_transfers: vec![BoundaryTransferImplementation::CanonicalBoundaryProjection],
        })
        .expect("canonical Tet4 exact realization should be valid");
        let mut different_projection = canonical.clone();
        different_projection.components[0].sources =
            vec![SourceImplementation::UniformNodeDistribution];

        let hypothetical_production = vec![canonical.clone()];
        assert_eq!(
            qualification_from_exact_whitelist(&canonical, &hypothetical_production),
            FormulationMaturity::Production
        );
        assert_eq!(
            qualification_from_exact_whitelist(&different_projection, &hypothetical_production),
            FormulationMaturity::Experimental
        );
        assert!(production_structural_realizations().is_empty());
    }
}
