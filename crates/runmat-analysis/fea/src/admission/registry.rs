use runmat_analysis_core::{
    is_exposed_analysis_identity, CouplingImplementation, ExactFormulationRealization,
    FormulationAdmissionRejectionReason, FormulationAxis, PhysicsFamily, RecoveryImplementation,
    SolutionSpace, SpatialOperatorImplementation, StructuralFormulationRealization,
};

pub(super) fn require_available(
    family: PhysicsFamily,
    procedure: runmat_analysis_core::AnalysisProcedure,
    solution_space: SolutionSpace,
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
    if !has_registered_formulation(family, procedure) {
        return Err(FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure);
    }
    Ok(())
}

use super::{
    realizations::registered_formulations,
    structural::{
        structural_exact_realization, thermomechanical_exact_realization,
        StructuralRealizationInput, ThermomechanicalRealizationInput,
    },
    FormulationAdmissionRequest,
};

pub(super) fn admit(
    request: &FormulationAdmissionRequest,
) -> Result<(), FormulationAdmissionRejectionReason> {
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

    if request.family == PhysicsFamily::Structural
        || request.family
            == PhysicsFamily::Coupled(runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical)
    {
        let expected =
            validate_structural_identity(request, request.family != PhysicsFamily::Structural)?;
        if request.realization != expected {
            return Err(first_incompatible_axis(&request.realization, &expected));
        }
    } else {
        let registered = registered_formulations(request.family, request.procedure);
        if registered.is_empty() {
            return Err(FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure);
        }
        if !registered
            .iter()
            .any(|entry| entry.realization() == &request.realization)
        {
            let reason = registered
                .first()
                .map(|expected| {
                    first_incompatible_axis(&request.realization, expected.realization())
                })
                .unwrap_or(FormulationAdmissionRejectionReason::UnregisteredRealization);
            return Err(reason);
        }
    }
    Ok(())
}

fn has_registered_formulation(
    family: PhysicsFamily,
    procedure: runmat_analysis_core::AnalysisProcedure,
) -> bool {
    if family
        == PhysicsFamily::Coupled(runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical)
        && procedure == runmat_analysis_core::AnalysisProcedure::Nonlinear
    {
        return false;
    }
    if matches!(
        family,
        PhysicsFamily::Structural
            | PhysicsFamily::Coupled(runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical)
    ) {
        is_exposed_analysis_identity(family, procedure, SolutionSpace::Direct)
            || is_exposed_analysis_identity(family, procedure, SolutionSpace::ModalReduced)
    } else {
        !registered_formulations(family, procedure).is_empty()
    }
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
        match request.realization.coupling {
            CouplingImplementation::OneWaySolvedThermoelasticState => {
                CouplingImplementation::OneWaySolvedThermoelasticState
            }
            _ => CouplingImplementation::OnePassPrescribedThermomechanicalProfile,
        }
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
        kinematics: requested_kinematics(request),
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

/// The kinematics a structural request realizes is carried by its spatial operator identity.
fn requested_kinematics(
    request: &FormulationAdmissionRequest,
) -> runmat_analysis_core::StructuralKinematics {
    let total_lagrangian = request.realization.components.iter().any(|component| {
        matches!(
            component.spatial_operator,
            SpatialOperatorImplementation::Structural(
                runmat_analysis_core::StructuralOperatorImplementation::Tet4TotalLagrangianContinuum
                    | runmat_analysis_core::StructuralOperatorImplementation::Tet10TotalLagrangianContinuum
            )
        )
    });
    if total_lagrangian {
        runmat_analysis_core::StructuralKinematics::LargeDeformation
    } else {
        runmat_analysis_core::StructuralKinematics::SmallStrain
    }
}
