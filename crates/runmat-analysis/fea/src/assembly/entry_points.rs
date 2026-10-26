//! Owned compatibility and borrowed-mesh entry points for structural assembly.

use super::*;

mod construction_storage;

enum StructuralCouplingInput<'a> {
    Authored {
        solved_nodal_temperature_k: Option<
            runmat_types::physical::Quantity<runmat_types::physical::kind::Temperature, &'a [f64]>,
        >,
        solid_energy: Option<&'a mut solver_solid::SolidEnergyPreparation>,
    },
    ElasticPerturbation,
}

/// Assemble the elastic operator about an already retained preload state.
/// The physical preload stress enters through its retained geometric operator;
/// its thermal profile is not reapplied as a new perturbation load.
pub(crate) fn assemble_structural_perturbation_system(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<SolverMeshArtifact>,
    requirement: StructuralFormulationRequirement,
) -> Result<AssemblySummary, LinearAssemblyError> {
    assemble_with_mesh(
        model,
        excitations,
        solver_mesh.as_ref(),
        Some(requirement),
        None,
        crate::planning::resource::LocalExecutionPlan::serial(),
        StructuralCouplingInput::ElasticPerturbation,
    )
}

pub(crate) fn assemble_structural_system(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<SolverMeshArtifact>,
    requirement: StructuralFormulationRequirement,
) -> Result<AssemblySummary, LinearAssemblyError> {
    assemble_structural_system_borrowed(model, excitations, solver_mesh.as_ref(), requirement)
}

/// Reassemble a time-dependent load without duplicating the immutable canonical mesh.
pub(crate) fn assemble_structural_system_borrowed(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<&SolverMeshArtifact>,
    requirement: StructuralFormulationRequirement,
) -> Result<AssemblySummary, LinearAssemblyError> {
    assemble_with_mesh(
        model,
        excitations,
        solver_mesh,
        Some(requirement),
        None,
        crate::planning::resource::LocalExecutionPlan::serial(),
        StructuralCouplingInput::Authored {
            solved_nodal_temperature_k: None,
            solid_energy: None,
        },
    )
}

pub(crate) fn assemble_structural_system_with_constraint_plan(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<SolverMeshArtifact>,
    requirement: StructuralFormulationRequirement,
    constraint_plan: &CanonicalStructuralConstraintPlan,
) -> Result<AssemblySummary, LinearAssemblyError> {
    assemble_structural_system_with_execution(
        model,
        excitations,
        solver_mesh,
        requirement,
        Some(constraint_plan),
        crate::planning::resource::LocalExecutionPlan::serial(),
        None,
    )
}

pub(crate) fn assemble_structural_system_with_execution(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<SolverMeshArtifact>,
    requirement: StructuralFormulationRequirement,
    constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
    local_execution: crate::planning::resource::LocalExecutionPlan,
    solved_nodal_temperature_k: Option<
        runmat_types::physical::Quantity<runmat_types::physical::kind::Temperature, &[f64]>,
    >,
) -> Result<AssemblySummary, LinearAssemblyError> {
    assemble_with_mesh(
        model,
        excitations,
        solver_mesh.as_ref(),
        Some(requirement),
        constraint_plan,
        local_execution,
        StructuralCouplingInput::Authored {
            solved_nodal_temperature_k,
            solid_energy: None,
        },
    )
}

fn assemble_with_mesh(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<&SolverMeshArtifact>,
    requirement: Option<StructuralFormulationRequirement>,
    constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
    local_execution: crate::planning::resource::LocalExecutionPlan,
    coupling: StructuralCouplingInput<'_>,
) -> Result<AssemblySummary, LinearAssemblyError> {
    let (
        thermo_mechanical_context,
        electro_thermal_context,
        solved_nodal_temperature_k,
        solid_energy,
    ) = match coupling {
        StructuralCouplingInput::Authored {
            solved_nodal_temperature_k,
            solid_energy,
        } => {
            let (thermal, electrical) =
                crate::excitation::realize_coupling_contexts(model, excitations)
                    .map_err(LinearAssemblyError::CouplingRealization)?;
            (
                thermal,
                electrical,
                solved_nodal_temperature_k,
                solid_energy,
            )
        }
        StructuralCouplingInput::ElasticPerturbation => {
            use crate::excitation::RealizedExcitationKind;
            if excitations.iter().any(|excitation| {
                matches!(
                    excitation.kind,
                    RealizedExcitationKind::TemperatureField(_)
                        | RealizedExcitationKind::TerminalVoltage(_)
                        | RealizedExcitationKind::ElectricCurrent(_)
                        | RealizedExcitationKind::ElectricCharge(_)
                        | RealizedExcitationKind::ElectricChargeDensity(_)
                        | RealizedExcitationKind::CurrentDensity(_)
                        | RealizedExcitationKind::MagneticVolumeCurrentDensity(_)
                        | RealizedExcitationKind::MagneticVectorPotential(_)
                )
            }) {
                return Err(LinearAssemblyError::CouplingRealization(
                        "elastic perturbation cannot reapply active temperature or electrical coupling excitations".into(),
                    ));
            }
            (None, None, None, None)
        }
    };
    if solved_nodal_temperature_k.is_some()
        && excitations.iter().any(|excitation| {
            matches!(
                excitation.kind,
                crate::excitation::RealizedExcitationKind::TemperatureField(_)
            )
        })
    {
        return Err(LinearAssemblyError::CouplingRealization(
            "solved thermal state conflicts with an active prescribed temperature field".into(),
        ));
    }
    assemble_linear_system_impl(
        model,
        excitations,
        solver_mesh,
        thermo_mechanical_context.filter(|_| solved_nodal_temperature_k.is_none()),
        electro_thermal_context,
        AssemblyExecution {
            structural_requirement: requirement,
            supplied_constraint_plan: constraint_plan,
            local_execution,
            solved_nodal_temperature_k,
            solid_energy,
        },
    )
}

pub(crate) fn assemble_structural_modal_system(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<&SolverMeshArtifact>,
    solid_energy: &mut solver_solid::SolidEnergyPreparation,
) -> Result<AssemblySummary, LinearAssemblyError> {
    assemble_with_mesh(
        model,
        excitations,
        solver_mesh,
        Some(StructuralFormulationRequirement::ExperimentalSyntheticOperator),
        None,
        crate::planning::resource::LocalExecutionPlan::serial(),
        StructuralCouplingInput::Authored {
            solved_nodal_temperature_k: None,
            solid_energy: Some(solid_energy),
        },
    )
}

pub(crate) struct AssemblyExecution<'a> {
    pub(super) structural_requirement: Option<StructuralFormulationRequirement>,
    pub(super) supplied_constraint_plan: Option<&'a CanonicalStructuralConstraintPlan>,
    pub(super) local_execution: crate::planning::resource::LocalExecutionPlan,
    pub(super) solved_nodal_temperature_k: Option<
        runmat_types::physical::Quantity<runmat_types::physical::kind::Temperature, &'a [f64]>,
    >,
    pub(super) solid_energy: Option<&'a mut solver_solid::SolidEnergyPreparation>,
}

#[cfg(test)]
impl AssemblyExecution<'_> {
    pub(super) fn serial_physical() -> Self {
        Self {
            structural_requirement: Some(StructuralFormulationRequirement::PhysicalDiscretization),
            supplied_constraint_plan: None,
            local_execution: crate::planning::resource::LocalExecutionPlan::serial(),
            solved_nodal_temperature_k: None,
            solid_energy: None,
        }
    }
}

#[cfg(test)]
mod physical_admission_tests {
    use super::*;
    #[test]
    fn absent_formulation_cannot_report_success_from_a_synthetic_multiphysics_rhs() {
        let model =
            crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
        let error = assemble_linear_system_impl(
            &model,
            &RealizedExcitations::default(),
            None,
            None,
            None,
            AssemblyExecution {
                structural_requirement: None,
                supplied_constraint_plan: None,
                local_execution: crate::planning::resource::LocalExecutionPlan::serial(),
                solved_nodal_temperature_k: None,
                solid_energy: None,
            },
        )
        .expect_err("missing physical formulation must reject before assembly");
        assert!(
            matches!(error, LinearAssemblyError::CouplingRealization(message) if message.contains("explicit structural formulation"))
        );
    }
}
