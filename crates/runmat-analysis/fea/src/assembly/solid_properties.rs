//! Capability-checked compiled solid ownership for structural assembly.

use runmat_analysis_core::AnalysisModel;
use runmat_meshing_core::SolverMeshArtifact;

use crate::properties::{compile_structural_property_map, CompiledPropertyMap};

use super::{
    solver_solid::SolverSolidAssemblyError, LinearAssemblyError, StructuralFormulationRequirement,
};

pub(super) fn compile_for_assembly(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    structural_requirement: Option<StructuralFormulationRequirement>,
    thermomechanical: bool,
) -> Result<Option<CompiledPropertyMap>, LinearAssemblyError> {
    let properties = mesh
        .filter(|_| structural_requirement.is_some())
        .map(|mesh| compile_structural_property_map(model, Some(mesh)))
        .transpose()
        .map_err(invalid_material_coverage)?;
    if thermomechanical {
        if let Some(properties) = &properties {
            properties
                .validate_thermal_solid_capabilities()
                .map_err(invalid_material_coverage)?;
        }
    }
    Ok(properties)
}

fn invalid_material_coverage(
    error: crate::properties::PropertyCompilationError,
) -> LinearAssemblyError {
    LinearAssemblyError::SolidOperator(SolverSolidAssemblyError::InvalidMaterialCoverage(
        error.to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assembly::{assemble_linear_system_impl, AssemblyExecution},
        excitation::RealizedExcitations,
        fixtures::{fixture_model, FixtureId},
        FeaThermoMechanicalContext,
    };

    #[test]
    fn direct_coupled_solid_assembly_rejects_missing_thermal_owner_before_operator_work() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
        model.materials[0].thermal = None;
        let context = FeaThermoMechanicalContext {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(10.0),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(1.0e-5),
            region_temperature_deltas: Vec::new(),
        };
        let error = assemble_linear_system_impl(
            &model,
            &RealizedExcitations::default(),
            Some(&mesh),
            Some(context),
            None,
            AssemblyExecution::serial_physical(),
        )
        .expect_err("uncapable coupled owner must fail before operator work");
        assert!(matches!(
            error,
            LinearAssemblyError::SolidOperator(
                SolverSolidAssemblyError::InvalidMaterialCoverage(message)
            ) if message.contains("thermal solid") && message.contains("Thermal")
        ));
    }
}
