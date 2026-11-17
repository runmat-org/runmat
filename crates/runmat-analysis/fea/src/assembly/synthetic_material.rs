//! Material coefficients read by the synthetic structural operator and its thermomechanical
//! scaling. Physical element operators take coefficients from compiled element owners. The
//! synthetic path averages the coefficients that the model's materials declare and fails when
//! no material declares the one it needs.

use runmat_analysis_core::{AnalysisModel, MaterialCapability, MaterialMechanicalModel};

use super::{mean, StructuralAssemblyError};

pub(super) fn mean_youngs_modulus_pa(
    model: &AnalysisModel,
) -> Result<f64, StructuralAssemblyError> {
    mean_mechanical(model, |mechanical| {
        mechanical.youngs_modulus_pa.canonical_value()
    })
}

pub(super) fn mean_poisson_ratio(model: &AnalysisModel) -> Result<f64, StructuralAssemblyError> {
    mean_mechanical(model, |mechanical| {
        mechanical.poisson_ratio.canonical_value()
    })
}

fn mean_mechanical(
    model: &AnalysisModel,
    value: impl Fn(&MaterialMechanicalModel) -> f64,
) -> Result<f64, StructuralAssemblyError> {
    mean(
        model
            .materials
            .iter()
            .filter_map(|material| material.require_mechanical().ok().map(&value)),
    )
    .ok_or_else(|| StructuralAssemblyError::SyntheticMaterial {
        capability: MaterialCapability::IsotropicLinearElasticity,
        undeclared_material_id: model
            .materials
            .first()
            .map(|material| material.material_id.clone()),
    })
}

#[cfg(test)]
mod tests {
    use super::super::{
        assemble_structural_system, LinearAssemblyError, StructuralFormulationRequirement,
    };
    use super::*;
    use crate::fixtures::{fixture_model, FixtureId};

    fn assemble_mesh_free(model: &AnalysisModel) -> Result<(), LinearAssemblyError> {
        let excitations = crate::excitation::realize_step_excitations(
            model,
            &model.sequence.steps[0],
            crate::excitation::ExcitationSamplePoint::Static,
        )
        .expect("fixture excitations realize at the static sample");
        assemble_structural_system(
            model,
            &excitations,
            None,
            StructuralFormulationRequirement::SyntheticOperator,
        )
        .map(|_| ())
    }

    #[test]
    fn mesh_free_operator_names_the_material_without_youngs_modulus() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        assemble_mesh_free(&model).expect("declared elasticity assembles");
        for material in &mut model.materials {
            material.elasticity = None;
        }
        let material_id = model.materials[0].material_id.clone();

        let error = assemble_mesh_free(&model).expect_err("missing modulus must not default");

        assert_eq!(
            error,
            LinearAssemblyError::Structural(StructuralAssemblyError::SyntheticMaterial {
                capability: MaterialCapability::IsotropicLinearElasticity,
                undeclared_material_id: Some(material_id.clone()),
            })
        );
        assert_eq!(
            error.to_string(),
            format!(
                "material `{material_id}` does not declare required `IsotropicLinearElasticity` capability; no model material declares it for the mesh-free structural operator"
            )
        );
    }

    #[test]
    fn mesh_free_operator_without_materials_reports_the_missing_capability() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.materials.clear();
        model.material_assignments.clear();

        let error = assemble_mesh_free(&model).expect_err("no material must not default");

        assert_eq!(
            error,
            LinearAssemblyError::Structural(StructuralAssemblyError::SyntheticMaterial {
                capability: MaterialCapability::IsotropicLinearElasticity,
                undeclared_material_id: None,
            })
        );
    }

    #[test]
    fn mesh_free_operator_ignores_missing_density() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        for material in &mut model.materials {
            material.mass_density_kg_per_m3 = None;
        }

        assemble_mesh_free(&model).expect("the mesh-free stiffness operator reads no density");
    }
}
