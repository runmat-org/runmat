//! Physical equivalent nodal loads for prescribed uniform thermoelastic strain on solid meshes.

use runmat_meshing_core::SolverMeshArtifact;
use thiserror::Error;

use crate::{
    properties::CompiledPropertyMap, topology::AnalysisProjection, FeaThermoMechanicalContext,
};

#[derive(Debug, Error)]
pub(super) enum ThermoelasticLoadError {
    #[error("thermoelastic element projection is invalid: {0}")]
    Projection(String),
    #[error("thermoelastic solid element {0} has no compiled material owner")]
    MissingOwner(u64),
    #[error("thermoelastic solid element {0} has invalid constitutive coefficients")]
    InvalidConstitutive(u64),
    #[error("region-specific temperature profiles require an exact element temperature map")]
    UnresolvedRegionTemperature,
}

/// Assemble ∫ Bᵀ D εₜₕ dV with εₜₕ = α(T − Tref)[1,1,1,0,0,0].
/// The isotropic hydrostatic stress is E α(T − Tref)/(1−2ν), so each nodal force component is
/// its volume integral against the corresponding physical shape gradient. No material average or
/// ordinal DOF distribution enters this load.
pub(super) fn assemble_uniform_temperature_load(
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    context: &FeaThermoMechanicalContext,
) -> Result<Vec<f64>, ThermoelasticLoadError> {
    if !context.region_temperature_deltas.is_empty() {
        return Err(ThermoelasticLoadError::UnresolvedRegionTemperature);
    }
    let projection = AnalysisProjection::derive(mesh)
        .map_err(|error| ThermoelasticLoadError::Projection(error.to_string()))?;
    let temperature_k = context.reference_temperature_k + context.applied_temperature_delta_k;
    let mut load = vec![0.0; projection.nodes.len() * 3];
    for element in &projection.elements {
        let owner = properties
            .solid_material(element.stable_identity)
            .ok_or(ThermoelasticLoadError::MissingOwner(element.element_id))?;
        let mechanical = owner
            .material
            .require_mechanical()
            .map_err(|_| ThermoelasticLoadError::InvalidConstitutive(element.element_id))?;
        let thermal = owner
            .material
            .require_thermal()
            .map_err(|_| ThermoelasticLoadError::InvalidConstitutive(element.element_id))?;
        let hydrostatic_stress_pa = mechanical.youngs_modulus_pa
            * thermal.expansion_coefficient_per_k
            * (temperature_k - thermal.reference_temperature_k)
            / (1.0 - 2.0 * mechanical.poisson_ratio);
        if !hydrostatic_stress_pa.is_finite() {
            return Err(ThermoelasticLoadError::InvalidConstitutive(
                element.element_id,
            ));
        }
        projection
            .visit_element_quadrature(element, |point| {
                for (&node_index, gradient) in element
                    .node_indices
                    .iter()
                    .zip(point.physical_shape_gradients())
                {
                    for axis in 0..3 {
                        load[node_index * 3 + axis] +=
                            hydrostatic_stress_pa * gradient[axis] * point.volume_weight_m3;
                    }
                }
            })
            .map_err(|error| ThermoelasticLoadError::Projection(error.to_string()))?;
    }
    Ok(load)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assembly::{
            assemble_linear_system_impl, AssemblyExecution, StructuralFormulationRequirement,
        },
        excitation::RealizedExcitations,
        fixtures::{fixture_model, FixtureId},
        properties::compile_structural_property_map,
    };
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

    fn context() -> FeaThermoMechanicalContext {
        FeaThermoMechanicalContext {
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 10.0,
            thermal_expansion_coefficient: 0.0,
            region_temperature_deltas: Vec::new(),
        }
    }

    #[test]
    fn uniform_tet4_thermal_load_matches_closed_form_nodal_forces() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.materials[0]
            .thermal
            .as_mut()
            .unwrap()
            .expansion_coefficient_per_k = 1.0e-5;
        let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        let load = assemble_uniform_temperature_load(&mesh, &properties, &context()).unwrap();
        let mechanical = model.materials[0].require_mechanical().unwrap();
        let stress = mechanical.youngs_modulus_pa * 1.0e-4 / (1.0 - 2.0 * mechanical.poisson_ratio);
        let expected = [
            -stress / 6.0,
            -stress / 6.0,
            -stress / 6.0,
            stress / 6.0,
            0.0,
            0.0,
            0.0,
            stress / 6.0,
            0.0,
            0.0,
            0.0,
            stress / 6.0,
        ];
        for (actual, expected) in load.iter().zip(expected) {
            assert!((actual - expected).abs() <= stress.abs() * 1.0e-12);
        }
        for axis in 0..3 {
            let resultant = load.chunks_exact(3).map(|force| force[axis]).sum::<f64>();
            assert!(resultant.abs() <= stress.abs() * 1.0e-12);
        }
    }

    #[test]
    fn uniform_tet10_load_has_zero_resultant_and_rejects_unresolved_region_field() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        let load = assemble_uniform_temperature_load(&mesh, &properties, &context()).unwrap();
        assert_eq!(load.len(), 30);
        assert!(load.iter().any(|force| force.abs() > 0.0));
        let scale = load.iter().map(|force| force.abs()).sum::<f64>();
        for axis in 0..3 {
            let resultant = load.chunks_exact(3).map(|force| force[axis]).sum::<f64>();
            assert!(resultant.abs() <= scale * 1.0e-12);
        }
        let mut unresolved = context();
        unresolved
            .region_temperature_deltas
            .push(crate::FeaThermoRegionTemperatureDelta {
                region_id: "region".into(),
                temperature_delta_k: 10.0,
            });
        assert!(matches!(
            assemble_uniform_temperature_load(&mesh, &properties, &unresolved),
            Err(ThermoelasticLoadError::UnresolvedRegionTemperature)
        ));
    }

    #[test]
    fn physical_structural_assembly_uses_element_thermal_load_not_proxy_pattern() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
        let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        let context = context();
        let expected = assemble_uniform_temperature_load(&mesh, &properties, &context).unwrap();
        let summary = assemble_linear_system_impl(
            &model,
            &RealizedExcitations::default(),
            Some(&mesh),
            Some(context),
            None,
            AssemblyExecution {
                structural_requirement: Some(
                    StructuralFormulationRequirement::PhysicalDiscretization,
                ),
                supplied_constraint_plan: None,
                local_execution: crate::planning::resource::LocalExecutionPlan::serial(),
            },
        )
        .expect("physical thermoelastic assembly");
        assert_eq!(
            &summary.physical_load[..expected.len()],
            expected.as_slice()
        );
        assert!(summary.physical_load.iter().any(|value| value.abs() > 0.0));
        assert!(summary.constrained_dof_count > 0);
        assert!(summary
            .operator
            .constrained_dofs()
            .iter()
            .enumerate()
            .any(|(row, constrained)| *constrained && summary.physical_load[row] != 0.0));
        for (row, is_constrained) in summary.operator.constrained_dofs().iter().enumerate() {
            if *is_constrained {
                assert_eq!(summary.operator.load()[row], 0.0);
            }
        }
    }
}
