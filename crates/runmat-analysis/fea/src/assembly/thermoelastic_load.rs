//! Physical equivalent nodal loads for prescribed uniform thermoelastic strain on solid meshes.

use runmat_meshing_core::SolverMeshArtifact;
use thiserror::Error;

use crate::{
    properties::{CompiledElementMaterial, CompiledPropertyMap},
    topology::AnalysisProjection,
    FeaThermoMechanicalContext,
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

pub(super) fn element_thermal_eigenstrain(
    owner: &CompiledElementMaterial,
    context: &FeaThermoMechanicalContext,
) -> Result<f64, ThermoelasticLoadError> {
    let thermal = owner
        .material
        .require_thermal()
        .map_err(|_| ThermoelasticLoadError::InvalidConstitutive(owner.element_id))?;
    let temperature_k = context.reference_temperature_k + context.applied_temperature_delta_k;
    let strain =
        thermal.expansion_coefficient_per_k * (temperature_k - thermal.reference_temperature_k);
    if strain.is_finite() {
        Ok(strain)
    } else {
        Err(ThermoelasticLoadError::InvalidConstitutive(
            owner.element_id,
        ))
    }
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
    let mut load = vec![0.0; projection.nodes.len() * 3];
    for element in &projection.elements {
        let owner = properties
            .solid_material(element.stable_identity)
            .ok_or(ThermoelasticLoadError::MissingOwner(element.element_id))?;
        let mechanical = owner
            .material
            .require_mechanical()
            .map_err(|_| ThermoelasticLoadError::InvalidConstitutive(element.element_id))?;
        let hydrostatic_stress_pa = mechanical.youngs_modulus_pa
            * element_thermal_eigenstrain(owner, context)?
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
            assemble_linear_system_impl, solver_solid::assemble_solver_solid_operators,
            AssemblyExecution, StructuralFormulationRequirement,
        },
        excitation::RealizedExcitations,
        fixtures::{fixture_model, FixtureId},
        properties::compile_structural_property_map,
    };
    use runmat_meshing_core::{
        fixtures::{canonical_tetrahedron_solver_mesh, three_region_tetrahedron_solver_mesh},
        ElementOrder,
    };

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
    fn tet4_and_tet10_free_expansion_satisfy_independent_mechanical_equilibrium() {
        // For a homogeneous traction-free solid, u = alpha * delta_T * x gives
        // epsilon(u) = epsilon_thermal at every integration point, hence K u = f_thermal.
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mesh = canonical_tetrahedron_solver_mesh(order);
            let mut model = fixture_model(FixtureId::CantileverLinearStatic);
            model.materials[0]
                .thermal
                .as_mut()
                .unwrap()
                .expansion_coefficient_per_k = 1.0e-5;
            let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
            let projected = AnalysisProjection::derive(&mesh).unwrap();
            let coefficient = model.materials[0]
                .require_thermal()
                .unwrap()
                .expansion_coefficient_per_k
                * context().applied_temperature_delta_k;
            let displacement = projected
                .nodes
                .iter()
                .flat_map(|node| {
                    node.coordinates_m
                        .map(|coordinate| coefficient * coordinate)
                })
                .collect::<Vec<_>>();
            let thermal_load =
                assemble_uniform_temperature_load(&mesh, &properties, &context()).unwrap();
            let stiffness = assemble_solver_solid_operators(&mesh, &properties)
                .unwrap()
                .stiffness;
            let dense = stiffness.to_dense();
            let n = stiffness.rows();
            assert_eq!(n, displacement.len());
            let scale = thermal_load
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            assert!(scale > 0.0);
            for row in 0..n {
                let internal_force = dense[row * n..(row + 1) * n]
                    .iter()
                    .zip(&displacement)
                    .map(|(stiffness, value)| stiffness * value)
                    .sum::<f64>();
                assert!(
                    (internal_force - thermal_load[row]).abs() <= scale * 1.0e-10,
                    "{order:?} row {row}: internal={internal_force}, thermal={}",
                    thermal_load[row]
                );
            }
        }
    }

    #[test]
    fn heterogeneous_regions_use_their_exact_constitutive_owner_independent_of_authored_order() {
        let mesh = three_region_tetrahedron_solver_mesh();
        let mut model = fixture_model(FixtureId::MultiMaterialAssembly);
        model
            .material_assignments
            .iter_mut()
            .find(|assignment| assignment.selected_region_id() == Some("mid_aluminum"))
            .unwrap()
            .assigned_material_id = "mat_aluminum".into();
        for (material, coefficient) in model.materials.iter_mut().zip([1.0e-5, 2.0e-5, 4.0e-5]) {
            material
                .thermal
                .as_mut()
                .unwrap()
                .expansion_coefficient_per_k = coefficient;
        }
        let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        let load = assemble_uniform_temperature_load(&mesh, &properties, &context()).unwrap();
        let projected = AnalysisProjection::derive(&mesh).unwrap();
        for element in &projected.elements {
            let region = element.region_id.source_topology_id.as_str();
            let (youngs, poisson, expansion) = match region {
                "tip_steel" => (200.0e9, 0.3, 1.0e-5),
                "mid_aluminum" => (69.0e9, 0.33, 2.0e-5),
                "polymer_segment" => (3.2e9, 0.37, 4.0e-5),
                _ => panic!("unexpected independent material region {region}"),
            };
            let stress = youngs * expansion * 10.0 / (1.0 - 2.0 * poisson);
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
            for (&node, expected_force) in element.node_indices.iter().zip(expected.chunks_exact(3))
            {
                for axis in 0..3 {
                    assert!(
                        (load[node * 3 + axis] - expected_force[axis]).abs()
                            <= stress.abs() * 1.0e-12,
                        "region {region}, node {node}, axis {axis}"
                    );
                }
            }
        }
        model.materials.reverse();
        model.material_assignments.reverse();
        let reordered = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        assert_eq!(
            assemble_uniform_temperature_load(&mesh, &reordered, &context()).unwrap(),
            load
        );
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
