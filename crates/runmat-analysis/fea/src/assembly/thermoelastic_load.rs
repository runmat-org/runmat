//! Physical equivalent nodal loads for element-owned prescribed thermoelastic strain.

use std::collections::BTreeSet;

use runmat_meshing_core::SolverMeshArtifact;
use thiserror::Error;

use crate::{
    properties::{CompiledElementMaterial, CompiledPropertyMap},
    topology::AnalysisProjection,
    FeaThermoMechanicalContext,
};

#[derive(Debug, Error)]
pub(crate) enum ThermoelasticLoadError {
    #[error("thermoelastic element projection is invalid: {0}")]
    Projection(String),
    #[error("thermoelastic solid element {0} has no compiled material owner")]
    MissingOwner(u64),
    #[error("thermoelastic solid element {0} has invalid constitutive coefficients")]
    InvalidConstitutive(u64),
    #[error("thermoelastic prescribed temperature is invalid for region {0}")]
    InvalidTemperature(String),
    #[error("thermoelastic temperature targets duplicate region {0}")]
    DuplicateRegion(String),
    #[error("thermoelastic temperature targets region {0} absent from the solver mesh")]
    UnknownRegion(String),
    #[error("solved thermoelastic temperature field is incomplete or inconsistent with the mesh")]
    InvalidNodalField,
}

pub(super) fn validate_temperature_regions(
    mesh: &SolverMeshArtifact,
    context: &FeaThermoMechanicalContext,
) -> Result<(), ThermoelasticLoadError> {
    let present = mesh
        .topology
        .volume_elements
        .iter()
        .map(|element| element.region_id.source_topology_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut targeted = BTreeSet::new();
    for target in &context.region_temperature_deltas {
        if !targeted.insert(target.region_id.as_str()) {
            return Err(ThermoelasticLoadError::DuplicateRegion(
                target.region_id.clone(),
            ));
        }
        if !present.contains(target.region_id.as_str()) {
            return Err(ThermoelasticLoadError::UnknownRegion(
                target.region_id.clone(),
            ));
        }
        let temperature = context.reference_temperature_k + target.temperature_delta_k;
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(ThermoelasticLoadError::InvalidTemperature(
                target.region_id.clone(),
            ));
        }
    }
    Ok(())
}

pub(super) fn element_temperature_k(
    region_id: &str,
    context: &FeaThermoMechanicalContext,
) -> Result<f64, ThermoelasticLoadError> {
    let delta = context
        .region_temperature_deltas
        .iter()
        .find(|target| target.region_id == region_id)
        .map_or(context.applied_temperature_delta_k, |target| {
            target.temperature_delta_k
        });
    let temperature = context.reference_temperature_k + delta;
    if temperature.is_finite() && temperature > 0.0 {
        Ok(temperature)
    } else {
        Err(ThermoelasticLoadError::InvalidTemperature(region_id.into()))
    }
}

pub(super) fn element_thermal_eigenstrain(
    owner: &CompiledElementMaterial,
    region_id: &str,
    context: &FeaThermoMechanicalContext,
) -> Result<f64, ThermoelasticLoadError> {
    let thermal = owner
        .material
        .require_thermal()
        .map_err(|_| ThermoelasticLoadError::InvalidConstitutive(owner.element_id))?;
    let temperature_k = element_temperature_k(region_id, context)?;
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

/// Assemble ∫ Bᵀ D εₜₕ dV with element-constant εₜₕ = α(T − Tref)[1,1,1,0,0,0].
/// The isotropic hydrostatic stress is E α(T − Tref)/(1−2ν), so each nodal force component is
/// its volume integral against the corresponding physical shape gradient. No material average or
/// ordinal DOF distribution enters this load.
pub(super) fn assemble_prescribed_temperature_load(
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    context: &FeaThermoMechanicalContext,
) -> Result<Vec<f64>, ThermoelasticLoadError> {
    validate_temperature_regions(mesh, context)?;
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
            * element_thermal_eigenstrain(owner, &element.region_id.source_topology_id, context)?
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

/// Integrate the transferred nodal solution as a physical temperature field at each solid
/// quadrature point. No element-average or ordinal region proxy replaces the field interpolation.
pub(crate) fn assemble_solved_temperature_load(
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    nodal_temperature_k: &[f64],
) -> Result<Vec<f64>, ThermoelasticLoadError> {
    let projection = AnalysisProjection::derive(mesh)
        .map_err(|error| ThermoelasticLoadError::Projection(error.to_string()))?;
    if projection.nodes.len() != nodal_temperature_k.len()
        || projection.nodes.len() != mesh.topology.nodes.len()
        || projection
            .nodes
            .iter()
            .zip(&mesh.topology.nodes)
            .any(|(projected, canonical)| projected.node_id != canonical.node_id)
        || nodal_temperature_k
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(ThermoelasticLoadError::InvalidNodalField);
    }
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
        let coefficient = mechanical.youngs_modulus_pa * thermal.expansion_coefficient_per_k
            / (1.0 - 2.0 * mechanical.poisson_ratio);
        if !coefficient.is_finite() {
            return Err(ThermoelasticLoadError::InvalidConstitutive(
                element.element_id,
            ));
        }
        projection
            .visit_element_quadrature(element, |point| {
                let temperature_k = element
                    .node_indices
                    .iter()
                    .zip(point.shape_values())
                    .map(|(&node_index, &shape)| shape * nodal_temperature_k[node_index])
                    .sum::<f64>();
                let stress_pa = coefficient * (temperature_k - thermal.reference_temperature_k);
                for (&node_index, gradient) in element
                    .node_indices
                    .iter()
                    .zip(point.physical_shape_gradients())
                {
                    for axis in 0..3 {
                        load[node_index * 3 + axis] +=
                            stress_pa * gradient[axis] * point.volume_weight_m3;
                    }
                }
            })
            .map_err(|error| ThermoelasticLoadError::Projection(error.to_string()))?;
    }
    if load.iter().any(|value| !value.is_finite()) {
        return Err(ThermoelasticLoadError::InvalidNodalField);
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
        let load = assemble_prescribed_temperature_load(&mesh, &properties, &context()).unwrap();
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
    fn solved_linear_tet4_temperature_integrates_closed_form_thermal_force() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.materials[0]
            .thermal
            .as_mut()
            .unwrap()
            .expansion_coefficient_per_k = 1.0e-5;
        let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        let temperature = mesh
            .topology
            .nodes
            .iter()
            .map(|node| {
                293.15 + 10.0 * node.coordinates_m[0] + 20.0 * node.coordinates_m[1]
                    - 5.0 * node.coordinates_m[2]
            })
            .collect::<Vec<_>>();
        let force = assemble_solved_temperature_load(&mesh, &properties, &temperature).unwrap();
        let mechanical = model.materials[0].require_mechanical().unwrap();
        let average_delta = 0.25 * (10.0 + 20.0 - 5.0);
        let hydrostatic_stress = mechanical.youngs_modulus_pa * 1.0e-5 * average_delta
            / (1.0 - 2.0 * mechanical.poisson_ratio);
        let expected = [
            -hydrostatic_stress / 6.0,
            -hydrostatic_stress / 6.0,
            -hydrostatic_stress / 6.0,
            hydrostatic_stress / 6.0,
            0.0,
            0.0,
            0.0,
            hydrostatic_stress / 6.0,
            0.0,
            0.0,
            0.0,
            hydrostatic_stress / 6.0,
        ];
        for (actual, reference) in force.iter().zip(expected) {
            assert!((actual - reference).abs() <= hydrostatic_stress.abs() * 1.0e-12);
        }
        let mut invalid = temperature;
        invalid[0] = -1.0;
        assert!(matches!(
            assemble_solved_temperature_load(&mesh, &properties, &invalid),
            Err(ThermoelasticLoadError::InvalidNodalField)
        ));
    }

    #[test]
    fn uniform_tet10_load_has_zero_resultant_and_rejects_unknown_region_field() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        let load = assemble_prescribed_temperature_load(&mesh, &properties, &context()).unwrap();
        assert_eq!(load.len(), 30);
        assert!(load.iter().any(|force| force.abs() > 0.0));
        let scale = load.iter().map(|force| force.abs()).sum::<f64>();
        for axis in 0..3 {
            let resultant = load.chunks_exact(3).map(|force| force[axis]).sum::<f64>();
            assert!(resultant.abs() <= scale * 1.0e-12);
        }
        let mut unknown = context();
        unknown
            .region_temperature_deltas
            .push(crate::FeaThermoRegionTemperatureDelta {
                region_id: "missing_region".into(),
                temperature_delta_k: 10.0,
            });
        assert!(matches!(
            assemble_prescribed_temperature_load(&mesh, &properties, &unknown),
            Err(ThermoelasticLoadError::UnknownRegion(_))
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
                assemble_prescribed_temperature_load(&mesh, &properties, &context()).unwrap();
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
        let mut prescribed = context();
        prescribed
            .region_temperature_deltas
            .push(crate::FeaThermoRegionTemperatureDelta {
                region_id: "mid_aluminum".into(),
                temperature_delta_k: 30.0,
            });
        let load = assemble_prescribed_temperature_load(&mesh, &properties, &prescribed).unwrap();
        let recovery =
            crate::assembly::solid_recovery::build(&mesh, &properties, Some(&prescribed), None)
                .unwrap();
        assert!(crate::assembly::solid_recovery::build(
            &mesh,
            &properties,
            Some(&prescribed),
            Some(&vec![300.0; mesh.topology.nodes.len()]),
        )
        .is_err());
        for element in &recovery {
            let expected = if element.region_id == "mid_aluminum" {
                323.15
            } else {
                303.15
            };
            assert_eq!(element.prescribed_temperature_k, Some(expected));
        }
        let projected = AnalysisProjection::derive(&mesh).unwrap();
        for element in &projected.elements {
            let region = element.region_id.source_topology_id.as_str();
            let (youngs, poisson, expansion) = match region {
                "tip_steel" => (200.0e9, 0.3, 1.0e-5),
                "mid_aluminum" => (69.0e9, 0.33, 2.0e-5),
                "polymer_segment" => (3.2e9, 0.37, 4.0e-5),
                _ => panic!("unexpected independent material region {region}"),
            };
            let delta = if region == "mid_aluminum" { 30.0 } else { 10.0 };
            let stress = youngs * expansion * delta / (1.0 - 2.0 * poisson);
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
            assemble_prescribed_temperature_load(&mesh, &reordered, &prescribed).unwrap(),
            load
        );
        prescribed
            .region_temperature_deltas
            .push(crate::FeaThermoRegionTemperatureDelta {
                region_id: "mid_aluminum".into(),
                temperature_delta_k: 40.0,
            });
        assert!(matches!(
            assemble_prescribed_temperature_load(&mesh, &reordered, &prescribed),
            Err(ThermoelasticLoadError::DuplicateRegion(_))
        ));
        prescribed.region_temperature_deltas.pop();
        prescribed.region_temperature_deltas[0].temperature_delta_k = -400.0;
        assert!(matches!(
            assemble_prescribed_temperature_load(&mesh, &reordered, &prescribed),
            Err(ThermoelasticLoadError::InvalidTemperature(_))
        ));
    }

    #[test]
    fn physical_structural_assembly_uses_element_thermal_load_not_proxy_pattern() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
        let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        let context = context();
        let expected = assemble_prescribed_temperature_load(&mesh, &properties, &context).unwrap();
        let summary = assemble_linear_system_impl(
            &model,
            &RealizedExcitations::default(),
            Some(&mesh),
            Some(context),
            None,
            AssemblyExecution {
                solved_nodal_temperature_k: None,
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
