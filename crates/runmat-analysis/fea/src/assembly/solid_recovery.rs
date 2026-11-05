//! Exact solid recovery geometry and element-owned thermal eigenstrain construction.

use std::collections::BTreeMap;

use crate::algebra::coercivity::NumericalCapacityError;
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};
use runmat_simulation_usage::OperationCount;

use crate::topology::quadrature::{volume_shape, TETRAHEDRON_FOUR_POINT_RULE};
use crate::{properties::CompiledPropertyMap, FeaThermoMechanicalContext};

use super::{
    elements::solid::SolidMaterial, solver_solid::resolve_element_material,
    solver_solid::SolverSolidAssemblyError, thermoelastic_load, SolidRecoveryElementSummary,
    SolidRecoveryGeometry, SolvedThermalRecoveryQuadrature,
};

pub(crate) fn build(
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    thermal_context: Option<&FeaThermoMechanicalContext>,
    solved_nodal_temperature_k: Option<&crate::execution::SolvedTemperatureField>,
) -> Result<Vec<SolidRecoveryElementSummary>, SolverSolidAssemblyError> {
    validate_inputs(mesh, thermal_context, solved_nodal_temperature_k)?;
    let node_indices = node_indices(mesh);
    let elements = &mesh.topology.volume_elements;
    // Independent per-element summaries on the admitted pool, collected in element order.
    crate::algebra::element_assembly::ordered_map_range(elements.len(), |index| {
        build_element(
            index,
            mesh,
            properties,
            thermal_context,
            solved_nodal_temperature_k,
            &node_indices,
        )
    })
    .into_iter()
    .collect()
}

pub(crate) async fn build_with_owner<O, F, A>(
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    thermal_context: Option<&FeaThermoMechanicalContext>,
    solved_nodal_temperature_k: Option<&crate::execution::SolvedTemperatureField>,
    before: &mut O,
) -> Result<Vec<SolidRecoveryElementSummary>, NumericalCapacityError<SolverSolidAssemblyError, A>>
where
    O: FnMut(usize, OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
{
    validate_inputs(mesh, thermal_context, solved_nodal_temperature_k)?;
    let node_indices = node_indices(mesh);
    let mut summaries = Vec::with_capacity(mesh.topology.volume_elements.len());
    for (index, element) in mesh.topology.volume_elements.iter().enumerate() {
        let work = if thermal_context.is_some() {
            1 + 1 + 2
        } else if solved_nodal_temperature_k.is_some() && element.order == ElementOrder::Tet10 {
            4 * 116
        } else {
            0
        };
        if work > 0 {
            before(index, OperationCount::from_operations(work))
                .await
                .map_err(NumericalCapacityError::Admission)?;
        }
        summaries.push(build_element(
            index,
            mesh,
            properties,
            thermal_context,
            solved_nodal_temperature_k,
            &node_indices,
        )?);
    }
    Ok(summaries)
}

fn validate_inputs(
    mesh: &SolverMeshArtifact,
    thermal_context: Option<&FeaThermoMechanicalContext>,
    solved_nodal_temperature_k: Option<&crate::execution::SolvedTemperatureField>,
) -> Result<(), SolverSolidAssemblyError> {
    if thermal_context.is_some() && solved_nodal_temperature_k.is_some() {
        return Err(SolverSolidAssemblyError::InvalidMaterialCoverage(
            "prescribed and solved thermal fields cannot both own solid recovery".into(),
        ));
    }
    if let Some(temperature) = solved_nodal_temperature_k {
        if temperature.node_count() != mesh.topology.nodes.len()
            || !temperature.all_positive_finite()
        {
            return Err(SolverSolidAssemblyError::InvalidMaterialCoverage(
                "solved thermal field does not cover every solver node".into(),
            ));
        }
    }
    Ok(())
}

fn node_indices(mesh: &SolverMeshArtifact) -> BTreeMap<u64, usize> {
    mesh.topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect()
}

fn thermal_error(error: thermoelastic_load::ThermoelasticLoadError) -> SolverSolidAssemblyError {
    match error {
        thermoelastic_load::ThermoelasticLoadError::Accounting(cause) => {
            SolverSolidAssemblyError::Accounting(cause)
        }
        thermoelastic_load::ThermoelasticLoadError::Cancelled => {
            SolverSolidAssemblyError::Cancelled
        }
        other => SolverSolidAssemblyError::InvalidMaterialCoverage(other.to_string()),
    }
}

fn build_element(
    index: usize,
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    thermal_context: Option<&FeaThermoMechanicalContext>,
    solved_nodal_temperature_k: Option<&crate::execution::SolvedTemperatureField>,
    node_indices: &BTreeMap<u64, usize>,
) -> Result<SolidRecoveryElementSummary, SolverSolidAssemblyError> {
    let element = &mesh.topology.volume_elements[index];
    let material = resolve_element_material(element, properties)?;
    let prescribed_temperature_k = thermal_context
        .map(|context| {
            thermoelastic_load::element_temperature_k(
                &element.region_id,
                &element.provenance,
                context,
            )
            .map_err(thermal_error)
        })
        .transpose()?;
    let thermal_eigenstrain = thermal_context
        .map(|context| {
            let owner = properties
                .solid_material(element.stable_identity)
                .ok_or_else(|| {
                    SolverSolidAssemblyError::InvalidMaterialCoverage(format!(
                        "thermoelastic element {} has no compiled material owner",
                        element.element_id
                    ))
                })?;
            thermoelastic_load::element_thermal_eigenstrain(
                owner,
                &element.region_id,
                &element.provenance,
                context,
            )
            .map_err(thermal_error)
        })
        .transpose()?;
    let geometry = match element.order {
        ElementOrder::Tet4 => {
            let (node_indices, coordinates_m) =
                recovery_geometry::<4>(element.element_id, &element.node_ids, node_indices, mesh)?;
            SolidRecoveryGeometry::Tetrahedron4 {
                node_indices,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates_m),
            }
        }
        ElementOrder::Tet10 => {
            let (node_indices, coordinates_m) =
                recovery_geometry::<10>(element.element_id, &element.node_ids, node_indices, mesh)?;
            SolidRecoveryGeometry::Tetrahedron10 {
                node_indices,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates_m),
            }
        }
    };
    let solved_thermal_quadrature = solved_nodal_temperature_k
        .map(|temperatures| {
            solved_thermal_quadrature(index, element, node_indices, properties, temperatures)
        })
        .transpose()?;
    Ok(SolidRecoveryElementSummary {
        element_id: element.element_id.to_string(),
        region_id: element.region_id.source_topology_id.clone(),
        material: SolidMaterial {
            youngs_modulus_pa: material.youngs_modulus_pa,
            poisson_ratio: runmat_types::physical::PoissonRatio::new(
                material.poisson_ratio.canonical_value(),
            ),
            density_kg_per_m3: runmat_types::physical::MassDensity::new(
                material.density_kg_per_m3.canonical_value(),
            ),
        },
        thermal_eigenstrain,
        prescribed_temperature_k: prescribed_temperature_k
            .map(runmat_types::physical::Temperature::new),
        solved_thermal_quadrature,
        geometry,
    })
}

fn solved_thermal_quadrature(
    element_index: usize,
    element: &runmat_meshing_core::SolverVolumeElement,
    node_indices: &BTreeMap<u64, usize>,
    properties: &CompiledPropertyMap,
    temperature: &crate::execution::SolvedTemperatureField,
) -> Result<SolvedThermalRecoveryQuadrature, SolverSolidAssemblyError> {
    let owner = properties
        .solid_material(element.stable_identity)
        .ok_or_else(|| {
            SolverSolidAssemblyError::InvalidMaterialCoverage(format!(
                "thermoelastic element {} has no compiled material owner",
                element.element_id
            ))
        })?;
    let thermal = owner
        .material
        .require_thermal()
        .map_err(|error| SolverSolidAssemblyError::InvalidMaterialCoverage(error.to_string()))?;
    let mut temperature_k = [0.0; 4];
    let mut eigenstrain = [0.0; 4];
    for (point, barycentric) in TETRAHEDRON_FOUR_POINT_RULE.iter().copied().enumerate() {
        let (shape, _) = volume_shape(element.order, barycentric)
            .map_err(SolverSolidAssemblyError::Accounting)?;
        let mut value = 0.0;
        for (local, node_id) in element.node_ids.iter().copied().enumerate() {
            let index =
                node_indices
                    .get(&node_id)
                    .ok_or(SolverSolidAssemblyError::UnknownElementNode {
                        element_id: element.element_id,
                        node_id,
                    })?;
            value += shape[local] * temperature.at(element_index, *index);
        }
        let strain = thermal.expansion_coefficient_per_k.canonical_value()
            * (value - thermal.reference_temperature_k.canonical_value());
        if !value.is_finite() || value <= 0.0 || !strain.is_finite() {
            return Err(SolverSolidAssemblyError::InvalidMaterialCoverage(format!(
                "thermoelastic element {} has nonphysical solved temperature",
                element.element_id
            )));
        }
        temperature_k[point] = value;
        eigenstrain[point] = strain;
    }
    Ok(SolvedThermalRecoveryQuadrature {
        nodal_temperature_k: runmat_types::physical::Quantity::new(
            element
                .node_ids
                .iter()
                .map(|node_id| temperature.at(element_index, node_indices[node_id]))
                .collect::<Vec<_>>(),
        ),
        expansion_coefficient_per_k: thermal.expansion_coefficient_per_k,
        reference_temperature_k: runmat_types::physical::Temperature::new(
            thermal.reference_temperature_k.canonical_value(),
        ),
        temperature_k: runmat_types::physical::Quantity::new(temperature_k),
        eigenstrain,
    })
}

fn recovery_geometry<const N: usize>(
    element_id: u64,
    node_ids: &[u64],
    node_indices: &BTreeMap<u64, usize>,
    mesh: &SolverMeshArtifact,
) -> Result<([usize; N], [[f64; 3]; N]), SolverSolidAssemblyError> {
    if node_ids.len() != N {
        return Err(SolverSolidAssemblyError::InvalidElementNodeCount {
            element_id,
            expected: N,
            actual: node_ids.len(),
        });
    }
    let mut indices = [0; N];
    let mut coordinates_m = [[0.0; 3]; N];
    for (local, node_id) in node_ids.iter().copied().enumerate() {
        let node_index =
            *node_indices
                .get(&node_id)
                .ok_or(SolverSolidAssemblyError::UnknownElementNode {
                    element_id,
                    node_id,
                })?;
        indices[local] = node_index;
        coordinates_m[local] = mesh.topology.nodes[node_index]
            .coordinates_m
            .canonical_value();
    }
    Ok((indices, coordinates_m))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        fixtures::{fixture_model, FixtureId},
        properties::compile_structural_property_map,
    };
    use runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh;

    #[test]
    fn solved_tet10_summary_owner_quotes_fixed_shape_and_preserves_denial() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        let temperature =
            crate::execution::SolvedTemperatureField::nodal(vec![
                293.15;
                mesh.topology.nodes.len()
            ]);
        let native = build(&mesh, &properties, None, Some(&temperature)).unwrap();
        let quotes = std::cell::RefCell::new(Vec::new());
        let mut grant = |element, count| {
            quotes.borrow_mut().push((element, count));
            std::future::ready(Ok::<(), &'static str>(()))
        };
        let awaited = futures::executor::block_on(build_with_owner(
            &mesh,
            &properties,
            None,
            Some(&temperature),
            &mut grant,
        ));
        assert!(matches!(awaited, Ok(ref summaries) if summaries == &native));
        assert_eq!(
            quotes.into_inner(),
            vec![(0, OperationCount::from_operations(4 * 116))]
        );
        let mut deny = |_, _| std::future::ready(Err::<(), _>("denied"));
        let denied = futures::executor::block_on(build_with_owner(
            &mesh,
            &properties,
            None,
            Some(&temperature),
            &mut deny,
        ));
        assert!(matches!(
            denied,
            Err(NumericalCapacityError::Admission("denied"))
        ));
    }
}
