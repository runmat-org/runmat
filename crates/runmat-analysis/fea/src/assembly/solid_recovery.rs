//! Exact solid recovery geometry and element-owned thermal eigenstrain construction.

use std::collections::BTreeMap;

use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

use crate::{properties::CompiledPropertyMap, FeaThermoMechanicalContext};

use super::{
    elements::solid::SolidMaterial, solver_solid::resolve_element_material,
    solver_solid::SolverSolidAssemblyError, thermoelastic_load, SolidRecoveryElementSummary,
    SolidRecoveryGeometry,
};

pub(super) fn build(
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    thermal_context: Option<&FeaThermoMechanicalContext>,
) -> Result<Vec<SolidRecoveryElementSummary>, SolverSolidAssemblyError> {
    let node_indices = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();

    mesh.topology
        .volume_elements
        .iter()
        .map(|element| {
            let material = resolve_element_material(element, properties)?;
            let prescribed_temperature_k = thermal_context
                .map(|context| {
                    thermoelastic_load::element_temperature_k(
                        &element.region_id.source_topology_id,
                        context,
                    )
                    .map_err(|error| {
                        SolverSolidAssemblyError::InvalidMaterialCoverage(error.to_string())
                    })
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
                        &element.region_id.source_topology_id,
                        context,
                    )
                    .map_err(|error| {
                        SolverSolidAssemblyError::InvalidMaterialCoverage(error.to_string())
                    })
                })
                .transpose()?;
            let geometry = match element.order {
                ElementOrder::Tet4 => {
                    let (node_indices, coordinates_m) = recovery_geometry::<4>(
                        element.element_id,
                        &element.node_ids,
                        &node_indices,
                        mesh,
                    )?;
                    SolidRecoveryGeometry::Tetrahedron4 {
                        node_indices,
                        coordinates_m,
                    }
                }
                ElementOrder::Tet10 => {
                    let (node_indices, coordinates_m) = recovery_geometry::<10>(
                        element.element_id,
                        &element.node_ids,
                        &node_indices,
                        mesh,
                    )?;
                    SolidRecoveryGeometry::Tetrahedron10 {
                        node_indices,
                        coordinates_m,
                    }
                }
            };
            Ok(SolidRecoveryElementSummary {
                element_id: element.element_id.to_string(),
                region_id: element.region_id.source_topology_id.clone(),
                material: SolidMaterial {
                    youngs_modulus_pa: material.youngs_modulus_pa,
                    poisson_ratio: material.poisson_ratio,
                    density_kg_per_m3: material.density_kg_per_m3,
                },
                thermal_eigenstrain,
                prescribed_temperature_k,
                geometry,
            })
        })
        .collect()
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
        coordinates_m[local] = mesh.topology.nodes[node_index].coordinates_m;
    }
    Ok((indices, coordinates_m))
}
