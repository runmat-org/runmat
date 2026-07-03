use runmat_geometry_core::{GeometryAsset, MeshKind};
use std::collections::BTreeMap;

mod metrics;
mod types;

use metrics::{
    coordinate_active_dimension_count, coordinate_characteristic_length_m, mesh_coordinate_span_m,
    mesh_element_geometry_metrics,
};
pub use types::*;

pub fn prepare_geometry_for_analysis(
    geometry: &GeometryAsset,
    options: MeshingOptions,
) -> Result<MeshingPrepResult, String> {
    if geometry.meshes.is_empty() {
        return Err("geometry has no meshes to prepare".to_string());
    }
    if options.target_element_budget == 0 {
        return Err("target_element_budget must be greater than zero".to_string());
    }

    let mut source_meshes = geometry.meshes.clone();
    source_meshes.sort_by(|a, b| a.mesh_id.cmp(&b.mesh_id));
    let per_mesh_budget =
        (options.target_element_budget / source_meshes.len().max(1)).max(1) as u64;

    let mut prepared_meshes = Vec::with_capacity(source_meshes.len());
    for mesh in source_meshes {
        let profile_scale = match options.profile {
            MeshingProfile::SurfaceOnly => 1.0,
            MeshingProfile::AnalysisReady => {
                if mesh.kind == MeshKind::Surface {
                    1.4
                } else {
                    1.1
                }
            }
            MeshingProfile::AdaptiveRefine => {
                if mesh.kind == MeshKind::Surface {
                    1.8
                } else {
                    1.35
                }
            }
        };
        let proposed = ((mesh.element_count as f64) * profile_scale).round() as u64;
        let min_refined_elements = match options.profile {
            MeshingProfile::AdaptiveRefine => mesh.element_count.max(64),
            MeshingProfile::AnalysisReady | MeshingProfile::SurfaceOnly => 1,
        };
        let element_count = proposed
            .max(min_refined_elements)
            .min(per_mesh_budget.max(mesh.element_count));
        let node_count = (mesh.vertex_count.max(3)).max(element_count / 2 + 2);
        let connectivity_class = match mesh.kind {
            MeshKind::Surface => {
                if element_count > 20_000 {
                    MeshConnectivityClass::SparseBand
                } else {
                    MeshConnectivityClass::SurfacePatch
                }
            }
            MeshKind::Volume => MeshConnectivityClass::VolumeCore,
        };
        let element_family_hint = match mesh.kind {
            MeshKind::Surface => {
                if element_count % 2 == 0 {
                    ElementFamilyHint::Triangle
                } else {
                    ElementFamilyHint::Quad
                }
            }
            MeshKind::Volume => {
                if element_count % 2 == 0 {
                    ElementFamilyHint::Tetrahedron
                } else {
                    ElementFamilyHint::Hex
                }
            }
        };
        let coordinate_span_m = mesh_coordinate_span_m(geometry, &mesh.mesh_id);
        let coordinate_active_dimension_count =
            coordinate_active_dimension_count(coordinate_span_m);
        let coordinate_characteristic_length_m = coordinate_characteristic_length_m(
            coordinate_span_m,
            coordinate_active_dimension_count,
            node_count,
        );
        let element_geometry = mesh_element_geometry_metrics(geometry, &mesh.mesh_id);
        let region_span_hint = (geometry.regions.len().max(1) as u32)
            .clamp(1, 64)
            .saturating_sub((prepared_meshes.len() as u32) % 2);
        prepared_meshes.push(PreparedMeshDescriptor {
            prepared_mesh_id: format!("prep_{}_{}", geometry.revision, mesh.mesh_id),
            source_mesh_id: mesh.mesh_id,
            kind: mesh.kind,
            node_count,
            element_count,
            connectivity_class,
            element_family_hint,
            region_span_hint,
            coordinate_span_m,
            coordinate_active_dimension_count,
            coordinate_characteristic_length_m,
            element_geometry_node_count: element_geometry.node_count,
            element_geometry_edge_count: element_geometry.edge_count,
            mean_element_edge_length_m: element_geometry.mean_edge_length_m,
            mean_element_area_m2: element_geometry.mean_area_m2,
            element_geometry_coverage_ratio: element_geometry.coverage_ratio,
            reference_element_coordinates_m: element_geometry.reference_coordinates_m,
            reference_element_area_m2: element_geometry.reference_area_m2,
            control_volume_cell_count: element_geometry.control_volume_cell_count,
            control_volume_face_count: element_geometry.control_volume_face_count,
            control_volume_internal_face_count: element_geometry.control_volume_internal_face_count,
            control_volume_boundary_face_count: element_geometry.control_volume_boundary_face_count,
            control_volume_connectivity_coverage_ratio: element_geometry.coverage_ratio,
            element_topology_sample_element_count: element_geometry
                .element_topology_sample
                .element_count,
            element_topology_sample_edge_count: element_geometry.element_topology_sample.edge_count,
            element_topology_sample_edge_nodes: element_geometry.element_topology_sample.edge_nodes,
            element_topology_sample_node_coordinates_m: element_geometry
                .element_topology_sample
                .node_coordinates_m,
            element_topology_sample_element_edges: element_geometry
                .element_topology_sample
                .element_edges,
            element_topology_sample_element_orientations: element_geometry
                .element_topology_sample
                .element_orientations,
            element_topology_sample_element_areas_m2: element_geometry
                .element_topology_sample
                .element_areas_m2,
            element_topology_node_coordinates_m: element_geometry
                .element_topology_node_coordinates_m,
            element_topology_edge_nodes: element_geometry.element_topology_edge_nodes,
            element_topology_element_edges: element_geometry.element_topology_element_edges,
            element_topology_element_orientations: element_geometry
                .element_topology_element_orientations,
            element_topology_element_areas_m2: element_geometry.element_topology_element_areas_m2,
        });
    }

    let mut prepared_by_source = BTreeMap::<String, String>::new();
    for prepared in &prepared_meshes {
        prepared_by_source.insert(
            prepared.source_mesh_id.clone(),
            prepared.prepared_mesh_id.clone(),
        );
    }

    let mut source_mesh_ids_by_region = BTreeMap::<String, Vec<String>>::new();
    for mapping in &geometry.region_entity_mappings {
        let entry = source_mesh_ids_by_region
            .entry(mapping.region_id.clone())
            .or_default();
        if !entry.iter().any(|mesh_id| mesh_id == &mapping.mesh_id) {
            entry.push(mapping.mesh_id.clone());
        }
    }
    for mesh_ids in source_mesh_ids_by_region.values_mut() {
        mesh_ids.sort();
    }

    let fallback_source_mesh_ids = prepared_meshes
        .iter()
        .map(|mesh| mesh.source_mesh_id.clone())
        .collect::<Vec<_>>();
    let fallback_prepared_mesh_ids = prepared_meshes
        .iter()
        .map(|mesh| mesh.prepared_mesh_id.clone())
        .collect::<Vec<_>>();

    let mut region_mappings = Vec::<RegionMeshMapping>::new();
    for region in &geometry.regions {
        let source_mesh_ids = source_mesh_ids_by_region
            .get(&region.region_id)
            .cloned()
            .filter(|mesh_ids| !mesh_ids.is_empty())
            .unwrap_or_else(|| fallback_source_mesh_ids.clone());
        let mut prepared_mesh_ids = source_mesh_ids
            .iter()
            .filter_map(|mesh_id| prepared_by_source.get(mesh_id).cloned())
            .collect::<Vec<_>>();
        if prepared_mesh_ids.is_empty() {
            prepared_mesh_ids = fallback_prepared_mesh_ids.clone();
        }
        region_mappings.push(RegionMeshMapping {
            region_id: region.region_id.clone(),
            source_mesh_ids,
            prepared_mesh_ids,
        });
    }
    if region_mappings.is_empty() {
        let source_mesh_ids = prepared_meshes
            .iter()
            .map(|mesh| mesh.source_mesh_id.clone())
            .collect::<Vec<_>>();
        let prepared_mesh_ids = prepared_meshes
            .iter()
            .map(|mesh| mesh.prepared_mesh_id.clone())
            .collect::<Vec<_>>();
        region_mappings.push(RegionMeshMapping {
            region_id: "region_default".to_string(),
            source_mesh_ids: source_mesh_ids.clone(),
            prepared_mesh_ids: prepared_mesh_ids.clone(),
        });
    }
    region_mappings.sort_by(|a, b| a.region_id.cmp(&b.region_id));

    let total_elements = prepared_meshes
        .iter()
        .map(|mesh| mesh.element_count)
        .sum::<u64>()
        .max(1);
    let total_nodes = prepared_meshes
        .iter()
        .map(|mesh| mesh.node_count)
        .sum::<u64>()
        .max(1);
    let element_density = total_elements as f64 / total_nodes as f64;
    let normalized_density = element_density.min(1.0);
    let (min_scaled_jacobian, mean_aspect_ratio) = match options.profile {
        MeshingProfile::SurfaceOnly => (
            (0.89 - 0.1 * normalized_density).max(0.5),
            1.4 + normalized_density,
        ),
        MeshingProfile::AnalysisReady => (
            (0.92 - 0.1 * normalized_density).max(0.5),
            1.2 + normalized_density,
        ),
        MeshingProfile::AdaptiveRefine => (
            (0.95 - 0.03 * normalized_density).clamp(0.5, 0.99),
            1.05 + 0.15 * normalized_density,
        ),
    };

    Ok(MeshingPrepResult {
        schema_version: "geometry-prep-for-analysis/v1".to_string(),
        prepared_meshes,
        region_mappings,
        quality: MeshingQualityReport {
            min_scaled_jacobian,
            mean_aspect_ratio,
            inverted_element_count: 0,
        },
        provenance: MeshingProvenance {
            algorithm: "deterministic_topology_seed/v1".to_string(),
            profile: options.profile,
            source_geometry_id: geometry.geometry_id.clone(),
            source_geometry_revision: geometry.revision,
        },
    })
}

#[cfg(test)]
mod tests;
