//! Disposable boundary rendering from the original solver nodes and element mappings.
use super::{
    AnalysisRenderMesh, AnalysisRenderRegion, AnalysisRenderTopology, AnalysisRenderTopologySource,
    AnalysisRenderTriangleRange,
};
use runmat_meshing_core::SolverMeshArtifact;
use std::collections::{BTreeMap, HashMap};

#[cfg(test)]
mod tests;

pub(super) fn render_topology_from_solver_mesh(
    mesh: Option<&SolverMeshArtifact>,
) -> Option<AnalysisRenderTopology> {
    let mesh = mesh?;
    if mesh.topology.nodes.is_empty() || mesh.topology.boundary_faces.is_empty() {
        return None;
    }
    u32::try_from(mesh.topology.nodes.len()).ok()?;
    let triangle_count = mesh
        .topology
        .boundary_faces
        .iter()
        .try_fold(0_usize, |count, face| {
            count.checked_add(match face.node_ids.len() {
                3 => 1,
                6 => 4,
                _ => return None,
            })
        })?;
    u32::try_from(triangle_count).ok()?;

    let node_index_by_id = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index as u32))
        .collect::<HashMap<_, _>>();
    let volume_index_by_id = mesh
        .topology
        .volume_elements
        .iter()
        .enumerate()
        .map(|(index, element)| (element.element_id, index))
        .collect::<HashMap<_, _>>();
    let mut triangles = Vec::with_capacity(triangle_count);
    let mut triangle_volume_element_indices = Vec::with_capacity(triangle_count);
    let mut region_triangles = BTreeMap::<String, Vec<u32>>::new();
    // Project one face at a time. Do not retain a second complete boundary
    // inventory or grow buffers from a linear-face capacity for quadratic faces.
    for face in &mesh.topology.boundary_faces {
        let face_triangles = boundary_triangles(&face.node_ids, &node_index_by_id)?;
        let volume_element_index = face
            .adjacent_volume_element_ids
            .iter()
            .find_map(|element_id| volume_index_by_id.get(element_id).copied());
        let region_ids: Vec<String> = face
            .provenance
            .iter()
            .flat_map(|entity| {
                std::iter::once(entity.source_topology_id.clone())
                    .chain(entity.analysis_selection_id().ok())
            })
            .collect();
        for triangle in face_triangles {
            let triangle_index = triangles.len() as u32;
            triangles.push(triangle);
            triangle_volume_element_indices.push(volume_element_index);
            for region_id in &region_ids {
                region_triangles
                    .entry(region_id.clone())
                    .or_default()
                    .push(triangle_index);
            }
        }
    }
    if triangles.is_empty() {
        return None;
    }
    let regions = render_regions_from_triangle_indices(region_triangles);

    Some(AnalysisRenderTopology {
        schema_version: "analysis_render_topology/v1".to_string(),
        source: AnalysisRenderTopologySource::AnalysisMesh,
        volume_element_count: mesh.topology.volume_elements.len(),
        meshes: vec![AnalysisRenderMesh {
            mesh_id: "solver_mesh_boundary".to_string(),
            vertices: mesh
                .topology
                .nodes
                .iter()
                .map(|node| node.coordinates_m)
                .collect(),
            triangles,
            regions,
            vertex_volume_node_indices: (0..mesh.topology.nodes.len()).map(Some).collect(),
            triangle_volume_element_indices,
        }],
    })
}

/// Tri6 midside order is 01, 12, 20. Render four consistently oriented pieces
/// using original nodes; field recovery remains owned by the numerical result.
fn boundary_triangles(ids: &[u64], node_indices: &HashMap<u64, u32>) -> Option<Vec<[u32; 3]>> {
    let patterns: &[[usize; 3]] = match ids.len() {
        3 => &[[0, 1, 2]],
        6 => &[[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]],
        _ => return None,
    };
    patterns
        .iter()
        .map(|corners| {
            Some([
                *node_indices.get(&ids[corners[0]])?,
                *node_indices.get(&ids[corners[1]])?,
                *node_indices.get(&ids[corners[2]])?,
            ])
        })
        .collect()
}

fn render_regions_from_triangle_indices(
    region_triangles: BTreeMap<String, Vec<u32>>,
) -> Vec<AnalysisRenderRegion> {
    region_triangles
        .into_iter()
        .filter_map(|(region_id, mut triangle_indices)| {
            triangle_indices.sort_unstable();
            triangle_indices.dedup();
            let mut ranges = Vec::<AnalysisRenderTriangleRange>::new();
            for triangle_index in triangle_indices {
                if let Some(last) = ranges.last_mut() {
                    if last.start.saturating_add(last.count) == triangle_index {
                        last.count = last.count.saturating_add(1);
                        continue;
                    }
                }
                ranges.push(AnalysisRenderTriangleRange {
                    start: triangle_index,
                    count: 1,
                });
            }
            if ranges.is_empty() {
                return None;
            }
            Some(AnalysisRenderRegion {
                region_id,
                label: None,
                tag: Some("boundary".to_string()),
                triangle_ranges: ranges,
            })
        })
        .collect()
}
