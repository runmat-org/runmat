//! Disposable boundary rendering from the original solver nodes and element mappings.
use super::{
    AnalysisRenderMesh, AnalysisRenderRegion, AnalysisRenderTopology, AnalysisRenderTopologySource,
    AnalysisRenderTriangleRange,
};
use runmat_meshing_core::SolverMeshArtifact;
use std::collections::{BTreeMap, HashMap};

#[cfg(test)]
mod tests;

/// How finely a quadratic (Tri6) boundary face is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceDetail {
    /// Four triangles through the corner and midside nodes: every node is drawn where it is.
    Full,
    /// One triangle through the corners: a quarter of the triangles, for a scene that must fit
    /// a payload budget. The midside nodes are dropped; [`DroppedMidsides`] lists them so the
    /// caller can state the position and value error.
    Corners,
}

/// The midside nodes a [`SurfaceDetail::Corners`] surface leaves out: (corner, corner, midside)
/// node indices per face edge.
pub type DroppedMidsides = Vec<[u32; 3]>;

pub(super) fn render_topology_from_solver_mesh(
    mesh: Option<&SolverMeshArtifact>,
) -> Option<AnalysisRenderTopology> {
    render_topology_from_solver_mesh_at(mesh, SurfaceDetail::Full).map(|(topology, _)| topology)
}

pub(super) fn render_topology_from_solver_mesh_at(
    mesh: Option<&SolverMeshArtifact>,
    detail: SurfaceDetail,
) -> Option<(AnalysisRenderTopology, DroppedMidsides)> {
    let mesh = mesh?;
    let mut dropped = DroppedMidsides::new();
    if mesh.topology.nodes.is_empty() || mesh.topology.boundary_faces.is_empty() {
        return None;
    }
    u32::try_from(mesh.topology.nodes.len()).ok()?;
    let triangle_count = mesh
        .topology
        .boundary_faces
        .iter()
        .try_fold(0_usize, |count, face| {
            count.checked_add(match (face.node_ids.len(), detail) {
                (3, _) | (6, SurfaceDetail::Corners) => 1,
                (6, SurfaceDetail::Full) => 4,
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
        let mut face_triangles = boundary_triangles(&face.node_ids, &node_index_by_id, detail)?;
        if detail == SurfaceDetail::Corners && face.node_ids.len() == 6 {
            for [a, b, midside] in [[0, 1, 3], [1, 2, 4], [2, 0, 5]] {
                dropped.push([a, b, midside].map(|k| node_index_by_id[&face.node_ids[k]]));
            }
        }
        let volume_element_index = face
            .adjacent_volume_element_ids
            .iter()
            .find_map(|element_id| volume_index_by_id.get(element_id).copied());
        // A boundary face's stored winding is not a rendering convention; draw it facing out of
        // its adjacent element so front faces, normals and culling agree on every face.
        if let Some(element) =
            volume_element_index.map(|index| &mesh.topology.volume_elements[index])
        {
            if faces_into_element(mesh, &face.node_ids, element, &node_index_by_id) {
                for triangle in &mut face_triangles {
                    triangle.swap(1, 2);
                }
            }
        }
        let region_ids: Vec<String> = face
            .provenance
            .iter()
            .filter_map(|entity| entity.analysis_selection_id().ok())
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

    Some((AnalysisRenderTopology {
        schema_version: "analysis_render_topology/v1".to_string(),
        source: AnalysisRenderTopologySource::AnalysisMesh,
        volume_element_count: mesh.topology.volume_elements.len(),
        meshes: vec![AnalysisRenderMesh {
            mesh_id: "solver_mesh_boundary".to_string(),
            vertices: mesh
                .topology
                .nodes
                .iter()
                .map(|node| node.coordinates_m.canonical_value())
                .collect(),
            triangles,
            regions,
            vertex_volume_node_indices: (0..mesh.topology.nodes.len()).map(Some).collect(),
            triangle_volume_element_indices,
        }],
    }, dropped))
}

/// Whether the face's corner winding normal points into its adjacent element (towards the
/// element's corner centroid).
fn faces_into_element(
    mesh: &SolverMeshArtifact,
    face_node_ids: &[u64],
    element: &runmat_meshing_core::SolverVolumeElement,
    node_indices: &HashMap<u64, u32>,
) -> bool {
    let position = |id: &u64| {
        node_indices.get(id).map(|index| {
            mesh.topology.nodes[*index as usize]
                .coordinates_m
                .canonical_value()
        })
    };
    let corners = face_node_ids
        .iter()
        .take(3)
        .map(position)
        .collect::<Option<Vec<_>>>();
    let element_corners = element
        .node_ids
        .iter()
        .take(4)
        .map(position)
        .collect::<Option<Vec<_>>>();
    let (Some(corners), Some(element_corners)) = (corners, element_corners) else {
        return false;
    };
    let edge = |a: [f64; 3], b: [f64; 3]| [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let (u, v) = (edge(corners[0], corners[1]), edge(corners[0], corners[2]));
    let normal = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let count = element_corners.len() as f64;
    let toward_element: [f64; 3] = std::array::from_fn(|axis| {
        element_corners
            .iter()
            .map(|corner| corner[axis])
            .sum::<f64>()
            / count
            - (corners[0][axis] + corners[1][axis] + corners[2][axis]) / 3.0
    });
    (0..3)
        .map(|axis| normal[axis] * toward_element[axis])
        .sum::<f64>()
        > 0.0
}

/// Tri6 midside order is 01, 12, 20. Render four consistently oriented pieces
/// using original nodes; field recovery remains owned by the numerical result.
fn boundary_triangles(
    ids: &[u64],
    node_indices: &HashMap<u64, u32>,
    detail: SurfaceDetail,
) -> Option<Vec<[u32; 3]>> {
    let patterns: &[[usize; 3]] = match (ids.len(), detail) {
        (3, _) | (6, SurfaceDetail::Corners) => &[[0, 1, 2]],
        (6, SurfaceDetail::Full) => &[[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]],
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
