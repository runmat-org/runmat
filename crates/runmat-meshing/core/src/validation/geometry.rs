use crate::{
    artifact::AnalysisMeshArtifact,
    topology::{BoundaryElementKind, VolumeElementKind},
};

pub(super) fn mesh_bounds_m(mesh: &AnalysisMeshArtifact) -> Option<[[f64; 3]; 2]> {
    let mut nodes = mesh.nodes.iter();
    let first = nodes.next()?.coordinates_m;
    let mut min = first;
    let mut max = first;
    for node in nodes {
        for axis in 0..3 {
            min[axis] = min[axis].min(node.coordinates_m[axis]);
            max[axis] = max[axis].max(node.coordinates_m[axis]);
        }
    }
    Some([min, max])
}

pub(super) fn mesh_volume_m3(mesh: &AnalysisMeshArtifact) -> f64 {
    mesh.volume_elements
        .iter()
        .filter(|element| {
            element.kind == VolumeElementKind::Tetrahedron4 && element.node_ids.len() == 4
        })
        .filter_map(|element| {
            Some(tetrahedron_volume_m3(element_tetrahedron_points(
                mesh,
                element.node_ids.as_slice(),
            )?))
        })
        .sum()
}

pub(super) fn element_tetrahedron_points(
    mesh: &AnalysisMeshArtifact,
    node_ids: &[u32],
) -> Option<[[f64; 3]; 4]> {
    Some([
        mesh_node(mesh, node_ids[0])?,
        mesh_node(mesh, node_ids[1])?,
        mesh_node(mesh, node_ids[2])?,
        mesh_node(mesh, node_ids[3])?,
    ])
}

pub(super) fn mesh_boundary_area_m2(mesh: &AnalysisMeshArtifact) -> f64 {
    mesh.boundary_faces
        .iter()
        .filter(|face| face.kind == BoundaryElementKind::Tri3 && face.node_ids.len() == 3)
        .filter_map(|face| {
            Some(triangle_area_m2([
                mesh_node(mesh, face.node_ids[0])?,
                mesh_node(mesh, face.node_ids[1])?,
                mesh_node(mesh, face.node_ids[2])?,
            ]))
        })
        .sum()
}

fn mesh_node(mesh: &AnalysisMeshArtifact, node_id: u32) -> Option<[f64; 3]> {
    mesh.nodes
        .iter()
        .find(|node| node.node_id == node_id)
        .map(|node| node.coordinates_m)
}

pub fn mesh_contains_point(mesh: &AnalysisMeshArtifact, point: [f64; 3]) -> bool {
    mesh.volume_elements
        .iter()
        .filter(|element| {
            element.kind == VolumeElementKind::Tetrahedron4 && element.node_ids.len() == 4
        })
        .filter_map(|element| {
            Some([
                mesh_node(mesh, element.node_ids[0])?,
                mesh_node(mesh, element.node_ids[1])?,
                mesh_node(mesh, element.node_ids[2])?,
                mesh_node(mesh, element.node_ids[3])?,
            ])
        })
        .any(|tetrahedron| point_in_tetrahedron(point, tetrahedron))
}

fn point_in_tetrahedron(point: [f64; 3], tetrahedron: [[f64; 3]; 4]) -> bool {
    let total = tetrahedron_volume_m3(tetrahedron);
    if !total.is_finite() || total <= f64::EPSILON {
        return false;
    }
    let subvolume_sum =
        tetrahedron_volume_m3([point, tetrahedron[1], tetrahedron[2], tetrahedron[3]])
            + tetrahedron_volume_m3([tetrahedron[0], point, tetrahedron[2], tetrahedron[3]])
            + tetrahedron_volume_m3([tetrahedron[0], tetrahedron[1], point, tetrahedron[3]])
            + tetrahedron_volume_m3([tetrahedron[0], tetrahedron[1], tetrahedron[2], point]);
    let tolerance = total * 1.0e-8 + f64::EPSILON;
    (subvolume_sum - total).abs() <= tolerance
}

pub(super) fn tetrahedron_volume_m3(points: [[f64; 3]; 4]) -> f64 {
    dot(
        sub(points[1], points[0]),
        cross(sub(points[2], points[0]), sub(points[3], points[0])),
    )
    .abs()
        / 6.0
}

fn triangle_area_m2(points: [[f64; 3]; 3]) -> f64 {
    0.5 * norm(cross(sub(points[1], points[0]), sub(points[2], points[0])))
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn norm(value: [f64; 3]) -> f64 {
    dot(value, value).sqrt()
}
