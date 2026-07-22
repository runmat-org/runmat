use std::collections::BTreeMap;

use runmat_meshing_core::contracts::{ProtectedBoundaryComplex, TopologyEntityId};

use crate::{
    cavity::constrained::{ConstrainedCavityBoundaryFace, ConstrainedCavityRefillTetrahedron},
    generate::TetrahedronGenerationError,
};

use super::{
    builder::PartitionBuilder,
    geometry::{cross, dot, norm, scale, sub},
};

pub(super) fn sorted_face_vec(node_ids: &[u32]) -> Vec<u32> {
    let mut sorted = node_ids.to_vec();
    sorted.sort();
    sorted
}

pub(super) fn triangulated_polygon_faces(
    face_key: &[u32],
    builder: &PartitionBuilder,
) -> Vec<[u32; 3]> {
    let ordered = polygon_order(face_key, builder);
    let Some((anchor_index, anchor)) = ordered
        .iter()
        .copied()
        .enumerate()
        .min_by_key(|(_, node_id)| *node_id)
    else {
        return Vec::new();
    };
    let mut rotated = ordered[anchor_index..].to_vec();
    rotated.extend_from_slice(&ordered[..anchor_index]);
    (1..rotated.len().saturating_sub(1))
        .map(|index| [anchor, rotated[index], rotated[index + 1]])
        .collect()
}

pub(super) fn partition_boundary_faces(
    tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    builder: &PartitionBuilder,
    inner_lower_bounds: [f64; 4],
    outer_source_faces: &BTreeMap<usize, usize>,
    inner_source_faces: &BTreeMap<usize, usize>,
) -> Result<Vec<ConstrainedCavityBoundaryFace>, TetrahedronGenerationError> {
    let mut face_counts = BTreeMap::<[u32; 3], ([u32; 3], usize)>::new();
    for tetrahedron in tetrahedra {
        for face in tetrahedron_faces(tetrahedron.node_ids) {
            let key = sorted_face(face);
            face_counts
                .entry(key)
                .and_modify(|(_, count)| *count += 1)
                .or_insert((face, 1));
        }
    }
    let mut boundary_faces = Vec::<ConstrainedCavityBoundaryFace>::new();
    for (_, (node_ids, count)) in face_counts {
        if count != 1 {
            continue;
        }
        let source_facet_index = boundary_source_facet_index(
            node_ids,
            builder,
            inner_lower_bounds,
            outer_source_faces,
            inner_source_faces,
        )?;
        boundary_faces.push(ConstrainedCavityBoundaryFace {
            node_ids,
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: Some(
                u32::try_from(source_facet_index).map_err(|_| {
                    TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc
                })?,
            ),
            source_edge_ids: [None, None, None],
            region_ids: Vec::new(),
        });
    }
    Ok(boundary_faces)
}

pub(super) fn shell_source_faces(
    plc: &ProtectedBoundaryComplex,
    builder: &PartitionBuilder,
    face_barycentric_values: [f64; 4],
) -> Result<BTreeMap<usize, usize>, TetrahedronGenerationError> {
    let mut source_faces = BTreeMap::<usize, usize>::new();
    for omitted_index in 0..4 {
        let mut source_face_id = None::<TopologyEntityId>;
        let mut source_facet_index = None::<usize>;
        for (facet_index, facet) in plc.facets.iter().enumerate().filter(|(_, facet)| {
            facet.node_ids.iter().all(|node_id| {
                builder
                    .barycentric_for_topology_node(node_id)
                    .map(|barycentric| {
                        (barycentric[omitted_index] - face_barycentric_values[omitted_index]).abs()
                            <= 1.0e-12
                    })
                    .unwrap_or(false)
            })
        }) {
            match &source_face_id {
                Some(existing_source_face_id)
                    if existing_source_face_id != &facet.source_face_id =>
                {
                    return Err(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc);
                }
                None => {
                    source_face_id = Some(facet.source_face_id.clone());
                    source_facet_index = Some(facet_index);
                }
                _ => {}
            }
        }
        let source_facet_index = source_facet_index
            .ok_or(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc)?;
        source_faces.insert(omitted_index, source_facet_index);
    }
    Ok(source_faces)
}

fn polygon_order(face_key: &[u32], builder: &PartitionBuilder) -> Vec<u32> {
    let mut center = [0.0; 3];
    for node_id in face_key {
        let point = builder.coordinates(*node_id);
        for axis in 0..3 {
            center[axis] += point[axis];
        }
    }
    for coordinate in &mut center {
        *coordinate /= face_key.len() as f64;
    }
    let points = face_key
        .iter()
        .map(|node_id| (*node_id, builder.coordinates(*node_id)))
        .collect::<Vec<_>>();
    let normal = polygon_normal(&points).unwrap_or([0.0, 0.0, 1.0]);
    let mut first_axis = sub(points[0].1, center);
    let first_axis_norm = norm(first_axis);
    if first_axis_norm <= f64::EPSILON {
        return face_key.to_vec();
    }
    first_axis = scale(first_axis, 1.0 / first_axis_norm);
    let second_axis = cross(normal, first_axis);
    let mut ordered = points
        .into_iter()
        .map(|(node_id, point)| {
            let relative = sub(point, center);
            let angle = dot(relative, second_axis).atan2(dot(relative, first_axis));
            (node_id, angle)
        })
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.1.total_cmp(&right.1));
    ordered.into_iter().map(|(node_id, _)| node_id).collect()
}

fn polygon_normal(points: &[(u32, [f64; 3])]) -> Option<[f64; 3]> {
    for first in 0..points.len() {
        for second in (first + 1)..points.len() {
            for third in (second + 1)..points.len() {
                let normal = cross(
                    sub(points[second].1, points[first].1),
                    sub(points[third].1, points[first].1),
                );
                let length = norm(normal);
                if length > 1.0e-12 {
                    return Some(scale(normal, 1.0 / length));
                }
            }
        }
    }
    None
}

fn boundary_source_facet_index(
    node_ids: [u32; 3],
    builder: &PartitionBuilder,
    inner_lower_bounds: [f64; 4],
    outer_source_faces: &BTreeMap<usize, usize>,
    inner_source_faces: &BTreeMap<usize, usize>,
) -> Result<usize, TetrahedronGenerationError> {
    for (coordinate_index, inner_lower_bound) in inner_lower_bounds.iter().enumerate() {
        if node_ids
            .iter()
            .all(|node_id| builder.barycentric_by_id[node_id][coordinate_index].abs() <= 1.0e-12)
        {
            return outer_source_faces
                .get(&coordinate_index)
                .copied()
                .ok_or(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc);
        }
        if node_ids.iter().all(|node_id| {
            (builder.barycentric_by_id[node_id][coordinate_index] - inner_lower_bound).abs()
                <= 1.0e-12
        }) {
            return inner_source_faces
                .get(&coordinate_index)
                .copied()
                .ok_or(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc);
        }
    }
    Err(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc)
}

fn tetrahedron_faces(node_ids: [u32; 4]) -> [[u32; 3]; 4] {
    [
        [node_ids[0], node_ids[1], node_ids[2]],
        [node_ids[0], node_ids[1], node_ids[3]],
        [node_ids[0], node_ids[2], node_ids[3]],
        [node_ids[1], node_ids[2], node_ids[3]],
    ]
}

fn sorted_face(mut node_ids: [u32; 3]) -> [u32; 3] {
    node_ids.sort();
    node_ids
}
