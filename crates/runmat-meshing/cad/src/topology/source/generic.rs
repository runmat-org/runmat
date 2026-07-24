use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::math::dot;
use crate::topology::{SourceTopologyFace, SourceTopologyModel};

pub(super) fn generic_coplanar_face_ids(
    topology: &SourceTopologyModel,
    face_region_by_source_face: &BTreeMap<u32, Vec<String>>,
) -> BTreeMap<u32, String> {
    let faces_by_id = topology
        .faces
        .iter()
        .map(|face| (face.face_id, face))
        .collect::<BTreeMap<_, _>>();
    let coordinates_by_vertex = topology
        .vertices
        .iter()
        .map(|vertex| (vertex.vertex_id, vertex.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let adjacent_faces = adjacent_faces_by_shared_edge(topology);
    let mut visited = BTreeSet::<u32>::new();
    let mut face_ids_by_source_face = BTreeMap::<u32, String>::new();
    let mut component_index = 0_usize;

    for face in &topology.faces {
        if !visited.insert(face.face_id) {
            continue;
        }
        let component_key =
            generic_face_key(face, &coordinates_by_vertex, face_region_by_source_face);
        let mut component_face_ids = Vec::<u32>::new();
        let mut queue = VecDeque::from([face.face_id]);
        while let Some(face_id) = queue.pop_front() {
            if !faces_by_id.contains_key(&face_id) {
                continue;
            }
            component_face_ids.push(face_id);
            for adjacent_face_id in adjacent_faces.get(&face_id).into_iter().flatten() {
                if visited.contains(adjacent_face_id) {
                    continue;
                }
                let Some(adjacent_face) = faces_by_id.get(adjacent_face_id).copied() else {
                    continue;
                };
                if generic_face_key(
                    adjacent_face,
                    &coordinates_by_vertex,
                    face_region_by_source_face,
                ) != component_key
                {
                    continue;
                }
                visited.insert(*adjacent_face_id);
                queue.push_back(*adjacent_face_id);
            }
        }
        component_face_ids.sort_unstable();
        let component_id = if component_face_ids.len() > 1 {
            let id = format!("cad_face_generic_{component_index:06}");
            component_index += 1;
            id
        } else {
            format!("cad_face_{}", component_face_ids[0])
        };
        for face_id in component_face_ids {
            face_ids_by_source_face.insert(face_id, component_id.clone());
        }
    }

    face_ids_by_source_face
}

fn adjacent_faces_by_shared_edge(topology: &SourceTopologyModel) -> BTreeMap<u32, Vec<u32>> {
    let mut adjacent_faces = BTreeMap::<u32, BTreeSet<u32>>::new();
    for edge in &topology.edges {
        for left in &edge.adjacent_face_ids {
            for right in &edge.adjacent_face_ids {
                if left != right {
                    adjacent_faces.entry(*left).or_default().insert(*right);
                }
            }
        }
    }
    adjacent_faces
        .into_iter()
        .map(|(face_id, neighbors)| (face_id, neighbors.into_iter().collect()))
        .collect()
}

fn generic_face_key(
    face: &SourceTopologyFace,
    coordinates_by_vertex: &BTreeMap<u32, [f64; 3]>,
    face_region_by_source_face: &BTreeMap<u32, Vec<String>>,
) -> (i64, i64, i64, i64, Vec<String>) {
    let normal = canonical_normal(face.unit_normal);
    let origin = coordinates_by_vertex
        .get(&face.node_ids[0])
        .copied()
        .unwrap_or([0.0; 3]);
    let offset = dot(normal, origin);
    let mut region_ids = face_region_by_source_face
        .get(&face.source_triangle_id)
        .cloned()
        .unwrap_or_else(|| face.region_ids.clone());
    region_ids.sort();
    region_ids.dedup();
    (
        quantize(normal[0]),
        quantize(normal[1]),
        quantize(normal[2]),
        quantize(offset),
        region_ids,
    )
}

fn canonical_normal(mut normal: [f64; 3]) -> [f64; 3] {
    if normal[0] < -f64::EPSILON
        || (normal[0].abs() <= f64::EPSILON && normal[1] < -f64::EPSILON)
        || (normal[0].abs() <= f64::EPSILON
            && normal[1].abs() <= f64::EPSILON
            && normal[2] < -f64::EPSILON)
    {
        normal = [-normal[0], -normal[1], -normal[2]];
    }
    normal
}

fn quantize(value: f64) -> i64 {
    (value * 1.0e9).round() as i64
}
