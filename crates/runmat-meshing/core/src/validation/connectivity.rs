use std::collections::{BTreeMap, BTreeSet};

use crate::{
    artifact::AnalysisMeshArtifact,
    topology::{BoundaryElementKind, VolumeElementKind},
};

pub(super) fn boundary_face_edges(mesh: &AnalysisMeshArtifact) -> BTreeSet<[u32; 2]> {
    let mut edges = BTreeSet::<[u32; 2]>::new();
    for face in &mesh.boundary_faces {
        if face.kind != BoundaryElementKind::Tri3 || face.node_ids.len() != 3 {
            continue;
        }
        edges.insert(sorted_edge(face.node_ids[0], face.node_ids[1]));
        edges.insert(sorted_edge(face.node_ids[1], face.node_ids[2]));
        edges.insert(sorted_edge(face.node_ids[2], face.node_ids[0]));
    }
    edges
}

pub fn volume_component_count(mesh: &AnalysisMeshArtifact) -> usize {
    volume_component_element_counts(mesh).len()
}

pub fn volume_component_element_counts(mesh: &AnalysisMeshArtifact) -> Vec<usize> {
    if mesh.volume_elements.is_empty() {
        return Vec::new();
    }
    let mut face_to_elements = BTreeMap::<[u32; 3], Vec<usize>>::new();
    for (element_index, element) in mesh.volume_elements.iter().enumerate() {
        if element.kind != VolumeElementKind::Tetrahedron4 || element.node_ids.len() != 4 {
            continue;
        }
        for face in tetrahedron_element_faces(element.node_ids.as_slice()) {
            face_to_elements
                .entry(face)
                .or_default()
                .push(element_index);
        }
    }
    let mut adjacency = vec![Vec::<usize>::new(); mesh.volume_elements.len()];
    for element_indices in face_to_elements.values() {
        for left_position in 0..element_indices.len() {
            for right_position in (left_position + 1)..element_indices.len() {
                let left = element_indices[left_position];
                let right = element_indices[right_position];
                adjacency[left].push(right);
                adjacency[right].push(left);
            }
        }
    }

    let mut visited = vec![false; mesh.volume_elements.len()];
    let mut component_element_counts = Vec::<usize>::new();
    for start in 0..mesh.volume_elements.len() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut component_element_count = 0_usize;
        let mut stack = vec![start];
        while let Some(current) = stack.pop() {
            component_element_count += 1;
            for neighbor in &adjacency[current] {
                if visited[*neighbor] {
                    continue;
                }
                visited[*neighbor] = true;
                stack.push(*neighbor);
            }
        }
        component_element_counts.push(component_element_count);
    }
    component_element_counts
}

fn tetrahedron_element_faces(node_ids: &[u32]) -> [[u32; 3]; 4] {
    [
        sorted_node_face([node_ids[0], node_ids[1], node_ids[2]]),
        sorted_node_face([node_ids[0], node_ids[1], node_ids[3]]),
        sorted_node_face([node_ids[0], node_ids[2], node_ids[3]]),
        sorted_node_face([node_ids[1], node_ids[2], node_ids[3]]),
    ]
}

pub(super) fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

fn sorted_node_face(mut node_ids: [u32; 3]) -> [u32; 3] {
    node_ids.sort_unstable();
    node_ids
}
