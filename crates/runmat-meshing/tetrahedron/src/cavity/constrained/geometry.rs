use std::collections::{BTreeMap, BTreeSet};

pub(super) fn centroid_of_node_set(
    node_ids: &BTreeSet<u32>,
    node_coordinates: &BTreeMap<u32, [f64; 3]>,
) -> Option<[f64; 3]> {
    if node_ids.is_empty() {
        return None;
    }
    let mut centroid = [0.0; 3];
    for node_id in node_ids {
        let point = node_coordinates.get(node_id)?;
        centroid[0] += point[0];
        centroid[1] += point[1];
        centroid[2] += point[2];
    }
    let scale = 1.0 / node_ids.len() as f64;
    Some([
        centroid[0] * scale,
        centroid[1] * scale,
        centroid[2] * scale,
    ])
}

pub(super) fn face_centroid(
    face: [u32; 3],
    node_coordinates: &BTreeMap<u32, [f64; 3]>,
) -> Option<[f64; 3]> {
    let first = node_coordinates.get(&face[0]).copied()?;
    let second = node_coordinates.get(&face[1]).copied()?;
    let third = node_coordinates.get(&face[2]).copied()?;
    Some([
        (first[0] + second[0] + third[0]) / 3.0,
        (first[1] + second[1] + third[1]) / 3.0,
        (first[2] + second[2] + third[2]) / 3.0,
    ])
}

pub(super) fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

pub(super) fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

pub(super) fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let norm = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return None;
    }
    Some([vector[0] / norm, vector[1] / norm, vector[2] / norm])
}

pub(super) fn midpoint(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        (left[0] + right[0]) * 0.5,
        (left[1] + right[1]) * 0.5,
        (left[2] + right[2]) * 0.5,
    ]
}

pub(super) fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    let delta = [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
    (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt()
}
