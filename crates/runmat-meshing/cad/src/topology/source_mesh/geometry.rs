use super::SourceTopologyInput;

pub(super) fn triangle_vertices(
    input: &SourceTopologyInput,
    node_ids: [u32; 3],
) -> Option<[[f64; 3]; 3]> {
    Some([
        *input.vertices.get(node_ids[0] as usize)?,
        *input.vertices.get(node_ids[1] as usize)?,
        *input.vertices.get(node_ids[2] as usize)?,
    ])
}

pub(super) fn triangle_area(vertices: [[f64; 3]; 3]) -> f64 {
    0.5 * norm(cross(
        sub(vertices[1], vertices[0]),
        sub(vertices[2], vertices[0]),
    ))
}

pub(super) fn triangle_unit_normal(vertices: [[f64; 3]; 3]) -> [f64; 3] {
    let normal = cross(sub(vertices[1], vertices[0]), sub(vertices[2], vertices[0]));
    let length = norm(normal);
    if !length.is_finite() || length <= f64::EPSILON {
        return [0.0, 0.0, 0.0];
    }
    [normal[0] / length, normal[1] / length, normal[2] / length]
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

pub(super) fn norm(value: [f64; 3]) -> f64 {
    distance([0.0, 0.0, 0.0], value)
}

pub(super) fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2))
        .sqrt()
}
