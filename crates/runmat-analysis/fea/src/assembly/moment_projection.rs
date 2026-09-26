use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MomentProjectionSummary {
    pub load_id: String,
    pub region_id: String,
    pub target_node_count: usize,
    pub projection_point_m: [f64; 3],
    pub applied_force: [f64; 3],
    pub applied_moment: [f64; 3],
    pub force_residual: [f64; 3],
    pub moment_residual: [f64; 3],
    pub projection_applied: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct MomentProjection {
    pub point_m: [f64; 3],
    pub nodal_forces: Vec<[f64; 3]>,
    pub applied_force: [f64; 3],
    pub applied_moment: [f64; 3],
    pub projection_applied: bool,
}

pub(super) fn project_moment(
    node_coordinates_m: &[[f64; 3]],
    target_nodes: &[usize],
    requested_moment: [f64; 3],
) -> MomentProjection {
    if target_nodes.is_empty() {
        return MomentProjection {
            point_m: [0.0; 3],
            nodal_forces: Vec::new(),
            applied_force: [0.0; 3],
            applied_moment: [0.0; 3],
            projection_applied: false,
        };
    }

    let point_m = target_centroid(node_coordinates_m, target_nodes);
    let offsets = target_nodes
        .iter()
        .map(|&node_index| subtract(node_coordinates_m[node_index], point_m))
        .collect::<Vec<_>>();
    let mut nodal_forces = vec![[0.0; 3]; target_nodes.len()];
    let mut projection_applied = false;

    if requested_moment
        .iter()
        .any(|component| component.abs() > f64::EPSILON)
    {
        let mut coupling = [[0.0_f64; 3]; 3];
        for offset in &offsets {
            let radius_squared = dot(*offset, *offset);
            for row in 0..3 {
                coupling[row][row] += radius_squared;
                for col in 0..3 {
                    coupling[row][col] -= offset[row] * offset[col];
                }
            }
        }

        if let Some(inverse) = invert_3x3(coupling) {
            let multiplier = matrix_vector_product(inverse, requested_moment);
            for (offset, nodal_force) in offsets.iter().zip(nodal_forces.iter_mut()) {
                *nodal_force = cross(multiplier, *offset);
            }
            projection_applied = true;
        }
    }

    let (applied_force, applied_moment) = resultants(&offsets, &nodal_forces);
    MomentProjection {
        point_m,
        nodal_forces,
        applied_force,
        applied_moment,
        projection_applied,
    }
}

pub(super) fn target_centroid(node_coordinates_m: &[[f64; 3]], target_nodes: &[usize]) -> [f64; 3] {
    let mut centroid = [0.0_f64; 3];
    for &node_index in target_nodes {
        let node = node_coordinates_m[node_index];
        for axis in 0..3 {
            centroid[axis] += node[axis];
        }
    }
    scale(centroid, 1.0 / target_nodes.len() as f64)
}

fn resultants(offsets: &[[f64; 3]], nodal_forces: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut force = [0.0_f64; 3];
    let mut moment = [0.0_f64; 3];
    for (&offset, &nodal_force) in offsets.iter().zip(nodal_forces) {
        for axis in 0..3 {
            force[axis] += nodal_force[axis];
        }
        let nodal_moment = cross(offset, nodal_force);
        for axis in 0..3 {
            moment[axis] += nodal_moment[axis];
        }
    }
    (force, moment)
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn scale(value: [f64; 3], factor: f64) -> [f64; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn matrix_vector_product(matrix: [[f64; 3]; 3], value: [f64; 3]) -> [f64; 3] {
    [
        dot(matrix[0], value),
        dot(matrix[1], value),
        dot(matrix[2], value),
    ]
}

fn invert_3x3(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c00 = matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1];
    let c01 = -(matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]);
    let c02 = matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0];
    let c10 = -(matrix[0][1] * matrix[2][2] - matrix[0][2] * matrix[2][1]);
    let c11 = matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0];
    let c12 = -(matrix[0][0] * matrix[2][1] - matrix[0][1] * matrix[2][0]);
    let c20 = matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1];
    let c21 = -(matrix[0][0] * matrix[1][2] - matrix[0][2] * matrix[1][0]);
    let c22 = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    let determinant = matrix[0][0] * c00 + matrix[0][1] * c01 + matrix[0][2] * c02;
    if determinant.abs() <= 1.0e-18 {
        return None;
    }
    let inverse_determinant = determinant.recip();
    Some([
        [
            c00 * inverse_determinant,
            c10 * inverse_determinant,
            c20 * inverse_determinant,
        ],
        [
            c01 * inverse_determinant,
            c11 * inverse_determinant,
            c21 * inverse_determinant,
        ],
        [
            c02 * inverse_determinant,
            c12 * inverse_determinant,
            c22 * inverse_determinant,
        ],
    ])
}
