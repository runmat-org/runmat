use super::{StructuralRecoveryEdge, TENSOR_COMPONENT_COUNT, VECTOR_COMPONENT_COUNT};

pub(super) fn edge_strain_tensor(displacement: &[f64], edge: &StructuralRecoveryEdge) -> [f64; 6] {
    let length = if edge.edge_length_m.is_finite() && edge.edge_length_m > 0.0 {
        edge.edge_length_m
    } else {
        edge.hop.max(1) as f64
    };
    let from_node = edge.from_dof / VECTOR_COMPONENT_COUNT;
    let to_node = edge.to_dof / VECTOR_COMPONENT_COUNT;
    if from_node == to_node {
        let jump = displacement.get(edge.to_dof).copied().unwrap_or(0.0)
            - displacement.get(edge.from_dof).copied().unwrap_or(0.0);
        let mut tensor = [0.0; TENSOR_COMPONENT_COUNT];
        tensor[edge.component] = jump / length;
        return tensor;
    }

    let du = nodal_displacement(displacement, to_node);
    let u0 = nodal_displacement(displacement, from_node);
    let gradient = [
        (du[0] - u0[0]) / length,
        (du[1] - u0[1]) / length,
        (du[2] - u0[2]) / length,
    ];
    [
        gradient[0],
        gradient[1],
        gradient[2],
        0.5 * (gradient[0] + gradient[1]),
        0.5 * (gradient[1] + gradient[2]),
        0.5 * (gradient[0] + gradient[2]),
    ]
}

fn nodal_displacement(displacement: &[f64], node: usize) -> [f64; VECTOR_COMPONENT_COUNT] {
    let base = node * VECTOR_COMPONENT_COUNT;
    [
        displacement.get(base).copied().unwrap_or(0.0),
        displacement.get(base + 1).copied().unwrap_or(0.0),
        displacement.get(base + 2).copied().unwrap_or(0.0),
    ]
}
