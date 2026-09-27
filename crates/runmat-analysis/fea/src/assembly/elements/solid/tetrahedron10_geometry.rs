//! Isoparametric Tet10 geometry, gradients, and whole-element Jacobian admission.

use runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;

use super::{
    jacobian_certificate::{certify_positive_jacobian, CertificateFailure},
    tetrahedron10::{Tetrahedron10ElementError, TETRAHEDRON10_ELEMENT_NODE_COUNT},
};

const BARYCENTRIC_DERIVATIVES: [[f64; 3]; 4] = [
    [-1.0, -1.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

pub(super) fn physical_shape_gradients_and_measure(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
    barycentric: [f64; 4],
) -> Result<([[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT], f64), Tetrahedron10ElementError> {
    let reference = reference_shape_gradients(barycentric);
    let (inverse, determinant) = inverse_jacobian(nodes_m, reference)?;
    let gradients = reference.map(|gradient| {
        std::array::from_fn(|axis| {
            inverse[axis][0] * gradient[0]
                + inverse[axis][1] * gradient[1]
                + inverse[axis][2] * gradient[2]
        })
    });
    Ok((gradients, determinant))
}

pub(super) fn reference_shape_gradients(
    barycentric: [f64; 4],
) -> [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT] {
    let mut gradients = [[0.0; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT];
    for node in 0..4 {
        let factor = 4.0 * barycentric[node] - 1.0;
        gradients[node] = BARYCENTRIC_DERIVATIVES[node].map(|derivative| factor * derivative);
    }
    for (local_edge, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
        gradients[4 + local_edge] = std::array::from_fn(|axis| {
            4.0 * (barycentric[right] * BARYCENTRIC_DERIVATIVES[left][axis]
                + barycentric[left] * BARYCENTRIC_DERIVATIVES[right][axis])
        });
    }
    gradients
}

pub(super) fn jacobian_determinant(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
    barycentric: [f64; 4],
) -> Result<f64, Tetrahedron10ElementError> {
    let reference = reference_shape_gradients(barycentric);
    let jacobian = jacobian(nodes_m, reference);
    let determinant = dot(jacobian[0], cross(jacobian[1], jacobian[2]));
    if determinant.is_finite() && determinant > 0.0 {
        Ok(determinant)
    } else {
        Err(Tetrahedron10ElementError::DegenerateOrInverted)
    }
}

fn inverse_jacobian(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
    reference: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
) -> Result<([[f64; 3]; 3], f64), Tetrahedron10ElementError> {
    let matrix = jacobian(nodes_m, reference);
    let determinant = dot(matrix[0], cross(matrix[1], matrix[2]));
    if !determinant.is_finite() || determinant <= 0.0 {
        return Err(Tetrahedron10ElementError::DegenerateOrInverted);
    }
    let inverse_determinant = 1.0 / determinant;
    Ok((
        [
            [
                (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inverse_determinant,
                (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inverse_determinant,
                (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inverse_determinant,
            ],
            [
                (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inverse_determinant,
                (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inverse_determinant,
                (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inverse_determinant,
            ],
            [
                (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inverse_determinant,
                (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inverse_determinant,
                (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inverse_determinant,
            ],
        ],
        determinant,
    ))
}

fn jacobian(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
    reference: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
) -> [[f64; 3]; 3] {
    std::array::from_fn(|reference_axis| {
        std::array::from_fn(|physical_axis| {
            (0..TETRAHEDRON10_ELEMENT_NODE_COUNT)
                .map(|node| nodes_m[node][physical_axis] * reference[node][reference_axis])
                .sum()
        })
    })
}

pub(super) fn validate_nodes(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
) -> Result<(), Tetrahedron10ElementError> {
    if !nodes_m.iter().flatten().all(|value| value.is_finite()) {
        return Err(Tetrahedron10ElementError::NonFiniteCoordinate);
    }
    let vertex_jacobians = std::array::from_fn(|vertex| {
        let mut barycentric = [0.0; 4];
        barycentric[vertex] = 1.0;
        jacobian(nodes_m, reference_shape_gradients(barycentric))
    });
    certify_positive_jacobian(vertex_jacobians).map_err(|failure| match failure {
        CertificateFailure::InvertedOrDegenerate => Tetrahedron10ElementError::DegenerateOrInverted,
        CertificateFailure::Inconclusive => Tetrahedron10ElementError::UncertifiedJacobian,
    })
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
