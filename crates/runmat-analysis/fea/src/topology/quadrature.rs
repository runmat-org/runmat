use runmat_meshing_core::BoundaryTriangleOrder;

mod boundary_nodes;
mod boundary_tensor;
mod volume;
pub(crate) use volume::{
    tensor_tetrahedron_rule, visit_volume_quadrature, visit_volume_rule, volume_rule_work,
    volume_shape, volume_shape_work,
};

pub(super) use boundary_nodes::boundary_nodal_area_vectors;
pub(super) use boundary_tensor::visit as visit_boundary_tensor_rule;

use super::{
    contract::BoundaryQuadraturePoint,
    element::{cross, norm},
};

// The four-point rule is owned by meshing core with the Tet10 element-validity rule.
pub(crate) use runmat_meshing_core::TETRAHEDRON_FOUR_POINT_RULE;

// Degree-four tetrahedral rule: exact for the cubic Tet10 Jacobian volume integral.
const TETRAHEDRON_ELEVEN_A: f64 = 0.785_714_285_714_285_7;
const TETRAHEDRON_ELEVEN_B: f64 = 0.071_428_571_428_571_42;
const TETRAHEDRON_ELEVEN_C: f64 = 0.399_403_576_166_799_2;
const TETRAHEDRON_ELEVEN_D: f64 = 0.100_596_423_833_200_8;
const TETRAHEDRON_ELEVEN_VERTEX_WEIGHT: f64 = 0.007_622_222_222_222_222;
const TETRAHEDRON_ELEVEN_EDGE_WEIGHT: f64 = 0.024_888_888_888_888_89;
pub(crate) const TETRAHEDRON_ELEVEN_POINT_RULE: [([f64; 4], f64); 11] = [
    ([0.25; 4], -0.013_155_555_555_555_556),
    (
        [
            TETRAHEDRON_ELEVEN_A,
            TETRAHEDRON_ELEVEN_B,
            TETRAHEDRON_ELEVEN_B,
            TETRAHEDRON_ELEVEN_B,
        ],
        TETRAHEDRON_ELEVEN_VERTEX_WEIGHT,
    ),
    (
        [
            TETRAHEDRON_ELEVEN_B,
            TETRAHEDRON_ELEVEN_A,
            TETRAHEDRON_ELEVEN_B,
            TETRAHEDRON_ELEVEN_B,
        ],
        TETRAHEDRON_ELEVEN_VERTEX_WEIGHT,
    ),
    (
        [
            TETRAHEDRON_ELEVEN_B,
            TETRAHEDRON_ELEVEN_B,
            TETRAHEDRON_ELEVEN_A,
            TETRAHEDRON_ELEVEN_B,
        ],
        TETRAHEDRON_ELEVEN_VERTEX_WEIGHT,
    ),
    (
        [
            TETRAHEDRON_ELEVEN_B,
            TETRAHEDRON_ELEVEN_B,
            TETRAHEDRON_ELEVEN_B,
            TETRAHEDRON_ELEVEN_A,
        ],
        TETRAHEDRON_ELEVEN_VERTEX_WEIGHT,
    ),
    (
        [
            TETRAHEDRON_ELEVEN_C,
            TETRAHEDRON_ELEVEN_C,
            TETRAHEDRON_ELEVEN_D,
            TETRAHEDRON_ELEVEN_D,
        ],
        TETRAHEDRON_ELEVEN_EDGE_WEIGHT,
    ),
    (
        [
            TETRAHEDRON_ELEVEN_C,
            TETRAHEDRON_ELEVEN_D,
            TETRAHEDRON_ELEVEN_C,
            TETRAHEDRON_ELEVEN_D,
        ],
        TETRAHEDRON_ELEVEN_EDGE_WEIGHT,
    ),
    (
        [
            TETRAHEDRON_ELEVEN_C,
            TETRAHEDRON_ELEVEN_D,
            TETRAHEDRON_ELEVEN_D,
            TETRAHEDRON_ELEVEN_C,
        ],
        TETRAHEDRON_ELEVEN_EDGE_WEIGHT,
    ),
    (
        [
            TETRAHEDRON_ELEVEN_D,
            TETRAHEDRON_ELEVEN_C,
            TETRAHEDRON_ELEVEN_C,
            TETRAHEDRON_ELEVEN_D,
        ],
        TETRAHEDRON_ELEVEN_EDGE_WEIGHT,
    ),
    (
        [
            TETRAHEDRON_ELEVEN_D,
            TETRAHEDRON_ELEVEN_C,
            TETRAHEDRON_ELEVEN_D,
            TETRAHEDRON_ELEVEN_C,
        ],
        TETRAHEDRON_ELEVEN_EDGE_WEIGHT,
    ),
    (
        [
            TETRAHEDRON_ELEVEN_D,
            TETRAHEDRON_ELEVEN_D,
            TETRAHEDRON_ELEVEN_C,
            TETRAHEDRON_ELEVEN_C,
        ],
        0.024_888_888_888_89,
    ),
];

// Five-point Gauss-Legendre on [0,1] is exact through degree nine. A 5³
// Duffy product integrates curved Tet10 N_i N_j det(J) exactly (degree ≤7 on
// the tetrahedron and degree ≤9 per transformed cube coordinate).
pub(crate) const GAUSS_LEGENDRE_FIVE_UNIT: [(f64, f64); 5] = [
    (0.046_910_077_030_668, 0.118_463_442_528_094_5),
    (0.230_765_344_947_158_5, 0.239_314_335_249_683_25),
    (0.5, 0.284_444_444_444_444_44),
    (0.769_234_655_052_841_5, 0.239_314_335_249_683_25),
    (0.953_089_922_969_332, 0.118_463_442_528_094_5),
];

// Gauss-Legendre rules on [0, 1] are owned by meshing core, which also decides which rule
// integrates a curved boundary face (one face-validity authority for mesher and analysis).
pub(crate) use runmat_meshing_core::{
    GAUSS_LEGENDRE_EIGHT_UNIT, GAUSS_LEGENDRE_SIXTEEN_UNIT, GAUSS_LEGENDRE_TWELVE_UNIT,
};

const TETRAHEDRON_DERIVATIVES: [[f64; 3]; 4] = [
    [-1.0, -1.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

pub(crate) const DUNAVANT_SEVEN: [([f64; 3], f64); 7] = [
    ([1.0 / 3.0; 3], 0.225),
    (
        [0.059715871789770, 0.470142064105115, 0.470142064105115],
        0.132394152788506,
    ),
    (
        [0.470142064105115, 0.059715871789770, 0.470142064105115],
        0.132394152788506,
    ),
    (
        [0.470142064105115, 0.470142064105115, 0.059715871789770],
        0.132394152788506,
    ),
    (
        [0.797426985353087, 0.101286507323456, 0.101286507323456],
        0.125939180544827,
    ),
    (
        [0.101286507323456, 0.797426985353087, 0.101286507323456],
        0.125939180544827,
    ),
    (
        [0.101286507323456, 0.101286507323456, 0.797426985353087],
        0.125939180544827,
    ),
];

pub(crate) const KEAST_FIFTEEN: [([f64; 4], f64); 15] = [
    ([0.25; 4], 0.030283678097089),
    ([0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 0.006026785714286),
    ([1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0], 0.006026785714286),
    ([1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0], 0.006026785714286),
    ([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], 0.006026785714286),
    (
        [8.0 / 11.0, 1.0 / 11.0, 1.0 / 11.0, 1.0 / 11.0],
        0.011645249086029,
    ),
    (
        [1.0 / 11.0, 8.0 / 11.0, 1.0 / 11.0, 1.0 / 11.0],
        0.011645249086029,
    ),
    (
        [1.0 / 11.0, 1.0 / 11.0, 8.0 / 11.0, 1.0 / 11.0],
        0.011645249086029,
    ),
    (
        [1.0 / 11.0, 1.0 / 11.0, 1.0 / 11.0, 8.0 / 11.0],
        0.011645249086029,
    ),
    (
        [
            0.433449846426336,
            0.433449846426336,
            0.066550153573664,
            0.066550153573664,
        ],
        0.010949141561386,
    ),
    (
        [
            0.433449846426336,
            0.066550153573664,
            0.433449846426336,
            0.066550153573664,
        ],
        0.010949141561386,
    ),
    (
        [
            0.433449846426336,
            0.066550153573664,
            0.066550153573664,
            0.433449846426336,
        ],
        0.010949141561386,
    ),
    (
        [
            0.066550153573664,
            0.433449846426336,
            0.433449846426336,
            0.066550153573664,
        ],
        0.010949141561386,
    ),
    (
        [
            0.066550153573664,
            0.433449846426336,
            0.066550153573664,
            0.433449846426336,
        ],
        0.010949141561386,
    ),
    (
        [
            0.066550153573664,
            0.066550153573664,
            0.433449846426336,
            0.433449846426336,
        ],
        0.010949141561386,
    ),
];

pub(super) fn visit_boundary_quadrature(
    order: BoundaryTriangleOrder,
    coordinates: &[[f64; 3]],
    mut visit: impl FnMut(BoundaryQuadraturePoint),
) -> Option<()> {
    let node_count = match order {
        BoundaryTriangleOrder::Tri3 => 3,
        BoundaryTriangleOrder::Tri6 => 6,
    };
    if coordinates.len() != node_count {
        return None;
    }
    // One face-validity rule shared with the mesher's Tet10 certification: the projected
    // Jacobian's exact minimum over the face stays a fixed fraction of its largest magnitude,
    // so no published face is folded or edge-on to its corner plane.
    // Curved area and consistent nodal traction are non-polynomial; the
    // seven-point planar rule does not reliably integrate their moments. The shared
    // rule names the tensor Gauss rule on which the face's area moments converge.
    match runmat_meshing_core::boundary_face_rule(order, coordinates)? {
        runmat_meshing_core::BoundaryFaceRule::Planar => {}
        runmat_meshing_core::BoundaryFaceRule::CurvedTwelve => {
            return boundary_tensor::visit(order, coordinates, &GAUSS_LEGENDRE_TWELVE_UNIT, visit);
        }
        runmat_meshing_core::BoundaryFaceRule::CurvedSixteen => {
            return boundary_tensor::visit(order, coordinates, &GAUSS_LEGENDRE_SIXTEEN_UNIT, visit);
        }
    }
    for (barycentric, normalized_weight) in DUNAVANT_SEVEN {
        let (shape_values, derivatives) = boundary_shape(order, barycentric);
        let position_m = weighted_position(coordinates, &shape_values[..node_count]);
        let tangent_xi = weighted_derivative(coordinates, &derivatives[..node_count], 0);
        let tangent_eta = weighted_derivative(coordinates, &derivatives[..node_count], 1);
        let jacobian_vector = cross(tangent_xi, tangent_eta);
        let jacobian = norm(jacobian_vector);
        let reference_weight = 0.5 * normalized_weight;
        visit(BoundaryQuadraturePoint {
            reference_barycentric: barycentric,
            reference_weight,
            position_m,
            area_vector_weight_m2: jacobian_vector.map(|value| value * reference_weight),
            area_weight_m2: jacobian * reference_weight,
            shape_values,
            node_count,
        });
    }
    Some(())
}

pub(crate) fn boundary_shape(
    order: BoundaryTriangleOrder,
    [l1, l2, l3]: [f64; 3],
) -> ([f64; 6], [[f64; 2]; 6]) {
    match order {
        BoundaryTriangleOrder::Tri3 => (
            [l1, l2, l3, 0.0, 0.0, 0.0],
            [
                [-1.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0; 2],
                [0.0; 2],
                [0.0; 2],
            ],
        ),
        BoundaryTriangleOrder::Tri6 => (
            [
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                l3 * (2.0 * l3 - 1.0),
                4.0 * l1 * l2,
                4.0 * l2 * l3,
                4.0 * l3 * l1,
            ],
            [
                [1.0 - 4.0 * l1, 1.0 - 4.0 * l1],
                [4.0 * l2 - 1.0, 0.0],
                [0.0, 4.0 * l3 - 1.0],
                [4.0 * (l1 - l2), -4.0 * l2],
                [4.0 * l3, 4.0 * l2],
                [-4.0 * l3, 4.0 * (l1 - l3)],
            ],
        ),
    }
}

/// Boundary shape derivatives are affine for Tri6; differentiate the shared owner exactly.
pub(super) fn boundary_shape_second_derivatives(
    order: BoundaryTriangleOrder,
) -> [[[f64; 2]; 2]; 6] {
    let (_, origin) = boundary_shape(order, [1.0, 0.0, 0.0]);
    let (_, u) = boundary_shape(order, [0.0, 1.0, 0.0]);
    let (_, v) = boundary_shape(order, [0.0, 0.0, 1.0]);
    std::array::from_fn(|node| {
        std::array::from_fn(|i| [u[node][i] - origin[node][i], v[node][i] - origin[node][i]])
    })
}

pub(super) fn weighted_position(coordinates: &[[f64; 3]], weights: &[f64]) -> [f64; 3] {
    std::array::from_fn(|axis| {
        coordinates
            .iter()
            .zip(weights)
            .map(|(point, weight)| point[axis] * weight)
            .sum()
    })
}

pub(super) fn weighted_derivative(
    coordinates: &[[f64; 3]],
    derivatives: &[[f64; 2]],
    reference_axis: usize,
) -> [f64; 3] {
    std::array::from_fn(|physical_axis| {
        coordinates
            .iter()
            .zip(derivatives)
            .map(|(point, derivative)| point[physical_axis] * derivative[reference_axis])
            .sum()
    })
}

#[cfg(test)]
mod gauss_rule_tests {
    use super::{GAUSS_LEGENDRE_SIXTEEN_UNIT, GAUSS_LEGENDRE_TWELVE_UNIT};

    #[test]
    fn curved_stiffness_rules_integrate_unit_interval_moments() {
        for rule in [
            GAUSS_LEGENDRE_TWELVE_UNIT.as_slice(),
            GAUSS_LEGENDRE_SIXTEEN_UNIT.as_slice(),
        ] {
            for degree in 0..(2 * rule.len()) {
                let integral: f64 = rule
                    .iter()
                    .map(|(point, weight)| weight * point.powi(degree as i32))
                    .sum();
                let exact = 1.0 / (degree + 1) as f64;
                assert!((integral - exact).abs() < 1.0e-14);
            }
        }
    }
}

#[cfg(test)]
mod boundary_quality_tests;

/// Diagnostic for a published mesh: which boundary-face check rejects a face, with numbers.
/// Run with `RUNMAT_FACE_MESH=<solver-mesh.cbor> RUNMAT_FACE_ID=<id>`.
#[cfg(test)]
mod face_diagnostic {
    use super::*;
    use crate::topology::element::{dot, subtract};
    use runmat_meshing_core::CanonicalMeshingContract;

    #[test]
    #[ignore]
    fn report_boundary_face_checks() {
        let path = std::env::var("RUNMAT_FACE_MESH").expect("RUNMAT_FACE_MESH");
        let mesh = runmat_meshing_core::SolverMeshArtifact::canonical_decode(
            &std::fs::read(path).unwrap(),
        )
        .unwrap();
        // RUNMAT_FACE_ID=all integrates every boundary face and reports the ones it rejects.
        if std::env::var("RUNMAT_FACE_ID").as_deref() == Ok("all") {
            let position = |id: &u64| {
                mesh.topology
                    .nodes
                    .binary_search_by_key(id, |node| node.node_id)
                    .map(|index| mesh.topology.nodes[index].coordinates_m.canonical_value())
                    .unwrap()
            };
            let rejected = mesh
                .topology
                .boundary_faces
                .iter()
                .filter(|face| {
                    let coordinates = face.node_ids.iter().map(position).collect::<Vec<_>>();
                    visit_boundary_quadrature(face.order, &coordinates, |_| {}).is_none()
                })
                .map(|face| face.face_id)
                .collect::<Vec<_>>();
            eprintln!(
                "{} of {} boundary faces rejected: {rejected:?}",
                rejected.len(),
                mesh.topology.boundary_faces.len()
            );
            return;
        }
        let face_id: u64 = std::env::var("RUNMAT_FACE_ID").unwrap().parse().unwrap();
        let face = mesh
            .topology
            .boundary_faces
            .iter()
            .find(|face| face.face_id == face_id)
            .expect("face");
        let coordinates = face
            .node_ids
            .iter()
            .map(|id| {
                mesh.topology
                    .nodes
                    .iter()
                    .find(|node| node.node_id == *id)
                    .unwrap()
                    .coordinates_m
                    .canonical_value()
            })
            .collect::<Vec<_>>();
        let corner_normal = cross(
            subtract(coordinates[1], coordinates[0]),
            subtract(coordinates[2], coordinates[0]),
        );
        eprintln!(
            "face {face_id} order {:?} nodes {coordinates:?}",
            face.order
        );
        eprintln!("corner normal norm {:e}", norm(corner_normal));
        if face.order == BoundaryTriangleOrder::Tri6 {
            eprintln!(
                "shared face validity {:?} valid {}",
                runmat_meshing_core::boundary_face_validity(face.order, &coordinates),
                runmat_meshing_core::is_valid_boundary_face(face.order, &coordinates)
            );
            eprintln!(
                "shared face rule {:?}",
                runmat_meshing_core::boundary_face_rule(face.order, &coordinates)
            );
        }
        for (barycentric, _) in DUNAVANT_SEVEN {
            let (_, derivatives) = boundary_shape(face.order, barycentric);
            let n = coordinates.len();
            let tangent_xi = weighted_derivative(&coordinates, &derivatives[..n], 0);
            let tangent_eta = weighted_derivative(&coordinates, &derivatives[..n], 1);
            let jacobian = cross(tangent_xi, tangent_eta);
            eprintln!(
                "point {barycentric:?} jacobian {:e} along corner normal {:e}",
                norm(jacobian),
                dot(jacobian, corner_normal)
            );
        }
        eprintln!(
            "visit_boundary_quadrature: {}",
            visit_boundary_quadrature(face.order, &coordinates, |_| {}).is_some()
        );
    }
}
