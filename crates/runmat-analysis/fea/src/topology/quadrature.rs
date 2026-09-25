use runmat_meshing_core::{BoundaryTriangleOrder, ElementOrder, TETRAHEDRON_MIDSIDE_EDGE_CORNERS};

use super::{
    contract::{BoundaryQuadraturePoint, VolumeQuadraturePoint},
    element::{cross, dot, invert, norm},
};

pub(crate) const TETRAHEDRON_FOUR_POINT_RULE: [[f64; 4]; 4] = [
    [
        0.585_410_196_624_968_5,
        0.138_196_601_125_010_5,
        0.138_196_601_125_010_5,
        0.138_196_601_125_010_5,
    ],
    [
        0.138_196_601_125_010_5,
        0.585_410_196_624_968_5,
        0.138_196_601_125_010_5,
        0.138_196_601_125_010_5,
    ],
    [
        0.138_196_601_125_010_5,
        0.138_196_601_125_010_5,
        0.585_410_196_624_968_5,
        0.138_196_601_125_010_5,
    ],
    [
        0.138_196_601_125_010_5,
        0.138_196_601_125_010_5,
        0.138_196_601_125_010_5,
        0.585_410_196_624_968_5,
    ],
];

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

// Eight-point Gauss-Legendre on [0,1] for nonpolynomial curved Tet10 stiffness.
pub(crate) const GAUSS_LEGENDRE_EIGHT_UNIT: [(f64, f64); 8] = [
    (0.019_855_071_751_231_85, 0.050_614_268_145_188_13),
    (0.101_666_761_293_186_65, 0.111_190_517_226_687_24),
    (0.237_233_795_041_835_5, 0.156_853_322_938_943_64),
    (0.408_282_678_752_175_1, 0.181_341_891_689_181),
    (0.591_717_321_247_825, 0.181_341_891_689_181),
    (0.762_766_204_958_164_5, 0.156_853_322_938_943_64),
    (0.898_333_238_706_813_4, 0.111_190_517_226_687_24),
    (0.980_144_928_248_768_1, 0.050_614_268_145_188_13),
];

const TETRAHEDRON_DERIVATIVES: [[f64; 3]; 4] = [
    [-1.0, -1.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

const DUNAVANT_SEVEN: [([f64; 3], f64); 7] = [
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

const KEAST_FIFTEEN: [([f64; 4], f64); 15] = [
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
    for (barycentric, normalized_weight) in DUNAVANT_SEVEN {
        let (shape_values, derivatives) = boundary_shape(order, barycentric);
        let position_m = weighted_position(coordinates, &shape_values[..node_count]);
        let tangent_xi = weighted_derivative(coordinates, &derivatives[..node_count], 0);
        let tangent_eta = weighted_derivative(coordinates, &derivatives[..node_count], 1);
        let jacobian_vector = cross(tangent_xi, tangent_eta);
        let jacobian = norm(jacobian_vector);
        if !jacobian.is_finite() || jacobian <= 0.0 {
            return None;
        }
        let reference_weight = 0.5 * normalized_weight;
        visit(BoundaryQuadraturePoint {
            position_m,
            area_vector_weight_m2: jacobian_vector.map(|value| value * reference_weight),
            area_weight_m2: jacobian * reference_weight,
            shape_values,
            node_count,
        });
    }
    Some(())
}

fn boundary_shape(
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

fn weighted_position(coordinates: &[[f64; 3]], weights: &[f64]) -> [f64; 3] {
    std::array::from_fn(|axis| {
        coordinates
            .iter()
            .zip(weights)
            .map(|(point, weight)| point[axis] * weight)
            .sum()
    })
}

fn weighted_derivative(
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

pub(super) fn visit_volume_quadrature(
    order: ElementOrder,
    coordinates: &[[f64; 3]],
    mut visit: impl FnMut(VolumeQuadraturePoint),
) -> Option<()> {
    let node_count = match order {
        ElementOrder::Tet4 => 4,
        ElementOrder::Tet10 => 10,
    };
    if coordinates.len() != node_count {
        return None;
    }
    for (barycentric, reference_weight) in KEAST_FIFTEEN {
        let (shape_values, derivatives) = volume_shape(order, barycentric);
        let jacobian = std::array::from_fn::<_, 3, _>(|reference_axis| {
            std::array::from_fn(|physical_axis| {
                coordinates
                    .iter()
                    .zip(&derivatives[..node_count])
                    .map(|(node, derivative)| node[physical_axis] * derivative[reference_axis])
                    .sum::<f64>()
            })
        });
        let determinant = dot(jacobian[0], cross(jacobian[1], jacobian[2]));
        if !determinant.is_finite() || determinant <= 0.0 {
            return None;
        }
        let inverse = invert(jacobian)?;
        let mut physical_shape_gradients = [[0.0; 3]; 10];
        for node in 0..node_count {
            physical_shape_gradients[node] = std::array::from_fn(|physical_axis| {
                (0..3)
                    .map(|reference_axis| {
                        inverse[physical_axis][reference_axis] * derivatives[node][reference_axis]
                    })
                    .sum()
            });
        }
        visit(VolumeQuadraturePoint {
            position_m: weighted_position(coordinates, &shape_values[..node_count]),
            volume_weight_m3: reference_weight * determinant,
            shape_values,
            physical_shape_gradients,
            node_count,
        });
    }
    Some(())
}

pub(super) fn volume_shape(
    order: ElementOrder,
    barycentric: [f64; 4],
) -> ([f64; 10], [[f64; 3]; 10]) {
    let mut values = [0.0; 10];
    let mut derivatives = [[0.0; 3]; 10];
    if order == ElementOrder::Tet4 {
        values[..4].copy_from_slice(&barycentric);
        derivatives[..4].copy_from_slice(&TETRAHEDRON_DERIVATIVES);
        return (values, derivatives);
    }
    for node in 0..4 {
        values[node] = barycentric[node] * (2.0 * barycentric[node] - 1.0);
        derivatives[node] = TETRAHEDRON_DERIVATIVES[node]
            .map(|derivative| (4.0 * barycentric[node] - 1.0) * derivative);
    }
    for (edge, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
        values[4 + edge] = 4.0 * barycentric[left] * barycentric[right];
        derivatives[4 + edge] = std::array::from_fn(|axis| {
            4.0 * (barycentric[right] * TETRAHEDRON_DERIVATIVES[left][axis]
                + barycentric[left] * TETRAHEDRON_DERIVATIVES[right][axis])
        });
    }
    (values, derivatives)
}
