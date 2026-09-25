use runmat_meshing_core::{BoundaryTriangleOrder, ElementOrder, TETRAHEDRON_MIDSIDE_EDGE_CORNERS};

mod boundary_nodes;

pub(super) use boundary_nodes::boundary_nodal_area_vectors;

use super::{
    contract::{BoundaryQuadraturePoint, VolumeQuadraturePoint},
    element::{cross, dot, invert, norm, subtract},
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

pub(crate) const GAUSS_LEGENDRE_TWELVE_UNIT: [(f64, f64); 12] = [
    (0.009_219_682_876_640_378, 0.023_587_668_193_255_64),
    (0.047_941_371_814_762_6, 0.053_469_662_997_659_38),
    (0.115_048_662_902_847_65, 0.080_039_164_271_673_18),
    (0.206_341_022_856_691_26, 0.101_583_713_361_532_95),
    (0.316_084_250_500_909_94, 0.116_746_268_269_177_44),
    (0.437_383_295_744_265_54, 0.124_573_522_906_701_48),
    (0.562_616_704_255_734_5, 0.124_573_522_906_701_48),
    (0.683_915_749_499_09, 0.116_746_268_269_177_44),
    (0.793_658_977_143_308_7, 0.101_583_713_361_532_95),
    (0.884_951_337_097_152_4, 0.080_039_164_271_673_18),
    (0.952_058_628_185_237_4, 0.053_469_662_997_659_38),
    (0.990_780_317_123_359_6, 0.023_587_668_193_255_64),
];

pub(crate) const GAUSS_LEGENDRE_SIXTEEN_UNIT: [(f64, f64); 16] = [
    (0.005_299_532_504_175_031, 0.013_576_229_705_877_027),
    (0.027_712_488_463_383_7, 0.031_126_761_969_323_805),
    (0.067_184_398_806_084_18, 0.047_579_255_841_246_5),
    (0.122_297_795_822_498_45, 0.062_314_485_627_766_97),
    (0.191_061_877_798_678_1, 0.074_797_994_408_288_33),
    (0.270_991_611_171_386_3, 0.084_578_259_697_501_31),
    (0.359_198_224_610_370_54, 0.091_301_707_522_461_8),
    (0.452_493_745_081_181_3, 0.094_725_305_227_534_3),
    (0.547_506_254_918_818_8, 0.094_725_305_227_534_3),
    (0.640_801_775_389_629_5, 0.091_301_707_522_461_8),
    (0.729_008_388_828_613_6, 0.084_578_259_697_501_31),
    (0.808_938_122_201_321_9, 0.074_797_994_408_288_33),
    (0.877_702_204_177_501_5, 0.062_314_485_627_766_97),
    (0.932_815_601_193_915_8, 0.047_579_255_841_246_5),
    (0.972_287_511_536_616_3, 0.031_126_761_969_323_805),
    (0.994_700_467_495_825, 0.013_576_229_705_877_027),
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
    // A Tri6 map has linear tangents, so its signed area projected onto the
    // corner normal is quadratic over the reference triangle. Checking only
    // quadrature points misses folds near a corner or edge.
    let corner_normal = cross(
        subtract(coordinates[1], coordinates[0]),
        subtract(coordinates[2], coordinates[0]),
    );
    if !norm(corner_normal).is_finite() || norm(corner_normal) <= 0.0 {
        return None;
    }
    if order == BoundaryTriangleOrder::Tri6
        && signed_boundary_area_minimum(coordinates, corner_normal)? <= 0.0
    {
        return None;
    }
    for (barycentric, normalized_weight) in DUNAVANT_SEVEN {
        let (shape_values, derivatives) = boundary_shape(order, barycentric);
        let position_m = weighted_position(coordinates, &shape_values[..node_count]);
        let tangent_xi = weighted_derivative(coordinates, &derivatives[..node_count], 0);
        let tangent_eta = weighted_derivative(coordinates, &derivatives[..node_count], 1);
        let jacobian_vector = cross(tangent_xi, tangent_eta);
        let jacobian = norm(jacobian_vector);
        if !jacobian.is_finite() || jacobian <= 0.0 || dot(jacobian_vector, corner_normal) <= 0.0 {
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

fn signed_boundary_area_minimum(coordinates: &[[f64; 3]], corner_normal: [f64; 3]) -> Option<f64> {
    let signed_area = |u: f64, v: f64| {
        let (_, derivatives) = boundary_shape(BoundaryTriangleOrder::Tri6, [1.0 - u - v, u, v]);
        let tangent_u = weighted_derivative(coordinates, &derivatives, 0);
        let tangent_v = weighted_derivative(coordinates, &derivatives, 1);
        dot(cross(tangent_u, tangent_v), corner_normal)
    };
    let [q00, q10, q01, qh0, q0h, qhh] = [
        signed_area(0.0, 0.0),
        signed_area(1.0, 0.0),
        signed_area(0.0, 1.0),
        signed_area(0.5, 0.0),
        signed_area(0.0, 0.5),
        signed_area(0.5, 0.5),
    ];
    if [q00, q10, q01, qh0, q0h, qhh]
        .iter()
        .any(|value| !value.is_finite())
    {
        return None;
    }
    // q(u,v) = a*u² + b*v² + c*u*v + d*u + e*v + f.
    let a = 2.0 * (q10 + q00 - 2.0 * qh0);
    let b = 2.0 * (q01 + q00 - 2.0 * q0h);
    let d = q10 - q00 - a;
    let e = q01 - q00 - b;
    let c = 4.0 * (qhh - 0.25 * (a + b) - 0.5 * (d + e) - q00);
    if [a, b, c, d, e].iter().any(|value| !value.is_finite()) {
        return None;
    }
    let edge_minima = [
        quadratic_edge_minimum(q00, q10, qh0),
        quadratic_edge_minimum(q00, q01, q0h),
        quadratic_edge_minimum(q10, q01, qhh),
    ];
    if edge_minima.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut minimum = edge_minima.into_iter().fold(f64::INFINITY, f64::min);
    // A strict interior minimum exists only for a positive-definite Hessian.
    let determinant = 4.0 * a * b - c * c;
    if !determinant.is_finite() {
        return None;
    }
    if a > 0.0 && determinant > 0.0 {
        let u = (c * e - 2.0 * b * d) / determinant;
        let v = (c * d - 2.0 * a * e) / determinant;
        if !u.is_finite() || !v.is_finite() {
            return None;
        }
        if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
            let interior = a * u * u + b * v * v + c * u * v + d * u + e * v + q00;
            if !interior.is_finite() {
                return None;
            }
            minimum = minimum.min(interior);
        }
    }
    minimum.is_finite().then_some(minimum)
}

fn quadratic_edge_minimum(start: f64, end: f64, midpoint: f64) -> f64 {
    let curvature = 2.0 * (start + end - 2.0 * midpoint);
    if curvature <= 0.0 {
        return start.min(end);
    }
    let slope = end - start - curvature;
    let position = (-slope / (2.0 * curvature)).clamp(0.0, 1.0);
    (curvature * position + slope) * position + start
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
