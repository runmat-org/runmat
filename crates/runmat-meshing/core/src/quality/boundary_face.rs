//! One validity rule for a published boundary triangle, shared by the mesher's Tet10
//! certification and every analysis consumer that integrates over the face.
//!
//! A six-node face maps the reference triangle by quadratic shape functions; its area
//! Jacobian J(u, v) = dX/du x dX/dv is quadratic. The face is valid when its Jacobian,
//! projected on the unit corner normal, stays at least
//! [`BOUNDARY_FACE_MINIMUM_NORMAL_JACOBIAN_RATIO`] of the largest Jacobian magnitude over the
//! face: the projected minimum is the exact minimum of that quadratic over the triangle, and
//! the magnitude maximum is taken over a fixed sample set (nodes, centroid and the seven
//! Dunavant points). A face that is folded, or so nearly edge-on to its own corner plane that
//! curved quadrature cannot converge on it, is invalid. A curved face is admitted only when its
//! area moments converge under the tensor Gauss rules that consumers integrate it with
//! ([`boundary_face_rule`]). A three-node face is valid when its corner normal is finite and
//! nonzero.
use crate::BoundaryTriangleOrder;

/// Smallest admitted ratio of the projected Jacobian to the largest Jacobian magnitude: the
/// surface may turn at most about 87 degrees from its corner plane anywhere on the face.
pub const BOUNDARY_FACE_MINIMUM_NORMAL_JACOBIAN_RATIO: f64 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryFaceValidity {
    /// Exact minimum over the face of the area Jacobian projected on the unit corner normal.
    pub minimum_normal_jacobian: f64,
    /// Largest area-Jacobian magnitude over the fixed sample set.
    pub maximum_jacobian: f64,
}

impl BoundaryFaceValidity {
    pub fn is_valid(&self) -> bool {
        self.minimum_normal_jacobian > 0.0
            && self.minimum_normal_jacobian
                >= BOUNDARY_FACE_MINIMUM_NORMAL_JACOBIAN_RATIO * self.maximum_jacobian
    }
}

/// The face's validity measures, or `None` for a malformed face (wrong node count,
/// nonfinite coordinates, or a zero corner normal).
pub fn boundary_face_validity(
    order: BoundaryTriangleOrder,
    coordinates: &[[f64; 3]],
) -> Option<BoundaryFaceValidity> {
    let count = match order {
        BoundaryTriangleOrder::Tri3 => 3,
        BoundaryTriangleOrder::Tri6 => 6,
    };
    if coordinates.len() != count || coordinates.iter().flatten().any(|value| !value.is_finite()) {
        return None;
    }
    let corner_normal = cross(
        subtract(coordinates[1], coordinates[0]),
        subtract(coordinates[2], coordinates[0]),
    );
    let corner_area = norm(corner_normal);
    if !corner_area.is_finite() || corner_area <= 0.0 {
        return None;
    }
    if order == BoundaryTriangleOrder::Tri3 {
        return Some(BoundaryFaceValidity {
            minimum_normal_jacobian: corner_area,
            maximum_jacobian: corner_area,
        });
    }
    let unit_normal = corner_normal.map(|value| value / corner_area);
    let jacobian = |u: f64, v: f64| tri6_area_jacobian(coordinates, u, v);
    let minimum_normal_jacobian =
        quadratic_triangle_minimum(|u, v| dot(jacobian(u, v), unit_normal))?;
    let maximum_jacobian = SAMPLE_POINTS
        .iter()
        .map(|[u, v]| norm(jacobian(*u, *v)))
        .fold(0.0_f64, f64::max);
    (minimum_normal_jacobian.is_finite() && maximum_jacobian.is_finite()).then_some(
        BoundaryFaceValidity {
            minimum_normal_jacobian,
            maximum_jacobian,
        },
    )
}

/// How an admitted face is integrated: the planar seven-point rule, or the tensor
/// Gauss-Legendre rule on which a curved face's area moments converged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryFaceRule {
    Planar,
    CurvedTwelve,
    CurvedSixteen,
}

/// The face's integration rule, or `None` when the face is not admitted: malformed, below the
/// projected-Jacobian margin, or curved such that its area moments (area, nodal areas, first
/// moment, area vector) do not agree to a relative 1e-10 between successive tensor
/// Gauss-Legendre rules (8 against 12, then 12 against 16). This is the single admission
/// authority for published boundary faces.
pub fn boundary_face_rule(
    order: BoundaryTriangleOrder,
    coordinates: &[[f64; 3]],
) -> Option<BoundaryFaceRule> {
    if !boundary_face_validity(order, coordinates)?.is_valid() {
        return None;
    }
    if order == BoundaryTriangleOrder::Tri3 || !is_curved(coordinates) {
        return Some(BoundaryFaceRule::Planar);
    }
    let eight = moments(coordinates, &GAUSS_LEGENDRE_EIGHT_UNIT)?;
    let twelve = moments(coordinates, &GAUSS_LEGENDRE_TWELVE_UNIT)?;
    if moments_converged(&eight, &twelve, coordinates) {
        return Some(BoundaryFaceRule::CurvedTwelve);
    }
    let sixteen = moments(coordinates, &GAUSS_LEGENDRE_SIXTEEN_UNIT)?;
    moments_converged(&twelve, &sixteen, coordinates).then_some(BoundaryFaceRule::CurvedSixteen)
}

/// True when the face is admitted by [`boundary_face_rule`].
pub fn is_valid_boundary_face(order: BoundaryTriangleOrder, coordinates: &[[f64; 3]]) -> bool {
    boundary_face_rule(order, coordinates).is_some()
}

// Eight-point Gauss-Legendre on [0,1] for nonpolynomial curved Tet10 stiffness.
pub const GAUSS_LEGENDRE_EIGHT_UNIT: [(f64, f64); 8] = [
    (0.019_855_071_751_231_85, 0.050_614_268_145_188_13),
    (0.101_666_761_293_186_65, 0.111_190_517_226_687_24),
    (0.237_233_795_041_835_5, 0.156_853_322_938_943_64),
    (0.408_282_678_752_175_1, 0.181_341_891_689_181),
    (0.591_717_321_247_825, 0.181_341_891_689_181),
    (0.762_766_204_958_164_5, 0.156_853_322_938_943_64),
    (0.898_333_238_706_813_4, 0.111_190_517_226_687_24),
    (0.980_144_928_248_768_1, 0.050_614_268_145_188_13),
];

pub const GAUSS_LEGENDRE_TWELVE_UNIT: [(f64, f64); 12] = [
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

pub const GAUSS_LEGENDRE_SIXTEEN_UNIT: [(f64, f64); 16] = [
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

const MOMENT_RELATIVE_TOLERANCE: f64 = 1.0e-10;

fn is_curved(coordinates: &[[f64; 3]]) -> bool {
    let scale = (0..3)
        .map(|index| norm(subtract(coordinates[(index + 1) % 3], coordinates[index])))
        .fold(0.0_f64, f64::max);
    [(0, 1), (1, 2), (2, 0)]
        .iter()
        .enumerate()
        .any(|(index, edge)| {
            let midpoint: [f64; 3] = std::array::from_fn(|axis| {
                0.5 * (coordinates[edge.0][axis] + coordinates[edge.1][axis])
            });
            norm(subtract(coordinates[index + 3], midpoint)) > 1.0e-12 * scale
        })
}

#[derive(Default)]
struct Moments {
    area: f64,
    nodal_area: [f64; 6],
    first_moment: [f64; 3],
    area_vector: [f64; 3],
}

/// Collapsed-tensor (Duffy) integration of the area moments with a Gauss-Legendre rule.
fn moments(coordinates: &[[f64; 3]], rule: &[(f64, f64)]) -> Option<Moments> {
    let mut result = Moments::default();
    for &(r, wr) in rule {
        for &(s, ws) in rule {
            let (u, v) = (r, (1.0 - r) * s);
            let l1 = 1.0 - u - v;
            let shapes = [
                l1 * (2.0 * l1 - 1.0),
                u * (2.0 * u - 1.0),
                v * (2.0 * v - 1.0),
                4.0 * l1 * u,
                4.0 * u * v,
                4.0 * v * l1,
            ];
            let area_vector = tri6_area_jacobian(coordinates, u, v);
            let jacobian = norm(area_vector);
            if !jacobian.is_finite() || jacobian <= 0.0 {
                return None;
            }
            let weight = wr * ws * (1.0 - r);
            result.area += jacobian * weight;
            for (sum, shape) in result.nodal_area.iter_mut().zip(shapes) {
                *sum += shape * jacobian * weight;
            }
            for (axis, component) in area_vector.iter().enumerate() {
                let position = (0..6)
                    .map(|node| shapes[node] * coordinates[node][axis])
                    .sum::<f64>();
                result.first_moment[axis] += position * jacobian * weight;
                result.area_vector[axis] += component * weight;
            }
        }
    }
    result.area.is_finite().then_some(result)
}

fn moments_converged(previous: &Moments, refined: &Moments, coordinates: &[[f64; 3]]) -> bool {
    let area_scale = previous.area.abs().max(refined.area.abs());
    let length_scale = coordinates
        .iter()
        .map(|point| norm(subtract(*point, coordinates[0])))
        .fold(0.0_f64, f64::max);
    let close = |left: f64, right: f64, scale: f64| {
        left.is_finite()
            && right.is_finite()
            && (left - right).abs() <= MOMENT_RELATIVE_TOLERANCE * scale
    };
    close(previous.area, refined.area, area_scale)
        && previous
            .nodal_area
            .iter()
            .zip(refined.nodal_area)
            .all(|(left, right)| close(*left, right, area_scale))
        && previous
            .first_moment
            .iter()
            .zip(refined.first_moment)
            .all(|(left, right)| close(*left, right, area_scale * length_scale))
        && previous
            .area_vector
            .iter()
            .zip(refined.area_vector)
            .all(|(left, right)| close(*left, right, area_scale))
}

/// Reference points (u, v) = (l2, l3): nodes, centroid and the seven Dunavant points.
const SAMPLE_POINTS: [[f64; 2]; 13] = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.0],
    [0.5, 0.5],
    [0.0, 0.5],
    [1.0 / 3.0, 1.0 / 3.0],
    [0.059_715_871_789_770, 0.470_142_064_105_115],
    [0.470_142_064_105_115, 0.059_715_871_789_770],
    [0.470_142_064_105_115, 0.470_142_064_105_115],
    [0.797_426_985_353_087, 0.101_286_507_323_456],
    [0.101_286_507_323_456, 0.797_426_985_353_087],
    [0.101_286_507_323_456, 0.101_286_507_323_456],
];

/// dX/du x dX/dv of the six-node map (corners 0..3, midsides 01, 12, 20) at (u, v) = (l2, l3).
fn tri6_area_jacobian(coordinates: &[[f64; 3]], u: f64, v: f64) -> [f64; 3] {
    let l1 = 1.0 - u - v;
    let derivatives: [[f64; 2]; 6] = [
        [1.0 - 4.0 * l1, 1.0 - 4.0 * l1],
        [4.0 * u - 1.0, 0.0],
        [0.0, 4.0 * v - 1.0],
        [4.0 * (l1 - u), -4.0 * u],
        [4.0 * v, 4.0 * u],
        [-4.0 * v, 4.0 * (l1 - v)],
    ];
    let tangent = |axis: usize| -> [f64; 3] {
        std::array::from_fn(|component| {
            (0..6)
                .map(|node| derivatives[node][axis] * coordinates[node][component])
                .sum::<f64>()
        })
    };
    cross(tangent(0), tangent(1))
}

/// Exact minimum over the reference triangle of a quadratic q(u, v), from its six nodal values.
fn quadratic_triangle_minimum(q: impl Fn(f64, f64) -> f64) -> Option<f64> {
    let [q00, q10, q01, qh0, q0h, qhh] = [
        q(0.0, 0.0),
        q(1.0, 0.0),
        q(0.0, 1.0),
        q(0.5, 0.0),
        q(0.0, 0.5),
        q(0.5, 0.5),
    ];
    if [q00, q10, q01, qh0, q0h, qhh]
        .iter()
        .any(|value| !value.is_finite())
    {
        return None;
    }
    // q(u,v) = a*u² + b*v² + c*u*v + d*u + e*v + q00.
    let a = 2.0 * (q10 + q00 - 2.0 * qh0);
    let b = 2.0 * (q01 + q00 - 2.0 * q0h);
    let d = q10 - q00 - a;
    let e = q01 - q00 - b;
    let c = 4.0 * (qhh - 0.25 * (a + b) - 0.5 * (d + e) - q00);
    if [a, b, c, d, e].iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut minimum = quadratic_edge_minimum(q00, q10, qh0)
        .min(quadratic_edge_minimum(q00, q01, q0h))
        .min(quadratic_edge_minimum(q10, q01, qhh));
    // A strict interior minimum exists only for a positive-definite Hessian.
    let determinant = 4.0 * a * b - c * c;
    if a > 0.0 && determinant > 0.0 {
        let u = (c * e - 2.0 * b * d) / determinant;
        let v = (c * d - 2.0 * a * e) / determinant;
        if u.is_finite() && v.is_finite() && u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
            minimum = minimum.min(a * u * u + b * v * v + c * u * v + d * u + e * v + q00);
        }
    }
    minimum.is_finite().then_some(minimum)
}

/// Minimum over [0, 1] of the quadratic through (0, start), (1/2, midpoint), (1, end).
fn quadratic_edge_minimum(start: f64, end: f64, midpoint: f64) -> f64 {
    let curvature = 2.0 * (start + end - 2.0 * midpoint);
    if curvature <= 0.0 {
        return start.min(end);
    }
    let slope = end - start - curvature;
    let position = (-slope / (2.0 * curvature)).clamp(0.0, 1.0);
    (curvature * position + slope) * position + start
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}
fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}
fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}
fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat() -> [[f64; 3]; 6] {
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ]
    }

    // witness: FEA-MESH-DISCRETIZATION-001/volume_quality_convergence @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
    //   boundary: The shared boundary-face rule admits flat and gently curved six-node faces
    //   and rejects folded faces, faces nearly edge-on to their corner plane, and malformed
    //   faces; the bracket Tet10 study exercises publication.
    #[test]
    fn shared_face_rule_rejects_folded_and_edge_on_faces() {
        assert!(is_valid_boundary_face(BoundaryTriangleOrder::Tri6, &flat()));
        let mut curved = flat();
        curved[4][2] = 0.1;
        assert!(is_valid_boundary_face(BoundaryTriangleOrder::Tri6, &curved));
        // A fold missed by every Dunavant point: q(u, v) = 2.04 - 2.08u - 1.04v.
        let mut folded = flat();
        folded[3][0] = 0.76;
        let validity = boundary_face_validity(BoundaryTriangleOrder::Tri6, &folded).unwrap();
        assert!((validity.minimum_normal_jacobian + 0.04).abs() < 1.0e-14);
        assert!(!validity.is_valid());
        // A sliver whose midsides stand it nearly edge-on to its tiny corner plane.
        let mut edge_on = flat();
        edge_on[2] = [0.5, 1.0e-6, 0.0];
        edge_on[4] = [0.75, 0.0, 0.3];
        edge_on[5] = [0.25, 0.0, 0.3];
        assert!(!is_valid_boundary_face(
            BoundaryTriangleOrder::Tri6,
            &edge_on
        ));
        assert!(boundary_face_validity(BoundaryTriangleOrder::Tri6, &flat()[..3]).is_none());
        let mut collapsed = flat();
        collapsed[2] = [0.5, 0.0, 0.0];
        assert!(!is_valid_boundary_face(
            BoundaryTriangleOrder::Tri6,
            &collapsed
        ));
        assert!(is_valid_boundary_face(
            BoundaryTriangleOrder::Tri3,
            &flat()[..3]
        ));
        assert_eq!(
            boundary_face_rule(BoundaryTriangleOrder::Tri6, &flat()),
            Some(BoundaryFaceRule::Planar)
        );
        assert!(matches!(
            boundary_face_rule(BoundaryTriangleOrder::Tri6, &curved),
            Some(BoundaryFaceRule::CurvedTwelve | BoundaryFaceRule::CurvedSixteen)
        ));
        // Bracket fillet face 22538: above the projected-Jacobian margin, but its area
        // moments do not converge under the tensor rules, so it is not admitted.
        let fillet = [
            [0.02, 0.017, 0.029],
            [0.02, 0.017359901803675984, 0.029022258384304236],
            [0.01994235584120969, 0.017585270966048387, 0.029],
            [0.02, 0.017225653696809735, 0.029008841735619875],
            [
                0.019972589040368874,
                0.017476895463202486,
                0.0290106004194752,
            ],
            [0.01997743860379613, 0.01736723202559765, 0.029],
        ];
        assert!(boundary_face_validity(BoundaryTriangleOrder::Tri6, &fillet)
            .unwrap()
            .is_valid());
        assert_eq!(
            boundary_face_rule(BoundaryTriangleOrder::Tri6, &fillet),
            None
        );
    }
}
