//! Independent polynomial kinetic moments, not a second coefficient table.
use super::*;
use crate::topology::quadrature::GAUSS_LEGENDRE_EIGHT_UNIT;
use runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;

fn material() -> SolidMaterial {
    SolidMaterial {
        youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(100.0),
        poisson_ratio: runmat_types::physical::PoissonRatio::new(0.2),
        density_kg_per_m3: runmat_types::physical::MassDensity::new(3.0),
    }
}

fn geometry(scale: f64, origin: f64) -> Tetrahedron10ElementGeometry {
    let corners = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.5, 3.0, 0.0],
        [0.25, 0.5, 4.0],
    ]
    .map(|point| point.map(|coordinate| origin + scale * coordinate));
    let mut nodes_m = [[0.0; 3]; 10];
    nodes_m[..4].copy_from_slice(&corners);
    for (edge, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
        nodes_m[4 + edge] =
            std::array::from_fn(|axis| 0.5 * (corners[left][axis] + corners[right][axis]));
    }
    Tetrahedron10ElementGeometry { nodes_m }
}

fn factorial(n: u32) -> f64 {
    (1..=n).map(f64::from).product()
}

fn nodes() -> [[f64; 4]; 10] {
    let mut nodes = [[0.0; 4]; 10];
    for (i, node) in nodes[..4].iter_mut().enumerate() {
        node[i] = 1.0;
    }
    for (edge, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
        nodes[4 + edge][left] = 0.5;
        nodes[4 + edge][right] = 0.5;
    }
    nodes
}

fn monomial(point: [f64; 4], exponent: [u32; 4]) -> f64 {
    point
        .into_iter()
        .zip(exponent)
        .map(|(x, n)| x.powi(n as i32))
        .product()
}

// witness: FEA-STRUCTURAL-SOLID-001/quadrature_mass @ linear_structural/tet10/numerical_kernel;
//   boundary: Affine Tet10 consistent mass preserves independent quadratic kinetic cross moments
//   across scale/translation. No lumped mass, body-load, damping or mesh-convergence claim.
#[test]
fn affine_mass_preserves_all_quadratic_kinetic_cross_moments_at_scale_and_translation() {
    let basis = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 2],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ];
    for scale in [0.125_f64, 1.0, 16.0] {
        for origin in [0.0, -32.0, 2.0_f64.powi(40)] {
            let matrix = consistent_mass_matrix(material(), geometry(scale, origin)).unwrap();
            let determinant = 24.0 * scale.powi(3);
            for left in basis {
                for right in basis {
                    let exponent = std::array::from_fn::<_, 4, _>(|i| left[i] + right[i]);
                    let exact = material().density_kg_per_m3.canonical_value()
                        * determinant
                        * exponent.into_iter().map(factorial).product::<f64>()
                        / factorial(3 + exponent.into_iter().sum::<u32>());
                    for axis in 0..3 {
                        let actual = nodes()
                            .into_iter()
                            .enumerate()
                            .map(|(i, x)| {
                                nodes()
                                    .into_iter()
                                    .enumerate()
                                    .map(|(j, y)| {
                                        monomial(x, left)
                                            * matrix[3 * i + axis][3 * j + axis]
                                            * monomial(y, right)
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>();
                        assert!((actual - exact).abs() <= 3e-14 * exact.abs(),
                            "scale={scale} origin={origin} left={left:?} right={right:?}: {actual:e} != {exact:e}");
                    }
                }
            }
            for i in 0..30 {
                for j in 0..30 {
                    assert_eq!(matrix[i][j], matrix[j][i]);
                    if i % 3 != j % 3 {
                        assert_eq!(matrix[i][j], 0.0);
                    }
                }
            }
            assert!(
                matrix[0][12] < 0.0,
                "consistent corner/edge coupling is retained"
            );
        }
    }
}

#[test]
fn curved_geometry_keeps_its_original_rule_and_richer_independent_integral() {
    for perturbation in [2.0_f64.powi(-40), 0.03125] {
        let mut geometry = geometry(1.0, 0.0);
        geometry.nodes_m[4][1] += perturbation;
        assert!(!super::super::tetrahedron10_geometry::is_affine(geometry));
        let actual = consistent_mass_matrix(material(), geometry).unwrap();
        // Direct physical kinetic moment from an independent 512-point rule.
        // All 100 quadratic cross products span the consistent mass bilinear form.
        let mut reference = [[0.0; 10]; 10];
        for (point, weight) in tensor_tetrahedron_rule(&GAUSS_LEGENDRE_EIGHT_UNIT) {
            let shape = shape_values(point);
            let measure = material().density_kg_per_m3.canonical_value()
                * weight
                * jacobian_determinant(geometry.nodes_m, point).unwrap();
            for i in 0..10 {
                for j in 0..10 {
                    reference[i][j] += measure * shape[i] * shape[j];
                }
            }
        }
        for i in 0..10 {
            for j in 0..10 {
                assert!((actual[3 * i][3 * j] - reference[i][j]).abs() < 3e-14);
            }
        }
    }
}

// witness: FEA-STRUCTURAL-SOLID-001/quadrature_mass @ linear_structural/tet10/numerical_kernel;
//   coverage: partial;
//   boundary: Tet10 consistent mass preserves physical rigid-translation mass under reciprocal
//   density and geometry scaling.
#[test]
fn reciprocal_density_and_geometry_scales_preserve_physical_rigid_translation_mass() {
    for (length_exponent, density_exponent) in [(-350, 900), (300, -900)] {
        let mut material = material();
        material.density_kg_per_m3 =
            runmat_types::physical::MassDensity::new(2.0_f64.powi(density_exponent));
        let geometry = geometry(2.0_f64.powi(length_exponent), 0.0);
        let matrix = consistent_mass_matrix(material, geometry).unwrap();
        // Exact volume for the independently constructed skew tetrahedron is 4*s^3.
        // Combine exponents first so the continuum reference has no tiny intermediate.
        let expected = 4.0 * 2.0_f64.powi(density_exponent + 3 * length_exponent);
        let actual = (0..10)
            .map(|i| (0..10).map(|j| matrix[3 * i][3 * j]).sum::<f64>())
            .sum::<f64>();
        assert!(
            (actual - expected).abs() < 3e-14 * expected,
            "length={length_exponent} density={density_exponent}: {actual:e} != {expected:e}"
        );
    }
}
