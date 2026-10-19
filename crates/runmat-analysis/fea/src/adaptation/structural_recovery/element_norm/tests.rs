use super::*;
use runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;

const CORNERS: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

fn discrepancy(order: ElementOrder, coordinates: &[[f64; 3]]) -> (f64, f64) {
    let nodal: Vec<_> = coordinates
        .iter()
        .map(|p| [1.0 + p[0] * p[0], 0.0, 0.0, 0.0, 0.0, 0.0])
        .collect();
    integrate(
        order,
        coordinates,
        &nodal,
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &mut 0,
        StructuralRecoveryEstimatorOptions::default(),
    )
    .unwrap()
}

#[test]
fn affine_tet4_integrates_squared_linear_discrepancy() {
    // Corner interpolation of x² is x; ∫tet x² dV = 2!/5!.
    let (error, reference) = discrepancy(ElementOrder::Tet4, &CORNERS);
    assert!((error - 1.0 / 60.0).abs() < 1.0e-14);
    assert!((reference - 1.0 / 6.0).abs() < 1.0e-14);
}

#[test]
fn curved_tet10_integrates_polynomial_density_against_exact_moments() {
    for curvature in [0.0, 0.4] {
        let mut coordinates = CORNERS.to_vec();
        for [a, b] in TETRAHEDRON_MIDSIDE_EDGE_CORNERS {
            coordinates.push(std::array::from_fn(|i| {
                (CORNERS[a][i] + CORNERS[b][i]) / 2.0
            }));
        }
        // X=ξ, Y=η, Z=ζ(1+aξ), det(J)=1+aξ. Recovery discrepancy is ξ².
        for p in &mut coordinates {
            p[2] *= 1.0 + curvature * p[0];
        }
        let (error, reference) = discrepancy(ElementOrder::Tet10, &coordinates);
        // ∫ ξ^p dξdηdζ = p!/(p+3)!, independent of the integration implementation.
        assert!((error - (1.0 / 210.0 + curvature / 336.0)).abs() < 1.0e-14);
        assert!((reference - (1.0 / 6.0 + curvature / 24.0)).abs() < 1.0e-14);
    }
}

#[test]
fn invalid_inventory_and_overflow_fail_closed() {
    let options = StructuralRecoveryEstimatorOptions::default();
    assert_eq!(
        integrate(
            ElementOrder::Tet10,
            &CORNERS,
            &[[1.0; 6]; 4],
            [1.0; 6],
            &mut 0,
            options
        ),
        Err(StructuralRecoveryEstimatorError::InvalidElementGeometry)
    );
    assert_eq!(
        integrate(
            ElementOrder::Tet4,
            &CORNERS,
            &[[f64::MAX; 6]; 4],
            [1.0; 6],
            &mut 0,
            options
        ),
        Err(StructuralRecoveryEstimatorError::InvalidStressField)
    );
}
