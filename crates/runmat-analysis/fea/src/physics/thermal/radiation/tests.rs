use super::*;
use runmat_meshing_core::{fixtures, ElementOrder};

fn fixture(order: ElementOrder) -> (AnalysisProjection, ProjectedBoundaryFace) {
    let mesh = fixtures::canonical_tetrahedron_solver_mesh(order);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let face = projection
        .boundary_faces
        .iter()
        .find(|face| {
            face.node_indices
                .iter()
                .all(|&node| projection.nodes[node].coordinates_m.canonical_value()[2] == 0.0)
        })
        .unwrap()
        .clone();
    (projection, face)
}

#[test]
fn gray_law_sign_equilibrium_derivative_and_invalid_states() {
    let epsilon = SurfaceEmissivity::new(0.8);
    let sigma = STEFAN_BOLTZMANN_W_PER_M2_K4;
    for (temperature, ambient) in [(400.0, 300.0), (300.0, 400.0), (0.0, 300.0), (400.0, 0.0)] {
        let (flux, tangent) = outward_flux_and_tangent(temperature, ambient, epsilon).unwrap();
        let reference = 0.8 * sigma * (temperature.powi(4) - ambient.powi(4));
        assert!((flux - reference).abs() <= reference.abs() * 1e-14);
        assert_eq!(tangent, 3.2 * sigma * temperature.powi(3));
    }
    assert_eq!(
        outward_flux_and_tangent(300.0, 300.0, epsilon).unwrap().0,
        0.0
    );
    assert_eq!(
        outward_flux_and_tangent(400.0, 300.0, SurfaceEmissivity::new(0.0)).unwrap(),
        (0.0, 0.0)
    );
    for (t, a, e) in [
        (-1.0, 300.0, 0.8),
        (300.0, -1.0, 0.8),
        (f64::NAN, 300.0, 0.8),
        (300.0, f64::INFINITY, 0.8),
        (300.0, 300.0, -0.1),
        (300.0, 300.0, 1.1),
        (1e100, 300.0, 0.8),
    ] {
        assert_eq!(
            outward_flux_and_tangent(t, a, SurfaceEmissivity::new(e)),
            Err(RadiationError::InvalidState)
        );
    }
    // A fourth-power subtraction loses digits here; the factored law agrees
    // with the differential limit without an ambient-temperature scale floor.
    let t = 300.0 + 1e-10;
    let flux = outward_flux_and_tangent(t, 300.0, epsilon).unwrap().0;
    let differential = 4.0 * 0.8 * sigma * 300.0_f64.powi(3) * (t - 300.0);
    assert!((flux / differential - 1.0).abs() < 1e-12);
}

#[test]
fn tri3_and_tri6_constant_forms_conserve_power_and_have_consistent_jacobians() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (projection, face) = fixture(order);
        let temperatures = vec![400.0; face.node_indices.len()];
        let equilibrium = integrate_face(
            &projection,
            &face,
            &vec![300.0; face.node_indices.len()],
            300.0,
            SurfaceEmissivity::new(0.7),
        )
        .unwrap();
        assert_eq!(equilibrium.outward_power_w, 0.0);
        assert_eq!(equilibrium.outward_nodal_power_w, [0.0; 6]);
        let form = integrate_face(
            &projection,
            &face,
            &temperatures,
            300.0,
            SurfaceEmissivity::new(0.7),
        )
        .unwrap();
        let (flux, tangent) =
            outward_flux_and_tangent(400.0, 300.0, SurfaceEmissivity::new(0.7)).unwrap();
        assert!((form.outward_power_w / (0.5 * flux) - 1.0).abs() < 1e-13);
        assert!(
            (form.outward_nodal_power_w.iter().sum::<f64>() / form.outward_power_w - 1.0).abs()
                < 1e-13
        );
        assert!(
            (form.tangent_w_per_k.iter().flatten().sum::<f64>() / (0.5 * tangent) - 1.0).abs()
                < 1e-13
        );
        for column in 0..form.node_count {
            let mut plus = temperatures.clone();
            let mut minus = temperatures.clone();
            plus[column] += 1e-3;
            minus[column] -= 1e-3;
            let plus = integrate_face(
                &projection,
                &face,
                &plus,
                300.0,
                SurfaceEmissivity::new(0.7),
            )
            .unwrap();
            let minus = integrate_face(
                &projection,
                &face,
                &minus,
                300.0,
                SurfaceEmissivity::new(0.7),
            )
            .unwrap();
            for row in 0..form.node_count {
                let derivative =
                    (plus.outward_nodal_power_w[row] - minus.outward_nodal_power_w[row]) / 0.002;
                assert!((derivative - form.tangent_w_per_k[row][column]).abs() < 1e-8 * tangent);
                assert!(
                    (form.tangent_w_per_k[row][column] - form.tangent_w_per_k[column][row]).abs()
                        < 1e-14 * tangent
                );
            }
        }
    }
}

#[test]
fn flat_tri6_quadratic_temperature_integrates_degree_ten_form() {
    let (projection, face) = fixture(ElementOrder::Tet10);
    let temperatures = face
        .node_indices
        .iter()
        .map(|&node| {
            let x = projection.nodes[node].coordinates_m.canonical_value()[0];
            300.0 + 100.0 * x * x
        })
        .collect::<Vec<_>>();
    let form = integrate_face(
        &projection,
        &face,
        &temperatures,
        300.0,
        SurfaceEmissivity::new(1.0),
    )
    .unwrap();
    // On the unit right triangle, integral x^n dA = 1/((n+1)(n+2)).
    // Binomial expansion of (300 + 100 x^2)^4 - 300^4 is independent
    // of shape-function quadrature and includes the x^8 term.
    let expected = STEFAN_BOLTZMANN_W_PER_M2_K4
        * [4.0, 6.0, 4.0, 1.0]
            .iter()
            .enumerate()
            .map(|(i, coefficient)| {
                let j = i + 1;
                coefficient * 300.0_f64.powi((4 - j) as i32) * 100.0_f64.powi(j as i32)
                    / ((2 * j + 1) * (2 * j + 2)) as f64
            })
            .sum::<f64>();
    assert!((form.outward_power_w / expected - 1.0).abs() < 2e-13);
    let mut low_order = 0.0;
    projection
        .visit_boundary_quadrature(&face, |point| {
            let t = temperatures
                .iter()
                .zip(point.shape_values())
                .map(|(t, n)| t * n)
                .sum();
            low_order += outward_flux_and_tangent(t, 300.0, SurfaceEmissivity::new(1.0))
                .unwrap()
                .0
                * point.area_weight_m2;
        })
        .unwrap();
    assert!(
        (low_order / expected - 1.0).abs() > 1e-5,
        "degree-five rule must expose underintegration"
    );
}

#[test]
fn curved_tri6_constant_flux_matches_analytic_parabolic_surface_area() {
    let (mut projection, face) = fixture(ElementOrder::Tet10);
    let a = 0.2;
    // Isoparametric quadratic graph z=a*x^2 over the unit right triangle.
    for &node in &face.node_indices {
        let x = projection.nodes[node].coordinates_m.canonical_value()[0];
        {
            let quantity_change = a * x * x;
            let quantity_index = 2;
            let quantity_slot = &mut projection.nodes[node].coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] = quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
    }
    let b = 2.0_f64 * a;
    let area = 0.5 * ((1.0 + b * b).sqrt() + b.asinh() / b)
        - ((1.0 + b * b).powf(1.5) - 1.0) / (3.0 * b * b);
    let form = integrate_face(
        &projection,
        &face,
        &[400.0; 6],
        300.0,
        SurfaceEmissivity::new(0.8),
    )
    .unwrap();
    let (flux, tangent) =
        outward_flux_and_tangent(400.0, 300.0, SurfaceEmissivity::new(0.8)).unwrap();
    assert!((form.outward_power_w / (flux * area) - 1.0).abs() < 1e-12);
    assert!((form.integrated_tangent_w_per_k / (tangent * area) - 1.0).abs() < 1e-12);
}

#[test]
fn tri6_interpolation_rejects_negative_sample_even_with_positive_nodes() {
    let (projection, face) = fixture(ElementOrder::Tet10);
    // All corner values positive, all mid-edge values zero: the quadratic
    // interpolant is negative in the face interior.
    assert!(matches!(
        integrate_face(
            &projection,
            &face,
            &[300.0, 300.0, 300.0, 0.0, 0.0, 0.0],
            300.0,
            SurfaceEmissivity::new(0.8)
        ),
        Err(RadiationError::InvalidState)
    ));
}
