use super::*;
use crate::assembly::{elements::solid::SolidMaterial, SolvedThermalRecoveryQuadrature};
use runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh;

pub(super) fn element(order: ElementOrder) -> (SolidRecoveryElementSummary, Vec<[f64; 3]>) {
    let mesh = canonical_tetrahedron_solver_mesh(order);
    let coordinates: Vec<_> = mesh
        .topology
        .nodes
        .iter()
        .map(|node| node.coordinates_m.canonical_value())
        .collect();
    let geometry = match order {
        ElementOrder::Tet4 => SolidRecoveryGeometry::Tetrahedron4 {
            node_indices: std::array::from_fn(|i| i),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(coordinates.clone().try_into().unwrap()),
        },
        ElementOrder::Tet10 => SolidRecoveryGeometry::Tetrahedron10 {
            node_indices: std::array::from_fn(|i| i),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(coordinates.clone().try_into().unwrap()),
        },
    };
    (
        SolidRecoveryElementSummary {
            element_id: "patch".into(),
            region_id: "solid".into(),
            material: SolidMaterial {
                youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(100.0),
                poisson_ratio: runmat_types::physical::PoissonRatio::new(0.25),
                density_kg_per_m3: runmat_types::physical::MassDensity::new(1.0),
            },
            thermal_eigenstrain: None,
            prescribed_temperature_k: None,
            solved_thermal_quadrature: None,
            geometry,
        },
        coordinates,
    )
}

fn thermal(element: &mut SolidRecoveryElementSummary, coordinates: &[[f64; 3]], gradient: f64) {
    element.solved_thermal_quadrature = Some(SolvedThermalRecoveryQuadrature {
        nodal_temperature_k: runmat_types::physical::Quantity::new(
            coordinates
                .iter()
                .map(|x| 300.0 + 10.0 + gradient * x[0])
                .collect::<Vec<_>>(),
        ),
        expansion_coefficient_per_k: runmat_types::physical::ThermalExpansion::new(0.001),
        reference_temperature_k: runmat_types::physical::Temperature::new(300.0),
        // These legacy recovery samples must not drive integration at different points.
        temperature_k: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Temperature,
            _,
        >::new([999.0; 4]),
        eigenstrain: [123.0; 4],
    });
}

#[test]
fn free_expansion_and_restrained_heating_have_physical_geometric_stress() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (mut element, coordinates) = element(order);
        thermal(&mut element, &coordinates, 0.0);
        let displacement: Vec<_> = coordinates.iter().map(|x| x.map(|v| v * 0.01)).collect();
        let free = integrate::<30>(&element, &displacement, order, &coordinates, false).unwrap();
        assert!(free
            .stresses
            .iter()
            .flatten()
            .flatten()
            .all(|v| v.abs() < 1e-12));
        assert!(free.matrix.iter().flatten().all(|v| v.abs() < 1e-12));
        let fixed = integrate::<30>(
            &element,
            &vec![[0.0; 3]; coordinates.len()],
            order,
            &coordinates,
            false,
        )
        .unwrap();
        assert_eq!(fixed.stresses.len(), 15);
        for stress in &fixed.stresses {
            for (i, row) in stress.iter().enumerate() {
                for (j, value) in row.iter().enumerate() {
                    assert!((value - if i == j { -2.0 } else { 0.0 }).abs() < 1e-12);
                }
            }
        }
        // v=(x,0,0): integral grad(v)^T sigma grad(v) dV = -2/6.
        let moment: f64 = coordinates
            .iter()
            .enumerate()
            .flat_map(|(i, a)| {
                coordinates
                    .iter()
                    .enumerate()
                    .map(move |(j, b)| (i, j, a[0] * b[0]))
            })
            .map(|(i, j, w)| w * fixed.matrix[3 * i][3 * j])
            .sum();
        assert!((moment + 2.0 / 6.0).abs() < 1e-12);
    }
}

#[test]
fn quadratic_temperature_geometric_moment_requires_higher_than_four_point_rule() {
    let (mut element, coordinates) = element(ElementOrder::Tet10);
    thermal(&mut element, &coordinates, 0.0);
    element
        .solved_thermal_quadrature
        .as_mut()
        .unwrap()
        .nodal_temperature_k = runmat_types::physical::Quantity::new(
        coordinates
            .iter()
            .map(|x| 300.0 + x[0] * x[0])
            .collect::<Vec<_>>(),
    );
    let result = integrate::<30>(
        &element,
        &[[0.0; 3]; 10],
        ElementOrder::Tet10,
        &coordinates,
        false,
    )
    .unwrap();
    // u_test=(x²,0,0): stress_xx=-0.2 x²; integrand=-0.8 x^4.
    // Unit-simplex integral(x^4)=4!/(4+3)!=1/210.
    let mut moment = 0.0;
    for (i, a) in coordinates.iter().enumerate() {
        for (j, b) in coordinates.iter().enumerate() {
            moment += a[0].powi(2) * result.matrix[3 * i][3 * j] * b[0].powi(2);
        }
    }
    assert!((moment + 0.8 / 210.0).abs() < 1e-12, "{moment}");
}

#[test]
fn compatible_linear_temperature_has_zero_stress_at_every_integration_point() {
    let (mut element, coordinates) = element(ElementOrder::Tet10);
    thermal(&mut element, &coordinates, 20.0);
    let displacement: Vec<_> = coordinates
        .iter()
        .map(|&[x, y, z]| {
            [
                0.001 * (10.0 * x + 10.0 * (x * x - y * y - z * z)),
                0.001 * (10.0 * y + 20.0 * x * y),
                0.001 * (10.0 * z + 20.0 * x * z),
            ]
        })
        .collect();
    let result = integrate::<30>(
        &element,
        &displacement,
        ElementOrder::Tet10,
        &coordinates,
        false,
    )
    .unwrap();
    assert!(result
        .stresses
        .iter()
        .flatten()
        .flatten()
        .all(|v| v.abs() < 1e-11));
    assert!(result.matrix.iter().flatten().all(|v| v.abs() < 1e-11));
}

#[test]
fn curved_rule_preserves_accepted_samples_and_fails_closed_without_agreement() {
    let (mut element, mut coordinates) = element(ElementOrder::Tet10);
    coordinates[4][0] += 0.02;
    thermal(&mut element, &coordinates, 0.0);
    let result = integrate::<30>(
        &element,
        &[[0.0; 3]; 10],
        ElementOrder::Tet10,
        &coordinates,
        true,
    )
    .unwrap();
    assert!(matches!(result.stresses.len(), 1728 | 4096));
    coordinates[4][0] += 0.22;
    let error = integrate::<30>(
        &element,
        &[[0.0; 3]; 10],
        ElementOrder::Tet10,
        &coordinates,
        true,
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("did not converge"));
    let zero = Integrated {
        matrix: [[0.0; 3]; 3],
        stresses: Vec::new(),
        roundoff_scale: 0.0,
    };
    let nonzero = Integrated {
        matrix: [[1.0; 3]; 3],
        stresses: Vec::new(),
        roundoff_scale: 1.0,
    };
    assert!(converged(&zero, &zero));
    assert!(!converged(&nonzero, &zero));
    let invalid = Integrated {
        roundoff_scale: f64::INFINITY,
        ..nonzero
    };
    assert!(!converged(&invalid, &zero));
}

#[test]
fn genuinely_curved_free_expansion_converges_at_physical_roundoff_scale() {
    for modulus_factor in [1.0e-12, 1.0, 1.0e12] {
        let (mut element, mut coordinates) = element(ElementOrder::Tet10);
        coordinates[4][0] += 0.02;
        element.geometry = SolidRecoveryGeometry::Tetrahedron10 {
            node_indices: std::array::from_fn(|i| i),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(coordinates.clone().try_into().unwrap()),
        };
        assert!(!tetrahedron10_is_affine(Tetrahedron10ElementGeometry {
            nodes_m: coordinates.clone().try_into().unwrap(),
        }));
        element.material.youngs_modulus_pa = runmat_types::physical::Quantity::new(
            element.material.youngs_modulus_pa.canonical_value() * (modulus_factor),
        );
        thermal(&mut element, &coordinates, 0.0);
        // Isoparametric partition of unity reproduces u=alpha*deltaT*x exactly,
        // including on genuinely curved geometry. The exact physical stress is zero.
        let displacement: Vec<_> = coordinates.iter().map(|x| x.map(|v| 0.01 * v)).collect();
        let result = integrate::<30>(
            &element,
            &displacement,
            ElementOrder::Tet10,
            &coordinates,
            true,
        )
        .expect("curved free expansion must not fail a relative-to-zero stress test");
        assert!(matches!(result.stresses.len(), 1728 | 4096));
        assert!(result.roundoff_scale.is_finite() && result.roundoff_scale > 0.0);
        let tolerance = 256.0 * f64::EPSILON * result.roundoff_scale;
        assert!(result
            .matrix
            .iter()
            .flatten()
            .all(|value| value.abs() <= tolerance));
        assert!(result
            .stresses
            .iter()
            .flatten()
            .flatten()
            .all(|value| value.abs()
                <= 256.0 * f64::EPSILON * element.material.youngs_modulus_pa.canonical_value()));
    }
}
