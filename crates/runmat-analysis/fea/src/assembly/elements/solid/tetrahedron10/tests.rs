use super::super::tetrahedron10_mass::consistent_mass_matrix;
use super::super::tetrahedron10_stiffness::global_stiffness_matrix;
use super::*;
use crate::assembly::elements::solid::Tetrahedron4ElementGeometry;
use runmat_simulation_usage::UsageLedger;
use std::sync::Arc;

fn straight_unit_tetrahedron() -> Tetrahedron10ElementGeometry {
    let corners = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    Tetrahedron10ElementGeometry {
        nodes_m: [
            corners[0],
            corners[1],
            corners[2],
            corners[3],
            midpoint(corners[0], corners[1]),
            midpoint(corners[1], corners[2]),
            midpoint(corners[2], corners[0]),
            midpoint(corners[0], corners[3]),
            midpoint(corners[1], corners[3]),
            midpoint(corners[2], corners[3]),
        ],
    }
}

fn steel() -> SolidMaterial {
    SolidMaterial {
        youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(200.0e9),
        poisson_ratio: runmat_types::physical::PoissonRatio::new(0.3),
        density_kg_per_m3: runmat_types::physical::MassDensity::new(7_850.0),
    }
}

#[test]
fn admitted_affine_tetrahedron10_geometry_conserves_reference_work() {
    // A reference gradient plus affine determinant (92 + 23), followed
    // by a reference gradient, affine inverse (60), and 150 transforms.
    let limit = OperationCount::from_operations(92 + 23 + 92 + 60 + 150);
    let ledger = Arc::new(UsageLedger::new(limit, limit).unwrap());
    let context = UsageContext::new(Arc::clone(&ledger));
    context.run_sync(|| {
        jacobian_determinant(straight_unit_tetrahedron().nodes_m, [0.25; 4]).unwrap();
        physical_shape_gradients_and_measure(straight_unit_tetrahedron().nodes_m, [0.25; 4])
            .unwrap();
    });
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.completed.assembly, limit);
    assert_eq!(snapshot.reserved, OperationCount::ZERO);
}

#[test]
fn quadratic_shape_gradients_preserve_partition_of_unity() {
    for barycentric in QUADRATURE {
        let gradients = reference_shape_gradients(barycentric);
        for axis in 0..3 {
            assert!(
                gradients
                    .iter()
                    .map(|gradient| gradient[axis])
                    .sum::<f64>()
                    .abs()
                    < 1.0e-14
            );
        }
    }
}

#[test]
fn skew_tetrahedron10_gradients_reproduce_physical_coordinates() {
    let corners = [
        [0.2, -0.4, 0.7],
        [2.1, 0.3, 0.9],
        [0.8, 1.7, 0.5],
        [0.5, 0.2, 2.4],
    ];
    let geometry = Tetrahedron10ElementGeometry {
        nodes_m: [
            corners[0],
            corners[1],
            corners[2],
            corners[3],
            midpoint(corners[0], corners[1]),
            midpoint(corners[1], corners[2]),
            midpoint(corners[2], corners[0]),
            midpoint(corners[0], corners[3]),
            midpoint(corners[1], corners[3]),
            midpoint(corners[2], corners[3]),
        ],
    };
    for barycentric in QUADRATURE {
        let gradients = physical_shape_gradients_and_measure(geometry.nodes_m, barycentric)
            .unwrap()
            .0;
        for coordinate in 0..3 {
            for derivative in 0..3 {
                let value = geometry
                    .nodes_m
                    .iter()
                    .zip(gradients)
                    .map(|(node, gradient)| node[coordinate] * gradient[derivative])
                    .sum::<f64>();
                let expected = f64::from(coordinate == derivative);
                assert!((value - expected).abs() < 1.0e-11);
            }
        }
    }
}

#[test]
fn straight_tetrahedron10_stiffness_is_symmetric_and_rejects_rigid_translation() {
    let stiffness = global_stiffness_matrix(steel(), straight_unit_tetrahedron()).unwrap();
    for row in 0..TETRAHEDRON10_ELEMENT_DOF_COUNT {
        for column in 0..TETRAHEDRON10_ELEMENT_DOF_COUNT {
            assert!((stiffness[row][column] - stiffness[column][row]).abs() < 1.0e-4);
        }
    }
    let mut translation = [0.0; TETRAHEDRON10_ELEMENT_DOF_COUNT];
    for displacement in translation.chunks_exact_mut(3) {
        displacement.copy_from_slice(&[1.0, -2.0, 3.0]);
    }
    let residual = std::array::from_fn::<_, TETRAHEDRON10_ELEMENT_DOF_COUNT, _>(|row| {
        stiffness[row]
            .iter()
            .zip(translation)
            .map(|(left, right)| left * right)
            .sum::<f64>()
    });
    assert!(residual.iter().all(|value| value.abs() < 2.0e-3));
}

#[test]
fn tetrahedron10_rejects_nonfinite_and_inverted_geometry() {
    let mut nonfinite = straight_unit_tetrahedron();
    nonfinite.nodes_m[4][0] = f64::NAN;
    assert_eq!(
        global_stiffness_matrix(steel(), nonfinite).unwrap_err(),
        Tetrahedron10ElementError::NonFiniteCoordinate
    );

    let mut inverted = straight_unit_tetrahedron();
    inverted.nodes_m.swap(1, 2);
    inverted.nodes_m.swap(4, 6);
    inverted.nodes_m.swap(8, 9);
    assert_eq!(
        global_stiffness_matrix(steel(), inverted).unwrap_err(),
        Tetrahedron10ElementError::DegenerateOrInverted
    );
}

#[test]
fn tetrahedron10_rejects_a_corner_fold_missed_by_all_quadrature_points() {
    let mut folded = straight_unit_tetrahedron();
    // The 01 midside node moves from x=0.5 to x=0.8. Along that edge,
    // det(J)=1+1.2(1-2*x), so it is negative at corner 1 even though every
    // current stiffness and volume quadrature point still has a positive determinant.
    folded.nodes_m[4][0] += 0.3;
    for barycentric in QUADRATURE {
        assert!(jacobian_determinant(folded.nodes_m, barycentric).is_ok());
    }
    for (barycentric, _) in VOLUME_QUADRATURE {
        assert!(jacobian_determinant(folded.nodes_m, barycentric).is_ok());
    }
    assert_eq!(
        validate_nodes(folded.nodes_m),
        Err(Tetrahedron10ElementError::DegenerateOrInverted)
    );
    assert_eq!(
        global_stiffness_matrix(steel(), folded),
        Err(Tetrahedron10ElementError::DegenerateOrInverted)
    );
    assert_eq!(
        consistent_mass_matrix(steel(), folded),
        Err(Tetrahedron10ElementError::DegenerateOrInverted)
    );
    assert_eq!(
        folded.volume_m3(),
        Err(Tetrahedron10ElementError::DegenerateOrInverted)
    );
}

#[test]
fn curved_tetrahedron10_mass_matches_exact_barycentric_moments() {
    let mut geometry = straight_unit_tetrahedron();
    let warp = 0.1;
    geometry.nodes_m[4][0] += warp;
    let mass = consistent_mass_matrix(steel(), geometry).unwrap();
    let density = steel().density_kg_per_m3.canonical_value();
    // For this warp det(J)=1+4δ(L0-L1). Exact simplex moments give
    // ∫N0² dV = 1/420 + δ/315 and ∫N4² dV = 4/315.
    assert!((mass[0][0] - density * (1.0 / 420.0 + warp / 315.0)).abs() < 1.0e-10);
    assert!((mass[12][12] - density * 4.0 / 315.0).abs() < 1.0e-10);
    let rigid_translation_mass = (0..10)
        .flat_map(|left| (0..10).map(move |right| mass[left * 3][right * 3]))
        .sum::<f64>();
    assert!((rigid_translation_mass - density * geometry.volume_m3().unwrap()).abs() < 1.0e-9);
}

#[test]
fn quadratic_recovery_exhibits_expected_h_and_p_convergence() {
    let tet4_coarse = manufactured_quadratic_recovery_error(1, false);
    let tet4_medium = manufactured_quadratic_recovery_error(2, false);
    let tet4_fine = manufactured_quadratic_recovery_error(4, false);
    let tet10 = manufactured_quadratic_recovery_error(1, true);

    assert!((tet4_coarse / tet4_medium - 2.0).abs() < 1.0e-12);
    assert!((tet4_medium / tet4_fine - 2.0).abs() < 1.0e-12);
    assert!(tet10 < 1.0e-13 * tet4_coarse);
}

fn manufactured_quadratic_recovery_error(divisions: usize, quadratic: bool) -> f64 {
    let h = 1.0 / divisions as f64;
    let mut error_squared = 0.0;
    for i in 0..divisions {
        for j in 0..divisions {
            for k in 0..divisions {
                let origin = [i as f64 * h, j as f64 * h, k as f64 * h];
                let vertex = |dx: f64, dy: f64, dz: f64| {
                    [origin[0] + dx * h, origin[1] + dy * h, origin[2] + dz * h]
                };
                let v000 = vertex(0.0, 0.0, 0.0);
                let v100 = vertex(1.0, 0.0, 0.0);
                let v010 = vertex(0.0, 1.0, 0.0);
                let v110 = vertex(1.0, 1.0, 0.0);
                let v001 = vertex(0.0, 0.0, 1.0);
                let v101 = vertex(1.0, 0.0, 1.0);
                let v011 = vertex(0.0, 1.0, 1.0);
                let v111 = vertex(1.0, 1.0, 1.0);
                for mut corners in [
                    [v000, v100, v110, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v011, v001, v111],
                    [v000, v001, v101, v111],
                    [v000, v101, v100, v111],
                ] {
                    if (Tetrahedron4ElementGeometry { nodes_m: corners })
                        .volume_m3()
                        .is_err()
                    {
                        corners.swap(1, 2);
                    }
                    error_squared += if quadratic {
                        quadratic_element_recovery_error(corners)
                    } else {
                        linear_element_recovery_error(corners)
                    };
                }
            }
        }
    }
    error_squared.sqrt()
}

fn linear_element_recovery_error(corners: [[f64; 3]; 4]) -> f64 {
    let geometry = Tetrahedron4ElementGeometry { nodes_m: corners };
    let b = super::super::tetrahedron4::strain_displacement_matrix(geometry).unwrap();
    let displacement = corners
        .into_iter()
        .flat_map(|node| [node[0].powi(2), 0.0, 0.0])
        .collect::<Vec<_>>();
    let recovered = b[0]
        .iter()
        .zip(displacement)
        .map(|(shape, value)| shape * value)
        .sum::<f64>();
    let volume = geometry.volume_m3().unwrap();
    QUADRATURE
        .iter()
        .map(|barycentric| {
            let x = (0..4)
                .map(|node| barycentric[node] * corners[node][0])
                .sum::<f64>();
            (recovered - 2.0 * x).powi(2) * volume / 4.0
        })
        .sum()
}

fn quadratic_element_recovery_error(corners: [[f64; 3]; 4]) -> f64 {
    let nodes_m = [
        corners[0],
        corners[1],
        corners[2],
        corners[3],
        midpoint(corners[0], corners[1]),
        midpoint(corners[1], corners[2]),
        midpoint(corners[2], corners[0]),
        midpoint(corners[0], corners[3]),
        midpoint(corners[1], corners[3]),
        midpoint(corners[2], corners[3]),
    ];
    let geometry = Tetrahedron10ElementGeometry { nodes_m };
    let displacement = std::array::from_fn(|dof| {
        let node = dof / 3;
        if dof % 3 == 0 {
            nodes_m[node][0].powi(2)
        } else {
            0.0
        }
    });
    geometry
        .quadrature_strain(displacement)
        .unwrap()
        .iter()
        .map(|point| {
            let x = (0..10)
                .map(|node| shape_values(point.barycentric)[node] * nodes_m[node][0])
                .sum::<f64>();
            (point.strain[0] - 2.0 * x).powi(2) * point.integration_weight_m3
        })
        .sum()
}

fn midpoint(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| (left[axis] + right[axis]) * 0.5)
}

/// The browser worker's owner quotes capacity only with no batch outstanding. A curved
/// element's awaited stiffness and mass match the synchronous kernels and leave no
/// reservation behind; the first quadratic element no longer fails with OutstandingWork.
#[test]
fn awaited_tetrahedron10_kernels_request_capacity_with_no_batch_outstanding() {
    use super::super::tetrahedron10_mass::consistent_mass_matrix_with_capacity;
    use super::super::tetrahedron10_stiffness::global_stiffness_matrix_with_capacity;
    use runmat_simulation_usage::{AsyncCapacityOwner, LedgerError};
    struct ExclusiveOwner<'a>(&'a UsageLedger);
    impl AsyncCapacityOwner for ExclusiveOwner<'_> {
        fn ensure_capacity(
            &self,
            additional: OperationCount,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), LedgerError>> + '_>>
        {
            let quoted = self.0.exclusive_work_requirement(additional).map(|_| ());
            Box::pin(async move { quoted })
        }
    }
    fn ready<R>(future: impl std::future::Future<Output = R>) -> R {
        let mut future = std::pin::pin!(future);
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(result) => result,
            std::task::Poll::Pending => unreachable!("an immediate owner never suspends"),
        }
    }
    let mut curved = straight_unit_tetrahedron();
    curved.nodes_m[4][1] -= 0.05;
    for geometry in [straight_unit_tetrahedron(), curved] {
        let limit = OperationCount::from_operations(1 << 40);
        let ledger = Arc::new(UsageLedger::new(limit, limit).unwrap());
        let context = UsageContext::new(Arc::clone(&ledger));
        let owner = ExclusiveOwner(&ledger);
        let (stiffness, mass) = context.run_sync(|| {
            (
                ready(global_stiffness_matrix_with_capacity(
                    steel(),
                    geometry,
                    &owner,
                )),
                ready(consistent_mass_matrix_with_capacity(
                    steel(),
                    geometry,
                    &owner,
                )),
            )
        });
        let admitted = |result| match result {
            Ok(matrix) => matrix,
            Err(Tetrahedron10CapacityError::Admission(cause)) => {
                panic!("capacity refused: {cause:?}")
            }
            Err(Tetrahedron10CapacityError::Numerical(error)) => panic!("{error:?}"),
        };
        assert_eq!(
            admitted(stiffness),
            global_stiffness_matrix(steel(), geometry).unwrap()
        );
        assert_eq!(
            admitted(mass),
            consistent_mass_matrix(steel(), geometry).unwrap()
        );
        let snapshot = ledger.snapshot().unwrap();
        assert!(snapshot.completed.assembly > OperationCount::ZERO);
        assert_eq!(snapshot.reserved, OperationCount::ZERO);
    }
}
