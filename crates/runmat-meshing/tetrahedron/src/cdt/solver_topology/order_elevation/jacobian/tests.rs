use super::exact::{exact_jacobian_at, exact_jacobian_from_reference_basis};
use super::*;
use runmat_meshing_core::{quality::predicate::ExactDyadic, NeverCancelled};

#[test]
fn borrowed_coordinate_gather_preserves_sparse_ids_connectivity_order_and_errors() {
    use runmat_meshing_core::{ElementOrder, SolverMeshNode, SolverVolumeElement, StableDigest};
    let original = straight();
    let nodes = original
        .iter()
        .enumerate()
        .map(|(index, coordinates)| SolverMeshNode {
            node_id: (index as u64 + 1) * 17,
            stable_identity: StableDigest::from_bytes([index as u8 + 1; 32]),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(*coordinates),
            provenance: Vec::new(),
            exact_parameters: Vec::new(),
        })
        .collect();
    let topology = SolverMeshTopology {
        nodes,
        volume_elements: Vec::new(),
        neighbors: Vec::new(),
        boundary_faces: Vec::new(),
        boundary_edges: Vec::new(),
        regions: Vec::new(),
        conformal_interfaces: Vec::new(),
        contacts: Vec::new(),
        field_topologies: Vec::new(),
    };
    let order = [7, 0, 9, 3, 1, 6, 4, 2, 5, 8];
    let mut element = SolverVolumeElement {
        element_id: 1,
        stable_identity: StableDigest::from_bytes([99; 32]),
        order: ElementOrder::Tet10,
        node_ids: order.map(|index| (index as u64 + 1) * 17).into(),
        region_id: runmat_geometry_core::PersistentEntityId::new(
            runmat_geometry_core::PersistentEntityKind::Region,
            "region".into(),
            Vec::new(),
        ),
        provenance: Vec::new(),
    };
    assert_eq!(
        element_coordinates_from_topology(&element, &topology).unwrap(),
        order.map(|index| original[index])
    );
    element.node_ids[0] = 0;
    assert!(element_coordinates_from_topology(&element, &topology).is_err());
    element.node_ids.pop();
    assert!(element_coordinates_from_topology(&element, &topology).is_err());
}

fn straight() -> [[f64; 3]; 10] {
    let corners = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    [
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
    ]
}

fn exact_reference_basis(coordinates: [[f64; 3]; 10]) -> [[[ExactDyadic; 3]; 3]; 4] {
    let mut work = 0;
    build_exact_reference_jacobians(coordinates, 10_000, &NeverCancelled, &mut work).unwrap()
}

fn exact_jacobian(coordinates: [[f64; 3]; 10], barycentric: [f64; 4]) -> [[ExactDyadic; 3]; 3] {
    let mut work = 0;
    exact_jacobian_at(coordinates, barycentric, 10_000, &NeverCancelled, &mut work).unwrap()
}

// witness: FEA-MESH-DISCRETIZATION-001/tet10_elevation @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: The Jacobian interval certificate accepts a linear mapping and rejects a curved
//   inversion.
#[test]
fn interval_certificate_accepts_linear_mapping_and_rejects_curved_inversion() {
    let mut work = 0;
    certify_element(
        straight(),
        10_000,
        16,
        1,
        &NeverCancelled,
        ValidationWork {
            counter: &mut work,
            storage: Storage(None),
        },
    )
    .unwrap();

    let mut inverted = straight();
    inverted[4] = [0.5, 0.0, 4.0];
    let orientation = match orient3d(inverted[..4].try_into().unwrap()).unwrap() {
        PredicateSign::Positive => -1.0,
        PredicateSign::Negative => 1.0,
        PredicateSign::Zero => panic!("test corners must be noncoplanar"),
    };
    assert!(has_certified_nonpositive_vertex(
        inverted,
        REFERENCE_TETRAHEDRON,
        orientation
    ));
    let mut work = 0;
    assert_eq!(
        certify_element(
            inverted,
            100_000,
            24,
            1,
            &NeverCancelled,
            ValidationWork {
                counter: &mut work,
                storage: Storage(None)
            }
        )
        .unwrap_err()
        .kind,
        DelaunaySolverTopologyErrorKind::InvalidMesh
    );
}

#[test]
fn exact_fallback_certifies_positive_mapping_below_interval_resolution() {
    let mut coordinates = straight();
    coordinates[4][0] = 0.25 + 1.0e-16;
    let collapsed_cell = [REFERENCE_TETRAHEDRON[0]; 4];
    let enclosure = determinant_enclosure(coordinates, collapsed_cell).unwrap();
    assert!(enclosure.lower <= 0.0);
    let orientation = match orient3d(coordinates[..4].try_into().unwrap()).unwrap() {
        PredicateSign::Positive => -1.0,
        PredicateSign::Negative => 1.0,
        PredicateSign::Zero => panic!("test corners must be noncoplanar"),
    };
    let reference_jacobians = exact_reference_basis(coordinates);
    let mut exact_work = 0;
    assert!(exact_bernstein_coefficients_are_positive(
        &reference_jacobians,
        collapsed_cell,
        orientation,
        10_000,
        &NeverCancelled,
        &mut exact_work,
    )
    .unwrap());

    // The exact fallback proves the Jacobian positive everywhere ...
    let mut work = 0;
    certify_element_within(
        coordinates,
        JacobianSearchLimits {
            maximum_search_work: 10_000,
            maximum_recursion_depth: 0,
            cancellation_check_interval: 1,
        },
        &NeverCancelled,
        &Storage(None),
        &mut work,
        &mut 0,
    )
    .unwrap();
    // ... but a corner determinant of order 1e-16 has no scale-relative margin, so the shared
    // Tet10 element rule, and with it the mesher's certification, rejects the element.
    let mut work = 0;
    let failure = certify_element(
        coordinates,
        10_000,
        0,
        1,
        &NeverCancelled,
        ValidationWork {
            counter: &mut work,
            storage: Storage(None),
        },
    )
    .unwrap_err();
    assert!(
        failure.reason.contains("InsufficientJacobianMargin"),
        "{failure:?}"
    );
}

#[test]
fn exact_reference_basis_matches_direct_jacobian_evaluation() {
    let mut coordinates = straight();
    coordinates[4] = [0.375, 0.0, 0.125];
    coordinates[8] = [0.5, 0.0, 0.625];
    let reference_jacobians = exact_reference_basis(coordinates);
    let barycentric = [0.125, 0.25, 0.375, 0.25];
    let direct = exact_jacobian(coordinates, barycentric);
    let mut work = 0;
    let interpolated = exact_jacobian_from_reference_basis(
        &reference_jacobians,
        barycentric,
        10_000,
        &NeverCancelled,
        &mut work,
    )
    .unwrap();

    for row in 0..3 {
        for column in 0..3 {
            assert_eq!(
                direct[row][column]
                    .clone()
                    .subtract_exact(interpolated[row][column].clone())
                    .sign(),
                PredicateSign::Zero
            );
        }
    }
}

#[test]
fn exact_fallback_observes_cancellation_between_coefficients() {
    struct Cancelled;
    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }

    let coordinates = straight();
    let orientation = match orient3d(coordinates[..4].try_into().unwrap()).unwrap() {
        PredicateSign::Positive => -1.0,
        PredicateSign::Negative => 1.0,
        PredicateSign::Zero => panic!("test corners must be noncoplanar"),
    };
    let reference_jacobians = exact_reference_basis(coordinates);
    let mut work = 0;
    let failure = exact_bernstein_coefficients_are_positive(
        &reference_jacobians,
        REFERENCE_TETRAHEDRON,
        orientation,
        10_000,
        &Cancelled,
        &mut work,
    )
    .unwrap_err();

    assert_eq!(failure.kind, DelaunaySolverTopologyErrorKind::Cancelled);
    assert_eq!(work, 1);
}

#[test]
fn exact_jacobian_construction_observes_cancellation_between_terms() {
    struct Cancelled;
    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }

    let mut work = 0;
    let failure = match exact_jacobian_at(
        straight(),
        REFERENCE_TETRAHEDRON[0],
        10_000,
        &Cancelled,
        &mut work,
    ) {
        Ok(_) => panic!("cancelled exact Jacobian construction must fail"),
        Err(failure) => failure,
    };

    assert_eq!(failure.kind, DelaunaySolverTopologyErrorKind::Cancelled);
    assert_eq!(work, 1);
}

#[test]
fn interval_certificate_bounds_each_element_and_honors_cancellation() {
    let mut work = 0;
    assert_eq!(
        certify_element(
            straight(),
            3,
            16,
            1,
            &NeverCancelled,
            ValidationWork {
                counter: &mut work,
                storage: Storage(None)
            }
        )
        .unwrap_err()
        .kind,
        // An element that cannot be certified within its own work share is not admitted.
        DelaunaySolverTopologyErrorKind::InvalidMesh
    );
    assert!(
        work > 3,
        "the element's actual work is charged to the shared counter"
    );

    struct Cancelled;
    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    let mut work = 0;
    assert_eq!(
        certify_element(
            straight(),
            10_000,
            16,
            1,
            &Cancelled,
            ValidationWork {
                counter: &mut work,
                storage: Storage(None)
            }
        )
        .unwrap_err()
        .kind,
        DelaunaySolverTopologyErrorKind::Cancelled
    );
}

fn midpoint(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] * 0.5 + right[axis] * 0.5)
}
