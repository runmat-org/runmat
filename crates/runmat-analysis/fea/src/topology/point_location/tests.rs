use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

use super::{solve_3x3, volume_shape, AnalysisProjection};

#[test]
fn locates_linear_and_quadratic_tetrahedron_points_with_partition_neutral_weights() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection =
            AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(order)).unwrap();
        let location = projection
            .locate_volume_point([0.2, 0.3, 0.1])
            .unwrap()
            .expect("interior point should resolve");
        assert_eq!(location.element_index, 0);
        assert_eq!(
            location.node_indices.len(),
            match order {
                ElementOrder::Tet4 => 4,
                ElementOrder::Tet10 => 10,
            }
        );
        assert!((location.shape_values.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
        let recovered = std::array::from_fn::<_, 3, _>(|axis| {
            location
                .node_indices
                .iter()
                .zip(&location.shape_values)
                .map(|(node, weight)| {
                    projection.nodes[*node].coordinates_m.canonical_value()[axis] * weight
                })
                .sum::<f64>()
        });
        for (actual, expected) in recovered.into_iter().zip([0.2, 0.3, 0.1]) {
            assert!((actual - expected).abs() < 1.0e-11);
        }
    }
}

#[test]
fn rejects_points_outside_the_solid_projection() {
    let projection =
        AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
            .unwrap();
    assert!(projection
        .locate_volume_point([0.5, 0.5, 0.5])
        .unwrap()
        .is_none());
}

#[test]
fn reference_solve_is_scale_aware_for_small_physical_elements() {
    let scale = 1.0e-8;
    let solution = solve_3x3(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]],
        [0.2 * scale, 0.3 * scale, 0.1 * scale],
    )
    .expect("well-conditioned small element Jacobian must remain invertible");
    for (actual, expected) in solution.into_iter().zip([0.2, 0.3, 0.1]) {
        assert!((actual - expected).abs() < 1.0e-14);
    }
}

#[test]
fn quadratic_location_inverts_a_curved_isoparametric_map() {
    let mut projection =
        AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
            .unwrap();
    {
        let quantity_change = 0.2;
        let quantity_index = 1;
        let quantity_slot = &mut projection.nodes[4].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    {
        let quantity_change = 0.15;
        let quantity_index = 2;
        let quantity_slot = &mut projection.nodes[5].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    {
        let quantity_change = 0.12;
        let quantity_index = 0;
        let quantity_slot = &mut projection.nodes[7].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    let expected_barycentric = [0.4, 0.2, 0.3, 0.1];
    let (expected_weights, _) = volume_shape(ElementOrder::Tet10, expected_barycentric).unwrap();
    let position = std::array::from_fn(|axis| {
        projection.elements[0]
            .node_indices
            .iter()
            .zip(expected_weights)
            .map(|(node, weight)| {
                projection.nodes[*node].coordinates_m.canonical_value()[axis] * weight
            })
            .sum::<f64>()
    });
    let location = projection
        .locate_volume_point(position)
        .unwrap()
        .expect("curved Tet10 point should resolve through the isoparametric map");
    for (actual, expected) in location.shape_values.iter().zip(expected_weights) {
        assert!((actual - expected).abs() < 1.0e-10);
    }
}

#[test]
fn shared_vertex_ownership_and_heap_are_independent_of_element_inventory_and_order() {
    use runmat_meshing_core::fixtures::{quadratic_slab_mesh, slab_mesh};
    for mesh in [
        slab_mesh(1),
        slab_mesh(2),
        quadratic_slab_mesh(1),
        quadratic_slab_mesh(2),
    ] {
        let mut projection = AnalysisProjection::derive(&mesh).unwrap();
        let position = [0., 0., 0.];
        // Independent full-match reference retains the original ownership definition.
        let expected = projection
            .elements
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                super::locate_in_element(&projection, e, position)
                    .unwrap()
                    .map(|weights| (i, e.stable_identity, weights))
            })
            .min_by_key(|(_, identity, _)| *identity)
            .unwrap();
        let bound = projection.volume_point_location_resident_bytes().unwrap();
        let (actual, retained, peak) =
            crate::topology::measure_test_allocations(|| projection.locate_volume_point(position));
        let actual = actual.unwrap().unwrap();
        assert_eq!(actual.element_identity, expected.1);
        assert_eq!(actual.element_index, expected.0);
        assert_eq!(actual.shape_values, expected.2[..actual.node_indices.len()]);
        assert_eq!(peak as u64, bound);
        assert_eq!(retained, peak);
        projection.elements.reverse();
        let reverse = projection.locate_volume_point(position).unwrap().unwrap();
        assert_eq!(reverse.element_identity, actual.element_identity);
        assert_eq!(reverse.node_identities, actual.node_identities);
        assert_eq!(reverse.shape_values, actual.shape_values);
        let (outside, retained, peak) = crate::topology::measure_test_allocations(|| {
            projection.locate_volume_point([2., 2., 2.])
        });
        assert!(outside.unwrap().is_none());
        assert_eq!((retained, peak), (0, 0));
    }
}

#[test]
fn point_search_cancellation_is_distinct_from_an_outside_point() {
    let projection =
        AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
            .unwrap();
    let token = crate::FeaCancellationToken::new();
    token.cancel();
    let _guard = crate::replace_fea_progress_context(crate::FeaProgressContext {
        cancellation: token,
        ..Default::default()
    });
    assert_eq!(
        projection.locate_volume_point([0.2, 0.3, 0.1]).unwrap_err(),
        crate::topology::AnalysisProjectionError::Cancelled
    );
}
