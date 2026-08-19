use super::*;

#[test]
fn boundary_driven_refill_recovers_large_nonstar_tetrahelix() {
    const TETRAHEDRA: usize = 20;
    let mut coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 3.0_f64.sqrt() * 0.5, 0.0],
        [0.5, 3.0_f64.sqrt() / 6.0, (2.0_f64 / 3.0).sqrt()],
    ];
    for index in 0..TETRAHEDRA - 1 {
        coordinates.push(reflect_across_face(
            coordinates[index],
            coordinates[index + 1],
            coordinates[index + 2],
            coordinates[index + 3],
        ));
    }
    let nodes = coordinates
        .iter()
        .enumerate()
        .map(|(index, coordinates_m)| DelaunayVolumeNode {
            identity: StableDigest::from_bytes([(index + 1) as u8; 32]),
            coordinates_m: *coordinates_m,
        })
        .collect::<Vec<_>>();
    let tetrahedra = (0..TETRAHEDRA)
        .map(|index| {
            [
                index as u32,
                index as u32 + 1,
                index as u32 + 2,
                index as u32 + 3,
            ]
        })
        .collect::<Vec<_>>();
    let topology = build_delaunay_volume_topology(
        nodes,
        tetrahedra,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let cavity = crate::cavity::constrained::ConstrainedCavity {
        removed_tetrahedron_ids: (0..TETRAHEDRA as u32).collect(),
        boundary_faces: topology
            .incidence
            .boundary_facets
            .iter()
            .map(
                |facet| crate::cavity::constrained::ConstrainedCavityBoundaryFace {
                    node_ids: facet.vertex_indices,
                    outside_tetrahedron_ids: Vec::new(),
                    source_face_id: None,
                    source_edge_ids: [None; 3],
                    region_ids: Vec::new(),
                },
            )
            .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: topology
            .tetrahedra
            .iter()
            .map(|tetrahedron| {
                tetrahedron_volume(
                    tetrahedron
                        .vertex_indices
                        .map(|node| topology.nodes[node as usize].coordinates_m),
                )
            })
            .sum(),
    };
    assert!(topology.nodes.len() > 20);
    assert!(cavity.boundary_faces.len() > 40);
    let mut work = FacetRecoveryWork::new(DelaunayFacetRecoveryOptions::default(), &NeverCancelled);
    assert!(
        super::super::cavity::star::star_refill(&cavity, &topology, 0, &mut work)
            .unwrap()
            .is_none()
    );

    let cavity_nodes = topology
        .nodes
        .iter()
        .enumerate()
        .map(
            |(node_id, node)| crate::cavity::constrained::ConstrainedCavityNode {
                node_id: node_id as u32,
                coordinates_m: node.coordinates_m,
            },
        )
        .collect::<Vec<_>>();
    let refill = super::super::cavity::refill_side(&cavity, &cavity_nodes, &topology, 0, &mut work)
        .unwrap()
        .expect("the facet fallback should refill the large non-star tetrahelix");
    assert_eq!(refill.len(), TETRAHEDRA);
    let mut reordered_nodes = cavity_nodes.clone();
    reordered_nodes.reverse();
    let mut repeated_work =
        FacetRecoveryWork::new(DelaunayFacetRecoveryOptions::default(), &NeverCancelled);
    let repeated = super::super::cavity::refill_side(
        &cavity,
        &reordered_nodes,
        &topology,
        0,
        &mut repeated_work,
    )
    .unwrap()
    .expect("reordered cavity nodes should produce the same refill");
    assert_eq!(refill, repeated);

    let options = crate::cavity::constrained::ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..crate::cavity::constrained::ConstrainedCavityRefillOptions::default()
    };
    for budget in [
        crate::cavity::constrained::ConstrainedCavityRefillBudget {
            maximum_candidate_tetrahedra: 1,
            ..crate::cavity::constrained::ConstrainedCavityRefillBudget::default()
        },
        crate::cavity::constrained::ConstrainedCavityRefillBudget {
            maximum_candidate_evaluations: 1,
            ..crate::cavity::constrained::ConstrainedCavityRefillBudget::default()
        },
        crate::cavity::constrained::ConstrainedCavityRefillBudget {
            maximum_search_attempts: 1,
            ..crate::cavity::constrained::ConstrainedCavityRefillBudget::default()
        },
    ] {
        assert!(matches!(
            crate::cavity::constrained::retriangulate_constrained_cavity_from_nodes(
                &cavity,
                &cavity_nodes,
                options,
                budget,
                &NeverCancelled,
            ),
            Err(crate::cavity::constrained::ConstrainedCavityRefillError::ResourceLimit { .. })
        ));
    }

    struct CancelDuringRefill;
    impl MeshingCancellationSignal for CancelDuringRefill {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    assert_eq!(
        crate::cavity::constrained::retriangulate_constrained_cavity_from_nodes(
            &cavity,
            &cavity_nodes,
            options,
            crate::cavity::constrained::ConstrainedCavityRefillBudget {
                cancellation_check_interval: 1,
                ..crate::cavity::constrained::ConstrainedCavityRefillBudget::default()
            },
            &CancelDuringRefill,
        )
        .unwrap_err(),
        crate::cavity::constrained::ConstrainedCavityRefillError::Cancelled
    );
}

fn reflect_across_face(
    point: [f64; 3],
    first: [f64; 3],
    second: [f64; 3],
    third: [f64; 3],
) -> [f64; 3] {
    let subtract = |left: [f64; 3], right: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|axis| left[axis] - right[axis])
    };
    let left = subtract(second, first);
    let right = subtract(third, first);
    let normal = [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ];
    let squared_length = normal.into_iter().map(|value| value * value).sum::<f64>();
    let distance_scale = subtract(point, first)
        .into_iter()
        .zip(normal)
        .map(|(left, right)| left * right)
        .sum::<f64>()
        / squared_length;
    std::array::from_fn(|axis| point[axis] - 2.0 * distance_scale * normal[axis])
}
