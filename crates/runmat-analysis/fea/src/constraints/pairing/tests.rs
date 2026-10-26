use super::*;
use runmat_analysis_core::RegionTarget;
use runmat_execution::Digest;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

pub(super) fn target(ids: Vec<StableDigest>) -> ResolvedConstraintTarget {
    ResolvedConstraintTarget {
        authored: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("pair")),
        matched_entities: Vec::new(),
        matched_node_identities: ids,
        selection_digest: Digest::sha256("pair"),
    }
}

#[test]
fn streaming_matching_keeps_identity_ties_and_does_not_skip_later_errors() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let [a, b, c, d] = [1, 2, 3, 4].map(|id| StableDigest::from_bytes([id; 32]));
    let epsilon = matching_tolerance(&mesh) / 4.0;
    let values = [
        (a, [-epsilon, 0.0, 0.0]),
        (b, [epsilon, 0.0, 0.0]),
        (c, [0.0; 3]),
        (d, [epsilon, 0.0, 0.0]),
    ];
    let nodes = NodeCoordinates::derive(values.into_iter(), values.len()).unwrap();
    let secondary = target(vec![c, d]);
    for primary in [target(vec![a, b]), target(vec![b, a])] {
        let pairs = pair_matching(
            &mesh,
            &nodes,
            &primary,
            &secondary,
            &PeriodicTransform::Identity,
            "tie",
        )
        .unwrap();
        assert_eq!(pairs[0].primary_weights[0].primary_node_identity, a);
        assert_eq!(pairs[1].primary_weights[0].primary_node_identity, b);
    }
    let nodes = NodeCoordinates::derive(values.into_iter().filter(|(id, _)| *id != b), 3).unwrap();
    // A valid nearest candidate must not short-circuit validation of later candidates.
    assert!(pair_matching(
        &mesh,
        &nodes,
        &target(vec![a, b]),
        &secondary,
        &PeriodicTransform::Identity,
        "missing"
    )
    .is_err());
}

#[test]
fn quadratic_weights_preserve_negative_values_and_affine_trace() {
    let points = [
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0.5, 0., 0.],
        [0.5, 0.5, 0.],
        [0., 0.5, 0.],
    ];
    let nodes: Vec<_> = points
        .into_iter()
        .enumerate()
        .map(|(i, p)| (StableDigest::from_bytes([(i + 1) as u8; 32]), p))
        .collect();
    let weights =
        interpolation_weights(BoundaryTriangleOrder::Tri6, &nodes, [0.2, 0.3, 0.5]).unwrap();
    let expected = [-0.12, -0.12, 0., 0.24, 0.6, 0.4];
    for (i, value) in expected.into_iter().enumerate() {
        let actual = weights
            .iter()
            .find(|w| w.primary_node_identity == nodes[i].0)
            .map_or(0., |w| w.weight);
        assert!((actual - value).abs() < 1e-14);
    }
    let sum: f64 = weights.iter().map(|w| w.weight).sum();
    assert!((sum - 1.).abs() < 1e-14);
    let trace: f64 = weights
        .iter()
        .map(|w| {
            let p = nodes
                .iter()
                .find(|n| n.0 == w.primary_node_identity)
                .unwrap()
                .1;
            w.weight * (2. + 3. * p[0] - 4. * p[1])
        })
        .sum();
    assert!((trace - (2. + 3. * 0.3 - 4. * 0.5)).abs() < 1e-14);
}

#[test]
fn counted_indexes_reject_duplicate_or_mismatched_inventories() {
    let a = StableDigest::from_bytes([1; 32]);
    assert!(NodeCoordinates::derive([(a, [0.; 3]), (a, [1.; 3])].into_iter(), 2).is_err());
    assert!(NodeCoordinates::derive([(a, [0.; 3])].into_iter(), 0).is_err());
    assert!(NodeCoordinates::derive([(a, [0.; 3])].into_iter(), 2).is_err());
}

#[test]
fn empty_pairing_targets_reject_without_indexing_an_absent_secondary() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let valid = target(vec![mesh.topology.nodes[0].stable_identity]);
    let empty = target(Vec::new());
    for (primary, secondary) in [(&valid, &empty), (&empty, &valid)] {
        for policy in [
            PairingPolicy::Matching,
            PairingPolicy::GeometricTolerance(runmat_analysis_core::Length::new(0.02)),
        ] {
            assert!(matches!(
                pair_targets(
                    &mesh,
                    primary,
                    secondary,
                    &policy,
                    &PeriodicTransform::Identity,
                    "empty"
                ),
                Err(ConstraintPlanError::EmptyTarget { .. })
            ));
        }
    }
}

#[test]
fn nonconformal_quadratic_pairing_preserves_full_surface_trace_and_row_identity() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    // Pairing-owner fixture: off-surface secondary sample, no volume solve is claimed.
    mesh.topology.nodes[3].coordinates_m = runmat_types::physical::Quantity::<
        runmat_types::physical::kind::Position,
        _,
    >::new([0.3, 0.5, 0.01]);
    let primary = super::super::target::resolve_nodal_target(
        &mesh,
        &RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:0")),
        "trace",
    )
    .unwrap();
    let secondary =
        super::super::target::resolve_nodal_target(&mesh, &RegionTarget::point("node:4"), "trace")
            .unwrap();
    let policy = PairingPolicy::GeometricTolerance(runmat_analysis_core::Length::new(0.02));
    let pairs = pair_targets(
        &mesh,
        &primary,
        &secondary,
        &policy,
        &PeriodicTransform::Identity,
        "trace",
    )
    .unwrap();
    assert!((pairs[0].geometric_error_meters - 0.01).abs() < 1e-14);
    assert!(pairs[0].primary_weights.iter().any(|w| w.weight < 0.));
    let rows = pairing_rows("trace", &pairs, &PeriodicTransform::Identity, false).unwrap();
    for row in &rows {
        let balance: f64 = row
            .terms
            .iter()
            .map(|term| {
                let CanonicalStructuralDofOwner::StructuralNode(id) = term.owner else {
                    panic!("surface node")
                };
                let p = mesh
                    .topology
                    .nodes
                    .iter()
                    .find(|n| n.stable_identity == id)
                    .unwrap()
                    .coordinates_m
                    .canonical_value();
                // Independent degree-two trace at both original nodes and the new sample.
                term.coefficient
                    * (2. + 3. * p[0] - 4. * p[1] + p[0] * p[0] + p[0] * p[1] + p[1] * p[1])
            })
            .sum();
        assert!(balance.abs() < 1e-13);
    }
    mesh.topology.nodes.reverse();
    mesh.topology.boundary_faces.reverse();
    let reordered = pair_targets(
        &mesh,
        &primary,
        &secondary,
        &policy,
        &PeriodicTransform::Identity,
        "trace",
    )
    .unwrap();
    assert_eq!(pairs, reordered);
    assert_eq!(
        rows,
        pairing_rows("trace", &reordered, &PeriodicTransform::Identity, false).unwrap()
    );
}
