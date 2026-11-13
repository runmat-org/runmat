//! A diametral witness does not preclude a checked Delaunay reconnection.
use super::*;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An exact five-point encroached segment reconnects using its original points and
//   independently passes physical Delaunay validation. This bounded-cavity degeneracy fixture does
//   not establish large-input recovery, asymptotic scaling, whole-task memory or original heat-sink
//   acceptance.
#[test]
fn encroached_cospherical_edge_recovers_without_subdivision() {
    // All five points lie exactly on the sphere centered at (0, 2, 0),
    // radius 2.5. The middle face contains a point strictly inside the
    // requested edge's diameter sphere. Both triangulations are Delaunay.
    let coordinates = [
        [-1.5, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [0.0, -0.5, 0.0],
        [0.0, 2.0, 2.5],
        [0.0, 2.0, -2.5],
    ];
    let constraints = DelaunayConstraints {
        nodes: coordinates
            .into_iter()
            .enumerate()
            .map(|(index, coordinates_m)| DelaunayConstraintNode {
                identity: StableDigest::from_bytes([(index + 1) as u8; 32]),
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates_m),
                source_vertex_id: None,
            })
            .collect(),
        segments: vec![DelaunayConstraintSegment {
            vertex_indices: [0, 1],
            source_edge_id: None,
            source_edge_parameters: None,
        }],
        facets: Vec::new(),
    };
    let topology = build_delaunay_volume_topology(
        constraints.volume_nodes(),
        vec![[0, 2, 3, 4], [1, 2, 4, 3]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let options = DelaunaySegmentRecoveryOptions::default();
    let [left, right] = [constraints.nodes[0].identity, constraints.nodes[1].identity];
    let mut work = RecoveryWork::new(options, &NeverCancelled);
    let witness = encroaching_point(&topology, left, right, 0, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(witness.relation, PredicateSign::Negative);
    let mut candidate = crate::cdt::mutation::TopologyMutation::from_canonical(topology.clone());
    assert!(flip::try_recover_edge_with_face_flip(
        &mut candidate,
        left,
        right,
        &[],
        &[],
        0,
        &mut work,
    )
    .unwrap());
    let candidate = candidate
        .freeze(options.insertion.topology, &NeverCancelled)
        .unwrap();
    validate_delaunay_volume_topology(&candidate, options.insertion, &NeverCancelled).unwrap();
    let recovered =
        recover_delaunay_segments(topology, &constraints, options, &NeverCancelled).unwrap();
    assert!(recovered.steiner_node_identities.is_empty());
    assert_eq!(recovered.topology.nodes, candidate.nodes);
    validate_delaunay_segment_recovery(&recovered, &constraints, options, &NeverCancelled).unwrap();
}

#[test]
fn exact_boundary_reconnection_preserves_hull_and_protected_interfaces() {
    let nodes = [
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    .into_iter()
    .enumerate()
    .map(|(i, p)| DelaunayVolumeNode::explicit(StableDigest::from_bytes([(i + 1) as u8; 32]), p))
    .collect();
    let topology = build_delaunay_volume_topology(
        nodes,
        vec![[0, 2, 3, 4], [1, 3, 2, 4]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let [left, right] = [topology.nodes[0].identity, topology.nodes[1].identity];
    let protected_edge = [topology.nodes[2].identity, topology.nodes[3].identity];
    let protected_face = [
        topology.nodes[2].identity,
        topology.nodes[3].identity,
        topology.nodes[4].identity,
    ];
    for (edges, faces) in [
        (vec![protected_edge], vec![]),
        (vec![], vec![protected_face]),
    ] {
        let mut mutation = TopologyMutation::from_canonical(topology.clone());
        let mut work =
            RecoveryWork::new(DelaunaySegmentRecoveryOptions::default(), &NeverCancelled);
        assert!(!local_reconnection::recover(
            &mut mutation,
            left,
            right,
            &edges,
            &faces,
            0,
            &mut work
        )
        .unwrap());
        assert_eq!(
            mutation
                .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
                .unwrap(),
            topology
        );
    }
    let mut mutation = TopologyMutation::from_canonical(topology);
    let mut work = RecoveryWork::new(DelaunaySegmentRecoveryOptions::default(), &NeverCancelled);
    assert!(
        local_reconnection::recover(&mut mutation, left, right, &[], &[], 0, &mut work).unwrap()
    );
    let result = mutation
        .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    assert_eq!(result.nodes.len(), 5);
    assert_eq!(result.incidence.boundary_facets.len(), 6);
    validate_delaunay_volume_topology(&result, work.options.insertion, &NeverCancelled).unwrap();
}

#[test]
fn completed_fragment_retains_depth_and_acute_corner_lineage() {
    let constraints = octahedron_constraints();
    let mut topology = TopologyMutation::from_canonical(octahedron_topology(&constraints));
    let [first, last] = [&constraints.nodes[0], &constraints.nodes[5]];
    let context = SegmentContext {
        constraint_index: 0,
        first_identity: first.identity,
        last_identity: last.identity,
        first_coordinates: first.coordinates_m.canonical_value(),
        last_coordinates: last.coordinates_m.canonical_value(),
    };
    let options = DelaunaySegmentRecoveryOptions::default();
    let mut work = RecoveryWork::new(options, &NeverCancelled);
    let mut identity_for_node = |context, midpoint, _| steiner_identity(context, midpoint);
    let mut completed = Vec::new();
    let mut nodes = Vec::new();
    let original_depth = options.maximum_split_depth - 1;
    for acute_lineage in [[true, false], [false, true], [true, true], [false, false]] {
        completed.clear();
        nodes.clear();
        let mut interval = IntervalRecoveryContext {
            segment: context,
            protected_edges: &[],
            protected_faces: &[],
            split_strong_gabriel: true,
            reported_unrecovered_interval: false,
            completed_intervals: Some(&mut completed),
            identity_for_node: &mut identity_for_node,
            work: &mut work,
        };
        recover_interval(
            &mut topology,
            DyadicNode::endpoint(first.identity, false),
            DyadicNode::endpoint(last.identity, true),
            acute_lineage,
            original_depth,
            &mut nodes,
            &mut interval,
        )
        .unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].depth, original_depth);
        assert_eq!(completed[0].acute_lineage, acute_lineage);
    }
    assert_eq!(topology.nodes().len(), constraints.nodes.len());
    let canonical = topology
        .freeze(options.insertion.topology, &NeverCancelled)
        .unwrap();
    validate_delaunay_volume_topology(&canonical, options.insertion, &NeverCancelled).unwrap();
}

fn noncospherical_neighborhood(exterior_neighbor: bool) -> DelaunayVolumeTopology {
    let mut points = vec![
        [-1.5, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [0.0, -0.5, 0.0],
        [0.0, 2.0, 2.5],
        [0.0, 2.0, -2.5],
    ];
    let mut tetrahedra = vec![[0, 2, 3, 4], [1, 2, 4, 3]];
    if exterior_neighbor {
        points.push([-1.74, 0.2, 0.0]);
        tetrahedra.push([0, 3, 4, 5]);
    }
    let nodes = points
        .into_iter()
        .enumerate()
        .map(|(index, point)| {
            DelaunayVolumeNode::explicit(StableDigest::from_bytes([(index + 1) as u8; 32]), point)
        })
        .collect();
    build_delaunay_volume_topology(
        nodes,
        tetrahedra,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

#[test]
fn noncospherical_reconnection_checks_every_changed_sphere() {
    let topology = noncospherical_neighborhood(false);
    let options = DelaunaySegmentRecoveryOptions::default();
    assert!(
        validate_delaunay_volume_topology(&topology, options.insertion, &NeverCancelled).is_err()
    );
    let [left, right] = [topology.nodes[0].identity, topology.nodes[1].identity];
    let mut mutation = TopologyMutation::from_canonical(topology);
    let mut work = RecoveryWork::new(options, &NeverCancelled);
    assert!(
        local_reconnection::recover(&mut mutation, left, right, &[], &[], 0, &mut work).unwrap()
    );
    assert_eq!(mutation.nodes().len(), 5);
    let recovered = mutation
        .freeze(options.insertion.topology, &NeverCancelled)
        .unwrap();
    // The independent validator checks all points against all tetrahedra, not
    // merely the neighbor checks used by the local candidate provider.
    validate_delaunay_volume_topology(&recovered, options.insertion, &NeverCancelled).unwrap();
}

#[test]
fn noncospherical_reconnection_rejects_an_exterior_sphere_violation() {
    let topology = noncospherical_neighborhood(true);
    let options = DelaunaySegmentRecoveryOptions::default();
    let unsafe_candidate = build_delaunay_volume_topology(
        topology.nodes.clone(),
        vec![[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2], [0, 3, 4, 5]],
        options.insertion.topology,
        &NeverCancelled,
    )
    .unwrap();
    assert!(validate_delaunay_volume_topology(
        &unsafe_candidate,
        options.insertion,
        &NeverCancelled
    )
    .is_err());
    let [left, right] = [topology.nodes[0].identity, topology.nodes[1].identity];
    let mut mutation = TopologyMutation::from_canonical(topology.clone());
    let mut work = RecoveryWork::new(options, &NeverCancelled);
    assert!(
        !local_reconnection::recover(&mut mutation, left, right, &[], &[], 0, &mut work).unwrap()
    );
    assert_eq!(
        mutation
            .freeze(options.insertion.topology, &NeverCancelled)
            .unwrap(),
        topology
    );
}
