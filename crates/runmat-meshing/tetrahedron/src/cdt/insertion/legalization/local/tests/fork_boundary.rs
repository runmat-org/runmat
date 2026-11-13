//! Reduced original double-thickness fork neighborhood, before subdivision.
//! Exact original coordinates and construction provenance are retained.
use super::*;

#[test]
fn fork_coplanar_hull_cavity_recovers_the_strongly_gabriel_edge() {
    let nodes = vec![
        DelaunayVolumeNode {
            identity: StableDigest::from_bytes([
                2, 128, 216, 113, 98, 66, 211, 64, 37, 190, 54, 158, 250, 76, 220, 32, 150, 83,
                204, 129, 151, 81, 5, 139, 200, 171, 214, 70, 229, 151, 147, 91,
            ]),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([
                0.7706721243303036,
                0.0491925522510428,
                0.34345345944166183,
            ]),
            construction: runmat_meshing_core::quality::predicate::Point3Construction::Segment {
                endpoints: [
                    [0.7834244556725025, 0.11448448523879051, 0.34345345944166183],
                    [
                        0.7387039512395859,
                        -0.11448448523879051,
                        0.34345345944166183,
                    ],
                ],
                parameter_numerator: 73,
                parameter_exponent: 8,
            },
        },
        DelaunayVolumeNode {
            identity: StableDigest::from_bytes([
                7, 49, 23, 152, 114, 181, 214, 168, 235, 15, 123, 199, 216, 168, 36, 47, 131, 0,
                72, 250, 129, 101, 216, 103, 183, 162, 89, 61, 175, 79, 164, 22,
            ]),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([
                0.7625548876821995,
                0.11448448523879051,
                0.19080747663974762,
            ]),
            construction: runmat_meshing_core::quality::predicate::Point3Construction::Explicit,
        },
        DelaunayVolumeNode {
            identity: StableDigest::from_bytes([
                73, 188, 43, 154, 220, 120, 250, 84, 214, 164, 40, 111, 8, 183, 84, 62, 76, 110,
                105, 69, 174, 67, 29, 20, 248, 16, 196, 120, 37, 137, 230, 166,
            ]),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([
                0.7237971164286137,
                0.11448448523879051,
                0.19080747663974762,
            ]),
            construction: runmat_meshing_core::quality::predicate::Point3Construction::Explicit,
        },
        DelaunayVolumeNode {
            identity: StableDigest::from_bytes([
                62, 34, 234, 21, 114, 118, 187, 203, 210, 2, 159, 242, 184, 138, 205, 111, 205, 93,
                233, 51, 223, 41, 118, 162, 131, 241, 59, 10, 73, 187, 145, 140,
            ]),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([
                0.7387039512395859,
                0.11448448523879051,
                0.34345345944166183,
            ]),
            construction: runmat_meshing_core::quality::predicate::Point3Construction::Explicit,
        },
        DelaunayVolumeNode {
            identity: StableDigest::from_bytes([
                61, 59, 224, 82, 87, 200, 73, 165, 155, 50, 150, 17, 226, 213, 55, 110, 13, 66, 67,
                26, 156, 52, 14, 168, 95, 41, 249, 43, 38, 218, 5, 94,
            ]),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([
                0.7387039512395859,
                0.041590066903154366,
                0.34345345944166183,
            ]),
            construction: runmat_meshing_core::quality::predicate::Point3Construction::Segment {
                endpoints: [
                    [0.7387039512395859, 0.11448448523879051, 0.34345345944166183],
                    [
                        0.7387039512395859,
                        -0.11448448523879051,
                        0.34345345944166183,
                    ],
                ],
                parameter_numerator: 163,
                parameter_exponent: 9,
            },
        },
        DelaunayVolumeNode {
            identity: StableDigest::from_bytes([
                189, 97, 142, 162, 113, 179, 169, 218, 116, 98, 247, 189, 204, 29, 18, 147, 217,
                82, 255, 52, 0, 114, 81, 181, 223, 9, 135, 116, 239, 66, 108, 155,
            ]),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([
                0.6939834505319595,
                0.11448448523879051,
                0.34345345944166183,
            ]),
            construction: runmat_meshing_core::quality::predicate::Point3Construction::Explicit,
        },
    ];
    let original = canonical_neighborhood(
        nodes,
        vec![
            ([0, 1, 2, 4], None),
            ([0, 1, 3, 2], None),
            ([0, 2, 5, 4], None),
            ([3, 0, 2, 5], None),
        ],
    );
    let options = DelaunayInsertionOptions::default();
    validate_delaunay_volume_topology(&original, options, &NeverCancelled).unwrap_err();
    let mut work = InsertionWork::new(options, &NeverCancelled);
    let repaired = restore(original.clone(), &BTreeSet::new(), true, &mut work).unwrap();
    assert_eq!(repaired.nodes, original.nodes);
    validate_delaunay_volume_topology(&repaired, options, &NeverCancelled).unwrap();
}

#[test]
fn fork_original_neighborhood_passes_independent_empty_sphere_admission() {
    let captured = captured_neighborhood();
    assert_eq!(
        (captured.nodes.len(), captured.tetrahedra.len()),
        (905, 3572)
    );
    let original = canonical_neighborhood(captured.nodes, captured.tetrahedra);
    let options = DelaunayInsertionOptions::default();
    crate::cdt::insertion::validate_constrained_delaunay_volume_topology(
        &original,
        &[],
        options,
        &NeverCancelled,
    )
    .unwrap_err();
    let mut work = InsertionWork::new(options, &NeverCancelled);
    let repaired = restore(original.clone(), &BTreeSet::new(), true, &mut work).unwrap();
    assert_eq!(repaired.nodes, original.nodes);
    crate::cdt::insertion::validate_constrained_delaunay_volume_topology(
        &repaired,
        &[],
        options,
        &NeverCancelled,
    )
    .unwrap();

    validate_delaunay_volume_topology(&repaired, options, &NeverCancelled).unwrap();
}

fn canonical_neighborhood(
    nodes: Vec<DelaunayVolumeNode>,
    tetrahedra: Vec<([u32; 4], Option<runmat_geometry_core::PersistentEntityId>)>,
) -> DelaunayVolumeTopology {
    let mut order = (0..nodes.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| nodes[*index].identity);
    let mut remap = vec![0; nodes.len()];
    for (canonical, original) in order.iter().copied().enumerate() {
        remap[original] = canonical as u32;
    }
    build_delaunay_volume_topology_with_regions(
        order.into_iter().map(|index| nodes[index]).collect(),
        tetrahedra
            .into_iter()
            .map(|(vertices, region)| (vertices.map(|vertex| remap[vertex as usize]), region))
            .collect(),
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

#[derive(serde::Deserialize)]
struct CapturedNeighborhood {
    nodes: Vec<DelaunayVolumeNode>,
    tetrahedra: Vec<([u32; 4], Option<runmat_geometry_core::PersistentEntityId>)>,
}

fn captured_neighborhood() -> CapturedNeighborhood {
    serde_json::from_str(include_str!("fixtures/fork-double-neighborhood.json")).unwrap()
}

#[test]
fn original_fork_four_to_four_rotation_preserves_nodes_and_cavity_boundary() {
    let mut captured = captured_neighborhood();
    captured.nodes.sort_by_key(|node| node.identity);
    let nodes = [28, 226, 776, 852, 481, 817]
        .map(|index| captured.nodes[index])
        .to_vec();
    let requested = [nodes[0].identity, nodes[4].identity];
    let original = canonical_neighborhood(
        nodes,
        vec![
            ([0, 1, 2, 3], None),
            ([0, 2, 5, 3], None),
            ([1, 4, 2, 3], None),
            ([2, 4, 5, 3], None),
        ],
    );
    let options = DelaunayInsertionOptions::default();
    validate_delaunay_volume_topology(&original, options, &NeverCancelled).unwrap_err();
    let mut work = InsertionWork::new(options, &NeverCancelled);
    let repaired = rotation::first_candidate(
        &original,
        &BTreeSet::new(),
        &BTreeSet::new(),
        search_state::seed(&original, &mut work).unwrap(),
        &mut work,
    )
    .unwrap()
    .expect("the original four-tet cavity requires an admissible rotation")
    .0;
    assert_eq!(repaired.nodes, original.nodes);
    assert_eq!(repaired.tetrahedra.len(), 4);
    let boundary = |topology: &DelaunayVolumeTopology| {
        topology
            .incidence
            .boundary_facets
            .iter()
            .map(|facet| facet.vertex_indices)
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(boundary(&repaired), boundary(&original));
    validate_delaunay_volume_topology(&repaired, options, &NeverCancelled).unwrap();
    let mut edge = requested.map(|identity| {
        repaired
            .nodes
            .binary_search_by_key(&identity, |node| node.identity)
            .unwrap() as u32
    });
    edge.sort_unstable();
    assert!(topology_has_edge(&repaired, edge));
}
