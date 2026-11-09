use super::*;
use crate::cdt::{build_delaunay_volume_topology, DelaunayTopologyOptions, DelaunayVolumeNode};
use runmat_meshing_core::{quality::predicate::Point3Construction, NeverCancelled};
#[derive(serde::Deserialize)]
struct Fixture {
    coordinates_m_bits: [[u64; 3]; 4],
    source_vertices: [usize; 3],
    split_node: usize,
    split_endpoints: [usize; 2],
    parameter_numerator: u64,
    parameter_exponent: u8,
    source_edge_membership: [Vec<usize>; 4],
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A captured four-node facet with a rounded exact midpoint recovers both support
//   triangles, each original boundary edge once and the interior diagonal twice. This local fixture
//   does not discharge original-model meshing, physical quality, resource or scientific gates.
#[test]
fn captured_rounded_facet_recovers_every_original_edge_without_a_degenerate_ear() {
    let fixture: Fixture = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../core/tests/fixtures/rounded-facet.json"
    )))
    .unwrap();
    let mut nodes = fixture
        .coordinates_m_bits
        .into_iter()
        .enumerate()
        .map(|(index, bits)| {
            DelaunayVolumeNode::explicit(
                StableDigest::from_bytes([(index + 1) as u8; 32]),
                bits.map(f64::from_bits),
            )
        })
        .collect::<Vec<_>>();
    nodes[fixture.split_node].construction = Point3Construction::Segment {
        endpoints: fixture
            .split_endpoints
            .map(|index| nodes[index].coordinates_m.canonical_value()),
        parameter_numerator: fixture.parameter_numerator,
        parameter_exponent: fixture.parameter_exponent,
    };
    let boundary = nodes.iter().map(|node| node.identity).collect::<Vec<_>>();
    let source = fixture.source_vertices.map(|index| nodes[index].identity);
    let memberships = boundary
        .iter()
        .copied()
        .zip(
            fixture
                .source_edge_membership
                .map(|edges| edges.into_iter().collect::<BTreeSet<_>>()),
        )
        .collect::<BTreeMap<_, _>>();
    let mut apex = nodes[0].coordinates_m.canonical_value();
    apex[0] += 0.01;
    nodes.push(DelaunayVolumeNode::explicit(
        StableDigest::from_bytes([5; 32]),
        apex,
    ));
    let topology = build_delaunay_volume_topology(
        nodes,
        vec![[0, 1, 3, 4], [1, 2, 3, 4]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let view = FacetSupportView {
        topology: &topology,
        segments: &[],
    };
    let mut work = FacetRecoveryWork::new(
        super::super::DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    );
    let triangles = triangulate_boundary(
        &view,
        boundary.clone(),
        source,
        &memberships,
        71194,
        &mut work,
    )
    .unwrap();
    assert_eq!(triangles.len(), 2);
    let mut edges = BTreeMap::new();
    for triangle in triangles {
        for mut edge in triangle_edges(triangle.node_identities) {
            edge.sort_unstable();
            *edges.entry(edge).or_insert(0usize) += 1;
        }
    }
    for index in 0..boundary.len() {
        let mut edge = [boundary[index], boundary[(index + 1) % boundary.len()]];
        edge.sort_unstable();
        assert_eq!(
            edges.remove(&edge),
            Some(1),
            "every original edge survives exactly once"
        );
    }
    assert_eq!(edges.len(), 1);
    assert!(edges.values().all(|count| *count == 2));
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An acute tilted facet with one constructed node on each source edge takes its
//   exact 2D Delaunay support, the medial subdivision, even when the local topology holds a
//   non-Delaunay diagonal. This local fixture does not discharge original-model meshing,
//   physical quality, resource or scientific gates.
#[test]
fn facet_support_is_the_exact_delaunay_triangulation_of_the_boundary_polygon() {
    let corners = [[0.0, 0.0, 0.0], [1.0, 0.1, 0.3], [0.4, 0.9, 0.7]];
    let identity = |index: usize| StableDigest::from_bytes([(index + 1) as u8; 32]);
    // Boundary order: corner 0, midpoint 0-1, corner 1, midpoint 1-2, corner 2, midpoint 2-0.
    let mut nodes = Vec::new();
    for edge in 0..3 {
        let [first, last] = [corners[edge], corners[(edge + 1) % 3]];
        nodes.push(DelaunayVolumeNode::explicit(identity(2 * edge), first));
        let mut midpoint = DelaunayVolumeNode::explicit(
            identity(2 * edge + 1),
            [0, 1, 2].map(|axis| 0.5 * (first[axis] + last[axis])),
        );
        midpoint.construction = Point3Construction::Segment {
            endpoints: [first, last],
            parameter_numerator: 1,
            parameter_exponent: 1,
        };
        nodes.push(midpoint);
    }
    nodes.push(DelaunayVolumeNode::explicit(identity(6), [0.3, 0.2, 2.0]));
    // The local cells hold the diagonals 1-3, 1-5 and the non-Delaunay 1-4 but not the boundary
    // edge 4-5, so the topology does not cover the facet, and ear clipping that prefers existing
    // diagonals would close on 1-4.
    let topology = build_delaunay_volume_topology(
        nodes,
        vec![[0, 1, 5, 6], [1, 2, 3, 6], [1, 3, 4, 6]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let view = FacetSupportView {
        topology: &topology,
        segments: &[],
    };
    let mut work = FacetRecoveryWork::new(
        super::super::DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    );
    let boundary = (0..6).map(identity).collect::<Vec<_>>();
    let memberships = (0..6)
        .map(|index| {
            let edges = if index % 2 == 0 {
                [index / 2, (index / 2 + 2) % 3].into_iter().collect()
            } else {
                [index / 2].into_iter().collect()
            };
            (identity(index), edges)
        })
        .collect::<BTreeMap<_, BTreeSet<usize>>>();
    let source = [0, 2, 4].map(identity);
    let projection = projection_axes(&view, source, 1, &mut work).unwrap();
    let sign = orientation(&view, source, projection, 1, &mut work).unwrap();
    let triangles = delaunay::triangulate_delaunay(
        &view,
        &boundary,
        source,
        delaunay::FacetProjection {
            axes: projection,
            sign,
        },
        &memberships,
        1,
        &mut work,
    )
    .unwrap()
    .expect("an acute facet has a unique Delaunay support");
    let mut found = triangles
        .iter()
        .map(|triangle| {
            let mut nodes = triangle.node_identities;
            nodes.sort_unstable();
            nodes
        })
        .collect::<Vec<_>>();
    found.sort_unstable();
    let mut expected = [[0, 1, 5], [1, 2, 3], [3, 4, 5], [1, 3, 5]]
        .map(|triangle| triangle.map(identity))
        .to_vec();
    for triangle in &mut expected {
        triangle.sort_unstable();
    }
    expected.sort_unstable();
    assert_eq!(found, expected);
    // The full support path takes the same triangles, not the ears on existing diagonals.
    let support =
        triangulate_boundary(&view, boundary, source, &memberships, 1, &mut work).unwrap();
    let mut support = support
        .iter()
        .map(|triangle| {
            let mut nodes = triangle.node_identities;
            nodes.sort_unstable();
            nodes
        })
        .collect::<Vec<_>>();
    support.sort_unstable();
    assert_eq!(support, expected);
}
