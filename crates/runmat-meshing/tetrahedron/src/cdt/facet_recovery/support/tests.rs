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
