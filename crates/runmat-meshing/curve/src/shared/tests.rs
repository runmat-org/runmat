use std::f64::consts::{PI, TAU};

use runmat_geometry_core::ParameterRange;

use super::*;

#[test]
fn shared_curve_contract_covers_exact_edges_and_face_uses() {
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    let mesh = circle_mesh(&topology);
    mesh.validate_against(&topology).unwrap();

    let mut reordered = mesh.clone();
    reordered.edges[0].nodes.swap(0, 1);
    assert!(reordered.validate_against(&topology).is_err());

    let mut missing_use = mesh.clone();
    missing_use.edges[0].face_uses.clear();
    assert!(missing_use.validate_against(&topology).is_err());

    let mut tampered_identity = mesh.clone();
    tampered_identity.edges[0].nodes[1].node_id = tampered_identity.edges[0].nodes[0].node_id;
    assert!(tampered_identity.validate_against(&topology).is_err());

    let mut missed_bound = mesh;
    missed_bound.edges[0].achieved.maximum_chordal_deviation_m = 0.2;
    assert!(missed_bound.validate_against(&topology).is_err());
}

#[test]
fn shared_curve_codec_is_canonical_bounded_and_topology_admitted() {
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    let mesh = circle_mesh(&topology);
    let encoded = encode_shared_curve_mesh(&mesh, &topology).unwrap();
    let decoded = decode_shared_curve_mesh(&encoded, &topology).unwrap();
    assert_eq!(decoded, mesh);
    assert_eq!(
        encode_shared_curve_mesh(&decoded, &topology).unwrap(),
        encoded
    );

    let mut trailing = encoded.clone();
    trailing.push(0);
    assert!(decode_shared_curve_mesh(&trailing, &topology).is_err());

    let mut corrupt = encoded;
    corrupt[0] ^= 1;
    assert!(decode_shared_curve_mesh(&corrupt, &topology).is_err());

    let encoded = encode_shared_curve_mesh(&mesh, &topology).unwrap();
    assert!(super::codec::decode_with_byte_limit(&encoded, &topology, encoded.len() - 1).is_err());

    let mut different_topology = topology.clone();
    different_topology.edges.clear();
    different_topology.coedges.clear();
    assert!(decode_shared_curve_mesh(&encoded, &different_topology,).is_err());
}

fn circle_mesh(topology: &runmat_geometry_core::ExactBRepTopology) -> SharedCurveMesh {
    let edge = &topology.edges[0];
    let coedge = &topology.coedges[0];
    let parameter_range = ParameterRange {
        start: 0.0,
        end: TAU,
    };
    let nodes = [
        (0.0, 0.0, [1.0, 0.0, 0.0], edge.start_vertex_id.clone()),
        (PI, PI, [-1.0, 0.0, 0.0], None),
        (TAU, TAU, [1.0, 0.0, 0.0], edge.end_vertex_id.clone()),
    ]
    .into_iter()
    .map(
        |(parameter, arc_length_m, coordinates_m, source_vertex_id)| SharedCurveNode {
            node_id: shared_curve_node_id(&edge.id, parameter),
            source_vertex_id,
            parameter,
            arc_length_m,
            coordinates_m,
        },
    )
    .collect();
    SharedCurveMesh {
        schema_version: SHARED_CURVE_MESH_SCHEMA_VERSION,
        edges: vec![SharedCurve {
            source_edge_id: edge.id.clone(),
            parameter_range,
            nodes,
            face_uses: vec![SharedCurveFaceUse {
                coedge_id: coedge.id.clone(),
                face_id: coedge.face_id.clone(),
                orientation: coedge.orientation,
                seam_image: coedge.seam_image,
                node_uv: vec![[0.0, 0.0], [PI, 0.0], [TAU, 0.0]],
            }],
            requested: CurveResolutionPolicy {
                maximum_chordal_deviation_m: 0.1,
                maximum_tangent_change_rad: PI,
                minimum_metric_edge_length: 0.5,
                maximum_metric_edge_length: 4.0,
            },
            achieved: CurveResolutionEvidence {
                maximum_chordal_deviation_m: 0.05,
                maximum_tangent_change_rad: PI,
                minimum_metric_edge_length: 2.0,
                maximum_metric_edge_length: 2.0,
            },
        }],
    }
}
