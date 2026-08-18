use runmat_meshing_core::{
    contracts::{
        MeshingStage, PlcFacet, PlcNode, PlcProtectedEdge, PlcValidationSummary, StageEvidence,
    },
    NeverCancelled,
};

use super::*;
use crate::cdt::{build_delaunay_volume_point_set, DelaunayPointSetOptions};

fn tetrahedron_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "tetrahedron".to_owned(),
        nodes: vec![
            node("0", [0.0, 0.0, 0.0]),
            node("1", [1.0, 0.0, 0.0]),
            node("2", [0.0, 1.0, 0.0]),
            node("3", [0.0, 0.0, 1.0]),
        ],
        facets: vec![
            facet("f0", ["0", "2", "1"]),
            facet("f1", ["0", "1", "3"]),
            facet("f2", ["1", "2", "3"]),
            facet("f3", ["2", "0", "3"]),
        ],
        protected_edges: vec![PlcProtectedEdge {
            edge_id: plc_id("protected:0-1"),
            node_ids: [plc_id("0"), plc_id("1")],
            source_edge_id: id(MeshingStage::CurveMesh, "curve:0"),
            cad_curve_boundary: None,
        }],
        validation: PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
    }
}

#[test]
fn constraint_build_is_canonical_and_retains_plc_provenance() {
    let plc = tetrahedron_plc();
    let expected =
        build_delaunay_constraints(&plc, DelaunayConstraintOptions::default(), &NeverCancelled)
            .unwrap();
    let mut permuted = plc;
    permuted.nodes.reverse();
    permuted.facets.reverse();
    let actual = build_delaunay_constraints(
        &permuted,
        DelaunayConstraintOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(actual, expected);
    assert_eq!(actual.nodes.len(), 4);
    assert_eq!(actual.segments.len(), 6);
    assert_eq!(actual.facets.len(), 4);
    assert!(actual
        .nodes
        .iter()
        .all(|node| node.identity != StableDigest::ZERO));
    assert_eq!(
        actual
            .nodes
            .iter()
            .find(|node| node.source_node_id.id == "0")
            .unwrap()
            .identity,
        StableDigest::from_bytes([
            0x12, 0xce, 0x0c, 0x2a, 0x9c, 0xa3, 0x14, 0x29, 0xb8, 0x6b, 0xa1, 0xbd, 0xfc, 0xf3,
            0x58, 0x71, 0x7c, 0x0c, 0xe9, 0x7d, 0x4f, 0x4e, 0xab, 0x32, 0xf8, 0x8d, 0xe4, 0x44,
            0x81, 0x53, 0x19, 0xa2,
        ])
    );
    let protected = actual
        .segments
        .iter()
        .find(|segment| segment.protected_edge_id.is_some())
        .unwrap();
    assert_eq!(protected.protected_edge_id, Some(plc_id("protected:0-1")));
    assert_eq!(
        protected.source_edge_id,
        Some(id(MeshingStage::CurveMesh, "curve:0"))
    );

    let topology = build_delaunay_volume_point_set(
        actual.volume_nodes(),
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(topology.tetrahedra.len(), 1);
    assert_eq!(topology.incidence.boundary_facets.len(), 4);
}

#[test]
fn constraint_build_rejects_invalid_plc_and_noncanonical_identity() {
    let mut invalid = tetrahedron_plc();
    invalid.validation.watertight = false;
    assert_eq!(
        build_delaunay_constraints(
            &invalid,
            DelaunayConstraintOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayConstraintErrorKind::InvalidPlc
    );

    let mut empty_identity = tetrahedron_plc();
    empty_identity.nodes[0].node_id.id.clear();
    for facet in &mut empty_identity.facets {
        for node_id in &mut facet.node_ids {
            if node_id.id == "0" {
                node_id.id.clear();
            }
        }
    }
    empty_identity.protected_edges[0].node_ids[0].id.clear();
    assert_eq!(
        build_delaunay_constraints(
            &empty_identity,
            DelaunayConstraintOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayConstraintErrorKind::InvalidIdentity
    );
}

#[test]
fn constraint_validation_rejects_tampered_canonical_state() {
    let constraints = build_delaunay_constraints(
        &tetrahedron_plc(),
        DelaunayConstraintOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    let mut unordered_nodes = constraints.clone();
    unordered_nodes.nodes.swap(0, 1);
    assert_eq!(
        validate_delaunay_constraints(
            &unordered_nodes,
            DelaunayConstraintOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayConstraintErrorKind::InvalidIdentity
    );

    let mut incomplete_provenance = constraints.clone();
    incomplete_provenance.segments[0].protected_edge_id = Some(plc_id("protected:tampered"));
    assert_eq!(
        validate_delaunay_constraints(
            &incomplete_provenance,
            DelaunayConstraintOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayConstraintErrorKind::InvalidPlc
    );

    let mut degenerate_facet = constraints;
    degenerate_facet.facets[0].vertex_indices[1] = degenerate_facet.facets[0].vertex_indices[0];
    assert_eq!(
        validate_delaunay_constraints(
            &degenerate_facet,
            DelaunayConstraintOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayConstraintErrorKind::InvalidPlc
    );
}

struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn constraint_build_enforces_inventory_limits_and_cancellation() {
    let options = DelaunayConstraintOptions {
        maximum_segments: 5,
        ..DelaunayConstraintOptions::default()
    };
    assert_eq!(
        build_delaunay_constraints(&tetrahedron_plc(), options, &NeverCancelled)
            .unwrap_err()
            .kind,
        DelaunayConstraintErrorKind::ResourceLimit
    );
    assert_eq!(
        build_delaunay_constraints(
            &tetrahedron_plc(),
            DelaunayConstraintOptions {
                cancellation_check_interval: 1,
                ..DelaunayConstraintOptions::default()
            },
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayConstraintErrorKind::Cancelled
    );
}

fn node(node_id: &str, coordinates_m: [f64; 3]) -> PlcNode {
    PlcNode {
        node_id: plc_id(node_id),
        coordinates_m,
    }
}

fn facet(facet_id: &str, nodes: [&str; 3]) -> PlcFacet {
    PlcFacet {
        facet_id: plc_id(facet_id),
        node_ids: nodes.map(plc_id),
        source_face_id: id(MeshingStage::SurfaceMesh, facet_id),
        material_interface_ids: Vec::new(),
    }
}

fn plc_id(value: &str) -> TopologyEntityId {
    id(MeshingStage::ProtectedBoundaryComplex, value)
}

fn id(stage: MeshingStage, value: &str) -> TopologyEntityId {
    TopologyEntityId {
        stage,
        id: value.to_owned(),
    }
}
