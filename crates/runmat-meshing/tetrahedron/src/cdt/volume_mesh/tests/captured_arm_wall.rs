//! The 20 surface nodes, 45 segments and 26 facets of the motor-pod arm (`arm_study.step` at a
//! 3 mm request) within 1.8 mm of one corner of a thin-wall facet. The wall faces a 0.2 mm slot.
//! The facet and its neighbour across one of its segment edges are nearly coplanar: the
//! neighbour's far corner lies 3.5 um off the facet's plane. The facet needs an interior split,
//! and its 2:1:1 split node encroaches that segment edge and lies inside the neighbour's
//! circumcircle, so no constrained Delaunay complex holds the neighbour and the node. Facet
//! recovery splits the encroached segment edge first.
use super::*;
use crate::cdt::{
    DelaunayConstraintFacet, DelaunayConstraintFacetSide, DelaunayConstraintNode,
    DelaunayConstraintSegment, DelaunayConstraints,
};
use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CapturedFacet {
    id: [u8; 32],
    chart: [u8; 32],
    vertices: [u32; 3],
    face: String,
    positive: String,
    negative: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CapturedWall {
    nodes: Vec<([u8; 32], [f64; 3])>,
    segments: Vec<[u32; 2]>,
    facets: Vec<CapturedFacet>,
}

fn side(name: &str) -> DelaunayConstraintFacetSide {
    match name {
        "exterior" => DelaunayConstraintFacetSide::Exterior,
        "void" => DelaunayConstraintFacetSide::Void,
        region => DelaunayConstraintFacetSide::Region(PersistentEntityId::new(
            PersistentEntityKind::Region,
            region.strip_prefix("region:").unwrap().to_owned(),
            Vec::new(),
        )),
    }
}

fn captured_constraints() -> DelaunayConstraints {
    let captured: CapturedWall =
        serde_json::from_str(include_str!("fixtures/arm-thin-wall-facets.json")).unwrap();
    DelaunayConstraints {
        nodes: captured
            .nodes
            .into_iter()
            .map(|(identity, coordinates_m)| DelaunayConstraintNode {
                identity: StableDigest::from_bytes(identity),
                source_vertex_id: None,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates_m),
            })
            .collect(),
        segments: captured
            .segments
            .into_iter()
            .map(|vertex_indices| DelaunayConstraintSegment {
                vertex_indices,
                source_edge_id: None,
                source_edge_parameters: None,
            })
            .collect(),
        facets: captured
            .facets
            .into_iter()
            .map(|facet| DelaunayConstraintFacet {
                facet_id: StableDigest::from_bytes(facet.id),
                chart_id: StableDigest::from_bytes(facet.chart),
                vertex_indices: facet.vertices,
                source_face_id: PersistentEntityId::new(
                    PersistentEntityKind::Face,
                    facet.face,
                    Vec::new(),
                ),
                positive_side: side(&facet.positive),
                negative_side: side(&facet.negative),
                contact_ids: Vec::new(),
            })
            .collect(),
    }
}

#[test]
fn an_interior_split_node_first_splits_the_segment_edge_it_encroaches() {
    let constraints = captured_constraints();
    let frame = frame::CanonicalVolumeFrame::from_constraints(&constraints).unwrap();
    let options = volume_options();
    let initial = crate::cdt::point_set::build_delaunay_volume_point_set_with_order(
        constraints.volume_nodes(),
        point_set_options(options),
        &NeverCancelled,
        |left, right| frame.compare_nodes(left, right),
    )
    .unwrap();
    let initial = normalize_point_set_coordinates(
        initial,
        &[],
        options.refinement.step.insertion,
        &NeverCancelled,
    )
    .unwrap();
    let segments = recover_delaunay_segments_from_point_set(
        initial,
        &constraints,
        options.carving.facet_recovery.segment_recovery,
        &NeverCancelled,
    )
    .unwrap();
    let recovered = recover_delaunay_facets(
        segments,
        &constraints,
        options.carving.facet_recovery,
        &NeverCancelled,
    )
    .expect("every facet of the thin wall must recover");
    assert!(recovered.topology.nodes.len() > constraints.nodes.len());
    crate::cdt::validate_delaunay_facet_recovery(
        &recovered,
        &constraints,
        options.carving.facet_recovery,
        &NeverCancelled,
    )
    .unwrap();
}
