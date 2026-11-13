//! Replay the authenticated full native surface inventory for recovery diagnostics.
//! Native study execution remains the authority for whole-task resource and physics gates.
use super::*;
#[path = "captured_heat/local.rs"]
mod local;
use crate::cdt::{
    DelaunayConstraintNode, DelaunayConstraintSegment, DelaunayConstraints,
    DelaunaySegmentRecoveryOptions,
};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CapturedSegments {
    source_surface_chunks: Vec<(u32, String)>,
    nodes: Vec<([u8; 32], [f64; 3])>,
    segments: Vec<[u32; 2]>,
}

#[test]
#[ignore = "requires the authenticated native heat-sink boundary diagnostic fixture"]
fn native_heat_sink_preserves_every_surface_segment() {
    replay_original_segments(true, false);
}

#[test]
#[ignore = "requires the authenticated native heat-sink boundary diagnostic fixture"]
fn native_heat_sink_physical_coordinates_preserve_every_surface_segment() {
    replay_original_segments(false, false);
}

#[test]
#[ignore = "requires the authenticated native heat-sink boundary diagnostic fixture"]
fn native_heat_sink_physical_ordering_preserves_every_surface_segment() {
    replay_original_segments(false, true);
}

fn authenticated_constraints() -> DelaunayConstraints {
    let path =
        std::env::var("RUNMAT_HEAT_SINK_SEGMENTS").expect("native captured segment fixture path");
    use sha2::{Digest, Sha256};
    assert!(std::fs::metadata(&path).unwrap().len() <= 10_000_000);
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        hex::encode(Sha256::digest(&bytes)),
        "3ab5566217d37ac41f3d493c4d681753417eacbee73969e58b129fca795c9e4b"
    );
    let captured: CapturedSegments = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(captured.source_surface_chunks.len(), 19);
    assert_eq!(
        (captured.nodes.len(), captured.segments.len()),
        (45603, 136821)
    );
    DelaunayConstraints {
        nodes: captured
            .nodes
            .into_iter()
            .map(|(id, coordinates_m)| DelaunayConstraintNode {
                identity: StableDigest::from_bytes(id),
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates_m),
                source_vertex_id: None,
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
        facets: Vec::new(),
    }
}

fn replay_original_segments(canonical: bool, canonical_order: bool) {
    let constraints = authenticated_constraints();
    let frame = frame::CanonicalVolumeFrame::from_constraints(&constraints).unwrap();
    let constraints = if canonical {
        frame.canonicalize_constraints(&constraints)
    } else {
        constraints
    };
    let options = DelaunayVolumeMeshOptions::default();
    let initial = if canonical_order {
        crate::cdt::point_set::build_delaunay_volume_point_set_with_order(
            constraints.volume_nodes(),
            point_set_options(options),
            &NeverCancelled,
            |left, right| frame.compare_nodes(left, right),
        )
    } else {
        build_delaunay_volume_point_set(
            constraints.volume_nodes(),
            point_set_options(options),
            &NeverCancelled,
        )
    }
    .unwrap();
    let initial = normalize_point_set_coordinates(
        initial,
        &[],
        options.refinement.step.insertion,
        &NeverCancelled,
    )
    .unwrap();
    let result = tracing::subscriber::with_default(
        crate::cdt::test_fixtures::diagnostics::RecoveryDiagnostics::segment_recovery(),
        || {
            recover_delaunay_segments_from_point_set(
                initial,
                &constraints,
                DelaunaySegmentRecoveryOptions::default(),
                &NeverCancelled,
            )
        },
    )
    .expect("all original heat-sink surface edges must recover simultaneously");
    assert_eq!(result.segments.len(), constraints.segments.len());
    crate::cdt::validate_delaunay_segment_recovery(
        &result,
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}
