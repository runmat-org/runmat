//! Original native surface nodes and every triangle edge of a concave tuning fork.
//! The 100 x 36 x 6 mm extruded profile has 65 mm tines and a 20 mm gap.
//! This isolates segment recovery; it does not certify facets, volume quality or physics.
use super::*;
#[path = "captured_fork/family.rs"]
mod family;
use crate::cdt::{
    DelaunayConstraintNode, DelaunayConstraintSegment, DelaunayConstraints,
    DelaunaySegmentRecoveryOptions,
};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CapturedSegments {
    source_surface_chunk_sha256: String,
    nodes: Vec<([u8; 32], [f64; 3])>,
    segments: Vec<[u32; 2]>,
}

fn original_constraints() -> DelaunayConstraints {
    let captured: CapturedSegments =
        serde_json::from_str(include_str!("fixtures/tuning-fork-segments.json")).unwrap();
    assert_eq!(
        captured.source_surface_chunk_sha256,
        "dcac9c0cf70a8b42b92494fa7248748e6c83ca1052af72587ff732af2ed7e379"
    );
    assert_eq!((captured.nodes.len(), captured.segments.len()), (424, 1266));
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
        facets: Vec::new(),
    }
}

fn recover_constraints(constraints: DelaunayConstraints) -> crate::cdt::DelaunaySegmentRecovery {
    let frame = frame::CanonicalVolumeFrame::from_constraints(&constraints).unwrap();
    let constraints = frame.canonicalize_constraints(&constraints);
    let options = volume_options();
    let initial = build_delaunay_volume_point_set(
        constraints.volume_nodes(),
        point_set_options(options),
        &NeverCancelled,
    )
    .unwrap();
    let initial = normalize_point_set_coordinates(
        initial,
        &[],
        options.refinement.step.insertion,
        &NeverCancelled,
    )
    .unwrap();
    let recovery = tracing::subscriber::with_default(
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
    .expect("all original native surface segments must recover simultaneously");
    assert_eq!(recovery.segments.len(), constraints.segments.len());
    crate::cdt::validate_delaunay_segment_recovery(
        &recovery,
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    recovery
}

#[test]
fn native_tuning_fork_preserves_every_surface_segment() {
    recover_constraints(original_constraints());
}
