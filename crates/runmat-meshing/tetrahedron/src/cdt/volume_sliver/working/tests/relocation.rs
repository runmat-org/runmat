use super::*;
use crate::cdt::volume_sliver::working::CaptureSeed;

fn query_identity(topology: &TopologyMutation, point: [f64; 3]) -> Vec<StableDigest> {
    let cells = topology
        .spatial_node_candidates(
            Some(runmat_meshing_core::Aabb3 {
                min_m: point,
                max_m: point,
            }),
            &mut || Ok::<_, ()>(()),
        )
        .unwrap();
    cells
        .into_iter()
        .map(|index| topology.node(index).identity)
        .collect()
}

#[test]
fn local_relocation_replays_cached_geometry_and_matches_full_quality_reference() {
    let (base, provenance) = interior_sliver_fixture();
    let reference = quality(&base, &provenance);
    let request = metric();
    let prepared = PreparedLocalQuality::prepare(&request, &provenance, quality_options()).unwrap();
    let mut state = WorkingState::new(
        base.clone(),
        reference.clone(),
        quality_options(),
        &prepared,
        &[],
        &NeverCancelled,
    )
    .unwrap();
    let source = StableDigest::from_bytes([5; 32]);
    let replacement = DelaunayVolumeNode::explicit(StableDigest::from_bytes([6; 32]), [0.25; 3]);
    assert_eq!(
        query_identity(state.topology(), [0.3, 0.3, 1.0e-6]),
        vec![source]
    );
    let seed = {
        let mut trial = state.trial();
        let index = trial.topology().node_index(source).unwrap();
        trial
            .replace_node(
                index,
                replacement,
                DelaunayVolumeSliverOptions::default(),
                &NeverCancelled,
            )
            .unwrap();
        assert_eq!(
            query_identity(
                trial.topology(),
                replacement.coordinates_m.canonical_value()
            ),
            vec![replacement.identity]
        );
        assert!(query_identity(trial.topology(), [0.3, 0.3, 1.0e-6]).is_empty());
        trial.capture(Vec::new())
    };
    assert_eq!(
        query_identity(state.topology(), [0.3, 0.3, 1.0e-6]),
        vec![source]
    );
    {
        let mut trial = state.trial();
        seed.replay(
            &mut trial,
            DelaunayVolumeSliverOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let replayed = trial.capture(Vec::new());
        assert_eq!(seed.connectivity_key(), replayed.connectivity_key());
        trial.commit();
    }
    let (actual, actual_quality) = state
        .finish(DelaunayVolumeSliverOptions::default(), &NeverCancelled)
        .unwrap();
    let full = quality(&actual, &provenance);
    assert_eq!(actual_quality, full);
    assert!(!actual.nodes.iter().any(|node| node.identity == source));
    assert!(actual.nodes.contains(&replacement));
}

#[test]
fn protected_and_inverted_relocation_reject_before_mutating_either_owner() {
    let (base, provenance) = interior_sliver_fixture();
    let reference = quality(&base, &provenance);
    let request = metric();
    let prepared = PreparedLocalQuality::prepare(&request, &provenance, quality_options()).unwrap();
    let mut state = WorkingState::new(
        base.clone(),
        reference.clone(),
        quality_options(),
        &prepared,
        &[],
        &NeverCancelled,
    )
    .unwrap();
    for (source, point) in [(1, [0.01; 3]), (5, [2.0; 3])] {
        let mut trial = state.trial();
        let index = trial
            .topology()
            .node_index(StableDigest::from_bytes([source; 32]))
            .unwrap();
        assert_eq!(
            trial
                .replace_node(
                    index,
                    DelaunayVolumeNode::explicit(StableDigest::from_bytes([7; 32]), point),
                    DelaunayVolumeSliverOptions::default(),
                    &NeverCancelled
                )
                .unwrap_err()
                .kind,
            VolumeStateErrorKind::InvalidTopology
        );
        assert!(trial.topology().capture_edits().is_empty());
    }
    let (actual, actual_quality) = state
        .finish(DelaunayVolumeSliverOptions::default(), &NeverCancelled)
        .unwrap();
    assert_eq!(actual, base);
    assert_eq!(actual_quality, reference);
}
