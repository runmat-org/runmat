use super::*;
use crate::cdt::volume_sliver::working::CaptureSeed;
mod cavity_relocation;
mod relocation;
mod scaling;
mod search;
use crate::cdt::volume_sliver::tests::{interior_sliver_fixture, metric, quality, quality_options};
use runmat_meshing_core::NeverCancelled;

fn split(trial: &mut WorkingTrial<'_, '_>, identity: StableDigest) {
    let source = trial
        .quality()
        .worst(QualityObjective::Shape)
        .unwrap()
        .node_identities;
    let vertices = source.map(|identity| trial.topology().node_index(identity).unwrap());
    let index = trial.topology().tetrahedron_index(vertices).unwrap();
    let region = trial.topology().cell(index).region_id.clone();
    let point = std::array::from_fn(|axis| {
        vertices
            .iter()
            .map(|vertex| {
                trial
                    .topology()
                    .node(*vertex)
                    .coordinates_m
                    .canonical_value()[axis]
            })
            .sum::<f64>()
            / 4.0
    });
    let node = trial
        .append_node(
            DelaunayVolumeNode::explicit(identity, point),
            DelaunayVolumeSliverOptions::default(),
        )
        .unwrap();
    let created = (0..4)
        .map(|opposite| {
            let mut corners = vertices;
            corners[opposite] = node;
            (corners, region.clone())
        })
        .collect();
    trial
        .replace(
            &BTreeSet::from([index]),
            created,
            DelaunayVolumeSliverOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
}

#[test]
fn nested_working_trials_replay_only_retained_edits_and_restore_both_owners() {
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
    let seed;
    {
        let mut outer = state.trial();
        split(&mut outer, StableDigest::from_bytes([6; 32]));
        {
            let mut rejected = outer.trial();
            split(&mut rejected, StableDigest::from_bytes([7; 32]));
        }
        {
            let mut retained = outer.trial();
            split(&mut retained, StableDigest::from_bytes([8; 32]));
            retained.commit();
        }
        seed = outer.capture(Vec::new());
        assert_eq!(seed.edits.len(), 4);
        assert!(seed.edits.iter().all(|edit| !matches!(edit,
            TopologyEdit::AppendNode(node) if node.identity == StableDigest::from_bytes([7; 32]))));
        assert_eq!(outer.quality().cell_count(), reference.tetrahedra.len() + 6);
        {
            let mut continuation = outer.trial();
            seed.replay(
                &mut continuation,
                DelaunayVolumeSliverOptions::default(),
                &NeverCancelled,
            )
            .unwrap();
            assert_eq!(
                continuation.topology().capture_edits().len(),
                seed.edits.len()
            );
            assert_eq!(
                continuation.quality().cell_count(),
                reference.tetrahedra.len() + 6
            );
        }
    }
    assert_eq!(state.quality().cell_count(), reference.tetrahedra.len());
    {
        let mut replay = state.trial();
        seed.replay(
            &mut replay,
            DelaunayVolumeSliverOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let replayed = replay.capture(Vec::new());
        for objective in [QualityObjective::Shape, QualityObjective::Refinement] {
            assert_eq!(
                seed.compare_spectrum(&replayed, objective),
                std::cmp::Ordering::Equal
            );
        }
        assert_eq!(seed.connectivity_key(), replayed.connectivity_key());
        assert_eq!(
            seed.compare_connectivity(&replayed, replay.quality()),
            std::cmp::Ordering::Equal
        );
    }
    let (actual, actual_quality) = state
        .finish(DelaunayVolumeSliverOptions::default(), &NeverCancelled)
        .unwrap();
    assert_eq!(actual, base);
    assert_eq!(actual_quality, reference);
}

#[test]
fn retained_seed_rejects_foreign_and_rebased_owners_before_any_edit() {
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
    let seed = {
        let mut trial = state.trial();
        split(&mut trial, StableDigest::from_bytes([6; 32]));
        trial.capture(Vec::new())
    };
    let mut foreign = WorkingState::new(
        base.clone(),
        reference.clone(),
        quality_options(),
        &prepared,
        &[],
        &NeverCancelled,
    )
    .unwrap();
    for owner in [&mut foreign, &mut state] {
        owner.rebase(&NeverCancelled).unwrap();
        {
            let mut trial = owner.trial();
            assert_eq!(
                seed.replay(
                    &mut trial,
                    DelaunayVolumeSliverOptions::default(),
                    &NeverCancelled
                )
                .unwrap_err()
                .kind,
                VolumeStateErrorKind::InvalidInput
            );
            assert!(trial.topology().capture_edits().is_empty());
        }
        assert_eq!(owner.quality().cell_count(), reference.tetrahedra.len());
    }
}

mod scale_codec;
mod scale_inventory;
