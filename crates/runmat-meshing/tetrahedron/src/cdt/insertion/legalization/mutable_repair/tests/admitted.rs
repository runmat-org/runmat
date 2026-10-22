use super::*;
use crate::cdt::insertion::AdmittedPhysicalTopology;

fn original(count: usize) -> DelaunayVolumeTopology {
    let options = super::super::super::super::DelaunayInsertionOptions::default();
    restore(
        neighborhoods(count),
        &BTreeSet::new(),
        &mut InsertionWork::new(options, &NeverCancelled),
    )
    .unwrap()
}

fn changed(count: usize) -> (TopologyMutation, DelaunayVolumeTopology) {
    let original = original(count);
    let options = super::super::super::super::DelaunayInsertionOptions::default();
    let admitted =
        AdmittedPhysicalTopology::verify(original.clone(), options, &NeverCancelled).unwrap();
    let mut topology = TopologyMutation::from_admitted(admitted);
    let removed = (0..topology.tetrahedron_slot_count())
        .filter(|slot| {
            topology
                .tetrahedron(*slot as u32)
                .unwrap()
                .vertex_indices
                .iter()
                .all(|vertex| *vertex < 5)
        })
        .map(|slot| slot as u32)
        .collect::<BTreeSet<_>>();
    assert_eq!(removed.len(), 3);
    topology
        .replace_cavity(&removed, vec![([0, 2, 3, 4], None), ([1, 2, 4, 3], None)])
        .unwrap();
    assert!(!topology.physical_dirty_faces().unwrap().is_empty());
    (topology, original)
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A construction-verified physical prefix repairs one changed cavity across four and
//   sixty-four disconnected neighborhoods, retains original points and passes independent
//   exhaustive empty-sphere validation with bounded checkpoint growth. This component size series
//   does not establish connected-mesh asymptotics, native heat recovery or whole-task memory.
#[test]
fn admitted_dirty_repair_is_local_and_matches_independent_whole_mesh_validation() {
    let mut counts = Vec::new();
    for size in [4, 64] {
        let (mut topology, original) = changed(size);
        let mut options = super::super::super::super::DelaunayInsertionOptions::default();
        options.topology.cancellation_check_interval = 1;
        let probe = Probe::default();
        let mut work = InsertionWork::new(options, &probe);
        assert!(restore_admitted(&mut topology, &mut work).unwrap());
        assert_eq!(work.legalization_flips, 1);
        assert!(topology.physical_dirty_faces().unwrap().is_empty());
        counts.push(probe.0.load(Ordering::Relaxed));
        let repaired = topology.freeze(options.topology, &NeverCancelled).unwrap();
        assert_eq!(repaired, original);
        crate::cdt::validate_delaunay_volume_topology(&repaired, options, &NeverCancelled).unwrap();
    }
    assert!(
        counts[1] <= counts[0] * 2,
        "single dirty cavity across distant inventory: {counts:?}"
    );
}

#[test]
fn rejected_admitted_repair_retains_dirty_work_and_does_not_authorize_unchecked_input() {
    let (mut topology, original) = changed(4);
    let limited = super::super::super::super::DelaunayInsertionOptions {
        maximum_predicate_evaluations: 1,
        ..Default::default()
    };
    assert_eq!(
        restore_admitted(
            &mut topology,
            &mut InsertionWork::new(limited, &NeverCancelled)
        )
        .unwrap_err()
        .kind,
        DelaunayInsertionErrorKind::ResourceLimit
    );
    assert!(!topology.physical_dirty_faces().unwrap().is_empty());
    let options = super::super::super::super::DelaunayInsertionOptions::default();
    assert!(restore_admitted(
        &mut topology,
        &mut InsertionWork::new(options, &NeverCancelled)
    )
    .unwrap());
    assert_eq!(
        topology.freeze(options.topology, &NeverCancelled).unwrap(),
        original
    );
    assert!(AdmittedPhysicalTopology::verify(neighborhoods(1), options, &NeverCancelled).is_err());
    let mut unchecked = TopologyMutation::from_canonical(original.clone());
    assert!(!restore_admitted(
        &mut unchecked,
        &mut InsertionWork::new(options, &NeverCancelled)
    )
    .unwrap());
    assert_eq!(
        unchecked.freeze(options.topology, &NeverCancelled).unwrap(),
        original
    );
}

// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Cancellation before local repair preserves the original trial topology and complete
//   pending physical-adjacency inventory. This component entry cancellation does not establish
//   every mid-repair cancellation point or native task rollback.
#[test]
fn cancelled_admitted_repair_preserves_original_trial_and_pending_authority() {
    struct Cancelled;
    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    let (mut topology, _) = changed(4);
    let options = super::super::super::super::DelaunayInsertionOptions::default();
    let before = topology
        .clone()
        .freeze(options.topology, &NeverCancelled)
        .unwrap();
    let dirty = topology.physical_dirty_faces().unwrap().clone();
    let mut cancellable = options;
    cancellable.topology.cancellation_check_interval = 1;
    assert_eq!(
        restore_admitted(
            &mut topology,
            &mut InsertionWork::new(cancellable, &Cancelled)
        )
        .unwrap_err()
        .kind,
        DelaunayInsertionErrorKind::Cancelled
    );
    assert_eq!(topology.physical_dirty_faces().unwrap(), &dirty);
    assert_eq!(
        topology.freeze(options.topology, &NeverCancelled).unwrap(),
        before
    );
}
