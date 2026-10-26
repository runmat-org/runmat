use super::*;
use crate::cdt::insertion::AdmittedPhysicalTopology;
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(serde::Deserialize)]
struct Captured {
    nodes: Vec<crate::cdt::DelaunayVolumeNode>,
}

fn changed(count: usize) -> TopologyMutation {
    let mut captured: Captured = serde_json::from_str(include_str!(
        "../../local/tests/fixtures/fork-double-neighborhood.json"
    ))
    .unwrap();
    captured.nodes.sort_by_key(|node| node.identity);
    let mut nodes = [28, 226, 776, 852, 481, 817]
        .map(|index| captured.nodes[index])
        .to_vec();
    for (index, node) in nodes.iter_mut().enumerate() {
        let mut identity = [1; 32];
        identity[..8].copy_from_slice(&((index + 1) as u64).to_be_bytes());
        node.identity = StableDigest::from_bytes(identity);
    }
    let bad = [[0, 1, 2, 3], [0, 2, 5, 3], [1, 4, 2, 3], [2, 4, 5, 3]];
    let options = crate::cdt::insertion::DelaunayInsertionOptions::default();
    let original = crate::cdt::build_delaunay_volume_topology(
        nodes,
        bad.to_vec(),
        options.topology,
        &NeverCancelled,
    )
    .unwrap();
    let good = super::super::super::local::restore(
        original,
        &BTreeSet::new(),
        true,
        &mut InsertionWork::new(options, &NeverCancelled),
    )
    .unwrap();
    let mut nodes = Vec::new();
    let mut cells = Vec::new();
    for cluster in 0..count {
        let first = nodes.len() as u32;
        for mut node in good.nodes.iter().copied() {
            let mut identity = [1; 32];
            identity[..8].copy_from_slice(&((nodes.len() + 1) as u64).to_be_bytes());
            node.identity = StableDigest::from_bytes(identity);
            {
                let quantity_change = cluster as f64 * 8.0;
                let quantity_index = 0;
                let quantity_slot = &mut node.coordinates_m;
                let mut quantity_coordinates = quantity_slot.canonical_value();
                quantity_coordinates[quantity_index] += quantity_change;
                *quantity_slot = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(quantity_coordinates);
            };
            if let runmat_meshing_core::quality::predicate::Point3Construction::Segment {
                ref mut endpoints,
                ..
            } = node.construction
            {
                for endpoint in endpoints {
                    endpoint[0] += cluster as f64 * 8.0;
                }
            }
            nodes.push(node);
        }
        cells.extend(
            good.tetrahedra
                .iter()
                .map(|cell| cell.vertex_indices.map(|vertex| vertex + first)),
        );
    }
    let original =
        crate::cdt::build_delaunay_volume_topology(nodes, cells, options.topology, &NeverCancelled)
            .unwrap();
    let admitted = AdmittedPhysicalTopology::verify(original, options, &NeverCancelled).unwrap();
    let mut topology = TopologyMutation::from_admitted(admitted);
    let removed = (0..topology.tetrahedron_slot_count())
        .filter(|index| {
            topology
                .tetrahedron(*index as u32)
                .unwrap()
                .vertex_indices
                .iter()
                .all(|vertex| *vertex < 6)
        })
        .map(|index| index as u32)
        .collect();
    topology
        .replace_cavity(&removed, bad.into_iter().map(|cell| (cell, None)).collect())
        .unwrap();
    // Index startup is separate from retained-neighborhood repair work.
    topology
        .connectivity_search_fingerprint(&mut || Ok::<(), ()>(()))
        .unwrap();
    topology
}

#[derive(Default)]
struct Checkpoints {
    calls: AtomicUsize,
    stop: Option<usize>,
}
impl MeshingCancellationSignal for Checkpoints {
    fn is_cancelled(&self) -> bool {
        let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        self.stop.is_some_and(|stop| call >= stop)
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Captured four-to-four geometry repairs one dirty physical neighborhood with equal
//   checkpoint work across 4/64 disconnected neighborhoods, after explicit index startup. Preserves
//   point constructions and passes independent whole-result empty-sphere validation. Does not
//   establish arbitrary neighborhood/valence/depth bounds, cold-index locality, full native timing
//   or whole-task admission.
#[test]
fn advanced_physical_repair_is_local_and_independently_admitted() {
    let mut counts = Vec::new();
    for size in [4, 64] {
        let mut topology = changed(size);
        let points = topology.nodes.clone();
        let signal = Checkpoints::default();
        let mut options = crate::cdt::insertion::DelaunayInsertionOptions::default();
        options.topology.cancellation_check_interval = 1;
        let mut work = InsertionWork::new(options, &signal);
        assert!(violations(&topology, &mut work).unwrap().0 > 0);
        assert!(super::super::restore_admitted(&mut topology, &mut work).unwrap());
        assert!(work.legalization_flips > 0);
        counts.push(signal.calls.load(Ordering::Relaxed));
        assert!(topology.physical_dirty_faces().unwrap().is_empty());
        let result = topology
            .validated_snapshot(options.topology, &NeverCancelled)
            .unwrap();
        assert_eq!(result.nodes, points);
        crate::cdt::insertion::validate_delaunay_volume_topology(&result, options, &NeverCancelled)
            .unwrap();
    }
    assert_eq!(
        counts[0], counts[1],
        "retained advanced repair checkpoints: {counts:?}"
    );
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Cancellation at every exercised checkpoint of a captured advanced physical-repair
//   trial restores the original canonical topology and dirty inventory; a complete retry succeeds.
//   Does not establish remote cancellation, fenced publication, arbitrary repair paths or
//   whole-task resource/performance acceptance.
#[test]
fn advanced_physical_repair_cancellation_preserves_the_original_trial() {
    let original = changed(1);
    let mut options = crate::cdt::insertion::DelaunayInsertionOptions::default();
    options.topology.cancellation_check_interval = 1;
    let before = original
        .validated_snapshot(options.topology, &NeverCancelled)
        .unwrap();
    let signal = Checkpoints::default();
    let mut completed = original.clone();
    assert!(super::super::restore_admitted(
        &mut completed,
        &mut InsertionWork::new(options, &signal)
    )
    .unwrap());
    for stop in 1..=signal.calls.load(Ordering::Relaxed) {
        let cancellation = Checkpoints {
            calls: AtomicUsize::new(0),
            stop: Some(stop),
        };
        let mut topology = original.clone();
        {
            let mut trial = topology.cavity_trial();
            let failure = super::super::restore_admitted(
                &mut trial,
                &mut InsertionWork::new(options, &cancellation),
            )
            .unwrap_err();
            assert_eq!(
                failure.kind,
                DelaunayInsertionErrorKind::Cancelled,
                "checkpoint {stop}"
            );
        }
        assert_eq!(
            topology
                .validated_snapshot(options.topology, &NeverCancelled)
                .unwrap(),
            before
        );
        assert_eq!(
            topology.physical_dirty_faces(),
            original.physical_dirty_faces()
        );
        assert!(super::super::restore_admitted(
            &mut topology,
            &mut InsertionWork::new(options, &NeverCancelled)
        )
        .unwrap());
    }
}
