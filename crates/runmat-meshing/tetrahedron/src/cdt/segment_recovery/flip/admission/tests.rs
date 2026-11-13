use super::*;
use crate::cdt::{
    build_delaunay_volume_topology, DelaunaySegmentRecoveryOptions, DelaunayTopologyOptions,
    DelaunayVolumeNode,
};
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled};
use std::sync::atomic::{AtomicUsize, Ordering};

fn neighborhoods(count: usize) -> DelaunayVolumeTopology {
    let mut nodes = Vec::new();
    let mut cells = Vec::new();
    for copy in 0..count {
        let offset = nodes.len() as u32;
        for point in [
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0.2, 0.2, 1.],
            [0.2, 0.2, -1.],
        ] {
            let mut identity = [0; 32];
            identity[..8].copy_from_slice(&((nodes.len() + 1) as u64).to_be_bytes());
            nodes.push(DelaunayVolumeNode::explicit(
                StableDigest::from_bytes(identity),
                [point[0] + copy as f64 * 10., point[1], point[2]],
            ));
        }
        cells.extend([
            [offset, offset + 1, offset + 2, offset + 3],
            [offset, offset + 2, offset + 1, offset + 4],
        ]);
    }
    build_delaunay_volume_topology(
        nodes,
        cells,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

struct Counting(AtomicUsize);
impl MeshingCancellationSignal for Counting {
    fn is_cancelled(&self) -> bool {
        self.0.fetch_add(1, Ordering::Relaxed);
        false
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Across four and sixty-four disconnected physical neighborhoods, raw reconnection
//   creates a requested edge and exact physical repair removes it. Rejection preserves the original
//   topology and retained admission, with bounded checkpoint growth after shared index preparation.
//   This component series does not qualify arbitrary connected meshes, complete native recovery or
//   whole-task memory. Unverified states and protected faces retain full admission.
#[test]
fn rejected_physical_edge_trial_preserves_original_and_avoids_global_readmission() {
    let mut costs = Vec::new();
    for count in [4, 64] {
        let original = neighborhoods(count);
        let endpoints = [original.nodes[3].identity, original.nodes[4].identity];
        let options = DelaunaySegmentRecoveryOptions::default();
        crate::cdt::validate_delaunay_volume_topology(
            &original,
            options.insertion,
            &NeverCancelled,
        )
        .unwrap();
        let admitted = crate::cdt::insertion::AdmittedPhysicalTopology::verify(
            original.clone(),
            options.insertion,
            &NeverCancelled,
        )
        .unwrap();
        let mut topology = TopologyMutation::from_admitted(admitted);
        topology
            .prepare_cell_spatial_index(&mut || Ok::<_, ()>(()))
            .unwrap();
        // Exercise the exact failure mechanism: raw reconnection creates the
        // edge, then strict physical repair removes it. This is not an early
        // geometric rejection masquerading as an admission regression.
        let mut trial = topology.clone();
        let mut trial_work = RecoveryWork::new(options, &NeverCancelled);
        assert!(try_recover_edge_with_face_flip_mutation(
            &mut trial,
            endpoints[0],
            endpoints[1],
            &[],
            &[],
            0,
            &mut trial_work
        )
        .unwrap());
        assert!(crate::cdt::insertion::restore_admitted_mutation(
            &mut trial,
            options.insertion,
            &NeverCancelled
        )
        .unwrap());
        assert!(!super::super::super::candidate::preserves_edges(
            &trial,
            endpoints,
            &[],
            0,
            &mut trial_work
        )
        .unwrap());
        let signal = Counting(AtomicUsize::new(0));
        let mut options = options;
        options.constraints.cancellation_check_interval = 1;
        options.insertion.topology.cancellation_check_interval = 1;
        let mut work = RecoveryWork::new(options, &signal);
        assert!(!try_recover_edge_with_face_flip(
            &mut topology,
            endpoints[0],
            endpoints[1],
            &[],
            &[],
            0,
            &mut work
        )
        .unwrap());
        costs.push(signal.0.load(Ordering::Relaxed));
        assert_eq!(
            topology
                .snapshot(options.insertion.topology, &NeverCancelled)
                .unwrap(),
            original
        );
        assert!(topology.physical_dirty_faces().unwrap().is_empty());
    }
    assert!(
        costs[1] <= costs[0] * 4,
        "single rejected edge across distant inventory: {costs:?}"
    );
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Every tested checkpoint of an originally admitted edge-recovery transaction returns
//   cancellation while preserving original canonical topology and an empty physical dirty
//   inventory; retry reaches the complete original rejection. This verifies local transaction
//   cancellation, not worker cancellation, original heat-sink acceptance or whole-task scaling.
#[test]
fn cancelled_local_recovery_trials_preserve_original_at_every_checkpoint() {
    struct CancelAt {
        calls: AtomicUsize,
        stop: usize,
    }
    impl MeshingCancellationSignal for CancelAt {
        fn is_cancelled(&self) -> bool {
            self.calls.fetch_add(1, Ordering::Relaxed) >= self.stop
        }
    }
    let original = neighborhoods(1);
    let endpoints = [original.nodes[3].identity, original.nodes[4].identity];
    let mut options = DelaunaySegmentRecoveryOptions::default();
    options.constraints.cancellation_check_interval = 1;
    options.insertion.topology.cancellation_check_interval = 1;
    let make = || {
        let admitted = crate::cdt::insertion::AdmittedPhysicalTopology::verify(
            original.clone(),
            options.insertion,
            &NeverCancelled,
        )
        .unwrap();
        let topology = TopologyMutation::from_admitted(admitted);
        topology
            .prepare_cell_spatial_index(&mut || Ok::<_, ()>(()))
            .unwrap();
        topology
    };
    let count = Counting(AtomicUsize::new(0));
    assert!(!try_recover_edge_with_face_flip(
        &mut make(),
        endpoints[0],
        endpoints[1],
        &[],
        &[],
        0,
        &mut RecoveryWork::new(options, &count)
    )
    .unwrap());
    for stop in 0..count.0.load(Ordering::Relaxed) {
        let signal = CancelAt {
            calls: AtomicUsize::new(0),
            stop,
        };
        let mut topology = make();
        let failure = try_recover_edge_with_face_flip(
            &mut topology,
            endpoints[0],
            endpoints[1],
            &[],
            &[],
            0,
            &mut RecoveryWork::new(options, &signal),
        )
        .unwrap_err();
        assert_eq!(
            failure.kind,
            DelaunaySegmentRecoveryErrorKind::Cancelled,
            "checkpoint {stop}"
        );
        assert_eq!(
            topology
                .snapshot(options.insertion.topology, &NeverCancelled)
                .unwrap(),
            original,
            "checkpoint {stop}"
        );
        assert!(topology.physical_dirty_faces().unwrap().is_empty());
        assert!(!try_recover_edge_with_face_flip(
            &mut topology,
            endpoints[0],
            endpoints[1],
            &[],
            &[],
            0,
            &mut RecoveryWork::new(options, &NeverCancelled)
        )
        .unwrap());
    }
}
