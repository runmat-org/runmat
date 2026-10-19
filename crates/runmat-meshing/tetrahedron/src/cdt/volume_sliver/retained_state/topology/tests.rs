use super::super::test_fixtures::assigned;
use super::*;
use runmat_meshing_core::NeverCancelled;

#[test]
fn compact_seed_restores_exact_topology_and_order_with_unrelated_cells() {
    for unrelated in [0, 64] {
        let (base, candidate) = super::super::test_fixtures::pair(unrelated);
        let base = assigned(base);
        let candidate = assigned(candidate);
        let patch = TopologyPatch::between(&base, &candidate).unwrap();
        let unchanged = TopologyPatch::between(&base, &base).unwrap();
        assert_eq!(patch.removed.len(), 3);
        assert_eq!(patch.created.len(), 2);
        assert_eq!(
            patch.keys(&base).collect::<Vec<_>>(),
            connectivity::keys(&candidate).collect::<Vec<_>>()
        );
        assert_eq!(
            patch.compare(&unchanged, &base),
            connectivity::compare_connectivity(&candidate, &base)
        );
        let restored = patch
            .materialize(
                &base,
                DelaunayVolumeSliverOptions::default(),
                &NeverCancelled,
            )
            .unwrap();
        // Includes physical coordinates, corner orientation, neighbors, region
        // assignments and all incidence, not just an unordered cell set.
        assert_eq!(restored, candidate);
        let mut reassigned = base.clone();
        reassigned.tetrahedra[0]
            .region_id
            .as_mut()
            .unwrap()
            .source_topology_id = "foreign-part".into();
        assert_eq!(
            TopologyPatch::between(&base, &reassigned)
                .err()
                .unwrap()
                .kind,
            DelaunayVolumeSliverErrorKind::InvalidTopology
        );
        let mut foreign = candidate;
        foreign.nodes[0].coordinates_m[0] += 0.01;
        assert_eq!(
            TopologyPatch::between(&base, &foreign).err().unwrap().kind,
            DelaunayVolumeSliverErrorKind::InvalidTopology
        );
    }
}

#[test]
fn compact_seed_materialization_preserves_cancellation_and_resource_rejection() {
    struct Cancelled;
    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    let (base, candidate) = super::super::test_fixtures::pair(0);
    let patch = TopologyPatch::between(&base, &candidate).unwrap();
    assert_eq!(
        patch
            .materialize(&base, DelaunayVolumeSliverOptions::default(), &Cancelled)
            .unwrap_err()
            .kind,
        DelaunayVolumeSliverErrorKind::Cancelled
    );
    let mut options = DelaunayVolumeSliverOptions::default();
    options.insertion.topology.maximum_tetrahedra = 1;
    assert_eq!(
        patch
            .materialize(&base, options, &NeverCancelled)
            .unwrap_err()
            .kind,
        DelaunayVolumeSliverErrorKind::ResourceLimit
    );
}
