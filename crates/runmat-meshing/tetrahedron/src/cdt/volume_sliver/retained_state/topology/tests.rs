use super::super::test_fixtures::assigned;
use super::*;
use runmat_meshing_core::NeverCancelled;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An exact three-to-two replacement retains only three removed and two created cells
//   against one baseline, with zero or 64 unrelated cells and distinct region assignments.
//   Materialization restores complete physical topology, orientation, neighbors, regions and
//   incidence; streamed ordering matches the original full candidate. Changed nodes or reassigned
//   unchanged cells reject. No sublinear work or whole-task resource claim.
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
        {
            let quantity_change = 0.01;
            let quantity_index = 0;
            let quantity_slot = &mut foreign.nodes[0].coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] += quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
        assert_eq!(
            TopologyPatch::between(&base, &foreign).err().unwrap().kind,
            DelaunayVolumeSliverErrorKind::InvalidTopology
        );
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Compact reconnection seed materialization rejects cancellation and a one-cell
//   topology ceiling for an exact two-cell candidate. This is a construction component boundary,
//   not whole-task admission or measured native memory evidence.
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
