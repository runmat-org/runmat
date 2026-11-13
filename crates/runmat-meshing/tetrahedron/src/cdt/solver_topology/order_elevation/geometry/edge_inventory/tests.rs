//! Metadata preparation only: repeated observations are not a geometric qualification.
use super::*;
use runmat_geometry_core::{GeometryEvaluationError, GeometryEvaluationErrorKind};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
struct Budget {
    live: AtomicU64,
    peak: AtomicU64,
    maximum: u64,
}
impl GeometryEvaluationControl for Budget {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, bytes: u64) -> Result<(), GeometryEvaluationError> {
        let before = self
            .live
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
                live.checked_add(bytes).filter(|next| *next <= self.maximum)
            })
            .map_err(|_| {
                GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                    "flat metadata denied",
                )
            })?;
        self.peak.fetch_max(before + bytes, Ordering::Relaxed);
        Ok(())
    }
    fn release_allocation_bytes(&self, bytes: u64) {
        self.live.fetch_sub(bytes, Ordering::Relaxed);
    }
}
#[test]
fn borrowed_flat_owners_match_reference_and_release_every_admitted_buffer() {
    let base = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
        runmat_meshing_core::ElementOrder::Tet4,
    );
    let mut owners = base
        .topology
        .volume_elements
        .iter()
        .flat_map(|element| {
            element
                .provenance
                .iter()
                .chain(std::iter::once(&element.region_id))
        })
        .collect::<Vec<_>>();
    owners.sort();
    owners.dedup();
    for observations in [1, 64, 4096] {
        let mut topology = (*base.topology).clone();
        topology
            .volume_elements
            .resize(observations, topology.volume_elements[0].clone());
        let mut expected = BTreeMap::<[u64; 2], BTreeSet<&PersistentEntityId>>::new();
        for cell in &topology.volume_elements {
            for edge in TETRAHEDRON_MIDSIDE_EDGE_CORNERS {
                let edge = sorted_edge([cell.node_ids[edge[0]], cell.node_ids[edge[1]]]);
                expected.entry(edge).or_default().extend(
                    cell.provenance
                        .iter()
                        .chain(std::iter::once(&cell.region_id)),
                );
            }
        }
        let budget = Budget {
            live: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            maximum: 16 << 20,
        };
        let inventory = EdgeOwners::prepare(&topology, &owners, &budget).unwrap();
        assert_eq!(inventory.groups.len(), expected.len());
        for (edge, range) in &inventory.groups {
            let actual = inventory.owners(range.clone()).collect::<Vec<_>>();
            assert_eq!(actual, expected[edge].iter().copied().collect::<Vec<_>>());
            assert!(actual
                .iter()
                .all(|entity| owners.iter().any(|source| std::ptr::eq(*entity, *source))));
        }
        assert!(budget.live.load(Ordering::Relaxed) > 0);
        drop(inventory);
        assert_eq!(budget.live.load(Ordering::Relaxed), 0);
        assert!(budget.peak.load(Ordering::Relaxed) < 16 << 20);
        let denied = Budget {
            live: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            maximum: 0,
        };
        assert!(EdgeOwners::prepare(&topology, &owners, &denied).is_err());
        assert_eq!(denied.live.load(Ordering::Relaxed), 0);
    }
}
