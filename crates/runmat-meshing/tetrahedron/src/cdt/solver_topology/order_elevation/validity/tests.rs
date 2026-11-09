use super::*;
use crate::cdt::test_fixtures::allocation::AllocationBudget;
use runmat_meshing_core::{fixtures::quadratic_slab_mesh, NeverCancelled};

const LIMITS: CertificationLimits = CertificationLimits {
    maximum_search_work: jacobian::ELEMENT_SEARCH_WORK,
    maximum_recursion_depth: 32,
    cancellation_check_interval: 1,
};

/// Deterministic pseudo-random sequence for node moves.
struct Sequence(u64);
impl Sequence {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f64 {
        (self.next() % 2001) as f64 / 1000.0 - 1.0
    }
}

fn incidence(topology: &SolverMeshTopology) -> Vec<Vec<usize>> {
    let mut incident = vec![Vec::new(); topology.nodes.len()];
    for (index, element) in topology.volume_elements.iter().enumerate() {
        for id in &element.node_ids {
            incident[*id as usize - 1].push(index);
        }
    }
    incident
}

fn full(topology: &SolverMeshTopology, budget: &AllocationBudget) -> Vec<usize> {
    jacobian::invalid_element_indices(
        topology,
        LIMITS.maximum_search_work,
        LIMITS.maximum_recursion_depth,
        LIMITS.cancellation_check_interval,
        &NeverCancelled,
        jacobian::ValidationWork {
            counter: &mut 0,
            storage: Storage(Some(budget)),
        },
    )
    .unwrap()
    .to_vec()
}

// witness: FEA-MESH-DISCRETIZATION-001/volume_quality_convergence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: The incremental Tet10 invalid-element inventory equals a full certification of
//   every element after arbitrary node moves, including moves that invalidate elements and moves
//   back to an earlier position.
#[test]
fn incremental_inventory_matches_full_certification_after_node_moves() {
    let budget = AllocationBudget::new(1 << 30);
    let mut topology = quadratic_slab_mesh(3).topology.into_topology();
    let incident = incidence(&topology);
    let counters = ElevationCounters::default();
    let mut validity = ElementValidity::certify(
        &topology,
        LIMITS,
        &budget,
        &NeverCancelled,
        &mut 0,
        &counters,
    )
    .unwrap();
    assert_eq!(
        validity.indices(Storage(None)).unwrap().to_vec(),
        full(&topology, &budget)
    );
    let original = topology.clone();
    let mut sequence = Sequence(7);
    let mut saw_invalid = false;
    for round in 0..40 {
        let moves = 1 + sequence.next() as usize % 6;
        for _ in 0..moves {
            let slot = sequence.next() as usize % topology.nodes.len();
            let mut point = if round % 10 == 9 {
                // Return the node to its original position.
                original.nodes[slot].coordinates_m.canonical_value()
            } else {
                topology.nodes[slot].coordinates_m.canonical_value()
            };
            let scale = if sequence.next() % 3 == 0 { 0.4 } else { 0.04 };
            if round % 10 != 9 {
                for value in &mut point {
                    *value += scale * sequence.unit();
                }
            }
            topology.nodes[slot].coordinates_m = runmat_types::physical::Quantity::new(point);
        }
        validity
            .refresh(
                &topology,
                |id| Ok(incident[id as usize - 1].as_slice()),
                &budget,
                &NeverCancelled,
                &mut 0,
                &counters,
            )
            .unwrap();
        let expected = full(&topology, &budget);
        saw_invalid |= !expected.is_empty();
        assert_eq!(
            validity.indices(Storage(None)).unwrap().to_vec(),
            expected,
            "round {round}"
        );
    }
    assert!(saw_invalid, "the moves must invalidate some elements");
    // Elements with folded boundary faces are merged into the inventory, so it has room for
    // every element.
    let mut indices = validity.indices(Storage(None)).unwrap();
    let present = indices.to_vec();
    for element in 0..topology.volume_elements.len() {
        if !present.contains(&element) {
            indices.push(element).unwrap();
        }
    }
    let reused = counters
        .inventory_reused
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(reused > 0, "refreshes must reuse unchanged elements");
}
