use super::*;
use crate::cdt::solver_topology::order_elevation::counters::ElevationCounters;
use crate::cdt::test_fixtures::allocation::AllocationBudget;
use runmat_meshing_core::{fixtures::quadratic_slab_mesh, NeverCancelled};

fn reference_profile(topology: &SolverMeshTopology, elements: &[usize]) -> Vec<u64> {
    jacobian::optimization_margin_profile_elements(
        topology,
        elements.iter().copied(),
        &NeverCancelled,
        crate::cdt::solver_topology::storage::Storage(None),
    )
    .unwrap()
    .to_vec()
}

fn incident(topology: &SolverMeshTopology, id: u64) -> Vec<usize> {
    topology
        .volume_elements
        .iter()
        .enumerate()
        .filter(|(_, element)| element.node_ids.contains(&id))
        .map(|(index, _)| index)
        .collect()
}

fn set(topology: &mut SolverMeshTopology, slot: usize, point: [f64; 3]) {
    topology.nodes[slot].coordinates_m = runmat_types::physical::Quantity::new(point);
}

// witness: FEA-MESH-DISCRETIZATION-001/volume_quality_convergence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Cached margin profiles and the parallel candidate profiles equal profiles taken
//   from scratch with the node placed at each candidate, so repair decisions are unchanged.
#[test]
fn cached_and_candidate_profiles_match_profiles_from_scratch() {
    let budget = AllocationBudget::new(1 << 30);
    let mut topology = quadratic_slab_mesh(2).topology.into_topology();
    let counters = ElevationCounters::default();
    let mut state = RepairState::new(&budget);
    // An interior-ish midside node: the one with the most incident elements.
    let midside = topology.volume_elements[0].node_ids[4];
    let node_id = (5..10)
        .flat_map(|slot| {
            topology
                .volume_elements
                .iter()
                .map(move |e| e.node_ids[slot])
        })
        .max_by_key(|id| incident(&topology, *id).len())
        .unwrap_or(midside);
    let elements = incident(&topology, node_id);
    assert!(elements.len() > 1);
    let slot = topology
        .nodes
        .binary_search_by_key(&node_id, |node| node.node_id)
        .unwrap();
    let initial = topology.nodes[slot].coordinates_m.canonical_value();
    let movable = MovableNode {
        edge: [0, 0],
        node_id,
        stable_identity: topology.nodes[slot].stable_identity,
        incident_elements: &elements,
        surface_uses: &[],
        exact_edge_id: None,
    };
    let offsets = [
        [0.0, 0.0, 0.0],
        [0.01, -0.02, 0.005],
        [0.2, 0.1, -0.15],
        [-0.3, 0.25, 0.1],
    ];
    let candidates = offsets
        .iter()
        .enumerate()
        .map(|(index, offset)| Candidate {
            fraction: index as f64,
            objective: 0.0,
            coordinates_m: runmat_types::physical::Quantity::new(std::array::from_fn(|axis| {
                initial[axis] + offset[axis]
            })),
            exact_parameters: Vec::new(),
        })
        .collect::<Vec<_>>();
    let profiles =
        node::candidate_profiles(&topology, &movable, &candidates, &NeverCancelled, &counters)
            .unwrap();
    for (candidate, profile) in candidates.iter().zip(&profiles) {
        set(
            &mut topology,
            slot,
            candidate.coordinates_m.canonical_value(),
        );
        assert_eq!(profile, &reference_profile(&topology, &elements));
    }
    set(&mut topology, slot, initial);
    // The cache reuses unchanged elements and recomputes moved ones.
    assert_eq!(
        state
            .profile(&topology, &elements, &NeverCancelled, &counters)
            .unwrap(),
        reference_profile(&topology, &elements)
    );
    for candidate in &candidates {
        set(
            &mut topology,
            slot,
            candidate.coordinates_m.canonical_value(),
        );
        assert_eq!(
            state
                .profile(&topology, &elements, &NeverCancelled, &counters)
                .unwrap(),
            reference_profile(&topology, &elements)
        );
        assert_eq!(
            state
                .profile(&topology, &elements, &NeverCancelled, &counters)
                .unwrap(),
            reference_profile(&topology, &elements)
        );
    }
    assert!(
        counters
            .profile_margins_reused
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0
    );
}

#[test]
fn a_declined_repair_is_remembered_until_its_star_changes() {
    let budget = AllocationBudget::new(1 << 30);
    let mut topology = quadratic_slab_mesh(2).topology.into_topology();
    let mut state = RepairState::new(&budget);
    let node_id = topology.volume_elements[0].node_ids[4];
    let elements = incident(&topology, node_id);
    let slot = |topology: &SolverMeshTopology, id: u64| {
        topology
            .nodes
            .binary_search_by_key(&id, |node| node.node_id)
            .unwrap()
    };
    let movable = MovableNode {
        edge: [0, 0],
        node_id,
        stable_identity: topology.nodes[slot(&topology, node_id)].stable_identity,
        incident_elements: &elements,
        surface_uses: &[],
        exact_edge_id: None,
    };
    let signature = |topology: &SolverMeshTopology| {
        let node = &topology.nodes[slot(topology, node_id)];
        StarSignature::of(topology, &movable, [node, node, node]).unwrap()
    };
    state.decline(node_id, signature(&topology)).unwrap();
    assert!(state.declined_unchanged(node_id, &signature(&topology)));
    // Moving a node of a neighboring element in the star invalidates the memory.
    let other = topology.volume_elements[elements[elements.len() - 1]].node_ids[0];
    let other_slot = slot(&topology, other);
    let mut point = topology.nodes[other_slot].coordinates_m.canonical_value();
    point[2] += 1.0e-9;
    set(&mut topology, other_slot, point);
    assert!(!state.declined_unchanged(node_id, &signature(&topology)));
    point[2] -= 1.0e-9;
    set(&mut topology, other_slot, point);
    state.moved(node_id);
    assert!(!state.declined_unchanged(node_id, &signature(&topology)));
}
