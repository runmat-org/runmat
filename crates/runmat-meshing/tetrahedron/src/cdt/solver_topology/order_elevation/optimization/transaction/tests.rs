use super::*;
use crate::cdt::test_fixtures::allocation::AllocationBudget;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[test]
fn rejected_edits_restore_original_buffers_without_cumulative_candidate_storage() {
    let budget = AllocationBudget::new(512 << 20);
    {
        let scope = GeometryAllocationScope::new(&budget);
        let mut topology = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10)
            .topology
            .into_topology();
        for node in &mut topology.nodes {
            *node = Storage(Some(&scope)).node(node).unwrap();
        }
        let original = topology.clone();
        let pointers = topology
            .nodes
            .iter()
            .map(|node| (node.provenance.as_ptr(), node.exact_parameters.as_ptr()))
            .collect::<Vec<_>>();
        let live = budget.live_bytes();
        for attempt in 0..64 {
            let result = edit_nodes(&mut topology, &[0, 1, 2], &scope, |topology, control| {
                for slot in 0..3 {
                    let replacement = Storage(Some(control)).node(&topology.nodes[slot])?;
                    topology.nodes[slot] = replacement;
                    {
                        let quantity_change = 0.5;
                        let quantity_index = 0;
                        let quantity_slot = &mut topology.nodes[slot].coordinates_m;
                        let mut quantity_coordinates = quantity_slot.canonical_value();
                        quantity_coordinates[quantity_index] += quantity_change;
                        *quantity_slot = runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::Position,
                            _,
                        >::new(quantity_coordinates);
                    };
                }
                if attempt % 2 == 0 {
                    Ok(false)
                } else {
                    Err(invalid("injected candidate failure"))
                }
            });
            if attempt % 2 == 0 {
                assert_eq!(result.unwrap(), 0);
            } else {
                assert!(result.is_err());
            }
            assert_eq!(topology, original);
            for (slot, (node, pointer)) in topology.nodes.iter().zip(&pointers).enumerate() {
                assert_eq!(
                    (node.provenance.as_ptr(), node.exact_parameters.as_ptr()),
                    *pointer,
                    "attempt={attempt}, slot={slot}, node_id={}",
                    node.node_id
                );
            }
            assert_eq!(budget.live_bytes(), live);
        }
        let moved = edit_nodes(&mut topology, &[0, 1], &scope, |topology, _| {
            {
                let quantity_change = 0.25;
                let quantity_index = 0;
                let quantity_slot = &mut topology.nodes[0].coordinates_m;
                let mut quantity_coordinates = quantity_slot.canonical_value();
                quantity_coordinates[quantity_index] += quantity_change;
                *quantity_slot = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(quantity_coordinates);
            };
            {
                let quantity_change = 0.5;
                let quantity_index = 0;
                let quantity_slot = &mut topology.nodes[1].coordinates_m;
                let mut quantity_coordinates = quantity_slot.canonical_value();
                quantity_coordinates[quantity_index] += quantity_change;
                *quantity_slot = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(quantity_coordinates);
            };
            Ok(true)
        })
        .unwrap();
        assert_eq!(moved, 2);
        let retained: u64 = topology
            .nodes
            .iter()
            .map(|node| node.owned_buffer_bytes().unwrap())
            .sum();
        assert_eq!(budget.live_bytes(), retained);
    }
    assert_eq!(budget.live_bytes(), 0);
}

#[test]
fn denied_edit_keeps_original_nodes_and_releases_all_partial_replacements() {
    let mut topology = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10)
        .topology
        .into_topology();
    let original = topology.clone();
    let budget = AllocationBudget::new(1);
    assert_eq!(
        edit_nodes(&mut topology, &[0, 1, 2], &budget, |_, _| panic!(
            "denial must precede candidate execution"
        ))
        .unwrap_err()
        .kind,
        DelaunaySolverTopologyErrorKind::MemoryBudgetExceeded
    );
    assert_eq!(topology, original);
    assert_eq!(budget.live_bytes(), 0);
}
