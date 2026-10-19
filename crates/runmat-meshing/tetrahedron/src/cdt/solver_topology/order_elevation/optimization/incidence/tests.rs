use super::*;
use crate::cdt::test_fixtures::allocation::AllocationBudget;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
#[test]
fn shared_incidence_covers_corners_and_midsides_and_releases_its_full_inventory() {
    let topology = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10).topology;
    let budget = AllocationBudget::new(512 << 20);
    let peak;
    {
        let incidence = Incidence::prepare(&topology, &budget).unwrap();
        for node in &topology.nodes {
            let expected = topology
                .volume_elements
                .iter()
                .enumerate()
                .filter_map(|(index, element)| {
                    element.node_ids.contains(&node.node_id).then_some(index)
                })
                .collect::<Vec<_>>();
            assert_eq!(incidence.elements(node.node_id).unwrap(), expected);
        }
        assert_eq!(
            budget.live_bytes(),
            ((incidence.offsets.capacity() + incidence.elements.capacity())
                * std::mem::size_of::<usize>()) as u64
        );
        peak = budget.peak_bytes();
    }
    assert_eq!(budget.live_bytes(), 0);
    let short = AllocationBudget::new(peak - 1);
    assert!(Incidence::prepare(&topology, &short).is_err());
    assert_eq!(short.live_bytes(), 0);
}
