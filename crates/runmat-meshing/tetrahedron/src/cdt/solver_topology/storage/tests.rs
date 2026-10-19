use super::*;
use crate::cdt::test_fixtures::allocation::AllocationBudget;
use runmat_geometry_core::GeometryAllocationScope;
#[test]
fn reallocation_admits_overlapping_buffers_and_denial_preserves_original_values() {
    for limit in [71, 72] {
        let budget = AllocationBudget::new(limit);
        {
            let scope = GeometryAllocationScope::new(&budget);
            let storage = Storage(Some(&scope));
            let mut values = storage.vector::<u64>(3).unwrap();
            values.extend([11, 22, 33]);
            let original_pointer = values.as_ptr();
            let result = storage.push(&mut values, 44);
            if limit == 71 {
                assert_eq!(
                    result.unwrap_err().kind,
                    DelaunaySolverTopologyErrorKind::MemoryBudgetExceeded
                );
                assert_eq!(values, [11, 22, 33]);
                assert_eq!(values.as_ptr(), original_pointer);
                assert_eq!(budget.live_bytes(), 24);
            } else {
                result.unwrap();
                assert_eq!(values, [11, 22, 33, 44]);
                assert_eq!(budget.peak_bytes(), 72);
                assert_eq!(budget.live_bytes(), 48);
            }
        }
        assert_eq!(budget.live_bytes(), 0);
    }
}
#[test]
fn retained_node_copy_matches_nested_capacity_inventory_and_releases_on_failure() {
    let artifact = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
        runmat_meshing_core::ElementOrder::Tet10,
    );
    let source = &artifact.topology.nodes[0];
    let budget = AllocationBudget::new(512 << 20);
    let required;
    {
        let scope = GeometryAllocationScope::new(&budget);
        let copy = Storage(Some(&scope)).node(source).unwrap();
        assert_eq!(copy, *source);
        required = copy.owned_buffer_bytes().unwrap();
        assert_eq!(budget.live_bytes(), required);
    }
    assert_eq!(budget.live_bytes(), 0);
    let short = AllocationBudget::new(required - 1);
    {
        let scope = GeometryAllocationScope::new(&short);
        assert_eq!(
            Storage(Some(&scope)).node(source).unwrap_err().kind,
            DelaunaySolverTopologyErrorKind::MemoryBudgetExceeded
        );
    }
    assert_eq!(short.live_bytes(), 0);
}
