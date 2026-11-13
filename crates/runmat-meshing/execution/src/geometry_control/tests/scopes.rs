use super::*;
use runmat_geometry_core::{GeometryAllocationReservation, GeometryAllocationScope};

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Nested iteration scopes release temporary allocations, transfer only retained storage
//   and preserve global work counts and the original shared ceiling. This is declared storage
//   accounting, not allocator RSS or whole-model acceptance.
#[test]
fn nested_scopes_release_scratch_and_transfer_only_live_results() {
    let control = bounded_control(100);
    {
        let face = GeometryAllocationScope::new(&control);
        for _ in 0..1000 {
            let iteration = GeometryAllocationScope::new(&face);
            iteration.consume_allocation_bytes(60).unwrap();
            iteration.consume_search_work(1).unwrap();
            iteration.consume_iterations(1).unwrap();
            {
                let _scratch = GeometryAllocationReservation::new(&iteration, 40).unwrap();
            }
            assert_eq!(control.usage().live_allocation_bytes, 60);
        }
        assert_eq!(control.usage().live_allocation_bytes, 0);
        let iteration = GeometryAllocationScope::new(&face);
        iteration.consume_allocation_bytes(30).unwrap();
        iteration.retain_bytes(20);
        drop(iteration);
        assert_eq!(control.usage().live_allocation_bytes, 20);
        face.retain_bytes(20);
    }
    assert_eq!(control.usage().live_allocation_bytes, 20);
    assert_eq!(control.usage().allocation_bytes, 100);
    assert_eq!(control.usage().iterations, 1000);
    assert_eq!(control.usage().search_work, 1000);
    assert!(GeometryAllocationReservation::new(&control, 81).is_err());
}

#[test]
fn scope_denial_and_unwind_preserve_parent_inventory() {
    let control = bounded_control(100);
    control.consume_allocation_bytes(20).unwrap();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let scope = GeometryAllocationScope::new(&control);
        scope.consume_allocation_bytes(80).unwrap();
        assert!(scope.consume_allocation_bytes(1).is_err());
        panic!("exercise nested allocation cleanup");
    }));
    assert!(outcome.is_err());
    assert_eq!(control.usage().live_allocation_bytes, 20);
    assert_eq!(control.usage().allocation_bytes, 100);
}

#[test]
fn retained_scope_transfer_owns_its_lifetime_and_denial_keeps_the_scope_charged() {
    let control = bounded_control(100);
    let scope = GeometryAllocationScope::new(&control);
    scope.consume_allocation_bytes(80).unwrap();
    assert!(scope.transfer_reservation(81).is_err());
    assert_eq!(control.usage().live_allocation_bytes, 80);
    let retained = scope.transfer_reservation(30).unwrap();
    assert_eq!(retained.bytes(), 30);
    drop(scope);
    assert_eq!(control.usage().live_allocation_bytes, 30);
    assert!(control.consume_allocation_bytes(71).is_err());
    drop(retained);
    assert_eq!(control.usage().live_allocation_bytes, 0);
    assert_eq!(control.usage().allocation_bytes, 80);
}
