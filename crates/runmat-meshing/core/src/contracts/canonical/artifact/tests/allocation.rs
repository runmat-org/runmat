//! Independent requested-allocation reconciliation; this allocator exists only in core unit tests.
use super::super::super::artifact_resources::validation_scratch_bytes;
use super::{canonical_tetrahedron_solver_mesh, validate_solver_mesh_topology, ElementOrder};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

thread_local! {
    static ALLOCATIONS: Cell<(bool, usize, usize)> = const { Cell::new((false, 0, 0)) };
}
struct TrackedSystem;
#[global_allocator]
static ALLOCATOR: TrackedSystem = TrackedSystem;

fn record(added: usize, removed: usize) {
    // A constant TLS cell allocates no tracking buffers and isolates simultaneous test threads.
    let _ = ALLOCATIONS.try_with(|state| {
        let (enabled, live, peak) = state.get();
        if enabled {
            let next = live.saturating_sub(removed).saturating_add(added);
            state.set((enabled, next, peak.max(next)));
        }
    });
}
unsafe impl GlobalAlloc for TrackedSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout.size(), 0);
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record(layout.size(), 0);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record(0, layout.size());
        unsafe {
            System.dealloc(pointer, layout);
        }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let next = unsafe { System.realloc(pointer, layout, size) };
        if !next.is_null() {
            record(size, layout.size());
        }
        next
    }
}

#[test]
fn successful_topology_validation_reconciles_actual_requested_allocations() {
    use crate::fixtures::{
        connected_quadratic_tetrahedron_solver_mesh, connected_tetrahedron_solver_mesh,
        quadratic_slab_mesh, slab_mesh, three_region_tetrahedron_solver_mesh_with_order,
    };
    let meshes = [
        canonical_tetrahedron_solver_mesh(ElementOrder::Tet4),
        canonical_tetrahedron_solver_mesh(ElementOrder::Tet10),
        connected_tetrahedron_solver_mesh(),
        connected_quadratic_tetrahedron_solver_mesh(),
        three_region_tetrahedron_solver_mesh_with_order(ElementOrder::Tet4),
        three_region_tetrahedron_solver_mesh_with_order(ElementOrder::Tet10),
        slab_mesh(2),
        quadratic_slab_mesh(2),
    ];
    for mesh in meshes {
        let order = mesh.resolved_request.element_order;
        let expected = validation_scratch_bytes(&mesh.topology, order).unwrap();
        ALLOCATIONS.with(|state| state.set((true, 0, 0)));
        let result = validate_solver_mesh_topology(&mesh.topology, &mesh.resolved_request);
        let (_, live, peak) = ALLOCATIONS.with(|state| {
            let record = state.get();
            state.set((false, 0, 0));
            record
        });
        result.unwrap();
        println!("validation {order:?} nodes={} elements={} requested_peak={peak} expected={expected} released={live}", mesh.topology.nodes.len(), mesh.topology.volume_elements.len());
        assert_eq!(live, 0, "validation must release scratch for {order:?}");
        assert_eq!(
            peak as u64, expected,
            "requested layout peak must match the owning phase inventory for {order:?}"
        );
    }
}

#[test]
fn compact_topology_decode_peak_stays_within_its_typed_admission() {
    use crate::fixtures::{
        connected_quadratic_tetrahedron_solver_mesh, quadratic_slab_mesh, slab_mesh,
        three_region_tetrahedron_solver_mesh_with_order,
    };
    let limits = runmat_canonical_codec::CanonicalLimits::new(1 << 30, 1 << 24, 1 << 20, 64);
    let control =
        super::super::super::codec::MeshingCanonicalCodecControl::new(&crate::NeverCancelled);
    for mesh in [
        canonical_tetrahedron_solver_mesh(ElementOrder::Tet10),
        connected_quadratic_tetrahedron_solver_mesh(),
        three_region_tetrahedron_solver_mesh_with_order(ElementOrder::Tet4),
        slab_mesh(6),
        quadratic_slab_mesh(4),
        slab_mesh(16),
        quadratic_slab_mesh(10),
    ] {
        let bytes = runmat_canonical_codec::encode_contract(
            b"decode-peak\0",
            "topology",
            &mesh.topology,
            limits,
        )
        .unwrap();
        let value = runmat_canonical_codec::validated_contract_value(
            b"decode-peak\0",
            "topology",
            &bytes,
            limits,
            &control,
        )
        .unwrap();
        let admitted =
            super::super::super::codec::topology_decode_admission(value, &control).unwrap();
        ALLOCATIONS.with(|state| state.set((true, 0, 0)));
        let decoded: super::SolverMeshTopology =
            runmat_canonical_codec::decode_contract(b"decode-peak\0", "topology", &bytes, limits)
                .unwrap();
        let (_, _, peak) = ALLOCATIONS.with(|state| {
            let record = state.get();
            state.set((false, 0, 0));
            record
        });
        assert_eq!(decoded, *mesh.topology);
        println!(
            "decode elements={} peak={peak} admitted={admitted} typed={}",
            mesh.topology.volume_elements.len(),
            runmat_canonical_codec::canonical_value_typed_resident_bytes_estimate(value, &control)
                .unwrap()
        );
        assert!(
            peak as u64 <= admitted,
            "decode peak {peak} exceeds its admitted {admitted}"
        );
    }
}
