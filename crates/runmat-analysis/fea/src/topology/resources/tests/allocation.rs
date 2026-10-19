//! Independent requested-layout reconciliation, isolated to the FEA unit-test process.
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

pub(crate) fn measured<T>(run: impl FnOnce() -> T) -> (T, usize, usize) {
    ALLOCATIONS.with(|state| state.set((true, 0, 0)));
    let value = run();
    let (_, live, peak) = ALLOCATIONS.with(|state| {
        let record = state.get();
        state.set((false, 0, 0));
        record
    });
    (value, live, peak)
}

#[test]
fn projection_derivation_and_clone_fit_the_shared_requested_inventory() {
    use crate::topology::AnalysisProjection;
    use runmat_meshing_core::{
        fixtures::{
            canonical_tetrahedron_solver_mesh, connected_quadratic_tetrahedron_solver_mesh,
            connected_tetrahedron_solver_mesh, quadratic_slab_mesh, slab_mesh,
        },
        ElementOrder,
    };
    for mesh in [
        canonical_tetrahedron_solver_mesh(ElementOrder::Tet4),
        canonical_tetrahedron_solver_mesh(ElementOrder::Tet10),
        connected_tetrahedron_solver_mesh(),
        connected_quadratic_tetrahedron_solver_mesh(),
        slab_mesh(2),
        quadratic_slab_mesh(2),
    ] {
        let estimate = AnalysisProjection::derivation_resident_bytes_estimate(&mesh).unwrap();
        let (projection, live, peak) = measured(|| AnalysisProjection::derive(&mesh));
        let projection = projection.unwrap();
        assert!(live <= peak);
        assert!(
            peak as u64 <= estimate,
            "requested projection peak {peak} exceeds {estimate}"
        );
        let clone_estimate = projection.clone_resident_bytes_estimate().unwrap();
        let (cold, cloned, clone_peak) = measured(|| projection.clone());
        assert_eq!(clone_peak as u64, clone_estimate);
        assert_eq!(cloned, clone_peak);
        assert_eq!(cold.node_neighbors, projection.node_neighbors);
        println!("projection {:?} nodes={} elements={} requested_peak={peak} retained={live} estimate={estimate} clone_peak={clone_peak} clone_estimate={clone_estimate}",mesh.resolved_request.element_order,projection.nodes.len(),projection.elements.len());
    }
}

#[test]
fn native_field_validation_fits_its_own_counted_and_codec_workspace() {
    use crate::{
        results::field::FiniteVolumeFieldTopology,
        topology::{AnalysisProjection, FiniteVolumeTopology},
    };
    let mesh = runmat_meshing_core::fixtures::slab_mesh(2);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let topology = FiniteVolumeTopology::derive(&projection, Default::default()).unwrap();
    let layout = FiniteVolumeFieldTopology::derive(&projection, &topology).unwrap();
    let counts = layout.entity_counts();
    let estimate = FiniteVolumeFieldTopology::validation_workspace_bytes(
        counts.nodes,
        counts.volume_elements,
        counts.oriented_faces,
    )
    .unwrap();
    let (verdict, live, peak) = measured(|| layout.validate());
    verdict.unwrap();
    assert_eq!(live, 0);
    assert!(
        peak as u64 <= estimate,
        "native validator peak {peak} exceeds {estimate}"
    );
    println!("native field validation requested_peak={peak} estimate={estimate} released={live}");
}
