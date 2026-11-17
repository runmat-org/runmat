//! Imported-source meshing holds its Rust heap within the admitted memory ceiling.
//!
//! The browser geometry host runs `mesh_imported_source_serial` in a wasm heap that cannot grow
//! past its link-time maximum. A mesh that needs more than the admitted ceiling must stop with a
//! typed memory refusal before it allocates past the ceiling. These tests mesh the prism the
//! browser host received (fixtures/prism.step with the host's request) and compare the Rust
//! heap this test binary measures against the ceiling and against the peak the evidence reports.
//! OCCT's own allocations are outside this allocator; native pointers are twice the size of
//! wasm32's, so the native measurement overstates the browser heap.
#![cfg(feature = "occt-native")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use runmat_analysis_execution::{
    mesh_imported_source_serial, ImportedSourceMeshingRequest, ImportedSourceMeshingResult,
    SerialGeometryMeshingError,
};
use runmat_execution_artifact::{
    cache::MemoryObjectStore, object::ObjectInventoryLimits, LogicalObject, ObjectNamespace,
};
use runmat_meshing_core::{
    MeshingDiagnosticValue, MeshingExecutionCeilings, MeshingFailureCategory, MeshingProgress,
    MeshingStageKind, NeverCancelled, PlatformBuildIdentity, StableDigest,
};
use runmat_meshing_execution::MeshingProgressSink;

struct Counting;

static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static TRACKING: AtomicBool = AtomicBool::new(false);
// One measured run at a time: the counters are process-wide.
static SERIAL: Mutex<()> = Mutex::new(());

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn grow(size: usize) {
    let live = LIVE.fetch_add(size as u64, Ordering::Relaxed) + size as u64;
    if TRACKING.load(Ordering::Relaxed) {
        PEAK.fetch_max(live, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            grow(layout.size());
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            grow(layout.size());
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if !replacement.is_null() {
            LIVE.fetch_sub(layout.size() as u64, Ordering::Relaxed);
            grow(new_size);
        }
        replacement
    }
}

/// Measured heap growth above the run's baseline at each stage's progress reports.
#[derive(Default)]
struct StagePeaks {
    baseline: u64,
    stages: Vec<(MeshingStageKind, u64, u64)>,
}

impl MeshingProgressSink for StagePeaks {
    fn record(&mut self, progress: &MeshingProgress) {
        let measured = PEAK
            .swap(LIVE.load(Ordering::Relaxed), Ordering::Relaxed)
            .saturating_sub(self.baseline);
        match self.stages.last_mut() {
            Some((stage, reported, peak)) if *stage == progress.stage => {
                *reported = (*reported).max(progress.peak_memory_bytes);
                *peak = (*peak).max(measured);
            }
            _ => self
                .stages
                .push((progress.stage, progress.peak_memory_bytes, measured)),
        }
    }
}

fn request(target_edge_length_m: f64) -> ImportedSourceMeshingRequest {
    let bytes = include_bytes!("fixtures/prism.step").to_vec();
    let source = LogicalObject::new(
        ObjectNamespace::InputValue,
        "input/exact-cad-source",
        "application/octet-stream",
        bytes,
    )
    .unwrap();
    let mut request: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/prism-imported-meshing-request.json")).unwrap();
    request["source"] = serde_json::to_value(&source).unwrap();
    request["settings"]["target_edge_length_m"] = target_edge_length_m.into();
    serde_json::from_value(request).unwrap()
}

fn platform() -> PlatformBuildIdentity {
    PlatformBuildIdentity {
        capability_cohort: "native-serial-occt-test".into(),
        target_triple: "native-test".into(),
        build_digest: StableDigest::from_bytes([1; 32]),
        exact_kernel_abi: runmat_geometry_io::native_exact_kernel_abi(),
    }
}

struct Measured {
    result: Result<ImportedSourceMeshingResult, SerialGeometryMeshingError>,
    peak_bytes: u64,
    stages: StagePeaks,
}

fn mesh(target_edge_length_m: f64, memory_bytes: u64) -> Measured {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let request = request(target_edge_length_m);
    let ceilings = MeshingExecutionCeilings::for_host(memory_bytes, &request.settings.limits);
    let inventory = ObjectInventoryLimits {
        max_objects: 100_000,
        max_object_bytes: 1 << 30,
        max_total_bytes: 4 << 30,
    };
    let mut store = MemoryObjectStore::new(inventory).unwrap();
    let context = runmat_geometry_io::GeometryImportContext::default();
    let baseline = LIVE.load(Ordering::Relaxed);
    let mut stages = StagePeaks {
        baseline,
        ..StagePeaks::default()
    };
    PEAK.store(baseline, Ordering::Relaxed);
    TRACKING.store(true, Ordering::Relaxed);
    let result = mesh_imported_source_serial(
        request,
        &mut store,
        &context,
        ceilings,
        platform(),
        inventory,
        &NeverCancelled,
        &mut stages,
    );
    TRACKING.store(false, Ordering::Relaxed);
    let peak_bytes = stages
        .stages
        .iter()
        .map(|(_, _, measured)| *measured)
        .chain(std::iter::once(
            PEAK.load(Ordering::Relaxed).saturating_sub(baseline),
        ))
        .max()
        .unwrap_or(0);
    Measured {
        result,
        peak_bytes,
        stages,
    }
}

fn report(label: &str, measured: &Measured) {
    eprintln!(
        "{label}: measured Rust heap peak {} bytes",
        measured.peak_bytes
    );
    for (stage, reported, peak) in &measured.stages.stages {
        eprintln!("  {stage:?}: reported {reported} measured {peak}");
    }
    match &measured.result {
        Ok(result) => {
            eprintln!(
                "  completed: reported peak {} bytes, {} stages",
                result.evidence.resources.peak_memory_bytes,
                result.evidence.stages.len()
            );
            for stage in &result.evidence.stages {
                eprintln!(
                    "    evidence {:?}: peak {}",
                    stage.stage, stage.peak_memory_bytes
                );
            }
        }
        Err(error) => {
            eprintln!("  refused: {error}");
            if let SerialGeometryMeshingError::Meshing(run) = error {
                eprintln!("  failure: {run:?}");
            }
        }
    }
}

fn memory_refusal(measured: &Measured) -> (u64, u64) {
    let Err(SerialGeometryMeshingError::Meshing(
        runmat_meshing_execution::ExactMeshingDagRunError::Stage(error),
    )) = &measured.result
    else {
        panic!("expected a stage refusal");
    };
    let failure = error.failure().expect("typed meshing failure");
    assert_eq!(
        failure.category,
        MeshingFailureCategory::MemoryBudgetExceeded
    );
    assert_eq!(failure.stage, MeshingStageKind::Tetrahedralization);
    let count = |values: &[runmat_meshing_core::MeshingDiagnosticEntry]| {
        values
            .iter()
            .find_map(|entry| match (&*entry.name, &entry.value) {
                ("memory_bytes", MeshingDiagnosticValue::Count(count)) => Some(*count),
                _ => None,
            })
            .expect("memory_bytes diagnostic")
    };
    (
        count(&failure.achieved_values),
        count(&failure.request_values),
    )
}

/// The browser's 2.2 mm prism needs more than 256 MiB. The volume stage refuses it with a typed
/// memory refusal that names the requirement and the ceiling, and the Rust heap never grows past
/// the ceiling on the way.
#[test]
fn prism_below_its_need_is_refused_before_the_heap_passes_the_ceiling() {
    const CEILING: u64 = 256 << 20;
    let measured = mesh(0.0022, CEILING);
    report("2.2 mm at 256 MiB", &measured);
    let (required, maximum) = memory_refusal(&measured);
    assert_eq!(maximum, CEILING);
    assert!(required > CEILING);
    assert!(
        measured.peak_bytes <= CEILING,
        "measured {} bytes past the {CEILING}-byte ceiling",
        measured.peak_bytes
    );
}

/// With an ample ceiling the prism meshes, and the tetrahedral stage reports at least the heap
/// this binary measured while that stage ran.
#[test]
fn prism_with_an_ample_ceiling_reports_at_least_its_measured_heap() {
    let measured = mesh(0.0022, 4 << 30);
    report("2.2 mm at 4 GiB", &measured);
    let result = measured.result.as_ref().expect("ample ceiling meshes");
    let reported = result
        .evidence
        .stages
        .iter()
        .find(|stage| stage.stage == MeshingStageKind::Tetrahedralization)
        .expect("tetrahedral stage evidence")
        .peak_memory_bytes;
    let observed = measured
        .stages
        .stages
        .iter()
        .find(|(stage, _, _)| *stage == MeshingStageKind::Tetrahedralization)
        .expect("tetrahedral stage progress")
        .2;
    assert!(
        reported >= observed,
        "tetrahedral stage reported {reported} bytes but held {observed}"
    );
    assert!(result.evidence.resources.peak_memory_bytes >= reported);
}
