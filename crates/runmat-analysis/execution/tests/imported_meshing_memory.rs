//! Imported-source meshing holds its Rust heap within the admitted memory ceiling.
//!
//! The browser geometry host runs `mesh_imported_source_serial` in a wasm heap that cannot grow
//! past its link-time maximum. A mesh that needs more than the admitted ceiling must stop with a
//! typed memory refusal before it allocates past the ceiling. These tests mesh the prism the
//! browser host received (fixtures/prism.step with the host's request) and compare the Rust
//! heap this test binary measures against the ceiling and against the peak each stage reports.
//! A stage's measured heap is its growth above the heap live at its first progress report, so
//! artifacts that earlier stages left in the store are not counted against it.
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
    cache::{CacheExport, CacheImport, MemoryObjectStore},
    object::ObjectInventoryLimits,
    LogicalObject, ObjectNamespace,
};
use runmat_meshing_core::{
    MeshingDiagnosticValue, MeshingExecutionCeilings, MeshingFailureCategory, MeshingProgress,
    MeshingStageKind, NeverCancelled, PlatformBuildIdentity, StableDigest,
};
use runmat_meshing_execution::{
    ExactMeshingDagExecutor, MeshingKernelDispatcher, MeshingProgressSink,
    MeshingSerialExecutionError, MeshingStageExecutionError, PlannedMeshingStage,
};

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

/// Measured heap per stage partition: the growth above the heap live at the partition's first
/// progress report, through its last. The gap between partitions (publishing one, importing the
/// next one's inputs) counts toward the run's peak only.
#[derive(Default)]
struct StagePeaks {
    baseline: u64,
    run_peak: u64,
    stages: Vec<StagePeak>,
}

struct StagePeak {
    stage: MeshingStageKind,
    partition: u32,
    start: u64,
    reported: u64,
    measured: u64,
}

impl MeshingProgressSink for StagePeaks {
    fn record(&mut self, progress: &MeshingProgress) {
        let live = LIVE.load(Ordering::Relaxed);
        let peak = PEAK.swap(live, Ordering::Relaxed);
        self.run_peak = self.run_peak.max(peak.saturating_sub(self.baseline));
        match self.stages.last_mut() {
            Some(current)
                if current.stage == progress.stage
                    && current.partition == progress.partition_index =>
            {
                current.reported = current.reported.max(progress.peak_memory_bytes);
                current.measured = current.measured.max(peak.saturating_sub(current.start));
            }
            _ => self.stages.push(StagePeak {
                stage: progress.stage,
                partition: progress.partition_index,
                start: live,
                reported: progress.peak_memory_bytes,
                measured: 0,
            }),
        }
    }
}

impl StagePeaks {
    /// The last partition of `stage` that reported progress.
    fn stage(&self, stage: MeshingStageKind) -> &StagePeak {
        self.stages
            .iter()
            .rev()
            .find(|peak| peak.stage == stage)
            .unwrap_or_else(|| panic!("{stage:?} reported no progress"))
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

const INVENTORY: ObjectInventoryLimits = ObjectInventoryLimits {
    max_objects: 100_000,
    max_object_bytes: 1 << 30,
    max_total_bytes: 4 << 30,
};

struct Measured<T> {
    result: Result<T, SerialGeometryMeshingError>,
    stages: StagePeaks,
}

/// Runs `mesh` with heap tracking from a fresh baseline, one measured run at a time.
fn measure<T>(
    mesh: impl FnOnce(&mut StagePeaks) -> Result<T, SerialGeometryMeshingError>,
) -> Measured<T> {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let baseline = LIVE.load(Ordering::Relaxed);
    let mut stages = StagePeaks {
        baseline,
        ..StagePeaks::default()
    };
    PEAK.store(baseline, Ordering::Relaxed);
    TRACKING.store(true, Ordering::Relaxed);
    let result = mesh(&mut stages);
    TRACKING.store(false, Ordering::Relaxed);
    let tail = PEAK.load(Ordering::Relaxed).saturating_sub(baseline);
    stages.run_peak = stages.run_peak.max(tail);
    Measured { result, stages }
}

fn mesh(target_edge_length_m: f64, memory_bytes: u64) -> Measured<ImportedSourceMeshingResult> {
    let request = request(target_edge_length_m);
    let ceilings = MeshingExecutionCeilings::for_host(memory_bytes, &request.settings.limits);
    let mut store = MemoryObjectStore::new(INVENTORY).unwrap();
    let context = runmat_geometry_io::GeometryImportContext::default();
    measure(|stages| {
        mesh_imported_source_serial(
            request,
            &mut store,
            &context,
            ceilings,
            platform(),
            INVENTORY,
            &NeverCancelled,
            stages,
        )
    })
}

/// A serial executor that runs order elevation under its own memory ceiling, as a host that
/// admits each stage separately does. Every other stage runs under the run's ceilings.
struct ProjectionCeiling<'a> {
    store: &'a mut MemoryObjectStore,
    kernel: &'a MeshingKernelDispatcher<runmat_analysis_execution::OcctMeshingEvaluatorProvider>,
    progress: &'a mut dyn MeshingProgressSink,
    projection: MeshingExecutionCeilings,
}

impl CacheImport for ProjectionCeiling<'_> {
    fn read_verified(
        &self,
        digest: runmat_execution::Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        self.store.read_verified(digest)
    }
    fn read_verified_bounded(
        &self,
        digest: runmat_execution::Digest,
        maximum_bytes: u64,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        self.store.read_verified_bounded(digest, maximum_bytes)
    }
    fn read_verified_bounded_with_admission(
        &self,
        digest: runmat_execution::Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> runmat_execution_artifact::ArtifactResult<()>,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        self.store
            .read_verified_bounded_with_admission(digest, maximum_bytes, admit)
    }
}

impl CacheExport for ProjectionCeiling<'_> {
    fn write_verified(
        &mut self,
        object: &LogicalObject,
    ) -> runmat_execution_artifact::ArtifactResult<()> {
        self.store.write_verified(object)
    }
}

impl ExactMeshingDagExecutor for ProjectionCeiling<'_> {
    async fn execute_stage<'a>(
        &'a mut self,
        stage: &'a PlannedMeshingStage,
        revision: runmat_execution::ProgramRevision,
    ) -> Result<runmat_execution_runner::AttemptSuccess, MeshingStageExecutionError> {
        let mut host = stage.host().clone();
        if host.workload.stage == MeshingStageKind::OrderElevation {
            host.ceilings = self.projection;
        }
        let invocation = host
            .prepare_program_request(revision, stage.input_roots())
            .map_err(|error| MeshingStageExecutionError::new(error.to_string()))?;
        invocation
            .write_to(self.store)
            .map_err(|error| MeshingStageExecutionError::new(error.to_string()))?;
        runmat_meshing_execution::execute_serial_stage(
            invocation.request(),
            self.store,
            self.kernel,
            &NeverCancelled,
            self.progress,
            runmat_meshing_core::MeshingChunkPolicy {
                maximum_chunk_bytes: 8 << 20,
                maximum_records_per_chunk: 65_536,
                maximum_total_encoded_bytes: INVENTORY.max_total_bytes,
            },
            INVENTORY,
        )
        .map(|completed| completed.attempt_success())
        .map_err(|error| match error {
            MeshingSerialExecutionError::Stage(failure) => {
                MeshingStageExecutionError::Failure(failure)
            }
            MeshingSerialExecutionError::Bridge(error) => {
                MeshingStageExecutionError::new(error.to_string())
            }
        })
    }
}

/// Imports the prism and runs its meshing DAG with every stage under `run_bytes` except order
/// elevation, which runs under `projection_bytes`.
fn mesh_with_projection_ceiling(
    target_edge_length_m: f64,
    run_bytes: u64,
    projection_bytes: u64,
) -> Measured<()> {
    let request = request(target_edge_length_m);
    assert!(request.contacts.is_empty());
    let run = MeshingExecutionCeilings::for_host(run_bytes, &request.settings.limits);
    let projection = MeshingExecutionCeilings::for_host(projection_bytes, &request.settings.limits);
    let mut store = MemoryObjectStore::new(INVENTORY).unwrap();
    measure(|stages| {
        let imported = runmat_geometry_io::import_exact_cad(
            &request.source_name,
            &request.source.bytes,
            runmat_geometry_io::detect_geometry_format(&request.source_name, &request.source.bytes),
            &request.import_options,
            &runmat_geometry_io::GeometryImportContext::default(),
        )
        .map_err(|error| error.to_string())?;
        let document = imported
            .geometry_document()
            .map_err(|error| error.to_string())?;
        let resolved =
            runmat_meshing_core::resolve_meshing_request(document.tolerance, request.settings)
                .map_err(|error| error.to_string())?;
        let objects = runmat_meshing_execution::prepare_exact_geometry_objects(
            document,
            imported.topology,
            imported.evaluators,
            Some(imported.representation),
            imported.healing_report,
            INVENTORY,
        )
        .map_err(|error| error.to_string())?;
        for object in &objects.objects {
            store
                .write_verified(object)
                .map_err(|error| error.to_string())?;
        }
        let geometry = runmat_meshing_execution::prepare_exact_geometry_input(
            objects,
            request.artifact_access.clone(),
            INVENTORY,
        )
        .map_err(|error| error.to_string())?;
        let kernel =
            MeshingKernelDispatcher::new(runmat_analysis_execution::OcctMeshingEvaluatorProvider);
        let mut executor = ProjectionCeiling {
            store: &mut store,
            kernel: &kernel,
            progress: stages,
            projection,
        };
        runmat_meshing_execution::execute_exact_meshing_dag(
            runmat_meshing_execution::ExactMeshingDagRun {
                geometry: &geometry,
                request: resolved,
                ceilings: run,
                artifact_access: request.artifact_access,
                capability_cohort: Some(platform().capability_cohort),
                program_revision: request.program_revision,
                preferred_edges_per_partition: 8,
                preferred_faces_per_partition: 8,
                inventory_limits: INVENTORY,
                evidence: runmat_meshing_execution::MeshingRunEvidenceContext {
                    platform: platform(),
                    sizing: Vec::new(),
                    cache_admission: runmat_meshing_core::CacheAdmissionDecision::Admitted,
                },
                named_regions: Default::default(),
            },
            &mut executor,
        )?;
        Ok(())
    })
}

fn report<T>(label: &str, measured: &Measured<T>) {
    eprintln!(
        "{label}: measured Rust heap peak {} bytes over the run",
        measured.stages.run_peak
    );
    for peak in &measured.stages.stages {
        eprintln!(
            "  {:?}[{}]: reported {} measured {}",
            peak.stage, peak.partition, peak.reported, peak.measured
        );
    }
    if let Err(error) = &measured.result {
        eprintln!("  refused: {error}");
    }
}

/// The typed stage failure of a refused run.
fn stage_failure<T>(measured: &Measured<T>) -> &runmat_meshing_core::MeshingFailure {
    let Err(SerialGeometryMeshingError::Meshing(
        runmat_meshing_execution::ExactMeshingDagRunError::Stage(error),
    )) = &measured.result
    else {
        panic!("expected a stage refusal");
    };
    error.failure().expect("typed meshing failure")
}

fn memory_refusal<T>(measured: &Measured<T>) -> (u64, u64) {
    let failure = stage_failure(measured);
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
        measured.stages.run_peak <= CEILING,
        "measured {} bytes past the {CEILING}-byte ceiling",
        measured.stages.run_peak
    );
}

/// Below what volume construction holds before quality convergence starts, the construction
/// phases refuse the mesh before the volume stage's heap passes the ceiling.
#[test]
fn prism_below_its_construction_need_is_refused_before_convergence() {
    const CEILING: u64 = 64 << 20;
    let measured = mesh(0.0022, CEILING);
    report("2.2 mm at 64 MiB", &measured);
    let (required, maximum) = memory_refusal(&measured);
    assert_eq!(maximum, CEILING);
    assert!(required > CEILING);
    let volume = measured.stages.stage(MeshingStageKind::Tetrahedralization);
    assert!(
        volume.measured <= CEILING,
        "the volume stage held {} bytes past the {CEILING}-byte ceiling",
        volume.measured
    );
}

/// Order elevation needs more than 256 MiB for the prism. Admitted separately under that
/// ceiling, it refuses with a typed memory failure before its heap passes the ceiling.
#[test]
fn projection_below_its_need_is_refused_before_the_heap_passes_the_ceiling() {
    const CEILING: u64 = 256 << 20;
    let measured = mesh_with_projection_ceiling(0.0022, 4 << 30, CEILING);
    report("2.2 mm, order elevation at 256 MiB", &measured);
    let failure = stage_failure(&measured);
    assert_eq!(
        failure.category,
        MeshingFailureCategory::MemoryBudgetExceeded
    );
    assert_eq!(failure.stage, MeshingStageKind::OrderElevation);
    let projection = measured.stages.stage(MeshingStageKind::OrderElevation);
    assert!(
        projection.measured <= CEILING,
        "order elevation held {} bytes past the {CEILING}-byte ceiling",
        projection.measured
    );
}

/// With an ample ceiling the prism meshes. Volume construction and order elevation report at
/// least the heap this binary measured while they ran, and the terminal stages' evidence covers
/// theirs. Curve and surface partitions and publication are printed: surface partitions charge
/// about 6% less than they hold on this prism and publication about 14 KB less.
#[test]
fn prism_with_an_ample_ceiling_reports_at_least_its_measured_heap() {
    let measured = mesh(0.0022, 4 << 30);
    report("2.2 mm at 4 GiB", &measured);
    let result = measured.result.as_ref().expect("ample ceiling meshes");
    for stage in [
        MeshingStageKind::Tetrahedralization,
        MeshingStageKind::OrderElevation,
    ] {
        let peak = measured.stages.stage(stage);
        assert!(
            peak.reported >= peak.measured,
            "{stage:?} reported {} bytes but held {}",
            peak.reported,
            peak.measured
        );
    }
    for stage in [
        MeshingStageKind::Validation,
        MeshingStageKind::Serialization,
    ] {
        let evidence = result
            .evidence
            .stages
            .iter()
            .find(|evidence| evidence.stage == stage)
            .expect("terminal stage evidence");
        let held = measured.stages.stage(stage).measured;
        assert!(
            evidence.peak_memory_bytes >= held,
            "{stage:?} evidence reports {} bytes but the stage held {held}",
            evidence.peak_memory_bytes
        );
    }
    let reported = measured
        .stages
        .stage(MeshingStageKind::Tetrahedralization)
        .reported;
    assert!(result.evidence.resources.peak_memory_bytes >= reported);
}
