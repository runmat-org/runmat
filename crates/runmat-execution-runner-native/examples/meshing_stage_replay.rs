//! Bounded diagnostic replay through the production native meshing stage bridge.
//! Usage: meshing_stage_replay OBJECTS HOST_DIGEST OUTPUT SECONDS (bounded by the original stage budget).
//! Only CurveMesh/SurfaceMesh are eligible; final publication is rejected.
//! Revalidates original inputs and controls. It cannot complete a study or publish
//! a solver mesh; intermediate diagnostic objects stay in OUTPUT only.
use runmat_execution::value::{ValuePayload, ValueRef};
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{cache::CacheImport, object::ObjectInventoryLimits};
use runmat_execution_runner_native::{
    native_meshing_kernel_dispatcher, FilesystemObjectStore, NativeMeshingHostLimits,
};
use runmat_meshing_core::{
    MeshingCancellationSignal, MeshingInputKind, MeshingProgress, MeshingStageKind,
};
use runmat_meshing_execution::{
    execute_serial_stage, import_exact_geometry_objects, import_stage_objects,
    prepare_exact_geometry_input, prepare_result_publication, ExactGeometryObjectRoot,
    MeshingHostWorkload, MeshingProgressSink, MeshingStageObjectRoot,
};
use std::{
    io::Write,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
#[path = "meshing_stage_replay/store.rs"]
mod store;
use store::ReplayStore;
const MAXIMUM_HOST_BYTES: u64 = 8 * 1024 * 1024;

struct Deadline {
    start: Instant,
    cancelled: AtomicBool,
}
impl MeshingCancellationSignal for Deadline {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}
#[derive(Default)]
struct Progress {
    records: u64,
    completed_work: u64,
    search_work: u64,
    peak_memory_bytes: u64,
}
impl MeshingProgressSink for Progress {
    fn record(&mut self, progress: &MeshingProgress) {
        self.records += 1;
        self.completed_work = progress.completed_work;
        self.search_work = progress.consumed_search_work;
        self.peak_memory_bytes = self.peak_memory_bytes.max(progress.peak_memory_bytes);
        if self.records <= 16 {
            eprintln!(
                "REPLAY_PROGRESS {}",
                serde_json::to_string(progress).unwrap()
            );
        }
    }
}

fn roots(
    host: &MeshingHostWorkload,
    store: &ReplayStore,
    limits: ObjectInventoryLimits,
) -> Result<Vec<ValueRef>, Box<dyn std::error::Error>> {
    host.workload
        .inputs
        .iter()
        .map(|input| {
            let digest = Digest::from_bytes(*input.digest.bytes());
            let length = std::fs::metadata(
                store
                    .source
                    .root()
                    .join(digest.to_string().trim_start_matches("sha256:")),
            )?
            .len();
            let root = match input.kind {
                MeshingInputKind::ExactGeometry => {
                    let document = host
                        .geometry_document
                        .clone()
                        .ok_or("exact geometry document missing")?;
                    let objects = import_exact_geometry_objects(
                        store,
                        document,
                        ExactGeometryObjectRoot {
                            digest,
                            encoded_length: length,
                        },
                        limits,
                    )?;
                    prepare_exact_geometry_input(objects, host.artifact_access.clone(), limits)?
                        .root_input()
                        .clone()
                }
                MeshingInputKind::StageArtifact => {
                    let objects = import_stage_objects(
                        store,
                        MeshingStageObjectRoot {
                            digest,
                            encoded_length: length,
                        },
                        limits,
                    )?;
                    let publication =
                        prepare_result_publication(objects, host.artifact_access.clone(), limits)?;
                    let ValuePayload::Object(root) = publication.root_output() else {
                        return Err("stage root is not externalized".into());
                    };
                    root.as_ref().clone()
                }
                _ => {
                    return Err(
                        "diagnostic currently requires exact CAD and accepted meshing stage inputs"
                            .into(),
                    )
                }
            };
            Ok(root)
        })
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    runmat_logging::set_runtime_log_hook(|record| {
        eprintln!("REPLAY_LOG {}", serde_json::to_string(record).unwrap());
    });
    let _logging = runmat_logging::init_logging(runmat_logging::LoggingOptions::default());
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments.len() != 4 {
        return Err("expected OBJECTS HOST_DIGEST OUTPUT SECONDS".into());
    }
    let seconds: u64 = arguments[3].to_str().ok_or("invalid seconds")?.parse()?;
    if seconds == 0 {
        return Err("diagnostic interval must be positive".into());
    }
    let digest: Digest = arguments[1].to_str().ok_or("invalid digest")?.parse()?;
    let host_limits = NativeMeshingHostLimits::default();
    host_limits.validate()?;
    let limits = host_limits.inventory;
    let source = FilesystemObjectStore::open(Path::new(&arguments[0]), limits.max_object_bytes)?;
    let bytes = source
        .read_verified_bounded(digest, MAXIMUM_HOST_BYTES)?
        .ok_or("host descriptor missing")?;
    let host = MeshingHostWorkload::from_canonical_bytes(&bytes)?;
    if !matches!(
        host.workload.stage,
        MeshingStageKind::CurveMesh | MeshingStageKind::SurfaceMesh
    ) {
        return Err("diagnostic only accepts curve/surface stages; publication is excluded".into());
    }
    if seconds
        .checked_mul(1000)
        .is_none_or(|milliseconds| milliseconds > host.ceilings.maximum_wall_time_ms)
    {
        return Err("diagnostic interval exceeds the original stage wall-time budget".into());
    }
    let output = Path::new(&arguments[2]);
    if output.exists() {
        return Err("diagnostic output must be fresh".into());
    }
    runmat_filesystem::NativeFsProvider.create_private_directory_durable(output)?;
    let mut store = ReplayStore {
        source,
        output: FilesystemObjectStore::open(output.join("objects"), limits.max_object_bytes)?,
    };
    let input_roots = roots(&host, &store, limits)?;
    // The revision names this diagnostic host invocation, never the original study.
    let environment = ProgramEnvironment::new(
        1,
        1,
        Digest::sha256(runmat_meshing_execution::BUILD_CONFIGURATION.as_bytes()),
        Digest::sha256(b"native-meshing-stage-diagnostic/v1"),
        "matlab",
    )?;
    let revision = ProgramRevision::new(
        Digest::sha256(b"native-meshing-stage-diagnostic/v1"),
        digest,
        environment,
    )?;
    let invocation = host.prepare_program_request(revision, &input_roots)?;
    invocation.write_to(&mut store)?;
    let deadline = Deadline {
        start: Instant::now(),
        cancelled: AtomicBool::new(false),
    };
    println!(
        "REPLAY_EXECUTE_START stage={:?} partition={} host={digest}",
        host.workload.stage, host.workload.partition.partition_index
    );
    std::io::stdout().flush()?;
    let mut progress = Progress::default();
    // Match production's cheap cancellation probe. A clock read at every geometry
    // checkpoint would distort the very stage this diagnostic measures. The scoped
    // timer exits immediately on completion, including errors and unwinding.
    let result = std::thread::scope(|scope| {
        let (completed, completion) = mpsc::sync_channel::<()>(1);
        let cancellation = &deadline.cancelled;
        scope.spawn(move || {
            if matches!(
                completion.recv_timeout(Duration::from_secs(seconds)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ) {
                cancellation.store(true, Ordering::Relaxed);
            }
        });
        let result = runmat_cpu_execution::with_cpu_pool_cancellable(
            1,
            || deadline.is_cancelled(),
            || {
                execute_serial_stage(
                    invocation.request(),
                    &mut store,
                    &native_meshing_kernel_dispatcher(),
                    &deadline,
                    &mut progress,
                    host_limits.chunk_policy,
                    limits,
                )
            },
        );
        let _ = completed.send(());
        result
    })?;
    let summary = match result {
        Ok(stage) => {
            serde_json::json!({"status":"diagnostic-stage-completed-not-study-acceptance", "result":stage.workload_result()})
        }
        Err(error) => {
            serde_json::json!({"status":"diagnostic-stopped-not-acceptance", "error":error.to_string()})
        }
    };
    let summary = serde_json::json!({"host_digest":digest,"original_controls":host.resolved_request,
        "elapsed_seconds":deadline.start.elapsed().as_secs_f64(), "diagnostic_seconds":seconds,
        "numerical_source":option_env!("RUNMAT_MESHING_UPSTREAM_IMPLEMENTATION"),
        "numerical_build_configuration":runmat_meshing_execution::meshing_build_configuration(host.workload.stage),
        "subtask_cache_root":std::env::var_os("RUNMAT_EXECUTION_MESHING_SUBTASK_CACHE_ROOT").map(std::path::PathBuf::from),
        "progress":{"records":progress.records,"completed_work":progress.completed_work,"search_work":progress.search_work,"peak_memory_bytes":progress.peak_memory_bytes},
        "outcome":summary});
    std::fs::write(
        output.join("diagnostic.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("REPLAY_RESULT {summary}");
    Ok(())
}
