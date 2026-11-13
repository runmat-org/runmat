use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use runmat_execution::cancellation::CancellationSource;
use runmat_execution::security::{NodeClockEndV1, NodeClockStartV1};
use runmat_execution::state::TaskState;
use runmat_execution::{CancellationReason, Digest, ExecutionScopeId, PoolId, TaskId};
use runmat_execution_artifact::archive::{read_bundle, ArchiveLimits};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::{
    ArtifactResult, GraphExecutionRequest, ProgramExecutionRequest, ProjectRevisionRecord,
    WorkerProgramRequest,
};
use runmat_execution_runner::pool::ResizeRequest;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptReport, AttemptRequest, AttemptSuccess, Driver, DriverAction, DriverCommand,
    DriverConfig, PoolSpec, TaskSubmission,
};
use runmat_execution_transport_native::control::DriverArtifactDownload;
use tokio::sync::mpsc;
use tokio::sync::{oneshot, Mutex as AsyncMutex, RwLock};

use super::pool_progress::RemoteTaskCompletion;
use super::RemoteWorkerChannel;
use crate::{NativeExecutionError, NativeExecutionResult};

type CompletionResult = Result<AttemptSuccess, crate::NativeProgramFailure>;

/// Signed transfer authority for one already-admitted, immutable Graph
/// Program. The scheduler retains the source for retries; attempt frames carry
/// only the digest and length issued by a worker after authentication.
pub struct RemoteGraphTransfer {
    pub(super) encrypted_manifest: Arc<[u8]>,
    pub(super) ordered_downloads: Arc<[DriverArtifactDownload]>,
    pub(super) maximum_program_bytes: u64,
    pub(super) program_digest: Digest,
    pub(super) program_length: u64,
}
pub(super) type SignedGraphTransfer = RemoteGraphTransfer;
mod graph_transfer;

mod admission;
mod attempt_execution;
mod clock;
mod completion;
mod console;
mod input_inventory;
mod inspection;
mod membership;
mod object_registration;
mod recovery;
mod route_loss;
mod timing;

pub(super) use recovery::{AuthoritativeUnstartedInventory, OriginalWorkerOwner};
pub use recovery::{RemotePoolCheckpoint, RemotePoolRestoreAuthority};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct WorkerRouteIdentity {
    worker_id: runmat_execution::identity::WorkerId,
    generation: u64,
}

#[derive(Clone)]
struct WorkerRoute {
    identity: WorkerRouteIdentity,
    channel: Arc<dyn RemoteWorkerChannel>,
}

enum FailedRoute {
    Observed(WorkerRouteIdentity),
    Missing,
}

// A terminal run checkpoint must not outrun a worker's cancellation acknowledgement.
// Keep this guard even if the dispatch future is dropped or panics: in that case
// cancellation control cleanup is incomplete and recovery must be indeterminate.
// Successful control cleanup does not certify physical worker retirement.
struct CancellationDispatchGuard(Arc<RemotePoolDriver>, bool);

impl Drop for CancellationDispatchGuard {
    fn drop(&mut self) {
        if !self.1 {
            self.0
                .cancellation_ownership_failed
                .store(true, Ordering::Release);
        }
        self.0.cancellation_in_flight.fetch_sub(1, Ordering::AcqRel);
        self.0.cancellation_settled.notify_waiters();
    }
}

/// Portable scheduler composition over allocation-scoped remote worker routes.
pub struct RemotePoolDriver {
    pub(super) scope_id: ExecutionScopeId,
    pub(super) pool_id: PoolId,
    direct_bundle: Option<DirectPoolBundle>,
    pub(super) driver: Mutex<Driver>,
    console_retention: Mutex<Option<Arc<crate::console_retention::ConsoleRetentionOwner>>>,
    console_max_in_flight: u32,
    worker_registration: AsyncMutex<()>,
    channels: RwLock<HashMap<runmat_execution::identity::WorkerId, WorkerRoute>>,
    original_owners:
        Mutex<HashMap<runmat_execution::identity::WorkerId, recovery::OriginalWorkerOwner>>,
    applied_software_readiness: Mutex<HashSet<runmat_execution::identity::WorkerId>>,
    next_route_generation: AtomicU64,
    installed_nodes: AsyncMutex<BTreeSet<String>>,
    value_scope: String,
    values: super::pool_values::RemoteValueCatalog,
    execution_objects: super::pool_objects::RemoteObjectCatalog,
    pub(super) programs: Mutex<HashMap<TaskId, Arc<WorkerProgramRequest>>>,
    graph_transfers: Mutex<HashMap<TaskId, Arc<SignedGraphTransfer>>>,
    // The exact authenticated parent stage must outlive every active Graph
    // attempt. Spawned attempt tasks retain this pool Arc even if their
    // caller's bounded cancellation drain becomes indeterminate.
    _graph_stage: Mutex<Option<super::program_staging::StagedGraphProgram>>,
    attempt_cancellation: Mutex<HashMap<runmat_execution::identity::AttemptId, CancellationSource>>,
    input_inventories: Mutex<input_inventory::InputInventories>,
    pub(super) completions: Mutex<HashMap<TaskId, oneshot::Sender<(u64, CompletionResult)>>>,
    completion_sequence: AtomicU64,
    pub(super) progress: Mutex<HashMap<TaskId, Arc<super::pool_progress::RemoteProgressBuffer>>>,
    pub(super) collectives: Arc<crate::driver::collective::ProcessCollectiveBroker>,
    pub(super) fea_provider: crate::fea_proxy::Provider,
    admission: Mutex<()>,
    restored_completions: Mutex<HashMap<TaskId, RemoteTaskCompletion>>,
    task_wall_intervals: Mutex<HashMap<TaskId, timing::TaskWallInterval>>,
    clock_state: Mutex<()>,
    wave_admission: AsyncMutex<()>,
    clock_commands: Mutex<Option<mpsc::Sender<super::clock_observer::ClockCommand>>>,
    wave_clocks: Mutex<HashMap<super::clock_observer::RemoteWaveIdentity, [u8; 16]>>,
    certified_waves: Mutex<
        HashMap<super::clock_observer::RemoteWaveIdentity, super::clock_observer::CertifiedWave>,
    >,
    clock_starts: Mutex<HashMap<[u8; 16], NodeClockStartV1>>,
    clock_ends: Mutex<HashMap<[u8; 16], NodeClockEndV1>>,
    task_clocks: Mutex<HashMap<TaskId, [u8; 16]>>,
    uncheckpointed_tasks: Mutex<HashSet<TaskId>>,
    held_launches: Mutex<Vec<AttemptRequest>>,
    committed_task: tokio::sync::Notify,
    committed_event_watermark: AtomicU64,
    cancellation_in_flight: AtomicUsize,
    cancellation_settled: tokio::sync::Notify,
    cancellation_ownership_failed: AtomicBool,
}

/// Direct V7 admission is backed by the installed bundle. Graph V10 instead
/// carries exact ordered selected archives in each task request.
struct DirectPoolBundle {
    digest: Digest,
    identity: Digest,
    project_revision: ProjectRevisionRecord,
    bytes: Arc<[u8]>,
}

impl CacheImport for RemotePoolDriver {
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.execution_objects
            .get_by_digest_admitted(digest, maximum_bytes, admit)
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.execution_objects
            .get_by_digest_bounded(digest, maximum_bytes)
    }

    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.read_verified_bounded(digest, u64::MAX)
    }
}

impl RemotePoolDriver {
    pub fn new(
        scope_id: ExecutionScopeId,
        pool: PoolSpec,
        driver_fence: u64,
        bundle: impl Into<Arc<[u8]>>,
    ) -> NativeExecutionResult<Arc<Self>> {
        Self::new_with_value_scope(scope_id, pool, driver_fence, bundle, scope_id.to_string())
    }

    pub fn new_with_value_scope(
        scope_id: ExecutionScopeId,
        pool: PoolSpec,
        driver_fence: u64,
        bundle: impl Into<Arc<[u8]>>,
        value_scope: impl Into<String>,
    ) -> NativeExecutionResult<Arc<Self>> {
        let bundle = bundle.into();
        let decoded = read_bundle(bundle.as_ref(), ArchiveLimits::default()).map_err(|error| {
            NativeExecutionError::Protocol(format!("remote pool bundle is invalid: {error}"))
        })?;
        let bundle_identity = decoded.identity().map_err(|error| {
            NativeExecutionError::Protocol(format!(
                "remote pool bundle identity is invalid: {error}"
            ))
        })?;
        let direct_bundle = DirectPoolBundle {
            digest: Digest::sha256(bundle.as_ref()),
            identity: bundle_identity,
            project_revision: decoded.manifest.project_revision,
            bytes: bundle,
        };
        Self::new_with_authority(
            scope_id,
            pool,
            driver_fence,
            Some(direct_bundle),
            value_scope,
        )
    }

    pub fn new_graph(
        scope_id: ExecutionScopeId,
        pool: PoolSpec,
        driver_fence: u64,
        value_scope: impl Into<String>,
    ) -> NativeExecutionResult<Arc<Self>> {
        Self::new_with_authority(scope_id, pool, driver_fence, None, value_scope)
    }

    fn new_with_authority(
        scope_id: ExecutionScopeId,
        pool: PoolSpec,
        driver_fence: u64,
        direct_bundle: Option<DirectPoolBundle>,
        value_scope: impl Into<String>,
    ) -> NativeExecutionResult<Arc<Self>> {
        let pool_id = pool.id;
        let console_max_in_flight = pool.max_in_flight;
        let mut driver = Driver::new(DriverConfig::default(), driver_fence)?;
        driver.handle(DriverCommand::RegisterScope {
            scope_id,
            parent: None,
        })?;
        driver.handle(DriverCommand::CreatePool(pool))?;
        driver.handle(DriverCommand::Tick {
            now_millis: now_millis(),
        })?;
        Ok(Arc::new(Self {
            scope_id,
            pool_id,
            direct_bundle,
            driver: Mutex::new(driver),
            console_retention: Mutex::new(None),
            console_max_in_flight,
            worker_registration: AsyncMutex::new(()),
            channels: RwLock::new(HashMap::new()),
            original_owners: Mutex::new(HashMap::new()),
            applied_software_readiness: Mutex::new(HashSet::new()),
            next_route_generation: AtomicU64::new(1),
            installed_nodes: AsyncMutex::new(BTreeSet::new()),
            value_scope: value_scope.into(),
            values: super::pool_values::RemoteValueCatalog::default(),
            execution_objects: super::pool_objects::RemoteObjectCatalog::new()?,
            programs: Mutex::new(HashMap::new()),
            graph_transfers: Mutex::new(HashMap::new()),
            _graph_stage: Mutex::new(None),
            attempt_cancellation: Mutex::new(HashMap::new()),
            input_inventories: Mutex::new(input_inventory::InputInventories::default()),
            completions: Mutex::new(HashMap::new()),
            completion_sequence: AtomicU64::new(0),
            progress: Mutex::new(HashMap::new()),
            collectives: Arc::new(crate::driver::collective::ProcessCollectiveBroker::default()),
            fea_provider: crate::fea_proxy::capture(),
            admission: Mutex::new(()),
            restored_completions: Mutex::new(HashMap::new()),
            task_wall_intervals: Mutex::new(HashMap::new()),
            clock_state: Mutex::new(()),
            clock_commands: Mutex::new(None),
            wave_admission: AsyncMutex::new(()),
            wave_clocks: Mutex::new(HashMap::new()),
            certified_waves: Mutex::new(HashMap::new()),
            clock_starts: Mutex::new(HashMap::new()),
            clock_ends: Mutex::new(HashMap::new()),
            task_clocks: Mutex::new(HashMap::new()),
            uncheckpointed_tasks: Mutex::new(HashSet::new()),
            held_launches: Mutex::new(Vec::new()),
            committed_task: tokio::sync::Notify::new(),
            committed_event_watermark: AtomicU64::new(0),
            cancellation_in_flight: AtomicUsize::new(0),
            cancellation_settled: tokio::sync::Notify::new(),
            cancellation_ownership_failed: AtomicBool::new(false),
        }))
    }

    pub fn register_value_object(
        &self,
        reference: runmat_execution::value::ValueRef,
        encoded: impl Into<Arc<[u8]>>,
    ) -> NativeExecutionResult<()> {
        self.values
            .register(reference, encoded.into(), &self.value_scope)
    }

    pub(super) fn resolve_value_object(
        &self,
        reference: &runmat_execution::value::ValueRef,
    ) -> NativeExecutionResult<runmat_execution::value::ValuePayload> {
        self.values.resolve(reference, &self.value_scope)
    }

    pub const fn scope_id(&self) -> ExecutionScopeId {
        self.scope_id
    }

    pub const fn pool_id(&self) -> PoolId {
        self.pool_id
    }

    pub fn value_scope(&self) -> &str {
        &self.value_scope
    }

    /// Return an accepted commit event even if its notification was already consumed.
    pub(super) async fn committed_task_after(&self, published_watermark: u64) -> u64 {
        loop {
            let notified = self.committed_task.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let watermark = self.committed_event_watermark.load(Ordering::Acquire);
            if watermark > published_watermark {
                return watermark;
            }
            notified.await;
        }
    }

    pub(super) fn retain_graph_stage(
        &self,
        stage: Option<super::program_staging::StagedGraphProgram>,
    ) {
        *self
            ._graph_stage
            .lock()
            .expect("remote Graph stage owner poisoned") = stage;
    }

    pub fn cancel(self: &Arc<Self>, reason: CancellationReason) -> NativeExecutionResult<()> {
        let actions = self.driver.lock().expect("remote driver poisoned").handle(
            DriverCommand::CancelScope {
                scope_id: self.scope_id,
                reason,
                now_millis: now_millis(),
            },
        )?;
        self.dispatch(actions);
        Ok(())
    }

    /// Advances deadline, wall-time, and cancellation escalation state using wall-clock time.
    pub fn tick(self: &Arc<Self>) -> NativeExecutionResult<()> {
        let actions =
            self.driver
                .lock()
                .expect("remote driver poisoned")
                .handle(DriverCommand::Tick {
                    now_millis: now_millis(),
                })?;
        self.dispatch(actions);
        self.resolve_non_success_terminals();
        self.cancellation_settled.notify_waiters();
        Ok(())
    }

    pub(super) fn dispatch(self: &Arc<Self>, actions: Vec<DriverAction>) {
        for action in actions {
            match action {
                DriverAction::Launch(request) => {
                    let barrier = self
                        .uncheckpointed_tasks
                        .lock()
                        .expect("clock launch barrier poisoned");
                    if barrier.contains(&request.task_id) {
                        self.held_launches
                            .lock()
                            .expect("held launches poisoned")
                            .push(request);
                    } else {
                        drop(barrier);
                        self.launch(request);
                    }
                }
                DriverAction::Cancel(request) | DriverAction::Terminate(request) => {
                    self.cancel_attempt(request)
                }
                DriverAction::Checkpoint => {}
                DriverAction::GarbageCollectResults { objects, .. } => {
                    self.execution_objects.discard_results(&objects);
                }
                DriverAction::ResizePool { .. } => {
                    // Coarse allocation is owned by the Server/client adapter;
                    // workers are registered only after fenced routes exist.
                }
            }
        }
    }

    fn cancel_attempt(self: &Arc<Self>, request: AttemptRequest) {
        let reason = self
            .driver
            .lock()
            .expect("remote driver poisoned")
            .task_record(request.task_id)
            .map_err(|error| error.to_string())
            .and_then(|task| {
                task.cancellation
                    .ok_or_else(|| "scheduler stop has no original cancellation cause".to_string())
            })
            .and_then(|cancellation| {
                crate::process_provider::cancellation::reason(&request, cancellation.cause)
            });
        let reason = match reason {
            Ok(reason) => reason,
            Err(message) => {
                self.cancellation_ownership_failed
                    .store(true, Ordering::Release);
                self.apply_report(BackendReport::for_request(
                    &request,
                    AttemptReport::RetirementUnverified { message },
                ));
                return;
            }
        };
        // A remote Cancel acknowledgement proves delivery, not retirement of
        // the attempt's runtime, foreign host or private staging. Both Direct
        // and Graph attempts close through their eventual authenticated owner
        // report, never through this control acknowledgement.
        if let Some(source) = self
            .attempt_cancellation
            .lock()
            .expect("remote attempt cancellation registry poisoned")
            .get(&request.id)
            .cloned()
        {
            source.cancel(reason);
        }
        self.cancellation_in_flight.fetch_add(1, Ordering::AcqRel);
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let mut guard = CancellationDispatchGuard(Arc::clone(&this), false);
            let channel = this
                .channels
                .read()
                .await
                .get(&request.worker_id)
                .map(|route| Arc::clone(&route.channel));
            match channel {
                Some(channel) => {
                    if channel.cancel(&request, reason).await.is_err() {
                        if this.remove_worker(request.worker_id, true).await.is_err() {
                            this.apply_report(BackendReport::for_request(
                                &request,
                                AttemptReport::Lost {
                                    message: "remote worker cancellation acknowledgement was lost"
                                        .into(),
                                },
                            ));
                        } else {
                            guard.1 = true;
                        }
                        return;
                    }
                }
                None => {
                    if this.remove_worker(request.worker_id, true).await.is_err() {
                        this.apply_report(BackendReport::for_request(
                            &request,
                            AttemptReport::Lost {
                                message: "remote worker route disappeared during cancellation"
                                    .into(),
                            },
                        ));
                    } else {
                        guard.1 = true;
                    }
                    return;
                }
            }
            guard.1 = true;
        });
    }

    /// Wait for cancellation dispatch acknowledgements or route fencing, not
    /// physical retirement. The original scheduler evidence retains that proof.
    pub(crate) async fn wait_for_cancellation_control(&self) -> bool {
        loop {
            let notified = self.cancellation_settled.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.cancellation_in_flight.load(Ordering::Acquire) == 0 {
                return !self.cancellation_ownership_failed.load(Ordering::Acquire);
            }
            notified.await;
        }
    }

    /// Wait for control delivery and the scheduler's original cancellation
    /// dispositions. A control ACK alone cannot close an active attempt.
    pub(crate) async fn wait_for_cancellation_settlement(&self) -> bool {
        loop {
            let notified = self.cancellation_settled.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.cancellation_ownership_failed.load(Ordering::Acquire) {
                return false;
            }
            if self.cancellation_in_flight.load(Ordering::Acquire) == 0 {
                let pending = self
                    .driver
                    .lock()
                    .expect("remote driver poisoned")
                    .task_records()
                    .any(|task| {
                        task.cancellation.is_some()
                            && match task.state {
                                TaskState::Deferred
                                | TaskState::Ready
                                | TaskState::Assigned
                                | TaskState::Running
                                | TaskState::Committing => true,
                                TaskState::Succeeded
                                | TaskState::Failed
                                | TaskState::Cancelled
                                | TaskState::Indeterminate => false,
                            }
                    });
                if !pending {
                    return true;
                }
            }
            notified.await;
        }
    }
}

pub(super) fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_millis().try_into().unwrap_or(u64::MAX))
        .unwrap_or(0)
}
