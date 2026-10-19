use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use runmat_execution::{CancellationReason, Digest, ExecutionScopeId, PoolId, TaskId};
use runmat_execution_artifact::archive::{read_bundle, ArchiveLimits};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::{ArtifactResult, ProgramExecutionRequest, ProjectRevisionRecord};
use runmat_execution_runner::pool::ResizeRequest;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptReport, AttemptRequest, AttemptSuccess, Driver, DriverAction, DriverCommand,
    DriverConfig, PoolSpec, TaskSubmission,
};
use tokio::sync::{oneshot, Mutex as AsyncMutex, RwLock};

use super::pool_progress::RemoteTaskCompletion;
use super::RemoteWorkerChannel;
use crate::{NativeExecutionError, NativeExecutionResult};

type CompletionResult = Result<AttemptSuccess, String>;

mod admission;
mod attempt_execution;
mod completion;
mod input_inventory;
mod inspection;
mod membership;
mod object_registration;
mod recovery;
mod route_loss;

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
// ownership has not been proven quiescent and recovery must be indeterminate.
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
    scope_id: ExecutionScopeId,
    pool_id: PoolId,
    bundle_digest: Digest,
    bundle_identity: Digest,
    project_revision: ProjectRevisionRecord,
    bundle: Arc<[u8]>,
    driver: Mutex<Driver>,
    worker_registration: AsyncMutex<()>,
    channels: RwLock<HashMap<runmat_execution::identity::WorkerId, WorkerRoute>>,
    next_route_generation: AtomicU64,
    installed_nodes: AsyncMutex<BTreeSet<String>>,
    value_scope: String,
    values: super::pool_values::RemoteValueCatalog,
    execution_objects: super::pool_objects::RemoteObjectCatalog,
    programs: Mutex<HashMap<TaskId, ProgramExecutionRequest>>,
    input_inventories: Mutex<input_inventory::InputInventories>,
    completions: Mutex<HashMap<TaskId, oneshot::Sender<CompletionResult>>>,
    progress: Mutex<HashMap<TaskId, Arc<super::pool_progress::RemoteProgressBuffer>>>,
    admission: Mutex<()>,
    restored_completions: Mutex<HashMap<TaskId, RemoteTaskCompletion>>,
    committed_task: tokio::sync::Notify,
    committed_event_watermark: AtomicU64,
    cancellation_in_flight: AtomicUsize,
    cancellation_settled: tokio::sync::Notify,
    cancellation_ownership_failed: AtomicBool,
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
        let pool_id = pool.id;
        let bundle = bundle.into();
        let decoded = read_bundle(bundle.as_ref(), ArchiveLimits::default()).map_err(|error| {
            NativeExecutionError::Protocol(format!("remote pool bundle is invalid: {error}"))
        })?;
        let bundle_identity = decoded.identity().map_err(|error| {
            NativeExecutionError::Protocol(format!(
                "remote pool bundle identity is invalid: {error}"
            ))
        })?;
        let project_revision = decoded.manifest.project_revision;
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
            bundle_digest: Digest::sha256(bundle.as_ref()),
            bundle_identity,
            project_revision,
            bundle,
            driver: Mutex::new(driver),
            worker_registration: AsyncMutex::new(()),
            channels: RwLock::new(HashMap::new()),
            next_route_generation: AtomicU64::new(1),
            installed_nodes: AsyncMutex::new(BTreeSet::new()),
            value_scope: value_scope.into(),
            values: super::pool_values::RemoteValueCatalog::default(),
            execution_objects: super::pool_objects::RemoteObjectCatalog::new()?,
            programs: Mutex::new(HashMap::new()),
            input_inventories: Mutex::new(input_inventory::InputInventories::default()),
            completions: Mutex::new(HashMap::new()),
            progress: Mutex::new(HashMap::new()),
            admission: Mutex::new(()),
            restored_completions: Mutex::new(HashMap::new()),
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
        Ok(())
    }

    fn dispatch(self: &Arc<Self>, actions: Vec<DriverAction>) {
        for action in actions {
            match action {
                DriverAction::Launch(request) => self.launch(request),
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
                    if channel.cancel(&request).await.is_err() {
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
            this.apply_report(BackendReport::for_request(
                &request,
                AttemptReport::Cancelled,
            ));
            guard.1 = true;
        });
    }

    pub(crate) async fn wait_for_cancellation_quiescence(&self) -> bool {
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
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_millis().try_into().unwrap_or(u64::MAX))
        .unwrap_or(0)
}
