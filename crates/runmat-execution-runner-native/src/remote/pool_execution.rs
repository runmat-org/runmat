use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use runmat_execution::identity::PoolId;
use runmat_execution::identity::ValueId;
use runmat_execution::value::GraphRequiredSchema;
use runmat_execution::{CancellationReason, ExecutionScopeId};
use runmat_execution_artifact::encryption::RunKeyMaterial;
use runmat_execution_artifact::{
    GraphExecutionRequest, ProgramExecutionObject, ProgramExecutionObjectLocation,
    ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_execution_runner::PoolSpec;
use runmat_execution_transport_native::control::{
    DriverAuthority, DriverControlPlane, ResourceRequest as AllocationResources,
    WorkerRetirementLookup,
};

use super::driver::AdmittedRemoteProgram;
use super::pool_reconcile::{reconcile_workers, validate_pool_intent, WorkerAuthorizationContext};
use super::pool_resources::pool_inventory;
use super::pool_wait::{
    await_running_operation, await_startup_operation, RunningWait, StartupWait,
};
use super::RemotePoolDriver;
use crate::{
    NativeExecutionError, NativeExecutionResult, EXACT_MESHING_ROOT_EXECUTION_MODE,
    EXACT_STRUCTURAL_STUDY_ROOT_EXECUTION_MODE, EXACT_STRUCTURAL_SWEEP_ROOT_EXECUTION_MODE,
};

const WORKER_READY_TIMEOUT: Duration = Duration::from_secs(90);
const RECONCILE_INTERVAL: Duration = Duration::from_secs(1);
const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(2);
const DRIVER_TICK_INTERVAL: Duration = Duration::from_millis(250);
const CANCELLATION_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// In-process completion transfers its original values without a control heap
/// allocation. Only the authenticated root driver funds a retained publication.
pub(crate) type RemotePoolExecutionOutcome = Result<RemotePoolExecutionResult, RemoteExecutionStop>;

pub(crate) enum RemoteExecutionStop {
    Cancelled,
    Indeterminate(String),
}

pub(crate) struct RemotePoolExecutionResult {
    pub(crate) response: ProgramExecutionResponse,
    pub(crate) console: Option<RemoteTerminalConsole>,
    pub(crate) cancelled: bool,
    pub(super) objects: super::result_objects::RemoteResultObjects,
}

/// Scheduler retention remains the original evidence owner. Only encoding
/// borrows it; no complete transcript is cloned to publish the terminal result.
pub(crate) enum RemoteTerminalConsole {
    Owned(crate::console_bridge::NativeCapturedConsole),
    LocalSources {
        root: Option<runmat_execution::console::ConsoleSourceIdentity>,
        source: Arc<dyn runmat_runtime::context::RuntimeRetainedConsoleSource>,
        budget: runmat_execution::console::ConsoleBudget,
    },
    Task {
        pool: Arc<RemotePoolDriver>,
        task_id: runmat_execution::TaskId,
    },
}

impl RemoteTerminalConsole {
    pub(crate) fn encode(
        &self,
        outputs: &runmat_execution_artifact::ProgramExecutionOutputs,
    ) -> NativeExecutionResult<Vec<u8>> {
        match self {
            Self::LocalSources {
                root,
                source,
                budget,
            } => outputs
                .encode_with_console_source(
                    None,
                    &crate::console_bridge::RuntimeConsoleSourceAdapter(source.as_ref()),
                    Some(*budget),
                    *root,
                )
                .map_err(protocol),
            Self::Owned(console) => outputs
                .encode_with_console_source(
                    Some(&console.transcript),
                    &console.source(),
                    Some(console.budget),
                    None,
                )
                .map_err(protocol),
            Self::Task { pool, task_id } => pool.encode_terminal_console(*task_id, outputs),
        }
    }
}

impl RemotePoolExecutionResult {
    /// A response whose values are complete without driver-held result objects.
    pub(crate) fn without_objects(response: ProgramExecutionResponse) -> Self {
        Self {
            response,
            console: None,
            cancelled: false,
            objects: super::result_objects::RemoteResultObjects::None,
        }
    }

    #[cfg(all(test, feature = "occt-native"))]
    pub(crate) fn read_result_object(
        &self,
        reference: &runmat_execution::value::ValueRef,
        authorization_scope: &str,
    ) -> NativeExecutionResult<Vec<u8>> {
        self.objects.read(reference, authorization_scope)
    }
}

pub(super) struct RemotePoolExecution<'a> {
    pub(super) control: &'a dyn DriverControlPlane,
    pub(super) authority: &'a DriverAuthority,
    pub(super) run_key: &'a RunKeyMaterial,
    pub(super) program: AdmittedRemoteProgram,
    pub(super) input_objects: Vec<ProgramExecutionObject>,
    pub(super) recovered: Option<super::recovery::RestoredObjectClosure>,
    pub(super) recovered_pool: Option<super::pool_checkpoint::RestoredPoolCheckpoint>,
    pub(super) project_revision: String,
    pub(super) checkpoint_sequence: u64,
    pub(super) desired_workers: u32,
    pub(super) resources: AllocationResources,
    pub(super) driver_resources: AllocationResources,
    pub(super) cancellation: tokio::sync::watch::Receiver<bool>,
    pub(super) ready_v1: bool,
    pub(super) clock_parent: Option<super::config::DriverClockParent>,
}

/// The admitted root program: Direct V7 runs against the installed bundle;
/// a Graph V10 Program is staged to each worker from its signed transfer.
enum RootProgram {
    // Retain only request control state, not the decoded admission bundle.
    Direct(Box<ProgramExecutionRequest>),
    // The stage has transferred to the pool; remaining graph fields are needed.
    Graph(Box<super::driver::program::AdmittedGraphProgram>),
}

pub(super) struct GraphRoot {
    pub(super) request: GraphExecutionRequest,
    pub(super) encoded_bytes: usize,
    pub(super) transfer: Option<super::pool::SignedGraphTransfer>,
}

pub(super) async fn execute(
    execution: RemotePoolExecution<'_>,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let RemotePoolExecution {
        control,
        authority,
        run_key,
        program,
        input_objects,
        recovered,
        recovered_pool,
        project_revision,
        checkpoint_sequence,
        desired_workers,
        resources,
        driver_resources,
        mut cancellation,
        ready_v1,
        clock_parent,
    } = execution;
    let authorization = WorkerAuthorizationContext {
        authority,
        run_key,
        ready_v1,
    };
    let scope_id = ExecutionScopeId::derive(&[authority.run_id.as_bytes(), b"remote-root"]);
    let pool_id = PoolId::derive(&[authority.run_id.as_bytes(), b"remote-pool"]);
    let pool_inventory = pool_inventory(&resources, desired_workers)?;
    let pool_spec = PoolSpec {
        id: pool_id,
        min_workers: 1,
        max_workers: desired_workers,
        max_in_flight: desired_workers,
        resource_limit: pool_inventory,
    };
    let pool_catalog = recovered_pool
        .as_ref()
        .map(|restored| restored.catalog.clone());
    let pool_locations = recovered_pool
        .as_ref()
        .map(|restored| restored.locations.clone())
        .unwrap_or_default();
    let authoritative_unstarted = recovered_pool
        .as_ref()
        .map(|restored| restored.authoritative_unstarted_inventory())
        .transpose()?;
    let (pool, root) = match program {
        AdmittedRemoteProgram::Direct(admitted) => {
            let super::driver::program::AdmittedDirectProgram {
                bundle_archive,
                request,
                ..
            } = *admitted;
            let pool = match recovered_pool {
                Some(restored) => restored.into_pool(
                    super::RemotePoolRestoreAuthority {
                        scope_id,
                        pool: pool_spec,
                        value_scope: authority.run_id.clone(),
                        driver_fence: authority.fencing_token,
                    },
                    Arc::from(bundle_archive),
                )?,
                None => RemotePoolDriver::new_with_value_scope(
                    scope_id,
                    pool_spec,
                    authority.fencing_token,
                    bundle_archive,
                    authority.run_id.clone(),
                )?,
            };
            // This ordinary transport-control Box moves the request without
            // copying Value payloads or retaining redundant decoded objects.
            (pool, RootProgram::Direct(Box::new(request)))
        }
        AdmittedRemoteProgram::Graph(mut admitted) => {
            if recovered_pool.is_some() || !input_objects.is_empty() {
                return Err(protocol(
                    "graph programs carry neither pool checkpoints nor Direct input objects",
                ));
            }
            let pool = RemotePoolDriver::new_graph(
                scope_id,
                pool_spec,
                authority.fencing_token,
                authority.run_id.clone(),
            )?;
            pool.retain_graph_stage(admitted._staged.take());
            (pool, RootProgram::Graph(admitted))
        }
    };
    pool.configure_console_retention(&driver_resources)?;
    for object in input_objects {
        if object.is_externalized_value() {
            pool.register_value_object(object.reference, object.bytes)?;
        } else {
            pool.register_execution_object(object.reference, object.bytes)?;
        }
    }
    let deadline = tokio::time::Instant::now() + WORKER_READY_TIMEOUT;
    let mut driver_ticks = tokio::time::interval(DRIVER_TICK_INTERVAL);
    driver_ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    driver_ticks.tick().await;
    // A replacement driver must retire every uncertain original allocation
    // before requesting another physical worker at the finite pool limit.
    // Neither absence in list_workers nor a release/resize ACK is evidence.
    for (worker_id, owner) in pool.unresolved_original_owners()? {
        let original = WorkerRetirementLookup {
            run_id: authority.run_id.clone(),
            allocation_lease_id: owner.allocation_lease_id.clone(),
            allocation_fencing_token: owner.allocation_fencing_token,
            driver_fencing_token: owner.driver_fencing_token,
        };
        loop {
            match await_startup_operation(
                control.worker_retirement(authority, &original),
                &mut driver_ticks,
                &pool,
                &mut cancellation,
                deadline,
            )
            .await?
            {
                StartupWait::Completed(Ok(Some(evidence)), cancelled) => {
                    pool.accept_node_tree_retirement(
                        authority,
                        worker_id,
                        &owner,
                        &evidence,
                        run_key,
                        authoritative_unstarted.as_ref(),
                    )?;
                    if cancelled {
                        return Ok(Err(
                            crate::remote::pool_execution::RemoteExecutionStop::Cancelled,
                        ));
                    }
                    break;
                }
                StartupWait::Completed(Ok(None), cancelled) => {
                    if cancelled {
                        return Ok(Err(
                            crate::remote::pool_execution::RemoteExecutionStop::Cancelled,
                        ));
                    }
                    match await_startup_operation(
                        tokio::time::sleep(Duration::from_millis(250)),
                        &mut driver_ticks,
                        &pool,
                        &mut cancellation,
                        deadline,
                    )
                    .await?
                    {
                        StartupWait::Completed((), false) => {}
                        StartupWait::Completed((), true) | StartupWait::Deadline(true) => {
                            return Ok(Err(
                                crate::remote::pool_execution::RemoteExecutionStop::Cancelled,
                            ))
                        }
                        StartupWait::Deadline(false) => {
                            return Err(protocol(
                                "original worker tree retirement remained unverified",
                            ))
                        }
                    }
                }
                StartupWait::Completed(Err(error), _) => return Err(protocol(error)),
                StartupWait::Deadline(true) => {
                    return Ok(Err(
                        crate::remote::pool_execution::RemoteExecutionStop::Cancelled,
                    ))
                }
                StartupWait::Deadline(false) => {
                    return Err(protocol(
                        "original worker tree retirement evidence timed out",
                    ))
                }
            }
        }
    }
    let worker_pool = match await_startup_operation(
        control.resize_workers(authority, 0, desired_workers, resources.clone()),
        &mut driver_ticks,
        &pool,
        &mut cancellation,
        deadline,
    )
    .await?
    {
        StartupWait::Completed(Ok(worker_pool), false) => worker_pool,
        StartupWait::Completed(Ok(worker_pool), true) => {
            return super::pool_allocation::finalize(
                control,
                authority,
                &worker_pool.resources,
                worker_pool.generation,
                Ok(Err(
                    crate::remote::pool_execution::RemoteExecutionStop::Cancelled,
                )),
            )
            .await;
        }
        StartupWait::Completed(Err(primary), cancelled) => {
            return super::pool_allocation::resolve_failed_initial(
                control,
                authority,
                desired_workers,
                &resources,
                primary.to_string(),
                cancelled,
            )
            .await;
        }
        StartupWait::Deadline(cancelled) => {
            return super::pool_allocation::resolve_failed_initial(
                control,
                authority,
                desired_workers,
                &resources,
                "initial worker allocation exceeded its deadline".into(),
                cancelled,
            )
            .await;
        }
    };
    let generation = Arc::new(std::sync::atomic::AtomicU64::new(worker_pool.generation));
    let outcome = match validate_pool_intent(&worker_pool, desired_workers, &resources) {
        Ok(()) => {
            execute_with_allocated_workers(AllocatedExecution {
                control,
                authorization: &authorization,
                pool,
                pool_id,
                root,
                recovered,
                pool_catalog,
                pool_locations,
                project_revision,
                checkpoint_sequence,
                desired_workers,
                resources: resources.clone(),
                cancellation,
                worker_pool,
                generation: Arc::clone(&generation),
                deadline,
                driver_ticks,
                clock_parent,
            })
            .await
        }
        Err(error) => Err(error),
    };
    super::pool_allocation::finalize(
        control,
        authority,
        &resources,
        generation.load(std::sync::atomic::Ordering::Acquire),
        outcome,
    )
    .await
}

struct AllocatedExecution<'a> {
    control: &'a dyn DriverControlPlane,
    authorization: &'a WorkerAuthorizationContext<'a>,
    pool: Arc<RemotePoolDriver>,
    pool_id: PoolId,
    root: RootProgram,
    recovered: Option<super::recovery::RestoredObjectClosure>,
    pool_catalog: Option<ProgramExecutionObjectLocation>,
    pool_locations: BTreeMap<ValueId, ProgramExecutionObjectLocation>,
    project_revision: String,
    checkpoint_sequence: u64,
    desired_workers: u32,
    resources: AllocationResources,
    cancellation: tokio::sync::watch::Receiver<bool>,
    worker_pool: runmat_execution_transport_native::control::DriverWorkerPool,
    generation: Arc<std::sync::atomic::AtomicU64>,
    deadline: tokio::time::Instant,
    driver_ticks: tokio::time::Interval,
    clock_parent: Option<super::config::DriverClockParent>,
}

async fn execute_with_allocated_workers(
    execution: AllocatedExecution<'_>,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let AllocatedExecution {
        control,
        authorization,
        pool,
        pool_id,
        root,
        recovered,
        pool_catalog,
        pool_locations,
        project_revision,
        checkpoint_sequence,
        desired_workers,
        resources,
        mut cancellation,
        mut worker_pool,
        generation,
        deadline,
        mut driver_ticks,
        clock_parent,
    } = execution;
    let authority = authorization.authority;
    let run_key = authorization.run_key;
    let mut registered = HashMap::new();
    while registered.len() < desired_workers as usize {
        if *cancellation.borrow() {
            pool.cancel(CancellationReason::User)?;
            return Ok(Err(
                crate::remote::pool_execution::RemoteExecutionStop::Cancelled,
            ));
        }
        match await_startup_operation(
            reconcile_workers(
                control,
                authorization,
                pool_id,
                &pool,
                &worker_pool,
                &mut registered,
            ),
            &mut driver_ticks,
            &pool,
            &mut cancellation,
            deadline,
        )
        .await?
        {
            StartupWait::Completed(result, cancelled) => {
                result?;
                if cancelled {
                    return Ok(Err(
                        crate::remote::pool_execution::RemoteExecutionStop::Cancelled,
                    ));
                }
            }
            StartupWait::Deadline(cancelled) => {
                if cancelled {
                    return Ok(Err(
                        crate::remote::pool_execution::RemoteExecutionStop::Cancelled,
                    ));
                }
                return Err(protocol(
                    "remote workers did not become ready before the deadline",
                ));
            }
        }
        if registered.len() == desired_workers as usize {
            break;
        }
        match await_startup_operation(
            tokio::time::sleep(Duration::from_millis(250)),
            &mut driver_ticks,
            &pool,
            &mut cancellation,
            deadline,
        )
        .await?
        {
            StartupWait::Completed((), true) | StartupWait::Deadline(true) => {
                return Ok(Err(
                    crate::remote::pool_execution::RemoteExecutionStop::Cancelled,
                ));
            }
            StartupWait::Completed((), false) => {}
            StartupWait::Deadline(false) => {
                return Err(protocol(
                    "remote workers did not become ready before the deadline",
                ));
            }
        }
        match await_startup_operation(
            control.resize_workers(
                authority,
                worker_pool
                    .generation
                    .checked_sub(1)
                    .ok_or_else(|| protocol("Server returned an invalid worker pool generation"))?,
                desired_workers,
                resources.clone(),
            ),
            &mut driver_ticks,
            &pool,
            &mut cancellation,
            deadline,
        )
        .await?
        {
            StartupWait::Completed(result, cancelled) => {
                worker_pool = result.map_err(protocol)?;
                generation.store(worker_pool.generation, std::sync::atomic::Ordering::Release);
                validate_pool_intent(&worker_pool, desired_workers, &resources)?;
                if cancelled {
                    return Ok(Err(
                        crate::remote::pool_execution::RemoteExecutionStop::Cancelled,
                    ));
                }
            }
            StartupWait::Deadline(true) => {
                return Ok(Err(
                    crate::remote::pool_execution::RemoteExecutionStop::Cancelled,
                ));
            }
            StartupWait::Deadline(false) => {
                return Err(protocol(
                    "remote workers did not become ready before the deadline",
                ));
            }
        }
    }

    let run_id = authority.run_id.clone();
    let request = match root {
        RootProgram::Direct(request) => *request,
        RootProgram::Graph(graph) => {
            // V10 work proceeds as soon as workers register. Newer graph
            // requests wait for an encrypted, binary-issued capability proof.
            let required = GraphRequiredSchema::from_version(graph.request.schema_version)
                .ok_or_else(|| protocol("remote Graph request has an unsupported schema"))?
                .required_capability();
            if let Some(required) = &required {
                let supported = await_graph_capability(GraphCapabilityWait {
                    control,
                    authorization,
                    pool: &pool,
                    pool_id,
                    required,
                    desired_workers,
                    resources: &resources,
                    cancellation: &mut cancellation,
                    worker_pool: &mut worker_pool,
                    generation: generation.as_ref(),
                    registered: &mut registered,
                    deadline,
                })
                .await?;
                if !supported {
                    pool.cancel(CancellationReason::User)?;
                    return Ok(Err(
                        crate::remote::pool_execution::RemoteExecutionStop::Cancelled,
                    ));
                }
            }
            return run_graph_root(
                &pool,
                super::pool_tasks::execute_graph_root(
                    Arc::clone(&pool),
                    GraphRoot {
                        request: graph.request,
                        encoded_bytes: graph.encoded_bytes,
                        transfer: graph.transfer,
                    },
                    required,
                    run_id,
                    resources.clone(),
                ),
                &mut cancellation,
                &mut driver_ticks,
            )
            .await;
        }
    };
    let exact_meshing = request.recipe.execution_mode == EXACT_MESHING_ROOT_EXECUTION_MODE;
    let geometry_meshing =
        request.recipe.execution_mode == crate::GEOMETRY_MESHING_ROOT_EXECUTION_MODE;
    let exact_structural_study =
        request.recipe.execution_mode == EXACT_STRUCTURAL_STUDY_ROOT_EXECUTION_MODE;
    let structural_study =
        request.recipe.execution_mode == crate::STRUCTURAL_STUDY_ROOT_EXECUTION_MODE;
    let exact_structural_sweep =
        request.recipe.execution_mode == EXACT_STRUCTURAL_SWEEP_ROOT_EXECUTION_MODE;
    let pool_for_execution = Arc::clone(&pool);
    let resources_for_execution = resources.clone();
    // Scheduler recovery belongs to every replay-safe pool root. Study roots
    // additionally publish physical state through the same transaction owner.
    let mut recovery_publisher = matches!(
        super::checkpoint::execution_start_phase(&request),
        super::checkpoint::RemoteDriverCheckpointPhase::ReplayableExecutionStarted
    )
    .then(|| {
        super::recovery::RemoteRecoveryPublisher::new(
            control,
            run_key,
            authority,
            &project_revision,
            &request,
            checkpoint_sequence,
            recovered.as_ref(),
        )
    })
    .transpose()?;
    if let Some(publisher) = recovery_publisher.as_mut() {
        publisher.bind_pool(Arc::clone(&pool), pool_catalog, pool_locations)?;
    }
    // The study owns physical publications; this clone owns periodic scheduler
    // publications. Both serialize through the same acknowledged sequence.
    let periodic_publisher = recovery_publisher.clone();
    let (clock_sender, mut clock_receiver) = tokio::sync::mpsc::channel(8);
    // Effects-unknown roots have no resume-safe pool publisher and cannot
    // claim recovered wall time. Preserve their non-replayable execution path;
    // every replayable root installs the observed checkpoint barrier.
    if clock_parent.is_some() && periodic_publisher.is_some() {
        pool.install_clock_commands(clock_sender)?;
    }
    let execution = if geometry_meshing {
        Box::pin(super::exact_meshing::execute_geometry_root(
            pool_for_execution,
            request,
            run_id,
            resources_for_execution,
        )) as std::pin::Pin<Box<dyn std::future::Future<Output = _> + '_>>
    } else if exact_meshing {
        Box::pin(super::exact_meshing::execute(
            pool_for_execution,
            request,
            run_id,
            resources_for_execution,
        )) as std::pin::Pin<Box<dyn std::future::Future<Output = _> + '_>>
    } else if structural_study {
        Box::pin(super::structural_study::execute(
            pool_for_execution,
            request,
            run_id,
            resources_for_execution,
            recovered.as_ref(),
            recovery_publisher
                .as_mut()
                .ok_or_else(|| protocol("structural study recovery publisher is absent"))?,
        )) as std::pin::Pin<Box<dyn std::future::Future<Output = _> + '_>>
    } else if exact_structural_study {
        Box::pin(super::exact_structural_study::execute(
            pool_for_execution,
            request,
            run_id,
            resources_for_execution,
            recovered.as_ref(),
            recovery_publisher
                .as_mut()
                .ok_or_else(|| protocol("structural study recovery publisher is absent"))?,
        )) as std::pin::Pin<Box<dyn std::future::Future<Output = _> + '_>>
    } else if exact_structural_sweep {
        Box::pin(super::exact_structural_sweep::execute(
            pool_for_execution,
            request,
            run_id,
            resources_for_execution,
        )) as std::pin::Pin<Box<dyn std::future::Future<Output = _> + '_>>
    } else {
        Box::pin(super::pool_tasks::execute_single_root(
            pool_for_execution,
            request,
            run_id,
            resources_for_execution,
        ))
    };
    let terminal_publisher = periodic_publisher.clone();
    let terminal_pool = Arc::clone(&pool);
    let mut execution = Box::pin(async move {
        let outcome = execution.await;
        // A task waiter can resolve as soon as cancellation is requested, while
        // the worker is still executing. The final checkpoint must include the
        // original scheduler disposition, not just the control delivery ACK.
        let cancellation_settled = tokio::time::timeout(
            WORKER_READY_TIMEOUT,
            terminal_pool.wait_for_cancellation_settlement(),
        )
        .await
        .unwrap_or(false);
        if !cancellation_settled {
            return Ok(Err(
                crate::remote::pool_execution::RemoteExecutionStop::Indeterminate(
                    "remote cancellation settlement did not complete before its deadline".into(),
                ),
            ));
        }
        // Persist the final scheduler state before exposing any terminal outcome,
        // including when the last task and packaging finish in one executor turn.
        if let Some(publisher) = terminal_publisher {
            tokio::time::timeout(WORKER_READY_TIMEOUT, publisher.publish_pool())
                .await
                .map_err(|_| protocol("terminal recovery publication exceeded its deadline"))??;
        }
        outcome
    }) as std::pin::Pin<Box<dyn std::future::Future<Output = _> + '_>>;
    let mut reconcile_ticks = tokio::time::interval(RECONCILE_INTERVAL);
    reconcile_ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut checkpoint_ticks = tokio::time::interval(CHECKPOINT_INTERVAL);
    checkpoint_ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    driver_ticks.tick().await;
    reconcile_ticks.tick().await;
    checkpoint_ticks.tick().await;
    let mut cancellation_requested = false;
    let mut cancellation_open = true;
    let mut checkpointed_event_watermark = 0;
    loop {
        if *cancellation.borrow() && !cancellation_requested {
            pool.cancel(CancellationReason::User)?;
            cancellation_requested = true;
        }
        let observed_watermark = checkpointed_event_watermark;
        let checkpoint_due = async {
            tokio::select! {
                _ = checkpoint_ticks.tick() => None,
                watermark = pool.committed_task_after(observed_watermark) => Some(watermark),
            }
        };
        tokio::select! {
            command = clock_receiver.recv(), if clock_parent.is_some() => {
                if let Some(command) = command {
                    match command {
                        super::clock_observer::ClockCommand::Begin(reply) => {
                            let result = clock_parent.as_ref().expect("clock branch is enabled")
                                .begin(authority, control).await;
                            let _ = reply.send(result);
                        }
                        super::clock_observer::ClockCommand::Checkpoint(reply) => {
                            let result = match periodic_publisher.as_ref() {
                                Some(publisher) => publisher.publish_pool().await.map(|_| ()),
                                None => Err(protocol("wave clock requires a recovery publisher")),
                            };
                            let _ = reply.send(result);
                        }
                        super::clock_observer::ClockCommand::Close(reply) => {
                            let result = match periodic_publisher.as_ref() {
                                Some(publisher) => publisher.publish_closed_wave().await,
                                None => Err(protocol("closed wave requires a recovery publisher")),
                            };
                            let _ = reply.send(result);
                        }
                        super::clock_observer::ClockCommand::End { start, respond } => {
                            let result = super::clock_observer::observe_end(control, authority, &start).await;
                            let _ = respond.send(result);
                        }
                    }
                }
            }
            result = &mut execution => {
                return if cancellation_requested {
                    super::pool_tasks::observed_execution(&pool, result, true)
                } else {
                    result
                };
            },
            changed = cancellation.changed(), if cancellation_open => {
                if changed.is_err() {
                    cancellation_open = false;
                } else if *cancellation.borrow() && !cancellation_requested {
                    pool.cancel(CancellationReason::User)?;
                    cancellation_requested = true;
                }
            }
            _ = driver_ticks.tick() => {
                pool.tick()?;
            }
            requested_watermark = checkpoint_due, if periodic_publisher.is_some() && !cancellation_requested => {
                let publisher = periodic_publisher.as_ref().expect("checkpoint branch is enabled");
                match await_running_operation(
                    publisher.publish_pool_through(requested_watermark.unwrap_or(0)),
                    &mut execution,
                    &mut driver_ticks,
                    &pool,
                    &mut cancellation,
                    &mut cancellation_requested,
                    tokio::time::Instant::now() + WORKER_READY_TIMEOUT,
                ).await? {
                    RunningWait::Execution(result) => {
                        return super::pool_tasks::observed_execution(&pool, result, cancellation_requested);
                    }
                    RunningWait::Operation(result, completed) => {
                        checkpointed_event_watermark = result?;
                        if let Some(result) = completed {
                            return super::pool_tasks::observed_execution(&pool, result, cancellation_requested);
                        }
                    }
                    RunningWait::Deadline(completed) => {
                        if let Some(result) = completed {
                            return super::pool_tasks::observed_execution(&pool, result, cancellation_requested);
                        }
                        if cancellation_requested {
                            return Ok(Err(crate::remote::pool_execution::RemoteExecutionStop::Indeterminate(
                                "remote cancellation was not quiescent before recovery publication timed out".into(),
                            )));
                        }
                        return Err(protocol("remote recovery publication exceeded its deadline"));
                    }
                }
            }
            _ = reconcile_ticks.tick(), if !cancellation_requested => {
                let resize = async {
                    control.resize_workers(
                        authority,
                        worker_pool.generation.checked_sub(1).ok_or_else(|| {
                            protocol("Server returned an invalid worker pool generation")
                        })?,
                        desired_workers,
                        resources.clone(),
                    )
                    .await
                    .map_err(protocol)
                };
                let (resized, completed) = match await_running_operation(
                    resize,
                    &mut execution,
                    &mut driver_ticks,
                    &pool,
                    &mut cancellation,
                    &mut cancellation_requested,
                    tokio::time::Instant::now() + WORKER_READY_TIMEOUT,
                ).await? {
                    RunningWait::Execution(result) => {
                        return super::pool_tasks::observed_execution(&pool, result, cancellation_requested);
                    }
                    RunningWait::Operation(resized, completed) => match resized {
                        Ok(resized) => (resized, completed),
                        Err(error) => {
                            if let Err(outcome) = super::pool_allocation::reconcile_or_indeterminate(
                                control,
                                authority,
                                desired_workers,
                                &resources,
                                generation.as_ref(),
                                format!("remote worker control failed: {error}"),
                            ).await {
                                return Ok(Err(outcome));
                            }
                            return Err(error);
                        }
                    },
                    RunningWait::Deadline(completed) => {
                        if let Err(outcome) = super::pool_allocation::reconcile_or_indeterminate(
                            control,
                            authority,
                            desired_workers,
                            &resources,
                            generation.as_ref(),
                            "remote worker control operation exceeded its deadline",
                        ).await {
                            return Ok(Err(outcome));
                        }
                        if let Some(result) = completed {
                            return super::pool_tasks::observed_execution(&pool, result, cancellation_requested);
                        }
                        if cancellation_requested {
                            return Ok(Err(crate::remote::pool_execution::RemoteExecutionStop::Indeterminate(
                                "remote cancellation was not quiescent before worker control timed out".into(),
                            )));
                        }
                        return Err(protocol("remote worker control operation exceeded its deadline"));
                    }
                };
                worker_pool = resized;
                generation.store(worker_pool.generation, std::sync::atomic::Ordering::Release);
                validate_pool_intent(&worker_pool, desired_workers, &resources)?;
                if let Some(result) = completed {
                    return super::pool_tasks::observed_execution(&pool, result, cancellation_requested);
                }
                if cancellation_requested {
                    continue;
                }
                let reconcile = reconcile_workers(
                        control,
                        authorization,
                        pool_id,
                        &pool,
                        &worker_pool,
                        &mut registered,
                    );
                match await_running_operation(
                    reconcile,
                    &mut execution,
                    &mut driver_ticks,
                    &pool,
                    &mut cancellation,
                    &mut cancellation_requested,
                    tokio::time::Instant::now() + WORKER_READY_TIMEOUT,
                ).await? {
                    RunningWait::Execution(result) => {
                        return super::pool_tasks::observed_execution(&pool, result, cancellation_requested);
                    }
                    RunningWait::Operation(result, completed) => {
                        result?;
                        if let Some(result) = completed {
                            return super::pool_tasks::observed_execution(&pool, result, cancellation_requested);
                        }
                    }
                    RunningWait::Deadline(completed) => {
                        if let Err(outcome) = super::pool_allocation::reconcile_or_indeterminate(
                            control,
                            authority,
                            desired_workers,
                            &resources,
                            generation.as_ref(),
                            "remote worker reconciliation exceeded its deadline",
                        ).await {
                            return Ok(Err(outcome));
                        }
                        if let Some(result) = completed {
                            return super::pool_tasks::observed_execution(&pool, result, cancellation_requested);
                        }
                        if cancellation_requested {
                            return Ok(Err(crate::remote::pool_execution::RemoteExecutionStop::Indeterminate(
                                "remote cancellation control cleanup remained incomplete when worker reconciliation timed out".into(),
                            )));
                        }
                        return Err(protocol("remote worker reconciliation exceeded its deadline"));
                    }
                }
            }
        }
    }
}

struct GraphCapabilityWait<'a, 'b> {
    control: &'a dyn DriverControlPlane,
    authorization: &'a WorkerAuthorizationContext<'a>,
    pool: &'b Arc<RemotePoolDriver>,
    pool_id: PoolId,
    required: &'b runmat_execution::resource::Capability,
    desired_workers: u32,
    resources: &'b AllocationResources,
    cancellation: &'b mut tokio::sync::watch::Receiver<bool>,
    worker_pool: &'b mut runmat_execution_transport_native::control::DriverWorkerPool,
    generation: &'b std::sync::atomic::AtomicU64,
    registered: &'b mut HashMap<String, runmat_execution::identity::WorkerId>,
    deadline: tokio::time::Instant,
}

/// Returns false when cancellation arrives before an authenticated worker
/// advertises the selected Graph schema.
async fn await_graph_capability(wait: GraphCapabilityWait<'_, '_>) -> NativeExecutionResult<bool> {
    let GraphCapabilityWait {
        control,
        authorization,
        pool,
        pool_id,
        required,
        desired_workers,
        resources,
        cancellation,
        worker_pool,
        generation,
        registered,
        deadline,
    } = wait;
    loop {
        pool.refresh_worker_software_capabilities().await?;
        let supported = pool
            .snapshot()
            .pools
            .get(&pool_id)
            .is_some_and(|pool| pool.spec.resource_limit.capabilities.contains(required));
        if supported {
            return Ok(true);
        }
        if *cancellation.borrow() {
            return Ok(false);
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(protocol(
                "no authenticated remote worker advertised the selected Graph schema",
            ));
        }
        tokio::select! {
            changed = cancellation.changed() => {
                if changed.is_ok() && *cancellation.borrow() {
                    return Ok(false);
                }
            }
            _ = tokio::time::sleep(RECONCILE_INTERVAL) => {}
        }
        *worker_pool = control
            .resize_workers(
                authorization.authority,
                worker_pool
                    .generation
                    .checked_sub(1)
                    .ok_or_else(|| protocol("Server returned an invalid worker pool generation"))?,
                desired_workers,
                resources.clone(),
            )
            .await
            .map_err(protocol)?;
        generation.store(worker_pool.generation, std::sync::atomic::Ordering::Release);
        validate_pool_intent(worker_pool, desired_workers, resources)?;
        reconcile_workers(
            control,
            authorization,
            pool_id,
            pool,
            worker_pool,
            registered,
        )
        .await?;
    }
}

/// A Graph root keeps scheduler time moving and, on cancellation, waits a
/// bounded interval for the worker's retirement report before resolving.
async fn run_graph_root(
    pool: &Arc<RemotePoolDriver>,
    execution: impl std::future::Future<Output = NativeExecutionResult<RemotePoolExecutionOutcome>>,
    cancellation: &mut tokio::sync::watch::Receiver<bool>,
    driver_ticks: &mut tokio::time::Interval,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    tokio::pin!(execution);
    let mut cancellation_open = true;
    loop {
        if *cancellation.borrow() {
            pool.cancel(CancellationReason::User)?;
            let drained = tokio::time::timeout(CANCELLATION_DRAIN_TIMEOUT, &mut execution).await;
            return Ok(match drained {
                Ok(result) => super::pool_tasks::observed_execution(pool, result, true)?,
                Err(_) => Err(crate::remote::pool_execution::RemoteExecutionStop::Indeterminate(
                    "remote Graph cancellation did not prove worker retirement before the drain deadline"
                        .into(),
                )),
            });
        }
        tokio::select! {
            result = &mut execution => return result,
            changed = cancellation.changed(), if cancellation_open => {
                if changed.is_err() {
                    cancellation_open = false;
                }
            }
            _ = driver_ticks.tick() => {
                pool.tick()?;
            }
        }
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
