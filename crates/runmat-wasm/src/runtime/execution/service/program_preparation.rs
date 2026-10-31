//! Canonical browser program preparation, before taking the service borrow.
use super::*;
use runmat_execution::host::ExecutionHostRequirement;
use runmat_execution::resource::ResourceRequest;
use runmat_runtime::execution::GraphResultReceipt;

pub(crate) struct PreparedProgram {
    request: BrowserWorkerRequest,
    receipt: Option<Rc<dyn GraphResultReceipt>>,
    callable: runmat_execution::ProgramCallable,
    resources: ResourceRequest,
    host: ExecutionHostRequirement,
}
impl PreparedProgram {
    pub(in crate::runtime::execution) fn prepare(
        call: &DeferredCall,
        outputs: OutputContract,
        capabilities: BrowserExecutionCapabilities,
    ) -> Result<Self, ExecutionServiceError> {
        match &call.program {
            runmat_runtime::execution::DeferredProgram::SelectedGraph(_) => {
                Self::graph(call, outputs, capabilities)
            }
            _ => Self::direct(call, outputs, capabilities),
        }
    }

    pub(in crate::runtime::execution) fn admit_registered_worker(
        &self,
        service: &BrowserExecutionService,
    ) -> Result<(), ExecutionServiceError> {
        let state = service.state.borrow();
        let pool = state
            .driver
            .pool_record(service.pool_id)
            .map_err(driver_error)?;
        if !pool.workers.values().any(|worker| worker.accepts_work()) {
            return Err(ExecutionServiceError::WorkerLost(
                "browser pool has no admitted available worker owner".into(),
            ));
        }
        let empty = runmat_execution_runner::scheduler::ResourceAllocation::default();
        if pool
            .workers
            .values()
            .filter(|worker| worker.accepts_work())
            .any(|worker| {
                self.host.is_satisfied_by(&worker.spec.host).is_ok()
                    && runmat_execution_runner::scheduler::fits(
                        &worker.spec.resources,
                        &empty,
                        &self.resources,
                    )
            })
        {
            return Ok(());
        }
        Err(ExecutionServiceError::Infrastructure(
            "no registered browser worker satisfies the program's original host and resource requirements".into()))
    }

    fn direct(
        call: &DeferredCall,
        outputs: OutputContract,
        capabilities: BrowserExecutionCapabilities,
    ) -> Result<Self, ExecutionServiceError> {
        let (callable, context, recipe, artifact, arguments) =
            runmat_vm::materialize_deferred_call(
                call,
                outputs.clone(),
                runmat_execution_artifact::ProgramTarget::portable(
                    "wasm32-browser-interpreter-bytecode-v2",
                ),
            )?;
        let mut resources = browser_request(capabilities, recipe.accelerators.clone());
        let context_bytes =
            runmat_execution::console::invocation_memory_bytes(&call.environment, call.console)
                .map_err(|e| ExecutionServiceError::Failed(e.to_string()))?;
        resources.memory_bytes = resources
            .memory_bytes
            .checked_add(context_bytes)
            .ok_or_else(|| {
                ExecutionServiceError::Failed("invocation memory reservation overflow".into())
            })?;
        recipe.validate_resource_request(&resources).map_err(|e| {
            ExecutionServiceError::Failed(format!(
                "browser scheduler resources do not satisfy the program: {e}"
            ))
        })?;
        let host = artifact
            .execution_host_requirement(
                &recipe,
                BTreeSet::from([runmat_execution::security::ExecutionTrustTier::CustomerTrusted]),
            )
            .map_err(|e| ExecutionServiceError::Failed(e.to_string()))?;
        let request = ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            recipe,
            artifact,
            callable: callable.clone(),
            context,
            assignment: None,
            job_id: None,
            arguments,
            requested_outputs: outputs.requested_outputs,
            environment: call.environment.clone(),
            console: call.console,
        };
        Ok(Self {
            request: BrowserWorkerRequest::Direct(request),
            receipt: None,
            callable,
            resources,
            host,
        })
    }
    fn graph(
        call: &DeferredCall,
        outputs: OutputContract,
        capabilities: BrowserExecutionCapabilities,
    ) -> Result<Self, ExecutionServiceError> {
        let (callable, request, receipt) = runmat_vm::materialize_deferred_graph_call(
            call,
            outputs,
            runmat_execution_artifact::ProgramTarget::portable(
                "wasm32-browser-interpreter-bytecode-v2",
            ),
            runmat_execution::value::ValueLimits::default(),
        )?;
        let mut resources = browser_request(capabilities, request.recipe.accelerators.clone());
        let bytes = serde_json::to_vec(&request)
            .map_err(|e| ExecutionServiceError::Failed(e.to_string()))?;
        resources.memory_bytes = request
            .input_allocation_floor_bytes(bytes.len())
            .map_err(|e| ExecutionServiceError::Failed(e.to_string()))?;
        resources.scratch_bytes = u64::try_from(bytes.len())
            .map_err(|_| {
                ExecutionServiceError::Failed("browser callback wire length overflows".into())
            })?
            .max(1024 * 1024);
        let schema =
            runmat_execution::value::GraphRequiredSchema::from_version(request.schema_version)
                .ok_or_else(|| {
                    ExecutionServiceError::Failed("unsupported callback Graph schema".into())
                })?;
        if let Some(capability) = schema.required_capability() {
            resources.required_capabilities.insert(capability);
        }
        request
            .recipe
            .validate_resource_request(&resources)
            .map_err(|e| ExecutionServiceError::Failed(e.to_string()))?;
        let host = request
            .execution_host_requirement(BTreeSet::from([
                runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
            ]))
            .map_err(|e| ExecutionServiceError::Failed(e.to_string()))?;
        Ok(Self {
            request: BrowserWorkerRequest::Graph(request),
            receipt: Some(receipt),
            callable,
            resources,
            host,
        })
    }
    pub(in crate::runtime::execution) fn into_submission(
        self,
        id: TaskId,
        scope_id: ExecutionScopeId,
        pool_id: PoolId,
        outputs: OutputContract,
        retry: runmat_execution::RetryPolicy,
    ) -> (
        TaskSubmission,
        BrowserWorkerRequest,
        Option<Rc<dyn GraphResultReceipt>>,
    ) {
        let (artifact, context, inputs, environment, console) = match &self.request {
            BrowserWorkerRequest::Direct(request) => (
                request.artifact.id,
                request.context.clone(),
                request.arguments.clone(),
                request.environment.clone(),
                request.console,
            ),
            BrowserWorkerRequest::Graph(request) => (
                request.artifact.id,
                request.context.clone(),
                Vec::new(),
                request
                    .invocation_controls
                    .as_ref()
                    .map(|controls| controls.environment.clone())
                    .unwrap_or_default(),
                request
                    .invocation_controls
                    .as_ref()
                    .and_then(|controls| controls.console),
            ),
            BrowserWorkerRequest::OwnedBuiltin(_) => {
                unreachable!("program preparation never creates owned-builtin requests")
            }
        };
        let task = TaskSubmission {
            request: TaskRequest {
                id,
                scope_id,
                pool_id,
                schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
                executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                    ArtifactId::derive(&[artifact.0.bytes()]),
                ),
                callable: Callable::for_program("browser-session", &self.callable),
                invocation_context: context,
                inputs,
                environment,
                console,
                outputs,
                resources: self.resources,
                host: self.host,
                retry,
                deadline_unix_millis: None,
            },
            dependencies: BTreeSet::new(),
            priority: 0,
        };
        (task, self.request, self.receipt)
    }
}
