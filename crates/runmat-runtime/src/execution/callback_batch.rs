//! Bounded, ordered numerical callback waves under the original authority.
use crate::call::descriptor::CallableDescriptor;
use crate::context::RuntimeContext;
use crate::execution::{
    AwaitAction, DeferredCall, DeferredGraphProgram, DeferredInvocation, DeferredProgram,
    ExecutionServiceError, RootedValue, RootedValueLease, RootedValueSequence,
};
use crate::{GpuGatherRetry, RuntimeError};
use runmat_execution::{
    CancellationReason, PoolBackend, PoolHandle, PoolRequest, RetryPolicy, TaskHandle,
};
use runmat_value::Value;
mod pending;
mod selection;
#[cfg(test)]
mod tests;
use pending::PreparedWave;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum CallbackParallelMode {
    #[default]
    Off,
    Auto,
    On,
}

/// Exact selected code plus its actual source root. This is not a registry or
/// scheduler; the existing selected program issuer supplies its descriptor.
/// Builtin program/task metadata currently requires unbounded host workspace;
/// this does not change finite admission for exact bound Graph callables.
pub struct SelectedWorkerCallable {
    descriptor: CallableDescriptor,
    capabilities: runmat_types::CapabilitySet,
    spawn_safety: runmat_types::SpawnSafetyFact,
    _source: crate::call::descriptor::RepeatableCallable,
    direct_builtin: Option<std::sync::Arc<[u8]>>,
}
impl SelectedWorkerCallable {
    pub fn new(
        descriptor: CallableDescriptor,
        capabilities: runmat_types::CapabilitySet,
        spawn_safety: runmat_types::SpawnSafetyFact,
        source: crate::call::descriptor::RepeatableCallable,
        direct_builtin: Option<std::sync::Arc<[u8]>>,
    ) -> Self {
        Self {
            descriptor,
            capabilities,
            spawn_safety,
            _source: source,
            direct_builtin,
        }
    }
    fn deferred_program(
        &self,
        invocation: &DeferredInvocation,
        selected: &DeferredGraphProgram,
        _runtime: &RuntimeContext,
    ) -> Result<DeferredProgram, RuntimeError> {
        if let Some(program) = &self.direct_builtin {
            if crate::execution::value_codec::deferred_invocation_requires_graph(invocation) {
                // The selected snapshot already owns builtin Graph roots and
                // authenticates callable arguments against their issued V3
                // products. V2's numeric codec is not authority to reject or
                // flatten that existing Graph route.
                return Ok(DeferredProgram::SelectedGraph(selected.clone()));
            }
            return Ok(DeferredProgram::InterpreterV2(program.clone()));
        }
        Ok(DeferredProgram::SelectedGraph(selected.clone()))
    }
    fn invocation(
        &self,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: &RuntimeContext,
    ) -> Result<DeferredInvocation, RuntimeError> {
        if let Some(name) = self.descriptor.target.selected_builtin_name() {
            // Keep the selected source backing until this actual task's wire
            // metadata is published. A name-byte quote alone would not bound
            // the complete program/task metadata path.
            crate::context::host_allocation::require_unbounded_workspace(
                Some(runtime),
                "parallel builtin callback",
                "selected builtin program and task metadata have no complete finite workspace bound",
            )?;
            return Ok(DeferredInvocation::Program {
                callable: runmat_execution::ProgramCallable::Builtin {
                    name: runmat_types::BuiltinId(name.to_owned()),
                },
                context: runmat_execution::ProgramInvocationContext::Direct,
                arguments,
                requested_outputs,
            });
        }
        let mut descriptor = self.descriptor.clone();
        descriptor.args = arguments;
        descriptor.requested_outputs = requested_outputs;
        Ok(DeferredInvocation::callable(descriptor))
    }
}

fn unavailable(runtime: Option<&RuntimeContext>, message: &str) -> RuntimeError {
    let mut error = if let Some(runtime) = runtime {
        let _entry = runtime.enter();
        crate::runtime_error::semantic_error(
            "RunMat:Execution:ParallelCallbackUnavailable",
            message,
        )
    } else {
        crate::build_runtime_error(message)
            .with_identifier("RunMat:Execution:ParallelCallbackUnavailable")
            .build()
    };
    error.gpu_gather_retry = GpuGatherRetry::Never;
    error
}

fn execution_error(runtime: &RuntimeContext, error: ExecutionServiceError) -> RuntimeError {
    let _entry = runtime.enter();
    let original_callback = matches!(
        &error,
        ExecutionServiceError::LocalizedRuntimeFailure(_)
            | ExecutionServiceError::RuntimeFailure(_)
    );
    let mut error = error.into_runtime_error_in_context(Some(runtime));
    if !original_callback {
        error.gpu_gather_retry = GpuGatherRetry::Never;
    }
    error
}

pub(crate) struct CallbackBatch {
    runtime: RuntimeContext,
    selected: SelectedWorkerCallable,
    program: DeferredGraphProgram,
    pool: PoolHandle,
    width: usize,
}

impl CallbackBatch {
    pub fn width(&self) -> usize {
        self.width
    }

    /// One worker-width wave. Submission remains lazy until the returned
    /// future is polled; numerical callers admit their whole sweep first.
    pub fn execute<'a>(
        &'a self,
        arguments: Vec<Vec<Value>>,
        requested_outputs: usize,
        cooperation: &crate::execution::CpuCooperationBudget,
    ) -> Result<
        impl std::future::Future<Output = Result<Vec<RootedValueSequence>, RuntimeError>> + 'a,
        RuntimeError,
    > {
        // Keep the existing wave-width diagnostic ahead of token admission.
        if arguments.len() > self.width {
            return Err(unavailable(
                Some(&self.runtime),
                "callback wave exceeds selected worker count",
            ));
        }
        let cancellation = self.original_cancellation(cooperation)?;
        let wave = self.prepare_wave(arguments, requested_outputs, &cancellation)?;
        Ok(async move {
            match wave.start(&self.runtime, &self.pool, cancellation)? {
                Some(pending) => pending.collect().await,
                None => Ok(Vec::new()),
            }
        })
    }
    /// Submit one actual call now. The returned future owns PendingBatch from
    /// submission, even if it is dropped before its first poll. It owns the
    /// original token, not a borrow of a mutable caller budget or poll scope.
    pub fn start_call<'a>(
        &'a self,
        arguments: Vec<Value>,
        requested_outputs: usize,
        cooperation: &crate::execution::CpuCooperationBudget,
    ) -> Result<
        impl std::future::Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a,
        RuntimeError,
    > {
        let cancellation = self.original_cancellation(cooperation)?;
        crate::context::host_allocation::require_unbounded_workspace(
            Some(&self.runtime),
            "callback queue",
            "finite host-memory immediate callback queues have no complete metadata bound",
        )?;
        let wave = self.prepare_wave(vec![arguments], requested_outputs, &cancellation)?;
        let pending = wave
            .start(&self.runtime, &self.pool, cancellation)?
            .ok_or_else(|| {
                unavailable(Some(&self.runtime), "single callback has no accepted task")
            })?;
        Ok(async move {
            let mut results = pending.collect().await?;
            let result = results.pop().ok_or_else(|| {
                unavailable(
                    Some(&self.runtime),
                    "single callback has no completed result",
                )
            })?;
            if !results.is_empty() {
                return Err(unavailable(
                    Some(&self.runtime),
                    "single callback returned extra results",
                ));
            }
            Ok(result)
        })
    }
    fn original_cancellation(
        &self,
        cooperation: &crate::execution::CpuCooperationBudget,
    ) -> Result<runmat_execution::cancellation::CancellationToken, RuntimeError> {
        let cancellation = cooperation.cancellation_token();
        if !cancellation.same_generation(&self.runtime.cancellation()) {
            return Err(
                crate::context::host_allocation::host_allocation_runtime_error(
                    Some(&self.runtime),
                    "RunMat:ParallelCallbacks:Authority",
                    "callback CPU budget belongs to another cancellation generation",
                    crate::context::host_allocation::HostAllocationError::StaleAuthority,
                ),
            );
        }
        if cancellation.is_cancelled() {
            return Err(execution_error(
                &self.runtime,
                ExecutionServiceError::Cancelled,
            ));
        }
        Ok(cancellation)
    }
    fn prepare_wave(
        &self,
        arguments: Vec<Vec<Value>>,
        requested_outputs: usize,
        cancellation: &runmat_execution::cancellation::CancellationToken,
    ) -> Result<PreparedWave, RuntimeError> {
        if arguments.len() > self.width {
            return Err(unavailable(
                Some(&self.runtime),
                "callback wave exceeds selected worker count",
            ));
        }
        let runtime = &self.runtime;
        let validation_runtime = runtime
            .clone()
            .with_cancellation_token(cancellation.clone());
        let program = self.program.clone().with_validation_workspace(
            crate::execution::value_codec::ValidationWorkspaceOwner::capture(&validation_runtime),
        );
        let _entry = runtime.enter(); // synchronous admission only; never captured by the future.
        let environment = crate::execution::capture_deferred_environment(runtime)?;
        let console = runtime
            .service_ports()
            .host()
            .and_then(|host| host.console_limits());
        let mut wave = PreparedWave {
            calls: Vec::with_capacity(arguments.len()),
            roots: Vec::with_capacity(arguments.len()),
        };
        for arguments in arguments {
            wave.roots.push(RootedValueLease::for_values_in_context(
                &arguments, runtime,
            )?);
            crate::execution::validate_spawn_captures(&arguments).map_err(|mut error| {
                error.gpu_gather_retry = GpuGatherRetry::Never;
                error
            })?;
            let invocation =
                self.selected
                    .invocation(arguments, requested_outputs, &validation_runtime)?;
            let program = self
                .selected
                .deferred_program(&invocation, &program, runtime)?;
            let revision = program
                .program_revision(runtime.program_revision())
                .cloned();
            wave.calls.push(DeferredCall {
                spawn_safety: self.selected.spawn_safety.clone(),
                invocation,
                retry: RetryPolicy::Never,
                program_revision: revision,
                environment: environment.clone(),
                console,
                capabilities: self.selected.capabilities.clone(),
                program,
            });
        }
        Ok(wave)
    }
}
