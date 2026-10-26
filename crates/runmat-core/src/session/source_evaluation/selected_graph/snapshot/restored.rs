//! Producer-checked restoration, without old runtime or invoker retention.

use super::*;
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::{AdmittedCallableReentry, RestoredCallableGraph};

struct RestoredGraphSnapshot {
    code: runmat_vm::RestoredSelectedGraph,
    // This concrete original producer owns immutable code/source data and a
    // weak evaluator, not a RuntimeBound unit or arbitrary host invoker.
    fallback: Rc<SelectedGraphSnapshot>,
}

impl SelectedGraphSnapshot {
    pub(super) fn restore_selected_callables(
        &self,
        runtime: &RuntimeContext,
        portable: runmat_execution::value::PortableSelectedCode,
    ) -> Result<RestoredCallableGraph, RuntimeError> {
        if !self.primary.source_owner().is_bound_in(runtime) {
            return Err(unavailable(
                "callable restoration requires its current original source invocation",
            ));
        }
        runmat_runtime::context::host_allocation::require_unbounded_workspace(
            Some(runtime),
            "selected callable restore",
            "selected executable and capture admission has no complete finite workspace bound",
        )?;
        check_cancelled(runtime)?;
        let evaluator = self
            .evaluator
            .upgrade()
            .ok_or_else(|| unavailable("selected graph session has ended"))?;
        let expected = evaluator.current_generation();
        let epoch = expected.epoch();
        let mut registry = expected.registry()?.as_ref().clone();
        let mut artifacts = expected.artifacts()?.as_ref().clone();
        let mut sources = expected.source_owners()?.as_ref().clone();
        let admitted = runmat_execution_artifact::CapturedGraphCode::new(portable)
            .into_validated()
            .map_err(|error| {
                runmat_runtime::build_runtime_error("saved callable code failed archive admission")
                    .with_identifier("RunMat:Execution:SelectedCallableUnavailable")
                    .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                    .with_source(error)
                    .build()
            })?;
        let (admitted, bundles) = admitted.into_admitted_with_bundles();
        let (code, values) =
            runmat_vm::RestoredSelectedGraph::admit_and_decode(&admitted, bundles, runtime, None)?;
        code.retain_definitions_into(&mut registry)?;
        for bytes in code.canonical_units() {
            artifacts
                .insert_canonical_arc_unit(Arc::clone(bytes?))
                .map_err(|error| {
                    runmat_runtime::build_runtime_error(
                        "saved callable code failed definition publication",
                    )
                    .with_identifier("RunMat:Execution:SelectedCallableUnavailable")
                    .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                    .with_source(error)
                    .build()
                })?;
        }
        sources.extend(code.retained_sources()?);
        check_cancelled(runtime)?;
        let prepared = evaluator.prepare_import(registry, artifacts, sources);
        let retired = evaluator.commit_import(&expected, &epoch, prepared)?;
        drop(retired);
        let fallback = Rc::new(SelectedGraphSnapshot {
            primary: Rc::clone(&self.primary),
            evaluator: self.evaluator.clone(),
        });
        let program = DeferredGraphProgram::new(Rc::new(RestoredGraphSnapshot { code, fallback }));
        Ok(RestoredCallableGraph { values, program })
    }
}

impl DeferredGraphSnapshot for RestoredGraphSnapshot {
    fn select_retained_worker_callable(
        &self,
        runtime: &RuntimeContext,
        selected: runmat_runtime::call::descriptor::RepeatableCallable,
    ) -> Result<runmat_runtime::execution::SelectedWorkerCallable, RuntimeError> {
        self.code
            .admit_callable(runtime, selected.selected_descriptor())?;
        let registry = self
            .code
            .registry_for_descriptor(selected.selected_descriptor())?;
        runmat_vm::select_retained_worker_callable(runtime, selected, registry)
    }

    fn select_repeatable_callable(
        &self,
        runtime: &RuntimeContext,
        target: runmat_runtime::execution::RootedValue,
    ) -> Result<runmat_runtime::call::descriptor::RepeatableCallable, RuntimeError> {
        require_bound_handle(&target)?;
        crate::callable_selection::select(runtime, self.code.primary_registry(), target)
    }

    fn retained_callable_reentry(
        &self,
        runtime: &RuntimeContext,
        selected: &runmat_runtime::call::descriptor::CallableDescriptor,
    ) -> Result<Option<AdmittedCallableReentry>, RuntimeError> {
        self.code.admit_callable(runtime, selected)
    }

    fn invoke_retained_reentry(
        self: Rc<Self>,
        runtime: RuntimeContext,
        receipt: AdmittedCallableReentry,
        selected: runmat_runtime::call::descriptor::CallableDescriptor<
            runmat_runtime::call::BuiltinArguments,
        >,
    ) -> Result<runmat_runtime::user_functions::BoundFunctionFuture, RuntimeError> {
        self.code.invoke(runtime, receipt, selected)
    }

    fn select_callable(
        &self,
        runtime: &RuntimeContext,
        target: runmat_runtime::execution::RootedValue,
    ) -> Result<runmat_runtime::execution::SelectedWorkerCallable, RuntimeError> {
        require_bound_handle(&target)?;
        runmat_vm::select_worker_callable(runtime, target, self.code.primary_registry())
    }

    fn snapshot(
        &self,
        invocation: &DeferredInvocation,
        limits: ValueLimits,
        workspace: Option<&runmat_runtime::execution::value_codec::ValidationWorkspaceOwner>,
    ) -> Result<DeferredGraphCapture, ValueCodecError> {
        self.fallback.snapshot(invocation, limits, workspace)
    }

    fn capture_callables(
        &self,
        selected: &[runmat_runtime::call::descriptor::CallableDescriptor],
        limits: ValueLimits,
        workspace: Option<&runmat_runtime::execution::value_codec::ValidationWorkspaceOwner>,
    ) -> Result<runmat_runtime::execution::DeferredSelectedCodeCapture, ValueCodecError> {
        self.fallback.capture_callables(selected, limits, workspace)
    }

    fn restore_callables(
        &self,
        runtime: &RuntimeContext,
        portable: runmat_execution::value::PortableSelectedCode,
    ) -> Result<RestoredCallableGraph, RuntimeError> {
        self.fallback.restore_selected_callables(runtime, portable)
    }
}

fn check_cancelled(runtime: &RuntimeContext) -> Result<(), RuntimeError> {
    if runtime.cancellation().is_cancelled() {
        Err(ExecutionServiceError::Cancelled.into_runtime_error_in_context(Some(runtime)))
    } else {
        Ok(())
    }
}

fn require_bound_handle(
    value: &runmat_runtime::execution::RootedValue,
) -> Result<(), RuntimeError> {
    if matches!(
        value.value(),
        runmat_value::Value::BoundFunctionHandle(_) | runmat_value::Value::Closure(_)
    ) {
        Ok(())
    } else {
        Err(unavailable(
            "restored user-code selection requires its actual bound callable handle",
        ))
    }
}

fn unavailable(message: impl Into<String>) -> RuntimeError {
    runmat_runtime::build_runtime_error(message)
        .with_identifier("RunMat:Execution:SelectedCallableUnavailable")
        .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
        .build()
}
