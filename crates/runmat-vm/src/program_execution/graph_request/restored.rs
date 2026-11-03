//! Current-session restoration of portable, immutable selected code.

use super::*;
use runmat_execution_artifact::{AdmittedGraphUnitBundle, ArchiveAdmittedCapturedGraphCode};

#[path = "restored/retention.rs"]
mod retention;
use runmat_runtime::call::descriptor::{CallableDescriptor, CallableTarget};
use runmat_runtime::call::BuiltinArguments;
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::AdmittedCallableReentry;
use runmat_runtime::RuntimeError;

/// Executable/source/static ownership, never an invocation runtime or invoker.
/// The actual snapshot producer retains this and selects through the canonical
/// resolver; it does not publish a merged graph-unit name namespace.
#[derive(Clone)]
pub struct RestoredSelectedGraph {
    code: Rc<selected::RetainedSelectedGraph>,
}

impl RestoredSelectedGraph {
    pub fn registry_for_descriptor(
        &self,
        selected: &CallableDescriptor,
    ) -> Result<&crate::FunctionRegistry, RuntimeError> {
        if selected.target.selected_builtin_name().is_some() {
            return Ok(self.primary_registry());
        }
        let binding = match &selected.target {
            CallableTarget::Bound(binding) | CallableTarget::BoundLexical { binding, .. } => {
                binding
            }
            _ => {
                return Err(rejected(
                    "restored worker target has no exact admitted definition",
                ))
            }
        };
        let owner = self.code.owner_for_binding(binding).map_err(rejected)?;
        self.code
            .unit(owner)
            .map(|unit| unit.registry())
            .map_err(rejected)
    }

    pub fn admit_and_decode(
        admitted: &ArchiveAdmittedCapturedGraphCode,
        bundles: Vec<AdmittedGraphUnitBundle>,
        runtime: &RuntimeContext,
        foreign_ports: Option<&SelectedGraphForeignPorts>,
    ) -> Result<(Self, RootedValues), RuntimeError> {
        runmat_runtime::context::host_allocation::require_unbounded_workspace(
            Some(runtime),
            "selected callable restore",
            "selected executable and capture admission has no complete finite workspace bound",
        )?;
        let _entry = runtime.enter();
        let cancellation = runtime.cancellation();
        if cancellation.is_cancelled() {
            return Err(runmat_runtime::execution::ExecutionServiceError::Cancelled
                .into_runtime_error_in_context(Some(runtime)));
        }
        let selected = SelectedWorkerGraph::admit_captured(
            admitted,
            bundles,
            runtime,
            foreign_ports.map(SelectedGraphForeignPorts::as_slice),
        );
        if cancellation.is_cancelled() {
            return Err(runmat_runtime::execution::ExecutionServiceError::Cancelled
                .into_runtime_error_in_context(Some(runtime)));
        }
        let selected = selected.map_err(rejected)?;
        let primary = selected
            .runtime_for_unit(runmat_execution::value::GraphProgramUnitId(0))
            .map_err(rejected)?;
        let values = runmat_runtime::execution::value_codec::decode_payload_graph_with_selected_classes_in_context(
            &admitted.code().graph,
            &primary,
            &selected.source_map,
            &selected.issuer,
            &selected.layouts,
            &selected.proofs,
            &selected.bindings,
            &selected.object_classes,
            runmat_execution::value::ValueLimits::default(),
        );
        if cancellation.is_cancelled() {
            return Err(runmat_runtime::execution::ExecutionServiceError::Cancelled
                .into_runtime_error_in_context(Some(runtime)));
        }
        let values = values.map_err(|source| {
            runmat_runtime::build_runtime_error("selected callable capture decoding failed")
                .with_identifier("RunMat:Execution:SelectedCallableUnavailable")
                .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                .with_source(source)
                .build()
        })?;
        Ok((
            Self {
                code: selected.into_code(),
            },
            values,
        ))
    }

    /// Only this immutable selected unit is used for name visibility during
    /// canonical descriptor selection. Actual handles retain their exact lease.
    pub fn primary_registry(&self) -> &crate::FunctionRegistry {
        self.code.units[0].registry()
    }

    pub fn admit_callable(
        &self,
        runtime: &RuntimeContext,
        descriptor: &CallableDescriptor,
    ) -> Result<Option<AdmittedCallableReentry>, RuntimeError> {
        let _entry = runtime.enter();
        let binding = match &descriptor.target {
            CallableTarget::Bound(binding) => binding,
            CallableTarget::BoundLexical { binding, closure } => {
                if !matches!(closure.as_value(), Value::Closure(value) if value.binding.as_ref() == Some(binding))
                {
                    return Err(rejected(
                        "restored closure differs from its admitted binding",
                    ));
                }
                binding
            }
            // Builtins use the ordinary compiled runtime dispatch path. This
            // producer does not attach source authority to any other target.
            _ => return Ok(None),
        };
        let owner = self.code.owner_for_binding(binding).map_err(rejected)?;
        let unit = self.code.unit(owner).map_err(rejected)?;
        let body = selected::selected_bound_body(unit.registry(), binding).map_err(rejected)?;
        AdmittedCallableReentry::from_selected_definition(
            runtime,
            unit.selected_source_owner()
                .ok_or_else(|| rejected("restored source owner is absent"))?
                .clone(),
            binding,
            owner.origin,
            Arc::clone(&body.definition_owner),
            body.source_id
                .and_then(|source| source.runtime())
                .ok_or_else(|| rejected("restored callable has no issued source member"))?,
        )
        .map(Some)
    }

    pub fn invoke(
        &self,
        runtime: RuntimeContext,
        receipt: AdmittedCallableReentry,
        descriptor: CallableDescriptor<BuiltinArguments>,
    ) -> Result<runmat_runtime::user_functions::BoundFunctionFuture, RuntimeError> {
        let cancellation = runtime.cancellation();
        let runtime = runtime.with_cancellation_token(cancellation);
        let active = runtime.enter();
        // Recheck the actual producer-owned body; a receipt from a different
        // restored graph with equal serialized bytes is not interchangeable.
        let target_only = CallableDescriptor {
            target: descriptor.target.clone(),
            args: Vec::new(),
            requested_outputs: descriptor.requested_outputs,
            metadata: descriptor.metadata.clone(),
        };
        let expected = self
            .admit_callable(&runtime, &target_only)?
            .ok_or_else(|| rejected("restored reentry has no selected bound target"))?;
        if !receipt.same_selected_definition(&expected) {
            return Err(rejected(
                "restored reentry differs from its exact admitted definition",
            ));
        }
        // Per-unit foreign services are not retained with code. A host must
        // provide a new exact archive-materialized projection for such calls;
        // the general current foreign catalog cannot substitute for it.
        if self.code.units.iter().any(|unit| {
            let interop = unit.interop();
            !interop.adapters.is_empty()
                || !interop.foreign_types.is_empty()
                || !interop.adapter_contracts.is_empty()
        }) {
            return Err(rejected(
                "restored interop requires a current selected-unit foreign projection",
            ));
        }
        let code = Rc::clone(&self.code);
        // Capture the existing invocation's weak driver before projecting its
        // selected code/source view. Reuse its canonical sibling queue rather
        // than recursively driving a second VM beneath the caller's builtin.
        let driver =
            crate::interpreter::process_driver::ProcessDriverHandle::current_in_context(&runtime)?;
        let invoker = runmat_runtime::user_functions::install_local_bound_function_invoker(
            Rc::new(move |current, binding, arguments, outputs, environment| {
                let code = Rc::clone(&code);
                let driver = driver.clone();
                runmat_runtime::execution::admit_executable_sequence_with(
                    &current.clone(),
                    move || async move {
                        invoke_selected_code_bound(
                            &code,
                            crate::interpreter::runner::BoundInvocation {
                                runtime: current,
                                binding,
                                args: arguments,
                                requested_outputs: outputs,
                                driver,
                                lexical_environment: environment,
                            },
                            None,
                        )
                        .await
                    },
                )
            }),
        )?;
        // Snapshot the exact code-bound invoker before its temporary install
        // is retired. A funding/cancellation continuation alone does not retain
        // call projections. The same captured context owns poll and drop.
        let scope = runtime.prepare_poll_scope()?;
        let future =
            runmat_runtime::execution::admit_executable_sequence_with(&runtime, move || {
                scope.construct_result_unboxed(move |runtime| async move {
                    runmat_runtime::call::descriptor::execute_retained_callable_descriptor(
                        Some(&runtime),
                        descriptor,
                    )
                    .await
                })
            });
        drop(invoker);
        drop(active);
        future
    }
}

fn rejected(message: impl Into<String>) -> RuntimeError {
    runmat_runtime::build_runtime_error(message)
        .with_identifier("RunMat:Execution:SelectedCallableUnavailable")
        .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
        .build()
}
