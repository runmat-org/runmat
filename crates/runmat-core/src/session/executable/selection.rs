//! Executable backend selection and concrete future construction.
//!
//! Leaf constructors keep unselected backend state out of invocation poll
//! frames. Each selected future uses the existing single admitted owner.

use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::{
    admit_executable_sequence, RootedValueSequence, RuntimeContinuationFuture,
};
use runmat_runtime::RuntimeError;
use runmat_value::Value;

#[cfg(not(target_arch = "wasm32"))]
use super::NativeProcedureTarget;
use super::{runtime_error, RunError, RunMatSession};

type SelectedInvocation<'a> =
    Result<RuntimeContinuationFuture<'a, RootedValueSequence>, RuntimeError>;

#[derive(Clone, Copy)]
struct SelectionContext<'a> {
    unit: &'a crate::ExecutableUnit,
    runtime: &'a RuntimeContext,
    bytecode: &'a runmat_vm::Bytecode,
}

struct ResolvedProcedure {
    function: runmat_hir::FunctionId,
    #[cfg(not(target_arch = "wasm32"))]
    name: String,
    arguments: Vec<Value>,
    requested_outputs: usize,
}

impl RunMatSession {
    pub(super) fn select_executable_continuation<'a>(
        &'a mut self,
        unit: &'a crate::ExecutableUnit,
        invocation: crate::ProcedureInvocation,
        runtime: &'a RuntimeContext,
        bytecode: &'a runmat_vm::Bytecode,
        #[cfg(not(target_arch = "wasm32"))] backend: crate::ExecutableBackendPolicy,
    ) -> Result<RuntimeContinuationFuture<'a, RootedValueSequence>, RunError> {
        let context = SelectionContext {
            unit,
            runtime,
            bytecode,
        };
        let selected = match invocation.target {
            crate::ProcedureTarget::Entrypoint => {
                if !invocation.arguments.is_empty() {
                    return Err(runtime_error(
                        "RunMat:ExecutableArity",
                        "executable entrypoint does not accept invocation arguments",
                    ));
                }
                self.select_entrypoint(
                    context,
                    #[cfg(not(target_arch = "wasm32"))]
                    backend,
                )
            }
            crate::ProcedureTarget::Function(name) => {
                let function = unit.functions().resolve_name(&name).ok_or_else(|| {
                    runtime_error(
                        "RunMat:UndefinedExecutableProcedure",
                        format!("executable unit does not contain semantic procedure '{name}'"),
                    )
                })?;
                self.select_procedure(
                    context,
                    ResolvedProcedure {
                        function,
                        #[cfg(not(target_arch = "wasm32"))]
                        name,
                        arguments: invocation.arguments,
                        requested_outputs: invocation.requested_outputs,
                    },
                    #[cfg(not(target_arch = "wasm32"))]
                    backend,
                )
            }
        };
        selected.map_err(RunError::Runtime)
    }

    fn select_entrypoint<'a>(
        &'a mut self,
        context: SelectionContext<'a>,
        #[cfg(not(target_arch = "wasm32"))] backend: crate::ExecutableBackendPolicy,
    ) -> SelectedInvocation<'a> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if backend == crate::ExecutableBackendPolicy::ForcedGenericNative {
                return self.construct_generic_entrypoint(context);
            }
            if self.native_tiering_enabled {
                return self.construct_tiered_entrypoint(context);
            }
        }
        self.construct_interpreted_entrypoint(context)
    }

    fn select_procedure<'a>(
        &'a mut self,
        context: SelectionContext<'a>,
        target: ResolvedProcedure,
        #[cfg(not(target_arch = "wasm32"))] backend: crate::ExecutableBackendPolicy,
    ) -> SelectedInvocation<'a> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if backend == crate::ExecutableBackendPolicy::ForcedGenericNative {
                return self.construct_generic_procedure(context, target);
            }
            if self.native_tiering_enabled {
                return self.construct_tiered_procedure(context, target);
            }
        }
        self.construct_interpreted_procedure(context, target)
    }

    fn construct_interpreted_entrypoint<'a>(
        &'a mut self,
        context: SelectionContext<'a>,
    ) -> SelectedInvocation<'a> {
        admit_executable_sequence(
            context.runtime,
            self.invoke_entrypoint_interpreted(context.unit, context.runtime, context.bytecode),
        )
    }

    fn construct_interpreted_procedure<'a>(
        &'a mut self,
        context: SelectionContext<'a>,
        target: ResolvedProcedure,
    ) -> SelectedInvocation<'a> {
        admit_executable_sequence(
            context.runtime,
            self.invoke_procedure_interpreted(
                context.unit,
                context.runtime,
                context.bytecode,
                target.function,
                target.arguments,
                target.requested_outputs,
            ),
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn construct_generic_entrypoint<'a>(
        &'a mut self,
        context: SelectionContext<'a>,
    ) -> SelectedInvocation<'a> {
        admit_executable_sequence(
            context.runtime,
            self.invoke_generic_native(
                context.unit,
                context.runtime,
                &context.bytecode.function_registry,
                None,
                Vec::new(),
                0,
            ),
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn construct_tiered_entrypoint<'a>(
        &'a mut self,
        context: SelectionContext<'a>,
    ) -> SelectedInvocation<'a> {
        admit_executable_sequence(
            context.runtime,
            self.invoke_tiered_entrypoint(context.unit, context.runtime, context.bytecode),
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn construct_generic_procedure<'a>(
        &'a mut self,
        context: SelectionContext<'a>,
        target: ResolvedProcedure,
    ) -> SelectedInvocation<'a> {
        admit_executable_sequence(context.runtime, async move {
            self.invoke_generic_native(
                context.unit,
                context.runtime,
                &context.bytecode.function_registry,
                Some(&target.name),
                target.arguments,
                target.requested_outputs,
            )
            .await
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn construct_tiered_procedure<'a>(
        &'a mut self,
        context: SelectionContext<'a>,
        target: ResolvedProcedure,
    ) -> SelectedInvocation<'a> {
        admit_executable_sequence(context.runtime, async move {
            self.invoke_tiered_procedure(
                context.unit,
                context.runtime,
                context.bytecode,
                NativeProcedureTarget {
                    function: target.function,
                    name: &target.name,
                },
                target.arguments,
                target.requested_outputs,
            )
            .await
        })
    }
}
