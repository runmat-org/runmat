//! Executable backend selection and concrete future construction.
//!
//! Leaf constructors keep unselected backend state out of invocation poll
//! frames. Each selected future uses the existing single admitted owner.

use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::{
    admit_executable_sequence, RootedValueSequence, RuntimeContinuationFuture,
};
use runmat_runtime::RuntimeError;

#[cfg(not(target_arch = "wasm32"))]
use super::NativeProcedureTarget;
use super::{runtime_error, ProcedureCall, RunError, RunMatSession};

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
    call: ProcedureCall,
}

impl RunMatSession {
    pub(super) fn select_executable_continuation<'a>(
        &'a mut self,
        unit: &'a crate::ExecutableUnit,
        invocation: crate::executable::AdmittedProcedureInvocation,
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
                if invocation.requested_outputs > 1 {
                    return Err(runtime_error(
                        "RunMat:ExecutableArity",
                        "scripts do not define multiple output ordinals",
                    ));
                }
                if !invocation.arguments.is_empty() {
                    return Err(runtime_error(
                        "RunMat:ExecutableArity",
                        "executable entrypoint does not accept invocation arguments",
                    ));
                }
                self.select_entrypoint(
                    context,
                    invocation.requested_outputs,
                    #[cfg(not(target_arch = "wasm32"))]
                    backend,
                )
            }
            crate::ProcedureTarget::SemanticFunction(function) => {
                let origin = runmat_execution::FunctionBindingOrigin::from_revision(
                    &unit.revision().program_revision,
                    function,
                )
                .map_err(|error| {
                    runtime_error("RunMat:UndefinedExecutableProcedure", error.to_string())
                })?;
                let function = unit
                    .local_function_for_origin(origin)
                    .map_err(|error| runtime_error("RunMat:UndefinedExecutableProcedure", error))?
                    .ok_or_else(|| {
                        runtime_error(
                            "RunMat:UndefinedExecutableProcedure",
                            "executable unit does not own the semantic procedure",
                        )
                    })?;
                let name = unit
                    .functions()
                    .get(function)
                    .ok_or_else(|| {
                        runtime_error(
                            "RunMat:UndefinedExecutableProcedure",
                            "executable unit does not contain the semantic procedure",
                        )
                    })?
                    .display_name
                    .clone();
                #[cfg(not(target_arch = "wasm32"))]
                if backend == crate::ExecutableBackendPolicy::ForcedGenericNative
                    && (unit.native_function_id(&name) != Some(function)
                        || !unit.owns_callable(function))
                {
                    return Err(runtime_error(
                        "RunMat:ExecutableBackend",
                        "semantic function is not owned by this native compiler product",
                    ));
                }
                #[cfg(target_arch = "wasm32")]
                let _ = name;
                self.select_procedure(
                    context,
                    ResolvedProcedure {
                        function,
                        #[cfg(not(target_arch = "wasm32"))]
                        name,
                        call: ProcedureCall {
                            arguments: invocation.arguments,
                            requested_outputs: invocation.requested_outputs,
                            lexical_environment: None,
                        },
                    },
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
                        call: ProcedureCall {
                            arguments: invocation.arguments,
                            requested_outputs: invocation.requested_outputs,
                            lexical_environment: None,
                        },
                    },
                    #[cfg(not(target_arch = "wasm32"))]
                    backend,
                )
            }
            crate::ProcedureTarget::RetainedCallable(callback) => {
                let (selected, function) = crate::executable::callback::select_owned(
                    unit,
                    runtime,
                    &bytecode.function_registry,
                    callback,
                )
                .map_err(RunError::Runtime)?;
                let lexical_environment = match selected.selected_target() {
                    runmat_runtime::call::descriptor::CallableTarget::BoundLexical {
                        closure,
                        ..
                    } => closure.as_closure().lexical_environment.clone(),
                    _ => None,
                };
                let arguments = selected
                    .prepare_arguments(Some(runtime), invocation.arguments)
                    .map_err(RunError::Runtime)?;
                #[cfg(not(target_arch = "wasm32"))]
                let name = bytecode
                    .function_registry
                    .get(function)
                    .ok_or_else(|| {
                        runtime_error(
                            "RunMat:UndefinedExecutableProcedure",
                            "selected callable definition is absent",
                        )
                    })?
                    .display_name
                    .clone();
                self.select_procedure(
                    context,
                    ResolvedProcedure {
                        function,
                        #[cfg(not(target_arch = "wasm32"))]
                        name,
                        call: ProcedureCall {
                            arguments: runmat_runtime::call::BuiltinArguments::new(arguments),
                            requested_outputs: invocation.requested_outputs,
                            lexical_environment,
                        },
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
        requested_outputs: usize,
        #[cfg(not(target_arch = "wasm32"))] backend: crate::ExecutableBackendPolicy,
    ) -> SelectedInvocation<'a> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Native entrypoints do not expose workspace assignments as
            // ordinal returns. Keep an explicit script `ans` request on the
            // interpreter path unless the caller required native execution.
            if requested_outputs == 1 {
                if backend == crate::ExecutableBackendPolicy::ForcedGenericNative {
                    return Err(runmat_runtime::build_runtime_error(
                        "native script entrypoints do not expose the ans binding",
                    )
                    .with_identifier("RunMat:ExecutableBackend")
                    .build());
                }
                return self.construct_interpreted_entrypoint(context, requested_outputs);
            }
            if backend == crate::ExecutableBackendPolicy::ForcedGenericNative {
                return self.construct_generic_entrypoint(context);
            }
            if self.native_tiering_enabled {
                return self.construct_tiered_entrypoint(context);
            }
        }
        self.construct_interpreted_entrypoint(context, requested_outputs)
    }

    fn select_procedure<'a>(
        &'a mut self,
        context: SelectionContext<'a>,
        target: ResolvedProcedure,
        #[cfg(not(target_arch = "wasm32"))] backend: crate::ExecutableBackendPolicy,
    ) -> SelectedInvocation<'a> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // The native cache's public entry lookup is name-based. Never
            // let that lookup select a different body from the authenticated
            // semantic target (for example a shadowed lexical name).
            if context.unit.native_function_id(&target.name) != Some(target.function) {
                if backend == crate::ExecutableBackendPolicy::ForcedGenericNative {
                    return Err(runmat_runtime::build_runtime_error(
                        "semantic function does not identify this native compiler entry",
                    )
                    .with_identifier("RunMat:ExecutableBackend")
                    .build());
                }
                return self.construct_interpreted_procedure(context, target);
            }
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
        requested_outputs: usize,
    ) -> SelectedInvocation<'a> {
        admit_executable_sequence(
            context.runtime,
            self.invoke_entrypoint_interpreted(
                context.unit,
                context.runtime,
                context.bytecode,
                requested_outputs,
            ),
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
                target.call,
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
                ProcedureCall {
                    arguments: runmat_runtime::call::BuiltinArguments::from_borrowed(
                        &[],
                        Some(context.runtime),
                    )?,
                    requested_outputs: 0,
                    lexical_environment: None,
                },
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
                target.call,
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
                target.call,
            )
            .await
        })
    }
}
