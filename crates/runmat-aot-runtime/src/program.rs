use std::{rc::Rc, sync::Arc};

use runmat_native_executor::execute::{NativeWorkspaceInput, RootedNativeExecution};
use runmat_types::FunctionId;
use runmat_value::{Value, ValueSequence};

mod bound;
mod class_registration;
mod external_sources;
mod name_selection;
#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "program/s1_aot_direct_call_tests.rs"]
mod s1_aot_direct_call_tests;
mod super_constructor;
#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "program/super_constructor_tests.rs"]
mod super_constructor_tests;
#[cfg(test)]
mod tests;

pub async fn invoke(
    executor: Rc<runmat_native_executor::NativeExecutor>,
    program: &runmat_native_codegen::aot::AotProgramManifest,
    function: FunctionId,
    arguments: Vec<Value>,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<RootedNativeExecution, runmat_runtime::RuntimeError> {
    let roots = runmat_runtime::execution::RootedValueLease::for_values(arguments.iter())
        .map_err(|error| error.into_runtime_error())?;
    let runtime = install_acceleration_services(runtime);
    let (runtime, placement) = runtime.native_invocation_context();
    let guards = ProgramInvocationGuards::install(Rc::clone(&executor), program, &runtime)
        .map_err(|error| *error)?;
    class_registration::install(&executor, &runtime).map_err(|error| *error)?;
    let runtime =
        super_constructor::bind_selected(&executor, function, runtime).map_err(|error| *error)?;
    let selected = runtime.clone();
    let execution = runmat_runtime::context::run_native_continuation_with(
        &runtime,
        placement,
        roots,
        move || async move {
            let _active =
                runmat_runtime::user_functions::push_active_semantic_function(function.0)?;
            executor
                .invoke_async(function, arguments, requested_outputs, selected)
                .await
                .map_err(native_executor_error)
        },
    )
    .await;
    drop(guards);
    execution
}

pub async fn invoke_workspace(
    executor: Rc<runmat_native_executor::NativeExecutor>,
    program: &runmat_native_codegen::aot::AotProgramManifest,
    function: FunctionId,
    workspace: NativeWorkspaceInput,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<RootedNativeExecution, runmat_runtime::RuntimeError> {
    let roots = runmat_runtime::execution::RootedValueLease::for_values(workspace.profile_values())
        .map_err(|error| error.into_runtime_error())?;
    let runtime = install_acceleration_services(runtime);
    let (runtime, placement) = runtime.native_invocation_context();
    let guards = ProgramInvocationGuards::install(Rc::clone(&executor), program, &runtime)
        .map_err(|error| *error)?;
    class_registration::install(&executor, &runtime).map_err(|error| *error)?;
    let runtime =
        super_constructor::bind_selected(&executor, function, runtime).map_err(|error| *error)?;
    let selected = runtime.clone();
    let execution = runmat_runtime::context::run_native_continuation_with(
        &runtime,
        placement,
        roots,
        move || async move {
            let _active =
                runmat_runtime::user_functions::push_active_semantic_function(function.0)?;
            executor
                .invoke_workspace_async(function, workspace, requested_outputs, selected)
                .await
                .map_err(native_executor_error)
        },
    )
    .await;
    drop(guards);
    execution
}

/// Direct rooted entry keeps the decoder's original argument owner and roots
/// through native admission, execution, and any queued continuation.
pub async fn invoke_rooted(
    executor: Rc<runmat_native_executor::NativeExecutor>,
    program: &runmat_native_codegen::aot::AotProgramManifest,
    function: FunctionId,
    arguments: runmat_runtime::execution::RootedValues,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<RootedNativeExecution, runmat_runtime::RuntimeError> {
    let runtime = install_acceleration_services(runtime);
    let (runtime, placement) = runtime.native_invocation_context();
    let guards = ProgramInvocationGuards::install(Rc::clone(&executor), program, &runtime)
        .map_err(|error| *error)?;
    class_registration::install(&executor, &runtime).map_err(|error| *error)?;
    let runtime =
        super_constructor::bind_selected(&executor, function, runtime).map_err(|error| *error)?;
    let selected = runtime.clone();
    let arguments = runmat_runtime::call::RootedFunctionArguments::ordinary(arguments);
    let execution = runmat_runtime::context::run_rooted_function_native_continuation(
        &runtime,
        placement,
        arguments,
        move |arguments| async move {
            let _active =
                runmat_runtime::user_functions::push_active_semantic_function(function.0)?;
            executor
                .invoke_rooted_function_arguments_async(
                    function,
                    arguments,
                    requested_outputs,
                    selected,
                    None,
                )
                .await
                .map_err(native_executor_error)
        },
    )
    .await;
    drop(guards);
    execution
}

struct ProgramInvocationGuards {
    _semantic: runmat_runtime::user_functions::FunctionInvokerGuard,
    _bound: runmat_runtime::user_functions::BoundFunctionInvokerGuard,
    _lexical: runmat_runtime::user_functions::LexicalFunctionInvokerGuard,
    _resolver: runmat_runtime::user_functions::FunctionResolverGuard,
    _scoped: runmat_runtime::user_functions::ScopedNameTargetResolverGuard,
    _name_candidates: runmat_runtime::user_functions::SemanticNameCandidatesGuard,
    _catalog: runmat_runtime::user_functions::SourceFunctionCatalogGuard,
    _execution_source: runmat_runtime::user_functions::ExecutionSourceResolverGuard,
}

impl ProgramInvocationGuards {
    fn install(
        executor: Rc<runmat_native_executor::NativeExecutor>,
        program: &runmat_native_codegen::aot::AotProgramManifest,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<Self, Box<runmat_runtime::RuntimeError>> {
        program
            .validate()
            .map_err(|error| Box::new(native_error(error)))?;
        if &program.executable.program != executor.program_revision() {
            return Err(Box::new(native_error(
                "standalone manifest and linked Native IR name different programs",
            )));
        }
        let bound_functions = Arc::new(bound::AdmittedBoundFunctions::from_program(program)?);
        let mut functions = program
            .functions
            .iter()
            .map(|function| {
                let source_ordinal = usize::try_from(function.source.0)
                    .map_err(|_| native_error("program source ordinal exceeds host indexing"))?;
                let function_id = function.function.0;
                let source_id = runtime
                    .admitted_program_source_id(function.source)
                    .map_err(source_admission_error)?
                    .ok_or_else(|| {
                        source_admission_error(
                            runmat_runtime::context::ProgramSourceSelectionError::Missing(
                                function.source,
                            ),
                        )
                    })?;
                Ok(runmat_runtime::user_functions::SourceFunctionInfo {
                    source_id,
                    name: function.name.clone(),
                    function: function_id,
                    role: function.role,
                    source_file: program
                        .source_map
                        .entries
                        .get(source_ordinal)
                        .map(|source| source.relative_path.clone()),
                    binding_origin: Some(
                        runmat_execution::FunctionBindingOrigin::from_revision(
                            &program.executable.program,
                            function.portable_function,
                        )
                        .map_err(native_error)?,
                    ),
                    definition_owners: None,
                })
            })
            .collect::<Result<Vec<_>, runmat_runtime::RuntimeError>>()?;
        functions.sort_by_key(|function| function.function);
        // Installation is synchronous, but it must target this program's
        // context state rather than the ambient compatibility slot. Native
        // invocation activates the same context around every callback.
        let _runtime_scope = runtime.enter();
        let previous_semantic = runmat_runtime::user_functions::current_semantic_function_invoker();
        let native_functions = Arc::new(
            program
                .functions
                .iter()
                .map(|function| function.function)
                .collect::<std::collections::BTreeSet<_>>(),
        );
        let semantic_executor = Rc::clone(&executor);
        let semantic_runtime = runtime.native_invocation_context().0;
        let semantic_native_functions = native_functions.clone();
        let semantic = runmat_runtime::user_functions::install_local_semantic_function_invoker(
            Rc::new(move |selected, function, arguments, requested_outputs| {
                let lexical_environment = arguments.callable().and_then(|callable| {
                    let Value::Closure(closure) = callable.value() else {
                        return None;
                    };
                    closure.lexical_environment.clone()
                });
                let executor = Rc::clone(&semantic_executor);
                let same_owner = selected.as_ref().is_none_or(|context| {
                    context.local_identity() == semantic_runtime.local_identity()
                        && context.program_revision() == semantic_runtime.program_revision()
                });
                let runtime = if same_owner {
                    selected.map_or_else(
                        || semantic_runtime.clone(),
                        |context| {
                            semantic_runtime
                                .clone()
                                .try_with_inherited_callback_invocation_authority_from(&context)
                                .expect(
                                    "callback Runtime owner checked before authority projection",
                                )
                        },
                    )
                } else {
                    semantic_runtime.clone()
                };
                let runtime = isolated_runtime(runtime);
                let previous = previous_semantic.clone();
                let native_functions = Arc::clone(&semantic_native_functions);
                let admission_runtime = runtime.clone();
                runmat_runtime::user_functions::admit_user_function_with(
                    Some(&admission_runtime),
                    move || async move {
                        if !same_owner {
                            return Err(native_error(
                                "semantic callback selected another standalone runtime owner",
                            ));
                        }
                        let function = FunctionId(function);
                        if !native_functions.contains(&function) {
                            if let Some(previous) = previous {
                                return runtime
                                    .scope(previous(
                                        Some(runtime.clone()),
                                        function.0,
                                        arguments,
                                        requested_outputs,
                                    )?)
                                    .await;
                            }
                            return Err(native_error(format!(
                                "semantic function {} is unavailable",
                                function.0
                            )));
                        }
                        invoke_native_values(
                            Rc::clone(&executor),
                            function,
                            arguments,
                            requested_outputs,
                            runtime,
                            lexical_environment,
                        )
                        .await
                    },
                )
            }),
        )?;

        let previous_bound = runmat_runtime::user_functions::current_bound_function_invoker();
        let bound_executor = Rc::clone(&executor);
        let bound = runmat_runtime::user_functions::install_local_bound_function_invoker(Rc::new(
            move |runtime, binding, arguments, requested_outputs, lexical_environment| {
                let executor = Rc::clone(&bound_executor);
                let functions = Arc::clone(&bound_functions);
                let previous = previous_bound.clone();
                runmat_runtime::execution::admit_executable_sequence_with(
                    &runtime.clone(),
                    move || async move {
                        let Some(function) = functions.resolve(&binding).map_err(|error| *error)?
                        else {
                            if let Some(previous) = previous {
                                return previous(
                                    runtime,
                                    binding,
                                    arguments,
                                    requested_outputs,
                                    lexical_environment,
                                )?
                                .await;
                            }
                            return Err(bound::unavailable(
                                "bound function has no admitted execution owner",
                            ));
                        };
                        let (runtime, super_body_owner) =
                            runtime.take_retained_super_body_for(&binding)?;
                        if let Some(expected) = super_body_owner {
                            let selected_owner = executor
                                .function_metadata(function)
                                .and_then(|metadata| metadata.class_method_owner.as_ref());
                            if selected_owner != Some(&expected)
                                || expected.role != runmat_types::ClassMethodRole::Constructor
                                || expected.is_static
                            {
                                return Err(native_error(
                                    "linked superclass body disagrees with its admitted owner",
                                ));
                            }
                        }
                        let runtime = isolated_runtime(runtime);
                        invoke_native_values(
                            Rc::clone(&executor),
                            function,
                            arguments,
                            requested_outputs,
                            runtime,
                            lexical_environment,
                        )
                        .await
                    },
                )
            },
        ))?;

        let previous_lexical = runmat_runtime::user_functions::current_lexical_function_invoker();
        let lexical_executor = Rc::clone(&executor);
        let lexical_runtime = runtime.native_invocation_context().0;
        let lexical_native_functions = native_functions;
        let lexical = runmat_runtime::user_functions::install_local_lexical_function_invoker(
            Rc::new(move |selected, call| {
                let call_roots = runmat_runtime::execution::RootedValueLease::for_traced(&call)
                    .map_err(|error| error.into_runtime_error());
                let executor = Rc::clone(&lexical_executor);
                let same_owner = selected.as_ref().is_none_or(|context| {
                    context.local_identity() == lexical_runtime.local_identity()
                        && context.program_revision() == lexical_runtime.program_revision()
                });
                let runtime = if same_owner {
                    selected.map_or_else(
                        || lexical_runtime.clone(),
                        |context| {
                            lexical_runtime
                                .clone()
                                .try_with_inherited_callback_invocation_authority_from(&context)
                                .expect(
                                    "callback Runtime owner checked before authority projection",
                                )
                        },
                    )
                } else {
                    lexical_runtime.clone()
                };
                let runtime = isolated_runtime(runtime);
                let previous = previous_lexical.clone();
                let native_functions = Arc::clone(&lexical_native_functions);
                let admission_runtime = runtime.clone();
                runmat_runtime::user_functions::admit_lexical_function_with(
                    Some(&admission_runtime),
                    move || async move {
                        let _call_roots = call_roots?;
                        if !same_owner {
                            return Err(native_error(
                                "lexical callback selected another standalone runtime owner",
                            ));
                        }
                        let function = FunctionId(call.function);
                        if !native_functions.contains(&function) {
                            if let Some(previous) = previous {
                                return runtime.scope(previous(Some(runtime.clone()), call)?).await;
                            }
                            return Err(native_error(format!(
                                "lexical function {} is unavailable",
                                function.0
                            )));
                        }
                        let roots = runmat_runtime::execution::RootedValueLease::for_values(
                            call.arguments
                                .iter()
                                .chain(call.captures.iter().map(|capture| &capture.value)),
                        )
                        .map_err(|error| error.into_runtime_error())?;
                        let environment_roots = call
                            .environment
                            .as_ref()
                            .map(runmat_runtime::call::lexical::root_environment)
                            .transpose()?;
                        let runtime =
                            super_constructor::bind_selected(&executor, function, runtime)
                                .map_err(|error| *error)?;
                        let (runtime, placement) = runtime.native_invocation_context();
                        let selected = runtime.clone();
                        let requested_outputs = call.requested_outputs;
                        let execution = runmat_runtime::context::run_native_continuation_with(
                        &runtime,
                        placement,
                        roots,
                        move || async move {
                            let _active = runmat_runtime::user_functions::push_active_semantic_function(
                                function.0,
                            )?;
                            let _environment_roots = environment_roots;
                            executor
                                .invoke_request_async_in_environment(
                                    runmat_native_executor::NativeInvocationRequest {
                                        function,
                                        captures: call.captures,
                                        arguments: call.arguments,
                                        requested_outputs,
                                        runtime: selected,
                                        deoptimization: runmat_native_executor::deopt::DeoptimizationPolicy::default(),
                                        osr_target: None,
                                        workspace: None,
                                    },
                                    call.environment,
                                )
                                .await
                                .map_err(native_executor_error)
                        },
                    )
                    .await?;
                        execution.try_map_owned(
                            |execution| {
                                Ok(runmat_runtime::call::lexical::LexicalCallResult {
                                    outputs: normalize_outputs(
                                        execution.outputs,
                                        requested_outputs,
                                    )
                                    .map_err(|error| *error)?,
                                    captures: execution.captures,
                                })
                            },
                            |error| error.into_runtime_error(),
                        )
                    },
                )
            }),
        );

        let previous_resolver =
            runmat_runtime::user_functions::current_semantic_function_resolver();
        let previous_candidates =
            runmat_runtime::user_functions::current_semantic_name_candidates();
        let previous_scoped = runmat_runtime::user_functions::current_scoped_name_target_resolver();
        let resolver_program = Arc::new(program.clone());
        let scoped_program = Arc::clone(&resolver_program);
        let source_facts = Arc::new(
            functions
                .iter()
                .map(|info| (info.function, info.clone()))
                .collect::<std::collections::HashMap<_, _>>(),
        );
        let candidates: Arc<runmat_runtime::user_functions::SemanticNameCandidates> =
            Arc::new(move |name| {
                let mut candidates = Vec::new();
                // The active caller's already-issued private/retained target
                // precedes a public name, including when a nested selected
                // source service installed that target before this AOT guard.
                let caller = runmat_runtime::user_functions::current_active_semantic_function();
                let own_scoped = name_selection::select_scoped(&resolver_program, caller, name);
                let scoped = runmat_runtime::user_functions::current_scoped_name_target_resolver()
                    .and_then(|resolver| resolver(name))
                    .map(|target| target.function().0);
                let previous_offers = previous_candidates
                    .as_ref()
                    .map(|previous| previous(name))
                    .unwrap_or_default();
                let selected = scoped.or_else(|| {
                    name_selection::select(&resolver_program, caller, name)
                        .map(|function| function.0)
                });
                if let Some(function) = selected {
                    let own_source = own_scoped
                        .filter(|selected| selected.0 == function)
                        .and_then(|_| source_facts.get(&function).cloned());
                    let previous = (scoped.is_some() && own_source.is_none())
                        .then(|| {
                            previous_offers
                                .iter()
                                .find(|candidate| candidate.function == function)
                                .cloned()
                        })
                        .flatten();
                    candidates.push(previous.unwrap_or_else(|| {
                        runmat_runtime::user_functions::SemanticNameCandidate {
                            function,
                            source: if scoped.is_some() && own_source.is_none() {
                                None
                            } else {
                                own_source.or_else(|| source_facts.get(&function).cloned())
                            },
                            definition_owner: None,
                        }
                    }));
                }
                for candidate in previous_offers {
                    if !candidates
                        .iter()
                        .any(|existing| existing.same_definition(&candidate))
                    {
                        candidates.push(candidate);
                    }
                }
                if let Some(previous) = previous_resolver
                    .as_ref()
                    .and_then(|resolver| resolver(name))
                {
                    if !candidates
                        .iter()
                        .any(|candidate| candidate.function == previous)
                    {
                        candidates.push(runmat_runtime::user_functions::SemanticNameCandidate {
                            function: previous,
                            source: None,
                            definition_owner: None,
                        });
                    }
                }
                candidates
            });
        let resolver_candidates = Arc::clone(&candidates);
        let resolver = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
            Arc::new(move |name| {
                resolver_candidates(name)
                    .into_iter()
                    .next()
                    .map(|candidate| candidate.function)
            }),
        ))?;
        let name_candidates =
            runmat_runtime::user_functions::install_semantic_name_candidates(Some(candidates))?;
        let installed_external =
            external_sources::install(Rc::clone(&executor), program, runtime, &functions)?;
        let execution_source = installed_external.guard;
        let scoped_bindings = installed_external.scoped_bindings;
        let scoped = runmat_runtime::user_functions::install_scoped_name_target_resolver(Some(
            Rc::new(move |name| {
                let selected = name_selection::select_scoped(
                    &scoped_program,
                    runmat_runtime::user_functions::current_active_semantic_function(),
                    name,
                );
                let Some(selected) = selected else {
                    return previous_scoped.as_ref().and_then(|resolver| resolver(name));
                };
                if scoped_program.product.selected().is_some() {
                    let binding = scoped_bindings.get(&selected)?.clone();
                    Some(
                        runmat_runtime::user_functions::ScopedSemanticNameTarget::Retained(binding),
                    )
                } else {
                    Some(
                        runmat_runtime::user_functions::ScopedSemanticNameTarget::Published(
                            selected,
                        ),
                    )
                }
            }),
        ))?;
        let catalog = runmat_runtime::user_functions::install_source_function_catalog(Some(
            Arc::new(functions),
        ))?;
        Ok(Self {
            _semantic: semantic,
            _bound: bound,
            _lexical: lexical,
            _resolver: resolver,
            _scoped: scoped,
            _name_candidates: name_candidates,
            _catalog: catalog,
            _execution_source: execution_source,
        })
    }
}

fn isolated_runtime(
    runtime: runmat_runtime::context::RuntimeContext,
) -> runmat_runtime::context::RuntimeContext {
    let ports = runtime.service_ports().clone().without_workspace();
    install_acceleration_services(runtime.with_service_ports(ports))
}

async fn invoke_native_values(
    executor: Rc<runmat_native_executor::NativeExecutor>,
    function: FunctionId,
    arguments: runmat_runtime::call::RootedFunctionArguments,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> Result<runmat_runtime::execution::RootedValueSequence, runmat_runtime::RuntimeError> {
    let environment_roots = if let Some(callable) = arguments.callable() {
        let Value::Closure(closure) = callable.value() else {
            unreachable!("rooted callable arguments validated their source")
        };
        if closure.lexical_environment.as_ref() != lexical_environment.as_ref() {
            return Err(native_error(
                "selected callable lexical activation differs from native invocation",
            ));
        }
        None
    } else {
        lexical_environment
            .as_ref()
            .map(runmat_runtime::call::lexical::root_environment)
            .transpose()?
    };
    let runtime =
        super_constructor::bind_selected(&executor, function, runtime).map_err(|error| *error)?;
    let (runtime, placement) = runtime.native_invocation_context();
    let selected = runtime.clone();
    let execution = runmat_runtime::context::run_rooted_function_native_continuation(
        &runtime,
        placement,
        arguments,
        move |arguments| async move {
            let _active =
                runmat_runtime::user_functions::push_active_semantic_function(function.0)?;
            let _environment_roots = environment_roots;
            executor
                .invoke_rooted_function_arguments_async(
                    function,
                    arguments,
                    requested_outputs,
                    selected,
                    lexical_environment,
                )
                .await
                .map_err(native_executor_error)
        },
    )
    .await?;
    execution.try_map_owned(
        |execution| normalize_outputs(execution.outputs, requested_outputs).map_err(|error| *error),
        |error| error.into_runtime_error(),
    )
}

/// A standalone root owns the same existing placement session used by Core.
/// Supplied services and all nested context clones retain their exact authority.
pub(crate) fn install_acceleration_services(
    runtime: runmat_runtime::context::RuntimeContext,
) -> runmat_runtime::context::RuntimeContext {
    let runtime = if runtime.service_ports().placement().is_none() {
        let ports = runtime.service_ports().clone().with_placement(Rc::new(
            runmat_accelerate::placement::PlacementSession::default(),
        ));
        runtime.with_service_ports(ports)
    } else {
        runtime
    };
    runmat_accelerate::native_auto::install_preparation_service(runtime)
}

fn normalize_outputs(
    outputs: Vec<Value>,
    requested_outputs: usize,
) -> Result<ValueSequence, Box<runmat_runtime::RuntimeError>> {
    if outputs.len() != requested_outputs {
        return Err(Box::new(native_error(format!(
            "native execution produced {} outputs for a request of {requested_outputs}",
            outputs.len()
        ))));
    }
    Ok(if requested_outputs == 1 {
        ValueSequence::single(
            outputs
                .into_iter()
                .next()
                .expect("one output was validated"),
        )
        .map_err(|error| Box::new(native_error(error)))?
    } else {
        ValueSequence::comma_separated(outputs).map_err(|error| Box::new(native_error(error)))?
    })
}

fn native_error(error: impl std::fmt::Display) -> runmat_runtime::RuntimeError {
    runmat_runtime::build_runtime_error(error.to_string())
        .with_identifier("RunMat:AotNativeCall")
        .build()
}

fn native_executor_error(
    error: runmat_native_executor::NativeExecutorError,
) -> runmat_runtime::RuntimeError {
    match error {
        runmat_native_executor::NativeExecutorError::Runtime(original) => *original,
        runmat_native_executor::NativeExecutorError::SourceAdmission(error) => {
            source_admission_error(error)
        }
        other => native_error(other),
    }
}

fn source_admission_error(
    error: runmat_runtime::context::ProgramSourceSelectionError,
) -> runmat_runtime::RuntimeError {
    runmat_runtime::build_runtime_error(error.to_string())
        .with_identifier("RunMat:AotSourceAdmission")
        .with_source(error)
        .build()
}
