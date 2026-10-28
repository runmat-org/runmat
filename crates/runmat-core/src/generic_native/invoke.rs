use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    sync::Arc,
};

use runmat_gc::{Trace, Tracer};
use runmat_runtime::user_functions;
use runmat_types::FunctionId;
use runmat_value::{Value, ValueSequence};

use crate::ExecutableUnit;

#[path = "invoke/bound_constructor.rs"]
mod bound_constructor;
#[path = "invoke/native_semantic.rs"]
mod native_semantic;
#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "invoke/native_semantic_tests.rs"]
mod native_semantic_tests;

pub(crate) struct NativeExecution {
    pub outputs: ValueSequence,
    pub loop_backedges: BTreeMap<runmat_types::ProgramPointId, u64>,
    pub osr_entry: Option<runmat_types::ProgramPointId>,
    pub vectorized_regions: u64,
    pub workspace: Option<runmat_native_executor::execute::NativeWorkspaceSnapshot>,
    pub expression: Option<Value>,
}

pub(crate) type RootedNativeExecution = runmat_runtime::execution::RootedResult<NativeExecution>;

impl Trace for NativeExecution {
    fn trace(&self, tracer: &mut dyn Tracer) {
        self.outputs.trace(tracer);
        if let Some(workspace) = &self.workspace {
            for (_, value) in workspace.iter() {
                value.trace(tracer);
            }
        }
        if let Some(expression) = &self.expression {
            expression.trace(tracer);
        }
    }
}

pub(crate) struct NativeInvocation<'a> {
    pub published: runmat_jit::entry::PublishedEntry,
    pub osr: Option<(
        runmat_jit::entry::PublishedEntry,
        runmat_types::ProgramPointId,
    )>,
    pub preferred_function: Option<&'a str>,
    pub arguments: runmat_runtime::call::BuiltinArguments,
    pub lexical_environment: Option<runmat_value::LexicalEnvironment>,
    pub requested_outputs: usize,
    pub runtime: runmat_runtime::context::RuntimeContext,
    pub dynamic_admission: Option<runmat_vm::DynamicEvalAdmissionPort>,
    pub runtime_functions: &'a runmat_vm::FunctionRegistry,
    pub workspace: Option<runmat_native_executor::execute::NativeWorkspaceInput>,
}

pub(crate) async fn invoke(
    unit: &ExecutableUnit,
    invocation: NativeInvocation<'_>,
) -> Result<RootedNativeExecution, runmat_runtime::RuntimeError> {
    let NativeInvocation {
        published,
        osr,
        preferred_function,
        arguments,
        lexical_environment,
        requested_outputs,
        runtime,
        dynamic_admission,
        runtime_functions,
        workspace,
    } = invocation;
    runtime_functions
        .validate_selected_source_owners(&runtime)
        .map_err(|message| super::error::stage("NativeSourceOwner", message))?;
    runmat_vm::prepare_native_execution_metadata(unit.bytecode())?;
    let function = preferred_function
        .map(|name| {
            unit.native_function_id(name)
                .ok_or_else(|| super::error::stage("NativeProcedure", name))
        })
        .transpose()?
        .unwrap_or(published.entrypoint);
    let selected_function = function;
    let selected_body = runtime_functions.get_current(function);
    let runtime = if selected_body.is_some_and(|body| {
        body.class_method_owner.as_ref().is_some_and(|owner| {
            owner.role == runmat_types::ClassMethodRole::Constructor && !owner.is_static
        })
    }) {
        selected_body
            .and_then(|body| body.selected_class_product.as_ref())
            .ok_or_else(|| {
                super::error::stage(
                    "NativeClassProduct",
                    "selected native constructor lacks its exact class product",
                )
            })?
            .invocation_runtime(runtime)?
    } else {
        runtime
    };
    let runtime = runtime.with_constructor_receiver_for_method(
        unit.mir()
            .functions
            .get(&selected_function)
            .and_then(|metadata| metadata.class_method_owner.as_ref()),
    )?;
    let runtime = if let Some(owner) = selected_body
        .and_then(|body| body.class_method_owner.as_ref())
        .filter(|owner| owner.role == runmat_types::ClassMethodRole::Constructor)
    {
        let binding = runtime_functions
            .exact_bound_binding(function)
            .ok_or_else(|| {
                super::error::stage(
                    "NativeClassProduct",
                    "constructor frame lacks its exact entered binding",
                )
            })?;
        runtime.with_admitted_constructor_frame(owner, binding)?
    } else {
        runtime
    };
    let registry = Rc::new(runtime_functions.clone());
    let runtime = if runtime.has_selected_graph_function_resolver() {
        runtime
    } else {
        crate::session::bind_function_resolvers(
            runtime,
            unit,
            Rc::clone(&registry),
            unit.original_static_targets().to_vec(),
        )
        .map_err(|error| super::error::stage("NativeStaticImport", error))?
    };
    let native_functions = unit.mir().bodies.keys().copied().collect::<BTreeSet<_>>();
    let previous_invoker = user_functions::current_semantic_function_invoker();
    let nested_executor = Rc::clone(&published.executor);
    let nested_runtime = runtime.native_invocation_context().0;
    let semantic_native_functions = native_functions.clone();
    let semantic_unit = unit.clone();
    let semantic_registry = Rc::clone(&registry);
    let semantic_dynamic_admission = dynamic_admission.clone();
    let invoker = user_functions::install_local_semantic_function_invoker(Rc::new(
        move |selected, function, arguments, requested_outputs| {
            let lexical_environment = arguments.callable().and_then(|callable| {
                let Value::Closure(closure) = callable.value() else {
                    return None;
                };
                closure.lexical_environment.clone()
            });
            let previous_invoker = previous_invoker.clone();
            let executor = Rc::clone(&nested_executor);
            let same_owner = selected
                .as_ref()
                .is_none_or(|context| context.local_identity() == nested_runtime.local_identity());
            let runtime = if same_owner {
                selected.map_or_else(
                    || nested_runtime.clone(),
                    |context| {
                        nested_runtime
                            .clone()
                            .try_with_inherited_callback_invocation_authority_from(&context)
                            .expect("callback Runtime owner checked before authority projection")
                    },
                )
            } else {
                nested_runtime.clone()
            };
            let unit = semantic_unit.clone();
            let registry = Rc::clone(&semantic_registry);
            let dynamic_admission = semantic_dynamic_admission.clone();
            let method_owner = unit
                .mir()
                .functions
                .get(&runmat_hir::FunctionId(function))
                .and_then(|metadata| metadata.class_method_owner.clone())
                .or_else(|| {
                    registry
                        .get(runmat_hir::FunctionId(function))
                        .and_then(|function| function.class_method_owner.clone())
                });
            let admission_runtime = runtime.clone();
            let runtime = runtime.with_constructor_receiver_for_method(method_owner.as_ref());
            let is_native = semantic_native_functions.contains(&FunctionId(function));
            if same_owner
                && registry
                    .requires_selected_constructor_admission(runmat_hir::FunctionId(function))
            {
                return user_functions::admit_user_function_with(
                    Some(&admission_runtime),
                    move || async move {
                        let runtime = runtime?;
                        let binding = registry
                            .exact_bound_binding(runmat_hir::FunctionId(function))
                            .ok_or_else(|| {
                                super::error::stage(
                                "ConstructorOwnerInvalid",
                                "selected semantic constructor lacks its exact definition lease",
                            )
                            })?;
                        user_functions::try_call_rooted_bound_function_with_environment_in_context(
                            &runtime,
                            binding,
                            arguments,
                            requested_outputs,
                            lexical_environment,
                        )
                        .await
                        .ok_or_else(|| {
                            super::error::stage(
                                "UnavailableBoundFunction",
                                "selected semantic constructor has no bound executor",
                            )
                        })?
                    },
                );
            }
            user_functions::admit_user_function_with(Some(&admission_runtime), move || async move {
                if !same_owner {
                    return Err(super::error::stage(
                        "NativeCallbackOwner",
                        "semantic callback selected another native session owner",
                    ));
                }
                let runtime = runtime?;
                let runtime = isolated_procedure_runtime(runtime);
                if is_native {
                    let execution_sources = registry.execution_source_resolver_in(&runtime);
                    let runtime = super::super_constructor::bind_selected(
                        runtime,
                        &registry,
                        runmat_hir::FunctionId(function),
                        &execution_sources,
                    )?;
                    let function = runmat_hir::FunctionId(function);
                    native_semantic::invoke_rooted(
                        &unit,
                        executor,
                        native_semantic::RootedInvocation {
                            function,
                            arguments,
                            requested_outputs,
                            runtime,
                            lexical_environment,
                            dynamic_admission,
                        },
                    )
                    .await
                } else if registry.get(runmat_hir::FunctionId(function)).is_some() {
                    // A retained session closure need not belong to this
                    // unit's native MIR. Its registry definition remains
                    // authoritative; arguments still include its captures.
                    invoke_vm_with_rooted_admission(
                        function,
                        arguments,
                        requested_outputs,
                        &registry,
                        runtime,
                        lexical_environment,
                        dynamic_admission,
                    )
                    .await
                } else if let Some(previous_invoker) = previous_invoker {
                    runtime
                        .scope(previous_invoker(
                            Some(runtime.clone()),
                            function,
                            arguments,
                            requested_outputs,
                        )?)
                        .await
                } else {
                    Err(super::error::stage(
                        "UndefinedFunction",
                        format!("semantic function {function} is unavailable"),
                    ))
                }
            })
        },
    ))?;

    let previous_bound_invoker = user_functions::current_bound_function_invoker();
    let bound_registry = Rc::clone(&registry);
    let bound_dynamic_admission = dynamic_admission.clone();
    let bound_executor = Rc::clone(&published.executor);
    let bound_unit = unit.clone();
    let bound_native_functions = native_functions.clone();
    let bound_program = unit
        .revision()
        .program_revision
        .identity_digest()
        .map_err(|error| super::error::stage("NativeProgramIdentity", error))?;
    let bound_invoker = user_functions::install_local_bound_function_invoker(Rc::new(
        move |runtime, binding, arguments, requested_outputs, lexical_environment| {
            let previous = previous_bound_invoker.clone();
            let registry = Rc::clone(&bound_registry);
            let dynamic_admission = bound_dynamic_admission.clone();
            let executor = Rc::clone(&bound_executor);
            let unit = bound_unit.clone();
            let is_native = bound_native_functions.contains(&FunctionId(binding.function));
            // Select the exact body before the sibling future can be queued.
            // A later same-name class publication must not replace its
            // invocation-owned class defaults or constructor role.
            let selected_body = match &binding.authority {
                runmat_value::BoundFunctionAuthority::Local { .. } => binding
                    .authority
                    .local_lease(binding.function)
                    .and_then(|lease| {
                        registry.get_owned(runmat_hir::FunctionId(binding.function), lease)
                    }),
                runmat_value::BoundFunctionAuthority::Portable { origin } => registry
                    .get_current(runmat_hir::FunctionId(binding.function))
                    .filter(|body| body.binding_origin == Some(*origin)),
                runmat_value::BoundFunctionAuthority::SelectedNative { .. } => None,
            };
            let native_selected_body = is_native
                && selected_body.is_some_and(|body| {
                    body.binding_origin
                        .is_some_and(|origin| origin.program == bound_program)
                        && unit
                            .functions()
                            .get_current(runmat_hir::FunctionId(binding.function))
                            .is_some_and(|compiled| {
                                Arc::ptr_eq(&compiled.definition_owner, &body.definition_owner)
                            })
                });
            let constructor_owner = selected_body
                .and_then(|body| body.class_method_owner.clone())
                .filter(|owner| owner.role == runmat_types::ClassMethodRole::Constructor);
            let selected_product =
                selected_body.and_then(|body| body.selected_class_product.clone());
            let (runtime, super_body_owner) = runtime.take_retained_super_body_for(&binding)?;
            runmat_runtime::execution::admit_executable_sequence_with(
                &runtime.clone(),
                move || async move {
                    if let Some(expected) = &super_body_owner {
                        if constructor_owner.as_ref() != Some(expected)
                            || expected.is_static
                            || expected.role != runmat_types::ClassMethodRole::Constructor
                        {
                            return Err(super::error::stage(
                                "NativeConstructorOwner",
                                "selected superclass body disagrees with its admitted owner",
                            ));
                        }
                    }
                    let function = runmat_hir::FunctionId(binding.function);
                    let (view, current_definition) = match &binding.authority {
                        runmat_value::BoundFunctionAuthority::Local {
                            definition,
                            dependencies,
                            ..
                        } => {
                            let current = std::iter::once(definition)
                                .chain(dependencies.iter())
                                .all(|lease| {
                                    registry
                                        .get(runmat_hir::FunctionId(lease.function()))
                                        .is_some_and(|body| lease.matches(&body.definition_owner))
                                });
                            (
                                (!current)
                                    .then(|| registry.owned_view(function, &binding.authority))
                                    .flatten(),
                                current,
                            )
                        }
                        runmat_value::BoundFunctionAuthority::Portable { origin } => (
                            None,
                            registry
                                .get(function)
                                .is_some_and(|body| body.binding_origin == Some(*origin)),
                        ),
                        runmat_value::BoundFunctionAuthority::SelectedNative { .. } => {
                            (None, false)
                        }
                    };
                    if let Some(owner) = constructor_owner.filter(|_| {
                        (super_body_owner.is_some()
                            || matches!(
                                &binding.authority,
                                runmat_value::BoundFunctionAuthority::Local { .. }
                            ))
                            && (current_definition || view.is_some())
                    }) {
                        return bound_constructor::invoke(
                            bound_constructor::Invocation {
                                runtime,
                                binding,
                                owner,
                                product: selected_product,
                                registry: view.map(Rc::new).unwrap_or_else(|| Rc::clone(&registry)),
                                unit,
                                executor,
                                arguments,
                                requested_outputs,
                                lexical_environment,
                                is_native: current_definition && native_selected_body,
                                dynamic_admission,
                            },
                            super_body_owner.is_some(),
                        )
                        .await;
                    }
                    if super_body_owner.is_some() {
                        return Err(super::error::stage(
                            "NativeConstructorOwner",
                            "selected superclass body has no exact executable definition",
                        ));
                    }
                    if current_definition {
                        let runtime = isolated_procedure_runtime(runtime);
                        if native_selected_body {
                            // This binding is owned here. Admit its one-shot
                            // receiver intent before entering native code;
                            // ordinary siblings must not inherit caller proof.
                            let (runtime, selected_method) =
                                registry.prepare_selected_receiver_method(runtime, &binding)?;
                            let runtime = if selected_method {
                                runtime
                            } else {
                                runtime.without_selected_object_class()
                            };
                            let method_owner = unit
                                .mir()
                                .functions
                                .get(&function)
                                .and_then(|metadata| metadata.class_method_owner.as_ref());
                            let runtime =
                                runtime.with_constructor_receiver_for_method(method_owner)?;
                            let execution_sources = registry.execution_source_resolver_in(&runtime);
                            let runtime = super::super_constructor::bind_selected(
                                runtime,
                                &registry,
                                function,
                                &execution_sources,
                            )?;
                            let selected = runtime.clone();
                            return selected
                                .scope(native_semantic::invoke_rooted(
                                    &unit,
                                    executor,
                                    native_semantic::RootedInvocation {
                                        function,
                                        arguments,
                                        requested_outputs,
                                        runtime,
                                        lexical_environment,
                                        dynamic_admission,
                                    },
                                ))
                                .await;
                        }
                        return invoke_vm_with_rooted_admission(
                            binding.function,
                            arguments,
                            requested_outputs,
                            &registry,
                            runtime,
                            lexical_environment,
                            dynamic_admission,
                        )
                        .await;
                    }
                    if let Some(view) = view {
                        let runtime = isolated_procedure_runtime(runtime);
                        return invoke_vm_with_rooted_admission(
                            binding.function,
                            arguments,
                            requested_outputs,
                            &view,
                            runtime,
                            lexical_environment,
                            dynamic_admission,
                        )
                        .await;
                    }
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
                    Err(super::error::stage(
                        "UnavailableBoundFunction",
                        "bound function definition is unavailable in this execution owner",
                    ))
                },
            )
        },
    ))?;

    let previous_external_invoker = user_functions::current_external_function_invoker();
    let external_registry = Rc::clone(&registry);
    let external_unit = unit.clone();
    let external_dynamic_admission = dynamic_admission.clone();
    let external_runtime = runtime.native_invocation_context().0;
    let external_invoker =
        user_functions::install_local_external_function_invoker(Rc::new(move |selected, call| {
            let previous = previous_external_invoker.clone();
            let registry = Rc::clone(&external_registry);
            let unit = external_unit.clone();
            let dynamic_admission = external_dynamic_admission.clone();
            let same_owner = selected.as_ref().is_none_or(|context| {
                context.local_identity() == external_runtime.local_identity()
            });
            let runtime = if same_owner {
                selected.map_or_else(
                    || external_runtime.clone(),
                    |context| {
                        external_runtime
                            .clone()
                            .try_with_inherited_callback_invocation_authority_from(&context)
                            .expect("callback Runtime owner checked before authority projection")
                    },
                )
            } else {
                external_runtime.clone()
            };
            let admission_runtime = runtime.clone();
            user_functions::admit_user_function_with(Some(&admission_runtime), move || async move {
                if !same_owner {
                    return Err(super::error::stage(
                        "NativeCallbackOwner",
                        "external callback selected another native session owner",
                    ));
                }
                let runtime = isolated_procedure_runtime(runtime);
                if unit.owns_callable(FunctionId(call.function)) {
                    return invoke_vm_with_rooted_admission(
                        call.function,
                        call.arguments,
                        call.requested_outputs,
                        &registry,
                        runtime,
                        None,
                        dynamic_admission,
                    )
                    .await;
                }
                if let Some(previous) = previous {
                    return runtime.scope(previous(Some(runtime.clone()), call)?).await;
                }
                Err(super::error::stage(
                    "UndefinedFunction",
                    format!("external function '{}' is unavailable", call.display_name),
                ))
            })
        }));

    let previous_lexical_invoker = user_functions::current_lexical_function_invoker();
    let lexical_executor = Rc::clone(&published.executor);
    let lexical_runtime = runtime.native_invocation_context().0;
    let lexical_native_functions = native_functions.clone();
    let lexical_unit = unit.clone();
    let lexical_dynamic_admission = dynamic_admission.clone();
    let lexical_invoker =
        user_functions::install_local_lexical_function_invoker(Rc::new(move |selected, call| {
            let call_roots = runmat_runtime::execution::RootedValueLease::for_traced(&call)
                .map_err(|error| error.into_runtime_error());
            let previous = previous_lexical_invoker.clone();
            let executor = Rc::clone(&lexical_executor);
            let same_owner = selected
                .as_ref()
                .is_none_or(|context| context.local_identity() == lexical_runtime.local_identity());
            let runtime = if same_owner {
                selected.map_or_else(
                    || lexical_runtime.clone(),
                    |context| {
                        lexical_runtime
                            .clone()
                            .try_with_inherited_callback_invocation_authority_from(&context)
                            .expect("callback Runtime owner checked before authority projection")
                    },
                )
            } else {
                lexical_runtime.clone()
            };
            let unit = lexical_unit.clone();
            let dynamic_admission = lexical_dynamic_admission.clone();
            let method_owner = unit
                .mir()
                .functions
                .get(&runmat_hir::FunctionId(call.function))
                .and_then(|metadata| metadata.class_method_owner.as_ref());
            let admission_runtime = runtime.clone();
            let runtime = runtime.with_constructor_receiver_for_method(method_owner);
            let is_native = lexical_native_functions.contains(&FunctionId(call.function));
            user_functions::admit_lexical_function_with(
                Some(&admission_runtime),
                move || async move {
                    let _call_roots = call_roots?;
                    if !same_owner {
                        return Err(super::error::stage(
                            "NativeCallbackOwner",
                            "lexical callback selected another native session owner",
                        ));
                    }
                    let runtime = runtime?;
                    let runtime = isolated_procedure_runtime(runtime);
                    if !is_native {
                        if let Some(previous) = previous {
                            return runtime.scope(previous(Some(runtime.clone()), call)?).await;
                        }
                        return Err(super::error::stage(
                            "UndefinedFunction",
                            format!("lexical function {} is unavailable", call.function),
                        ));
                    }
                    let function = runmat_hir::FunctionId(call.function);
                    let execution = super::deopt::invoke(
                        &unit,
                        super::deopt::DeoptimizationRequest {
                            executor,
                            function,
                            lexical_environment: call.environment,
                            inputs: super::deopt::Inputs::Source {
                                captures: call.captures,
                                arguments: call.arguments,
                            },
                            requested_outputs: call.requested_outputs,
                            runtime,
                            dynamic_admission,
                            osr: None,
                            workspace: None,
                        },
                    )
                    .await?;
                    execution.try_map_owned(
                        |execution| {
                            Ok(runmat_runtime::call::lexical::LexicalCallResult {
                                outputs: normalize_outputs(
                                    execution.outputs,
                                    call.requested_outputs,
                                )?,
                                captures: execution.captures,
                            })
                        },
                        |error| error.into_runtime_error(),
                    )
                },
            )
        }));

    let previous_resolver = user_functions::current_semantic_function_resolver();
    let resolver_names = registry.published_name_resolution_view();
    let resolver =
        user_functions::install_semantic_function_resolver(Some(Arc::new(move |name| {
            resolver_names
                .resolve_name(name)
                .map(|function| function.0)
                .or_else(|| {
                    previous_resolver
                        .as_ref()
                        .and_then(|resolver| resolver(name))
                })
        })))?;
    let execution_sources = registry.execution_source_resolver_in(&runtime);
    let execution_source =
        user_functions::install_execution_source_resolver(Some(Rc::clone(&execution_sources)))?;
    let mut source_functions = registry
        .functions
        .values()
        .filter_map(|function| {
            function
                .source_id
                .and_then(|source| source.resolve_in(&runtime))
                .map(|source_id| user_functions::SourceFunctionInfo {
                    source_id,
                    name: function.display_name.clone(),
                    function: function.function.0,
                    role: function.source_role,
                    binding_origin: function.binding_origin,
                    source_file: function.source_file.clone(),
                    definition_owners: None,
                })
        })
        .collect::<Vec<_>>();
    source_functions.sort_by_key(|function| function.function);
    let catalog =
        user_functions::install_source_function_catalog(Some(Arc::new(source_functions)))?;
    let active = user_functions::push_active_semantic_function(function.0)?;
    // The compiler deliberately omits the synthetic script entrypoint from
    // the semantic FunctionRegistry. It cannot own a superclass constructor;
    // every real function still has to pass the exact retained-owner check.
    let runtime = if Some(selected_function) == unit.script_entrypoint()
        && unit
            .mir()
            .functions
            .get(&selected_function)
            .is_some_and(|metadata| metadata.kind == runmat_hir::FunctionKind::SyntheticEntrypoint)
        && registry.get(selected_function).is_none()
    {
        runtime
    } else {
        super::super_constructor::bind_selected(
            runtime,
            &registry,
            selected_function,
            &execution_sources,
        )?
    };
    let osr = osr
        .map(|(target, point)| {
            runmat_native_executor::osr::OsrTarget::new(target.executor, function, point)
                .map_err(|error| super::error::stage("NativeOsrTarget", error))
        })
        .transpose()?;
    let execution = super::deopt::invoke(
        unit,
        super::deopt::DeoptimizationRequest {
            executor: published.executor,
            function,
            lexical_environment,
            inputs: super::deopt::Inputs::Retained(arguments.into_function_arguments()),
            requested_outputs,
            runtime,
            dynamic_admission,
            osr,
            workspace,
        },
    )
    .await;
    drop(active);
    drop(execution_source);
    drop(catalog);
    drop(resolver);
    drop(lexical_invoker);
    drop(external_invoker);
    drop(invoker);
    drop(bound_invoker);
    let execution = execution?;
    execution.try_map_owned(
        |execution| {
            Ok(NativeExecution {
                outputs: normalize_outputs(execution.outputs, requested_outputs)?,
                loop_backedges: execution.loop_backedges,
                osr_entry: execution.osr_entry,
                vectorized_regions: execution.vectorized_regions,
                workspace: execution.workspace,
                expression: execution.expression,
            })
        },
        |error| error.into_runtime_error(),
    )
}

async fn invoke_vm_with_rooted_admission(
    function: usize,
    arguments: runmat_runtime::call::RootedFunctionArguments,
    requested_outputs: usize,
    registry: &runmat_vm::FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
    dynamic_admission: Option<runmat_vm::DynamicEvalAdmissionPort>,
) -> Result<runmat_runtime::execution::RootedValueSequence, runmat_runtime::RuntimeError> {
    runmat_vm::invoke_semantic_function_with_rooted_arguments_and_dynamic_admission_in_context(
        function,
        arguments,
        requested_outputs,
        registry,
        runtime,
        lexical_environment,
        dynamic_admission,
    )
    .await
}

fn isolated_procedure_runtime(
    runtime: runmat_runtime::context::RuntimeContext,
) -> runmat_runtime::context::RuntimeContext {
    let runtime = runtime.inherit_same_session_environment();
    let ports = runtime.service_ports().clone().without_workspace();
    runtime.with_service_ports(ports)
}

fn normalize_outputs(
    outputs: Vec<Value>,
    requested_outputs: usize,
) -> Result<ValueSequence, runmat_runtime::RuntimeError> {
    if outputs.len() != requested_outputs {
        return Err(super::error::stage(
            "NativeOutputArity",
            format!(
                "native execution produced {} outputs for a request of {requested_outputs}",
                outputs.len()
            ),
        ));
    }
    Ok(if requested_outputs == 1 {
        ValueSequence::single(
            outputs
                .into_iter()
                .next()
                .expect("one output was validated"),
        )
        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?
    } else {
        ValueSequence::comma_separated(outputs)
            .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?
    })
}

#[cfg(test)]
mod native_failure_tests {
    use futures::executor::block_on;
    use std::error::Error as _;

    use crate::{ExecutableSource, InvocationControl, ProcedureInvocation, RunMatSession};

    const SOURCE: &str = r#"
function output = first(value)
    error('RunMat:FirstInvocation', 'the first invocation failed');
    output = struct('second', @(next) value + next);
end
function output = later()
    error('RunMat:LaterArgument', 'later preparation ran first');
    output = 3;
end
s = struct('first', @first);
C = {7};
s.first(C{:}).second(later());
"#;

    #[test]
    fn direct_native_entry_fails_before_later_dotted_argument_preparation() {
        runmat_gc::gc_test_context(|| {
            let mut session = RunMatSession::with_options(false, false).unwrap();
            let unit = block_on(session.compile_executable_unit(
                ExecutableSource::new("native-dotted-error@1", "dotted_error.m", SOURCE),
                None,
            ))
            .unwrap();
            // ForcedGenericNative bypasses the interpreter and tiering fallback.
            // Its failure comes from the compiled native entry itself.
            let result = block_on(session.invoke_executable(
                &unit,
                ProcedureInvocation::entrypoint(),
                &InvocationControl::default().force_generic_native(),
            ));
            assert_eq!(
                result.err().and_then(|error| match error {
                    crate::RunError::Runtime(error) => error.identifier().map(str::to_owned),
                    _ => None,
                }),
                Some("RunMat:FirstInvocation".to_string()),
            );
        });
    }

    pub(super) fn forced_native_failure(
        source_id: &str,
        program: &str,
    ) -> runmat_runtime::RuntimeError {
        let mut session = RunMatSession::with_options(false, false).expect("session init");
        session.set_compat_mode(runmat_parser::CompatMode::RunMat);
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(source_id, "native_exception.m", program),
            None,
        ))
        .expect("compile native exception fixture");
        let result = block_on(session.invoke_executable(
            &unit,
            ProcedureInvocation::entrypoint(),
            &InvocationControl::default().force_generic_native(),
        ));
        match result {
            Err(crate::RunError::Runtime(error)) => error,
            Err(other) => panic!("expected runtime error from native executor: {other:?}"),
            Ok(_) => panic!("native error must propagate"),
        }
    }

    #[test]
    fn native_direct_and_caught_rethrow_preserve_source_less_origin() {
        runmat_gc::gc_test_context(|| {
            const DIRECT: &str =
                "function native_exception()\nerror('RunMat:NativeOrigin', 'boom');\nend\n";
            const CAUGHT: &str = "function native_exception()\ntry\n error('RunMat:NativeOrigin', 'boom');\ncatch e\n rethrow(e);\nend\nend\n";
            let direct = forced_native_failure("native-direct-error@1", DIRECT);
            let reraised = forced_native_failure("native-rethrow-error@1", CAUGHT);
            for error in [&direct, &reraised] {
                assert_eq!(error.identifier(), Some("RunMat:NativeOrigin"));
                assert_eq!(error.message(), "boom");
                assert_eq!(
                    error.context.call_frames.len(),
                    1,
                    "rethrow must not add an origin frame"
                );
                assert_eq!(error.context.call_frames[0].function, "native_exception");
                let span = error.span.as_ref().expect("native source span");
                assert!(!span.is_empty());
                let frame_span = error.context.call_frames[0]
                    .span
                    .expect("native frame span");
                assert!(frame_span.1 >= frame_span.0);
            }
            let origin_call = CAUGHT.find("error(").expect("origin call text");
            let rethrow_call = CAUGHT.find("rethrow(").expect("rethrow call text");
            let origin_span = reraised.span.as_ref().expect("retained origin span");
            let origin_end = origin_span.offset() + origin_span.len();
            assert!(origin_span.offset() <= origin_call && origin_call < origin_end);
            assert!(
                origin_end <= rethrow_call,
                "rethrow must not replace the original span"
            );
            let frame_span = reraised.context.call_frames[0]
                .span
                .expect("origin frame span");
            assert!(frame_span.0 <= origin_call && origin_call < frame_span.1);
            assert!(
                frame_span.1 <= rethrow_call,
                "rethrow must not replace the origin frame"
            );
            let mut direct_original = &direct;
            while let Some(shared) = direct_original.source().and_then(|source| {
                source.downcast_ref::<runmat_runtime::runtime_error::SharedRuntimeErrorSource>()
            }) {
                direct_original = shared.original();
            }
            assert!(
                direct_original.source.is_none(),
                "direct source-less origin remains source-less"
            );
            let mut original = &reraised;
            while let Some(shared) = original.source().and_then(|source| {
                source.downcast_ref::<runmat_runtime::runtime_error::SharedRuntimeErrorSource>()
            }) {
                original = shared.original();
            }
            assert!(
                original.source.is_none(),
                "source-less original stays source-less"
            );
            assert_eq!(reraised.context.phase, direct.context.phase);
        });
    }

    #[test]
    fn native_caught_rethrow_keeps_downcastable_format_failure_cause() {
        runmat_gc::gc_test_context(|| {
            const WITH_CAUSE: &str = "function native_exception()\ntry\n error('bad %q', 1);\ncatch e\n rethrow(e);\nend\nend\n";
            let reraised = forced_native_failure("native-rethrow-cause@1", WITH_CAUSE);
            assert_eq!(reraised.identifier(), Some("RunMat:error"));
            assert_eq!(reraised.context.call_frames.len(), 1);
            assert_eq!(reraised.context.call_frames[0].function, "native_exception");
            let origin_call = WITH_CAUSE.find("error(").expect("origin call text");
            let rethrow_call = WITH_CAUSE.find("rethrow(").expect("rethrow call text");
            let span = reraised.span.as_ref().expect("retained cause span");
            assert!(span.offset() <= origin_call && origin_call < span.offset() + span.len());
            assert!(span.offset() + span.len() <= rethrow_call);
            let mut original = &reraised;
            while let Some(shared) = original.source().and_then(|source| {
                source.downcast_ref::<runmat_runtime::runtime_error::SharedRuntimeErrorSource>()
            }) {
                original = shared.original();
            }
            assert!(original
                .source()
                .is_some_and(|source| source.is::<runmat_runtime::RuntimeError>()));
        });
    }
}

#[cfg(test)]
#[path = "invoke/native_catch_tests.rs"]
mod native_catch_tests;
