use std::{collections::BTreeMap, rc::Rc};

use runmat_native_executor::{
    deopt::{DeoptimizationPolicy, MaterializedFrame, ResumeTarget},
    execute::{NativeExecution, NativeInvocation, NativeInvocationStep, RootedNativeExecution},
    NativeExecutor,
};
use runmat_value::Value;

use crate::ExecutableUnit;

mod inputs;
pub(super) use inputs::Inputs;

#[cfg(all(test, not(target_arch = "wasm32")))]
thread_local! {
    static NATIVE_LEXICAL_COMPLETIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(super) fn reset_native_lexical_completions() {
    NATIVE_LEXICAL_COMPLETIONS.with(|count| count.set(0));
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(super) fn native_lexical_completions() -> usize {
    NATIVE_LEXICAL_COMPLETIONS.with(std::cell::Cell::get)
}

struct DeoptimizationInvocation {
    executor: Rc<NativeExecutor>,
    function: runmat_types::FunctionId,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
    inputs: Inputs,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
    dynamic_admission: Option<runmat_vm::DynamicEvalAdmissionPort>,
    policy: DeoptimizationPolicy,
    osr: Option<runmat_native_executor::osr::OsrTarget>,
    workspace: Option<runmat_native_executor::execute::NativeWorkspaceInput>,
}

pub(super) struct DeoptimizationRequest {
    pub executor: Rc<NativeExecutor>,
    pub function: runmat_types::FunctionId,
    pub lexical_environment: Option<runmat_value::LexicalEnvironment>,
    pub inputs: Inputs,
    pub requested_outputs: usize,
    pub runtime: runmat_runtime::context::RuntimeContext,
    pub dynamic_admission: Option<runmat_vm::DynamicEvalAdmissionPort>,
    pub osr: Option<runmat_native_executor::osr::OsrTarget>,
    pub workspace: Option<runmat_native_executor::execute::NativeWorkspaceInput>,
}

pub(super) async fn invoke(
    unit: &ExecutableUnit,
    request: DeoptimizationRequest,
) -> Result<RootedNativeExecution, runmat_runtime::RuntimeError> {
    let runtime = request.runtime.clone();
    let cleanup = runtime.function_cleanup_scope()?;
    let (result, cleanup) = cleanup.run(&runtime, invoke_inner(unit, request)).await?;
    runmat_runtime::runtime_error::complete_with_cleanup(result, cleanup)
}

async fn invoke_inner(
    unit: &ExecutableUnit,
    request: DeoptimizationRequest,
) -> Result<RootedNativeExecution, runmat_runtime::RuntimeError> {
    let roots = match &request.inputs {
        Inputs::Source { .. } => Some(
            runmat_runtime::execution::RootedValueLease::for_values(
                request.inputs.source_values().chain(
                    request
                        .workspace
                        .iter()
                        .flat_map(|workspace| workspace.profile_values()),
                ),
            )
            .map_err(|error| error.into_runtime_error())?,
        ),
        Inputs::Retained(_) => None,
    };
    let environment_roots = request
        .lexical_environment
        .as_ref()
        .map(runmat_runtime::call::lexical::root_environment)
        .transpose()?;
    let (runtime, placement) = request.runtime.native_invocation_context();
    let policy = DeoptimizationPolicy {
        target: if request.workspace.is_some() {
            ResumeTarget::GenericNative
        } else {
            ResumeTarget::Interpreter
        },
        ..DeoptimizationPolicy::default()
    };
    let selected = runtime.clone();
    let make_invocation = move |inputs| DeoptimizationInvocation {
        executor: request.executor,
        function: request.function,
        lexical_environment: request.lexical_environment,
        inputs,
        requested_outputs: request.requested_outputs,
        runtime: selected,
        dynamic_admission: request.dynamic_admission,
        policy,
        osr: request.osr,
        workspace: request.workspace,
    };
    match request.inputs {
        Inputs::Retained(arguments) => {
            let unit = unit.clone();
            runmat_runtime::context::run_rooted_function_native_continuation(
                &runtime,
                placement,
                arguments,
                move |arguments| async move {
                    let _environment_roots = environment_roots;
                    invoke_with_policy(&unit, make_invocation(Inputs::Retained(arguments))).await
                },
            )
            .await
        }
        source @ Inputs::Source { .. } => {
            let invocation = make_invocation(source);
            let roots = roots.expect("source ingress acquired its own registration");
            if placement.is_direct() {
                let _roots = roots;
                let _environment_roots = environment_roots;
                return runtime.scope(invoke_with_policy(unit, invocation)).await;
            }
            let unit = unit.clone();
            runmat_runtime::context::run_native_continuation(
                &runtime,
                placement,
                roots,
                async move {
                    let _environment_roots = environment_roots;
                    invoke_with_policy(&unit, invocation).await
                },
            )
            .await
        }
    }
}
async fn invoke_with_policy(
    unit: &ExecutableUnit,
    invocation: DeoptimizationInvocation,
) -> Result<RootedNativeExecution, runmat_runtime::RuntimeError> {
    let DeoptimizationInvocation {
        executor,
        function,
        lexical_environment,
        inputs,
        requested_outputs,
        runtime,
        dynamic_admission,
        policy,
        osr,
        workspace,
    } = invocation;
    if workspace.is_some() && (inputs.has_captures() || lexical_environment.is_some()) {
        return Err(super::error::stage(
            "NativeWorkspace",
            "interactive entrypoints cannot carry lexical captures or an activation",
        ));
    }
    // Direct internal users (including deoptimization tests) retain their
    // activation independently of the outer continuation admission lease.
    let _environment_roots = lexical_environment
        .as_ref()
        .map(runmat_runtime::call::lexical::root_environment)
        .transpose()?;
    let had_lexical_environment = lexical_environment.is_some();
    let invocation = match inputs {
        Inputs::Source {
            captures,
            arguments,
        } => executor
            .begin_request_with_environment(
                runmat_native_executor::NativeInvocationRequest {
                    function,
                    captures,
                    arguments,
                    requested_outputs,
                    runtime: runtime.clone(),
                    deoptimization: policy,
                    osr_target: osr,
                    workspace,
                },
                lexical_environment,
            )
            .map_err(super::error::from_native_executor_error)?,
        Inputs::Retained(arguments) => executor
            .begin_rooted_request_async(runmat_native_executor::RootedNativeInvocationRequest {
                function,
                arguments,
                requested_outputs,
                runtime: runtime.clone(),
                lexical_environment,
                deoptimization: policy,
                osr_target: osr,
                workspace,
            })
            .await
            .map_err(super::error::from_native_executor_error)?,
    };
    drive_prepared_invocation(
        unit,
        invocation,
        requested_outputs,
        runtime,
        dynamic_admission,
        had_lexical_environment,
    )?
    .await
}

pub(super) fn drive_prepared_invocation(
    unit: &ExecutableUnit,
    invocation: NativeInvocation,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
    dynamic_admission: Option<runmat_vm::DynamicEvalAdmissionPort>,
    had_lexical_environment: bool,
) -> Result<
    runmat_runtime::execution::RuntimeContinuationFuture<'static, RootedNativeExecution>,
    runmat_runtime::RuntimeError,
> {
    let unit = unit.clone();
    let admission_runtime = runtime.clone();
    let cleanup = runtime.function_cleanup_scope()?;
    runmat_runtime::execution::admit_rooted_result_with(&admission_runtime, move || async move {
        let cleanup_runtime = runtime.clone();
        let (result, cleanup) = cleanup
            .run(
                &cleanup_runtime,
                drive_prepared_invocation_inner(
                    unit,
                    invocation,
                    requested_outputs,
                    runtime,
                    dynamic_admission,
                    had_lexical_environment,
                ),
            )
            .await?;
        runmat_runtime::runtime_error::complete_with_cleanup(result, cleanup)
    })
}

async fn drive_prepared_invocation_inner(
    unit: ExecutableUnit,
    mut invocation: NativeInvocation,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
    dynamic_admission: Option<runmat_vm::DynamicEvalAdmissionPort>,
    _had_lexical_environment: bool,
) -> Result<RootedNativeExecution, runmat_runtime::RuntimeError> {
    loop {
        match invocation
            .advance()
            .map_err(super::error::from_native_executor_error)?
        {
            NativeInvocationStep::Completed(execution) => {
                #[cfg(all(test, not(target_arch = "wasm32")))]
                if _had_lexical_environment {
                    NATIVE_LEXICAL_COMPLETIONS.with(|count| count.set(count.get() + 1));
                }
                return Ok(execution);
            }
            NativeInvocationStep::Suspended {
                continuation,
                generation,
            } => invocation
                .resume_suspension(continuation, generation)
                .await
                .map_err(super::error::from_native_executor_error)?,
            NativeInvocationStep::Deoptimized { target, .. }
                if target == runmat_runtime::native::NativeResumeKind::GENERIC_NATIVE =>
            {
                invocation
                    .resume_deoptimization()
                    .map_err(super::error::from_native_executor_error)?;
            }
            NativeInvocationStep::Deoptimized { target, .. }
                if target == runmat_runtime::native::NativeResumeKind::OPTIMIZED_NATIVE =>
            {
                invocation
                    .resume_optimization()
                    .map_err(super::error::from_native_executor_error)?;
            }
            NativeInvocationStep::Deoptimized { frame, .. } => {
                let resume_environment = invocation
                    .deoptimized_lexical_environment()
                    .map_err(super::error::from_native_executor_error)?;
                let _resume_roots = resume_environment
                    .as_ref()
                    .map(runmat_runtime::call::lexical::root_environment)
                    .transpose()?;
                return resume_interpreter(
                    &unit,
                    frame,
                    requested_outputs,
                    runtime,
                    resume_environment,
                    dynamic_admission,
                )
                .await;
            }
        }
    }
}

async fn resume_interpreter(
    unit: &ExecutableUnit,
    frame: MaterializedFrame,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
    dynamic_admission: Option<runmat_vm::DynamicEvalAdmissionPort>,
) -> Result<RootedNativeExecution, runmat_runtime::RuntimeError> {
    let function_id = frame.function;
    let layout = unit
        .vm_layout()
        .functions
        .get(&function_id)
        .ok_or_else(|| super::error::stage("NativeDeoptimization", "VM layout is unavailable"))?;
    let functions = unit
        .runtime_functions(&runtime)
        .map_err(|error| super::error::stage("NativeDeoptimization", error))?;
    let (bytecode, function_name, var_count, lexical_binding_slots, inherited_storage_slots) =
        if let Some(function) = functions.get(function_id) {
            (
                runmat_vm::Bytecode::for_function(
                    function,
                    functions.clone(),
                    unit.vm_layout().clone(),
                ),
                function.display_name.clone(),
                function.var_count,
                function.lexical_binding_slots.clone(),
                function.inherited_storage_slots.clone(),
            )
        } else {
            (
                unit.runtime_bytecode(&runtime)
                    .map_err(|error| super::error::stage("NativeDeoptimization", error))?,
                unit.mir()
                    .functions
                    .get(&function_id)
                    .map(|metadata| metadata.name.0.clone())
                    .unwrap_or_else(|| "<main>".into()),
                unit.bytecode().var_count,
                Vec::new(),
                Vec::new(),
            )
        };
    let pc = frame.site.bytecode_pc.ok_or_else(|| {
        super::error::stage(
            "NativeDeoptimization",
            "interpreter target has no verified bytecode resume point",
        )
    })?;
    let pc = usize::try_from(pc).map_err(|_| {
        super::error::stage(
            "NativeDeoptimization",
            "bytecode resume PC exceeds this host",
        )
    })?;
    let persistent_authority = runtime.session_variable_authority();
    for (local, continuation) in &frame.persistent_bindings {
        let slot = layout
            .mir_local_slots
            .get(&runmat_mir::MirLocalId(*local))
            .ok_or_else(|| {
                super::error::stage(
                    "NativeDeoptimization",
                    "native persistent alias has no canonical VM slot",
                )
            })?;
        let owner = inherited_storage_slots
            .iter()
            .find(|entry| {
                entry.slot == slot.0
                    && entry.kind == runmat_vm::InheritedStorageKind::Persistent
                    && entry.name == continuation.name
                    && entry.owner_function == continuation.function
            })
            .map_or(function_name.as_str(), |entry| {
                entry.owner_function.as_str()
            });
        continuation.validate_selected(&persistent_authority, owner)?;
    }
    let mut vars = vec![Value::Num(0.0); var_count];
    let mut assigned_slots = std::collections::HashSet::new();
    for local in frame.locals {
        let slot = layout
            .mir_local_slots
            .get(&runmat_mir::MirLocalId(local.local.0 as usize))
            .ok_or_else(|| {
                super::error::stage(
                    "NativeDeoptimization",
                    "native local has no canonical VM slot",
                )
            })?;
        if let Some(value) = local.value {
            vars[slot.0] = value;
            assigned_slots.insert(slot.0);
        }
    }
    // Native frame operands describe the MIR operation about to execute and
    // remain available to the generic-native continuation. The VM resumes at
    // the compiler-verified bytecode boundary before that operation, where its
    // own operand stack is empty, so it reconstructs those operands normally.
    let persistent_aliases = frame
        .persistent_bindings
        .into_iter()
        .map(|(local, continuation)| {
            let slot = layout
                .mir_local_slots
                .get(&runmat_mir::MirLocalId(local))
                .ok_or_else(|| {
                    super::error::stage(
                        "NativeDeoptimization",
                        "native persistent alias has no canonical VM slot",
                    )
                })?;
            let inherited = inherited_storage_slots.iter().find(|entry| {
                entry.slot == slot.0
                    && entry.kind == runmat_vm::InheritedStorageKind::Persistent
                    && entry.name == continuation.name
                    && entry.owner_function == continuation.function
            });
            let owner = inherited.map_or(function_name.as_str(), |entry| {
                entry.owner_function.as_str()
            });
            let binding = if inherited.is_some() {
                runmat_vm::PersistentBinding::from_inherited_native_continuation(
                    continuation,
                    &persistent_authority,
                    owner,
                )?
            } else {
                runmat_vm::PersistentBinding::from_native_continuation(
                    continuation,
                    &persistent_authority,
                    owner,
                )?
            };
            Ok((slot.0, binding))
        })
        .collect::<Result<std::collections::HashMap<_, _>, runmat_runtime::RuntimeError>>()?;
    let global_aliases = frame
        .global_bindings
        .into_iter()
        .map(|(local, binding)| {
            layout
                .mir_local_slots
                .get(&runmat_mir::MirLocalId(local))
                .map(|slot| (slot.0, binding))
                .ok_or_else(|| {
                    super::error::stage(
                        "NativeDeoptimization",
                        "native global binding has no canonical VM slot",
                    )
                })
        })
        .collect::<Result<std::collections::HashMap<_, _>, _>>()?;
    let missing_input_slots = frame
        .missing_input_locals
        .iter()
        .map(|local| {
            layout
                .mir_local_slots
                .get(&runmat_mir::MirLocalId(local.0 as usize))
                .map(|slot| slot.0)
                .ok_or_else(|| {
                    super::error::stage(
                        "NativeDeoptimization",
                        "omitted native input has no canonical VM slot",
                    )
                })
        })
        .collect::<Result<std::collections::HashSet<_>, _>>()?;
    let resume = runmat_vm::InterpreterResumeState {
        pc,
        completion_boundary: None,
        vars: runmat_vm::InterpreterVars::from_owned(vars),
        assigned_slots,
        supplied_inputs: frame.supplied_inputs,
        requested_outputs: frame.requested_outputs,
        missing_input_slots,
        global_aliases,
        persistent_aliases,
        side_effect_epoch: frame.site.side_effect_epoch,
    };
    let runmat_vm::InterpreterOutcome::Completed(completion) =
        runmat_vm::interpret_resume_with_dynamic_admission_in_context(
            &bytecode,
            resume,
            Some(&function_name),
            runtime.clone(),
            runmat_vm::DynamicResumeSelection {
                lexical_environment,
                lexical_binding_slots,
                inherited_storage_slots,
                dynamic_admission,
            },
        )
        .await?;
    let projection = completion
        .with_owned(|completion| runmat_vm::CompletedFrameProjection::new(completion.values))?;
    let values = projection.values();
    let body = unit.mir().bodies.get(&function_id).ok_or_else(|| {
        super::error::stage("NativeDeoptimization", "MIR function body is unavailable")
    })?;
    let fixed_outputs = body
        .abi
        .fixed_output_bindings()
        .map(|local| {
            layout
                .binding_slots
                .get(local)
                .and_then(|slot| values.get(slot.0))
                .as_deref()
                .cloned()
                .unwrap_or(Value::Num(0.0))
        })
        .collect::<Vec<_>>();
    let varargout = body
        .abi
        .varargout
        .and_then(|binding| layout.binding_slots.get(&binding))
        .and_then(|slot| values.get(slot.0));
    let outputs = runmat_runtime::call::function_abi::collect_function_outputs(
        &function_name,
        &fixed_outputs,
        varargout.as_deref(),
        requested_outputs,
    )?;
    drop(varargout);
    let captures: Vec<_> = unit
        .mir()
        .functions
        .get(&function_id)
        .into_iter()
        .flat_map(|metadata| &metadata.captures)
        .filter_map(|capture| {
            let local = body
                .locals
                .iter()
                .find(|local| local.binding == Some(capture.binding))?;
            let slot = layout.mir_local_slots.get(&local.id)?;
            Some(runmat_runtime::call::lexical::LexicalCapture {
                binding: capture.binding,
                value: values
                    .get(slot.0)
                    .as_deref()
                    .cloned()
                    .unwrap_or(Value::Num(0.0)),
            })
        })
        .collect();
    let execution = RootedNativeExecution::new(NativeExecution {
        outputs,
        captures,
        loop_backedges: BTreeMap::new(),
        osr_entry: None,
        vectorized_regions: 0,
        workspace: None,
        expression: None,
    })
    .map_err(|error| error.into_runtime_error())?;
    drop(projection);
    Ok(execution)
}

#[cfg(test)]
#[path = "deopt_deal_tests.rs"]
mod deal_tests;
#[cfg(test)]
#[path = "deopt_projection_tests.rs"]
mod projection_tests;

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use runmat_native_executor::deopt::{DeoptimizationPolicy, FaultInjection, ResumeTarget};
    use runmat_value::Value;

    use crate::{ExecutableSource, RunMatSession};

    #[test]
    fn core_resumes_native_execution_in_interpreter_with_exact_call_context() {
        let mut session = RunMatSession::with_options(false, false).expect("session init");
        session.configure_runtime_context();
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "core-native-deopt-test@1",
                "nativeDeopt.m",
                "function [y, observed] = nativeDeopt(x, optional)\ny = sqrt(x);\nif nargin < 2\nobserved = nargout;\nelse\nobserved = optional;\nend\nend\n",
            ),
            None,
        ))
        .expect("compile deoptimization fixture");
        let compiled = super::super::compile::compile(&unit, Some("nativeDeopt"))
            .expect("compile generic-native fixture");
        let safepoint = compiled
            .safepoints
            .get(&compiled.entrypoint)
            .and_then(|points| points.first())
            .copied()
            .expect("sqrt call should publish a native safepoint");
        let policy = DeoptimizationPolicy {
            target: ResumeTarget::Interpreter,
            ..DeoptimizationPolicy::default()
        }
        .inject(FaultInjection::Safepoint(safepoint));
        let runtime = unit
            .bind_runtime_sources(
                session.execution_runtime_context(unit.revision().program_revision.clone()),
            )
            .expect("bind selected deoptimization source");
        let execution = block_on(super::invoke_with_policy(
            &unit,
            super::DeoptimizationInvocation {
                executor: compiled.executor,
                function: compiled.entrypoint,
                lexical_environment: None,
                inputs: super::Inputs::Source {
                    captures: Vec::new(),
                    arguments: vec![Value::Num(81.0)],
                },
                requested_outputs: 2,
                runtime,
                dynamic_admission: None,
                policy,
                osr: None,
                workspace: None,
            },
        ))
        .expect("native execution should resume in the interpreter");
        assert_eq!(execution.outputs, vec![Value::Num(9.0), Value::Num(2.0)]);
    }

    #[test]
    fn native_deopt_resumes_with_its_owned_shared_activation() {
        runmat_gc::gc_test_context(|| {
            let mut session = RunMatSession::with_options(false, false).expect("session init");
            session.set_compat_mode(runmat_parser::CompatMode::RunMat);
            session.configure_runtime_context();
            let source_text = "function [y, h] = nativeLexicalDeopt(x)\nshared = x;\nh = @bump;\ny = sqrt(81);\ny = bump(2);\nfunction out = bump(delta)\nshared = shared + delta;\nout = shared;\nend\nend\n";
            let unit = block_on(session.compile_executable_unit(
                ExecutableSource::new(
                    "native-lexical-deopt@1",
                    "nativeLexicalDeopt.m",
                    source_text,
                ),
                None,
            ))
            .expect("compile nested deoptimization fixture");
            let compiled = super::super::compile::compile(&unit, Some("nativeLexicalDeopt"))
                .expect("compile generic-native nested function");
            let sqrt_safepoints = compiled
                .safepoint_locations
                .get(&compiled.entrypoint)
                .into_iter()
                .flatten()
                .filter(|(_, location)| {
                    source_text
                        .get(location.span.start as usize..location.span.end as usize)
                        .is_some_and(|site| site.contains("sqrt"))
                })
                .cloned()
                .collect::<Vec<_>>();
            assert_eq!(
                sqrt_safepoints.len(),
                1,
                "the lowered sqrt operation has one exact native safepoint"
            );
            let (safepoint, selected_source) = sqrt_safepoints
                .into_iter()
                .next()
                .expect("typed sqrt call has a native safepoint");
            let runtime = unit
                .bind_runtime_sources(
                    session.execution_runtime_context(unit.revision().program_revision.clone()),
                )
                .expect("bind selected nested source");
            let owned_bindings = unit
                .mir()
                .lexical_environment_projection(compiled.entrypoint)
                .expect("canonical nested ownership projection")
                .owned;
            assert_eq!(owned_bindings.len(), 1, "only shared is an owned cell");
            let shared_binding = *owned_bindings.iter().next().expect("shared binding");
            let (parent, _parent_roots) = runmat_runtime::call::lexical::allocate_environment(None)
                .expect("parent activation");
            let mut invocation = compiled
                .executor
                .begin_request_with_environment(
                    runmat_native_executor::NativeInvocationRequest {
                        function: compiled.entrypoint,
                        captures: Vec::new(),
                        arguments: vec![Value::Num(5.0)],
                        requested_outputs: 2,
                        runtime: runtime.clone(),
                        deoptimization: DeoptimizationPolicy {
                            target: ResumeTarget::Interpreter,
                            ..Default::default()
                        }
                        .inject(FaultInjection::Safepoint(safepoint)),
                        osr_target: None,
                        workspace: None,
                    },
                    Some(parent.clone()),
                )
                .expect("begin native child activation");
            let result = block_on(async {
                loop {
                    match invocation.advance().expect("advance native nested fixture") {
                        super::NativeInvocationStep::Completed(_) => {
                            panic!("selected safepoint must deoptimize")
                        }
                        super::NativeInvocationStep::Suspended {
                            continuation,
                            generation,
                        } => invocation
                            .resume_suspension(continuation, generation)
                            .await
                            .expect("resume native suspension"),
                        super::NativeInvocationStep::Deoptimized { target, frame, .. } => {
                            assert_eq!(
                                target,
                                runmat_runtime::native::NativeResumeKind::INTERPRETER
                            );
                            assert_eq!(frame.function, compiled.entrypoint);
                            assert_eq!(
                                frame.site.point.function,
                                unit.analysis()
                                    .function_for_local(compiled.entrypoint)
                                    .expect("compiled entry portable identity")
                                    .function
                            );
                            assert!(frame.site.bytecode_pc.is_some());
                            assert_eq!(frame.site.source, selected_source.source);
                            assert_eq!(frame.site.span, selected_source.span);
                            let active_environment = invocation
                                .deoptimized_lexical_environment()
                                .expect("retained native activation")
                                .expect("owned lexical cell activation");
                            assert_ne!(active_environment, parent);
                            assert_eq!(
                                runmat_runtime::call::lexical::read_binding(
                                    &active_environment,
                                    shared_binding,
                                )
                                .expect("read pre-resume shared cell"),
                                Some(Value::Num(5.0)),
                                "native deopt must precede bump's shared-cell mutation"
                            );
                            let _roots = runmat_runtime::call::lexical::root_environment(
                                &active_environment,
                            )
                            .expect("root resumed activation");
                            let execution = super::resume_interpreter(
                                &unit,
                                frame,
                                2,
                                runtime.clone(),
                                Some(active_environment.clone()),
                                None,
                            )
                            .await
                            .expect("resume in VM with the same child activation");
                            assert_eq!(
                                runmat_runtime::call::lexical::read_binding(
                                    &active_environment,
                                    shared_binding,
                                )
                                .expect("read post-resume shared cell"),
                                Some(Value::Num(7.0)),
                                "VM must mutate the same native-owned shared cell"
                            );
                            break (execution, active_environment);
                        }
                    }
                }
            });
            assert_eq!(result.0.outputs[0], Value::Num(7.0));
            let Value::Closure(handle) = &result.0.outputs[1] else {
                panic!("second output must be escaped nested handle")
            };
            assert_eq!(handle.lexical_environment.as_ref(), Some(&result.1));

            let production = block_on(super::invoke_with_policy(
                &unit,
                super::DeoptimizationInvocation {
                    executor: compiled.executor,
                    function: compiled.entrypoint,
                    lexical_environment: Some(parent.clone()),
                    inputs: super::Inputs::Source {
                        captures: Vec::new(),
                        arguments: vec![Value::Num(5.0)],
                    },
                    requested_outputs: 2,
                    runtime,
                    dynamic_admission: None,
                    policy: DeoptimizationPolicy {
                        target: ResumeTarget::Interpreter,
                        ..Default::default()
                    }
                    .inject(FaultInjection::Safepoint(safepoint)),
                    osr: None,
                    workspace: None,
                },
            ))
            .expect("production deoptimization resumes its owned child activation");
            assert_eq!(production.outputs[0], Value::Num(7.0));
            let Value::Closure(handle) = &production.outputs[1] else {
                panic!("production output must retain escaped nested handle")
            };
            assert_ne!(handle.lexical_environment.as_ref(), Some(&parent));
        });
    }

    #[test]
    fn function_clear_before_native_deopt_does_not_resurrect_persistent() {
        runmat_gc::gc_test_context(|| {
            let mut session = RunMatSession::with_options(false, false).expect("session init");
            session.configure_runtime_context();
            let unit = block_on(session.compile_executable_unit(
                ExecutableSource::new(
                    "persistent-deopt-clear@1",
                    "persistentDeopt.m",
                    "function y = persistentDeopt()\npersistent p\np = 7;\nclear('persistentDeopt');\ny = sqrt(9);\nend\n",
                ),
                None,
            ))
            .expect("compile persistent deopt fixture");
            let compiled = super::super::compile::compile(&unit, Some("persistentDeopt"))
                .expect("compile generic-native persistent function");
            let safepoint = compiled
                .safepoints
                .get(&compiled.entrypoint)
                .and_then(|points| points.last())
                .copied()
                .expect("sqrt call after clear has a native safepoint");
            let runtime = unit
                .bind_runtime_sources(
                    session.execution_runtime_context(unit.revision().program_revision.clone()),
                )
                .expect("bind selected persistent deoptimization source");
            let mut invocation = compiled
                .executor
                .begin_request(runmat_native_executor::NativeInvocationRequest {
                    function: compiled.entrypoint,
                    captures: Vec::new(),
                    arguments: Vec::new(),
                    requested_outputs: 1,
                    runtime: runtime.clone(),
                    deoptimization: DeoptimizationPolicy {
                        target: ResumeTarget::Interpreter,
                        ..Default::default()
                    }
                    .inject(FaultInjection::Safepoint(safepoint)),
                    osr_target: None,
                    workspace: None,
                })
                .expect("begin native persistent-clear invocation");
            // `clear` retires selected native extensions on the session's
            // local-task host. Drive that same host while this raw invocation
            // is suspended, as the public request path does.
            let local_tasks = session.local_task_driver();
            let result = block_on(local_tasks.participate_foreground(async {
                loop {
                    match invocation.advance().expect("advance native clear fixture") {
                        super::NativeInvocationStep::Completed(_) => {
                            panic!("post-clear safepoint must physically deoptimize")
                        }
                        super::NativeInvocationStep::Suspended {
                            continuation,
                            generation,
                        } => invocation
                            .resume_suspension(continuation, generation)
                            .await
                            .expect("resume native clear fixture suspension"),
                        super::NativeInvocationStep::Deoptimized { target, frame, .. } => {
                            assert_eq!(
                                target,
                                runmat_runtime::native::NativeResumeKind::INTERPRETER
                            );
                            assert_eq!(
                                runtime
                                    .session_variable_authority()
                                    .persistent_named_value("persistentDeopt", "p")
                                    .expect("read canonical persistent after native clear"),
                                None,
                                "clear must execute before physical deoptimization"
                            );
                            break super::resume_interpreter(
                                &unit,
                                frame,
                                1,
                                runtime.clone(),
                                None,
                                None,
                            )
                            .await
                            .expect("resume cleared persistent frame in interpreter");
                        }
                    }
                }
            }))
            .expect("drive native-clear lifecycle tasks");
            assert_eq!(result.outputs, vec![Value::Num(3.0)]);
            assert_eq!(
                session
                    .runtime_context()
                    .session_variable_authority()
                    .persistent_named_value("persistentDeopt", "p")
                    .unwrap(),
                None,
                "old native alias must not republish after selected function clear"
            );
        });
    }
}
