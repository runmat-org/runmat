use crate::accel::fusion as accel_fusion;
use crate::accel::residency as accel_residency;
use crate::bytecode::{Bytecode, FunctionRegistry, Instr};
use crate::interpreter::api::{InterpreterOutcome, InterpreterState};
use crate::interpreter::dispatch::{self as interp_dispatch, DispatchDecision};
use crate::interpreter::engine as interp_engine;
use crate::interpreter::errors::{attach_span_from_pc, mex, set_vm_pc};
use crate::interpreter::timing::InterpreterTiming;
use crate::runtime::call_stack::attach_call_frames;
use crate::runtime::globals as runtime_globals;
use crate::runtime::workspace::{
    refresh_workspace_state, workspace_assign, workspace_clear, workspace_lookup, workspace_remove,
    workspace_snapshot,
};
use runmat_runtime::call::function_abi::{
    collect_function_outputs, prepare_function_inputs, FunctionInputSpec,
};
use runmat_runtime::{
    user_functions,
    workspace::{self as runtime_workspace, WorkspaceResolver},
    RuntimeError,
};
use runmat_value::{CellArray, Value};
use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Once;
use tracing::{debug, info_span};

#[cfg(not(target_arch = "wasm32"))]
use std::future::Future;

#[cfg(feature = "native-accel")]
use runmat_accelerate::{
    activate_fusion_plan, active_group_plan_clone, deactivate_fusion_plan, set_current_pc,
};

#[cfg(feature = "native-accel")]
struct FusionPlanGuard;

#[cfg(feature = "native-accel")]
impl Drop for FusionPlanGuard {
    fn drop(&mut self) {
        deactivate_fusion_plan();
    }
}

type VmResult<T> = Result<T, RuntimeError>;
runmat_thread_local::runmat_thread_local! {
    static CALL_COUNTS: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

fn sync_initial_vars(initial: &mut Vec<Value>, vars: &[Value]) {
    initial.clear();
    initial.extend_from_slice(vars);
}

fn ensure_workspace_resolver_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        runtime_workspace::register_workspace_resolver(WorkspaceResolver {
            lookup: workspace_lookup,
            snapshot: workspace_snapshot,
            globals: runtime_globals::workspace_global_names,
            assign: Some(workspace_assign),
            clear: Some(workspace_clear),
            remove: Some(workspace_remove),
        });
    });
}

fn ensure_wasm_builtins_registered() {
    #[cfg(target_arch = "wasm32")]
    {
        static REGISTER: Once = Once::new();
        REGISTER.call_once(|| {
            runmat_runtime::builtins::wasm_registry::register_all();
        });
    }
}

#[cfg(feature = "native-accel")]
fn clear_residency(value: &Value) {
    if let Err(err) = accel_residency::clear_value(value) {
        log::warn!("failed to clear GPU residency: {err}");
    }
}

fn active_or_standalone_runtime_context() -> runmat_runtime::context::RuntimeContext {
    runmat_runtime::context::legacy::active().unwrap_or_else(|| {
        runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
    })
}

pub async fn invoke_semantic_function_value(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
) -> Result<Value, RuntimeError> {
    let runtime = active_or_standalone_runtime_context();
    let (value, _) = invoke_semantic_function_value_with_input_residency(
        function,
        args,
        requested_outputs,
        function_registry,
        InputResidency::Transferred,
        runtime,
    )
    .await?;
    Ok(value)
}

pub async fn invoke_semantic_function_value_in_context(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<Value, RuntimeError> {
    let (value, _) = invoke_semantic_function_value_with_input_residency(
        function,
        args,
        requested_outputs,
        function_registry,
        InputResidency::Transferred,
        runtime,
    )
    .await?;
    Ok(value)
}

pub(crate) async fn invoke_semantic_function_value_with_capture_updates(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<(Value, Vec<Value>), RuntimeError> {
    invoke_semantic_function_value_with_input_residency(
        function,
        args,
        requested_outputs,
        function_registry,
        InputResidency::Borrowed,
        runtime,
    )
    .await
}

#[derive(Clone, Copy)]
enum InputResidency {
    /// The caller retains the argument values as live workspace roots.
    Borrowed,
    /// The invocation owns the argument values and may release inputs the
    /// function neither returns nor stores in another live root.
    Transferred,
}

async fn invoke_semantic_function_value_with_input_residency(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    input_residency: InputResidency,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<(Value, Vec<Value>), RuntimeError> {
    runtime
        .scope(invoke_semantic_function_value_with_input_residency_inner(
            function,
            args,
            requested_outputs,
            function_registry,
            input_residency,
            runtime.clone(),
        ))
        .await
}

async fn invoke_semantic_function_value_with_input_residency_inner(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    input_residency: InputResidency,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<(Value, Vec<Value>), RuntimeError> {
    let function_id = runmat_hir::FunctionId(function);
    let func = function_registry.get(function_id).ok_or_else(|| {
        let message = format!("Undefined semantic function: {function}");
        mex("UndefinedSemanticFunction", &message)
    })?;
    if args.len() < func.capture_slots.len() {
        let message = format!(
            "semantic function {} received too few arguments",
            func.display_name
        );
        return Err(mex("SemanticFunctionArity", &message));
    }
    let runtime_args = &args[func.capture_slots.len()..];
    let mut vars = vec![Value::Num(0.0); func.var_count];
    for (slot, value) in func.capture_slots.iter().zip(args.iter()) {
        if *slot < vars.len() {
            vars[*slot] = value.clone();
        }
    }
    let input_specs = func
        .argument_validations
        .iter()
        .map(|validation| {
            let input_index = func
                .input_slots
                .iter()
                .position(|slot| *slot == validation.input_slot)
                .ok_or_else(|| mex("InvalidInputSlot", "function argument slot out of bounds"))?;
            Ok(FunctionInputSpec {
                input_index,
                size: validation.size.as_ref(),
                class_name: validation.class_name.as_deref(),
                validators: &validation.validators,
                default_value: validation.default_value.as_ref(),
            })
        })
        .collect::<Result<Vec<_>, RuntimeError>>()?;
    let prepared = prepare_function_inputs(
        &func.display_name,
        runtime_args,
        func.input_slots.len(),
        func.varargin_slot.is_some(),
        &input_specs,
    )?;
    let mut missing_input_slots = HashSet::new();
    for (slot, value) in func.input_slots.iter().zip(&prepared.fixed) {
        if let Some(value) = value {
            if *slot < vars.len() {
                vars[*slot] = value.clone();
            }
        } else {
            missing_input_slots.insert(*slot);
        }
    }
    if let (Some(slot), Some(cell)) = (func.varargin_slot, prepared.varargin) {
        if slot < vars.len() {
            vars[slot] = Value::Cell(cell);
        }
    }
    if let Some(slot) = func.varargout_slot {
        if slot < vars.len() {
            let cell = CellArray::new(Vec::new(), 1, 0)
                .map_err(|err| mex("VarargoutPack", &format!("varargout: {err}")))?;
            vars[slot] = Value::Cell(cell);
        }
    }
    if let Some(slot) = func.implicit_nargin_slot {
        if slot < vars.len() {
            vars[slot] = Value::Num(prepared.nargin as f64);
        }
    }
    if let Some(slot) = func.implicit_nargout_slot {
        if slot < vars.len() {
            vars[slot] = Value::Num(requested_outputs as f64);
        }
    }

    let _active_semantic_function_guard =
        user_functions::push_active_semantic_function(function_id.0);
    let mut bytecode = Bytecode::with_instructions(func.instructions.clone(), func.var_count);
    bytecode.instr_spans = func.instr_spans.clone();
    bytecode.call_arg_spans = func.call_arg_spans.clone();
    bytecode.coverage_sites = func.coverage_sites.clone();
    bytecode.source_id = func.source_id;
    bytecode.var_names = func.var_names.clone();
    let mut initially_unassigned_slots = func.initially_unassigned_slots.clone();
    for slot in &func.capture_slots {
        initially_unassigned_slots.remove(slot);
    }
    for (slot, value) in func.input_slots.iter().zip(&prepared.fixed) {
        if value.is_some() {
            initially_unassigned_slots.remove(slot);
        }
    }
    if let Some(slot) = func.varargin_slot {
        initially_unassigned_slots.remove(&slot);
    }
    if let Some(slot) = func.varargout_slot {
        initially_unassigned_slots.remove(&slot);
    }
    if let Some(slot) = func.implicit_nargin_slot {
        initially_unassigned_slots.remove(&slot);
    }
    if let Some(slot) = func.implicit_nargout_slot {
        initially_unassigned_slots.remove(&slot);
    }
    bytecode.initially_unassigned_slots = initially_unassigned_slots;
    bytecode.bound_functions = function_registry.functions.clone();
    bytecode.function_registry = function_registry.clone();
    let result_vars = {
        let future = interpret_function_with_counts_in_context(
            &bytecode,
            vars,
            &func.display_name,
            requested_outputs,
            prepared.nargin,
            missing_input_slots,
            runtime,
        );
        #[cfg(target_arch = "wasm32")]
        {
            future.await?
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // A semantic call recursively drives another async interpreter.
            // Polling it on the caller's small executor/test-thread stack makes
            // ordinary class method chains consume the full native stack long
            // before reaching a meaningful language recursion depth. Grow the
            // stack only while polling the nested interpreter; values remain on
            // the same thread and therefore retain their thread-confined GC
            // semantics.
            const SEMANTIC_CALL_STACK_BYTES: usize = 16 * 1024 * 1024;
            let mut future = Box::pin(future);
            futures::future::poll_fn(move |context| {
                stacker::grow(SEMANTIC_CALL_STACK_BYTES, || future.as_mut().poll(context))
            })
            .await?
        }
    };
    let fixed_outputs = func
        .output_slots
        .iter()
        .map(|slot| result_vars.get(*slot).cloned().unwrap_or(Value::Num(0.0)))
        .collect::<Vec<_>>();
    let varargout = func.varargout_slot.and_then(|slot| result_vars.get(slot));
    let output_values = collect_function_outputs(
        &func.display_name,
        &fixed_outputs,
        varargout,
        requested_outputs,
    )?;
    let updated_captures = func
        .capture_slots
        .iter()
        .map(|slot| result_vars.get(*slot).cloned().unwrap_or(Value::Num(0.0)))
        .collect::<Vec<_>>();
    #[cfg(feature = "native-accel")]
    clear_semantic_function_temp_residency(
        &result_vars,
        args,
        &output_values,
        &updated_captures,
        input_residency,
    );
    Ok((
        output_value(output_values, requested_outputs),
        updated_captures,
    ))
}

fn output_value(output_values: Vec<Value>, requested_outputs: usize) -> Value {
    match requested_outputs {
        0 => Value::OutputList(Vec::new()),
        1 => output_values.into_iter().next().unwrap_or(Value::Num(0.0)),
        _ => Value::OutputList(output_values.into_iter().take(requested_outputs).collect()),
    }
}

#[cfg(feature = "native-accel")]
fn clear_semantic_function_temp_residency(
    result_vars: &[Value],
    args: &[Value],
    output_values: &[Value],
    updated_captures: &[Value],
    input_residency: InputResidency,
) {
    let mut keep_values = output_values.to_vec();
    if matches!(input_residency, InputResidency::Borrowed) {
        // Interpreter-to-interpreter calls borrow arguments from the caller's
        // workspace. Those values remain live after the callee returns even if
        // the callee overwrites its input slots.
        keep_values.extend(args.iter().cloned());
    }
    keep_values.extend(updated_captures.iter().cloned());
    keep_values.extend(runtime_globals::collect_thread_roots());
    let keep = Value::OutputList(keep_values);
    for value in result_vars {
        if let Err(err) = accel_residency::clear_value_excluding(value, &keep) {
            log::warn!("failed to clear temporary semantic function GPU residency: {err}");
        }
    }
}

pub async fn interpret_with_vars(
    bytecode: &Bytecode,
    initial_vars: &mut Vec<Value>,
    current_function_name: Option<&str>,
) -> VmResult<InterpreterOutcome> {
    let runtime = active_or_standalone_runtime_context();
    interpret_with_vars_in_context(bytecode, initial_vars, current_function_name, runtime).await
}

pub async fn interpret_with_vars_in_context(
    bytecode: &Bytecode,
    initial_vars: &mut Vec<Value>,
    current_function_name: Option<&str>,
    runtime: runmat_runtime::context::RuntimeContext,
) -> VmResult<InterpreterOutcome> {
    runtime
        .scope(runmat_runtime::data::with_tx_registry_scope(
            interpret_with_vars_inner(
                bytecode,
                initial_vars,
                current_function_name,
                runtime.clone(),
            ),
        ))
        .await
}

async fn interpret_with_vars_inner(
    bytecode: &Bytecode,
    initial_vars: &mut Vec<Value>,
    current_function_name: Option<&str>,
    runtime: runmat_runtime::context::RuntimeContext,
) -> VmResult<InterpreterOutcome> {
    let _debug_frame_guard = runmat_runtime::debug_context::push_frame(
        current_function_name.unwrap_or("<main>"),
        bytecode.source_id,
        bytecode_frame_span(bytecode),
    );
    let call_counts = CALL_COUNTS.with(|cc| cc.borrow().clone());
    let state = Box::new(InterpreterState::new_in_context(
        bytecode.clone(),
        initial_vars,
        current_function_name,
        call_counts,
        runtime,
    ));
    match Box::pin(run_interpreter(state, initial_vars)).await {
        Ok(outcome) => Ok(outcome),
        Err(err) => {
            let err = attach_span_from_pc(bytecode, err);
            let current_name = current_function_name.unwrap_or("<main>");
            Err(attach_call_frames(bytecode, current_name, err))
        }
    }
}

fn bytecode_frame_span(bytecode: &Bytecode) -> Option<(usize, usize)> {
    bytecode
        .instr_spans
        .first()
        .map(|span| (span.start, span.end))
}

async fn run_interpreter(
    state: Box<InterpreterState>,
    initial_vars: &mut Vec<Value>,
) -> VmResult<InterpreterOutcome> {
    let state = *state;
    Box::pin(run_interpreter_inner(state, initial_vars)).await
}

// Semantic invokers use the runtime's established Arc callback ABI, while live
// execution contexts are intentionally thread-confined with the GC-owned Values
// they contain. The callback never crosses the invocation thread.
#[allow(clippy::arc_with_non_send_sync)]
async fn run_interpreter_inner(
    state: InterpreterState,
    initial_vars: &mut Vec<Value>,
) -> VmResult<InterpreterOutcome> {
    let run_span = info_span!(
        "interpreter.run",
        function = state.current_function_name.as_str()
    );
    let _run_guard = run_span.enter();
    ensure_wasm_builtins_registered();
    ensure_workspace_resolver_registered();
    #[cfg(feature = "native-accel")]
    activate_fusion_plan(state.fusion_plan.clone());
    #[cfg(feature = "native-accel")]
    let _fusion_guard = FusionPlanGuard;
    let InterpreterState {
        mut stack,
        mut vars,
        mut pc,
        mut context,
        mut try_stack,
        mut last_exception,
        mut imports,
        mut global_aliases,
        mut persistent_aliases,
        mut missing_input_slots,
        current_function_name,
        call_counts,
        initial_assigned_var_count,
        #[cfg(feature = "native-accel")]
            fusion_plan: _,
        #[cfg(feature = "native-accel")]
        fusion_accel_graph,
        bytecode,
    } = state;
    let _source_context_guard =
        runmat_runtime::source_context::replace_current_source_id(bytecode.source_id);
    let _arity_call_counts_guard =
        runmat_runtime::builtins::introspection::arity_check::replace_call_counts(
            call_counts.clone(),
        );
    let function_registry = Arc::new(bytecode.function_registry());
    let nested_runtime = context.runtime.clone();
    let previous_semantic_invoker = user_functions::current_semantic_function_invoker();
    let registry_for_function_invoker = Arc::clone(&function_registry);
    let _semantic_function_guard =
        user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function: usize, args: &[Value], requested_outputs: usize| {
                let args = args.to_vec();
                let previous_invoker = previous_semantic_invoker.clone();
                let function_registry = Arc::clone(&registry_for_function_invoker);
                let runtime = nested_runtime.clone();
                Box::pin(async move {
                    let local_function = function_registry
                        .get(runmat_hir::FunctionId(function))
                        .is_some();
                    if !local_function {
                        if let Some(invoker) = previous_invoker {
                            return invoker(function, &args, requested_outputs).await;
                        }
                    }
                    invoke_semantic_function_value_with_capture_updates(
                        function,
                        &args,
                        requested_outputs,
                        &function_registry,
                        runtime,
                    )
                    .await
                    .map(|(value, _)| value)
                })
            },
        )));
    let previous_semantic_resolver = user_functions::current_semantic_function_resolver();
    let registry_for_function_resolver = Arc::clone(&function_registry);
    let _semantic_resolver_guard =
        user_functions::install_semantic_function_resolver(Some(Arc::new(move |name: &str| {
            if let Some(active_function) = user_functions::current_active_semantic_function() {
                if let Some(function) =
                    registry_for_function_resolver.get(runmat_hir::FunctionId(active_function))
                {
                    if let Some(scoped_function) = registry_for_function_resolver
                        .resolve_name_in_private_scope(&function.private_owner_scope, name)
                    {
                        return Some(scoped_function.0);
                    }
                }
            }
            if let Some(function) = registry_for_function_resolver.resolve_name(name) {
                return Some(function.0);
            }
            previous_semantic_resolver
                .as_ref()
                .and_then(|resolver| resolver(name))
        })));
    let mut source_function_catalog = function_registry
        .functions
        .values()
        .filter_map(|function| {
            function.source_id.map(
                |source_id| runmat_runtime::user_functions::SourceFunctionInfo {
                    source_id,
                    name: function.display_name.clone(),
                    function: function.function.0,
                },
            )
        })
        .collect::<Vec<_>>();
    source_function_catalog.sort_by_key(|info| info.function);
    let _source_function_catalog_guard =
        user_functions::install_source_function_catalog(Some(Arc::new(source_function_catalog)));
    CALL_COUNTS.with(|cc| {
        *cc.borrow_mut() = call_counts.clone();
    });
    let _workspace_guard = interp_engine::prepare_workspace_guard(
        &bytecode.var_names,
        &mut vars,
        initial_assigned_var_count,
        &bytecode.initially_unassigned_slots,
    );
    let thread_roots: Vec<Value> = runtime_globals::collect_thread_roots();
    let mut _gc_context = interp_engine::create_gc_context(&stack, &vars, thread_roots)?;
    let debug_stack = interp_engine::debug_stack_enabled();
    let mut interpreter_timing = InterpreterTiming::new();
    while pc < bytecode.instructions.len() {
        if let Some(sites) = bytecode.coverage_sites.get(pc) {
            crate::coverage::hit_sites(sites);
        }
        set_vm_pc(pc);
        #[cfg(feature = "native-accel")]
        set_current_pc(pc);
        if let Err(err) = interp_engine::check_cancelled() {
            #[cfg(feature = "native-accel")]
            {
                for value in &stack {
                    clear_residency(value);
                }
                for value in &vars {
                    clear_residency(value);
                }
            }
            return Err(err);
        }
        #[cfg(feature = "native-accel")]
        if let (Some(plan), Some(graph)) = (active_group_plan_clone(), fusion_accel_graph.as_ref())
        {
            if plan.group.span.start == pc {
                #[cfg(feature = "native-accel")]
                {
                    interp_engine::note_fusion_gate(
                        &mut interpreter_timing,
                        &plan,
                        &bytecode,
                        pc,
                        accel_fusion::fusion_span_has_vm_barrier(
                            &bytecode.instructions,
                            &plan.group.span,
                        ),
                        accel_fusion::fusion_span_live_result_count(
                            &bytecode.instructions,
                            &plan.group.span,
                        ),
                    );
                }
                let span = plan.group.span.clone();
                let has_barrier =
                    accel_fusion::fusion_span_has_vm_barrier(&bytecode.instructions, &span);
                let _fusion_span = info_span!(
                    "fusion.execute",
                    span_start = plan.group.span.start,
                    span_end = plan.group.span.end,
                    kind = ?plan.group.kind
                )
                .entered();
                if !has_barrier {
                    match accel_fusion::try_execute_fusion_group(
                        &plan,
                        graph,
                        &mut stack,
                        &mut vars,
                        &mut context,
                    )
                    .await
                    {
                        Ok(result) => {
                            stack.push(result);
                            pc = plan.group.span.end + 1;
                            continue;
                        }
                        Err(err) => {
                            log::debug!("fusion fallback at pc {}: {}", pc, err);
                        }
                    }
                } else {
                    interp_engine::note_fusion_skip(pc, &span);
                }
            }
        }
        interp_engine::note_pre_dispatch(
            &mut interpreter_timing,
            debug_stack,
            pc,
            &bytecode.instructions[pc],
            stack.len(),
        );
        let call_counts_snapshot = CALL_COUNTS.with(|cc| cc.borrow().clone());
        let store_var_global_aliases = match &bytecode.instructions[pc] {
            Instr::StoreVar(_) => Some(global_aliases.clone()),
            _ => None,
        };
        let store_local_global_aliases = match &bytecode.instructions[pc] {
            Instr::StoreLocal(_) => Some(global_aliases.clone()),
            _ => None,
        };
        let mut clear_value_residency = |value: &Value| {
            #[cfg(feature = "native-accel")]
            clear_residency(value);
        };
        let mut store_var_before_overwrite = |_current: &Value, _incoming: &Value| {};
        let mut store_var_after_store = |stored_index: usize, stored_value: &Value| {
            if let Some(ref aliases) = store_var_global_aliases {
                runtime_globals::update_global_store(stored_index, stored_value, aliases);
            }
        };
        let mut store_local_before_local_overwrite = |_current: &Value, _incoming: &Value| {};
        let mut store_local_before_var_overwrite = |_current: &Value, _incoming: &Value| {};
        let mut store_local_after_store = |stored_offset: usize, stored_value: &Value| {
            if let Some(ref aliases) = store_local_global_aliases {
                runtime_globals::update_global_store(stored_offset, stored_value, aliases);
            }
        };
        let mut store_local_after_fallback_store =
            |func_name: &str, stored_offset: usize, stored_value: &Value| {
                if let Some(ref aliases) = store_local_global_aliases {
                    runtime_globals::update_global_store(stored_offset, stored_value, aliases);
                }
                runtime_globals::update_persistent_local_store(
                    func_name,
                    stored_offset,
                    stored_value,
                );
            };
        let dispatch_result = interp_dispatch::dispatch_instruction(
            interp_dispatch::DispatchMeta {
                instr: &bytecode.instructions[pc],
                var_names: &bytecode.var_names,
                function_registry: &function_registry,
                source_id: bytecode.source_id,
                call_arg_spans: bytecode.call_arg_spans.get(pc).cloned().flatten(),
                call_counts: &call_counts_snapshot,
                current_function_name: &current_function_name,
            },
            interp_dispatch::DispatchState {
                stack: &mut stack,
                vars: &mut vars,
                context: &mut context,
                try_stack: &mut try_stack,
                last_exception: &mut last_exception,
                imports: &mut imports,
                global_aliases: &mut global_aliases,
                persistent_aliases: &mut persistent_aliases,
                missing_input_slots: &mut missing_input_slots,
                pc: &mut pc,
            },
            interp_dispatch::DispatchHooks {
                clear_value_residency: &mut clear_value_residency,
                store_var_before_overwrite: &mut store_var_before_overwrite,
                store_var_after_store: &mut store_var_after_store,
                store_local_before_local_overwrite: &mut store_local_before_local_overwrite,
                store_local_before_var_overwrite: &mut store_local_before_var_overwrite,
                store_local_after_store: &mut store_local_after_store,
                store_local_after_fallback_store: &mut store_local_after_fallback_store,
            },
        )
        .await;
        let dispatch_result = match dispatch_result {
            Ok(result) => result,
            Err(err) => match interp_dispatch::redirect_exception_to_catch(
                err,
                &mut try_stack,
                &mut vars,
                &mut last_exception,
                &mut pc,
                refresh_workspace_state,
            ) {
                interp_dispatch::ExceptionHandling::Caught => {
                    continue;
                }
                interp_dispatch::ExceptionHandling::Uncaught(err) => return Err(*err),
            },
        };
        if let Some(decision) = dispatch_result {
            match decision {
                interp_dispatch::DispatchHandled::Generic(DispatchDecision::ContinueLoop) => {
                    continue;
                }
                interp_dispatch::DispatchHandled::Generic(DispatchDecision::FallThrough) => {
                    pc += 1;
                    continue;
                }
                interp_dispatch::DispatchHandled::Generic(DispatchDecision::Return) => {
                    interpreter_timing.flush_host_span("return", None);
                    break;
                }
                interp_dispatch::DispatchHandled::ReturnValue(DispatchDecision::ContinueLoop)
                | interp_dispatch::DispatchHandled::Return(DispatchDecision::ContinueLoop) => {
                    continue;
                }
                interp_dispatch::DispatchHandled::ReturnValue(DispatchDecision::Return) => {
                    interpreter_timing.flush_host_span("return_value", None);
                    break;
                }
                interp_dispatch::DispatchHandled::Return(DispatchDecision::Return) => {
                    interpreter_timing.flush_host_span("return", None);
                    break;
                }
                interp_dispatch::DispatchHandled::ReturnValue(DispatchDecision::FallThrough)
                | interp_dispatch::DispatchHandled::Return(DispatchDecision::FallThrough) => {
                    pc += 1;
                    continue;
                }
            }
        }
        match bytecode.instructions[pc].clone() {
            Instr::EmitStackTop { .. }
            | Instr::EmitVar { .. }
            | Instr::AndAnd(_)
            | Instr::OrOr(_)
            | Instr::JumpIfFalse(_)
            | Instr::Jump(_)
            | Instr::LoadConst(_)
            | Instr::LoadComplex(_, _)
            | Instr::LoadBool(_)
            | Instr::LoadString(_)
            | Instr::LoadCharRow(_)
            | Instr::LoadLocal(_)
            | Instr::LoadVar(_)
            | Instr::LoadVarForIndexAssignment(_)
            | Instr::StoreVar(_)
            | Instr::StoreLocal(_)
            | Instr::Swap
            | Instr::Pop
            | Instr::EnterTry { .. }
            | Instr::LeaveTry(_)
            | Instr::ReturnValue
            | Instr::Return
            | Instr::EnterScope(_)
            | Instr::LoadMember(_)
            | Instr::LoadMemberOrInit(_)
            | Instr::LoadMemberDynamic
            | Instr::LoadMemberDynamicOrInit
            | Instr::StoreMember(_)
            | Instr::StoreMemberOrInit(_)
            | Instr::StoreMemberDynamic
            | Instr::StoreMemberDynamicOrInit
            | Instr::Index(_)
            | Instr::IndexSlice(_, _, _, _)
            | Instr::IndexSliceExpr { .. }
            | Instr::IndexCell { .. }
            | Instr::IndexCellExpand { .. }
            | Instr::IndexCellList { .. }
            | Instr::StoreIndex(_)
            | Instr::StoreIndexCell { .. }
            | Instr::StoreIndexDelete(_)
            | Instr::StoreIndexCellDelete { .. }
            | Instr::StoreSlice(_, _, _, _)
            | Instr::StoreSliceDelete(_, _, _, _)
            | Instr::StoreSliceExpr { .. }
            | Instr::StoreSliceExprDelete { .. }
            | Instr::CallMethodOrMemberIndexMulti { .. }
            | Instr::CallMethodOrMemberIndexExpandMultiOutput { .. }
            | Instr::LoadMethod(_)
            | Instr::CreateFunctionHandle(_)
            | Instr::CreateExternalFunctionHandle(_)
            | Instr::CreateMethodFunctionHandle(_)
            | Instr::CreateBoundFunctionHandle(_, _)
            | Instr::CreateExternalBoundFunctionHandle(_, _)
            | Instr::CreateClosure(_, _)
            | Instr::CreateSemanticClosure(_, _, _)
            | Instr::LoadStaticProperty(_, _)
            | Instr::LoadWorkspaceFirstStaticProperty { .. }
            | Instr::RegisterClass { .. }
            | Instr::CallFevalMulti(_, _)
            | Instr::CallFevalMultiUsingOutputSlot(_, _)
            | Instr::CallFevalExpandMultiOutput(_, _)
            | Instr::CallFevalExpandMultiOutputUsingOutputSlot(_, _)
            | Instr::CreateSemanticFuture(_, _, _)
            | Instr::CreateSemanticFutureExpandMultiOutput(_, _, _)
            | Instr::Spawn
            | Instr::Await
            | Instr::CallBuiltinMulti(_, _, _)
            | Instr::CallBuiltinMultiUsingOutputSlot(_, _, _)
            | Instr::CallSuperConstructorMulti { .. }
            | Instr::CallSuperMethodMulti { .. }
            | Instr::CallSemanticFunctionMulti(_, _, _)
            | Instr::CallSemanticFunctionMultiUsingOutputSlot(_, _, _)
            | Instr::CallSemanticNestedFunctionMulti { .. }
            | Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { .. }
            | Instr::CallFunctionMulti { .. }
            | Instr::CallFunctionMultiUsingOutputSlot { .. }
            | Instr::CallFunctionExpandMultiOutput { .. }
            | Instr::CallWorkspaceFirstMulti { .. }
            | Instr::CallWorkspaceFirstMultiUsingOutputSlot { .. }
            | Instr::CallWorkspaceFirstExpandMultiOutput { .. }
            | Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot { .. }
            | Instr::CallSemanticFunctionExpandMultiOutput(_, _, _)
            | Instr::CallSemanticNestedFunctionExpandMultiOutput { .. }
            | Instr::CallBuiltinExpandMultiOutput(_, _, _)
            | Instr::CallSuperConstructorExpandMultiOutput { .. }
            | Instr::CallSuperMethodExpandMultiOutput { .. }
            | Instr::ExitScope(_)
            | Instr::RegisterImport { .. }
            | Instr::DeclareGlobal(_)
            | Instr::DeclareGlobalNamed(_, _)
            | Instr::DeclarePersistent(_)
            | Instr::DeclarePersistentNamed(_, _)
            | Instr::CreateCell2D(_, _)
            | Instr::CreateStructLiteral(_)
            | Instr::CreateObjectLiteral { .. }
            | Instr::Add
            | Instr::Sub
            | Instr::Mul
            | Instr::ElemMul
            | Instr::ElemDiv
            | Instr::ElemPow
            | Instr::ElemLeftDiv
            | Instr::Neg
            | Instr::UPlus
            | Instr::Transpose
            | Instr::ConjugateTranspose
            | Instr::Pow
            | Instr::RightDiv
            | Instr::LeftDiv
            | Instr::LessEqual
            | Instr::Less
            | Instr::Greater
            | Instr::GreaterEqual
            | Instr::Equal
            | Instr::NotEqual
            | Instr::LogicalNot
            | Instr::LogicalAnd
            | Instr::LogicalOr
            | Instr::Unpack(_)
            | Instr::CreateMatrix(_, _)
            | Instr::CreateMatrixDynamic(_)
            | Instr::CreateRange(_)
            | Instr::PackToRow(_)
            | Instr::PackToCol(_) => unreachable!("handled by dispatch_instruction"),
            Instr::StochasticEvolution => {
                let steps_value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let scale_value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let drift_value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let state_value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let evolved =
                    crate::accel::idioms::stochastic_evolution::execute_stochastic_evolution(
                        state_value,
                        drift_value,
                        scale_value,
                        steps_value,
                    )
                    .await?;
                stack.push(evolved);
            }
        }
        if debug_stack {
            debug!(pc, stack_len = stack.len(), "[vm] after exec");
        }
        pc += 1;
    }
    interpreter_timing.flush_host_span("loop_complete", None);
    #[cfg(feature = "native-accel")]
    {
        let mut live_values = Vec::with_capacity(vars.len() + context.locals.len());
        live_values.extend(vars.iter().cloned());
        live_values.extend(context.locals.iter().cloned());
        live_values.extend(runtime_globals::collect_thread_roots());
        let live_values = Value::OutputList(live_values);
        for value in &stack {
            if let Err(err) = accel_residency::clear_value_excluding(value, &live_values) {
                log::warn!("failed to clear stack GPU residency: {err}");
            }
        }
    }
    sync_initial_vars(initial_vars, &vars);
    Ok(InterpreterOutcome::Completed(vars))
}

pub async fn interpret(bytecode: &Bytecode) -> Result<Vec<Value>, RuntimeError> {
    let mut vars = vec![Value::Num(0.0); bytecode.var_count];
    match interpret_with_vars(bytecode, &mut vars, Some("<main>")).await {
        Ok(InterpreterOutcome::Completed(values)) => Ok(values),
        Err(e) => Err(e),
    }
}

pub async fn interpret_function(
    bytecode: &Bytecode,
    vars: Vec<Value>,
) -> Result<Vec<Value>, RuntimeError> {
    interpret_function_with_counts(bytecode, vars, "<anonymous>", 0, 0, HashSet::new()).await
}

pub async fn interpret_function_with_counts(
    bytecode: &Bytecode,
    vars: Vec<Value>,
    name: &str,
    out_count: usize,
    in_count: usize,
    missing_input_slots: HashSet<usize>,
) -> Result<Vec<Value>, RuntimeError> {
    let runtime = active_or_standalone_runtime_context();
    interpret_function_with_counts_in_context(
        bytecode,
        vars,
        name,
        out_count,
        in_count,
        missing_input_slots,
        runtime,
    )
    .await
}

async fn interpret_function_with_counts_in_context(
    bytecode: &Bytecode,
    vars: Vec<Value>,
    name: &str,
    out_count: usize,
    in_count: usize,
    missing_input_slots: HashSet<usize>,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<Vec<Value>, RuntimeError> {
    runtime
        .scope(interpret_function_with_counts_in_context_inner(
            bytecode,
            vars,
            name,
            out_count,
            in_count,
            missing_input_slots,
            runtime.clone(),
        ))
        .await
}

async fn interpret_function_with_counts_in_context_inner(
    bytecode: &Bytecode,
    vars: Vec<Value>,
    name: &str,
    out_count: usize,
    in_count: usize,
    missing_input_slots: HashSet<usize>,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<Vec<Value>, RuntimeError> {
    let mut vars = vars;
    CALL_COUNTS.with(|cc| {
        cc.borrow_mut().push((in_count, out_count));
    });
    let call_counts = CALL_COUNTS.with(|cc| cc.borrow().clone());
    let mut state = InterpreterState::new_in_context(
        bytecode.clone(),
        &mut vars,
        Some(name),
        call_counts,
        runtime,
    );
    state.missing_input_slots = missing_input_slots;
    let _debug_frame_guard = runmat_runtime::debug_context::push_frame(
        name,
        bytecode.source_id,
        bytecode_frame_span(bytecode),
    );
    let res = Box::pin(run_interpreter(Box::new(state), &mut vars)).await;
    CALL_COUNTS.with(|cc| {
        cc.borrow_mut().pop();
    });
    let res = match res {
        Ok(InterpreterOutcome::Completed(values)) => Ok(values),
        Err(e) => Err(e),
    }?;
    runtime_globals::persist_declared_for_bytecode(bytecode, name, &vars);
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::{interpret_with_vars, output_value, run_interpreter_inner};
    use crate::bytecode::program::Bytecode;
    use crate::bytecode::Instr;
    use crate::interpreter::api::InterpreterState;
    use futures::executor::block_on;
    use runmat_runtime::builtins::common::validation::{
        value_is_empty, value_is_greater_than, value_is_greater_than_or_equal, value_is_integer,
        value_is_less_than, value_is_less_than_or_equal, value_is_negative, value_is_nonnegative,
        value_is_nonpositive, value_is_nonzero, value_is_numeric_or_logical, value_is_positive,
        value_is_real, value_is_scalar_or_empty, value_is_text,
    };
    use runmat_value::{CellArray, HandleRef, StructValue, Tensor, Value};
    use std::sync::{atomic::AtomicBool, Arc};
    #[cfg(feature = "native-accel")]
    use {
        once_cell::sync::Lazy,
        runmat_accelerate::simple_provider::InProcessProvider,
        runmat_accelerate_api::{AccelProvider, HostTensorView, ThreadProviderGuard},
    };

    #[cfg(feature = "native-accel")]
    static TEST_PROVIDER: Lazy<InProcessProvider> = Lazy::new(InProcessProvider::new);

    #[cfg(feature = "native-accel")]
    fn upload_provider_handle(
        data: Vec<f64>,
        shape: Vec<usize>,
    ) -> runmat_accelerate_api::GpuTensorHandle {
        TEST_PROVIDER
            .upload(&HostTensorView {
                data: &data,
                shape: &shape,
            })
            .expect("upload should succeed")
    }

    #[test]
    fn output_value_zero_requested_is_empty_output_list() {
        let value = output_value(vec![Value::Num(1.0)], 0);
        assert_eq!(value, Value::OutputList(Vec::new()));
    }

    #[test]
    fn output_value_multi_requested_returns_output_list() {
        let value = output_value(vec![Value::Num(1.0), Value::Num(2.0)], 2);
        assert_eq!(
            value,
            Value::OutputList(vec![Value::Num(1.0), Value::Num(2.0)])
        );
    }

    #[test]
    fn numeric_or_logical_validator_accepts_expected_domains() {
        assert!(value_is_numeric_or_logical(&Value::Num(1.0)));
        assert!(value_is_numeric_or_logical(&Value::Bool(true)));
        assert!(value_is_numeric_or_logical(&Value::Complex(1.0, 2.0)));
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_numeric_or_logical(&Value::Tensor(tensor)));
        assert!(!value_is_numeric_or_logical(&Value::String(
            "x".to_string()
        )));
        assert!(!value_is_numeric_or_logical(&Value::CharArray(
            runmat_value::CharArray::new("x".chars().collect(), 1, 1).expect("char")
        )));
    }

    #[test]
    fn text_validator_accepts_string_char_vector_and_cellstr() {
        assert!(value_is_text(&Value::String("x".to_string())));
        assert!(value_is_text(&Value::CharArray(
            runmat_value::CharArray::new("abc".chars().collect(), 1, 3).expect("char")
        )));
        assert!(value_is_text(&Value::Cell(
            CellArray::new(
                vec![
                    Value::CharArray(
                        runmat_value::CharArray::new("a".chars().collect(), 1, 1).expect("char"),
                    ),
                    Value::String("b".to_string()),
                ],
                1,
                2,
            )
            .expect("cell"),
        )));
        assert!(!value_is_text(&Value::Num(1.0)));
    }

    #[test]
    fn nonempty_validator_rejects_empty_arrays_and_cells() {
        let empty_num = Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor");
        assert!(value_is_empty(&Value::Tensor(empty_num)));
        let empty_char = runmat_value::CharArray::new(Vec::new(), 1, 0).expect("empty char array");
        assert!(value_is_empty(&Value::CharArray(empty_char)));
        let empty_cell = CellArray::new(Vec::new(), 0, 0).expect("empty cell");
        assert!(value_is_empty(&Value::Cell(empty_cell)));
        assert!(!value_is_empty(&Value::String("".to_string())));
        assert!(!value_is_empty(&Value::Num(1.0)));
    }

    #[test]
    fn scalar_or_empty_validator_accepts_scalar_or_empty_shapes() {
        assert!(value_is_scalar_or_empty(&Value::Num(1.0)));
        assert!(value_is_scalar_or_empty(&Value::Bool(true)));
        let empty_num = Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor");
        assert!(value_is_scalar_or_empty(&Value::Tensor(empty_num)));
        let matrix = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("matrix");
        assert!(!value_is_scalar_or_empty(&Value::Tensor(matrix)));
    }

    #[test]
    fn real_validator_rejects_imaginary_values() {
        assert!(value_is_real(&Value::Num(1.0)));
        assert!(value_is_real(&Value::Complex(1.0, 0.0)));
        assert!(!value_is_real(&Value::Complex(1.0, 2.0)));
        let complex_real =
            runmat_value::ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).expect("complex tensor");
        let complex_imag =
            runmat_value::ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).expect("complex tensor");
        assert!(value_is_real(&Value::ComplexTensor(complex_real)));
        assert!(!value_is_real(&Value::ComplexTensor(complex_imag)));
    }

    #[test]
    fn integer_validator_accepts_integer_valued_numeric_and_logical_inputs() {
        assert!(value_is_integer(&Value::Int(runmat_value::IntValue::I64(
            3
        ))));
        assert!(value_is_integer(&Value::Num(3.0)));
        assert!(!value_is_integer(&Value::Num(3.5)));
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_integer(&Value::Tensor(tensor)));
        let non_integer = Tensor::new(vec![1.0, 2.5], vec![1, 2]).expect("tensor");
        assert!(!value_is_integer(&Value::Tensor(non_integer)));
        assert!(value_is_integer(&Value::Bool(true)));
        assert!(value_is_integer(&Value::LogicalArray(
            runmat_value::LogicalArray::new(vec![0, 1], vec![1, 2]).expect("logical array")
        )));
    }

    #[test]
    fn positive_validator_rejects_zero_and_negative_values() {
        assert!(value_is_positive(&Value::Num(1.0)));
        assert!(!value_is_positive(&Value::Num(0.0)));
        assert!(!value_is_positive(&Value::Num(-1.0)));
        assert!(value_is_positive(&Value::Int(runmat_value::IntValue::I64(
            2
        ))));
        assert!(!value_is_positive(&Value::Int(
            runmat_value::IntValue::I64(0)
        )));
        let positive = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_positive(&Value::Tensor(positive)));
        let mixed = Tensor::new(vec![1.0, 0.0], vec![1, 2]).expect("tensor");
        assert!(!value_is_positive(&Value::Tensor(mixed)));
    }

    #[test]
    fn negative_validator_rejects_zero_and_positive_values() {
        assert!(value_is_negative(&Value::Num(-1.0)));
        assert!(!value_is_negative(&Value::Num(0.0)));
        assert!(!value_is_negative(&Value::Num(1.0)));
        assert!(value_is_negative(&Value::Int(runmat_value::IntValue::I64(
            -2
        ))));
        let ok = Tensor::new(vec![-1.0, -2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_negative(&Value::Tensor(ok)));
        let bad = Tensor::new(vec![-1.0, 0.0], vec![1, 2]).expect("tensor");
        assert!(!value_is_negative(&Value::Tensor(bad)));
    }

    #[test]
    fn nonnegative_validator_accepts_zero_and_positive_values() {
        assert!(value_is_nonnegative(&Value::Num(0.0)));
        assert!(value_is_nonnegative(&Value::Num(2.0)));
        assert!(!value_is_nonnegative(&Value::Num(-1.0)));
        assert!(value_is_nonnegative(&Value::Int(
            runmat_value::IntValue::I64(0)
        )));
        let ok = Tensor::new(vec![0.0, 1.0], vec![1, 2]).expect("tensor");
        assert!(value_is_nonnegative(&Value::Tensor(ok)));
        let bad = Tensor::new(vec![0.0, -1.0], vec![1, 2]).expect("tensor");
        assert!(!value_is_nonnegative(&Value::Tensor(bad)));
    }

    #[test]
    fn nonzero_validator_rejects_zero_values() {
        assert!(value_is_nonzero(&Value::Num(1.0)));
        assert!(!value_is_nonzero(&Value::Num(0.0)));
        assert!(value_is_nonzero(&Value::Int(runmat_value::IntValue::I64(
            2
        ))));
        assert!(!value_is_nonzero(&Value::Int(runmat_value::IntValue::I64(
            0
        ))));
        assert!(value_is_nonzero(&Value::Complex(0.0, 1.0)));
        assert!(!value_is_nonzero(&Value::Complex(0.0, 0.0)));
        let ok = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_nonzero(&Value::Tensor(ok)));
        let bad = Tensor::new(vec![1.0, 0.0], vec![1, 2]).expect("tensor");
        assert!(!value_is_nonzero(&Value::Tensor(bad)));
    }

    #[test]
    fn nonpositive_validator_accepts_zero_and_negative_values() {
        assert!(value_is_nonpositive(&Value::Num(0.0)));
        assert!(value_is_nonpositive(&Value::Num(-2.0)));
        assert!(!value_is_nonpositive(&Value::Num(1.0)));
        assert!(value_is_nonpositive(&Value::Int(
            runmat_value::IntValue::I64(0)
        )));
        let ok = Tensor::new(vec![0.0, -1.0], vec![1, 2]).expect("tensor");
        assert!(value_is_nonpositive(&Value::Tensor(ok)));
        let bad = Tensor::new(vec![0.0, 1.0], vec![1, 2]).expect("tensor");
        assert!(!value_is_nonpositive(&Value::Tensor(bad)));
    }

    #[test]
    fn greater_than_or_equal_validator_uses_numeric_threshold() {
        assert!(value_is_greater_than_or_equal(&Value::Num(2.0), 0.0));
        assert!(value_is_greater_than_or_equal(&Value::Num(0.0), 0.0));
        assert!(!value_is_greater_than_or_equal(&Value::Num(-1.0), 0.0));
    }

    #[test]
    fn less_than_or_equal_validator_uses_numeric_threshold() {
        assert!(value_is_less_than_or_equal(&Value::Num(-1.0), 0.0));
        assert!(value_is_less_than_or_equal(&Value::Num(0.0), 0.0));
        assert!(!value_is_less_than_or_equal(&Value::Num(1.0), 0.0));
    }

    #[test]
    fn greater_than_and_less_than_validators_use_numeric_threshold() {
        assert!(value_is_greater_than(&Value::Num(2.0), 1.0));
        assert!(!value_is_greater_than(&Value::Num(1.0), 1.0));
        assert!(value_is_less_than(&Value::Num(-2.0), -1.0));
        assert!(!value_is_less_than(&Value::Num(-1.0), -1.0));
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn cancellation_clears_gpu_residency_for_live_values() {
        use runmat_accelerate::fusion_residency;
        use runmat_accelerate_api::GpuTensorHandle;

        let handle = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 777_001,
        };
        fusion_residency::mark(&handle);
        assert!(fusion_residency::is_resident(&handle));

        let mut vars = vec![Value::GpuTensor(handle.clone())];
        let bytecode = Bytecode::with_instructions(vec![Instr::Return], vars.len());
        let cancelled = Arc::new(AtomicBool::new(true));
        let _interrupt_guard = runmat_runtime::interrupt::replace_interrupt(Some(cancelled));

        let err = block_on(interpret_with_vars(&bytecode, &mut vars, Some("<main>")))
            .expect_err("cancelled execution should return error");
        assert_eq!(err.identifier(), Some("RunMat:ExecutionCancelled"));
        assert!(
            !fusion_residency::is_resident(&handle),
            "cancelled execution should clear residency marks for live GPU handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn completion_clears_stack_only_gpu_residency() {
        use runmat_accelerate::fusion_residency;
        use runmat_accelerate_api::GpuTensorHandle;

        let handle = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 777_002,
        };
        fusion_residency::mark(&handle);
        assert!(fusion_residency::is_resident(&handle));

        let bytecode = Bytecode::with_instructions(Vec::new(), 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::GpuTensor(handle.clone()));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let outcome = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("interpreter should complete");
        assert!(matches!(
            outcome,
            crate::interpreter::api::InterpreterOutcome::Completed(_)
        ));
        assert!(
            !fusion_residency::is_resident(&handle),
            "completion should clear residency marks for stack-only GPU handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn pop_releases_stack_only_provider_handle() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![9.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::Pop, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::GpuTensor(handle.clone()));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("interpreter should complete");
        assert!(
            !fusion_residency::is_resident(&handle),
            "pop should clear residency for stack-only handles"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "pop should release provider storage for stack-only handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn pop_preserves_provider_handle_when_still_live_in_vars() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![11.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::Pop, Instr::Return], 1);
        let mut seed_vars = vec![Value::GpuTensor(handle.clone())];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::GpuTensor(handle.clone()));
        state.vars = vec![Value::GpuTensor(handle.clone())];

        let mut result_vars = vec![Value::GpuTensor(handle.clone())];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("interpreter should complete");
        assert!(
            fusion_residency::is_resident(&handle),
            "pop should preserve residency for handles still referenced by vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "pop should not release provider storage for handles still referenced by vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn exit_scope_releases_local_only_provider_handle() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![15.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::ExitScope(1), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.context.locals.push(Value::GpuTensor(handle.clone()));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("exit scope should complete");
        assert!(
            !fusion_residency::is_resident(&handle),
            "exit scope should clear residency for local-only handles"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "exit scope should release provider storage for local-only handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn exit_scope_preserves_provider_handle_when_still_live_in_vars() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![17.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::ExitScope(1), Instr::Return], 1);
        let mut seed_vars = vec![Value::GpuTensor(handle.clone())];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.context.locals.push(Value::GpuTensor(handle.clone()));
        state.vars = vec![Value::GpuTensor(handle.clone())];

        let mut result_vars = vec![Value::GpuTensor(handle.clone())];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("exit scope should complete");
        assert!(
            fusion_residency::is_resident(&handle),
            "exit scope should preserve residency for handles still referenced by vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "exit scope should not release provider storage for handles still referenced by vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn exit_scope_releases_nested_handle_object_local_provider_handle() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![18.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::ExitScope(1), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        state.context.locals.push(Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        }));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("exit scope should complete for nested handle-object local");
        assert!(
            !fusion_residency::is_resident(&handle),
            "exit scope should clear residency for nested handle-object local-only handles"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "exit scope should release provider storage for nested handle-object local-only handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn exit_scope_preserves_nested_handle_object_provider_handle_when_still_live_in_vars() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![20.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::ExitScope(1), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let local_value = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        state.context.locals.push(local_value.clone());
        state.vars = vec![local_value.clone()];

        let mut result_vars = vec![local_value];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("exit scope should complete for aliased nested handle-object local");
        assert!(
            fusion_residency::is_resident(&handle),
            "exit scope should preserve residency for nested handle-object handles still referenced by vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "exit scope should not release provider storage for nested handle-object handles still referenced by vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_var_overwrite_preserves_provider_handle_when_shared_in_other_var() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![19.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreVar(0), Instr::Return], 2);
        let mut seed_vars = vec![
            Value::GpuTensor(handle.clone()),
            Value::GpuTensor(handle.clone()),
        ];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Num(0.0));
        state.vars = vec![
            Value::GpuTensor(handle.clone()),
            Value::GpuTensor(handle.clone()),
        ];

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store var should complete");
        assert!(
            fusion_residency::is_resident(&handle),
            "store var overwrite should preserve residency for handles still live in other vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store var overwrite should not release provider storage for handles still live in other vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_var_overwrite_preserves_provider_handle_when_shared_in_local() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![20.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreVar(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::GpuTensor(handle.clone())];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Num(0.0));
        state.vars = vec![Value::GpuTensor(handle.clone())];
        state.context.locals.push(Value::GpuTensor(handle.clone()));

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store var should complete when alias lives in locals");
        assert!(
            fusion_residency::is_resident(&handle),
            "store var overwrite should preserve residency for handles still live in locals"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store var overwrite should not release provider storage for handles still live in locals"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_var_overwrite_releases_nested_handle_object_provider_handle_when_unaliased() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![22.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreVar(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        state.vars = vec![Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        })];
        state.stack.push(Value::Num(0.0));

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store var overwrite should complete for nested handle-object value");
        assert!(
            !fusion_residency::is_resident(&handle),
            "store var overwrite should clear residency for nested handle-object handles when unaliased"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "store var overwrite should release provider storage for nested handle-object handles when unaliased"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_var_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_other_var()
    {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![24.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreVar(0), Instr::Return], 2);
        let mut seed_vars = vec![Value::Num(0.0), Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let nested = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        state.vars = vec![nested.clone(), nested.clone()];
        state.stack.push(Value::Num(0.0));

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store var overwrite should complete for aliased nested handle-object values");
        assert!(
            fusion_residency::is_resident(&handle),
            "store var overwrite should preserve residency for nested handle-object handles still live in other vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store var overwrite should not release provider storage for nested handle-object handles still live in other vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_var_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_local() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![27.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreVar(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let nested = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        state.vars = vec![nested.clone()];
        state.stack.push(Value::Num(0.0));
        state.context.locals.push(nested);

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store var overwrite should complete when alias lives in locals");
        assert!(
            fusion_residency::is_resident(&handle),
            "store var overwrite should preserve residency for nested handle-object handles still live in locals"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store var overwrite should not release provider storage for nested handle-object handles still live in locals"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_preserves_provider_handle_when_shared_in_var() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![23.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::GpuTensor(handle.clone())];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Num(0.0));
        state.vars = vec![Value::GpuTensor(handle.clone())];
        state
            .context
            .call_stack
            .push(crate::bytecode::program::CallFrame {
                function_name: "<local>".to_string(),
                return_address: 0,
                locals_start: 0,
                locals_count: 1,
                expected_outputs: 0,
            });
        state.context.locals.push(Value::GpuTensor(handle.clone()));

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store local should complete");
        assert!(
            fusion_residency::is_resident(&handle),
            "store local overwrite should preserve residency for handles still live in vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store local overwrite should not release provider storage for handles still live in vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_preserves_provider_handle_when_shared_in_other_local() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![24.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Num(0.0));
        state.vars = vec![Value::Num(0.0)];
        state
            .context
            .call_stack
            .push(crate::bytecode::program::CallFrame {
                function_name: "<local>".to_string(),
                return_address: 0,
                locals_start: 0,
                locals_count: 2,
                expected_outputs: 0,
            });
        state.context.locals.push(Value::GpuTensor(handle.clone()));
        state.context.locals.push(Value::GpuTensor(handle.clone()));

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store local should complete when alias lives in other local");
        assert!(
            fusion_residency::is_resident(&handle),
            "store local overwrite should preserve residency for handles still live in other locals"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store local overwrite should not release provider storage for handles still live in other locals"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_releases_provider_handle_when_unaliased() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![25.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Num(0.0));
        state.vars = vec![Value::Num(0.0)];
        state
            .context
            .call_stack
            .push(crate::bytecode::program::CallFrame {
                function_name: "<local>".to_string(),
                return_address: 0,
                locals_start: 0,
                locals_count: 1,
                expected_outputs: 0,
            });
        state.context.locals.push(Value::GpuTensor(handle.clone()));

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store local overwrite should complete");
        assert!(
            !fusion_residency::is_resident(&handle),
            "store local overwrite should clear residency for unaliased local handles"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "store local overwrite should release provider storage for unaliased local handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_releases_nested_handle_object_provider_handle_when_unaliased() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![26.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        state.stack.push(Value::Num(0.0));
        state.vars = vec![Value::Num(0.0)];
        state
            .context
            .call_stack
            .push(crate::bytecode::program::CallFrame {
                function_name: "<local>".to_string(),
                return_address: 0,
                locals_start: 0,
                locals_count: 1,
                expected_outputs: 0,
            });
        state.context.locals.push(Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        }));

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store local overwrite should complete for nested handle-object value");
        assert!(
            !fusion_residency::is_resident(&handle),
            "store local overwrite should clear residency for nested handle-object handles when unaliased"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "store local overwrite should release provider storage for nested handle-object handles when unaliased"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_var() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![28.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let local_value = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        state.stack.push(Value::Num(0.0));
        state.vars = vec![local_value.clone()];
        state
            .context
            .call_stack
            .push(crate::bytecode::program::CallFrame {
                function_name: "<local>".to_string(),
                return_address: 0,
                locals_start: 0,
                locals_count: 1,
                expected_outputs: 0,
            });
        state.context.locals.push(local_value);

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store local overwrite should complete for aliased nested handle-object value");
        assert!(
            fusion_residency::is_resident(&handle),
            "store local overwrite should preserve residency for nested handle-object handles still live in vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store local overwrite should not release provider storage for nested handle-object handles still live in vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_preserves_nested_handle_object_provider_alias() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![30.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let nested = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        state.stack.push(Value::Num(0.0));
        state.vars = vec![Value::Num(0.0)];
        state
            .context
            .call_stack
            .push(crate::bytecode::program::CallFrame {
                function_name: "<local>".to_string(),
                return_address: 0,
                locals_start: 0,
                locals_count: 2,
                expected_outputs: 0,
            });
        state.context.locals.push(nested.clone());
        state.context.locals.push(nested);

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store local overwrite should complete when alias lives in other local");
        assert!(
            fusion_residency::is_resident(&handle),
            "store local overwrite should preserve residency for nested handle-object handles still live in other locals"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store local overwrite should not release provider storage for nested handle-object handles still live in other locals"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }
}
