use crate::accel::fusion as accel_fusion;
use crate::accel::residency as accel_residency;
use crate::bytecode::{Bytecode, Instr, SemanticFunctionRegistry};
use crate::call::descriptor::{execute_callable_descriptor, CallableCallKind, CallableDescriptor};
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
use runmat_builtins::{CellArray, Value};
use runmat_hir::CallableFallbackPolicy;
use runmat_runtime::{
    user_functions,
    workspace::{self as runtime_workspace, WorkspaceResolver},
    RuntimeError,
};
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Once;
use tracing::{debug, info_span};

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

fn invoke_user_for_end_expr_adapter<'a>(
    name: &'a str,
    argv: Vec<Value>,
    vars_ref: &'a [Value],
) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>> {
    Box::pin(async move {
        let mut local_vars = vars_ref.to_owned();
        let registry_ptr = CURRENT_SEMANTIC_REGISTRY.with(|slot| *slot.borrow());
        if let Some(registry_ptr) = registry_ptr {
            // The guard installed for the active interpreter frame owns this pointer lifetime.
            let semantic_registry = unsafe { &*registry_ptr };
            invoke_user_function_value(name, &argv, semantic_registry, &mut local_vars).await
        } else {
            let semantic_registry = SemanticFunctionRegistry::default();
            invoke_user_function_value(name, &argv, &semantic_registry, &mut local_vars).await
        }
    })
}

runmat_thread_local! {
    static CALL_COUNTS: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

runmat_thread_local! {
    static CURRENT_SEMANTIC_REGISTRY: RefCell<Option<*const SemanticFunctionRegistry>> = const { RefCell::new(None) };
}

struct SemanticRegistryGuard {
    previous: Option<*const SemanticFunctionRegistry>,
}

impl Drop for SemanticRegistryGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        CURRENT_SEMANTIC_REGISTRY.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

fn install_semantic_registry(registry: &SemanticFunctionRegistry) -> SemanticRegistryGuard {
    let registry_ptr = registry as *const SemanticFunctionRegistry;
    let previous = CURRENT_SEMANTIC_REGISTRY.with(|slot| slot.borrow_mut().replace(registry_ptr));
    SemanticRegistryGuard { previous }
}

fn sync_initial_vars(initial: &mut [Value], vars: &[Value]) {
    for (i, var) in vars.iter().enumerate() {
        if i < initial.len() {
            initial[i] = var.clone();
        }
    }
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
    accel_residency::clear_value(value);
}

#[cfg(feature = "native-accel")]
fn same_gpu_handle(lhs: &Value, rhs: &Value) -> bool {
    accel_residency::same_gpu_handle(lhs, rhs)
}

async fn invoke_user_function_value(
    name: &str,
    args: &[Value],
    semantic_registry: &SemanticFunctionRegistry,
    _vars: &mut [Value],
) -> Result<Value, RuntimeError> {
    if let Some(function) = semantic_registry.resolve_name(name) {
        return execute_callable_descriptor(CallableDescriptor::semantic_named(
            function,
            name.to_string(),
            args.to_vec(),
            1,
        ))
        .await;
    }
    execute_callable_descriptor(CallableDescriptor::dynamic_named(
        name.to_string(),
        args.to_vec(),
        1,
        CallableFallbackPolicy::RuntimeNameResolution,
        CallableCallKind::EndExpr,
    ))
    .await
}

pub async fn invoke_semantic_function_value(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    semantic_registry: &SemanticFunctionRegistry,
) -> Result<Value, RuntimeError> {
    let function_id = runmat_hir::FunctionId(function);
    let func = semantic_registry.get(function_id).ok_or_else(|| {
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
    let runtime_arg_count = args.len() - func.capture_slots.len();
    if runtime_arg_count < func.input_slots.len() {
        let message = format!(
            "semantic function {} expected {} inputs, got {}",
            func.display_name,
            func.input_slots.len(),
            runtime_arg_count
        );
        return Err(mex("NotEnoughInputs", &message));
    }
    if runtime_arg_count > func.input_slots.len() && func.varargin_slot.is_none() {
        let message = format!(
            "semantic function {} expected {} inputs, got {}",
            func.display_name,
            func.input_slots.len(),
            runtime_arg_count
        );
        return Err(mex("TooManyInputs", &message));
    }
    if requested_outputs > func.output_slots.len() && func.varargout_slot.is_none() {
        let message = format!(
            "semantic function {} expected {} outputs, got {}",
            func.display_name,
            func.output_slots.len(),
            requested_outputs
        );
        return Err(mex("TooManyOutputs", &message));
    }

    let mut vars = vec![Value::Num(0.0); func.var_count];
    for (slot, value) in func.capture_slots.iter().zip(args.iter()) {
        if *slot < vars.len() {
            vars[*slot] = value.clone();
        }
    }
    for (slot, value) in func
        .input_slots
        .iter()
        .zip(args.iter().skip(func.capture_slots.len()))
    {
        if *slot < vars.len() {
            vars[*slot] = value.clone();
        }
    }
    if let Some(slot) = func.varargin_slot {
        let fixed_count = func.input_slots.len();
        let rest = if runtime_arg_count > fixed_count {
            args[func.capture_slots.len() + fixed_count..].to_vec()
        } else {
            Vec::new()
        };
        let cols = rest.len();
        let cell = CellArray::new(rest, 1, cols)
            .map_err(|err| mex("VararginPack", &format!("varargin: {err}")))?;
        if slot < vars.len() {
            vars[slot] = Value::Cell(cell);
        }
    }

    let mut bytecode = Bytecode::with_instructions(func.instructions.clone(), func.var_count);
    bytecode.instr_spans = func.instr_spans.clone();
    bytecode.call_arg_spans = func.call_arg_spans.clone();
    bytecode.semantic_functions = semantic_registry.functions.clone();
    bytecode.semantic_function_registry = semantic_registry.clone();
    let result_vars = interpret_function_with_counts(
        &bytecode,
        vars,
        &func.display_name,
        requested_outputs,
        runtime_arg_count,
    )
    .await?;
    let output_values = collect_semantic_outputs(func, &result_vars, requested_outputs)?;
    Ok(semantic_output_value(output_values, requested_outputs))
}

fn collect_semantic_outputs(
    func: &crate::bytecode::program::SemanticFunctionBytecode,
    result_vars: &[Value],
    requested_outputs: usize,
) -> Result<Vec<Value>, RuntimeError> {
    let mut values = Vec::with_capacity(requested_outputs.max(1));
    for slot in func.output_slots.iter().take(requested_outputs) {
        values.push(result_vars.get(*slot).cloned().unwrap_or(Value::Num(0.0)));
    }
    if values.len() < requested_outputs {
        if let Some(slot) = func.varargout_slot {
            let available = match result_vars.get(slot) {
                Some(Value::Cell(cell)) => {
                    let expanded = crate::call::shared::expand_all_cell(cell)?;
                    let available = expanded.len();
                    for value in expanded {
                        if values.len() >= requested_outputs {
                            break;
                        }
                        values.push(value);
                    }
                    available
                }
                _ => 0,
            };
            if values.len() < requested_outputs {
                let need = requested_outputs - func.output_slots.len();
                let message = format!(
                    "Function '{}' returned {available} varargout values, {need} requested",
                    func.display_name
                );
                return Err(mex("VarargoutMismatch", &message));
            }
        }
    }
    while values.len() < requested_outputs {
        values.push(Value::Num(0.0));
    }
    Ok(values)
}

fn semantic_output_value(output_values: Vec<Value>, requested_outputs: usize) -> Value {
    match requested_outputs {
        0 => Value::OutputList(Vec::new()),
        1 => output_values.into_iter().next().unwrap_or(Value::Num(0.0)),
        _ => Value::OutputList(output_values.into_iter().take(requested_outputs).collect()),
    }
}

pub async fn interpret_with_vars(
    bytecode: &Bytecode,
    initial_vars: &mut [Value],
    current_function_name: Option<&str>,
) -> VmResult<InterpreterOutcome> {
    let call_counts = CALL_COUNTS.with(|cc| cc.borrow().clone());
    let state = Box::new(InterpreterState::new(
        bytecode.clone(),
        initial_vars,
        current_function_name,
        call_counts,
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

async fn run_interpreter(
    state: Box<InterpreterState>,
    initial_vars: &mut [Value],
) -> VmResult<InterpreterOutcome> {
    let state = *state;
    Box::pin(run_interpreter_inner(state, initial_vars)).await
}

async fn run_interpreter_inner(
    state: InterpreterState,
    initial_vars: &mut [Value],
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
        current_function_name,
        call_counts,
        #[cfg(feature = "native-accel")]
            fusion_plan: _,
        bytecode,
    } = state;
    let semantic_registry = Arc::new(bytecode.semantic_registry());
    let _semantic_registry_guard = install_semantic_registry(&semantic_registry);
    let semantic_registry_for_semantic_invoker = Arc::clone(&semantic_registry);
    let _semantic_function_guard =
        user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function: usize, args: &[Value], requested_outputs: usize| {
                let args = args.to_vec();
                let semantic_registry = Arc::clone(&semantic_registry_for_semantic_invoker);
                Box::pin(async move {
                    invoke_semantic_function_value(
                        function,
                        &args,
                        requested_outputs,
                        &semantic_registry,
                    )
                    .await
                })
            },
        )));
    let semantic_registry_for_semantic_resolver = Arc::clone(&semantic_registry);
    let _semantic_resolver_guard =
        user_functions::install_semantic_function_resolver(Some(Arc::new(move |name: &str| {
            semantic_registry_for_semantic_resolver
                .resolve_name(name)
                .map(|function| function.0)
        })));
    CALL_COUNTS.with(|cc| {
        *cc.borrow_mut() = call_counts.clone();
    });
    let _workspace_guard = interp_engine::prepare_workspace_guard(&mut vars);
    let thread_roots: Vec<Value> = runtime_globals::collect_thread_roots();
    let mut _gc_context = interp_engine::create_gc_context(&stack, &vars, thread_roots)?;
    let debug_stack = interp_engine::debug_stack_enabled();
    let mut interpreter_timing = InterpreterTiming::new();
    while pc < bytecode.instructions.len() {
        set_vm_pc(pc);
        #[cfg(feature = "native-accel")]
        set_current_pc(pc);
        interp_engine::check_cancelled()?;
        #[cfg(feature = "native-accel")]
        if let (Some(plan), Some(graph)) =
            (active_group_plan_clone(), bytecode.accel_graph.as_ref())
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
        let mut clear_value_residency = |value: &Value| {
            #[cfg(feature = "native-accel")]
            clear_residency(value);
        };
        let mut store_var_before_overwrite = |current: &Value, incoming: &Value| {
            #[cfg(feature = "native-accel")]
            if !same_gpu_handle(current, incoming) {
                clear_residency(current);
            }
        };
        let mut store_var_after_store = |stored_index: usize, stored_value: &Value| {
            if let Some(ref aliases) = store_var_global_aliases {
                runtime_globals::update_global_store(stored_index, stored_value, aliases);
            }
        };
        let mut store_local_before_local_overwrite = |current: &Value, incoming: &Value| {
            #[cfg(feature = "native-accel")]
            if !same_gpu_handle(current, incoming) {
                clear_residency(current);
            }
        };
        let mut store_local_before_var_overwrite = |current: &Value, incoming: &Value| {
            #[cfg(feature = "native-accel")]
            if !same_gpu_handle(current, incoming) {
                clear_residency(current);
            }
        };
        let mut store_local_after_fallback_store =
            |func_name: &str, stored_offset: usize, stored_value: &Value| {
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
                semantic_registry: &semantic_registry,
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
                pc: &mut pc,
            },
            interp_dispatch::DispatchHooks {
                clear_value_residency: &mut clear_value_residency,
                invoke_user_for_end_expr: &invoke_user_for_end_expr_adapter,
                store_var_before_overwrite: &mut store_var_before_overwrite,
                store_var_after_store: &mut store_var_after_store,
                store_local_before_local_overwrite: &mut store_local_before_local_overwrite,
                store_local_before_var_overwrite: &mut store_local_before_var_overwrite,
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
                    continue
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
                    continue
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
            | Instr::StoreVar(_)
            | Instr::StoreLocal(_)
            | Instr::Swap
            | Instr::Pop
            | Instr::EnterTry(_, _)
            | Instr::PopTry
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
            | Instr::IndexCell(_)
            | Instr::IndexCellExpand(_, _)
            | Instr::IndexCellList(_)
            | Instr::StoreIndex(_)
            | Instr::StoreIndexCell(_)
            | Instr::StoreSlice(_, _, _, _)
            | Instr::StoreSliceExpr { .. }
            | Instr::CallMethodOrMemberIndexMulti(_, _, _)
            | Instr::CallMethodOrMemberIndexExpandMultiOutput(_, _, _)
            | Instr::LoadMethod(_)
            | Instr::CreateFunctionHandle(_)
            | Instr::CreateSemanticFunctionHandle(_, _)
            | Instr::CreateClosure(_, _)
            | Instr::CreateSemanticClosure(_, _, _)
            | Instr::LoadStaticProperty(_, _)
            | Instr::RegisterClass { .. }
            | Instr::CallFevalMulti(_, _)
            | Instr::CallFevalExpandMultiOutput(_, _)
            | Instr::CallBuiltinMulti(_, _, _)
            | Instr::CallSemanticFunction(_, _)
            | Instr::CallSemanticFunctionMulti(_, _, _)
            | Instr::CallFunctionMulti(_, _, _)
            | Instr::CallFunctionExpandMultiOutput(_, _, _)
            | Instr::CallSemanticFunctionExpandMultiOutput(_, _, _)
            | Instr::CallBuiltinExpandMultiOutput(_, _, _)
            | Instr::ExitScope(_)
            | Instr::RegisterImport { .. }
            | Instr::DeclareGlobal(_)
            | Instr::DeclareGlobalNamed(_, _)
            | Instr::DeclarePersistent(_)
            | Instr::DeclarePersistentNamed(_, _)
            | Instr::CreateCell2D(_, _)
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
    interpret_function_with_counts(bytecode, vars, "<anonymous>", 0, 0).await
}

pub async fn interpret_function_with_counts(
    bytecode: &Bytecode,
    vars: Vec<Value>,
    name: &str,
    out_count: usize,
    in_count: usize,
) -> Result<Vec<Value>, RuntimeError> {
    let mut vars = vars;
    CALL_COUNTS.with(|cc| {
        cc.borrow_mut().push((in_count, out_count));
    });
    let res = Box::pin(interpret_with_vars(bytecode, &mut vars, Some(name))).await;
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
    use super::{collect_semantic_outputs, semantic_output_value};
    use crate::bytecode::program::SemanticFunctionBytecode;
    use crate::bytecode::Instr;
    use runmat_builtins::{CellArray, Value};
    use runmat_hir::FunctionId;

    fn test_function(varargout_slot: Option<usize>) -> SemanticFunctionBytecode {
        SemanticFunctionBytecode {
            function: FunctionId(0),
            display_name: "f".into(),
            source_id: None,
            instructions: vec![Instr::Return],
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            var_count: 1,
            input_slots: Vec::new(),
            varargin_slot: None,
            output_slots: Vec::new(),
            varargout_slot,
            capture_slots: Vec::new(),
        }
    }

    #[test]
    fn collect_outputs_zero_requested_does_not_consume_varargout() {
        let func = test_function(Some(0));
        let varargout = CellArray::new(vec![Value::Num(7.0)], 1, 1).expect("cell");
        let result_vars = vec![Value::Cell(varargout)];
        let outputs = collect_semantic_outputs(&func, &result_vars, 0).expect("collect");
        assert!(outputs.is_empty());
    }

    #[test]
    fn collect_outputs_one_requested_reads_varargout() {
        let func = test_function(Some(0));
        let varargout = CellArray::new(vec![Value::Num(7.0)], 1, 1).expect("cell");
        let result_vars = vec![Value::Cell(varargout)];
        let outputs = collect_semantic_outputs(&func, &result_vars, 1).expect("collect");
        assert_eq!(outputs, vec![Value::Num(7.0)]);
    }

    #[test]
    fn semantic_output_value_zero_requested_is_empty_output_list() {
        let value = semantic_output_value(vec![Value::Num(1.0)], 0);
        assert_eq!(value, Value::OutputList(Vec::new()));
    }

    #[test]
    fn semantic_output_value_multi_requested_returns_output_list() {
        let value = semantic_output_value(vec![Value::Num(1.0), Value::Num(2.0)], 2);
        assert_eq!(
            value,
            Value::OutputList(vec![Value::Num(1.0), Value::Num(2.0)])
        );
    }
}
