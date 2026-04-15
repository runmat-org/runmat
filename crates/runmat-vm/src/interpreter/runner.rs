use crate::accel::fusion as accel_fusion;
use crate::accel::residency as accel_residency;
use crate::bytecode::{Bytecode, Instr, UserFunction};
use crate::call::shared as call_shared;
use crate::call::user as call_user;
use crate::interpreter::api::{InterpreterOutcome, InterpreterState};
use crate::interpreter::dispatch::{self as interp_dispatch, DispatchDecision};
use crate::interpreter::engine as interp_engine;
use crate::interpreter::errors::{attach_span_from_pc, mex, set_vm_pc};
use crate::interpreter::timing::InterpreterTiming;
use crate::runtime::call_stack::attach_call_frames;
use crate::runtime::globals as runtime_globals;
use crate::runtime::workspace::{
    workspace_assign, workspace_clear, workspace_lookup, workspace_remove, workspace_snapshot,
};
use runmat_builtins::Value;
use runmat_runtime::{
    user_functions,
    workspace::{self as runtime_workspace, WorkspaceResolver},
    RuntimeError,
};
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Once;
use tracing::{debug, info_span};

#[cfg(feature = "native-accel")]
use runmat_accelerate::{activate_fusion_plan, active_group_plan_clone, deactivate_fusion_plan, set_current_pc};

#[cfg(feature = "native-accel")]
struct FusionPlanGuard;

#[cfg(feature = "native-accel")]
impl Drop for FusionPlanGuard {
    fn drop(&mut self) {
        deactivate_fusion_plan();
    }
}

type VmResult<T> = Result<T, RuntimeError>;

runmat_thread_local! {
    static CALL_COUNTS: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

runmat_thread_local! {
    static USER_FUNCTION_VARS: RefCell<Option<*mut Vec<Value>>> = const { RefCell::new(None) };
}

struct UserFunctionVarsGuard {
    previous: Option<*mut Vec<Value>>,
}

impl Drop for UserFunctionVarsGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        USER_FUNCTION_VARS.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

fn install_user_function_vars(vars: &mut Vec<Value>) -> UserFunctionVarsGuard {
    let vars_ptr = vars as *mut Vec<Value>;
    let previous = USER_FUNCTION_VARS.with(|slot| slot.borrow_mut().replace(vars_ptr));
    UserFunctionVarsGuard { previous }
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
    functions: &HashMap<String, UserFunction>,
    vars: &mut [Value],
) -> Result<Value, RuntimeError> {
    let func = call_shared::lookup_user_function(name, functions)?;
    let arg_count = args.len();
    call_shared::validate_user_function_arity(name, &func, arg_count)?;
    let prepared = call_shared::prepare_user_call(func, args, vars)?;
    let crate::call::shared::PreparedUserCall {
        func,
        var_map,
        func_program,
        func_vars,
    } = prepared;
    let func_bytecode = crate::compile(&func_program, functions)?;
    let func_result_vars =
        interpret_function_with_counts(&func_bytecode, func_vars, name, 1, arg_count).await?;
    Ok(call_shared::first_output_value(&func, &var_map, &func_result_vars))
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
    let functions = Arc::new(context.functions.clone());
    let _user_function_vars_guard = install_user_function_vars(&mut vars);
    let _user_function_guard = user_functions::install_user_function_invoker(Some(Arc::new(
        move |name: &str, args: &[Value]| {
            let name = name.to_string();
            let args = args.to_vec();
            let functions = Arc::clone(&functions);
            Box::pin(async move {
                let vars_ptr = USER_FUNCTION_VARS.with(|slot| *slot.borrow());
                let Some(vars_ptr) = vars_ptr else {
                    return Err(mex(
                        "InternalStateUnavailable",
                        "user function vars not installed",
                    ));
                };
                let vars = unsafe { &mut *vars_ptr };
                invoke_user_function_value(&name, &args, &functions, vars).await
            })
        },
    )));
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
                        accel_fusion::fusion_span_has_vm_barrier(&bytecode.instructions, &plan.group.span),
                        accel_fusion::fusion_span_live_result_count(&bytecode.instructions, &plan.group.span),
                    );
                }
                let span = plan.group.span.clone();
                let has_barrier = accel_fusion::fusion_span_has_vm_barrier(&bytecode.instructions, &span);
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
        let next_instr = bytecode.instructions.get(pc + 1);
        let call_counts_snapshot = CALL_COUNTS.with(|cc| cc.borrow().clone());
        let store_var_global_aliases = match &bytecode.instructions[pc] {
            Instr::StoreVar(_) => Some(global_aliases.clone()),
            _ => None,
        };
        if let Some(decision) = interp_dispatch::dispatch_instruction(
            &bytecode.instructions[pc],
            &mut stack,
            &mut vars,
            &bytecode.var_names,
            &mut context,
            &bytecode.functions,
            &mut try_stack,
            &mut last_exception,
            &mut imports,
            bytecode.source_id,
            bytecode.call_arg_spans.get(pc).cloned().flatten(),
            &call_counts_snapshot,
            &current_function_name,
            &mut global_aliases,
            &mut persistent_aliases,
            &mut pc,
            |value| {
                #[cfg(feature = "native-accel")]
                clear_residency(value);
            },
            |name, argv, functions, vars_ref| {
                Box::pin(async move {
                    let mut local_vars = vars_ref.clone();
                    invoke_user_function_value(name, &argv, functions, &mut local_vars).await
                })
            },
            |name, args, out_count| {
                Box::pin(async move {
                    if out_count == 1 {
                        call_user::try_builtin_fallback_single(&name, &args).await
                    } else {
                        call_user::try_builtin_fallback_multi(&name, &args, out_count).await
                    }
                })
            },
            |bc, vars, name, out_count, in_count| {
                Box::pin(async move {
                    interpret_function_with_counts(&bc, vars, &name, out_count, in_count).await
                })
            },
            |current, incoming| {
                #[cfg(feature = "native-accel")]
                if !same_gpu_handle(current, incoming) {
                    clear_residency(current);
                }
            },
            |stored_index, stored_value| {
                if let Some(ref aliases) = store_var_global_aliases {
                    runtime_globals::update_global_store(stored_index, stored_value, aliases);
                }
            },
            |current, incoming| {
                #[cfg(feature = "native-accel")]
                if !same_gpu_handle(current, incoming) {
                    clear_residency(current);
                }
            },
            |current, incoming| {
                #[cfg(feature = "native-accel")]
                if !same_gpu_handle(current, incoming) {
                    clear_residency(current);
                }
            },
            |func_name, stored_offset, stored_value| {
                runtime_globals::update_persistent_local_store(
                    func_name,
                    stored_offset,
                    stored_value,
                );
            },
            next_instr,
        )
        .await?
        {
            match decision {
                interp_dispatch::DispatchHandled::Generic(DispatchDecision::ContinueLoop) => continue,
                interp_dispatch::DispatchHandled::Generic(DispatchDecision::FallThrough) => {
                    pc += 1;
                    continue;
                }
                interp_dispatch::DispatchHandled::Generic(DispatchDecision::Return) => {
                    interpreter_timing.flush_host_span("return", None);
                    break;
                }
                interp_dispatch::DispatchHandled::ReturnValue(DispatchDecision::ContinueLoop)
                | interp_dispatch::DispatchHandled::Return(DispatchDecision::ContinueLoop) => continue,
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
            | Instr::StoreIndex(_)
            | Instr::StoreIndexCell(_)
            | Instr::StoreSlice(_, _, _, _)
            | Instr::StoreSliceExpr { .. }
            | Instr::CallMethod(_, _)
            | Instr::CallMethodOrMemberIndex(_, _)
            | Instr::LoadMethod(_)
            | Instr::CreateClosure(_, _)
            | Instr::LoadStaticProperty(_, _)
            | Instr::CallStaticMethod(_, _, _)
            | Instr::RegisterClass { .. }
            | Instr::CallFeval(_)
            | Instr::CallFevalExpandMulti(_)
            | Instr::CallBuiltin(_, _)
            | Instr::CallFunction(_, _)
            | Instr::CallFunctionMulti(_, _, _)
            | Instr::CallFunctionExpandMulti(_, _)
            | Instr::CallBuiltinExpandLast(_, _, _)
            | Instr::CallBuiltinExpandAt(_, _, _, _)
            | Instr::CallBuiltinExpandMulti(_, _)
            | Instr::CallFunctionExpandAt(_, _, _, _)
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
                let evolved = crate::accel::idioms::stochastic_evolution::execute_stochastic_evolution(
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
