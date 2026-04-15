use crate::functions::{Bytecode, ExecutionContext, UserFunction};
use crate::instr::Instr;
#[cfg(test)]
use crate::instr::EndExpr;
#[cfg(feature = "native-accel")]
use runmat_accelerate::fusion_exec::{
    execute_centered_gram, execute_elementwise, execute_explained_variance,
    execute_image_normalize, execute_matmul_epilogue, execute_power_step_normalize,
    execute_reduction,
};
#[cfg(feature = "native-accel")]
use runmat_accelerate::{
    activate_fusion_plan, deactivate_fusion_plan, set_current_pc,
};
#[cfg(feature = "native-accel")]
use runmat_accelerate::{
    active_group_plan_clone, value_is_all_keyword, FusionKind, ReductionAxes, ShapeInfo,
    ValueOrigin, VarKind,
};
use runmat_builtins::Value;
use runmat_runtime::{
    user_functions,
    workspace::{self as runtime_workspace, WorkspaceResolver},
    RuntimeError,
};
#[cfg(test)]
use runmat_runtime::{build_runtime_error, builtins::common::shape::is_scalar_shape};
use runmat_thread_local::runmat_thread_local;
pub use runmat_vm::interpreter::api::{
    push_pending_workspace, set_call_stack_limit, set_error_namespace,
    take_updated_workspace_state, InterpreterOutcome, InterpreterState, PendingWorkspaceGuard,
    DEFAULT_CALLSTACK_LIMIT, DEFAULT_ERROR_NAMESPACE,
};
use runmat_vm::interpreter::engine as interp_engine;
use runmat_vm::interpreter::dispatch::{self as interp_dispatch, DispatchDecision};
use runmat_vm::interpreter::errors::{
    attach_span_from_pc, mex, set_vm_pc,
};
use runmat_vm::call::shared as call_shared;
use runmat_vm::call::user as call_user;
use runmat_vm::accel::fusion as accel_fusion;
use runmat_vm::accel::residency as accel_residency;
#[cfg(test)]
use runmat_vm::indexing::end_expr as idx_end_expr;
#[cfg(test)]
use runmat_vm::indexing::selectors as idx_selectors;
use runmat_vm::interpreter::timing::InterpreterTiming;
use runmat_vm::runtime::call_stack::{
    attach_call_frames,
};
use runmat_vm::runtime::globals as runtime_globals;
use runmat_vm::runtime::workspace::{
    workspace_assign, workspace_clear, workspace_lookup, workspace_remove, workspace_snapshot,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Once;
use tracing::{debug, info_span};

#[cfg(feature = "native-accel")]
struct FusionPlanGuard;

#[cfg(feature = "native-accel")]
impl Drop for FusionPlanGuard {
    fn drop(&mut self) {
        deactivate_fusion_plan();
    }
}

type VmResult<T> = Result<T, RuntimeError>;

#[cfg(test)]
#[derive(Clone)]
enum SliceSelector {
    Colon,
    Scalar(usize),
    Indices(Vec<usize>),
    LinearIndices {
        values: Vec<usize>,
        output_shape: Vec<usize>,
    },
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct SlicePlan {
    indices: Vec<u32>,
    output_shape: Vec<usize>,
    selection_lengths: Vec<usize>,
    dims: usize,
}

#[cfg(test)]
fn from_vm_slice_selector(selector: idx_selectors::SliceSelector) -> SliceSelector {
    match selector {
        idx_selectors::SliceSelector::Colon => SliceSelector::Colon,
        idx_selectors::SliceSelector::Scalar(i) => SliceSelector::Scalar(i),
        idx_selectors::SliceSelector::Indices(v) => SliceSelector::Indices(v),
        idx_selectors::SliceSelector::LinearIndices { values, output_shape } => {
            SliceSelector::LinearIndices { values, output_shape }
        }
    }
}

#[cfg(test)]
fn to_vm_slice_selector(selector: &SliceSelector) -> idx_selectors::SliceSelector {
    match selector {
        SliceSelector::Colon => idx_selectors::SliceSelector::Colon,
        SliceSelector::Scalar(i) => idx_selectors::SliceSelector::Scalar(*i),
        SliceSelector::Indices(v) => idx_selectors::SliceSelector::Indices(v.clone()),
        SliceSelector::LinearIndices { values, output_shape } => {
            idx_selectors::SliceSelector::LinearIndices {
                values: values.clone(),
                output_shape: output_shape.clone(),
            }
        }
    }
}

#[cfg(test)]
fn from_vm_slice_plan(plan: idx_selectors::SlicePlan) -> SlicePlan {
    SlicePlan {
        indices: plan.indices,
        output_shape: plan.output_shape,
        selection_lengths: plan.selection_lengths,
        dims: plan.dims,
    }
}


#[cfg(test)]
fn total_len_from_shape(shape: &[usize]) -> usize {
    if is_scalar_shape(shape) {
        1
    } else {
        shape.iter().copied().product()
    }
}

#[derive(Clone, Copy)]
#[cfg(test)]
struct IndexContext<'a> {
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    base_shape: &'a [usize],
}

#[cfg(test)]
impl<'a> IndexContext<'a> {
    fn new(dims: usize, colon_mask: u32, end_mask: u32, base_shape: &'a [usize]) -> Self {
        Self {
            dims,
            colon_mask,
            end_mask,
            base_shape,
        }
    }

    fn dim_len_for_numeric_position(&self, numeric_position: usize) -> usize {
        let mut seen_numeric = 0usize;
        let mut dim_for_pos = 0usize;
        for d in 0..self.dims {
            let is_colon = (self.colon_mask & (1u32 << d)) != 0;
            let is_end = (self.end_mask & (1u32 << d)) != 0;
            if is_colon || is_end {
                continue;
            }
            if seen_numeric == numeric_position {
                dim_for_pos = d;
                break;
            }
            seen_numeric += 1;
        }
        if self.dims == 1 {
            let n = self.base_shape.iter().copied().product::<usize>();
            n.max(1)
        } else {
            self.base_shape.get(dim_for_pos).copied().unwrap_or(1)
        }
    }
}

#[cfg(test)]
async fn index_scalar_from_value(value: &Value) -> VmResult<Option<i64>> {
    idx_selectors::index_scalar_from_value(value).await
}

#[cfg(test)]
async fn selector_from_value_dim(value: &Value, dim_len: usize) -> VmResult<SliceSelector> {
    idx_selectors::selector_from_value_dim(value, dim_len)
        .await
        .map(from_vm_slice_selector)
}

#[cfg(test)]
async fn build_slice_selectors(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
) -> VmResult<Vec<SliceSelector>> {
    idx_selectors::build_slice_selectors(dims, colon_mask, end_mask, numeric, base_shape)
        .await
        .map(|selectors| selectors.into_iter().map(from_vm_slice_selector).collect())
}

#[cfg(test)]
fn build_slice_plan(
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
) -> VmResult<SlicePlan> {
    let vm_selectors: Vec<_> = selectors.iter().map(to_vm_slice_selector).collect();
    idx_selectors::build_slice_plan(&vm_selectors, dims, base_shape).map(from_vm_slice_plan)
}

#[cfg(test)]
fn apply_end_offsets_to_numeric<'a>(
    numeric: &'a [Value],
    ctx: IndexContext<'a>,
    end_offsets: &'a [(usize, EndExpr)],
    vars: &'a mut Vec<Value>,
    functions: &'a HashMap<String, UserFunction>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = VmResult<Vec<Value>>> + 'a>> {
    Box::pin(async move {
        let mut adjusted = numeric.to_vec();
        for (position, end_expr) in end_offsets {
            if let Some(value) = adjusted.get_mut(*position) {
                let dim_len = ctx.dim_len_for_numeric_position(*position);
                let idx_val = resolve_range_end_index(dim_len, end_expr, vars, functions).await?;
                *value = Value::Num(idx_val as f64);
            }
        }
        Ok(adjusted)
    })
}

#[cfg(test)]
async fn resolve_range_end_index(
    dim_len: usize,
    end_expr: &EndExpr,
    vars: &Vec<Value>,
    functions: &HashMap<String, UserFunction>,
) -> VmResult<i64> {
    idx_end_expr::resolve_range_end_index(
        dim_len,
        end_expr,
        vars,
        functions,
        &|name, argv| {
            Box::pin(async move {
                match runmat_runtime::call_builtin_async(name, &argv).await {
                    Ok(v) => Ok(Some(v)),
                    Err(_) => Ok(None),
                }
            })
        },
        &|name, argv, functions, vars| {
            Box::pin(async move {
                let mut local_vars = vars.clone();
                invoke_user_function_value(name, &argv, functions, &mut local_vars).await
            })
        },
    )
    .await
}

#[cfg(test)]
fn encode_end_expr_value(expr: &EndExpr) -> VmResult<Value> {
    fn mk_cell(items: Vec<Value>) -> VmResult<Value> {
        let cols = items.len();
        let cell = runmat_builtins::CellArray::new(items, 1, cols)
            .map_err(|e| format!("end expression encoding: {e}"))?;
        Ok(Value::Cell(cell))
    }

    match expr {
        EndExpr::End => Ok(Value::String("end".to_string())),
        EndExpr::Const(v) => Ok(Value::Num(*v)),
        EndExpr::Var(i) => Ok(Value::String(format!("var:{i}"))),
        EndExpr::Call(name, args) => {
            let mut items = vec![
                Value::String("call".to_string()),
                Value::String(name.clone()),
            ];
            for a in args {
                items.push(encode_end_expr_value(a)?);
            }
            mk_cell(items)
        }
        EndExpr::Add(a, b) => mk_cell(vec![
            Value::String("+".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::Sub(a, b) => mk_cell(vec![
            Value::String("-".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::Mul(a, b) => mk_cell(vec![
            Value::String("*".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::Div(a, b) => mk_cell(vec![
            Value::String("/".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::LeftDiv(a, b) => mk_cell(vec![
            Value::String("\\".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::Pow(a, b) => mk_cell(vec![
            Value::String("^".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::Neg(a) => mk_cell(vec![
            Value::String("neg".to_string()),
            encode_end_expr_value(a)?,
        ]),
        EndExpr::Pos(a) => mk_cell(vec![
            Value::String("pos".to_string()),
            encode_end_expr_value(a)?,
        ]),
        EndExpr::Floor(a) => mk_cell(vec![
            Value::String("floor".to_string()),
            encode_end_expr_value(a)?,
        ]),
        EndExpr::Ceil(a) => mk_cell(vec![
            Value::String("ceil".to_string()),
            encode_end_expr_value(a)?,
        ]),
        EndExpr::Round(a) => mk_cell(vec![
            Value::String("round".to_string()),
            encode_end_expr_value(a)?,
        ]),
        EndExpr::Fix(a) => mk_cell(vec![
            Value::String("fix".to_string()),
            encode_end_expr_value(a)?,
        ]),
    }
}

#[cfg(test)]
fn build_end_range_descriptor(start: Value, step: Value, end_expr: &EndExpr) -> VmResult<Value> {
    let encoded_end = encode_end_expr_value(end_expr)?;
    let cell = runmat_builtins::CellArray::new(
        vec![
            start,
            step,
            Value::String("end_expr".to_string()),
            encoded_end,
        ],
        1,
        4,
    )
    .map_err(|e| format!("obj range: {e}"))?;
    Ok(Value::Cell(cell))
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

runmat_thread_local! {
    // (nargin, nargout) for current call
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
    // Helper to resolve unqualified static accesses if Class.* is imported
    let _resolve_static =
        |imports: &Vec<(Vec<String>, bool)>, name: &str| -> Option<(String, String)> {
            // Return (class_name, member) for unqualified 'member' where Class.* imported
            for (path, wildcard) in imports {
                if !*wildcard {
                    continue;
                }
                if path.len() == 1 {
                    // Class.* style
                    let class_name = path[0].clone();
                    // We cannot know member names here; VM paths for LoadMember/CallMethod will enforce static
                    return Some((class_name, name.to_string()));
                }
            }
            None
        };
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
                    match try_execute_fusion_group(
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
                let evolved = stochastic_evolution_dispatch(
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

async fn stochastic_evolution_dispatch(
    state: Value,
    drift: Value,
    scale: Value,
    steps: Value,
) -> VmResult<Value> {
    runmat_vm::accel::idioms::stochastic_evolution::execute_stochastic_evolution(
        state, drift, scale, steps,
    )
    .await
}

#[cfg(feature = "native-accel")]
async fn try_execute_fusion_group(
    plan: &runmat_accelerate::FusionGroupPlan,
    graph: &runmat_accelerate::AccelGraph,
    stack: &mut Vec<Value>,
    vars: &mut Vec<Value>,
    context: &mut ExecutionContext,
) -> VmResult<Value> {
    let (stack_guard, request, consumed_inputs) =
        accel_fusion::gather_fusion_inputs(plan, graph, stack, vars, context)?;
    log::debug!(
        "dispatch fusion kind {:?}, supported {}",
        plan.group.kind,
        plan.kernel.supported
    );
    if plan.group.kind.is_elementwise() {
        match execute_elementwise(request) {
            Ok(result) => {
                accel_fusion::write_elementwise_materialized_stores(result.materialized_stores, vars, context);
                stack_guard.commit();
                Ok(result.final_value)
            }
            Err(err) => Err(mex("FusionExecutionFailed", &err.to_string())),
        }
    } else if plan.group.kind.is_reduction() {
        // Determine reduction axis or 'all'. Prefer the builtin reduction op's dim argument (inputs[1]).
        // MATLAB dim is 1-based: dim=1 reduces rows (axis 0), dim=2 reduces cols (axis 1), 'all' reduces all elements.
        let mut axis = 0usize;
        let mut axis_explicit = false;
        let mut reduce_all = matches!(plan.reduction_axes, Some(ReductionAxes::All));
        if let Some(ReductionAxes::Explicit(dims)) = &plan.reduction_axes {
            if let Some(first) = dims.first().copied() {
                axis = first.saturating_sub(1);
                axis_explicit = true;
            }
        }
        // Debug: show input origins for reduction
        if log::log_enabled!(log::Level::Debug) {
            let meta: Vec<String> = plan
                .inputs
                .iter()
                .map(|vid| {
                    if let Some(info) = graph.value(*vid) {
                        format!(
                            "vid={} origin={:?} shape={:?}",
                            vid, info.origin, info.shape
                        )
                    } else {
                        format!("vid={} origin=<missing>", vid)
                    }
                })
                .collect();
            log::debug!("reduction gather meta: [{}]", meta.join(", "));
        }
        // Detect 'all' in constants or const_values
        let has_all = reduce_all
            || plan.constants.values().any(value_is_all_keyword)
            || plan.const_values.values().any(value_is_all_keyword);
        if has_all {
            reduce_all = true;
        }
        if reduce_all && interp_engine::fusion_debug_enabled() {
            log::debug!(
                "fusion reduction (all) meta: data_vid={:?} inputs={:?} stack_pattern={:?}",
                plan.reduction_data,
                plan.inputs,
                plan.stack_pattern
            );
        }
        if !reduce_all {
            for node_id in &plan.group.nodes {
                if let Some(node) = graph.node(*node_id) {
                    if let runmat_accelerate::graph::AccelNodeLabel::Builtin { name } = &node.label
                    {
                        if name.eq_ignore_ascii_case("mean") {
                            for input_vid in &node.inputs {
                                if let Some(info) = graph.value(*input_vid) {
                                    if let Some(constant) = &info.constant {
                                        if value_is_all_keyword(constant) {
                                            reduce_all = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if reduce_all {
                    break;
                }
            }
        }
        // Prefer plan.reduction_dim if available
        if !reduce_all {
            if let Some(dim_vid) = plan.reduction_dim {
                if let Some(cv) = plan.const_values.get(&dim_vid) {
                    axis = match cv {
                        Value::Num(n) if *n >= 1.0 => (*n as usize).saturating_sub(1),
                        Value::Int(i) => (i.to_f64() as usize).saturating_sub(1),
                        _ => axis,
                    };
                    axis_explicit = true;
                } else if let Some(input_idx) = plan.inputs.iter().position(|v| *v == dim_vid) {
                    if let Some(cv) = plan.constants.get(&input_idx) {
                        axis = match cv {
                            Value::Num(n) if *n >= 1.0 => (*n as usize).saturating_sub(1),
                            Value::Int(i) => (i.to_f64() as usize).saturating_sub(1),
                            _ => axis,
                        };
                        axis_explicit = true;
                    }
                }
            } else {
                // Legacy fallback: inspect any constant mapped to the second logical position
                if let Some(dim_const) = plan.constants.get(&1) {
                    axis = match dim_const {
                        Value::Num(n) if *n >= 1.0 => (*n as usize).saturating_sub(1),
                        Value::Int(i) => (i.to_f64() as usize).saturating_sub(1),
                        _ => axis,
                    };
                    axis_explicit = true;
                }
            }
        }
        let (reduce_len, num_slices) = {
            // Try to get the data tensor's shape via the reduction builtin op input id
            let mut rows_cols: Option<(usize, usize)> = None;
            // Prefer shape from fusion plan reduction_data if fully known in graph
            if let Some(shape) = plan.reduction_data_shape(graph) {
                if shape.len() >= 2 {
                    rows_cols = Some((shape[0].max(1), shape[1].max(1)));
                } else if shape.len() == 1 {
                    rows_cols = Some((shape[0].max(1), 1));
                }
            }
            // Early fallback: inspect runtime variable values for declared plan inputs
            if rows_cols.is_none() {
                for &vid in &plan.inputs {
                    if let Some(binding) = graph.var_binding(vid) {
                        let value_opt = match binding.kind {
                            VarKind::Global => vars.get(binding.index).cloned(),
                            VarKind::Local => {
                                if let Some(frame) = context.call_stack.last() {
                                    let absolute = frame.locals_start + binding.index;
                                    context.locals.get(absolute).cloned()
                                } else {
                                    vars.get(binding.index).cloned()
                                }
                            }
                        };
                        if let Some(value) = value_opt {
                            match value {
                                Value::GpuTensor(h) => {
                                    rows_cols = Some((
                                        h.shape.first().copied().unwrap_or(1).max(1),
                                        h.shape.get(1).copied().unwrap_or(1).max(1),
                                    ));
                                    break;
                                }
                                Value::Tensor(t) => {
                                    rows_cols = Some((
                                        t.shape.first().copied().unwrap_or(1).max(1),
                                        t.shape.get(1).copied().unwrap_or(1).max(1),
                                    ));
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            // Prefer immediately-consumed stack values (common for reductions over producer results)
            for v in consumed_inputs.iter().filter_map(|v| v.as_ref()) {
                match v {
                    Value::GpuTensor(h) => {
                        rows_cols = Some((
                            h.shape.first().copied().unwrap_or(1).max(1),
                            h.shape.get(1).copied().unwrap_or(1).max(1),
                        ));
                        break;
                    }
                    Value::Tensor(t) => {
                        rows_cols = Some((
                            t.shape.first().copied().unwrap_or(1).max(1),
                            t.shape.get(1).copied().unwrap_or(1).max(1),
                        ));
                        break;
                    }
                    _ => {}
                }
            }
            let data_value_id: Option<runmat_accelerate::graph::ValueId> = plan.reduction_data;

            if let Some(data_id) = data_value_id {
                // Map data_id to plan input index if external
                if let Some(input_index) = plan.inputs.iter().position(|vid| *vid == data_id) {
                    if let Some(val) = consumed_inputs.get(input_index).and_then(|v| v.as_ref()) {
                        match val {
                            Value::GpuTensor(h) => {
                                let r = h.shape.first().copied().unwrap_or(1).max(1);
                                let c = h.shape.get(1).copied().unwrap_or(1).max(1);
                                rows_cols = Some((r, c));
                            }
                            Value::Tensor(t) => {
                                let r = t.shape.first().copied().unwrap_or(1).max(1);
                                let c = t.shape.get(1).copied().unwrap_or(1).max(1);
                                rows_cols = Some((r, c));
                            }
                            _ => {}
                        }
                    }
                    // Otherwise, it was a variable/constant; use request.inputs
                    if rows_cols.is_none() {
                        if let Some(val) = request.inputs.get(input_index) {
                            match val {
                                Value::GpuTensor(h) => {
                                    let r = h.shape.first().copied().unwrap_or(1).max(1);
                                    let c = h.shape.get(1).copied().unwrap_or(1).max(1);
                                    rows_cols = Some((r, c));
                                }
                                Value::Tensor(t) => {
                                    let r = t.shape.first().copied().unwrap_or(1).max(1);
                                    let c = t.shape.get(1).copied().unwrap_or(1).max(1);
                                    rows_cols = Some((r, c));
                                }
                                _ => {}
                            }
                        }
                    }
                }
                if rows_cols.is_none() {
                    if let Some(info) = graph.value(data_id) {
                        // Try direct variable lookup to get runtime value shape
                        if let ValueOrigin::Variable { kind, index } = &info.origin {
                            let val = match kind {
                                VarKind::Global => vars.get(*index).cloned(),
                                VarKind::Local => {
                                    if let Some(frame) = context.call_stack.last() {
                                        let absolute = frame.locals_start + index;
                                        context.locals.get(absolute).cloned()
                                    } else {
                                        vars.get(*index).cloned()
                                    }
                                }
                            };
                            if let Some(v) = val {
                                match v {
                                    Value::GpuTensor(h) => {
                                        rows_cols = Some((
                                            h.shape.first().copied().unwrap_or(1).max(1),
                                            h.shape.get(1).copied().unwrap_or(1).max(1),
                                        ));
                                    }
                                    Value::Tensor(t) => {
                                        rows_cols = Some((
                                            t.shape.first().copied().unwrap_or(1).max(1),
                                            t.shape.get(1).copied().unwrap_or(1).max(1),
                                        ));
                                    }
                                    _ => {}
                                }
                            }
                        }
                        if rows_cols.is_none() {
                            if let ShapeInfo::Tensor(dims) = &info.shape {
                                if !dims.is_empty() {
                                    let r = dims.first().and_then(|d| *d).unwrap_or(1);
                                    let c = dims.get(1).and_then(|d| *d).unwrap_or(1);
                                    rows_cols = Some((r.max(1), c.max(1)));
                                }
                            }
                        }
                    }
                }
            }

            // Fallback: any tensor input
            if rows_cols.is_none() {
                for v in consumed_inputs.iter().filter_map(|v| v.as_ref()) {
                    match v {
                        Value::GpuTensor(h) => {
                            rows_cols = Some((
                                h.shape.first().copied().unwrap_or(1).max(1),
                                h.shape.get(1).copied().unwrap_or(1).max(1),
                            ));
                            break;
                        }
                        Value::Tensor(t) => {
                            rows_cols = Some((
                                t.shape.first().copied().unwrap_or(1).max(1),
                                t.shape.get(1).copied().unwrap_or(1).max(1),
                            ));
                            break;
                        }
                        _ => {}
                    }
                }
                if rows_cols.is_none() {
                    for v in &request.inputs {
                        match v {
                            Value::GpuTensor(h) => {
                                rows_cols = Some((
                                    h.shape.first().copied().unwrap_or(1).max(1),
                                    h.shape.get(1).copied().unwrap_or(1).max(1),
                                ));
                                break;
                            }
                            Value::Tensor(t) => {
                                rows_cols = Some((
                                    t.shape.first().copied().unwrap_or(1).max(1),
                                    t.shape.get(1).copied().unwrap_or(1).max(1),
                                ));
                                break;
                            }
                            _ => {}
                        }
                    }
                }
            }
            // Final fallback: group-level static shape if available
            if rows_cols.is_none() {
                if let ShapeInfo::Tensor(dims) = &plan.group.shape {
                    if !dims.is_empty() {
                        let r = dims.first().and_then(|d| *d).unwrap_or(1);
                        let c = dims.get(1).and_then(|d| *d).unwrap_or(1);
                        rows_cols = Some((r.max(1), c.max(1)));
                    }
                }
            }

            let (r, c) = rows_cols.unwrap_or((1, 1));
            if reduce_all {
                let mut total_elems: Option<usize> = None;
                let mut total_from_operand = false;
                // Prefer fully-known graph shape for the reduction operand
                if let Some(shape) = plan.reduction_data_shape(graph) {
                    let prod = shape.into_iter().fold(1usize, |acc, dim| {
                        let d = dim.max(1);
                        acc.saturating_mul(d)
                    });
                    total_from_operand = true;
                    total_elems = Some(prod.max(1));
                }
                // Fall back to runtime tensor shapes (consumed stack values first, then inputs)
                if total_elems.is_none() {
                    let inspect_value = |value: &Value| -> Option<usize> {
                        match value {
                            Value::GpuTensor(handle) => {
                                if handle.shape.is_empty() {
                                    Some(1)
                                } else {
                                    Some(
                                        handle
                                            .shape
                                            .iter()
                                            .copied()
                                            .map(|d| d.max(1))
                                            .fold(1usize, |acc, dim| acc.saturating_mul(dim)),
                                    )
                                }
                            }
                            Value::Tensor(tensor) => {
                                if tensor.shape.is_empty() {
                                    Some(1)
                                } else {
                                    Some(
                                        tensor
                                            .shape
                                            .iter()
                                            .copied()
                                            .map(|d| d.max(1))
                                            .fold(1usize, |acc, dim| acc.saturating_mul(dim)),
                                    )
                                }
                            }
                            _ => None,
                        }
                    };
                    for value in consumed_inputs.iter().filter_map(|v| v.as_ref()) {
                        if let Some(prod) = inspect_value(value) {
                            total_from_operand = true;
                            total_elems = Some(prod.max(1));
                            break;
                        }
                    }
                    if total_elems.is_none() {
                        for value in &request.inputs {
                            if let Some(prod) = inspect_value(value) {
                                total_from_operand = true;
                                total_elems = Some(prod.max(1));
                                break;
                            }
                        }
                    }
                }
                // Final fallback: use group-level element count or the 2-D heuristic
                if total_elems.is_none() {
                    if let Some(ec) = plan.element_count() {
                        total_elems = Some(ec.max(1));
                    }
                }
                if total_elems.is_none() || !total_from_operand {
                    if interp_engine::fusion_debug_enabled() {
                        log::debug!(
                            "fusion reduction (all): operand extent unknown (source: {:?}); falling back to provider path",
                            if total_from_operand { "runtime" } else { "output_shape" }
                        );
                    }
                    return Err(mex(
                        "FusionReductionExtentUnknown",
                        "fusion: reduction all extent unknown",
                    ));
                }
                let total = total_elems.unwrap();
                if interp_engine::fusion_debug_enabled() {
                    log::debug!(
                        "fusion reduction (all): total_elems={} fallback_rows={} fallback_cols={}",
                        total,
                        r,
                        c
                    );
                }
                (total, 1usize)
            } else {
                if !axis_explicit {
                    axis = if r == 1 && c > 1 {
                        1
                    } else if r > 1 {
                        0
                    } else {
                        axis
                    };
                }
                if interp_engine::fusion_debug_enabled() {
                    if r == 1 && c == 1 {
                        log::debug!(
                    "fusion reduction: unresolved shape (defaulted to 1x1); axis={}, constants={:?}",
                    axis, plan.constants
                );
                    } else {
                        log::debug!(
                    "fusion reduction: resolved shape rows={} cols={} axis={} constants={:?}",
                    r,
                    c,
                    axis,
                    plan.constants
                );
                    }
                }
                if axis == 0 {
                    (r, c)
                } else {
                    (c, r)
                }
            }
        };
        if interp_engine::fusion_debug_enabled() {
            log::debug!(
                "fusion reduction: axis={} reduce_len={} num_slices={} constants={:?}",
                axis,
                reduce_len,
                num_slices,
                plan.constants
            );
        }
        if log::log_enabled!(log::Level::Debug) && interp_engine::fusion_debug_enabled() {
            let _rt_inputs: Vec<String> = request
                .inputs
                .iter()
                .enumerate()
                .map(|(i, v)| accel_fusion::summarize_value(i, v))
                .collect();
            let _plan_inputs: Vec<String> = plan
                .inputs
                .iter()
                .map(|vid| {
                    if let Some(info) = graph.value(*vid) {
                        format!(
                            "vid={} origin={:?} shape={:?}",
                            vid, info.origin, info.shape
                        )
                    } else {
                        format!("vid={} origin=<missing>", vid)
                    }
                })
                .collect();
            // Summarize inputs once before execution (omit plan inputs to reduce noise)
            log::debug!("reduction inputs: [{}]", _rt_inputs.join(", "));
        }
        // If shape derivation failed (1x1) but inputs/consumed suggest a larger tensor, skip fusion
        let looks_wrong = reduce_len == 1 && num_slices == 1 && {
            let mut big = false;
            let mut check_val = |v: &Value| match v {
                Value::GpuTensor(h) => {
                    let prod = h.shape.iter().copied().product::<usize>();
                    if prod > 1 {
                        big = true;
                    }
                }
                Value::Tensor(t) => {
                    let prod = t.shape.iter().copied().product::<usize>();
                    if prod > 1 {
                        big = true;
                    }
                }
                _ => {}
            };
            for v in consumed_inputs.iter().filter_map(|v| v.as_ref()) {
                check_val(v);
            }
            for v in &request.inputs {
                check_val(v);
            }
            big
        };
        if looks_wrong {
            log::debug!(
                "fusion reduction: skipping fusion due to unresolved shape; falling back to provider path"
            );
            return Err(mex(
                "FusionReductionShapeUnresolved",
                "fusion: reduction shape unresolved",
            ));
        }

        // Optional escape hatch: disable fused reductions to force provider path
        if std::env::var("RUNMAT_DISABLE_FUSED_REDUCTION")
            .ok()
            .as_deref()
            == Some("1")
        {
            return Err(mex(
                "FusionReductionDisabled",
                "fusion: fused reductions disabled",
            ));
        }
        let workgroup_size = 256u32;
        if log::log_enabled!(log::Level::Debug) && interp_engine::fusion_debug_enabled() {
            let _rt_inputs: Vec<String> = request
                .inputs
                .iter()
                .enumerate()
                .map(|(i, v)| accel_fusion::summarize_value(i, v))
                .collect();
            let _plan_inputs: Vec<String> = plan
                .inputs
                .iter()
                .map(|vid| {
                    if let Some(info) = graph.value(*vid) {
                        format!(
                            "vid={} origin={:?} shape={:?}",
                            vid, info.origin, info.shape
                        )
                    } else {
                        format!("vid={} origin=<missing>", vid)
                    }
                })
                .collect();
            log::debug!(
                "reduction axis={} reduce_len={} num_slices={}",
                axis,
                reduce_len,
                num_slices
            );
        }
        match execute_reduction(request, reduce_len, num_slices, workgroup_size) {
            Ok(result) => {
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => Err(mex("FusionExecutionFailed", &err.to_string())),
        }
    } else if plan.group.kind == FusionKind::CenteredGram {
        match execute_centered_gram(request).await {
            Ok(result) => {
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => Err(mex("FusionExecutionFailed", &err.to_string())),
        }
    } else if plan.group.kind == FusionKind::PowerStepNormalize {
        match execute_power_step_normalize(request).await {
            Ok(result) => {
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => Err(mex("FusionExecutionFailed", &err.to_string())),
        }
    } else if plan.group.kind == FusionKind::ExplainedVariance {
        log::debug!("explained variance plan inputs {:?}", plan.inputs);
        match execute_explained_variance(request).await {
            Ok(result) => {
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => {
                log::debug!("explained variance fusion fallback: {}", err);
                Err(mex("FusionExecutionFailed", &err.to_string()))
            }
        }
    } else if plan.group.kind == FusionKind::MatmulEpilogue {
        match execute_matmul_epilogue(request).await {
            Ok(result) => {
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => Err(mex("FusionExecutionFailed", &err.to_string())),
        }
    } else if plan.group.kind == FusionKind::ImageNormalize {
        match execute_image_normalize(request).await {
            Ok(result) => {
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => Err(mex("FusionExecutionFailed", &err.to_string())),
        }
    } else {
        // Unknown fusion kind; restore stack and report
        Err(mex(
            "FusionUnsupportedKind",
            "fusion: unsupported fusion kind",
        ))
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

#[cfg(test)]
fn map_slice_plan_error(context: &str, err: RuntimeError) -> RuntimeError {
    let is_oob = err
        .identifier()
        .map(|id| id.contains("IndexOutOfBounds"))
        .unwrap_or_else(|| err.message().contains("IndexOutOfBounds"));
    if is_oob {
        err
    } else {
        let mut builder = build_runtime_error(format!("{context}: {}", err.message()));
        if let Some(identifier) = err.identifier() {
            builder = builder.with_identifier(identifier.to_string());
        }
        builder.build()
    }
}

/// Interpret bytecode with default variable initialization
pub async fn interpret(bytecode: &Bytecode) -> Result<Vec<Value>, RuntimeError> {
    let mut vars = vec![Value::Num(0.0); bytecode.var_count];
    match interpret_with_vars(bytecode, &mut vars, Some("<main>")).await {
        Ok(InterpreterOutcome::Completed(values)) => Ok(values),
        Err(e) => Err(e),
    }
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
    let runmat_vm::call::shared::PreparedUserCall {
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

pub async fn interpret_function(
    bytecode: &Bytecode,
    vars: Vec<Value>,
) -> Result<Vec<Value>, RuntimeError> {
    // Delegate to the counted variant with anonymous name and zero counts
    interpret_function_with_counts(bytecode, vars, "<anonymous>", 0, 0).await
}

async fn interpret_function_with_counts(
    bytecode: &Bytecode,
    mut vars: Vec<Value>,
    name: &str,
    out_count: usize,
    in_count: usize,
) -> Result<Vec<Value>, RuntimeError> {
    // Push (nargin, nargout), run, then pop
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
mod scalar_index_tests {
    use super::*;

    #[test]
    fn linear_false_bool_index_is_empty() {
        let indices =
            futures::executor::block_on(indices_from_value_linear(&Value::Bool(false), 4))
                .expect("false logical index should be empty");
        assert!(indices.is_empty());
    }

    #[test]
    fn linear_true_bool_index_selects_first() {
        let indices = futures::executor::block_on(indices_from_value_linear(&Value::Bool(true), 4))
            .expect("true logical index should select first element");
        assert_eq!(indices, vec![1]);
    }

    #[test]
    fn dim_false_bool_selector_is_empty() {
        let selector = futures::executor::block_on(selector_from_value_dim(&Value::Bool(false), 4))
            .expect("false logical selector should be empty");
        match selector {
            SliceSelector::Indices(indices) => assert!(indices.is_empty()),
            SliceSelector::Scalar(_)
            | SliceSelector::Colon
            | SliceSelector::LinearIndices { .. } => {
                panic!("expected empty indices selector")
            }
        }
    }
}

#[cfg(all(test, feature = "native-accel"))]
mod fusion_span_barrier_tests {
    use super::*;
    use runmat_accelerate::InstrSpan;

    #[test]
    fn store_reload_span_with_one_live_result_is_legal() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::LoadConst(0.0),
            Instr::Add,
            Instr::StoreVar(1),
            Instr::LoadVar(1),
        ];
        let span = InstrSpan { start: 0, end: 4 };

        assert_eq!(accel_fusion::fusion_span_live_result_count(&instructions, &span), Some(1));
        assert!(!accel_fusion::fusion_span_has_vm_barrier(&instructions, &span));
    }

    #[test]
    fn span_leaving_multiple_live_results_is_illegal() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::LoadConst(0.0),
            Instr::Add,
            Instr::LoadVar(1),
        ];
        let span = InstrSpan { start: 0, end: 3 };

        assert_eq!(accel_fusion::fusion_span_live_result_count(&instructions, &span), Some(2));
        assert!(accel_fusion::fusion_span_has_vm_barrier(&instructions, &span));
    }

    #[test]
    fn stored_value_observed_after_span_is_legal_when_materialized() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::LoadConst(0.0),
            Instr::Add,
            Instr::StoreVar(1),
            Instr::LoadVar(1),
            Instr::LoadVar(1),
        ];
        let span = InstrSpan { start: 0, end: 4 };

        assert!(!accel_fusion::fusion_span_has_vm_barrier(&instructions, &span));
    }

    #[test]
    fn overwritten_store_before_later_load_is_legal() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::LoadConst(0.0),
            Instr::Add,
            Instr::StoreVar(1),
            Instr::LoadVar(1),
            Instr::LoadConst(1.0),
            Instr::StoreVar(1),
            Instr::LoadVar(1),
        ];
        let span = InstrSpan { start: 0, end: 4 };

        assert!(!accel_fusion::fusion_span_has_vm_barrier(&instructions, &span));
    }
}
