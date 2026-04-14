use crate::functions::{Bytecode, ExecutionContext, UserFunction};
use crate::gc_roots::InterpretContext;
use crate::instr::{EndExpr, Instr};
#[cfg(feature = "native-accel")]
use runmat_accelerate::fusion_exec::{
    execute_centered_gram, execute_elementwise, execute_explained_variance,
    execute_image_normalize, execute_matmul_epilogue, execute_power_step_normalize,
    execute_reduction, FusionExecutionRequest,
};
#[cfg(feature = "native-accel")]
use runmat_accelerate::{
    activate_fusion_plan, deactivate_fusion_plan, fusion_residency, set_current_pc,
};
#[cfg(feature = "native-accel")]
use runmat_accelerate::{
    active_group_plan_clone, value_is_all_keyword, FusionKind, InstrSpan, ReductionAxes, ShapeInfo,
    ValueOrigin, VarKind,
};
use runmat_builtins::{Type, Value};
use runmat_runtime::{
    build_runtime_error,
    builtins::common::shape::is_scalar_shape,
    builtins::common::tensor,
    builtins::stats::random::stochastic_evolution::stochastic_evolution_host,
    gather_if_needed_async,
    output_context::push_output_count,
    user_functions,
    workspace::{self as runtime_workspace, WorkspaceResolver},
    RuntimeError,
};
use runmat_thread_local::runmat_thread_local;
pub use runmat_vm::interpreter::api::{
    push_pending_workspace, set_call_stack_limit, set_error_namespace,
    take_updated_workspace_state, InterpreterOutcome, InterpreterState, PendingWorkspaceGuard,
    DEFAULT_CALLSTACK_LIMIT, DEFAULT_ERROR_NAMESPACE,
};
use runmat_vm::interpreter::errors::{
    attach_span_at, attach_span_from_pc, ensure_runtime_error_identifier, mex, set_vm_pc,
};
use runmat_vm::interpreter::stack::pop_value;
use runmat_vm::call::shared as call_shared;
use runmat_vm::call::{builtins as call_builtins, feval as call_feval, user as call_user};
use runmat_vm::call::builtins::ImportedBuiltinResolution;
use runmat_vm::call::closures as call_closures;
use runmat_vm::call::feval::FevalDispatch;
use runmat_vm::indexing::end_expr as idx_end_expr;
use runmat_vm::indexing::read_linear as idx_read_linear;
use runmat_vm::indexing::read_slice as idx_read_slice;
use runmat_vm::indexing::selectors as idx_selectors;
use runmat_vm::indexing::write_linear as idx_write_linear;
use runmat_vm::indexing::write_slice as idx_write_slice;
use runmat_vm::object::{class_def as obj_class_def, resolve as obj_resolve};
use runmat_vm::ops::cells as cell_ops;
use runmat_vm::ops::{arithmetic as arithmetic_ops, arrays as array_ops, comparison as comparison_ops};
use runmat_vm::ops::control_flow::{self, ControlFlowAction};
use runmat_vm::ops::stack as stack_ops;
use runmat_vm::interpreter::timing::InterpreterTiming;
use runmat_vm::runtime::call_stack::{
    attach_call_frames, error_namespace, push_call_frame,
};
use runmat_vm::runtime::globals as runtime_globals;
use runmat_vm::runtime::workspace::{
    refresh_workspace_state, set_workspace_state, take_pending_workspace_state,
    workspace_assign, workspace_clear, workspace_lookup, workspace_remove, workspace_snapshot,
};
#[cfg(not(target_arch = "wasm32"))]
use runmat_time::Instant;
#[cfg(target_arch = "wasm32")]
type Instant = ();
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::sync::Arc;
use std::sync::Once;
#[cfg(feature = "native-accel")]
use std::sync::OnceLock;
use tracing::{debug, info_span, warn};

#[cfg(feature = "native-accel")]
struct FusionPlanGuard;

#[cfg(feature = "native-accel")]
impl Drop for FusionPlanGuard {
    fn drop(&mut self) {
        deactivate_fusion_plan();
    }
}

#[derive(Clone, Copy)]
enum AutoBinaryOp {
    Elementwise,
    MatMul,
}

#[derive(Clone, Copy)]
enum AutoUnaryOp {
    Transpose,
}

#[cfg(feature = "native-accel")]
async fn accel_promote_binary(op: AutoBinaryOp, a: &Value, b: &Value) -> VmResult<(Value, Value)> {
    use runmat_accelerate::{promote_binary, BinaryOp};
    let mapped = match op {
        AutoBinaryOp::Elementwise => BinaryOp::Elementwise,
        AutoBinaryOp::MatMul => BinaryOp::MatMul,
    };
    Ok(promote_binary(mapped, a, b)
        .await
        .map_err(|e| e.to_string())?)
}

#[cfg(not(feature = "native-accel"))]
async fn accel_promote_binary(_op: AutoBinaryOp, a: &Value, b: &Value) -> VmResult<(Value, Value)> {
    Ok((a.clone(), b.clone()))
}

#[cfg(feature = "native-accel")]
async fn accel_promote_unary(op: AutoUnaryOp, value: &Value) -> VmResult<Value> {
    use runmat_accelerate::{promote_unary, UnaryOp};
    let mapped = match op {
        AutoUnaryOp::Transpose => UnaryOp::Transpose,
    };
    Ok(promote_unary(mapped, value)
        .await
        .map_err(|e| e.to_string())?)
}

#[cfg(not(feature = "native-accel"))]
async fn accel_promote_unary(_op: AutoUnaryOp, value: &Value) -> VmResult<Value> {
    Ok(value.clone())
}

#[cfg(feature = "native-accel")]
async fn accel_prepare_args(name: &str, args: &[Value]) -> VmResult<Vec<Value>> {
    Ok(runmat_accelerate::prepare_builtin_args(name, args)
        .await
        .map_err(|e| e.to_string())?)
}

#[cfg(not(feature = "native-accel"))]
async fn accel_prepare_args(_name: &str, args: &[Value]) -> VmResult<Vec<Value>> {
    Ok(args.to_vec())
}

macro_rules! call_builtin_vm {
    ($name:expr, $args:expr $(,)?) => {
        runmat_runtime::call_builtin_async($name, $args).await
    };
}

async fn call_builtin_auto(name: &str, args: &[Value]) -> VmResult<Value> {
    let prepared = accel_prepare_args(name, args).await?;
    Ok(call_builtin_vm!(name, &prepared)?)
}

fn is_scalarish_for_division(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Num(_) | Value::Complex(_, _) | Value::Bool(_) => true,
        Value::LogicalArray(arr) => is_scalar_shape(&arr.shape),
        Value::Tensor(tensor) => is_scalar_shape(&tensor.shape),
        Value::ComplexTensor(tensor) => is_scalar_shape(&tensor.shape),
        Value::GpuTensor(handle) => is_scalar_shape(&handle.shape),
        _ => false,
    }
}

async fn execute_elementwise_division(lhs: &Value, rhs: &Value) -> VmResult<Value> {
    let (lhs_acc, rhs_acc) = accel_promote_binary(AutoBinaryOp::Elementwise, lhs, rhs).await?;
    Ok(call_builtin_vm!("rdivide", &[lhs_acc, rhs_acc])?)
}

async fn execute_elementwise_left_division(lhs: &Value, rhs: &Value) -> VmResult<Value> {
    let (rhs_acc, lhs_acc) = accel_promote_binary(AutoBinaryOp::Elementwise, rhs, lhs).await?;
    Ok(call_builtin_vm!("rdivide", &[rhs_acc, lhs_acc])?)
}

async fn execute_right_division(lhs: &Value, rhs: &Value) -> VmResult<Value> {
    match (lhs, rhs) {
        (Value::Object(obj), _) => {
            let args = vec![
                Value::Object(obj.clone()),
                Value::String("mrdivide".to_string()),
                rhs.clone(),
            ];
            match call_builtin_vm!("call_method", &args) {
                Ok(v) => Ok(v),
                Err(_) => {
                    if is_scalarish_for_division(rhs) {
                        execute_elementwise_division(lhs, rhs).await
                    } else {
                        call_builtin_auto("mrdivide", &[lhs.clone(), rhs.clone()]).await
                    }
                }
            }
        }
        (_, Value::Object(obj)) => {
            let args = vec![
                Value::Object(obj.clone()),
                Value::String("mrdivide".to_string()),
                lhs.clone(),
            ];
            match call_builtin_vm!("call_method", &args) {
                Ok(v) => Ok(v),
                Err(_) => {
                    if is_scalarish_for_division(rhs) {
                        execute_elementwise_division(lhs, rhs).await
                    } else {
                        call_builtin_auto("mrdivide", &[lhs.clone(), rhs.clone()]).await
                    }
                }
            }
        }
        _ => {
            if is_scalarish_for_division(rhs) {
                execute_elementwise_division(lhs, rhs).await
            } else {
                call_builtin_auto("mrdivide", &[lhs.clone(), rhs.clone()]).await
            }
        }
    }
}

async fn execute_left_division(lhs: &Value, rhs: &Value) -> VmResult<Value> {
    match (lhs, rhs) {
        (Value::Object(obj), _) => {
            let args = vec![
                Value::Object(obj.clone()),
                Value::String("mldivide".to_string()),
                rhs.clone(),
            ];
            match call_builtin_vm!("call_method", &args) {
                Ok(v) => Ok(v),
                Err(_) => {
                    if is_scalarish_for_division(lhs) {
                        execute_elementwise_left_division(lhs, rhs).await
                    } else {
                        call_builtin_auto("mldivide", &[lhs.clone(), rhs.clone()]).await
                    }
                }
            }
        }
        (_, Value::Object(obj)) => {
            let args = vec![
                Value::Object(obj.clone()),
                Value::String("mldivide".to_string()),
                lhs.clone(),
            ];
            match call_builtin_vm!("call_method", &args) {
                Ok(v) => Ok(v),
                Err(_) => {
                    if is_scalarish_for_division(lhs) {
                        execute_elementwise_left_division(lhs, rhs).await
                    } else {
                        call_builtin_auto("mldivide", &[lhs.clone(), rhs.clone()]).await
                    }
                }
            }
        }
        _ => {
            if is_scalarish_for_division(lhs) {
                execute_elementwise_left_division(lhs, rhs).await
            } else {
                call_builtin_auto("mldivide", &[lhs.clone(), rhs.clone()]).await
            }
        }
    }
}

fn output_hint_for_single_result(bytecode: &Bytecode, pc: usize) -> usize {
    match bytecode.instructions.get(pc + 1) {
        Some(Instr::Pop) | Some(Instr::EmitStackTop { .. }) => 0,
        _ => 1,
    }
}

#[cfg(feature = "native-accel")]
#[inline]
fn fusion_debug_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| match std::env::var("RUNMAT_DEBUG_FUSION") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"),
        Err(_) => false,
    })
}

#[cfg(feature = "native-accel")]
fn log_fusion_span_window(
    plan: &runmat_accelerate::FusionGroupPlan,
    bytecode: &Bytecode,
    pc: usize,
) {
    if !fusion_debug_enabled() || !log::log_enabled!(log::Level::Debug) {
        return;
    }
    if bytecode.instructions.is_empty() {
        return;
    }
    let window = 3usize;
    let span = plan.group.span.clone();
    let total = bytecode.instructions.len();
    let start = span.start.saturating_sub(window);
    let mut end = span.end + window;
    if end >= total {
        end = total.saturating_sub(1);
    }
    if end < span.end {
        end = span.end;
    }
    let mut ops: Vec<String> = Vec::new();
    for idx in start..=end {
        let instr = &bytecode.instructions[idx];
        let mut tags: Vec<&'static str> = Vec::new();
        if idx == pc {
            tags.push("pc");
        }
        if idx == span.start {
            tags.push("start");
        }
        if idx == span.end {
            tags.push("end");
        }
        let tag_str = if tags.is_empty() {
            String::new()
        } else {
            format!("<{}>", tags.join(","))
        };
        ops.push(format!("{}{} {:?}", idx, tag_str, instr));
    }
    log::debug!(
        "fusion plan {} span window [{}..{}]: {}",
        plan.index,
        start,
        end,
        ops.join(" | ")
    );
}

type VmResult<T> = Result<T, RuntimeError>;

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

#[derive(Debug, Clone)]
struct SlicePlan {
    indices: Vec<u32>,
    output_shape: Vec<usize>,
    selection_lengths: Vec<usize>,
    dims: usize,
}

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

fn from_vm_slice_plan(plan: idx_selectors::SlicePlan) -> SlicePlan {
    SlicePlan {
        indices: plan.indices,
        output_shape: plan.output_shape,
        selection_lengths: plan.selection_lengths,
        dims: plan.dims,
    }
}


fn total_len_from_shape(shape: &[usize]) -> usize {
    if is_scalar_shape(shape) {
        1
    } else {
        shape.iter().copied().product()
    }
}

#[derive(Clone, Copy)]
struct IndexContext<'a> {
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    base_shape: &'a [usize],
}

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

async fn index_scalar_from_value(value: &Value) -> VmResult<Option<i64>> {
    idx_selectors::index_scalar_from_value(value).await
}

async fn selector_from_value_dim(value: &Value, dim_len: usize) -> VmResult<SliceSelector> {
    idx_selectors::selector_from_value_dim(value, dim_len)
        .await
        .map(from_vm_slice_selector)
}

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

fn build_slice_plan(
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
) -> VmResult<SlicePlan> {
    let vm_selectors: Vec<_> = selectors.iter().map(to_vm_slice_selector).collect();
    idx_selectors::build_slice_plan(&vm_selectors, dims, base_shape).map(from_vm_slice_plan)
}

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
    let pending_state = take_pending_workspace_state();
    let _workspace_guard = pending_state.map(|(names, assigned)| {
        let filtered_assigned: HashSet<String> = assigned
            .into_iter()
            .filter(|name| names.contains_key(name))
            .collect();
        set_workspace_state(names, filtered_assigned, &mut vars)
    });
    refresh_workspace_state(&vars);
    let mut _gc_context = InterpretContext::new(&stack, &vars)?;
    // Register thread-local globals/persistents as GC roots for the duration of this execution
    let thread_roots: Vec<Value> = runtime_globals::collect_thread_roots();
    let _ = _gc_context.register_global_values(thread_roots, "thread_globals_persistents");
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
    #[inline]
    fn bench_start() -> Option<Instant> {
        None
    }
    #[inline]
    fn bench_end(_label: &str, _start: Option<Instant>) {}
    let debug_stack = std::env::var("RUNMAT_DEBUG_STACK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let mut interpreter_timing = InterpreterTiming::new();
    macro_rules! vm_bail {
        ($err:expr) => {{
            let err: RuntimeError = ensure_runtime_error_identifier(($err).into());
            let err = attach_span_at(&bytecode, pc, err);
            if let Some((catch_pc, catch_var)) = try_stack.pop() {
                if let Some(var_idx) = catch_var {
                    if var_idx >= vars.len() {
                        vars.resize(var_idx + 1, Value::Num(0.0));
                        refresh_workspace_state(&vars);
                    }
                    let mex = parse_exception(&err);
                    last_exception = Some(mex.clone());
                    vars[var_idx] = Value::MException(mex);
                }
                pc = catch_pc;
                continue;
            } else {
                return Err(err);
            }
        }};
    }
    while pc < bytecode.instructions.len() {
        set_vm_pc(pc);
        #[cfg(feature = "native-accel")]
        set_current_pc(pc);
        if runmat_runtime::interrupt::is_cancelled() {
            return Err(mex("ExecutionCancelled", "Execution cancelled by user"));
        }
        #[cfg(feature = "native-accel")]
        if let (Some(plan), Some(graph)) =
            (active_group_plan_clone(), bytecode.accel_graph.as_ref())
        {
            if plan.group.span.start == pc {
                #[cfg(feature = "native-accel")]
                {
                    let detail = format!(
                        "plan={} kind={:?} span=[{}..{}]",
                        plan.index, plan.group.kind, plan.group.span.start, plan.group.span.end
                    );
                    interpreter_timing.flush_host_span("before_fusion", Some(detail.as_str()));
                }
                #[cfg(feature = "native-accel")]
                log_fusion_span_window(&plan, &bytecode, pc);
                let span = plan.group.span.clone();
                let has_barrier = fusion_span_has_vm_barrier(&bytecode.instructions, &span);
                let live_result_count =
                    fusion_span_live_result_count(&bytecode.instructions, &span);
                if fusion_debug_enabled() {
                    log::trace!(
                        "fusion gate pc={} kind={:?} span={}..{} has_barrier={} live_results={:?}",
                        pc,
                        plan.group.kind,
                        span.start,
                        span.end,
                        has_barrier,
                        live_result_count
                    );
                }
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
                } else if fusion_debug_enabled() {
                    log::debug!(
                        "fusion skip at pc {}: side-effecting instrs in span {}..{}",
                        pc,
                        span.start,
                        span.end
                    );
                }
            }
        }
        interpreter_timing.note_host_instr(pc);
        if debug_stack {
            debug!(
                pc,
                instr = ?bytecode.instructions[pc],
                stack_len = stack.len(),
                "[vm] instr"
            );
        }
        match bytecode.instructions[pc].clone() {
            Instr::EmitStackTop { label } => {
                stack_ops::emit_stack_top(&stack, &label, &bytecode.var_names).await?;
            }
            Instr::EmitVar { var_index, label } => {
                stack_ops::emit_var(&vars, var_index, &label, &bytecode.var_names).await?;
            }
            Instr::AndAnd(target) => {
                match control_flow::and_and(&mut stack, target)? {
                    ControlFlowAction::Jump(target) => {
                        pc = target;
                        continue;
                    }
                    ControlFlowAction::Next | ControlFlowAction::Return => {}
                }
            }
            Instr::OrOr(target) => {
                match control_flow::or_or(&mut stack, target)? {
                    ControlFlowAction::Jump(target) => {
                        pc = target;
                        continue;
                    }
                    ControlFlowAction::Next | ControlFlowAction::Return => {}
                }
            }
            Instr::Swap => {
                stack_ops::swap(&mut stack)?;
            }
            Instr::CallFeval(argc) => {
                let args = call_builtins::collect_call_args(&mut stack, argc)?;
                let func_val = pop_value(&mut stack)?;
                match call_feval::execute_feval(
                    func_val,
                    args,
                    &context.functions,
                    &bytecode.functions,
                )
                .await
                {
                    Ok(FevalDispatch::Completed(result)) => stack.push(result),
                    Ok(FevalDispatch::InvokeUser {
                        name,
                        args,
                        functions,
                    }) => match invoke_user_function_value(&name, &args, &functions, &mut vars).await {
                        Ok(value) => stack.push(value),
                        Err(e) => vm_bail!(e),
                    },
                    Err(err) => vm_bail!(err),
                }
            }
            Instr::CallFevalExpandMulti(specs) => {
                let args = call_shared::build_expanded_args_from_specs(
                    &mut stack,
                    &specs,
                    "CallFevalExpandMulti requires cell or object for expand_all",
                    "CallFevalExpandMulti requires cell or object cell access",
                    |base| async move { match base {
                        Value::Object(obj) => {
                            let empty = call_shared::subsref_empty_brace_cell()?;
                            let args = vec![
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("{}".to_string()),
                                empty,
                            ];
                            let v = runmat_runtime::call_builtin_async("call_method", &args).await?;
                            Ok(match v {
                                Value::Cell(ca) => call_shared::expand_all_cell(&ca),
                                other => vec![other],
                            })
                        }
                        _ => Err(mex(
                            "InvalidExpandAllTarget",
                            "CallFevalExpandMulti requires cell or object for expand_all",
                        )),
                    }},
                    |base, indices| async move { match base {
                        Value::Object(obj) => {
                            let cell = call_shared::subsref_brace_index_cell_raw(&indices)?;
                            let args = vec![
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("{}".to_string()),
                                cell,
                            ];
                            let v = runmat_runtime::call_builtin_async("call_method", &args).await?;
                            Ok(vec![v])
                        }
                        _ => Err(mex(
                            "ExpandError",
                            "CallFevalExpandMulti requires cell or object cell access",
                        )),
                    }},
                ).await?;
                let func_val = pop_value(&mut stack)?;
                match call_feval::execute_feval(
                    func_val,
                    args,
                    &context.functions,
                    &bytecode.functions,
                )
                .await
                {
                    Ok(FevalDispatch::Completed(result)) => stack.push(result),
                    Ok(FevalDispatch::InvokeUser {
                        name,
                        args,
                        functions,
                    }) => match invoke_user_function_value(&name, &args, &functions, &mut vars).await {
                        Ok(value) => stack.push(value),
                        Err(e) => vm_bail!(e),
                    },
                    Err(err) => vm_bail!(err),
                }
            }
            Instr::LoadConst(c) => {
                stack_ops::load_const(&mut stack, c);
                if debug_stack {
                    debug!(const_value = c, stack_len = stack.len(), "[vm] load const");
                }
            }
            Instr::LoadComplex(re, im) => {
                stack_ops::load_complex(&mut stack, re, im);
                if debug_stack {
                    eprintln!(
                        "  -> LoadComplex pushed ({}, {}), new_len={}",
                        re,
                        im,
                        stack.len()
                    );
                }
            }
            Instr::LoadBool(b) => stack_ops::load_bool(&mut stack, b),
            Instr::LoadString(s) => stack_ops::load_string(&mut stack, s),
            Instr::LoadCharRow(s) => {
                stack_ops::load_char_row(&mut stack, s)?;
            }
            Instr::LoadVar(i) => {
                let v = vars[i].clone();
                if std::env::var("RUNMAT_DEBUG_VARS").as_deref() == Ok("1") {
                    match &v {
                        Value::Tensor(t) => {
                            eprintln!("[vm] LoadVar var={i} Tensor shape={:?}", t.shape);
                        }
                        Value::GpuTensor(h) => {
                            eprintln!("[vm] LoadVar var={i} GpuTensor shape={:?}", h.shape);
                        }
                        _ => {}
                    }
                }
                if std::env::var("RUNMAT_DEBUG_INDEX").as_deref() == Ok("1") {
                    match &v {
                        Value::GpuTensor(h) => {
                            debug!(pc, var = i, shape = ?h.shape, "[vm] LoadVar GPU tensor");
                        }
                        Value::Tensor(t) => {
                            debug!(pc, var = i, shape = ?t.shape, "[vm] LoadVar tensor");
                        }
                        _ => {}
                    }
                }
                stack_ops::load_var(&mut stack, &vars, i)
            }
            Instr::StoreVar(i) => {
                let preview = stack
                    .last()
                    .cloned()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                if std::env::var("RUNMAT_DEBUG_VARS").as_deref() == Ok("1") {
                    match &preview {
                        Value::Tensor(t) => {
                            eprintln!("[vm] StoreVar var={i} Tensor shape={:?}", t.shape);
                        }
                        Value::GpuTensor(h) => {
                            eprintln!("[vm] StoreVar var={i} GpuTensor shape={:?}", h.shape);
                        }
                        _ => {}
                    }
                }
                if let Ok(filter) = std::env::var("RUNMAT_DEBUG_STORE_VAR") {
                    let log_this = if filter.trim().eq_ignore_ascii_case("*") {
                        true
                    } else if let Ok(target) = filter.trim().parse::<usize>() {
                        target == i
                    } else {
                        false
                    };
                    if log_this {
                        debug!(pc, var = i, ?preview, "[vm] StoreVar value");
                    }
                }
                if std::env::var("RUNMAT_DEBUG_INDEX").as_deref() == Ok("1") {
                    match &preview {
                        Value::GpuTensor(h) => {
                            debug!(pc, var = i, shape = ?h.shape, "[vm] StoreVar GPU tensor");
                        }
                        Value::Tensor(t) => {
                            debug!(pc, var = i, shape = ?t.shape, "[vm] StoreVar tensor");
                        }
                        _ => {}
                    }
                }
                stack_ops::store_var(
                    &mut stack,
                    &mut vars,
                    i,
                    &bytecode.var_names,
                    |current, incoming| {
                        #[cfg(feature = "native-accel")]
                        if !same_gpu_handle(current, incoming) {
                            clear_residency(current);
                        }
                    },
                    |stored_index, stored_value| {
                        runtime_globals::update_global_store(
                            stored_index,
                            stored_value,
                            &global_aliases,
                        );
                    },
                )?;
            }
            Instr::LoadLocal(offset) => {
                if let Err(err) = stack_ops::load_local(&mut stack, &context, &vars, offset) {
                    vm_bail!(err);
                }
            }
            Instr::StoreLocal(offset) => {
                stack_ops::store_local(
                    &mut stack,
                    &mut context,
                    &mut vars,
                    offset,
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
                )?;
            }
            Instr::EnterScope(local_count) => {
                control_flow::enter_scope(&mut context.locals, local_count);
            }
            Instr::ExitScope(local_count) => {
                control_flow::exit_scope(&mut context.locals, local_count, |val| {
                    #[cfg(feature = "native-accel")]
                    clear_residency(val);
                });
            }
            Instr::RegisterImport { path, wildcard } => {
                imports.push((path, wildcard));
            }
            Instr::DeclareGlobal(indices) => {
                runtime_globals::declare_global(indices, &mut vars);
            }
            Instr::DeclareGlobalNamed(indices, names) => {
                runtime_globals::declare_global_named(
                    indices,
                    names,
                    &mut vars,
                    &mut global_aliases,
                );
            }
            Instr::DeclarePersistent(indices) => {
                runtime_globals::declare_persistent(&current_function_name, indices, &mut vars);
            }
            Instr::DeclarePersistentNamed(indices, names) => {
                runtime_globals::declare_persistent_named(
                    &current_function_name,
                    indices,
                    names,
                    &mut vars,
                    &mut persistent_aliases,
                );
            }
            Instr::Add => {
                arithmetic_ops::add(
                    &mut stack,
                    |obj, method, arg| async move {
                        let args = vec![obj, Value::String(method.to_string()), arg];
                        call_builtin_vm!("call_method", &args)
                    },
                    |a, b| async move {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                        call_builtin_vm!("plus", &[a_acc, b_acc])
                    },
                ).await?;
            }
            Instr::Sub => {
                arithmetic_ops::sub(
                    &mut stack,
                    |obj, method, arg| async move {
                        let args = vec![obj, Value::String(method.to_string()), arg];
                        call_builtin_vm!("call_method", &args)
                    },
                    |obj, lhs| async move {
                        let class_name = match &obj { Value::Object(o) => o.class_name.clone(), _ => String::new() };
                        let qualified = format!("{}.minus", class_name);
                        call_builtin_vm!(&qualified, &[lhs, obj])
                    },
                    |a, b| async move {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                        call_builtin_vm!("minus", &[a_acc, b_acc])
                    },
                ).await?;
            }
            Instr::Mul => {
                arithmetic_ops::mul(
                    &mut stack,
                    |obj, method, arg| async move {
                        let args = vec![obj, Value::String(method.to_string()), arg];
                        call_builtin_vm!("call_method", &args)
                    },
                    |a, b| async move {
                        let (a_acc, b_acc) = accel_promote_binary(AutoBinaryOp::MatMul, &a, &b).await?;
                        runmat_runtime::matrix::value_matmul(&a_acc, &b_acc).await
                    },
                ).await?;
            }
            Instr::RightDiv => {
                arithmetic_ops::binary_fallback(&mut stack, |a, b| async move {
                    execute_right_division(&a, &b).await
                }).await?;
            }
            Instr::LeftDiv => {
                arithmetic_ops::binary_fallback(&mut stack, |a, b| async move {
                    execute_left_division(&a, &b).await
                }).await?;
            }
            Instr::Pow => {
                arithmetic_ops::power(
                    &mut stack,
                    |obj, method, arg| async move {
                        let args = vec![obj, Value::String(method.to_string()), arg];
                        call_builtin_vm!("call_method", &args)
                    },
                    |a, b| async move {
                        let (a_acc, b_acc) = accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                        runmat_runtime::power(&a_acc, &b_acc).map_err(RuntimeError::from)
                    },
                ).await?;
            }
            Instr::Neg => {
                arithmetic_ops::unary(&mut stack, |value| async move {
                    match &value {
                        Value::Object(obj) => {
                            let args = vec![Value::Object(obj.clone())];
                            match call_builtin_vm!("uminus", &args) {
                                Ok(v) => Ok(v),
                                Err(_) => call_builtin_vm!("times", &[value.clone(), Value::Num(-1.0)]),
                            }
                        }
                        _ => call_builtin_vm!("times", &[value.clone(), Value::Num(-1.0)]),
                    }
                }).await?;
            }
            Instr::UPlus => {
                arithmetic_ops::unary(&mut stack, |value| async move {
                    match &value {
                        Value::Object(obj) => {
                            let args = vec![Value::Object(obj.clone())];
                            match call_builtin_vm!("uplus", &args) {
                                Ok(v) => Ok(v),
                                Err(_) => Ok(value),
                            }
                        }
                        _ => Ok(value),
                    }
                }).await?;
            }
            Instr::Transpose => {
                arithmetic_ops::unary(&mut stack, |value| async move {
                    let promoted = accel_promote_unary(AutoUnaryOp::Transpose, &value).await?;
                    call_builtin_vm!("transpose", &[promoted])
                }).await?;
            }
            Instr::ConjugateTranspose => {
                arithmetic_ops::unary(&mut stack, |value| async move {
                    let promoted = accel_promote_unary(AutoUnaryOp::Transpose, &value).await?;
                    call_builtin_vm!("ctranspose", &[promoted])
                }).await?;
            }
            Instr::ElemMul => {
                arithmetic_ops::binary_method(
                    &mut stack,
                    "times",
                    |obj, method, arg| async move {
                        let args = vec![obj, Value::String(method.to_string()), arg];
                        call_builtin_vm!("call_method", &args)
                    },
                    |a, b| async move {
                        let (a_acc, b_acc) = accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                        call_builtin_vm!("times", &[a_acc, b_acc])
                    },
                ).await?;
            }
            Instr::ElemDiv => {
                arithmetic_ops::binary_method(
                    &mut stack,
                    "rdivide",
                    |obj, method, arg| async move {
                        let args = vec![obj, Value::String(method.to_string()), arg];
                        call_builtin_vm!("call_method", &args)
                    },
                    |a, b| async move {
                        let (a_acc, b_acc) = accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                        call_builtin_vm!("rdivide", &[a_acc, b_acc])
                    },
                ).await?;
            }
            Instr::ElemPow => {
                arithmetic_ops::power(
                    &mut stack,
                    |obj, _method, arg| async move { call_builtin_vm!("power", &[obj, arg]) },
                    |a, b| async move {
                        let (a_acc, b_acc) = accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                        call_builtin_vm!("power", &[a_acc, b_acc])
                    },
                ).await?;
            }
            Instr::ElemLeftDiv => {
                arithmetic_ops::binary_method(
                    &mut stack,
                    "ldivide",
                    |obj, method, arg| async move {
                        let args = vec![obj, Value::String(method.to_string()), arg];
                        call_builtin_vm!("call_method", &args)
                    },
                    |a, b| async move {
                        let (b_acc, a_acc) = accel_promote_binary(AutoBinaryOp::Elementwise, &b, &a).await?;
                        call_builtin_vm!("rdivide", &[b_acc, a_acc])
                    },
                ).await?;
            }
            Instr::LessEqual => {
                comparison_ops::relation_inverted(
                    &mut stack,
                    "le",
                    "gt",
                    "ge",
                    "lt",
                    |aa, bb| aa <= bb,
                    |obj, method, arg| async move {
                        let args = vec![obj, Value::String(method.to_string()), arg];
                        call_builtin_vm!("call_method", &args)
                    },
                    |name, a, b| async move { call_builtin_vm!(name, &[a, b]) },
                    |v, label| async move { logical_truth_from_value(&v, &label).await },
                ).await?;
            }
            Instr::Less => {
                comparison_ops::relation(
                    &mut stack,
                    "lt",
                    "gt",
                    |aa, bb| aa < bb,
                    |obj, method, arg| async move {
                        let args = vec![obj, Value::String(method.to_string()), arg];
                        call_builtin_vm!("call_method", &args)
                    },
                    |name, a, b| async move { call_builtin_vm!(name, &[a, b]) },
                ).await?;
            }
            Instr::Greater => {
                comparison_ops::relation(
                    &mut stack,
                    "gt",
                    "lt",
                    |aa, bb| aa > bb,
                    |obj, method, arg| async move {
                        let args = vec![obj, Value::String(method.to_string()), arg];
                        call_builtin_vm!("call_method", &args)
                    },
                    |name, a, b| async move { call_builtin_vm!(name, &[a, b]) },
                ).await?;
            }
            Instr::GreaterEqual => {
                comparison_ops::relation_inverted(
                    &mut stack,
                    "ge",
                    "lt",
                    "le",
                    "gt",
                    |aa, bb| aa >= bb,
                    |obj, method, arg| async move {
                        let args = vec![obj, Value::String(method.to_string()), arg];
                        call_builtin_vm!("call_method", &args)
                    },
                    |name, a, b| async move { call_builtin_vm!(name, &[a, b]) },
                    |v, label| async move { logical_truth_from_value(&v, &label).await },
                ).await?;
            }
            Instr::Equal => {
                comparison_ops::equal(
                    &mut stack,
                    |obj, method, arg| async move {
                        let args = vec![obj, Value::String(method.to_string()), arg];
                        call_builtin_vm!("call_method", &args)
                    },
                    |name, a, b| async move { call_builtin_vm!(name, &[a, b]) },
                    |_v, _label| async move { Ok(false) },
                ).await?;
            }
            Instr::NotEqual => {
                comparison_ops::not_equal(
                    &mut stack,
                    |obj, method, arg| async move {
                        let args = vec![obj, Value::String(method.to_string()), arg];
                        call_builtin_vm!("call_method", &args)
                    },
                    |name, a, b| async move { call_builtin_vm!(name, &[a, b]) },
                    |v, label| async move { logical_truth_from_value(&v, &label).await },
                ).await?;
            }
            Instr::JumpIfFalse(target) => {
                let cond = pop_value(&mut stack)?;
                let truth = logical_truth_from_value(&cond, "if condition").await?;
                match control_flow::jump_if_false(truth, target) {
                    ControlFlowAction::Jump(target) => {
                        pc = target;
                        continue;
                    }
                    ControlFlowAction::Next | ControlFlowAction::Return => {}
                }
            }
            Instr::Jump(target) => {
                match control_flow::jump(target) {
                    ControlFlowAction::Jump(target) => {
                        pc = target;
                        continue;
                    }
                    ControlFlowAction::Next | ControlFlowAction::Return => unreachable!(),
                }
            }
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
            Instr::CallBuiltin(name, arg_count) => {
                if debug_stack {
                    debug!(
                        pc,
                        name,
                        arg_count,
                        stack_len = stack.len(),
                        top = ?stack.last(),
                        "[vm] CallBuiltin"
                    );
                }
                let call_counts = CALL_COUNTS.with(|cc| cc.borrow().clone());
                if let Some(value) = call_builtins::special_counter_builtin(&name, arg_count, &call_counts)? {
                    stack.push(value);
                    pc += 1;
                    continue;
                }
                let requested_outputs = call_builtins::requested_output_count(&bytecode.instructions, pc);
                let args = call_builtins::collect_call_args(&mut stack, arg_count)?;

                let _callsite_guard = runmat_runtime::callsite::push_callsite(
                    bytecode.source_id,
                    bytecode.call_arg_spans.get(pc).cloned().flatten(),
                );
                let output_hint = output_hint_for_single_result(&bytecode, pc);
                let _output_guard = push_output_count(output_hint);

                let prepared_primary = call_builtins::prepare_builtin_args(&name, &args).await?;
                let result = match requested_outputs {
                    Some(count) => {
                        runmat_runtime::call_builtin_async_with_outputs(&name, &prepared_primary, count)
                            .await
                    }
                    None => runmat_runtime::call_builtin_async(&name, &prepared_primary).await,
                };
                match result {
                    Ok(result) => stack.push(result),
                    Err(e) => {
                        let e = e;
                        let imported = call_builtins::resolve_imported_builtin(
                            &name,
                            &imports,
                            &prepared_primary,
                            requested_outputs,
                        )
                        .await?;
                        match imported {
                            ImportedBuiltinResolution::Resolved(value) => stack.push(value),
                            ImportedBuiltinResolution::Ambiguous(message) => vm_bail!(message),
                            ImportedBuiltinResolution::NotFound => {
                                if let Some(err) = call_builtins::rethrow_without_explicit_exception(
                                    &name,
                                    &args,
                                    last_exception.as_ref().map(|e| e.identifier.as_str()),
                                    last_exception.as_ref().map(|e| e.message.as_str()),
                                ) {
                                    vm_bail!(err);
                                }
                                if let Some((catch_pc, catch_var)) = try_stack.pop() {
                                    if let Some(var_idx) = catch_var {
                                        if var_idx >= vars.len() {
                                            vars.resize(var_idx + 1, Value::Num(0.0));
                                            refresh_workspace_state(&vars);
                                        }
                                        let mex = parse_exception(&e);
                                        last_exception = Some(mex.clone());
                                        vars[var_idx] = Value::MException(mex);
                                    }
                                    pc = catch_pc;
                                    continue;
                                } else {
                                    return Err(e);
                                }
                            }
                        }
                    }
                }
            }
            Instr::CallBuiltinExpandLast(name, fixed_argc, num_indices) => {
                // Stack layout: [..., a1, a2, ..., a_fixed, base_for_cell, idx1, idx2, ...]
                // Build args vector by first collecting fixed args, then expanding cell indexing into comma-list
                // Evaluate indices and base
                let mut indices = Vec::with_capacity(num_indices);
                for _ in 0..num_indices {
                    let v = stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?;
                    indices.push(v);
                }
                indices.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                // Collect fixed args
                let mut fixed = Vec::with_capacity(fixed_argc);
                for _ in 0..fixed_argc {
                    fixed.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                fixed.reverse();
                // Evaluate cell indexing, then flatten cell contents to extend args
                let expanded = match (base, indices.len()) {
                    (Value::Cell(ca), 1) | (Value::Cell(ca), 2) => {
                        call_shared::expand_cell_indices(&ca, &indices)?
                    }
                    (other, _) => {
                        // Route to subsref(obj,'{}',{indices...}) if object
                        match other {
                            Value::Object(obj) => {
                                let cell = call_shared::subsref_paren_index_cell(&indices)?;
                                let v = match call_builtin_vm!(
                                    "call_method",
                                    &[
                                        Value::Object(obj),
                                        Value::String("subsref".to_string()),
                                        Value::String("()".to_string()),
                                        cell,
                                    ],
                                ) {
                                    Ok(v) => v,
                                    Err(e) => vm_bail!(e),
                                };
                                vec![v]
                            }
                            _ => {
                                return Err(mex(
                                    "ExpandError",
                                    "CallBuiltinExpandLast requires cell or object cell access",
                                ))
                            }
                        }
                    }
                };
                let mut args = fixed;
                args.extend(expanded.into_iter());
                let output_hint = output_hint_for_single_result(&bytecode, pc);
                let _output_guard = push_output_count(output_hint);
                match call_builtin_auto(&name, &args).await {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::CallBuiltinExpandAt(name, before_count, num_indices, after_count) => {
                // Stack layout: [..., a1..abefore, base, idx..., a_after...]
                let mut after: Vec<Value> = Vec::with_capacity(after_count);
                for _ in 0..after_count {
                    after.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                after.reverse();
                let mut indices = Vec::with_capacity(num_indices);
                for _ in 0..num_indices {
                    indices.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                indices.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let mut before: Vec<Value> = Vec::with_capacity(before_count);
                for _ in 0..before_count {
                    before.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                before.reverse();
                let expanded = match (base, indices.len()) {
                    (Value::Cell(ca), 1) | (Value::Cell(ca), 2) => {
                        call_shared::expand_cell_indices(&ca, &indices)?
                    }
                    (Value::Object(obj), _) => {
                        let idx_vals = call_shared::subsref_brace_numeric_index_values(&indices);
                        let cell = call_builtin_vm!("__make_cell", &idx_vals)?;
                        let v = match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("{}".to_string()),
                                cell,
                            ],
                        ) {
                            Ok(v) => v,
                            Err(e) => vm_bail!(e),
                        };
                        vec![v]
                    }
                    _ => {
                        return Err(mex(
                            "ExpandError",
                            "CallBuiltinExpandAt requires cell or object cell access",
                        ))
                    }
                };
                let mut args = before;
                args.extend(expanded.into_iter());
                args.extend(after.into_iter());
                let output_hint = output_hint_for_single_result(&bytecode, pc);
                let _output_guard = push_output_count(output_hint);
                match call_builtin_auto(&name, &args).await {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::CallBuiltinExpandMulti(name, specs) => {
                let args = call_shared::build_expanded_args_from_specs(
                    &mut stack,
                    &specs,
                    "CallBuiltinExpandMulti requires cell or object for expand_all",
                    "CallBuiltinExpandMulti requires cell or object cell access",
                    |base| async move { match base {
                        Value::Object(obj) => {
                            let empty = call_shared::subsref_empty_brace_cell()?;
                            let args = vec![
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("{}".to_string()),
                                empty,
                            ];
                            let v = runmat_runtime::call_builtin_async("call_method", &args).await?;
                            Ok(match v {
                                Value::Cell(ca) => call_shared::expand_all_cell(&ca),
                                other => vec![other],
                            })
                        }
                        _ => Err(mex(
                            "ExpandError",
                            "CallBuiltinExpandMulti requires cell or object for expand_all",
                        )),
                    }},
                    |base, indices| async move { match base {
                        Value::Object(obj) => {
                            let idx_vals = call_shared::subsref_brace_numeric_index_values(&indices);
                            let cell = runmat_runtime::call_builtin_async("__make_cell", &idx_vals).await?;
                            let args = vec![
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("{}".to_string()),
                                cell,
                            ];
                            let v = runmat_runtime::call_builtin_async("call_method", &args).await?;
                            Ok(vec![v])
                        }
                        _ => Err(mex(
                            "ExpandError",
                            "CallBuiltinExpandMulti requires cell or object cell access",
                        )),
                    }},
                ).await?;
                let output_hint = output_hint_for_single_result(&bytecode, pc);
                let _output_guard = push_output_count(output_hint);
                match call_builtin_auto(&name, &args).await {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::PackToRow(count) => {
                array_ops::pack_to_row(&mut stack, count)?;
            }
            Instr::PackToCol(count) => {
                array_ops::pack_to_col(&mut stack, count)?;
            }
            Instr::CallFunctionExpandMulti(name, specs) => {
                // Build args via specs, then invoke user function similar to CallFunction
                let mut temp: Vec<Value> = Vec::new();
                for spec in specs.iter().rev() {
                    if spec.is_expand {
                        let mut indices = Vec::with_capacity(spec.num_indices);
                        for _ in 0..spec.num_indices {
                            indices.push(
                                stack
                                    .pop()
                                    .ok_or(mex("StackUnderflow", "stack underflow"))?,
                            );
                        }
                        indices.reverse();
                        let base = stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?;
                        let expanded = if spec.expand_all {
                            match base {
                                Value::Cell(ca) => ca.data.iter().map(|p| (*(*p)).clone()).collect::<Vec<Value>>(),
                                Value::Object(obj) => {
                                    let empty = runmat_builtins::CellArray::new(vec![], 1, 0).map_err(|e| format!("subsref build error: {e}"))?;
                                    let v = match call_builtin_vm!("call_method", &[
                                        Value::Object(obj),
                                        Value::String("subsref".to_string()),
                                        Value::String("{}".to_string()),
                                        Value::Cell(empty),
                                    ]) { Ok(v) => v, Err(e) => vm_bail!(e) };
                                    match v { Value::Cell(ca) => ca.data.iter().map(|p| (*(*p)).clone()).collect::<Vec<Value>>(), other => vec![other] }
                                }
                                _ => {
                                    return Err(mex(
                                        "InvalidExpandAllTarget",
                                        "CallFunctionExpandMulti requires cell or object for expand_all",
                                    ))
                                }
                            }
                        } else {
                            match (base, indices.len()) {
                                (Value::Cell(ca), 1) => match &indices[0] {
                                    Value::Num(n) => {
                                        let idx = *n as usize;
                                        if idx == 0 || idx > ca.data.len() {
                                            return Err(mex(
                                                "CellIndexOutOfBounds",
                                                "Cell index out of bounds",
                                            ));
                                        }
                                        vec![(*ca.data[idx - 1]).clone()]
                                    }
                                    Value::Int(i) => {
                                        let idx = i.to_i64() as usize;
                                        if idx == 0 || idx > ca.data.len() {
                                            return Err(mex(
                                                "CellIndexOutOfBounds",
                                                "Cell index out of bounds",
                                            ));
                                        }
                                        vec![(*ca.data[idx - 1]).clone()]
                                    }
                                    Value::Tensor(t) => {
                                        let mut out: Vec<Value> = Vec::with_capacity(t.data.len());
                                        for &val in &t.data {
                                            let iu = val as usize;
                                            if iu == 0 || iu > ca.data.len() {
                                                return Err(mex(
                                                    "CellIndexOutOfBounds",
                                                    "Cell index out of bounds",
                                                ));
                                            }
                                            out.push((*ca.data[iu - 1]).clone());
                                        }
                                        out
                                    }
                                    _ => {
                                        return Err(mex(
                                            "CellIndexType",
                                            "Unsupported cell index type",
                                        ))
                                    }
                                },
                                (Value::Cell(ca), 2) => {
                                    let r: f64 = (&indices[0]).try_into()?;
                                    let c: f64 = (&indices[1]).try_into()?;
                                    let (ir, ic) = (r as usize, c as usize);
                                    if ir == 0 || ir > ca.rows || ic == 0 || ic > ca.cols {
                                        return Err(mex(
                                            "CellSubscriptOutOfBounds",
                                            "Cell subscript out of bounds",
                                        ));
                                    }
                                    vec![(*ca.data[(ir - 1) * ca.cols + (ic - 1)]).clone()]
                                }
                                (Value::Object(obj), _) => {
                                    let cell = runmat_builtins::CellArray::new(
                                        indices.clone(),
                                        1,
                                        indices.len(),
                                    )
                                    .map_err(|e| format!("subsref build error: {e}"))?;
                                    let v = match call_builtin_vm!(
                                        "call_method",
                                        &[
                                            Value::Object(obj),
                                            Value::String("subsref".to_string()),
                                            Value::String("{}".to_string()),
                                            Value::Cell(cell),
                                        ],
                                    ) {
                                        Ok(v) => v,
                                        Err(e) => vm_bail!(e),
                                    };
                                    vec![v]
                                }
                                _ => return Err(
                                    "CallFunctionExpandMulti requires cell or object cell access"
                                        .to_string()
                                        .into(),
                                ),
                            }
                        };
                        for v in expanded {
                            temp.push(v);
                        }
                    } else {
                        temp.push(
                            stack
                                .pop()
                                .ok_or(mex("StackUnderflow", "stack underflow"))?,
                        );
                    }
                }
                temp.reverse();
                let args = temp;
                let func: UserFunction = match bytecode.functions.get(&name) {
                    Some(f) => f.clone(),
                    None => vm_bail!(mex(
                        "UndefinedFunction",
                        &format!("Undefined function: {name}")
                    )),
                };
                let var_map = runmat_hir::remapping::create_complete_function_var_map(
                    &func.params,
                    &func.outputs,
                    &func.body,
                );
                let local_var_count = var_map.len();
                let remapped_body =
                    runmat_hir::remapping::remap_function_body(&func.body, &var_map);
                let func_vars_count = local_var_count.max(func.params.len());
                let mut func_vars = vec![Value::Num(0.0); func_vars_count];
                for (i, _param_id) in func.params.iter().enumerate() {
                    if i < args.len() && i < func_vars.len() {
                        func_vars[i] = args[i].clone();
                    }
                }
                for (original_var_id, local_var_id) in &var_map {
                    let local_index = local_var_id.0;
                    let global_index = original_var_id.0;
                    if local_index < func_vars.len() && global_index < vars.len() {
                        let is_parameter = func
                            .params
                            .iter()
                            .any(|param_id| param_id == original_var_id);
                        if !is_parameter {
                            func_vars[local_index] = vars[global_index].clone();
                        }
                    }
                }
                let mut func_var_types = func.var_types.clone();
                if func_var_types.len() < local_var_count {
                    func_var_types.resize(local_var_count, Type::Unknown);
                }
                let func_program = runmat_hir::HirProgram {
                    body: remapped_body,
                    var_types: func_var_types,
                };
                let mut func_bytecode = crate::compile(&func_program, &bytecode.functions)?;
                func_bytecode.source_id = func.source_id;
                let _call_frame_guard = push_call_frame(&name, &bytecode, pc);
                // Make nested closures visible to outer frames
                for (k, v) in func_bytecode.functions.iter() {
                    context.functions.insert(k.clone(), v.clone());
                }
                let func_result_vars =
                    match Box::pin(interpret_function(&func_bytecode, func_vars)).await {
                        Ok(v) => v,
                        Err(e) => vm_bail!(e),
                    };
                if let Some(output_var_id) = func.outputs.first() {
                    let local_output_index = var_map.get(output_var_id).map(|id| id.0).unwrap_or(0);
                    if local_output_index < func_result_vars.len() {
                        stack.push(func_result_vars[local_output_index].clone());
                    } else {
                        stack.push(Value::Num(0.0));
                    }
                } else {
                    stack.push(Value::Num(0.0));
                }
            }
            Instr::CallFunctionExpandAt(name, before_count, num_indices, after_count) => {
                // Stack layout: [..., a1..abefore, base, idx..., a_after...]
                let mut after: Vec<Value> = Vec::with_capacity(after_count);
                for _ in 0..after_count {
                    after.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                after.reverse();
                let mut indices = Vec::with_capacity(num_indices);
                for _ in 0..num_indices {
                    indices.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                indices.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let mut before: Vec<Value> = Vec::with_capacity(before_count);
                for _ in 0..before_count {
                    before.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                before.reverse();
                let expanded = match (base, indices.len()) {
                    (Value::Cell(ca), 1) | (Value::Cell(ca), 2) => {
                        call_shared::expand_cell_indices(&ca, &indices)?
                    }
                    (Value::Object(obj), _) => {
                        let cell = call_shared::subsref_brace_index_cell_raw(&indices)?;
                        let v = match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("{}".to_string()),
                                cell,
                            ],
                        ) {
                            Ok(v) => v,
                            Err(e) => vm_bail!(e),
                        };
                        vec![v]
                    }
                    _ => {
                        return Err(mex(
                            "ExpandError",
                            "CallFunctionExpandAt requires cell or object cell access",
                        ))
                    }
                };
                let mut args = before;
                args.extend(expanded.into_iter());
                args.extend(after.into_iter());
                let func: UserFunction = match bytecode.functions.get(&name) {
                    Some(f) => f.clone(),
                    None => vm_bail!(mex(
                        "UndefinedFunction",
                        &format!("Undefined function: {name}")
                    )),
                };
                let var_map = runmat_hir::remapping::create_complete_function_var_map(
                    &func.params,
                    &func.outputs,
                    &func.body,
                );
                let local_var_count = var_map.len();
                let remapped_body =
                    runmat_hir::remapping::remap_function_body(&func.body, &var_map);
                let func_vars_count = local_var_count.max(func.params.len());
                let mut func_vars = vec![Value::Num(0.0); func_vars_count];
                for (i, _param_id) in func.params.iter().enumerate() {
                    if i < args.len() && i < func_vars.len() {
                        func_vars[i] = args[i].clone();
                    }
                }
                for (original_var_id, local_var_id) in &var_map {
                    let local_index = local_var_id.0;
                    let global_index = original_var_id.0;
                    if local_index < func_vars.len() && global_index < vars.len() {
                        let is_parameter = func
                            .params
                            .iter()
                            .any(|param_id| param_id == original_var_id);
                        if !is_parameter {
                            func_vars[local_index] = vars[global_index].clone();
                        }
                    }
                }
                let mut func_var_types = func.var_types.clone();
                if func_var_types.len() < local_var_count {
                    func_var_types.resize(local_var_count, Type::Unknown);
                }
                let func_program = runmat_hir::HirProgram {
                    body: remapped_body,
                    var_types: func_var_types,
                };
                let func_bytecode = crate::compile(&func_program, &bytecode.functions)?;
                // Make nested closures visible to outer frames
                for (k, v) in func_bytecode.functions.iter() {
                    context.functions.insert(k.clone(), v.clone());
                }
                let func_result_vars =
                    match Box::pin(interpret_function(&func_bytecode, func_vars)).await {
                        Ok(v) => v,
                        Err(e) => vm_bail!(e),
                    };
                if let Some(output_var_id) = func.outputs.first() {
                    let local_output_index = var_map.get(output_var_id).map(|id| id.0).unwrap_or(0);
                    if local_output_index < func_result_vars.len() {
                        stack.push(func_result_vars[local_output_index].clone());
                    } else {
                        stack.push(Value::Num(0.0));
                    }
                } else {
                    stack.push(Value::Num(0.0));
                }
            }
            Instr::CallFunction(name, arg_count) => {
                // First, try runtime builtin fallback (some helpers like call_method)
                {
                    let args = call_builtins::collect_call_args(&mut stack, arg_count)?;
                    if let Some(result) = call_user::try_builtin_fallback_single(&name, &args).await? {
                        stack.push(result);
                        pc += 1;
                        continue;
                    }
                    for v in args.into_iter().rev() { stack.push(v); }
                }
                let func = match call_shared::lookup_user_function(&name, &bytecode.functions) {
                    Ok(func) => func,
                    Err(err) => vm_bail!(err),
                };
                let mut args = Vec::new();
                for _ in 0..arg_count {
                    args.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                args.reverse();
                let out_count = 1usize;
                if let Err(err) = call_shared::validate_user_function_arity(&name, &func, arg_count) {
                    vm_bail!(err);
                }
                let prepared = match call_shared::prepare_user_call(func, &args, &vars) {
                    Ok(prepared) => prepared,
                    Err(err) => vm_bail!(err),
                };
                let runmat_vm::call::shared::PreparedUserCall {
                    func,
                    var_map,
                    func_program,
                    func_vars,
                } = prepared;
                let mut func_bytecode = crate::compile(&func_program, &bytecode.functions)?;
                func_bytecode.source_id = func.source_id;
                let _call_frame_guard = push_call_frame(&name, &bytecode, pc);
                let func_result_vars = match interpret_function_with_counts(
                    &func_bytecode,
                    func_vars,
                    &name,
                    1,
                    arg_count,
                )
                .await
                {
                    Ok(v) => v,
                    Err(e) => {
                        if let Some((catch_pc, catch_var)) = try_stack.pop() {
                            if let Some(var_idx) = catch_var {
                                if var_idx >= vars.len() {
                                    vars.resize(var_idx + 1, Value::Num(0.0));
                                    refresh_workspace_state(&vars);
                                }
                                let mex = parse_exception(&e);

                                last_exception = Some(mex.clone());
                                vars[var_idx] = Value::MException(mex);
                            }
                            pc = catch_pc;
                            continue;
                        } else {
                            vm_bail!(e);
                        }
                    }
                };
                let outputs = match call_shared::collect_multi_outputs(
                    &name,
                    &func,
                    &var_map,
                    &func_result_vars,
                    out_count,
                ) {
                    Ok(outputs) => outputs,
                    Err(err) => vm_bail!(err),
                };
                for value in outputs {
                    stack.push(value);
                }
            }
            Instr::CallFunctionMulti(name, arg_count, out_count) => {
                // First, try runtime builtin fallback (some helpers like call_method)
                {
                    let args = call_builtins::collect_call_args(&mut stack, arg_count)?;
                    if let Some(result) = call_user::try_builtin_fallback_multi(&name, &args, out_count).await? {
                        stack.push(result);
                        pc += 1;
                        continue;
                    }
                    for v in args.into_iter().rev() { stack.push(v); }
                }
                let func = match call_shared::lookup_user_function(&name, &bytecode.functions) {
                    Ok(func) => func,
                    Err(err) => vm_bail!(err),
                };
                let mut args = Vec::new();
                for _ in 0..arg_count {
                    args.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                args.reverse();
                if let Err(err) = call_shared::validate_user_function_arity(&name, &func, arg_count) {
                    vm_bail!(err);
                }
                let prepared = match call_shared::prepare_user_call(func, &args, &vars) {
                    Ok(prepared) => prepared,
                    Err(err) => vm_bail!(err),
                };
                let runmat_vm::call::shared::PreparedUserCall {
                    func,
                    var_map,
                    func_program,
                    func_vars,
                } = prepared;
                let func_bytecode = crate::compile(&func_program, &bytecode.functions)?;
                let func_result_vars = match interpret_function_with_counts(
                    &func_bytecode,
                    func_vars,
                    &name,
                    out_count,
                    arg_count,
                )
                .await
                {
                    Ok(v) => v,
                    Err(e) => {
                        if let Some((catch_pc, catch_var)) = try_stack.pop() {
                            if let Some(var_idx) = catch_var {
                                if var_idx >= vars.len() {
                                    vars.resize(var_idx + 1, Value::Num(0.0));
                                    refresh_workspace_state(&vars);
                                }
                                let mex = parse_exception(&e);
                                last_exception = Some(mex.clone());
                                vars[var_idx] = Value::MException(mex);
                            }
                            pc = catch_pc;
                            continue;
                        } else {
                            vm_bail!(e);
                        }
                    }
                };
                let outputs = match call_shared::collect_multi_outputs(
                    &name,
                    &func,
                    &var_map,
                    &func_result_vars,
                    out_count,
                ) {
                    Ok(outputs) => outputs,
                    Err(err) => vm_bail!(err),
                };
                stack.push(Value::OutputList(outputs));
            }
            Instr::EnterTry(catch_pc, catch_var) => {
                control_flow::enter_try(&mut try_stack, catch_pc, catch_var);
            }
            Instr::PopTry => {
                control_flow::pop_try(&mut try_stack);
            }
            Instr::CreateMatrix(rows, cols) => {
                array_ops::create_matrix(&mut stack, rows, cols)?;
            }
            Instr::CreateMatrixDynamic(num_rows) => {
                array_ops::create_matrix_dynamic(&mut stack, num_rows, |rows_data| async move {
                    runmat_runtime::create_matrix_from_values(&rows_data).await
                }).await?;
            }
            Instr::CreateRange(has_step) => {
                array_ops::create_range(&mut stack, has_step, |args| async move {
                    call_builtin_vm!("colon", &args)
                }).await?;
            }
            Instr::Index(num_indices) => {
                let indices = idx_read_linear::collect_linear_indices(&mut stack, num_indices).await?;
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                #[cfg(feature = "native-accel")]
                clear_residency(&base);
                match base {
                    Value::Object(obj) => {
                        let cell = idx_read_linear::build_object_subsref_cell(&indices)?;
                        match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("()".to_string()),
                                cell,
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e.to_string()),
                        }
                    }
                    Value::HandleObject(handle) => {
                        let cell = idx_read_linear::build_object_subsref_cell(&indices)?;
                        match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::HandleObject(handle),
                                Value::String("subsref".to_string()),
                                Value::String("()".to_string()),
                                cell,
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e.to_string()),
                        }
                    }
                    other => {
                        let result = match idx_read_linear::generic_index(&other, &indices).await {
                            Ok(v) => v,
                            Err(e) => vm_bail!(e),
                        };
                        stack.push(result);
                    }
                }
            }
            Instr::IndexSlice(dims, numeric_count, colon_mask, end_mask) => {
                let __b = bench_start();
                // Pop numeric indices in reverse order (they were pushed in order), then base
                let mut numeric: Vec<Value> = Vec::with_capacity(numeric_count);
                for _ in 0..numeric_count {
                    numeric.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                numeric.reverse();
                let mut base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let mut logical_base = false;
                base = match base {
                    Value::LogicalArray(la) => {
                        logical_base = true;
                        let data: Vec<f64> = la
                            .data
                            .iter()
                            .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                            .collect();
                        let tensor = runmat_builtins::Tensor::new(data, la.shape.clone())
                            .map_err(|e| format!("slice: {e}"))?;
                        Value::Tensor(tensor)
                    }
                    other => other,
                };
                match base {
                    Value::Object(obj) => {
                        match idx_read_slice::object_subsref_paren(Value::Object(obj), &numeric)
                            .await
                        {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e.to_string()),
                        }
                    }
                    Value::HandleObject(handle) => {
                        match idx_read_slice::object_subsref_paren(
                            Value::HandleObject(handle),
                            &numeric,
                        )
                        .await
                        {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e.to_string()),
                        }
                    }
                    Value::Tensor(t) => {
                        if dims == 1 {
                            match idx_read_slice::read_tensor_slice_1d(
                                &t,
                                colon_mask,
                                end_mask,
                                &numeric,
                            )
                            .await
                            {
                                Ok(v) => stack.push(v),
                                Err(e) => vm_bail!(e),
                            }
                        } else {
                            match idx_read_slice::read_tensor_slice_nd(
                                &t,
                                dims,
                                colon_mask,
                                end_mask,
                                &numeric,
                            )
                            .await
                            {
                                Ok(v) => stack.push(v),
                                Err(e) => vm_bail!(e),
                            }
                        }
                    }
                    Value::ComplexTensor(ct) => {
                        let result = idx_read_slice::read_complex_slice(
                            &ct,
                            dims,
                            colon_mask,
                            end_mask,
                            &numeric,
                        )
                        .await
                        .map_err(|e| format!("slice: {e}"))?;
                        stack.push(result);
                    }
                    Value::GpuTensor(handle) => {
                        let result = idx_read_slice::read_gpu_slice(
                            &handle,
                            dims,
                            colon_mask,
                            end_mask,
                            &numeric,
                        )
                        .await
                        .map_err(|e| format!("slice: {e}"))?;
                        stack.push(result);
                    }
                    Value::StringArray(sa) => {
                        match idx_read_slice::read_string_slice(
                            &sa,
                            dims,
                            colon_mask,
                            end_mask,
                            &numeric,
                        )
                        .await
                        {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    other => {
                        // Support 1-D linear indexing and scalar(1) on non-tensors
                        if dims == 1 {
                            let is_colon = (colon_mask & 1u32) != 0;
                            let is_end = (end_mask & 1u32) != 0;
                            if is_colon {
                                vm_bail!(mex(
                                    "SliceNonTensor",
                                    "Slicing only supported on tensors"
                                ));
                            }
                            let idx_val: f64 = if is_end {
                                1.0
                            } else {
                                match numeric.first() {
                                    Some(Value::Num(n)) => *n,
                                    Some(Value::Int(i)) => i.to_f64(),
                                    _ => 1.0,
                                }
                            };
                            let v = match runmat_runtime::perform_indexing(&other, &[idx_val]).await
                            {
                                Ok(v) => v,
                                Err(_e) => vm_bail!(mex(
                                    "SliceNonTensor",
                                    "Slicing only supported on tensors"
                                )),
                            };
                            stack.push(v);
                        } else {
                            vm_bail!(mex("SliceNonTensor", "Slicing only supported on tensors"));
                        }
                    }
                }
                if logical_base {
                    let result = stack
                        .pop()
                        .ok_or(mex("SliceNonTensor", "logical slice missing result"))?;
                    let converted = match result {
                        Value::Tensor(t) => {
                            let logical_data: Vec<u8> = t
                                .data
                                .iter()
                                .map(|&v| if v != 0.0 { 1 } else { 0 })
                                .collect();
                            if logical_data.len() <= 1 {
                                Value::Bool(logical_data.first().copied().unwrap_or(0) != 0)
                            } else {
                                let logical = runmat_builtins::LogicalArray::new(
                                    logical_data,
                                    t.shape.clone(),
                                )
                                .map_err(|e| mex("SliceNonTensor", &format!("slice: {e}")))?;
                                Value::LogicalArray(logical)
                            }
                        }
                        Value::Num(n) => Value::Bool(n != 0.0),
                        Value::Bool(_) | Value::LogicalArray(_) => result,
                        other => other,
                    };
                    stack.push(converted);
                }
                bench_end("IndexSlice", __b);
            }
            Instr::IndexSliceExpr {
                dims,
                numeric_count,
                colon_mask,
                end_mask,
                range_dims,
                range_has_step,
                range_start_exprs,
                range_step_exprs,
                range_end_exprs,
                end_numeric_exprs,
            } => {
                // Pop any numeric scalar indices (reverse), then for each range in reverse push step (if has), start; then base
                let mut numeric: Vec<Value> = Vec::with_capacity(numeric_count);
                for _ in 0..numeric_count {
                    numeric.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                numeric.reverse();
                // Gather per-range params in reverse order of pushes
                let mut range_params: Vec<(f64, f64)> = Vec::with_capacity(range_dims.len());
                for i in (0..range_dims.len()).rev() {
                    let has_step = range_has_step[i];
                    let step = if has_step {
                        let v = stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?;
                        match v {
                            Value::Num(n) => n,
                            Value::Int(i) => i.to_f64(),
                            Value::Tensor(t) if !t.data.is_empty() => t.data[0],
                            _ => 1.0,
                        }
                    } else {
                        1.0
                    };
                    let v = stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?;
                    let start: f64 = match v {
                        Value::Num(n) => n,
                        Value::Int(i) => i.to_f64(),
                        Value::Tensor(t) if !t.data.is_empty() => t.data[0],
                        _ => 1.0,
                    };
                    range_params.push((start, step));
                }
                range_params.reverse();
                let mut base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                #[cfg(feature = "native-accel")]
                clear_residency(&base);
                if !end_numeric_exprs.is_empty() {
                    numeric = match &base {
                        Value::GpuTensor(handle) => {
                            apply_end_offsets_to_numeric(
                                &numeric,
                                IndexContext::new(dims, colon_mask, end_mask, &handle.shape),
                                &end_numeric_exprs,
                                &mut vars,
                                &context.functions,
                            )
                            .await?
                        }
                        Value::Tensor(t) => {
                            apply_end_offsets_to_numeric(
                                &numeric,
                                IndexContext::new(dims, colon_mask, end_mask, &t.shape),
                                &end_numeric_exprs,
                                &mut vars,
                                &context.functions,
                            )
                            .await?
                        }
                        Value::ComplexTensor(t) => {
                            apply_end_offsets_to_numeric(
                                &numeric,
                                IndexContext::new(dims, colon_mask, end_mask, &t.shape),
                                &end_numeric_exprs,
                                &mut vars,
                                &context.functions,
                            )
                            .await?
                        }
                        _ => numeric,
                    };
                }
                if let Value::GpuTensor(handle) = &base {
                    if let Some(provider) = runmat_accelerate_api::provider() {
                        let normalized_numeric: Vec<Value> = {
                            let mut out = Vec::with_capacity(numeric.len());
                            for value in &numeric {
                                if let Some(idx) = index_scalar_from_value(value).await? {
                                    out.push(Value::Int(runmat_builtins::IntValue::I64(idx)));
                                } else {
                                    out.push(
                                        runmat_runtime::dispatcher::gather_if_needed_async(value)
                                            .await?,
                                    );
                                }
                            }
                            out
                        };
                        let attempt = async {
                            let rank = handle.shape.len();
                            #[derive(Clone)]
                            enum Sel {
                                Colon,
                                Scalar(usize),
                                Indices(Vec<usize>),
                                Range {
                                    start: i64,
                                    step: i64,
                                    end_off: EndExpr,
                                },
                            }
                            let full_shape: Vec<usize> = if dims == 1 {
                                vec![total_len_from_shape(&handle.shape)]
                            } else if rank < dims {
                                let mut s = handle.shape.clone();
                                s.resize(dims, 1);
                                s
                            } else {
                                handle.shape.clone()
                            };
                            let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                            let mut num_iter = 0usize;
                            let mut rp_iter = 0usize;
                            for d in 0..dims {
                                let is_colon = (colon_mask & (1u32 << d)) != 0;
                                let is_end = (end_mask & (1u32 << d)) != 0;
                                if is_colon {
                                    selectors.push(Sel::Colon);
                                } else if is_end {
                                    selectors.push(Sel::Scalar(*full_shape.get(d).unwrap_or(&1)));
                                } else if let Some(pos) = range_dims.iter().position(|&rd| rd == d)
                                {
                                    let (raw_st, raw_sp) = range_params[rp_iter];
                                    let dim_len = *full_shape.get(d).unwrap_or(&1);
                                    let st = if let Some(expr) = &range_start_exprs[rp_iter] {
                                        resolve_range_end_index(
                                            dim_len,
                                            expr,
                                            &mut vars,
                                            &context.functions,
                                        )
                                        .await? as f64
                                    } else {
                                        raw_st
                                    };
                                    let sp = if let Some(expr) = &range_step_exprs[rp_iter] {
                                        resolve_range_end_index(
                                            dim_len,
                                            expr,
                                            &mut vars,
                                            &context.functions,
                                        )
                                        .await? as f64
                                    } else {
                                        raw_sp
                                    };
                                    rp_iter += 1;
                                    let off = range_end_exprs[pos].clone();
                                    selectors.push(Sel::Range {
                                        start: st as i64,
                                        step: if sp >= 0.0 {
                                            sp as i64
                                        } else {
                                            -(sp.abs() as i64)
                                        },
                                        end_off: off.clone(),
                                    });
                                } else {
                                    let v = normalized_numeric.get(num_iter).ok_or(mex(
                                        "MissingNumericIndex",
                                        "missing numeric index",
                                    ))?;
                                    num_iter += 1;
                                    if let Value::Int(idx_val) = v {
                                        let idx = idx_val.to_i64();
                                        if idx < 1 {
                                            return Err(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds",
                                            ));
                                        }
                                        selectors.push(Sel::Scalar(idx as usize));
                                    } else {
                                        match v {
                                            Value::Tensor(idx_t) => {
                                                let dim_len = *full_shape.get(d).unwrap_or(&1);
                                                let len = idx_t.shape.iter().product::<usize>();
                                                if len == dim_len {
                                                    let mut v = Vec::new();
                                                    for (i, &val) in idx_t.data.iter().enumerate() {
                                                        if val != 0.0 {
                                                            v.push(i + 1);
                                                        }
                                                    }
                                                    selectors.push(Sel::Indices(v));
                                                } else {
                                                    let mut v = Vec::with_capacity(len);
                                                    for &val in &idx_t.data {
                                                        let idx = val as isize;
                                                        if idx < 1 {
                                                            return Err(mex(
                                                                "IndexOutOfBounds",
                                                                "Index out of bounds",
                                                            ));
                                                        }
                                                        v.push(idx as usize);
                                                    }
                                                    selectors.push(Sel::Indices(v));
                                                }
                                            }
                                            _ => {
                                                return Err(mex(
                                                    "UnsupportedIndexType",
                                                    "Unsupported index type",
                                                ))
                                            }
                                        }
                                    }
                                }
                            }
                            let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                            for (d, sel) in selectors.iter().enumerate().take(dims) {
                                let dim_len = *full_shape.get(d).unwrap_or(&1) as i64;
                                let idxs: Vec<usize> = match sel {
                                    Sel::Colon => (1..=dim_len as usize).collect(),
                                    Sel::Scalar(i) => vec![*i],
                                    Sel::Indices(v) => v.clone(),
                                    Sel::Range {
                                        start,
                                        step,
                                        end_off,
                                    } => {
                                        let mut v = Vec::new();
                                        let mut cur = *start;
                                        let end_i = resolve_range_end_index(
                                            dim_len as usize,
                                            end_off,
                                            &mut vars,
                                            &context.functions,
                                        )
                                        .await?;
                                        if *step == 0 {
                                            return Err(mex(
                                                "IndexStepZero",
                                                "Index step cannot be zero",
                                            ));
                                        }
                                        if *step > 0 {
                                            while cur <= end_i {
                                                if cur < 1 || cur > dim_len {
                                                    break;
                                                }
                                                v.push(cur as usize);
                                                cur += *step;
                                            }
                                        } else {
                                            while cur >= end_i {
                                                if cur < 1 || cur > dim_len {
                                                    break;
                                                }
                                                v.push(cur as usize);
                                                cur += *step;
                                            }
                                        }
                                        v
                                    }
                                };
                                if idxs.iter().any(|&i| i == 0 || i > dim_len as usize) {
                                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                                }
                                per_dim_indices.push(idxs);
                            }
                            let total_out: usize =
                                per_dim_indices.iter().map(|v| v.len()).product();
                            if total_out == 0 {
                                return Ok((Vec::new(), vec![0, 0]));
                            }
                            let mut strides: Vec<usize> = vec![0; dims];
                            let mut acc = 1usize;
                            for (d, stride) in strides.iter_mut().enumerate().take(dims) {
                                *stride = acc;
                                acc *= full_shape[d];
                            }
                            let mut indices: Vec<u32> = Vec::with_capacity(total_out);
                            let mut idx = vec![0usize; dims];
                            loop {
                                let mut lin = 0usize;
                                for d in 0..dims {
                                    let i0 = per_dim_indices[d][idx[d]] - 1;
                                    lin += i0 * strides[d];
                                }
                                indices.push(lin as u32);
                                let mut d = 0usize;
                                while d < dims {
                                    idx[d] += 1;
                                    if idx[d] < per_dim_indices[d].len() {
                                        break;
                                    }
                                    idx[d] = 0;
                                    d += 1;
                                }
                                if d == dims {
                                    break;
                                }
                            }
                            let output_shape = if dims == 1 {
                                if total_out <= 1 {
                                    vec![1, 1]
                                } else {
                                    vec![total_out, 1]
                                }
                            } else {
                                per_dim_indices.iter().map(|v| v.len().max(1)).collect()
                            };
                            Ok((indices, output_shape))
                        }
                        .await;
                        if let Ok((indices, output_shape)) = attempt {
                            let vm_plan = idx_selectors::SlicePlan {
                                indices,
                                output_shape,
                                selection_lengths: Vec::new(),
                                dims,
                            };
                            if let Ok(result) = idx_read_slice::read_gpu_slice_from_plan(handle, &vm_plan)
                            {
                                stack.push(result);
                                pc += 1;
                                continue;
                            }
                        }
                        let host = provider
                            .download(handle)
                            .await
                            .map_err(|e| format!("slice: {e}"))?;
                        let tensor = runmat_builtins::Tensor::new(host.data, host.shape)
                            .map_err(|e| format!("slice: {e}"))?;
                        base = Value::Tensor(tensor);
                    } else {
                        return Err(mex(
                            "AccelerationProviderUnavailable",
                            "No acceleration provider registered",
                        ));
                    }
                }
                match base {
                    Value::ComplexTensor(t) => {
                        let vm_plan = idx_read_slice::build_expr_gather_plan(
                            dims,
                            colon_mask,
                            end_mask,
                            &range_dims,
                            &range_params,
                            &range_start_exprs,
                            &range_step_exprs,
                            &range_end_exprs,
                            &numeric,
                            &t.shape,
                            |dim_len, expr| {
                                let expr = expr.clone();
                                let vars_ref = &vars;
                                let functions_ref = &context.functions;
                                async move {
                                    resolve_range_end_index(dim_len, &expr, vars_ref, functions_ref)
                                        .await
                                }
                            },
                        )
                        .await?;
                        let result = idx_read_slice::read_complex_slice_from_plan(&t, &vm_plan)
                            .map_err(|e| format!("Slice error: {e}"))?;
                        stack.push(result);
                    }
                    Value::Tensor(t) => {
                        let vm_plan = idx_read_slice::build_expr_gather_plan(
                            dims,
                            colon_mask,
                            end_mask,
                            &range_dims,
                            &range_params,
                            &range_start_exprs,
                            &range_step_exprs,
                            &range_end_exprs,
                            &numeric,
                            &t.shape,
                            |dim_len, expr| {
                                let expr = expr.clone();
                                let vars_ref = &vars;
                                let functions_ref = &context.functions;
                                async move {
                                    resolve_range_end_index(dim_len, &expr, vars_ref, functions_ref)
                                        .await
                                }
                            },
                        )
                        .await?;
                        let result = idx_read_slice::read_tensor_slice_from_plan(&t, &vm_plan)
                            .map_err(|e| format!("Slice error: {e}"))?;
                        stack.push(result);
                    }
                    Value::StringArray(sa) => {
                        let selectors =
                            build_slice_selectors(dims, colon_mask, end_mask, &numeric, &sa.shape)
                                .await
                                .map_err(|e| format!("slice: {e}"))?;
                        let plan = build_slice_plan(&selectors, dims, &sa.shape)
                            .map_err(|e| map_slice_plan_error("slice", e))?;
                        let vm_plan = idx_selectors::SlicePlan {
                            indices: plan.indices.clone(),
                            output_shape: plan.output_shape.clone(),
                            selection_lengths: plan.selection_lengths.clone(),
                            dims: plan.dims,
                        };
                        let result = idx_read_slice::gather_string_slice(&sa, &vm_plan)
                            .map_err(|e| format!("slice: {e}"))?;
                        stack.push(result);
                    }
                    _ => vm_bail!(mex("SliceNonTensor", "Slicing only supported on tensors")),
                }
            }

            Instr::StoreSlice(dims, numeric_count, colon_mask, end_mask) => {
                let __b = bench_start();
                // RHS value to scatter, then numeric indices, then base
                let rhs = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let mut numeric: Vec<Value> = Vec::with_capacity(numeric_count);
                for _ in 0..numeric_count {
                    numeric.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                numeric.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Object(obj) => {
                        match idx_write_slice::object_subsasgn_paren(
                            Value::Object(obj.clone()),
                            &numeric,
                            rhs.clone(),
                        )
                        .await
                        {
                            Ok(v) => stack.push(v),
                            Err(_e) => {
                                // Fallback to direct builtin OverIdx.subsasgn if class method isn't registered
                                // Determine class name and call fully qualified builtin if present
                                let qualified = format!("{}.subsasgn", obj.class_name);
                                let cell = idx_write_slice::build_subsasgn_paren_cell(&numeric)?;
                                match call_builtin_vm!(
                                    &qualified,
                                    &[
                                        Value::Object(obj),
                                        Value::String("()".to_string()),
                                        cell,
                                        rhs,
                                    ],
                                ) {
                                    Ok(v2) => stack.push(v2),
                                    Err(e2) => vm_bail!(e2),
                                }
                            }
                        }
                    }
                    Value::HandleObject(handle) => {
                        match idx_write_slice::object_subsasgn_paren(
                            Value::HandleObject(handle),
                            &numeric,
                            rhs,
                        )
                        .await
                        {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e.to_string()),
                        }
                    }
                    Value::Tensor(t) => {
                        // F4: write barrier hook (placeholder) – in a full GC integration, call into GC pre/post here
                        // Linear 1-D indexing assignment: A(I) = rhs
                        if dims == 1 {
                            stack.push(
                                idx_write_slice::assign_tensor_slice_1d(
                                    t,
                                    colon_mask,
                                    end_mask,
                                    &numeric,
                                    rhs,
                                )
                                .await?,
                            );
                        } else {
                            let mut selectors: Vec<SliceSelector> = Vec::with_capacity(dims);
                            let mut num_iter = 0usize;
                            for d in 0..dims {
                                let is_colon = (colon_mask & (1u32 << d)) != 0;
                                let is_end = (end_mask & (1u32 << d)) != 0;
                                if is_colon {
                                    selectors.push(SliceSelector::Colon);
                                } else if is_end {
                                    selectors.push(SliceSelector::Scalar(*t.shape.get(d).unwrap_or(&1)));
                                } else {
                                    let v = numeric.get(num_iter).ok_or(mex(
                                        "MissingNumericIndex",
                                        "missing numeric index",
                                    ))?;
                                    num_iter += 1;
                                    let dim_len = *t.shape.get(d).unwrap_or(&1);
                                    let selector = match selector_from_value_dim(v, dim_len).await {
                                        Ok(selector) => selector,
                                        Err(err) => vm_bail!(err),
                                    };
                                    selectors.push(selector);
                                }
                            }
                            let vm_selectors: Vec<_> = selectors
                                .iter()
                                .map(|s| match s {
                                    SliceSelector::Colon => idx_selectors::SliceSelector::Colon,
                                    SliceSelector::Scalar(i) => idx_selectors::SliceSelector::Scalar(*i),
                                    SliceSelector::Indices(v) => idx_selectors::SliceSelector::Indices(v.clone()),
                                    SliceSelector::LinearIndices { values, output_shape } => {
                                        idx_selectors::SliceSelector::LinearIndices {
                                            values: values.clone(),
                                            output_shape: output_shape.clone(),
                                        }
                                    }
                                })
                                .collect();
                            stack.push(idx_write_slice::assign_tensor_slice_nd(t, dims, &vm_selectors, rhs)?);
                        }
                    }
                    Value::GpuTensor(handle) => {
                        stack.push(
                            idx_write_slice::assign_gpu_store_slice(
                                &handle,
                                dims,
                                colon_mask,
                                end_mask,
                                &numeric,
                                rhs,
                            )
                            .await?,
                        );
                    }
                    Value::ComplexTensor(mut ct) => {
                        let selectors =
                            build_slice_selectors(dims, colon_mask, end_mask, &numeric, &ct.shape)
                                .await
                                .map_err(|e| format!("slice assign: {e}"))?;
                        let plan = build_slice_plan(&selectors, dims, &ct.shape)
                            .map_err(|e| map_slice_plan_error("slice assign", e))?;
                        if plan.indices.is_empty() {
                            stack.push(Value::ComplexTensor(ct));
                            bench_end("StoreSlice", __b);
                            pc += 1;
                            continue;
                        }
                        let vm_plan = idx_selectors::SlicePlan {
                            indices: plan.indices.clone(),
                            output_shape: plan.output_shape.clone(),
                            selection_lengths: plan.selection_lengths.clone(),
                            dims: plan.dims,
                        };
                        let rhs_view = idx_write_slice::build_complex_rhs_view(&rhs, &plan.selection_lengths)
                            .map_err(|e| format!("slice assign: {e}"))?;
                        idx_write_slice::scatter_complex_with_plan(&mut ct, &vm_plan, &rhs_view)
                            .map_err(|e| format!("slice assign: {e}"))?;
                        stack.push(Value::ComplexTensor(ct));
                        bench_end("StoreSlice", __b);
                        pc += 1;
                        continue;
                    }
                    Value::StringArray(mut sa) => {
                        let selectors =
                            build_slice_selectors(dims, colon_mask, end_mask, &numeric, &sa.shape)
                                .await
                                .map_err(|e| format!("slice assign: {e}"))?;
                        let plan = build_slice_plan(&selectors, dims, &sa.shape)
                            .map_err(|e| map_slice_plan_error("slice assign", e))?;
                        if plan.indices.is_empty() {
                            stack.push(Value::StringArray(sa));
                            bench_end("StoreSlice", __b);
                            pc += 1;
                            continue;
                        }
                        let vm_plan = idx_selectors::SlicePlan {
                            indices: plan.indices.clone(),
                            output_shape: plan.output_shape.clone(),
                            selection_lengths: plan.selection_lengths.clone(),
                            dims: plan.dims,
                        };
                        let rhs_view = idx_write_slice::build_string_rhs_view(&rhs, &plan.selection_lengths)
                            .map_err(|e| format!("slice assign: {e}"))?;
                        idx_write_slice::scatter_string_with_plan(&mut sa, &vm_plan, &rhs_view)
                            .map_err(|e| format!("slice assign: {e}"))?;
                        stack.push(Value::StringArray(sa));
                        bench_end("StoreSlice", __b);
                        pc += 1;
                        continue;
                        // legacy path removed in favor of scatter_string_with_plan
                    }
                    other => {
                        warn!(
                            "StoreSlice: unsupported base {:?} dims={} numeric={:?} rhs={:?}",
                            other, dims, numeric, rhs
                        );
                        vm_bail!(
                            "Slicing assignment only supported on tensors or string arrays"
                                .to_string()
                        )
                    }
                }
                bench_end("StoreSlice", __b);
            }

            Instr::StoreSliceExpr {
                dims,
                numeric_count,
                colon_mask,
                end_mask,
                range_dims,
                range_has_step,
                range_start_exprs,
                range_step_exprs,
                range_end_exprs,
                end_numeric_exprs,
            } => {
                // RHS, range params (per range dim), then base with numeric scalar indices interleaved
                let mut rhs = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                // Pop per-range params in reverse order
                let mut range_params: Vec<(f64, f64)> = Vec::with_capacity(range_dims.len());
                for i in (0..range_dims.len()).rev() {
                    let has = range_has_step[i];
                    let step = if has {
                        let v: f64 = (&stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?)
                            .try_into()?;
                        v
                    } else {
                        1.0
                    };
                    let st: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    range_params.push((st, step));
                }
                range_params.reverse();
                let mut numeric: Vec<Value> = Vec::with_capacity(numeric_count);
                for _ in 0..numeric_count {
                    numeric.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                numeric.reverse();
                let mut base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                #[cfg(feature = "native-accel")]
                clear_residency(&base);
                // If base is not assignable but rhs is, swap them to handle reversed emission order
                let base_assignable = matches!(
                    base,
                    Value::Object(_)
                        | Value::Tensor(_)
                        | Value::ComplexTensor(_)
                        | Value::GpuTensor(_)
                );
                if !base_assignable
                    && matches!(
                        rhs,
                        Value::Object(_)
                            | Value::Tensor(_)
                            | Value::ComplexTensor(_)
                            | Value::GpuTensor(_)
                    )
                {
                    std::mem::swap(&mut base, &mut rhs);
                }
                if !end_numeric_exprs.is_empty() {
                    numeric = match &base {
                        Value::GpuTensor(handle) => {
                            apply_end_offsets_to_numeric(
                                &numeric,
                                IndexContext::new(dims, colon_mask, end_mask, &handle.shape),
                                &end_numeric_exprs,
                                &mut vars,
                                &context.functions,
                            )
                            .await?
                        }
                        Value::Tensor(t) => {
                            apply_end_offsets_to_numeric(
                                &numeric,
                                IndexContext::new(dims, colon_mask, end_mask, &t.shape),
                                &end_numeric_exprs,
                                &mut vars,
                                &context.functions,
                            )
                            .await?
                        }
                        Value::ComplexTensor(t) => {
                            apply_end_offsets_to_numeric(
                                &numeric,
                                IndexContext::new(dims, colon_mask, end_mask, &t.shape),
                                &end_numeric_exprs,
                                &mut vars,
                                &context.functions,
                            )
                            .await?
                        }
                        _ => numeric,
                    };
                }
                match base {
                    Value::ComplexTensor(mut t) => {
                        let vm_plan = idx_read_slice::build_expr_gather_plan(
                            dims,
                            colon_mask,
                            end_mask,
                            &range_dims,
                            &range_params,
                            &range_start_exprs,
                            &range_step_exprs,
                            &range_end_exprs,
                            &numeric,
                            &t.shape,
                            |dim_len, expr| {
                                let expr = expr.clone();
                                let vars_ref = &vars;
                                let functions_ref = &context.functions;
                                async move {
                                    resolve_range_end_index(dim_len, &expr, vars_ref, functions_ref)
                                        .await
                                }
                            },
                        )
                        .await?;
                        if !vm_plan.indices.is_empty() {
                            let rhs_view = idx_write_slice::build_complex_rhs_view(
                                &rhs,
                                &vm_plan.selection_lengths,
                            )
                            .map_err(|e| format!("slice assign: {e}"))?;
                            idx_write_slice::scatter_complex_with_plan(&mut t, &vm_plan, &rhs_view)
                                .map_err(|e| format!("slice assign: {e}"))?;
                        }
                        stack.push(Value::ComplexTensor(t));
                    }
                    Value::Tensor(t) => {
                        let selectors = idx_write_slice::build_expr_selectors(
                            dims,
                            colon_mask,
                            end_mask,
                            &range_dims,
                            &range_params,
                            &range_start_exprs,
                            &range_step_exprs,
                            &range_end_exprs,
                            &numeric,
                            &t.shape,
                            |dim_len, expr| {
                                let expr = expr.clone();
                                let vars_ref = &vars;
                                let functions_ref = &context.functions;
                                async move {
                                    resolve_range_end_index(dim_len, &expr, vars_ref, functions_ref)
                                        .await
                                }
                            },
                        )
                        .await?;
                        let updated = idx_write_slice::assign_tensor_slice_nd(t, dims, &selectors, rhs)?;
                        stack.push(updated);
                    }
                    Value::GpuTensor(h) => {
                        let updated = idx_write_slice::assign_gpu_store_slice(
                            &h,
                            dims,
                            colon_mask,
                            end_mask,
                            &numeric,
                            rhs,
                        )
                        .await?;
                        stack.push(updated);
                    }
                    Value::Object(obj) => {
                        // Build cell of per-dim index descriptors to pass to subsasgn
                        let mut idx_values: Vec<Value> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        let mut rp_iter = 0usize;
                        for d in 0..dims {
                            let is_colon = (colon_mask & (1u32 << d)) != 0;
                            let is_end = (end_mask & (1u32 << d)) != 0;
                            if is_colon {
                                idx_values.push(Value::String(":".to_string()));
                                continue;
                            }
                            if is_end {
                                idx_values.push(Value::String("end".to_string()));
                                continue;
                            }
                            if let Some(pos) = range_dims.iter().position(|&rd| rd == d) {
                                let (raw_st, raw_sp) = range_params[rp_iter];
                                let st = if let Some(expr) = &range_start_exprs[rp_iter] {
                                    encode_end_expr_value(expr)?
                                } else {
                                    Value::Num(raw_st)
                                };
                                let sp = if let Some(expr) = &range_step_exprs[rp_iter] {
                                    encode_end_expr_value(expr)?
                                } else {
                                    Value::Num(raw_sp)
                                };
                                rp_iter += 1;
                                let off = range_end_exprs[pos].clone();
                                idx_values.push(build_end_range_descriptor(st, sp, &off)?);
                            } else {
                                let v = numeric
                                    .get(num_iter)
                                    .ok_or(mex("MissingNumericIndex", "missing numeric index"))?;
                                num_iter += 1;
                                match v {
                                    Value::Num(n) => idx_values.push(Value::Num(*n)),
                                    Value::Int(i) => idx_values.push(Value::Num(i.to_f64())),
                                    Value::Tensor(t) => idx_values.push(Value::Tensor(t.clone())),
                                    other => {
                                        return Err(format!(
                                            "Unsupported index type for object: {other:?}"
                                        )
                                        .into())
                                    }
                                }
                            }
                        }
                        let cell = runmat_builtins::CellArray::new(idx_values, 1, dims)
                            .map_err(|e| format!("subsasgn build error: {e}"))?;
                        match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsasgn".to_string()),
                                Value::String("()".to_string()),
                                Value::Cell(cell),
                                rhs,
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    _ => vm_bail!("StoreSliceExpr only supports tensors currently".to_string()),
                }
            }
            Instr::CreateCell2D(rows, cols) => {
                let mut elems = Vec::with_capacity(rows * cols);
                for _ in 0..rows * cols {
                    elems.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                elems.reverse();
                let cell = cell_ops::create_cell_2d(elems, rows, cols)?;
                stack.push(cell);
            }
            Instr::IndexCell(num_indices) => {
                // Pop indices first (in reverse), then base
                let mut indices = Vec::with_capacity(num_indices);
                for _ in 0..num_indices {
                    let v: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    indices.push(v as usize);
                }
                indices.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Object(obj) => {
                        // Route to subsref(obj, '{}', {indices})
                        let cell = call_builtin_vm!(
                            "__make_cell",
                            &indices
                                .iter()
                                .map(|n| Value::Num(*n as f64))
                                .collect::<Vec<_>>(),
                        )?;
                        match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("{}".to_string()),
                                cell,
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    Value::HandleObject(handle) => {
                        // Route to subsref(obj, '{}', {indices})
                        let cell = call_builtin_vm!(
                            "__make_cell",
                            &indices
                                .iter()
                                .map(|n| Value::Num(*n as f64))
                                .collect::<Vec<_>>(),
                        )?;
                        match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::HandleObject(handle),
                                Value::String("subsref".to_string()),
                                Value::String("{}".to_string()),
                                cell,
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e),
                        }
                    }
                    Value::Cell(ca) => stack.push(cell_ops::index_cell_value(&ca, &indices)?),
                    _ => return Err(mex("CellIndexingOnNonCell", "Cell indexing on non-cell")),
                }
            }
            Instr::IndexCellExpand(num_indices, out_count) => {
                // Same as IndexCell but flatten cell contents into multiple outputs
                let mut indices = Vec::with_capacity(num_indices);
                if num_indices > 0 {
                    for _ in 0..num_indices {
                        let v: f64 = (&stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?)
                            .try_into()?;
                        indices.push(v as usize);
                    }
                    indices.reverse();
                }
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Cell(ca) => {
                        let values = cell_ops::expand_cell_values(&ca, &indices, out_count)?;
                        for v in values {
                            stack.push(v);
                        }
                    }
                    Value::Object(obj) => {
                        // Defer to subsref; expect a cell back; then expand one element
                        let cell = call_builtin_vm!(
                            "__make_cell",
                            &indices
                                .iter()
                                .map(|n| Value::Num(*n as f64))
                                .collect::<Vec<_>>(),
                        )?;
                        let v = match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("{}".to_string()),
                                cell,
                            ],
                        ) {
                            Ok(v) => v,
                            Err(e) => vm_bail!(e.to_string()),
                        };
                        // Push returned value and pad to out_count
                        stack.push(v);
                        for _ in 1..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    Value::HandleObject(handle) => {
                        // Defer to subsref; expect a cell back; then expand one element
                        let cell = call_builtin_vm!(
                            "__make_cell",
                            &indices
                                .iter()
                                .map(|n| Value::Num(*n as f64))
                                .collect::<Vec<_>>(),
                        )?;
                        let v = match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::HandleObject(handle),
                                Value::String("subsref".to_string()),
                                Value::String("{}".to_string()),
                                cell,
                            ],
                        ) {
                            Ok(v) => v,
                            Err(e) => vm_bail!(e.to_string()),
                        };
                        // Push returned value and pad to out_count
                        stack.push(v);
                        for _ in 1..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    _ => return Err(mex("CellExpansionOnNonCell", "Cell expansion on non-cell")),
                }
            }
            Instr::Pop => {
                stack_ops::pop(&mut stack);
            }
            Instr::Unpack(out_count) => {
                if out_count == 0 {
                    pc += 1;
                    continue;
                }
                array_ops::unpack(&mut stack, out_count)?;
            }
            Instr::ReturnValue => {
                let action = control_flow::return_value(&mut stack)?;
                interpreter_timing.flush_host_span("return_value", None);
                if matches!(action, ControlFlowAction::Return) {
                    break;
                }
            }
            Instr::Return => {
                let action = control_flow::return_void();
                interpreter_timing.flush_host_span("return", None);
                if matches!(action, ControlFlowAction::Return) {
                    break;
                }
            }
            Instr::StoreIndex(num_indices) => {
                // RHS to assign, then indices, then base
                // Debug snapshot of top-of-stack types before mutation
                #[allow(unused)]
                if std::env::var("RUNMAT_DEBUG_INDEX").as_deref() == Ok("1") {
                    let snap = stack
                        .iter()
                        .rev()
                        .take(6)
                        .map(|v| match v {
                            Value::Object(_) => "Object",
                            Value::HandleObject(_) => "HandleObject",
                            Value::Tensor(t) => {
                                debug!(shape = ?t.shape, "[vm] StoreIndex pre-snap tensor");
                                "Tensor"
                            }
                            Value::GpuTensor(h) => {
                                debug!(shape = ?h.shape, "[vm] StoreIndex pre-snap GPU tensor");
                                "GpuTensor"
                            }
                            Value::Num(_) => "Num",
                            Value::Int(_) => "Int",
                            Value::String(_) => "String",
                            Value::Cell(_) => "Cell",
                            _ => "Other",
                        })
                        .collect::<Vec<_>>();
                    debug!(pc, stack_top_types = ?snap, "[vm] StoreIndex pre-snap");
                }
                let rhs = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                // We will determine indices relative to the base location to avoid RHS temporaries interfering
                // Select the correct base: scan from top for the first assignable container (Object/Tensor/GpuTensor)
                let assignable = |v: &Value| {
                    matches!(
                        v,
                        Value::Object(_)
                            | Value::HandleObject(_)
                            | Value::Tensor(_)
                            | Value::ComplexTensor(_)
                            | Value::GpuTensor(_)
                    )
                };
                let base_idx_opt = (0..stack.len()).rev().find(|&j| assignable(&stack[j]));
                let base_pos = if let Some(j) = base_idx_opt {
                    j
                } else {
                    return Err(mex(
                        "IndexAssignmentUnsupportedBase",
                        "Index assignment only for tensors",
                    ));
                };
                let base = stack.remove(base_pos);
                #[cfg(feature = "native-accel")]
                clear_residency(&base);
                // Deterministically extract indices: take exactly `num_indices` numeric values
                // that were immediately above the base position.
                let mut indices: Vec<usize> = Vec::new();
                if num_indices > 0 {
                    let mut contiguous_ok = true;
                    if base_pos + num_indices > stack.len() {
                        contiguous_ok = false;
                    } else {
                        for k in 0..num_indices {
                            let idx_pos = base_pos + k;
                            let idx_val = match index_scalar_from_value(&stack[idx_pos]).await {
                                Ok(Some(val)) => val,
                                Ok(None) => {
                                    contiguous_ok = false;
                                    indices.clear();
                                    break;
                                }
                                Err(err) => vm_bail!(err),
                            };
                            let idx_val = if idx_val <= 0 { 0 } else { idx_val as usize };
                            indices.push(idx_val);
                        }
                    }
                    if contiguous_ok {
                        // Remove the consumed index values from the stack (highest index first)
                        for k in (0..num_indices).rev() {
                            stack.remove(base_pos + k);
                        }
                    } else {
                        indices.clear();
                    }
                }
                // Determine expected bounds for fast validation
                let (rows_opt, cols_opt) = match &base {
                    Value::Tensor(t) => (Some(t.rows()), Some(t.cols())),
                    Value::GpuTensor(h) => (
                        Some(h.shape.first().copied().unwrap_or(1).max(1)),
                        Some(h.shape.get(1).copied().unwrap_or(1).max(1)),
                    ),
                    _ => (None, None),
                };
                // If deterministic path failed (unexpected stack form), fall back to nearest-fit heuristic
                if indices.is_empty() {
                    let mut numeric_above: Vec<(usize, usize)> = Vec::new(); // (stack_index, value)
                    let mut scan_limit = 12usize;
                    let mut kk = stack.len();
                    while kk > 0 && scan_limit > 0 {
                        let idx = kk - 1;
                        if assignable(&stack[idx]) {
                            break;
                        }
                        if let Some(v) = index_scalar_from_value(&stack[idx]).await? {
                            let v = if v <= 0 { 0 } else { v as usize };
                            numeric_above.push((idx, v));
                        }
                        kk -= 1;
                        scan_limit -= 1;
                    }
                    if numeric_above.len() >= 2 {
                        let mut picked: Option<((usize, usize), (usize, usize))> = None;
                        for w in (1..numeric_above.len()).rev() {
                            let (j_idx, j_val) = numeric_above[w];
                            let (i_idx, i_val) = numeric_above[w - 1];
                            let fits = match (rows_opt, cols_opt) {
                                (Some(r), Some(c)) => {
                                    i_val >= 1 && i_val <= r && j_val >= 1 && j_val <= c
                                }
                                _ => true,
                            };
                            if fits {
                                picked = Some(((i_idx, i_val), (j_idx, j_val)));
                                break;
                            }
                        }
                        if let Some(((i_idx, i_val), (j_idx, j_val))) = picked {
                            let mut to_remove = [i_idx, j_idx];
                            to_remove.sort_unstable();
                            stack.remove(to_remove[1]);
                            stack.remove(to_remove[0]);
                            indices = vec![i_val, j_val];
                        }
                    } else if numeric_above.len() == 1 {
                        let (k_idx, k_val) = numeric_above[0];
                        stack.remove(k_idx);
                        indices = vec![k_val];
                    }
                }
                if indices.is_empty() {
                    return Err(mex(
                        "IndexAssignmentUnsupportedBase",
                        "Index assignment only for tensors",
                    ));
                }
                // TODO(GC): write barrier hook if base is in older generation and rhs/indices reference younger objects
                match base {
                    Value::Object(obj) => {
                        // subsasgn(obj, '()', {indices...}, rhs)
                        let cell = call_builtin_vm!(
                            "__make_cell",
                            &indices
                                .iter()
                                .map(|n| Value::Num(*n as f64))
                                .collect::<Vec<_>>(),
                        )?;
                        match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsasgn".to_string()),
                                Value::String("()".to_string()),
                                cell,
                                rhs,
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e.to_string()),
                        }
                    }
                    Value::HandleObject(handle) => {
                        // subsasgn(obj, '()', {indices...}, rhs)
                        let cell = call_builtin_vm!(
                            "__make_cell",
                            &indices
                                .iter()
                                .map(|n| Value::Num(*n as f64))
                                .collect::<Vec<_>>(),
                        )?;
                        match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::HandleObject(handle),
                                Value::String("subsasgn".to_string()),
                                Value::String("()".to_string()),
                                cell,
                                rhs,
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e.to_string()),
                        }
                    }
                    Value::Tensor(t) => stack.push(idx_write_linear::assign_tensor_scalar(t, &indices, &rhs).await?),
                    Value::ComplexTensor(t) => stack.push(idx_write_linear::assign_complex_scalar(t, &indices, &rhs).await?),
                    Value::GpuTensor(h) => stack.push(idx_write_linear::assign_gpu_scalar(&h, &indices, &rhs).await?),
                    _ => {
                        if std::env::var("RUNMAT_DEBUG_INDEX").as_deref() == Ok("1") {
                            let kind = |v: &Value| match v {
                                Value::Object(_) => "Object",
                                Value::Tensor(_) => "Tensor",
                                Value::GpuTensor(_) => "GpuTensor",
                                Value::Num(_) => "Num",
                                Value::Int(_) => "Int",
                                _ => "Other",
                            };
                            debug!(
                                pc,
                                base_kind = kind(&base),
                                rhs_kind = kind(&rhs),
                                ?indices,
                                "[vm] StoreIndex default branch"
                            );
                        }
                        return Err(mex(
                            "IndexAssignmentUnsupportedBase",
                            "Index assignment only for tensors",
                        ));
                    }
                }
            }
            Instr::StoreIndexCell(num_indices) => {
                // RHS, then indices, then base cell
                let rhs = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let mut indices = Vec::new();
                for _ in 0..num_indices {
                    let v: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    indices.push(v as usize);
                }
                indices.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                #[cfg(feature = "native-accel")]
                clear_residency(&base);
                // TODO(GC): write barrier hook for cell element updates
                match base {
                    Value::Object(obj) => {
                        // subsasgn(obj, '{}', {indices}, rhs)
                        let cell = runmat_builtins::CellArray::new(
                            indices.iter().map(|n| Value::Num(*n as f64)).collect(),
                            1,
                            indices.len(),
                        )
                        .map_err(|e| format!("subsasgn build error: {e}"))?;
                        match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsasgn".to_string()),
                                Value::String("{}".to_string()),
                                Value::Cell(cell),
                                rhs,
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e.to_string()),
                        }
                    }
                    Value::HandleObject(handle) => {
                        // subsasgn(obj, '{}', {indices}, rhs)
                        let cell = runmat_builtins::CellArray::new(
                            indices.iter().map(|n| Value::Num(*n as f64)).collect(),
                            1,
                            indices.len(),
                        )
                        .map_err(|e| format!("subsasgn build error: {e}"))?;
                        match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::HandleObject(handle),
                                Value::String("subsasgn".to_string()),
                                Value::String("{}".to_string()),
                                Value::Cell(cell),
                                rhs,
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e.to_string()),
                        }
                    }
                    Value::Cell(ca) => {
                        let updated = cell_ops::assign_cell_value(ca, &indices, rhs, |oldv, newv| {
                            runmat_gc::gc_record_write(oldv, newv);
                        })?;
                        stack.push(updated);
                    }
                    _ => {
                        return Err(mex(
                            "CellAssignmentOnNonCell",
                            "Cell assignment on non-cell",
                        ))
                    }
                }
            }
            Instr::LoadMember(field) | Instr::LoadMemberOrInit(field) => {
                let allow_init = matches!(bytecode.instructions[pc], Instr::LoadMemberOrInit(_));
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match obj_resolve::load_member(base, field, allow_init).await {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::LoadMemberDynamic | Instr::LoadMemberDynamicOrInit => {
                let allow_init =
                    matches!(bytecode.instructions[pc], Instr::LoadMemberDynamicOrInit);
                let name_val = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let name: String = (&name_val).try_into()?;
                match obj_resolve::load_member_dynamic(base, name, allow_init).await {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::StoreMember(field) | Instr::StoreMemberOrInit(field) => {
                let allow_init = matches!(bytecode.instructions[pc], Instr::StoreMemberOrInit(_));
                let rhs = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match obj_resolve::store_member(base, field, rhs, allow_init, |oldv, newv| {
                    runmat_gc::gc_record_write(oldv, newv);
                })
                .await
                {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::StoreMemberDynamic | Instr::StoreMemberDynamicOrInit => {
                let allow_init =
                    matches!(bytecode.instructions[pc], Instr::StoreMemberDynamicOrInit);
                let rhs = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let name_val = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let name: String = (&name_val).try_into()?;
                match obj_resolve::store_member_dynamic(base, name, rhs, allow_init, |oldv, newv| {
                    runmat_gc::gc_record_write(oldv, newv);
                })
                .await
                {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::CallMethod(name, arg_count) => {
                let (base, args) = call_closures::collect_method_args(&mut stack, arg_count)?;
                match call_closures::call_method(base, &name, args).await {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::CallMethodOrMemberIndex(name, arg_count) => {
                let (base, args) = call_closures::collect_method_args(&mut stack, arg_count)?;
                match call_closures::call_method_or_member_index(base, name, args).await {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::LoadMethod(name) => {
                let base = pop_value(&mut stack)?;
                match call_closures::load_method_closure(base, name) {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::CreateClosure(func_name, capture_count) => {
                call_closures::create_closure(&mut stack, func_name, capture_count)?;
            }
            Instr::LoadStaticProperty(class_name, prop) => {
                match obj_resolve::load_static_member(&class_name, &prop) {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::CallStaticMethod(class_name, method, arg_count) => {
                let mut args = call_builtins::collect_call_args(&mut stack, arg_count)?;
                match call_closures::call_static_method(&class_name, &method, args.clone()).await {
                    Ok(v) => stack.push(v),
                    Err(_) => {
                        let is_type_class = matches!(
                            class_name.as_str(),
                            "gpuArray"
                                | "logical"
                                | "double"
                                | "single"
                                | "int8"
                                | "int16"
                                | "int32"
                                | "int64"
                                | "uint8"
                                | "uint16"
                                | "uint32"
                                | "uint64"
                                | "char"
                                | "string"
                                | "cell"
                                | "struct"
                        );
                        if is_type_class {
                            args.push(Value::from(class_name.as_str()));
                            let v = match call_builtin_vm!(&method, &args) {
                                Ok(v) => v,
                                Err(e) => vm_bail!(e),
                            };
                            stack.push(v);
                        } else {
                            vm_bail!(format!(
                                "Unknown static method '{}' on class {}",
                                method, class_name
                            ));
                        }
                    }
                }
            }
            Instr::RegisterClass {
                name,
                super_class,
                properties,
                methods,
            } => {
                obj_class_def::register_class(name, super_class, properties, methods)?;
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
    let steps_u32 = parse_steps_value(&steps).await?;
    if steps_u32 == 0 {
        return Ok(state);
    }

    #[cfg(feature = "native-accel")]
    {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let (state_handle, state_owned) =
                ensure_gpu_tensor_for_stochastic(provider, &state).await?;
            let drift_scalar =
                scalar_from_value_scalar(&drift, "stochastic_evolution drift").await?;
            let scale_scalar =
                scalar_from_value_scalar(&scale, "stochastic_evolution scale").await?;
            match provider.stochastic_evolution(
                &state_handle,
                drift_scalar,
                scale_scalar,
                steps_u32,
            ) {
                Ok(output) => {
                    if let Some(temp) = state_owned {
                        let _ = provider.free(&temp);
                    }
                    fusion_residency::mark(&output);
                    return Ok(Value::GpuTensor(output));
                }
                Err(err) => {
                    log::debug!("stochastic_evolution provider fallback to host: {}", err);
                    if let Some(temp) = state_owned {
                        let _ = provider.free(&temp);
                    }
                }
            }
        }
    }

    let gathered_state = gather_if_needed_async(&state)
        .await
        .map_err(|e| format!("stochastic_evolution: {e}"))?;
    let mut tensor_value = match gathered_state {
        Value::Tensor(t) => t,
        other => tensor::value_into_tensor_for("stochastic_evolution", other)?,
    };
    let drift_scalar = scalar_from_value_scalar(&drift, "stochastic_evolution drift").await?;
    let scale_scalar = scalar_from_value_scalar(&scale, "stochastic_evolution scale").await?;
    stochastic_evolution_host(&mut tensor_value, drift_scalar, scale_scalar, steps_u32)?;
    Ok(Value::Tensor(tensor_value))
}

async fn scalar_from_value_scalar(value: &Value, label: &str) -> VmResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        Value::Tensor(t) => Err(format!(
            "{label}: expected scalar tensor, got {} elements",
            t.data.len()
        )
        .into()),
        Value::GpuTensor(_) => {
            let gathered = gather_if_needed_async(value)
                .await
                .map_err(|e| format!("{label}: {e}"))?;
            match gathered {
                Value::Num(n) => Ok(n),
                Value::Int(i) => Ok(i.to_f64()),
                Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
                Value::Tensor(t) => Err(format!(
                    "{label}: expected scalar tensor, got {} elements",
                    t.data.len()
                )
                .into()),
                other => Err(format!("{label}: expected numeric scalar, got {:?}", other).into()),
            }
        }
        other => Err(format!("{label}: expected numeric scalar, got {:?}", other).into()),
    }
}

async fn logical_truth_from_value(value: &Value, label: &str) -> VmResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Int(i) => Ok(!i.is_zero()),
        Value::Num(n) => Ok(*n != 0.0),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        Value::LogicalArray(array) => Err(mex(
            "InvalidConditionType",
            &format!(
                "{label}: expected scalar logical or numeric value, got logical array with {} elements",
                array.data.len()
            ),
        )),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0] != 0.0),
        Value::Tensor(tensor) => Err(mex(
            "InvalidConditionType",
            &format!(
                "{label}: expected scalar logical or numeric value, got numeric array with {} elements",
                tensor.data.len()
            ),
        )),
        Value::GpuTensor(_) => {
            let gathered = gather_if_needed_async(value)
                .await
                .map_err(|e| format!("{label}: {e}"))?;
            Box::pin(logical_truth_from_value(&gathered, label)).await
        }
        other => Err(mex(
            "InvalidConditionType",
            &format!(
                "{label}: expected scalar logical or numeric value, got {other:?}"
            ),
        )),
    }
}

async fn parse_steps_value(value: &Value) -> VmResult<u32> {
    let raw = scalar_from_value_scalar(value, "stochastic_evolution steps").await?;
    if !raw.is_finite() || raw < 0.0 {
        return Err(mex(
            "InvalidSteps",
            "stochastic_evolution: steps must be a non-negative scalar",
        ));
    }
    Ok(raw.round() as u32)
}

#[cfg(feature = "native-accel")]
async fn ensure_gpu_tensor_for_stochastic(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    value: &Value,
) -> VmResult<(
    runmat_accelerate_api::GpuTensorHandle,
    Option<runmat_accelerate_api::GpuTensorHandle>,
)> {
    match value {
        Value::GpuTensor(handle) => Ok((handle.clone(), None)),
        Value::Tensor(tensor) => {
            let handle = upload_tensor_view(provider, tensor)?;
            Ok((handle.clone(), Some(handle)))
        }
        _ => {
            let gathered = gather_if_needed_async(value)
                .await
                .map_err(|e| format!("stochastic_evolution: {e}"))?;
            match gathered {
                Value::Tensor(t) => {
                    let handle = upload_tensor_view(provider, &t)?;
                    Ok((handle.clone(), Some(handle)))
                }
                other => {
                    let tensor = tensor::value_into_tensor_for("stochastic_evolution", other)?;
                    let handle = upload_tensor_view(provider, &tensor)?;
                    Ok((handle.clone(), Some(handle)))
                }
            }
        }
    }
}

#[cfg(feature = "native-accel")]
fn upload_tensor_view(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    tensor: &runmat_builtins::Tensor,
) -> VmResult<runmat_accelerate_api::GpuTensorHandle> {
    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    provider
        .upload(&view)
        .map_err(|e| mex("UploadFailed", &e.to_string()))
}

#[cfg(feature = "native-accel")]
#[inline]
fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Int(_) => "Int",
        Value::Num(_) => "Num",
        Value::Complex(_, _) => "Complex",
        Value::Bool(_) => "Bool",
        Value::LogicalArray(_) => "LogicalArray",
        Value::String(_) => "String",
        Value::StringArray(_) => "StringArray",
        Value::CharArray(_) => "CharArray",
        Value::Tensor(_) => "Tensor",
        Value::ComplexTensor(_) => "ComplexTensor",
        Value::Cell(_) => "Cell",
        Value::Struct(_) => "Struct",
        Value::GpuTensor(_) => "GpuTensor",
        Value::Object(_) => "Object",
        Value::HandleObject(_) => "HandleObject",
        Value::Listener(_) => "Listener",
        Value::FunctionHandle(_) => "FunctionHandle",
        Value::Closure(_) => "Closure",
        Value::ClassRef(_) => "ClassRef",
        Value::MException(_) => "MException",
        Value::OutputList(_) => "OutputList",
    }
}
#[cfg(feature = "native-accel")]
#[inline]
fn summarize_value(i: usize, v: &Value) -> String {
    match v {
        Value::GpuTensor(h) => format!("in#{i}:GpuTensor shape={:?}", h.shape),
        Value::Tensor(t) => format!("in#{i}:Tensor shape={:?}", t.shape),
        Value::Num(n) => format!("in#{i}:Num({n:.6})"),
        Value::Int(n) => format!("in#{i}:Int({})", n.to_i64()),
        Value::Bool(b) => format!("in#{i}:Bool({})", if *b { 1 } else { 0 }),
        Value::String(s) => format!("in#{i}:String({})", s),
        _ => format!("in#{i}:{}", value_kind(v)),
    }
}

#[cfg(feature = "native-accel")]
fn fusion_span_live_result_count(instructions: &[Instr], span: &InstrSpan) -> Option<usize> {
    if span.start > span.end || span.end >= instructions.len() {
        return None;
    }

    let mut current_depth = 0usize;
    for instr in &instructions[span.start..=span.end] {
        let effect = instr.stack_effect()?;
        if current_depth < effect.pops {
            current_depth = effect.pops;
        }
        current_depth = current_depth - effect.pops + effect.pushes;
    }
    Some(current_depth)
}

#[cfg(feature = "native-accel")]
fn fusion_span_has_vm_barrier(instructions: &[Instr], span: &InstrSpan) -> bool {
    if span.start > span.end || span.end >= instructions.len() {
        return true;
    }

    for instr in &instructions[span.start..=span.end] {
        if matches!(
            instr,
            Instr::StoreIndex(_)
                | Instr::StoreSlice(_, _, _, _)
                | Instr::StoreSliceExpr { .. }
                | Instr::StoreIndexCell(_)
                | Instr::StoreMember(_)
                | Instr::StoreMemberOrInit(_)
                | Instr::StoreMemberDynamic
                | Instr::StoreMemberDynamicOrInit
        ) {
            return true;
        }
    }

    if fusion_span_live_result_count(instructions, span) != Some(1) {
        return true;
    }

    false
}

#[cfg(feature = "native-accel")]
struct StackSliceGuard<'a> {
    stack: *mut Vec<Value>,
    slice: Option<Vec<Value>>,
    _marker: std::marker::PhantomData<&'a mut Vec<Value>>,
}

#[cfg(feature = "native-accel")]
impl<'a> StackSliceGuard<'a> {
    fn new(stack: &'a mut Vec<Value>, slice_start: usize) -> Self {
        let slice = stack.split_off(slice_start);
        Self {
            stack,
            slice: Some(slice),
            _marker: std::marker::PhantomData,
        }
    }

    fn slice(&self) -> &[Value] {
        self.slice.as_ref().expect("stack slice missing").as_slice()
    }

    fn commit(mut self) {
        self.slice = None;
    }
}

#[cfg(feature = "native-accel")]
impl Drop for StackSliceGuard<'_> {
    fn drop(&mut self) {
        if let Some(slice) = self.slice.take() {
            unsafe {
                (&mut *self.stack).extend(slice);
            }
        }
    }
}

#[cfg(feature = "native-accel")]
async fn try_execute_fusion_group(
    plan: &runmat_accelerate::FusionGroupPlan,
    graph: &runmat_accelerate::AccelGraph,
    stack: &mut Vec<Value>,
    vars: &mut Vec<Value>,
    context: &mut ExecutionContext,
) -> VmResult<Value> {
    if plan.group.stack_layout.is_none() && !plan.stack_pattern.is_empty() {
        return Err(mex(
            "FusionMissingStackLayout",
            "fusion: missing compile-time stack layout metadata",
        ));
    }
    let required_stack_operands = plan
        .group
        .stack_layout
        .as_ref()
        .map(|layout| layout.required_stack_operands)
        .unwrap_or_else(|| plan.stack_pattern.len());
    let mut inputs: Vec<Option<Value>> = vec![None; plan.inputs.len()];

    for (idx, value) in &plan.constants {
        if let Some(slot) = inputs.get_mut(*idx) {
            if slot.is_none() {
                *slot = Some(value.clone());
            }
        }
    }

    for (idx, value_id) in plan.inputs.iter().enumerate() {
        let info = graph
            .value(*value_id)
            .ok_or_else(|| format!("fusion: missing value metadata for id {value_id}"))?;
        match &info.origin {
            ValueOrigin::Variable { kind, index } => {
                let value =
                    match kind {
                        VarKind::Global => vars
                            .get(*index)
                            .cloned()
                            .ok_or_else(|| format!("fusion: global var {index} out of range"))?,
                        VarKind::Local => {
                            if let Some(frame) = context.call_stack.last() {
                                let absolute = frame.locals_start + index;
                                context.locals.get(absolute).cloned().ok_or_else(|| {
                                    format!("fusion: local var {index} unavailable")
                                })?
                            } else {
                                vars.get(*index).cloned().ok_or_else(|| {
                                    format!("fusion: local var {index} unavailable")
                                })?
                            }
                        }
                    };
                debug_assert!(
                    inputs[idx].is_none(),
                    "fusion: duplicate input slot {} for plan {}",
                    idx,
                    plan.index
                );
                inputs[idx] = Some(value);
            }
            ValueOrigin::Constant | ValueOrigin::NodeOutput { .. } | ValueOrigin::Unknown => {}
        }
    }

    if log::log_enabled!(log::Level::Debug) && fusion_debug_enabled() {
        let stack_needed_preview = required_stack_operands;
        let stack_snapshot: Vec<&Value> = stack.iter().rev().take(stack_needed_preview).collect();
        let stack_kinds: Vec<&'static str> =
            stack_snapshot.iter().rev().map(|v| value_kind(v)).collect();
        let input_meta: Vec<String> = plan
            .inputs
            .iter()
            .enumerate()
            .map(|(i, value_id)| {
                if let Some(info) = graph.value(*value_id) {
                    format!("#{i}:id={} origin={:?}", value_id, info.origin)
                } else {
                    format!("#{i}:id={} origin=<missing>", value_id)
                }
            })
            .collect();
        log::debug!(
            "fusion group {} gather: stack_depth={} stack_needed={} stack_kinds={:?} pattern={:?} inputs={:?}",
            plan.index,
            stack.len(),
            stack_needed_preview,
            stack_kinds,
            &plan.stack_pattern,
            input_meta
        );
    }

    if stack.len() < required_stack_operands {
        if fusion_debug_enabled() {
            log::debug!(
                "fusion stack underflow: plan={} needed={} available={} pattern={:?}",
                plan.index,
                required_stack_operands,
                stack.len(),
                plan.stack_pattern
            );
        }
        return Err(mex(
            "FusionStackUnderflow",
            "fusion: stack underflow gathering inputs",
        ));
    }
    let available = required_stack_operands;
    let slice_start = stack.len() - available;
    let stack_guard = StackSliceGuard::new(stack, slice_start);
    let slice = stack_guard.slice().to_vec();
    let mut consumed_inputs: Vec<Option<Value>> = vec![None; plan.inputs.len()];
    let input_positions: HashMap<runmat_accelerate::graph::ValueId, usize> = plan
        .inputs
        .iter()
        .enumerate()
        .map(|(idx, value_id)| (*value_id, idx))
        .collect();

    let allow_stack_value = |val: &Value| {
        if plan.group.kind.is_reduction() {
            matches!(val, Value::GpuTensor(_) | Value::Tensor(_))
        } else {
            true
        }
    };

    if let Some(layout) = plan.group.stack_layout.as_ref() {
        for binding in &layout.bindings {
            let Some(input_idx) = input_positions.get(&binding.value_id).copied() else {
                continue;
            };
            let Some(val) = slice.get(binding.stack_offset).cloned() else {
                continue;
            };
            consumed_inputs[input_idx] = Some(val.clone());
            if inputs[input_idx].is_none() && allow_stack_value(&val) {
                inputs[input_idx] = Some(val);
            }
        }
    } else {
        for (offset, input_idx) in plan.stack_pattern.iter().enumerate() {
            let Some(val) = slice.get(offset).cloned() else {
                continue;
            };
            consumed_inputs[*input_idx] = Some(val.clone());
            if inputs[*input_idx].is_none() && allow_stack_value(&val) {
                inputs[*input_idx] = Some(val);
            }
        }
    }

    for (idx, slot) in inputs.iter_mut().enumerate() {
        if slot.is_some() {
            continue;
        }
        let vid = plan.inputs[idx];
        let info = graph.value(vid);
        if let Some(info) = info {
            match &info.origin {
                ValueOrigin::Variable { kind, index } => {
                    let value_opt = match kind {
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
                    if let Some(value) = value_opt {
                        *slot = Some(value);
                        continue;
                    }
                }
                ValueOrigin::Constant => {
                    if let Some(value) = plan.const_values.get(&vid) {
                        *slot = Some(value.clone());
                        continue;
                    }
                }
                _ => {}
            }
        }
        if slot.is_none() {
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
                    *slot = Some(value);
                    continue;
                }
            }
        }
        if slot.is_none() {
            if let Some(info) = info {
                if let ValueOrigin::NodeOutput { node, .. } = info.origin {
                    if let Some(binding) = graph.node_binding(node) {
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
                            *slot = Some(value);
                            continue;
                        }
                    }
                }
            }
        }
        if slot.is_none() {
            if let Some(value) = plan.const_values.get(&vid) {
                *slot = Some(value.clone());
            }
        }
    }

    let inputs: Vec<Value> = inputs
        .into_iter()
        .map(|opt| opt.ok_or_else(|| mex("FusionMissingInput", "fusion: missing input value")))
        .collect::<Result<_, _>>()?;

    // Debug: summarize runtime input kinds/shapes
    if log::log_enabled!(log::Level::Debug) {
        let summaries: Vec<String> = inputs
            .iter()
            .enumerate()
            .map(|(i, v)| summarize_value(i, v))
            .collect();
        log::debug!("fusion inputs runtime: [{}]", summaries.join(", "));
    }

    let request = FusionExecutionRequest { plan, inputs };
    log::debug!(
        "dispatch fusion kind {:?}, supported {}",
        plan.group.kind,
        plan.kernel.supported
    );
    if plan.group.kind.is_elementwise() {
        match execute_elementwise(request) {
            Ok(result) => {
                for (store, value) in result.materialized_stores {
                    match store.binding.kind {
                        VarKind::Global => {
                            let i = store.binding.index;
                            #[cfg(feature = "native-accel")]
                            if i < vars.len() && !same_gpu_handle(&vars[i], &value) {
                                clear_residency(&vars[i]);
                            }
                            if i >= vars.len() {
                                vars.resize(i + 1, Value::Num(0.0));
                                refresh_workspace_state(vars);
                            }
                            vars[i] = value;
                        }
                        VarKind::Local => {
                            if let Some(frame) = context.call_stack.last() {
                                let absolute = frame.locals_start + store.binding.index;
                                while context.locals.len() <= absolute {
                                    context.locals.push(Value::Num(0.0));
                                }
                                #[cfg(feature = "native-accel")]
                                if !same_gpu_handle(&context.locals[absolute], &value) {
                                    clear_residency(&context.locals[absolute]);
                                }
                                context.locals[absolute] = value;
                            } else {
                                let i = store.binding.index;
                                #[cfg(feature = "native-accel")]
                                if i < vars.len() && !same_gpu_handle(&vars[i], &value) {
                                    clear_residency(&vars[i]);
                                }
                                if i >= vars.len() {
                                    vars.resize(i + 1, Value::Num(0.0));
                                    refresh_workspace_state(vars);
                                }
                                vars[i] = value;
                            }
                        }
                    }
                }
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
        if reduce_all && fusion_debug_enabled() {
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
                    if fusion_debug_enabled() {
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
                if fusion_debug_enabled() {
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
                if fusion_debug_enabled() {
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
        if fusion_debug_enabled() {
            log::debug!(
                "fusion reduction: axis={} reduce_len={} num_slices={} constants={:?}",
                axis,
                reduce_len,
                num_slices,
                plan.constants
            );
        }
        if log::log_enabled!(log::Level::Debug) && fusion_debug_enabled() {
            let _rt_inputs: Vec<String> = request
                .inputs
                .iter()
                .enumerate()
                .map(|(i, v)| summarize_value(i, v))
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
        if log::log_enabled!(log::Level::Debug) && fusion_debug_enabled() {
            let _rt_inputs: Vec<String> = request
                .inputs
                .iter()
                .enumerate()
                .map(|(i, v)| summarize_value(i, v))
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
    if let Value::GpuTensor(handle) = value {
        fusion_residency::clear(handle);
    }
}

#[cfg(feature = "native-accel")]
fn same_gpu_handle(lhs: &Value, rhs: &Value) -> bool {
    matches!(
        (lhs, rhs),
        (Value::GpuTensor(left), Value::GpuTensor(right)) if left.buffer_id == right.buffer_id
    )
}

fn parse_exception(err: &runmat_runtime::RuntimeError) -> runmat_builtins::MException {
    if let Some(identifier) = err.identifier() {
        return runmat_builtins::MException::new(identifier.to_string(), err.message().to_string());
    }
    let message = err.message();
    // Prefer the last occurrence of ": " to split IDENT: message, preserving nested identifiers
    if let Some(idx) = message.rfind(": ") {
        let (id, msg) = message.split_at(idx);
        let message = msg.trim_start_matches(':').trim().to_string();
        let ident = if id.trim().is_empty() {
            format!("{}:error", error_namespace())
        } else {
            id.trim().to_string()
        };
        return runmat_builtins::MException::new(ident, message);
    }
    // Fallback: if any ':' present, use the last as separator
    if let Some(idx) = message.rfind(':') {
        let (id, msg) = message.split_at(idx);
        let message = msg.trim_start_matches(':').trim().to_string();
        let ident = if id.trim().is_empty() {
            format!("{}:error", error_namespace())
        } else {
            id.trim().to_string()
        };
        runmat_builtins::MException::new(ident, message)
    } else {
        runmat_builtins::MException::new(
            format!("{}:error", error_namespace()),
            message.to_string(),
        )
    }
}

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

        assert_eq!(fusion_span_live_result_count(&instructions, &span), Some(1));
        assert!(!fusion_span_has_vm_barrier(&instructions, &span));
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

        assert_eq!(fusion_span_live_result_count(&instructions, &span), Some(2));
        assert!(fusion_span_has_vm_barrier(&instructions, &span));
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

        assert!(!fusion_span_has_vm_barrier(&instructions, &span));
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

        assert!(!fusion_span_has_vm_barrier(&instructions, &span));
    }
}
