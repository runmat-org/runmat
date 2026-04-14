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
use runmat_vm::interpreter::stack::{pop2, pop_value};
use runmat_vm::call::shared as call_shared;
use runmat_vm::call::{builtins as call_builtins, feval as call_feval, user as call_user};
use runmat_vm::call::builtins::ImportedBuiltinResolution;
use runmat_vm::call::closures as call_closures;
use runmat_vm::call::feval::FevalDispatch;
use runmat_vm::indexing::end_expr as idx_end_expr;
use runmat_vm::indexing::read_linear as idx_read_linear;
use runmat_vm::indexing::read_slice as idx_read_slice;
use runmat_vm::indexing::selectors as idx_selectors;
use runmat_vm::ops::control_flow::{self, ControlFlowAction};
use runmat_vm::ops::stack as stack_ops;
use runmat_vm::interpreter::timing::InterpreterTiming;
use runmat_vm::runtime::call_stack::{
    attach_call_frames, error_namespace, push_call_frame,
};
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

fn cartesian_product<F: FnMut(&[usize])>(lists: &[Vec<usize>], mut f: F) {
    let dims = lists.len();
    if dims == 0 {
        return;
    }
    let mut idx = vec![0usize; dims];
    loop {
        let current: Vec<usize> = (0..dims).map(|d| lists[d][idx[d]]).collect();
        f(&current);
        let mut d = 0usize;
        while d < dims {
            idx[d] += 1;
            if idx[d] < lists[d].len() {
                break;
            }
            idx[d] = 0;
            d += 1;
        }
        if d == dims {
            break;
        }
    }
}

fn cartesian_positions<F: FnMut(&[usize])>(lengths: &[usize], mut f: F) {
    if lengths.is_empty() || lengths.contains(&0) {
        return;
    }
    let dims = lengths.len();
    let mut idx = vec![0usize; dims];
    loop {
        f(&idx);
        let mut d = 0usize;
        while d < dims {
            idx[d] += 1;
            if idx[d] < lengths[d] {
                break;
            }
            idx[d] = 0;
            d += 1;
        }
        if d == dims {
            break;
        }
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

async fn indices_from_value_linear(value: &Value, total_len: usize) -> VmResult<Vec<usize>> {
    idx_selectors::indices_from_value_linear(value, total_len).await
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

/// Output shape after indexing:
/// - preserve non-trailing singleton dimensions
/// - drop trailing singleton dimensions only when produced by scalar selectors.
///
/// E.g. A(:, 2, :) on 3×4×5 -> [3, 1, 5], not [3, 5].
fn matlab_squeezed_shape(selection_lengths: &[usize], scalar_mask: &[bool]) -> Vec<usize> {
    let mut dims: Vec<(usize, usize, bool)> = selection_lengths
        .iter()
        .enumerate()
        .map(|(d, &len)| (d, len, scalar_mask.get(d).copied().unwrap_or(false)))
        .collect();
    while dims.len() > 2
        && dims
            .last()
            .map(|&(_, len, is_scalar)| len == 1 && is_scalar)
            .unwrap_or(false)
    {
        dims.pop();
    }
    if dims.is_empty() {
        return vec![1, 1];
    }
    if dims.len() == 1 {
        let (dim, len, _) = dims[0];
        if dim == 1 {
            return vec![1, len];
        }
        return vec![len, 1];
    }
    dims.into_iter().map(|(_, len, _)| len).collect()
}

#[derive(Clone)]
enum ComplexAssignView {
    Scalar((f64, f64)),
    Array {
        data: Vec<(f64, f64)>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
}

fn build_complex_rhs_view(rhs: &Value, selection_lengths: &[usize]) -> VmResult<ComplexAssignView> {
    let dims = selection_lengths.len().max(1);
    match rhs {
        Value::Complex(re, im) => Ok(ComplexAssignView::Scalar((*re, *im))),
        Value::Num(n) => Ok(ComplexAssignView::Scalar((*n, 0.0))),
        Value::Int(i) => Ok(ComplexAssignView::Scalar((i.to_f64(), 0.0))),
        Value::Bool(b) => Ok(ComplexAssignView::Scalar((if *b { 1.0 } else { 0.0 }, 0.0))),
        Value::Tensor(t) => {
            let mut shape = t.shape.clone();
            if shape.len() < dims {
                shape.resize(dims, 1);
            } else if shape.len() > dims {
                if shape.iter().skip(dims).any(|&s| s != 1) {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
                shape.truncate(dims);
            }
            for (rhs_len, sel_len) in shape.iter().zip(selection_lengths.iter()) {
                if !(*rhs_len == 1 || *rhs_len == *sel_len) {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
            }
            let mut strides = vec![0usize; dims];
            let mut acc = 1usize;
            for d in 0..dims {
                strides[d] = acc;
                acc *= shape[d];
            }
            if acc != t.data.len() {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            let data: Vec<(f64, f64)> = t.data.iter().map(|&v| (v, 0.0)).collect();
            Ok(ComplexAssignView::Array {
                data,
                shape,
                strides,
            })
        }
        Value::ComplexTensor(ct) => {
            let mut shape = ct.shape.clone();
            if shape.len() < dims {
                shape.resize(dims, 1);
            } else if shape.len() > dims {
                if shape.iter().skip(dims).any(|&s| s != 1) {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
                shape.truncate(dims);
            }
            for (rhs_len, sel_len) in shape.iter().zip(selection_lengths.iter()) {
                if !(*rhs_len == 1 || *rhs_len == *sel_len) {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
            }
            let mut strides = vec![0usize; dims];
            let mut acc = 1usize;
            for d in 0..dims {
                strides[d] = acc;
                acc *= shape[d];
            }
            if acc != ct.data.len() {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            Ok(ComplexAssignView::Array {
                data: ct.data.clone(),
                shape,
                strides,
            })
        }
        other => Err(mex(
            "InvalidSliceAssignmentRhs",
            &format!("slice assign: unsupported RHS type {:?}", other),
        )),
    }
}

fn scatter_complex_with_plan(
    ct: &mut runmat_builtins::ComplexTensor,
    plan: &SlicePlan,
    view: &ComplexAssignView,
) -> VmResult<()> {
    if plan.indices.is_empty() {
        return Ok(());
    }
    let mut idx_iter = plan.indices.iter();
    cartesian_positions(&plan.selection_lengths, |position| {
        if let Some(&lin) = idx_iter.next() {
            let replacement = match view {
                ComplexAssignView::Scalar(v) => *v,
                ComplexAssignView::Array {
                    data,
                    shape,
                    strides,
                } => {
                    let mut rlin = 0usize;
                    for (d, &pos_val) in position.iter().enumerate() {
                        let rhs_len = shape.get(d).copied().unwrap_or(1);
                        let pos = if rhs_len == 1 { 0 } else { pos_val };
                        rlin += pos * strides.get(d).copied().unwrap_or(1);
                    }
                    data.get(rlin).copied().unwrap_or((0.0, 0.0))
                }
            };
            if let Some(slot) = ct.data.get_mut(lin as usize) {
                *slot = replacement;
            }
        }
    });
    Ok(())
}

enum StringAssignView {
    Scalar(String),
    Array {
        data: Vec<String>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
}

fn build_string_rhs_view(rhs: &Value, selection_lengths: &[usize]) -> VmResult<StringAssignView> {
    let dims = selection_lengths.len().max(1);
    match rhs {
        Value::String(s) => Ok(StringAssignView::Scalar(s.clone())),
        Value::Num(n) => Ok(StringAssignView::Scalar(n.to_string())),
        Value::Int(i) => Ok(StringAssignView::Scalar(i.to_i64().to_string())),
        Value::Tensor(t) => {
            let mut shape = t.shape.clone();
            if shape.len() < dims {
                shape.resize(dims, 1);
            } else if shape.len() > dims {
                if shape.iter().skip(dims).any(|&s| s != 1) {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
                shape.truncate(dims);
            }
            for (rhs_len, sel_len) in shape.iter().zip(selection_lengths.iter()) {
                if !(*rhs_len == 1 || *rhs_len == *sel_len) {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
            }
            let mut strides = vec![1usize; dims];
            for d in 1..dims {
                strides[d] = strides[d - 1] * shape[d - 1].max(1);
            }
            let data = t.data.iter().map(|v| v.to_string()).collect();
            Ok(StringAssignView::Array {
                data,
                shape,
                strides,
            })
        }
        Value::StringArray(sa) => {
            let mut shape = sa.shape.clone();
            if shape.len() < dims {
                shape.resize(dims, 1);
            } else if shape.len() > dims {
                if shape.iter().skip(dims).any(|&s| s != 1) {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
                shape.truncate(dims);
            }
            for (rhs_len, sel_len) in shape.iter().zip(selection_lengths.iter()) {
                if !(*rhs_len == 1 || *rhs_len == *sel_len) {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
            }
            let mut strides = vec![1usize; dims];
            for d in 1..dims {
                strides[d] = strides[d - 1] * shape[d - 1].max(1);
            }
            Ok(StringAssignView::Array {
                data: sa.data.clone(),
                shape,
                strides,
            })
        }
        _ => Err(mex(
            "InvalidSliceAssignmentRhs",
            "rhs must be string or string array",
        )),
    }
}

fn scatter_string_with_plan(
    sa: &mut runmat_builtins::StringArray,
    plan: &SlicePlan,
    view: &StringAssignView,
) -> VmResult<()> {
    if plan.indices.is_empty() {
        return Ok(());
    }
    let mut idx_iter = plan.indices.iter();
    cartesian_positions(&plan.selection_lengths, |position| {
        if let Some(&lin) = idx_iter.next() {
            let replacement = match view {
                StringAssignView::Scalar(s) => s.clone(),
                StringAssignView::Array {
                    data,
                    shape,
                    strides,
                } => {
                    let mut rlin = 0usize;
                    for (d, &pos_val) in position.iter().enumerate() {
                        let rhs_len = shape.get(d).copied().unwrap_or(1);
                        let pos = if rhs_len == 1 { 0 } else { pos_val };
                        rlin += pos * strides.get(d).copied().unwrap_or(1);
                    }
                    data.get(rlin).cloned().unwrap_or_default()
                }
            };
            if let Some(slot) = sa.data.get_mut(lin as usize) {
                *slot = replacement;
            }
        }
    });
    Ok(())
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

async fn materialize_rhs_linear(rhs: &Value, count: usize) -> VmResult<Vec<f64>> {
    let host_rhs = runmat_runtime::dispatcher::gather_if_needed_async(rhs).await?;

    match host_rhs {
        Value::Num(n) => Ok(vec![n; count]),
        Value::Int(int_val) => Ok(vec![int_val.to_f64(); count]),
        Value::Bool(b) => Ok(vec![if b { 1.0 } else { 0.0 }; count]),
        Value::Tensor(t) => {
            if t.data.len() == count {
                Ok(t.data)
            } else if t.data.len() == 1 {
                Ok(vec![t.data[0]; count])
            } else {
                Err(mex("ShapeMismatch", "shape mismatch for slice assign"))
            }
        }
        Value::LogicalArray(la) => {
            if la.data.len() == count {
                let out: Vec<f64> = la
                    .data
                    .into_iter()
                    .map(|b| if b != 0 { 1.0 } else { 0.0 })
                    .collect();
                Ok(out)
            } else if la.data.len() == 1 {
                let val = if la.data[0] != 0 { 1.0 } else { 0.0 };
                Ok(vec![val; count])
            } else {
                Err(mex("ShapeMismatch", "shape mismatch for slice assign"))
            }
        }
        other => Err(mex(
            "InvalidSliceAssignmentRhs",
            &format!("slice assign: unsupported RHS type {:?}", other),
        )),
    }
}

async fn value_to_complex_scalar(rhs: &Value) -> VmResult<(f64, f64)> {
    match rhs {
        Value::Complex(re, im) => Ok((*re, *im)),
        Value::Num(n) => Ok((*n, 0.0)),
        Value::Int(i) => Ok((i.to_f64(), 0.0)),
        Value::Bool(b) => Ok((if *b { 1.0 } else { 0.0 }, 0.0)),
        Value::Tensor(t) => {
            if t.data.len() == 1 {
                Ok((t.data[0], 0.0))
            } else {
                Err(mex("ScalarRequired", "RHS must be scalar"))
            }
        }
        Value::ComplexTensor(ct) => {
            if ct.data.len() == 1 {
                Ok(ct.data[0])
            } else {
                Err(mex("ScalarRequired", "RHS must be scalar"))
            }
        }
        Value::GpuTensor(h) => {
            let total = h.shape.iter().copied().product::<usize>();
            if total != 1 {
                return Err(mex("ScalarRequired", "RHS must be scalar"));
            }
            let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                mex(
                    "AccelerationProviderUnavailable",
                    "No acceleration provider registered",
                )
            })?;
            let host = provider
                .download(h)
                .await
                .map_err(|e| format!("gather rhs: {e}"))?;
            Ok((host.data.first().copied().unwrap_or(0.0), 0.0))
        }
        _ => Err(mex("NumericRequired", "RHS must be numeric")),
    }
}

async fn materialize_rhs_nd(rhs: &Value, selection_lengths: &[usize]) -> VmResult<Vec<f64>> {
    let rhs_host = runmat_runtime::dispatcher::gather_if_needed_async(rhs).await?;

    enum RhsView {
        Scalar(f64),
        Tensor {
            data: Vec<f64>,
            shape: Vec<usize>,
            strides: Vec<usize>,
        },
    }
    let view = match rhs_host {
        Value::Num(n) => RhsView::Scalar(n),
        Value::Int(iv) => RhsView::Scalar(iv.to_f64()),
        Value::Bool(b) => RhsView::Scalar(if b { 1.0 } else { 0.0 }),
        Value::Tensor(t) => {
            let mut shape = t.shape.clone();
            if shape.len() < selection_lengths.len() {
                shape.resize(selection_lengths.len(), 1);
            }
            if shape.len() > selection_lengths.len() {
                if shape.iter().skip(selection_lengths.len()).any(|&s| s != 1) {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
                shape.truncate(selection_lengths.len());
            }
            for (dim_len, &sel_len) in shape.iter().zip(selection_lengths.iter()) {
                if *dim_len != 1 && *dim_len != sel_len {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
            }
            let mut strides = vec![1usize; selection_lengths.len()];
            for d in 1..selection_lengths.len() {
                strides[d] = strides[d - 1] * shape[d - 1].max(1);
            }
            if t.data.len()
                != shape
                    .iter()
                    .copied()
                    .fold(1usize, |acc, len| acc.saturating_mul(len.max(1)))
            {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            RhsView::Tensor {
                data: t.data,
                shape,
                strides,
            }
        }
        Value::LogicalArray(la) => {
            if la.shape.len() > selection_lengths.len()
                && la
                    .shape
                    .iter()
                    .skip(selection_lengths.len())
                    .any(|&s| s != 1)
            {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            let mut shape = la.shape.clone();
            if shape.len() < selection_lengths.len() {
                shape.resize(selection_lengths.len(), 1);
            } else {
                shape.truncate(selection_lengths.len());
            }
            for (dim_len, &sel_len) in shape.iter().zip(selection_lengths.iter()) {
                if *dim_len != 1 && *dim_len != sel_len {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
            }
            let mut strides = vec![1usize; selection_lengths.len()];
            for d in 1..selection_lengths.len() {
                strides[d] = strides[d - 1] * shape[d - 1].max(1);
            }
            if la.data.len()
                != shape
                    .iter()
                    .copied()
                    .fold(1usize, |acc, len| acc.saturating_mul(len.max(1)))
            {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            let data: Vec<f64> = la
                .data
                .into_iter()
                .map(|b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            RhsView::Tensor {
                data,
                shape,
                strides,
            }
        }
        other => {
            return Err(mex(
                "InvalidSliceAssignmentRhs",
                &format!("slice assign: unsupported RHS type {:?}", other),
            ))
        }
    };

    let total = selection_lengths
        .iter()
        .copied()
        .fold(1usize, |acc, len| acc.saturating_mul(len.max(1)));
    let mut out = Vec::with_capacity(total);
    cartesian_positions(selection_lengths, |positions| match &view {
        RhsView::Scalar(val) => out.push(*val),
        RhsView::Tensor {
            data,
            shape,
            strides,
        } => {
            let mut rlin = 0usize;
            for d in 0..positions.len() {
                let rhs_len = shape[d];
                let pos = if rhs_len == 1 { 0 } else { positions[d] };
                rlin += pos * strides[d];
            }
            let value = data.get(rlin).copied().unwrap_or(0.0);
            out.push(value);
        }
    });
    Ok(out)
}

runmat_thread_local! {
    static GLOBALS: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
}

runmat_thread_local! {
    static PERSISTENTS: RefCell<HashMap<(String, usize), Value>> = RefCell::new(HashMap::new());
}

runmat_thread_local! {
    static PERSISTENTS_BY_NAME: RefCell<HashMap<(String, String), Value>> = RefCell::new(HashMap::new());
}

fn workspace_global_names() -> Vec<String> {
    let mut names = Vec::new();
    GLOBALS.with(|globals| {
        let map = globals.borrow();
        for key in map.keys() {
            if !key.starts_with("var_") {
                names.push(key.clone());
            }
        }
    });
    names.sort();
    names
}

fn ensure_workspace_resolver_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        runtime_workspace::register_workspace_resolver(WorkspaceResolver {
            lookup: workspace_lookup,
            snapshot: workspace_snapshot,
            globals: workspace_global_names,
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

fn rel_binary_use_builtin(a: &Value, b: &Value) -> bool {
    !matches!(a, Value::Num(_) | Value::Int(_)) || !matches!(b, Value::Num(_) | Value::Int(_))
}

macro_rules! handle_rel_binary { ($op:tt, $name:literal, $stack:ident) => {{
    let (a, b) = pop2(&mut $stack)?;
    match (&a, &b) {
        (Value::Object(obj), _) => { let args = vec![Value::Object(obj.clone()), Value::String($name.to_string()), b.clone()]; match call_builtin_vm!("call_method", &args) { Ok(v) => $stack.push(v), Err(_) => { let aa: f64 = (&a).try_into()?; let bb: f64 = (&b).try_into()?; $stack.push(Value::Num(if aa $op bb {1.0}else{0.0})) } } }
        (_, Value::Object(obj)) => { let rev = match $name { "lt" => "gt", "le" => "ge", "gt" => "lt", "ge" => "le", other => other };
            let args = vec![Value::Object(obj.clone()), Value::String(rev.to_string()), a.clone()]; match call_builtin_vm!("call_method", &args) { Ok(v) => $stack.push(v), Err(_) => { let aa: f64 = (&a).try_into()?; let bb: f64 = (&b).try_into()?; $stack.push(Value::Num(if aa $op bb {1.0}else{0.0})) } } }
        _ => {
            if rel_binary_use_builtin(&a, &b) {
                let v = call_builtin_vm!($name, &[a.clone(), b.clone()])?;
                $stack.push(v);
            } else {
                let bb: f64 = (&b).try_into()?;
                let aa: f64 = (&a).try_into()?;
                $stack.push(Value::Num(if aa $op bb {1.0}else{0.0}))
            }
        }
    }
}}; }
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
    let mut thread_roots: Vec<Value> = Vec::new();
    GLOBALS.with(|g| {
        for v in g.borrow().values() {
            thread_roots.push(v.clone());
        }
    });
    PERSISTENTS.with(|p| {
        for v in p.borrow().values() {
            thread_roots.push(v.clone());
        }
    });
    // Name-based table may duplicate persistents; harmless if included
    PERSISTENTS_BY_NAME.with(|p| {
        for v in p.borrow().values() {
            thread_roots.push(v.clone());
        }
    });
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
                        let key = format!("var_{stored_index}");
                        GLOBALS.with(|g| {
                            let mut m = g.borrow_mut();
                            if m.contains_key(&key) {
                                m.insert(key, stored_value.clone());
                            }
                        });
                        if let Some(name) = global_aliases.get(&stored_index) {
                            GLOBALS.with(|g| {
                                g.borrow_mut().insert(name.clone(), stored_value.clone());
                            });
                        }
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
                        let key = (func_name.to_string(), stored_offset);
                        PERSISTENTS.with(|p| {
                            let mut m = p.borrow_mut();
                            if m.contains_key(&key) {
                                m.insert(key, stored_value.clone());
                            }
                        });
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
                // Bind local var slots to global table entries by name (var_N)
                for i in indices.into_iter() {
                    let key = format!("var_{i}");
                    let val_opt = GLOBALS.with(|g| g.borrow().get(&key).cloned());
                    if let Some(v) = val_opt {
                        if i >= vars.len() {
                            vars.resize(i + 1, Value::Num(0.0));
                            refresh_workspace_state(&vars);
                        }
                        vars[i] = v;
                    }
                }
            }
            Instr::DeclareGlobalNamed(indices, names) => {
                for (pos, i) in indices.into_iter().enumerate() {
                    let name = names
                        .get(pos)
                        .cloned()
                        .unwrap_or_else(|| format!("var_{i}"));
                    let val_opt = GLOBALS.with(|g| g.borrow().get(&name).cloned());
                    if let Some(v) = val_opt {
                        if i >= vars.len() {
                            vars.resize(i + 1, Value::Num(0.0));
                            refresh_workspace_state(&vars);
                        }
                        vars[i] = v;
                    }
                    GLOBALS.with(|g| {
                        let mut m = g.borrow_mut();
                        if let Some(v) = m.get(&name).cloned() {
                            m.insert(format!("var_{i}"), v);
                        }
                    });
                    global_aliases.insert(i, name);
                }
            }
            Instr::DeclarePersistent(indices) => {
                // Initialize locals from persistent table if present
                let func_name = current_function_name.clone();
                for i in indices.into_iter() {
                    let key = (func_name.clone(), i);
                    let val_opt = PERSISTENTS.with(|p| p.borrow().get(&key).cloned());
                    if let Some(v) = val_opt {
                        if i >= vars.len() {
                            vars.resize(i + 1, Value::Num(0.0));
                            refresh_workspace_state(&vars);
                        }
                        vars[i] = v;
                    }
                }
            }
            Instr::DeclarePersistentNamed(indices, names) => {
                let func_name = current_function_name.clone();
                for (pos, i) in indices.into_iter().enumerate() {
                    let name = names
                        .get(pos)
                        .cloned()
                        .unwrap_or_else(|| format!("var_{i}"));
                    let key = (func_name.clone(), i);
                    let val_opt = PERSISTENTS_BY_NAME
                        .with(|p| p.borrow().get(&(func_name.clone(), name.clone())).cloned())
                        .or_else(|| PERSISTENTS.with(|p| p.borrow().get(&key).cloned()));
                    if let Some(v) = val_opt {
                        if i >= vars.len() {
                            vars.resize(i + 1, Value::Num(0.0));
                            refresh_workspace_state(&vars);
                        }
                        vars[i] = v;
                    }
                    persistent_aliases.insert(i, name);
                }
            }
            Instr::Add => {
                // If either operand is an object, try operator overloading
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("plus".to_string()),
                            b.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = call_builtin_vm!("plus", &[a.clone(), b.clone()])?;
                                stack.push(v)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("plus".to_string()),
                            a.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = call_builtin_vm!("plus", &[a.clone(), b.clone()])?;
                                stack.push(v)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                        let v = call_builtin_vm!("plus", &[a_acc, b_acc])?;
                        stack.push(v)
                    }
                }
            }
            Instr::Sub => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("minus".to_string()),
                            b.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = call_builtin_vm!("minus", &[a.clone(), b.clone()])?;
                                stack.push(v)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        // Subtraction is non-commutative: dispatch to b's class method with (a, b)
                        // in the original order so the method receives lhs=a, rhs=b and computes
                        // a - b. Using call_method here would be wrong because call_method always
                        // prepends the object as the first argument, computing b - a instead.
                        let qualified = format!("{}.minus", obj.class_name);
                        match call_builtin_vm!(&qualified, &[a.clone(), b.clone()]) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = call_builtin_vm!("minus", &[a.clone(), b.clone()])?;
                                stack.push(v)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                        let v = call_builtin_vm!("minus", &[a_acc, b_acc])?;
                        stack.push(v)
                    }
                }
            }
            Instr::Mul => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("mtimes".to_string()),
                            b.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = runmat_runtime::matrix::value_matmul(&a, &b).await?;
                                stack.push(v)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("mtimes".to_string()),
                            a.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = runmat_runtime::matrix::value_matmul(&a, &b).await?;
                                stack.push(v)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::MatMul, &a, &b).await?;
                        let v = runmat_runtime::matrix::value_matmul(&a_acc, &b_acc).await?;
                        stack.push(v)
                    }
                }
            }
            Instr::RightDiv => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                stack.push(execute_right_division(&a, &b).await?)
            }
            Instr::LeftDiv => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                stack.push(execute_left_division(&a, &b).await?)
            }
            Instr::Pow => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) | (_, Value::Object(obj)) => {
                        let arg_val = if matches!(&a, Value::Object(_)) {
                            b.clone()
                        } else {
                            a.clone()
                        };
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("power".to_string()),
                            arg_val,
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = runmat_runtime::power(&a, &b)?;
                                stack.push(v)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                        let v = runmat_runtime::power(&a_acc, &b_acc)?;
                        stack.push(v)
                    }
                }
            }
            Instr::Neg => {
                let value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match &value {
                    Value::Object(obj) => {
                        let args = vec![Value::Object(obj.clone())];
                        match call_builtin_vm!("uminus", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let result = call_builtin_vm!(
                                    "times",
                                    &[value.clone(), runmat_builtins::Value::Num(-1.0)],
                                )?;
                                stack.push(result)
                            }
                        }
                    }
                    _ => {
                        let result = call_builtin_vm!(
                            "times",
                            &[value.clone(), runmat_builtins::Value::Num(-1.0)],
                        )?;
                        stack.push(result);
                    }
                }
            }
            Instr::UPlus => {
                let value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match &value {
                    Value::Object(obj) => {
                        let args = vec![Value::Object(obj.clone())];
                        match call_builtin_vm!("uplus", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => stack.push(value),
                        }
                    }
                    _ => stack.push(value),
                }
            }
            Instr::Transpose => {
                let value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let promoted = accel_promote_unary(AutoUnaryOp::Transpose, &value).await?;
                let args = [promoted];
                let result = call_builtin_vm!("transpose", &args)?;
                stack.push(result);
            }
            Instr::ConjugateTranspose => {
                let value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let promoted = accel_promote_unary(AutoUnaryOp::Transpose, &value).await?;
                let args = [promoted];
                let result = call_builtin_vm!("ctranspose", &args)?;
                stack.push(result);
            }
            Instr::ElemMul => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("times".to_string()),
                            b.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                                stack.push(call_builtin_vm!("times", &[a_acc, b_acc])?)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("times".to_string()),
                            a.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                                stack.push(call_builtin_vm!("times", &[a_acc, b_acc])?)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                        stack.push(call_builtin_vm!("times", &[a_acc, b_acc])?)
                    }
                }
            }
            Instr::ElemDiv => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("rdivide".to_string()),
                            b.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                                stack.push(call_builtin_vm!("rdivide", &[a_acc, b_acc])?)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("rdivide".to_string()),
                            a.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                                stack.push(call_builtin_vm!("rdivide", &[a_acc, b_acc])?)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                        stack.push(call_builtin_vm!("rdivide", &[a_acc, b_acc])?)
                    }
                }
            }
            Instr::ElemPow => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) | (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            if matches!(&a, Value::Object(_)) {
                                b.clone()
                            } else {
                                a.clone()
                            },
                        ];
                        match call_builtin_vm!("power", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                                stack.push(call_builtin_vm!("power", &[a_acc, b_acc])?)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                        stack.push(call_builtin_vm!("power", &[a_acc, b_acc])?)
                    }
                }
            }
            Instr::ElemLeftDiv => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("ldivide".to_string()),
                            b.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (b_acc, a_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &b, &a).await?;
                                stack.push(call_builtin_vm!("rdivide", &[b_acc, a_acc])?)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("ldivide".to_string()),
                            a.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (b_acc, a_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &b, &a).await?;
                                stack.push(call_builtin_vm!("rdivide", &[b_acc, a_acc])?)
                            }
                        }
                    }
                    _ => {
                        let (b_acc, a_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &b, &a).await?;
                        stack.push(call_builtin_vm!("rdivide", &[b_acc, a_acc])?)
                    }
                }
            }
            Instr::LessEqual => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("le".to_string()),
                            b.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: le(a,b) = ~gt(a,b)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("gt".to_string()),
                                    b.clone(),
                                ];
                                match call_builtin_vm!("call_method", &args2) {
                                    Ok(v) => {
                                        let truth =
                                            logical_truth_from_value(&v, "comparison result")
                                                .await?;
                                        stack.push(Value::Num(if !truth { 1.0 } else { 0.0 }));
                                    }
                                    Err(_) => {
                                        let aa: f64 = (&a).try_into()?;
                                        let bb: f64 = (&b).try_into()?;
                                        stack.push(Value::Num(if aa <= bb { 1.0 } else { 0.0 }));
                                    }
                                }
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("ge".to_string()),
                            a.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: ge(b,a) = ~lt(b,a) hence le(a,b) = ge(b,a)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("lt".to_string()),
                                    a.clone(),
                                ];
                                match call_builtin_vm!("call_method", &args2) {
                                    Ok(v) => {
                                        let truth =
                                            logical_truth_from_value(&v, "comparison result")
                                                .await?;
                                        stack.push(Value::Num(if !truth { 1.0 } else { 0.0 }));
                                    }
                                    Err(_) => {
                                        let aa: f64 = (&a).try_into()?;
                                        let bb: f64 = (&b).try_into()?;
                                        stack.push(Value::Num(if aa <= bb { 1.0 } else { 0.0 }));
                                    }
                                }
                            }
                        }
                    }
                    _ => {
                        if rel_binary_use_builtin(&a, &b) {
                            let v = call_builtin_vm!("le", &[a.clone(), b.clone()])?;
                            stack.push(v);
                        } else {
                            let bb: f64 = (&b).try_into()?;
                            let aa: f64 = (&a).try_into()?;
                            stack.push(Value::Num(if aa <= bb { 1.0 } else { 0.0 }));
                        }
                    }
                }
            }
            Instr::Less => {
                handle_rel_binary!(<, "lt", stack);
            }
            Instr::Greater => {
                handle_rel_binary!(>, "gt", stack);
            }
            Instr::GreaterEqual => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("ge".to_string()),
                            b.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: ge(a,b) = ~lt(a,b)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("lt".to_string()),
                                    b.clone(),
                                ];
                                match call_builtin_vm!("call_method", &args2) {
                                    Ok(v) => {
                                        let truth =
                                            logical_truth_from_value(&v, "comparison result")
                                                .await?;
                                        stack.push(Value::Num(if !truth { 1.0 } else { 0.0 }));
                                    }
                                    Err(_) => {
                                        let aa: f64 = (&a).try_into()?;
                                        let bb: f64 = (&b).try_into()?;
                                        stack.push(Value::Num(if aa >= bb { 1.0 } else { 0.0 }));
                                    }
                                }
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("le".to_string()),
                            a.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: le(b,a) = ~gt(b,a); hence ge(a,b) = le(b,a)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("gt".to_string()),
                                    a.clone(),
                                ];
                                match call_builtin_vm!("call_method", &args2) {
                                    Ok(v) => {
                                        let truth =
                                            logical_truth_from_value(&v, "comparison result")
                                                .await?;
                                        stack.push(Value::Num(if !truth { 1.0 } else { 0.0 }));
                                    }
                                    Err(_) => {
                                        let aa: f64 = (&a).try_into()?;
                                        let bb: f64 = (&b).try_into()?;
                                        stack.push(Value::Num(if aa >= bb { 1.0 } else { 0.0 }));
                                    }
                                }
                            }
                        }
                    }
                    _ => {
                        if rel_binary_use_builtin(&a, &b) {
                            let v = call_builtin_vm!("ge", &[a.clone(), b.clone()])?;
                            stack.push(v);
                        } else {
                            let bb: f64 = (&b).try_into()?;
                            let aa: f64 = (&a).try_into()?;
                            stack.push(Value::Num(if aa >= bb { 1.0 } else { 0.0 }));
                        }
                    }
                }
            }
            Instr::Equal => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let push_logical =
                    |data: Vec<u8>, shape: Vec<usize>, stack: &mut Vec<Value>| -> VmResult<()> {
                        if data.len() == 1 && is_scalar_shape(&shape) {
                            stack.push(Value::Bool(data[0] != 0));
                            return Ok(());
                        }
                        let logical = runmat_builtins::LogicalArray::new(data, shape)
                            .map_err(|e| format!("eq: {e}"))?;
                        stack.push(Value::LogicalArray(logical));
                        Ok(())
                    };
                let logical_eq_scalar = |array: &runmat_builtins::LogicalArray,
                                         scalar: f64,
                                         stack: &mut Vec<Value>|
                 -> VmResult<()> {
                    let mut out = Vec::with_capacity(array.data.len());
                    for &bit in &array.data {
                        let val = if bit != 0 { 1.0 } else { 0.0 };
                        out.push(if (val - scalar).abs() < 1e-12 { 1 } else { 0 });
                    }
                    push_logical(out, array.shape.clone(), stack)
                };
                let logical_eq_tensor = |array: &runmat_builtins::LogicalArray,
                                         tensor: &runmat_builtins::Tensor,
                                         stack: &mut Vec<Value>|
                 -> VmResult<()> {
                    if array.shape != tensor.shape {
                        return Err(mex(
                            "ShapeMismatch",
                            "shape mismatch for element-wise comparison",
                        ));
                    }
                    let mut out = Vec::with_capacity(array.data.len());
                    for i in 0..array.data.len() {
                        let val = if array.data[i] != 0 { 1.0 } else { 0.0 };
                        out.push(if (val - tensor.data[i]).abs() < 1e-12 {
                            1
                        } else {
                            0
                        });
                    }
                    push_logical(out, array.shape.clone(), stack)
                };
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("eq".to_string()),
                            b.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let aa: f64 = (&a).try_into()?;
                                let bb: f64 = (&b).try_into()?;
                                stack.push(Value::Num(if aa == bb { 1.0 } else { 0.0 }))
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("eq".to_string()),
                            a.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let aa: f64 = (&a).try_into()?;
                                let bb: f64 = (&b).try_into()?;
                                stack.push(Value::Num(if aa == bb { 1.0 } else { 0.0 }))
                            }
                        }
                    }
                    (Value::HandleObject(_), _) | (_, Value::HandleObject(_)) => {
                        // Delegate to runtime eq builtin which implements identity semantics
                        let v = call_builtin_vm!("eq", &[a.clone(), b.clone()])?;
                        stack.push(v);
                    }
                    (Value::LogicalArray(la), Value::LogicalArray(lb)) => {
                        if la.shape != lb.shape {
                            return Err(mex(
                                "ShapeMismatch",
                                "shape mismatch for element-wise comparison",
                            ));
                        }
                        let mut out = Vec::with_capacity(la.data.len());
                        for i in 0..la.data.len() {
                            out.push(if la.data[i] == lb.data[i] { 1 } else { 0 });
                        }
                        push_logical(out, la.shape.clone(), &mut stack)?;
                    }
                    (Value::LogicalArray(la), Value::Num(n)) => {
                        logical_eq_scalar(la, *n, &mut stack)?;
                    }
                    (Value::LogicalArray(la), Value::Int(i)) => {
                        logical_eq_scalar(la, i.to_f64(), &mut stack)?;
                    }
                    (Value::LogicalArray(la), Value::Bool(flag)) => {
                        logical_eq_scalar(la, if *flag { 1.0 } else { 0.0 }, &mut stack)?;
                    }
                    (Value::Num(n), Value::LogicalArray(lb)) => {
                        logical_eq_scalar(lb, *n, &mut stack)?;
                    }
                    (Value::Int(i), Value::LogicalArray(lb)) => {
                        logical_eq_scalar(lb, i.to_f64(), &mut stack)?;
                    }
                    (Value::Bool(flag), Value::LogicalArray(lb)) => {
                        logical_eq_scalar(lb, if *flag { 1.0 } else { 0.0 }, &mut stack)?;
                    }
                    (Value::LogicalArray(la), Value::Tensor(tb)) => {
                        logical_eq_tensor(la, tb, &mut stack)?;
                    }
                    (Value::Tensor(ta), Value::LogicalArray(lb)) => {
                        logical_eq_tensor(lb, ta, &mut stack)?;
                    }
                    (Value::Tensor(ta), Value::Tensor(tb)) => {
                        // Element-wise eq; shapes must match
                        if ta.shape != tb.shape {
                            return Err(mex(
                                "ShapeMismatch",
                                "shape mismatch for element-wise comparison",
                            ));
                        }
                        let mut out = Vec::with_capacity(ta.data.len());
                        for i in 0..ta.data.len() {
                            out.push(if (ta.data[i] - tb.data[i]).abs() < 1e-12 {
                                1.0
                            } else {
                                0.0
                            });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, ta.shape.clone())
                                .map_err(|e| format!("eq: {e}"))?,
                        ));
                    }
                    (Value::Tensor(t), Value::Num(_)) | (Value::Tensor(t), Value::Int(_)) => {
                        let s = match &b {
                            Value::Num(n) => *n,
                            Value::Int(i) => i.to_f64(),
                            _ => 0.0,
                        };
                        let out: Vec<f64> = t
                            .data
                            .iter()
                            .map(|x| if (*x - s).abs() < 1e-12 { 1.0 } else { 0.0 })
                            .collect();
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, t.shape.clone())
                                .map_err(|e| format!("eq: {e}"))?,
                        ));
                    }
                    (Value::Num(_), Value::Tensor(t)) | (Value::Int(_), Value::Tensor(t)) => {
                        let s = match &a {
                            Value::Num(n) => *n,
                            Value::Int(i) => i.to_f64(),
                            _ => 0.0,
                        };
                        let out: Vec<f64> = t
                            .data
                            .iter()
                            .map(|x| if (s - *x).abs() < 1e-12 { 1.0 } else { 0.0 })
                            .collect();
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, t.shape.clone())
                                .map_err(|e| format!("eq: {e}"))?,
                        ));
                    }
                    (Value::StringArray(sa), Value::StringArray(sb)) => {
                        if sa.shape != sb.shape {
                            return Err(mex(
                                "ShapeMismatch",
                                "shape mismatch for string array comparison",
                            ));
                        }
                        let mut out = Vec::with_capacity(sa.data.len());
                        for i in 0..sa.data.len() {
                            out.push(if sa.data[i] == sb.data[i] { 1.0 } else { 0.0 });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, sa.shape.clone())
                                .map_err(|e| format!("eq: {e}"))?,
                        ));
                    }
                    (Value::StringArray(sa), Value::String(s)) => {
                        let mut out = Vec::with_capacity(sa.data.len());
                        for i in 0..sa.data.len() {
                            out.push(if sa.data[i] == *s { 1.0 } else { 0.0 });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, sa.shape.clone())
                                .map_err(|e| format!("eq: {e}"))?,
                        ));
                    }
                    (Value::String(s), Value::StringArray(sa)) => {
                        let mut out = Vec::with_capacity(sa.data.len());
                        for i in 0..sa.data.len() {
                            out.push(if *s == sa.data[i] { 1.0 } else { 0.0 });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, sa.shape.clone())
                                .map_err(|e| format!("eq: {e}"))?,
                        ));
                    }
                    (Value::String(a_s), Value::String(b_s)) => {
                        stack.push(Value::Num(if a_s == b_s { 1.0 } else { 0.0 }));
                    }
                    _ => {
                        let bb: f64 = (&b).try_into()?;
                        let aa: f64 = (&a).try_into()?;
                        stack.push(Value::Num(if aa == bb { 1.0 } else { 0.0 }));
                    }
                }
            }
            Instr::NotEqual => {
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("ne".to_string()),
                            b.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: ne(a,b) = ~eq(a,b)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("eq".to_string()),
                                    b.clone(),
                                ];
                                match call_builtin_vm!("call_method", &args2) {
                                    Ok(v) => {
                                        let truth =
                                            logical_truth_from_value(&v, "comparison result")
                                                .await?;
                                        stack.push(Value::Num(if !truth { 1.0 } else { 0.0 }));
                                    }
                                    Err(_) => {
                                        let aa: f64 = (&a).try_into()?;
                                        let bb: f64 = (&b).try_into()?;
                                        stack.push(Value::Num(if aa != bb { 1.0 } else { 0.0 }));
                                    }
                                }
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("ne".to_string()),
                            a.clone(),
                        ];
                        match call_builtin_vm!("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: ne(b,a) = ~eq(b,a)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("eq".to_string()),
                                    a.clone(),
                                ];
                                match call_builtin_vm!("call_method", &args2) {
                                    Ok(v) => {
                                        let truth =
                                            logical_truth_from_value(&v, "comparison result")
                                                .await?;
                                        stack.push(Value::Num(if !truth { 1.0 } else { 0.0 }));
                                    }
                                    Err(_) => {
                                        let aa: f64 = (&a).try_into()?;
                                        let bb: f64 = (&b).try_into()?;
                                        stack.push(Value::Num(if aa != bb { 1.0 } else { 0.0 }));
                                    }
                                }
                            }
                        }
                    }
                    (Value::HandleObject(_), _) | (_, Value::HandleObject(_)) => {
                        let v = call_builtin_vm!("ne", &[a.clone(), b.clone()])?;
                        stack.push(v);
                    }
                    (Value::Tensor(ta), Value::Tensor(tb)) => {
                        if ta.shape != tb.shape {
                            return Err(mex(
                                "ShapeMismatch",
                                "shape mismatch for element-wise comparison",
                            ));
                        }
                        let mut out = Vec::with_capacity(ta.data.len());
                        for i in 0..ta.data.len() {
                            out.push(if (ta.data[i] - tb.data[i]).abs() >= 1e-12 {
                                1.0
                            } else {
                                0.0
                            });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, ta.shape.clone())
                                .map_err(|e| format!("ne: {e}"))?,
                        ));
                    }
                    (Value::Tensor(t), Value::Num(_)) | (Value::Tensor(t), Value::Int(_)) => {
                        let s = match &b {
                            Value::Num(n) => *n,
                            Value::Int(i) => i.to_f64(),
                            _ => 0.0,
                        };
                        let out: Vec<f64> = t
                            .data
                            .iter()
                            .map(|x| if (*x - s).abs() >= 1e-12 { 1.0 } else { 0.0 })
                            .collect();
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, t.shape.clone())
                                .map_err(|e| format!("ne: {e}"))?,
                        ));
                    }
                    (Value::Num(_), Value::Tensor(t)) | (Value::Int(_), Value::Tensor(t)) => {
                        let s = match &a {
                            Value::Num(n) => *n,
                            Value::Int(i) => i.to_f64(),
                            _ => 0.0,
                        };
                        let out: Vec<f64> = t
                            .data
                            .iter()
                            .map(|x| if (s - *x).abs() >= 1e-12 { 1.0 } else { 0.0 })
                            .collect();
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, t.shape.clone())
                                .map_err(|e| format!("ne: {e}"))?,
                        ));
                    }
                    (Value::StringArray(sa), Value::StringArray(sb)) => {
                        if sa.shape != sb.shape {
                            return Err(mex(
                                "ShapeMismatch",
                                "shape mismatch for string array comparison",
                            ));
                        }
                        let mut out = Vec::with_capacity(sa.data.len());
                        for i in 0..sa.data.len() {
                            out.push(if sa.data[i] != sb.data[i] { 1.0 } else { 0.0 });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, sa.shape.clone())
                                .map_err(|e| format!("ne: {e}"))?,
                        ));
                    }
                    (Value::StringArray(sa), Value::String(s)) => {
                        let mut out = Vec::with_capacity(sa.data.len());
                        for i in 0..sa.data.len() {
                            out.push(if sa.data[i] != *s { 1.0 } else { 0.0 });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, sa.shape.clone())
                                .map_err(|e| format!("ne: {e}"))?,
                        ));
                    }
                    (Value::String(s), Value::StringArray(sa)) => {
                        let mut out = Vec::with_capacity(sa.data.len());
                        for i in 0..sa.data.len() {
                            out.push(if *s != sa.data[i] { 1.0 } else { 0.0 });
                        }
                        stack.push(Value::Tensor(
                            runmat_builtins::Tensor::new(out, sa.shape.clone())
                                .map_err(|e| format!("ne: {e}"))?,
                        ));
                    }
                    (Value::String(a_s), Value::String(b_s)) => {
                        stack.push(Value::Num(if a_s != b_s { 1.0 } else { 0.0 }));
                    }
                    _ => {
                        let bb: f64 = (&b).try_into()?;
                        let aa: f64 = (&a).try_into()?;
                        stack.push(Value::Num(if aa != bb { 1.0 } else { 0.0 }));
                    }
                }
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
                // Pop count values and build a 1xN numeric tensor (Num only; others error)
                let mut vals: Vec<f64> = Vec::with_capacity(count);
                let mut tmp: Vec<Value> = Vec::with_capacity(count);
                for _ in 0..count {
                    tmp.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                tmp.reverse();
                for v in tmp {
                    let n: f64 = (&v).try_into()?;
                    vals.push(n);
                }
                let tens = runmat_builtins::Tensor::new(vals, vec![1, count])
                    .map_err(|e| format!("PackToRow: {e}"))?;
                stack.push(Value::Tensor(tens));
            }
            Instr::PackToCol(count) => {
                let mut vals: Vec<f64> = Vec::with_capacity(count);
                let mut tmp: Vec<Value> = Vec::with_capacity(count);
                for _ in 0..count {
                    tmp.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                tmp.reverse();
                for v in tmp {
                    let n: f64 = (&v).try_into()?;
                    vals.push(n);
                }
                let tens = runmat_builtins::Tensor::new(vals, vec![count, 1])
                    .map_err(|e| format!("PackToCol: {e}"))?;
                stack.push(Value::Tensor(tens));
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
                let total_elements = rows * cols;
                let mut row_major = Vec::with_capacity(total_elements);
                for _ in 0..total_elements {
                    let val: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    row_major.push(val);
                }
                row_major.reverse();
                // Reorder to column-major storage: cm[r + c*rows] = rm[r*cols + c]
                let mut data = vec![0.0; total_elements];
                for r in 0..rows {
                    for c in 0..cols {
                        data[r + c * rows] = row_major[r * cols + c];
                    }
                }
                let matrix = runmat_builtins::Tensor::new_2d(data, rows, cols)
                    .map_err(|e| format!("Matrix creation error: {e}"))?;
                stack.push(Value::Tensor(matrix));
            }
            Instr::CreateMatrixDynamic(num_rows) => {
                let mut row_lengths = Vec::new();
                for _ in 0..num_rows {
                    let row_len: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    row_lengths.push(row_len as usize);
                }
                row_lengths.reverse();
                let mut rows_data = Vec::new();
                for &row_len in row_lengths.iter().rev() {
                    let mut row_values = Vec::new();
                    for _ in 0..row_len {
                        row_values.push(
                            stack
                                .pop()
                                .ok_or(mex("StackUnderflow", "stack underflow"))?,
                        );
                    }
                    row_values.reverse();
                    rows_data.push(row_values);
                }
                rows_data.reverse();
                let result = runmat_runtime::create_matrix_from_values(&rows_data).await?;
                stack.push(result);
            }
            Instr::CreateRange(has_step) => {
                if has_step {
                    // NOTE: Do not coerce to f64 here; start/step/end may be GPU-backed scalar
                    // tensors (e.g. loop variable `t` living on GPU). Delegate to `colon` builtin
                    // which contains the correct scalar extraction and GPU preference semantics.
                    let end = stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?;
                    let step = stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?;
                    let start = stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?;
                    let args = vec![start, step, end];
                    match call_builtin_vm!("colon", &args) {
                        Ok(v) => stack.push(v),
                        Err(e) => vm_bail!(e.to_string()),
                    }
                } else {
                    let end = stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?;
                    let start = stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?;
                    let args = vec![start, end];
                    match call_builtin_vm!("colon", &args) {
                        Ok(v) => stack.push(v),
                        Err(e) => vm_bail!(e.to_string()),
                    }
                }
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
                            if indices.is_empty() {
                                if let Ok(zeros) = provider.zeros(&output_shape) {
                                    stack.push(Value::GpuTensor(zeros));
                                    pc += 1;
                                    continue;
                                }
                            } else if let Ok(result) =
                                provider.gather_linear(handle, &indices, &output_shape)
                            {
                                stack.push(Value::GpuTensor(result));
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
                        let cell =
                            runmat_builtins::CellArray::new(numeric.clone(), 1, numeric.len())
                                .map_err(|e| format!("subsasgn build error: {e}"))?;
                        match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::Object(obj.clone()),
                                Value::String("subsasgn".to_string()),
                                Value::String("()".to_string()),
                                Value::Cell(cell.clone()),
                                rhs.clone(),
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(_e) => {
                                // Fallback to direct builtin OverIdx.subsasgn if class method isn't registered
                                // Determine class name and call fully qualified builtin if present
                                let qualified = format!("{}.subsasgn", obj.class_name);
                                match call_builtin_vm!(
                                    &qualified,
                                    &[
                                        Value::Object(obj),
                                        Value::String("()".to_string()),
                                        Value::Cell(cell),
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
                        let cell =
                            runmat_builtins::CellArray::new(numeric.clone(), 1, numeric.len())
                                .map_err(|e| format!("subsasgn build error: {e}"))?;
                        match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::HandleObject(handle),
                                Value::String("subsasgn".to_string()),
                                Value::String("()".to_string()),
                                Value::Cell(cell),
                                rhs,
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e.to_string()),
                        }
                    }
                    Value::Tensor(mut t) => {
                        // F4: write barrier hook (placeholder) – in a full GC integration, call into GC pre/post here
                        // Linear 1-D indexing assignment: A(I) = rhs
                        if dims == 1 {
                            let total = t.data.len();
                            // Build linear index list
                            let is_colon = (colon_mask & 1u32) != 0;
                            let is_end = (end_mask & 1u32) != 0;
                            let lin_indices = if is_colon {
                                (1..=total).collect()
                            } else if is_end {
                                vec![total]
                            } else {
                                let v = numeric
                                    .first()
                                    .ok_or(mex("MissingNumericIndex", "missing numeric index"))?;
                                match indices_from_value_linear(v, total).await {
                                    Ok(idxs) => idxs,
                                    Err(err) => vm_bail!(err),
                                }
                            };
                            // Scatter RHS
                            match rhs {
                                Value::Num(v) => {
                                    for &li in &lin_indices {
                                        t.data[li - 1] = v;
                                    }
                                }
                                Value::Tensor(rt) => {
                                    if rt.data.len() == 1 {
                                        let v = rt.data[0];
                                        for &li in &lin_indices {
                                            t.data[li - 1] = v;
                                        }
                                    } else if rt.data.len() == lin_indices.len() {
                                        for (k, &li) in lin_indices.iter().enumerate() {
                                            t.data[li - 1] = rt.data[k];
                                        }
                                    } else {
                                        vm_bail!(
                                            "shape mismatch for linear slice assign".to_string()
                                        );
                                    }
                                }
                                _ => vm_bail!("rhs must be numeric or tensor".to_string()),
                            }
                            stack.push(Value::Tensor(t));
                        } else {
                            let rank = t.shape.len();
                            let mut selectors: Vec<SliceSelector> = Vec::with_capacity(dims);
                            let mut num_iter = 0usize;
                            for d in 0..dims {
                                let is_colon = (colon_mask & (1u32 << d)) != 0;
                                let is_end = (end_mask & (1u32 << d)) != 0;
                                if is_colon {
                                    selectors.push(SliceSelector::Colon);
                                } else if is_end {
                                    selectors
                                        .push(SliceSelector::Scalar(*t.shape.get(d).unwrap_or(&1)));
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
                            // 2-D write fast paths (full column/row) with strict broadcast checks
                            if dims == 2 {
                                let rows = if rank >= 1 { t.shape[0] } else { 1 };
                                let cols = if rank >= 2 { t.shape[1] } else { 1 };
                                match (&selectors[0], &selectors[1]) {
                                    // A(:, j) = rhs
                                    (SliceSelector::Colon, SliceSelector::Scalar(j)) => {
                                        let j0 = *j - 1;
                                        // Size growth semantics: extend columns if needed
                                        if j0 >= cols {
                                            let new_cols = j0 + 1;
                                            let new_rows = rows;
                                            let mut new_data = vec![0.0f64; new_rows * new_cols];
                                            for c in 0..cols {
                                                let src_off = c * rows;
                                                let dst_off = c * new_rows;
                                                new_data[dst_off..dst_off + rows].copy_from_slice(
                                                    &t.data[src_off..src_off + rows],
                                                );
                                            }
                                            t.data = new_data;
                                            t.shape = vec![new_rows, new_cols];
                                            t.rows = new_rows;
                                            t.cols = new_cols;
                                        }
                                        let start = j0 * rows;
                                        match rhs {
                                            Value::Num(v) => {
                                                for r in 0..rows {
                                                    t.data[start + r] = v;
                                                }
                                            }
                                            Value::Tensor(rt) => {
                                                let len = rt.data.len();
                                                if len == rows {
                                                    for r in 0..rows {
                                                        t.data[start + r] = rt.data[r];
                                                    }
                                                } else if len == 1 {
                                                    for r in 0..rows {
                                                        t.data[start + r] = rt.data[0];
                                                    }
                                                } else {
                                                    vm_bail!("shape mismatch for slice assign"
                                                        .to_string());
                                                }
                                            }
                                            _ => {
                                                vm_bail!("rhs must be numeric or tensor".to_string())
                                            }
                                        }
                                        stack.push(Value::Tensor(t));
                                        bench_end("StoreSlice2D.fast_col", __b);
                                        pc += 1;
                                        continue;
                                    }
                                    // A(i, :) = rhs
                                    (SliceSelector::Scalar(i), SliceSelector::Colon) => {
                                        let i0 = *i - 1;
                                        // Size growth semantics: extend rows if needed
                                        if i0 >= rows {
                                            let new_rows = i0 + 1;
                                            let new_cols = cols;
                                            let mut new_data = vec![0.0f64; new_rows * new_cols];
                                            for c in 0..cols {
                                                for r in 0..rows {
                                                    new_data[r + c * new_rows] =
                                                        t.data[r + c * rows];
                                                }
                                            }
                                            t.data = new_data;
                                            t.shape = vec![new_rows, new_cols];
                                            t.rows = new_rows;
                                            t.cols = new_cols;
                                        }
                                        match rhs {
                                            Value::Num(v) => {
                                                for c in 0..cols {
                                                    t.data[i0 + c * rows] = v;
                                                }
                                            }
                                            Value::Tensor(rt) => {
                                                let len = rt.data.len();
                                                if len == cols {
                                                    for c in 0..cols {
                                                        t.data[i0 + c * rows] = rt.data[c];
                                                    }
                                                } else if len == 1 {
                                                    for c in 0..cols {
                                                        t.data[i0 + c * rows] = rt.data[0];
                                                    }
                                                } else {
                                                    vm_bail!("shape mismatch for slice assign"
                                                        .to_string());
                                                }
                                            }
                                            _ => {
                                                vm_bail!("rhs must be numeric or tensor".to_string())
                                            }
                                        }
                                        stack.push(Value::Tensor(t));
                                        bench_end("StoreSlice2D.fast_row", __b);
                                        pc += 1;
                                        continue;
                                    }
                                    _ => {}
                                }
                            }
                            // Generic N-D writer path
                            // Build per-dim index lists and strides
                            let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                            let full_shape: Vec<usize> = if rank < dims {
                                let mut s = t.shape.clone();
                                s.resize(dims, 1);
                                s
                            } else {
                                t.shape.clone()
                            };
                            for d in 0..dims {
                                let dim_len = full_shape[d];
                                let idxs = match &selectors[d] {
                                    SliceSelector::Colon => (1..=dim_len).collect(),
                                    SliceSelector::Scalar(i) => vec![*i],
                                    SliceSelector::Indices(v) => v.clone(),
                                    SliceSelector::LinearIndices { values: v, .. } => v.clone(),
                                };
                                if idxs.iter().any(|&i| i == 0 || i > dim_len) {
                                    vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                                }
                                per_dim_indices.push(idxs);
                            }
                            // Column-major strides (first dimension fastest)
                            let mut strides: Vec<usize> = vec![0; dims];
                            let mut acc = 1usize;
                            for d in 0..dims {
                                strides[d] = acc;
                                acc *= full_shape[d];
                            }
                            let total_out: usize =
                                per_dim_indices.iter().map(|v| v.len()).product();
                            // Prepare RHS values
                            enum RhsView {
                                Scalar(f64),
                                Tensor {
                                    data: Vec<f64>,
                                    shape: Vec<usize>,
                                    strides: Vec<usize>,
                                },
                            }
                            let rhs_view = match rhs {
                                Value::Num(n) => RhsView::Scalar(n),
                                Value::Tensor(rt) => {
                                    // Allow exact match or N-D broadcasting where rhs_dim is 1 or equals out_dim
                                    let mut shape = rt.shape.clone();
                                    if shape.len() < dims {
                                        shape.resize(dims, 1);
                                    }
                                    if shape.len() > dims {
                                        if shape.iter().skip(dims).any(|&s| s != 1) {
                                            vm_bail!("shape mismatch for slice assign".to_string());
                                        }
                                        shape.truncate(dims);
                                    }
                                    let mut ok = true;
                                    for d in 0..dims {
                                        let out_len = per_dim_indices[d].len();
                                        let rhs_len = shape[d];
                                        if !(rhs_len == 1 || rhs_len == out_len) {
                                            ok = false;
                                            break;
                                        }
                                    }
                                    if !ok {
                                        vm_bail!("shape mismatch for slice assign".to_string());
                                    }
                                    let mut rstrides = vec![0usize; dims];
                                    let mut racc = 1usize;
                                    for d in 0..dims {
                                        rstrides[d] = racc;
                                        racc *= shape[d];
                                    }
                                    RhsView::Tensor {
                                        data: rt.data,
                                        shape,
                                        strides: rstrides,
                                    }
                                }
                                _ => vm_bail!("rhs must be numeric or tensor".to_string()),
                            };
                            // Iterate and scatter
                            let mut _k = 0usize;
                            let mut idx = vec![0usize; dims];
                            if total_out == 0 {
                                stack.push(Value::Tensor(t));
                            } else {
                                loop {
                                    let mut lin = 0usize;
                                    for d in 0..dims {
                                        let i0 = per_dim_indices[d][idx[d]] - 1;
                                        lin += i0 * strides[d];
                                    }
                                    match &rhs_view {
                                        RhsView::Scalar(val) => t.data[lin] = *val,
                                        RhsView::Tensor {
                                            data,
                                            shape,
                                            strides,
                                        } => {
                                            let mut rlin = 0usize;
                                            for d in 0..dims {
                                                let rhs_len = shape[d];
                                                let pos = if rhs_len == 1 { 0 } else { idx[d] };
                                                rlin += pos * strides[d];
                                            }
                                            t.data[lin] = data[rlin];
                                        }
                                    }
                                    _k += 1;
                                    // Increment first dim fastest
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
                                stack.push(Value::Tensor(t));
                            }
                        }
                    }
                    Value::GpuTensor(handle) => {
                        if let Some(provider) = runmat_accelerate_api::provider() {
                            let base_shape = handle.shape.clone();
                            if let Ok(selectors) = build_slice_selectors(
                                dims,
                                colon_mask,
                                end_mask,
                                &numeric,
                                &base_shape,
                            )
                            .await
                            {
                                if dims == 2 {
                                    if let (Some(sel0), Some(sel1)) =
                                        (selectors.first(), selectors.get(1))
                                    {
                                        let rows = base_shape.first().copied().unwrap_or(1);
                                        let cols = base_shape.get(1).copied().unwrap_or(1);
                                        if let Value::GpuTensor(vh) = &rhs {
                                            if let (
                                                SliceSelector::Colon,
                                                SliceSelector::Scalar(j),
                                            ) = (sel0, sel1)
                                            {
                                                let j0 = *j - 1;
                                                if j0 < cols {
                                                    let v_rows = match vh.shape.len() {
                                                        1 | 2 => vh.shape[0],
                                                        _ => 0,
                                                    };
                                                    if v_rows == rows {
                                                        if let Ok(new_h) =
                                                            provider.scatter_column(&handle, j0, vh)
                                                        {
                                                            stack.push(Value::GpuTensor(new_h));
                                                            bench_end("StoreSlice2D.fast_col", __b);
                                                            pc += 1;
                                                            continue;
                                                        }
                                                    }
                                                }
                                            }
                                            if let (
                                                SliceSelector::Scalar(i),
                                                SliceSelector::Colon,
                                            ) = (sel0, sel1)
                                            {
                                                let i0 = *i - 1;
                                                if i0 < rows {
                                                    let v_cols = match vh.shape.len() {
                                                        1 => vh.shape[0],
                                                        2 => vh.shape[1],
                                                        _ => 0,
                                                    };
                                                    if v_cols == cols {
                                                        if let Ok(new_h) =
                                                            provider.scatter_row(&handle, i0, vh)
                                                        {
                                                            stack.push(Value::GpuTensor(new_h));
                                                            bench_end("StoreSlice2D.fast_row", __b);
                                                            pc += 1;
                                                            continue;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                if let Ok(plan) = build_slice_plan(&selectors, dims, &base_shape) {
                                    if plan.indices.is_empty() {
                                        stack.push(Value::GpuTensor(handle));
                                        bench_end("StoreSlice", __b);
                                        pc += 1;
                                        continue;
                                    }
                                    let values_result = if plan.dims == 1 {
                                        let count =
                                            plan.selection_lengths.first().copied().unwrap_or(0);
                                        materialize_rhs_linear(&rhs, count).await
                                    } else {
                                        materialize_rhs_nd(&rhs, &plan.selection_lengths).await
                                    };
                                    if let Ok(values) = values_result {
                                        if values.len() == plan.indices.len() {
                                            let value_shape = vec![values.len().max(1), 1];
                                            let upload_result = if values.is_empty() {
                                                provider.zeros(&[0, 1])
                                            } else {
                                                provider.upload(
                                                    &runmat_accelerate_api::HostTensorView {
                                                        data: &values,
                                                        shape: &value_shape,
                                                    },
                                                )
                                            };
                                            if let Ok(values_handle) = upload_result {
                                                if provider
                                                    .scatter_linear(
                                                        &handle,
                                                        &plan.indices,
                                                        &values_handle,
                                                    )
                                                    .is_ok()
                                                {
                                                    stack.push(Value::GpuTensor(handle));
                                                    bench_end("StoreSlice", __b);
                                                    pc += 1;
                                                    continue;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // Gather–mutate–reupload fallback for slice assignment on GPU bases
                        let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                            mex(
                                "AccelerationProviderUnavailable",
                                "No acceleration provider registered",
                            )
                        })?;
                        debug!(
                            "StoreSlice: falling back to host tensor path base_shape={:?}",
                            handle.shape
                        );
                        let host = provider
                            .download(&handle)
                            .await
                            .map_err(|e| format!("gather for slice assign: {e}"))?;
                        let mut t = runmat_builtins::Tensor::new(host.data, host.shape)
                            .map_err(|e| format!("slice assign: {e}"))?;
                        // Linear 1-D indexing assignment: A(I) = rhs
                        if dims == 1 {
                            let total = t.data.len();
                            // Build linear index list
                            let mut lin_indices: Vec<usize> = Vec::new();
                            let is_colon = (colon_mask & 1u32) != 0;
                            let is_end = (end_mask & 1u32) != 0;
                            if is_colon {
                                lin_indices = (1..=total).collect();
                            } else if is_end {
                                lin_indices = vec![total];
                            } else {
                                let v = numeric
                                    .first()
                                    .ok_or(mex("MissingNumericIndex", "missing numeric index"))?;
                                if let Some(i) = index_scalar_from_value(v).await? {
                                    if i < 1 || (i as usize) > total {
                                        vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                                    }
                                    lin_indices.push(i as usize);
                                } else {
                                    match v {
                                        Value::Tensor(idx_t) => {
                                            let len = idx_t.shape.iter().product::<usize>();
                                            if len == total {
                                                for (i, &val) in idx_t.data.iter().enumerate() {
                                                    if val != 0.0 {
                                                        lin_indices.push(i + 1);
                                                    }
                                                }
                                            } else {
                                                for &val in &idx_t.data {
                                                    let i = val as isize;
                                                    if i < 1 || (i as usize) > total {
                                                        vm_bail!(mex(
                                                            "IndexOutOfBounds",
                                                            "Index out of bounds"
                                                        ));
                                                    }
                                                    lin_indices.push(i as usize);
                                                }
                                            }
                                        }
                                        _ => vm_bail!(mex(
                                            "UnsupportedIndexType",
                                            "Unsupported index type"
                                        )),
                                    }
                                }
                            }
                            // Scatter RHS
                            match rhs {
                                Value::Num(v) => {
                                    for &li in &lin_indices {
                                        t.data[li - 1] = v;
                                    }
                                }
                                Value::Tensor(rt) => {
                                    if rt.data.len() == 1 {
                                        let v = rt.data[0];
                                        for &li in &lin_indices {
                                            t.data[li - 1] = v;
                                        }
                                    } else if rt.data.len() == lin_indices.len() {
                                        for (k, &li) in lin_indices.iter().enumerate() {
                                            t.data[li - 1] = rt.data[k];
                                        }
                                    } else {
                                        vm_bail!(
                                            "shape mismatch for linear slice assign".to_string()
                                        );
                                    }
                                }
                                _ => vm_bail!("rhs must be numeric or tensor".to_string()),
                            }
                            let view = runmat_accelerate_api::HostTensorView {
                                data: &t.data,
                                shape: &t.shape,
                            };
                            let new_h = provider
                                .upload(&view)
                                .map_err(|e| format!("reupload after slice assign: {e}"))?;
                            stack.push(Value::GpuTensor(new_h));
                        } else {
                            let rank = t.shape.len();
                            #[derive(Clone)]
                            enum Sel {
                                Colon,
                                Scalar(usize),
                                Indices(Vec<usize>),
                            }
                            let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                            let mut num_iter = 0usize;
                            for d in 0..dims {
                                let is_colon = (colon_mask & (1u32 << d)) != 0;
                                let is_end = (end_mask & (1u32 << d)) != 0;
                                if is_colon {
                                    selectors.push(Sel::Colon);
                                } else if is_end {
                                    selectors.push(Sel::Scalar(*t.shape.get(d).unwrap_or(&1)));
                                } else {
                                    let v = numeric.get(num_iter).ok_or(mex(
                                        "MissingNumericIndex",
                                        "missing numeric index",
                                    ))?;
                                    num_iter += 1;
                                    if let Some(idx) = index_scalar_from_value(v).await? {
                                        if idx < 1 {
                                            vm_bail!(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds"
                                            ));
                                        }
                                        selectors.push(Sel::Scalar(idx as usize));
                                    } else {
                                        match v {
                                            Value::Tensor(idx_t) => {
                                                let dim_len = *t.shape.get(d).unwrap_or(&1);
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
                                                            vm_bail!(mex(
                                                                "IndexOutOfBounds",
                                                                "Index out of bounds"
                                                            ));
                                                        }
                                                        v.push(idx as usize);
                                                    }
                                                    selectors.push(Sel::Indices(v));
                                                }
                                            }
                                            _ => vm_bail!(mex(
                                                "UnsupportedIndexType",
                                                "Unsupported index type"
                                            )),
                                        }
                                    }
                                }
                            }
                            // 2-D write fast paths (full column/row) with strict broadcast checks
                            if dims == 2 {
                                let rows = if rank >= 1 { t.shape[0] } else { 1 };
                                let cols = if rank >= 2 { t.shape[1] } else { 1 };
                                match (&selectors[0], &selectors[1]) {
                                    // A(:, j) = rhs
                                    (Sel::Colon, Sel::Scalar(j)) => {
                                        let j0 = *j - 1;
                                        // Size growth semantics: extend columns if needed
                                        if j0 >= cols {
                                            let new_cols = j0 + 1;
                                            let new_rows = rows;
                                            let mut new_data = vec![0.0f64; new_rows * new_cols];
                                            for c in 0..cols {
                                                let src_off = c * rows;
                                                let dst_off = c * new_rows;
                                                new_data[dst_off..dst_off + rows].copy_from_slice(
                                                    &t.data[src_off..src_off + rows],
                                                );
                                            }
                                            t.data = new_data;
                                            t.shape = vec![new_rows, new_cols];
                                        }
                                        let start = j0 * rows;
                                        // F5: try provider-side contig column scatter to avoid host round-trip on next writes (future optimization)
                                        match rhs {
                                            Value::Num(v) => {
                                                for r in 0..rows {
                                                    t.data[start + r] = v;
                                                }
                                            }
                                            Value::Tensor(rt) => {
                                                let len = rt.data.len();
                                                if len == rows {
                                                    for r in 0..rows {
                                                        t.data[start + r] = rt.data[r];
                                                    }
                                                } else if len == 1 {
                                                    for r in 0..rows {
                                                        t.data[start + r] = rt.data[0];
                                                    }
                                                } else {
                                                    vm_bail!("shape mismatch for slice assign"
                                                        .to_string());
                                                }
                                            }
                                            _ => {
                                                vm_bail!("rhs must be numeric or tensor".to_string())
                                            }
                                        }
                                        let view = runmat_accelerate_api::HostTensorView {
                                            data: &t.data,
                                            shape: &t.shape,
                                        };
                                        let new_h = provider.upload(&view).map_err(|e| {
                                            format!("reupload after slice assign: {e}")
                                        })?;
                                        stack.push(Value::GpuTensor(new_h));
                                        bench_end("StoreSlice2D.fast_col", __b);
                                        pc += 1;
                                        continue;
                                    }
                                    // A(i, :) = rhs
                                    (Sel::Scalar(i), Sel::Colon) => {
                                        let i0 = *i - 1;
                                        // Size growth semantics: extend rows if needed
                                        if i0 >= rows {
                                            let new_rows = i0 + 1;
                                            let new_cols = cols;
                                            let mut new_data = vec![0.0f64; new_rows * new_cols];
                                            for c in 0..cols {
                                                for r in 0..rows {
                                                    new_data[r + c * new_rows] =
                                                        t.data[r + c * rows];
                                                }
                                            }
                                            t.data = new_data;
                                            t.shape = vec![new_rows, new_cols];
                                        }
                                        // F5: try provider-side contig row scatter (future optimization)
                                        match rhs {
                                            Value::Num(v) => {
                                                for c in 0..cols {
                                                    t.data[i0 + c * rows] = v;
                                                }
                                            }
                                            Value::Tensor(rt) => {
                                                let len = rt.data.len();
                                                if len == cols {
                                                    for c in 0..cols {
                                                        t.data[i0 + c * rows] = rt.data[c];
                                                    }
                                                } else if len == 1 {
                                                    for c in 0..cols {
                                                        t.data[i0 + c * rows] = rt.data[0];
                                                    }
                                                } else {
                                                    vm_bail!("shape mismatch for slice assign"
                                                        .to_string());
                                                }
                                            }
                                            _ => {
                                                vm_bail!("rhs must be numeric or tensor".to_string())
                                            }
                                        }
                                        let view = runmat_accelerate_api::HostTensorView {
                                            data: &t.data,
                                            shape: &t.shape,
                                        };
                                        let new_h = provider.upload(&view).map_err(|e| {
                                            format!("reupload after slice assign: {e}")
                                        })?;
                                        stack.push(Value::GpuTensor(new_h));
                                        bench_end("StoreSlice2D.fast_row", __b);
                                        pc += 1;
                                        continue;
                                    }
                                    _ => {}
                                }
                            }
                            // Generic N-D writer path (GPU gather-mutate-reupload)
                            // Build per-dim index lists and strides
                            let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                            let full_shape: Vec<usize> = if rank < dims {
                                let mut s = t.shape.clone();
                                s.resize(dims, 1);
                                s
                            } else {
                                t.shape.clone()
                            };
                            for d in 0..dims {
                                let dim_len = full_shape[d];
                                let idxs = match &selectors[d] {
                                    Sel::Colon => (1..=dim_len).collect(),
                                    Sel::Scalar(i) => vec![*i],
                                    Sel::Indices(v) => v.clone(),
                                };
                                if idxs.iter().any(|&i| i == 0 || i > dim_len) {
                                    vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                                }
                                per_dim_indices.push(idxs);
                            }
                            // Column-major strides (first dimension fastest)
                            let mut strides: Vec<usize> = vec![0; dims];
                            let mut acc = 1usize;
                            for d in 0..dims {
                                strides[d] = acc;
                                acc *= full_shape[d];
                            }
                            let total_out: usize =
                                per_dim_indices.iter().map(|v| v.len()).product();
                            // Prepare RHS values
                            enum RhsView {
                                Scalar(f64),
                                Tensor {
                                    data: Vec<f64>,
                                    shape: Vec<usize>,
                                    strides: Vec<usize>,
                                },
                            }
                            let rhs_view = match rhs {
                                Value::Num(n) => RhsView::Scalar(n),
                                Value::Tensor(rt) => {
                                    // Allow exact match or N-D broadcasting where rhs_dim is 1 or equals out_dim
                                    let mut shape = rt.shape.clone();
                                    if shape.len() < dims {
                                        shape.resize(dims, 1);
                                    }
                                    if shape.len() > dims {
                                        if shape.iter().skip(dims).any(|&s| s != 1) {
                                            vm_bail!("shape mismatch for slice assign".to_string());
                                        }
                                        shape.truncate(dims);
                                    }
                                    let mut ok = true;
                                    for d in 0..dims {
                                        let out_len = per_dim_indices[d].len();
                                        let rhs_len = shape[d];
                                        if !(rhs_len == 1 || rhs_len == out_len) {
                                            ok = false;
                                            break;
                                        }
                                    }
                                    if !ok {
                                        vm_bail!("shape mismatch for slice assign".to_string());
                                    }
                                    let mut rstrides = vec![0usize; dims];
                                    let mut racc = 1usize;
                                    for d in 0..dims {
                                        rstrides[d] = racc;
                                        racc *= shape[d];
                                    }
                                    RhsView::Tensor {
                                        data: rt.data,
                                        shape,
                                        strides: rstrides,
                                    }
                                }
                                _ => vm_bail!("rhs must be numeric or tensor".to_string()),
                            };
                            // Iterate and scatter
                            let mut _k = 0usize;
                            let mut idx = vec![0usize; dims];
                            if total_out == 0 {
                                let view = runmat_accelerate_api::HostTensorView {
                                    data: &t.data,
                                    shape: &t.shape,
                                };
                                let new_h = provider
                                    .upload(&view)
                                    .map_err(|e| format!("reupload after slice assign: {e}"))?;
                                stack.push(Value::GpuTensor(new_h));
                            } else {
                                loop {
                                    let mut lin = 0usize;
                                    for d in 0..dims {
                                        let i0 = per_dim_indices[d][idx[d]] - 1;
                                        lin += i0 * strides[d];
                                    }
                                    match &rhs_view {
                                        RhsView::Scalar(val) => t.data[lin] = *val,
                                        RhsView::Tensor {
                                            data,
                                            shape,
                                            strides,
                                        } => {
                                            let mut rlin = 0usize;
                                            for d in 0..dims {
                                                let rhs_len = shape[d];
                                                let pos = if rhs_len == 1 { 0 } else { idx[d] };
                                                rlin += pos * strides[d];
                                            }
                                            t.data[lin] = data[rlin];
                                        }
                                    }
                                    _k += 1;
                                    // Increment first dim fastest
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
                                let view = runmat_accelerate_api::HostTensorView {
                                    data: &t.data,
                                    shape: &t.shape,
                                };
                                let new_h = provider
                                    .upload(&view)
                                    .map_err(|e| format!("reupload after slice assign: {e}"))?;
                                stack.push(Value::GpuTensor(new_h));
                            }
                        }
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
                        let rhs_view = build_complex_rhs_view(&rhs, &plan.selection_lengths)
                            .map_err(|e| format!("slice assign: {e}"))?;
                        scatter_complex_with_plan(&mut ct, &plan, &rhs_view)
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
                        let rhs_view = build_string_rhs_view(&rhs, &plan.selection_lengths)
                            .map_err(|e| format!("slice assign: {e}"))?;
                        scatter_string_with_plan(&mut sa, &plan, &rhs_view)
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
                        let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        let mut rp_iter = 0usize;
                        for d in 0..dims {
                            if let Some(pos) = range_dims.iter().position(|&rd| rd == d) {
                                let (raw_st, raw_sp) = range_params[rp_iter];
                                let dim_len = *t.shape.get(d).unwrap_or(&1);
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
                                let step_i = if sp >= 0.0 {
                                    sp as i64
                                } else {
                                    -(sp.abs() as i64)
                                };
                                selectors.push(Sel::Range {
                                    start: st as i64,
                                    step: step_i,
                                    end_off: range_end_exprs[pos].clone(),
                                });
                                continue;
                            }
                            let is_colon = (colon_mask & (1u32 << d)) != 0;
                            let is_end = (end_mask & (1u32 << d)) != 0;
                            if is_colon {
                                selectors.push(Sel::Colon);
                                continue;
                            }
                            if is_end {
                                let dim_len = if dims == 1 {
                                    total_len_from_shape(&t.shape)
                                } else {
                                    *t.shape.get(d).unwrap_or(&1)
                                };
                                selectors.push(Sel::Scalar(dim_len));
                                continue;
                            }
                            let v = numeric
                                .get(num_iter)
                                .ok_or(mex("MissingNumericIndex", "missing numeric index"))?;
                            num_iter += 1;
                            if let Some(idx) = index_scalar_from_value(v).await? {
                                if idx < 1 {
                                    vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                                }
                                selectors.push(Sel::Scalar(idx as usize));
                            } else {
                                match v {
                                    Value::Tensor(idx_t) => {
                                        let dim_len = if dims == 1 {
                                            total_len_from_shape(&t.shape)
                                        } else {
                                            *t.shape.get(d).unwrap_or(&1)
                                        };
                                        let len = idx_t.shape.iter().product::<usize>();
                                        let mut vi = Vec::with_capacity(len);
                                        for &val in &idx_t.data {
                                            let idx = val as isize;
                                            if idx < 1 || (idx as usize) > dim_len {
                                                vm_bail!(mex(
                                                    "IndexOutOfBounds",
                                                    "Index out of bounds"
                                                ));
                                            }
                                            vi.push(idx as usize);
                                        }
                                        selectors.push(Sel::Indices(vi));
                                    }
                                    _ => {
                                        vm_bail!(mex(
                                            "UnsupportedIndexType",
                                            "Unsupported index type"
                                        ))
                                    }
                                }
                            }
                        }
                        let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                        let mut selection_lengths: Vec<usize> = Vec::with_capacity(dims);
                        let mut scalar_mask: Vec<bool> = Vec::with_capacity(dims);
                        let full_shape: Vec<usize> = if t.shape.len() < dims {
                            let mut s = t.shape.clone();
                            s.resize(dims, 1);
                            s
                        } else {
                            t.shape.clone()
                        };
                        for (d, sel) in selectors.iter().enumerate().take(dims) {
                            let dim_len = full_shape[d];
                            let idxs = match sel {
                                Sel::Colon => (1..=dim_len).collect::<Vec<usize>>(),
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
                                        dim_len,
                                        end_off,
                                        &mut vars,
                                        &context.functions,
                                    )
                                    .await?;
                                    let stp = *step;
                                    if stp == 0 {
                                        vm_bail!(mex("IndexStepZero", "Index step cannot be zero"));
                                    }
                                    if stp > 0 {
                                        while cur <= end_i {
                                            if cur < 1 || cur > dim_len as i64 {
                                                break;
                                            }
                                            v.push(cur as usize);
                                            cur += stp;
                                        }
                                    } else {
                                        while cur >= end_i {
                                            if cur < 1 || cur > dim_len as i64 {
                                                break;
                                            }
                                            v.push(cur as usize);
                                            cur += stp;
                                        }
                                    }
                                    v
                                }
                            };
                            if idxs.iter().any(|&i| i == 0 || i > dim_len) {
                                vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            selection_lengths.push(idxs.len());
                            per_dim_indices.push(idxs);
                            scalar_mask.push(matches!(sel, Sel::Scalar(_)));
                        }
                        if per_dim_indices.iter().any(|v| v.is_empty()) {
                            stack.push(Value::ComplexTensor(t));
                        } else {
                            let mut strides: Vec<usize> = vec![0; dims];
                            let mut acc = 1usize;
                            for (d, stride) in strides.iter_mut().enumerate().take(dims) {
                                *stride = acc;
                                acc *= full_shape[d];
                            }
                            let total_out: usize =
                                per_dim_indices.iter().map(|v| v.len()).product();
                            let mut indices: Vec<u32> = Vec::with_capacity(total_out);
                            cartesian_product(&per_dim_indices, |multi| {
                                let mut lin = 0usize;
                                for d in 0..dims {
                                    let i0 = multi[d] - 1;
                                    lin += i0 * strides[d];
                                }
                                indices.push(lin as u32);
                            });
                            let plan = SlicePlan {
                                indices,
                                output_shape: matlab_squeezed_shape(
                                    &selection_lengths,
                                    &scalar_mask,
                                ),
                                selection_lengths,
                                dims,
                            };
                            let rhs_view = build_complex_rhs_view(&rhs, &plan.selection_lengths)
                                .map_err(|e| format!("slice assign: {e}"))?;
                            scatter_complex_with_plan(&mut t, &plan, &rhs_view)
                                .map_err(|e| format!("slice assign: {e}"))?;
                            stack.push(Value::ComplexTensor(t));
                        }
                    }
                    Value::Tensor(mut t) => {
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
                        let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        let mut rp_iter = 0usize;
                        for d in 0..dims {
                            if let Some(pos) = range_dims.iter().position(|&rd| rd == d) {
                                let (raw_st, raw_sp) = range_params[rp_iter];
                                let dim_len = *t.shape.get(d).unwrap_or(&1);
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
                                let step_i = if sp >= 0.0 {
                                    sp as i64
                                } else {
                                    -(sp.abs() as i64)
                                };
                                selectors.push(Sel::Range {
                                    start: st as i64,
                                    step: step_i,
                                    end_off: range_end_exprs[pos].clone(),
                                });
                                continue;
                            }
                            let is_colon = (colon_mask & (1u32 << d)) != 0;
                            let is_end = (end_mask & (1u32 << d)) != 0;
                            if is_colon {
                                selectors.push(Sel::Colon);
                                continue;
                            }
                            if is_end {
                                let dim_len = if dims == 1 {
                                    total_len_from_shape(&t.shape)
                                } else {
                                    *t.shape.get(d).unwrap_or(&1)
                                };
                                selectors.push(Sel::Scalar(dim_len));
                                continue;
                            }
                            let v = numeric
                                .get(num_iter)
                                .ok_or(mex("MissingNumericIndex", "missing numeric index"))?;
                            num_iter += 1;
                            if let Some(idx) = index_scalar_from_value(v).await? {
                                if idx < 1 {
                                    vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                                }
                                selectors.push(Sel::Scalar(idx as usize));
                            } else {
                                match v {
                                    Value::Tensor(idx_t) => {
                                        let dim_len = if dims == 1 {
                                            total_len_from_shape(&t.shape)
                                        } else {
                                            *t.shape.get(d).unwrap_or(&1)
                                        };
                                        let len = idx_t.shape.iter().product::<usize>();
                                        let mut vi = Vec::with_capacity(len);
                                        for &val in &idx_t.data {
                                            let idx = val as isize;
                                            if idx < 1 || (idx as usize) > dim_len {
                                                vm_bail!(mex(
                                                    "IndexOutOfBounds",
                                                    "Index out of bounds"
                                                ));
                                            }
                                            vi.push(idx as usize);
                                        }
                                        selectors.push(Sel::Indices(vi));
                                    }
                                    _ => {
                                        vm_bail!(mex(
                                            "UnsupportedIndexType",
                                            "Unsupported index type"
                                        ))
                                    }
                                }
                            }
                        }
                        // Build index lists and scatter rhs with broadcasting
                        // debug removed
                        let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                        for (d, sel) in selectors.iter().enumerate().take(dims) {
                            let dim_len = if dims == 1 {
                                total_len_from_shape(&t.shape)
                            } else {
                                *t.shape.get(d).unwrap_or(&1)
                            };
                            let idxs = match sel {
                                Sel::Colon => (1..=dim_len).collect::<Vec<usize>>(),
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
                                        dim_len,
                                        end_off,
                                        &mut vars,
                                        &context.functions,
                                    )
                                    .await?;
                                    let stp = *step;
                                    if stp == 0 {
                                        vm_bail!(mex("IndexStepZero", "Index step cannot be zero"));
                                    }
                                    if stp > 0 {
                                        while cur <= end_i {
                                            if cur < 1 || cur > dim_len as i64 {
                                                break;
                                            }
                                            v.push(cur as usize);
                                            cur += stp;
                                        }
                                    } else {
                                        while cur >= end_i {
                                            if cur < 1 || cur > dim_len as i64 {
                                                break;
                                            }
                                            v.push(cur as usize);
                                            cur += stp;
                                        }
                                    }
                                    v
                                }
                            };
                            if idxs.iter().any(|&i| i == 0 || i > dim_len) {
                                vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            per_dim_indices.push(idxs);
                        }
                        let mut strides: Vec<usize> = vec![0; dims];
                        let mut acc = 1usize;
                        for (d, stride) in strides.iter_mut().enumerate().take(dims) {
                            *stride = acc;
                            acc *= *t.shape.get(d).unwrap_or(&1);
                        }
                        let selection_empty = per_dim_indices.iter().any(|v| v.is_empty());
                        if selection_empty {
                            stack.push(Value::Tensor(t));
                        } else {
                            // Build broadcasting view for RHS with per-dimension shape
                            enum RhsView {
                                Scalar(f64),
                                Tensor {
                                    data: Vec<f64>,
                                    shape: Vec<usize>,
                                    strides: Vec<usize>,
                                },
                            }
                            let rhs_view = match rhs {
                                Value::Num(n) => RhsView::Scalar(n),
                                Value::Tensor(rt) => {
                                    if rt.data.is_empty() {
                                        vm_bail!("shape mismatch for slice assign".to_string());
                                    }
                                    // Normalize RHS shape to dims by padding with ones or validating extra dims are ones
                                    let mut rshape = if dims == 1 {
                                        vec![rt.data.len()]
                                    } else {
                                        rt.shape.clone()
                                    };
                                    if rshape.len() < dims {
                                        rshape.resize(dims, 1);
                                    }
                                    if rshape.len() > dims {
                                        if rshape.iter().skip(dims).any(|&s| s != 1) {
                                            vm_bail!("shape mismatch for slice assign".to_string());
                                        }
                                        rshape.truncate(dims);
                                    }
                                    // Validate broadcasting compatibility
                                    for d in 0..dims {
                                        let out_len = per_dim_indices[d].len();
                                        let rhs_len = rshape[d];
                                        if !(rhs_len == 1 || rhs_len == out_len) {
                                            vm_bail!("shape mismatch for slice assign".to_string());
                                        }
                                    }
                                    // Build column-major strides for RHS
                                    let mut rstrides = vec![0usize; dims];
                                    let mut racc = 1usize;
                                    for d in 0..dims {
                                        rstrides[d] = racc;
                                        racc *= rshape[d];
                                    }
                                    if racc != rt.data.len() {
                                        vm_bail!("shape mismatch for slice assign".to_string());
                                    }
                                    RhsView::Tensor {
                                        data: rt.data,
                                        shape: rshape,
                                        strides: rstrides,
                                    }
                                }
                                _ => vm_bail!("rhs must be numeric or tensor".to_string()),
                            };
                            // Precompute mapping from absolute index to position-in-selection per dimension to ensure column-major consistent mapping
                            use std::collections::HashMap;
                            let mut pos_maps: Vec<HashMap<usize, usize>> = Vec::with_capacity(dims);
                            for dim_idxs in per_dim_indices.iter().take(dims) {
                                let mut m: HashMap<usize, usize> = HashMap::new();
                                for (p, &idx) in dim_idxs.iter().enumerate() {
                                    m.insert(idx, p);
                                }
                                pos_maps.push(m);
                            }
                            fn cartesian2<F: FnMut(&[usize])>(lists: &[Vec<usize>], mut f: F) {
                                let dims = lists.len();
                                let mut idx = vec![0usize; dims];
                                loop {
                                    let cur: Vec<usize> =
                                        (0..dims).map(|d| lists[d][idx[d]]).collect();
                                    f(&cur);
                                    let mut d = 0usize;
                                    while d < dims {
                                        idx[d] += 1;
                                        if idx[d] < lists[d].len() {
                                            break;
                                        }
                                        idx[d] = 0;
                                        d += 1;
                                    }
                                    if d == dims {
                                        break;
                                    }
                                }
                            }
                            // debug removed
                            let mut err_opt: Option<String> = None;
                            let mut _debug_count = 0usize;
                            cartesian2(&per_dim_indices, |multi| {
                                if err_opt.is_some() {
                                    return;
                                }
                                let mut lin = 0usize;
                                for d in 0..dims {
                                    let i0 = multi[d] - 1;
                                    lin += i0 * strides[d];
                                }
                                match &rhs_view {
                                    RhsView::Scalar(val) => t.data[lin] = *val,
                                    RhsView::Tensor {
                                        data,
                                        shape,
                                        strides: rstrides,
                                    } => {
                                        // Map selection coordinate to RHS coordinate with broadcasting
                                        let mut rlin = 0usize;
                                        for d in 0..dims {
                                            let rhs_len = shape[d];
                                            let pos_in_dim = if rhs_len == 1 {
                                                0
                                            } else {
                                                *pos_maps[d].get(&multi[d]).unwrap_or(&0)
                                            };
                                            rlin += pos_in_dim * rstrides[d];
                                        }
                                        if rlin >= data.len() {
                                            err_opt =
                                                Some("shape mismatch for slice assign".to_string());
                                            return;
                                        }
                                        t.data[lin] = data[rlin];
                                    }
                                }
                            });
                            let _ = (t.data.first(), t.data.len());
                            if let Some(e) = err_opt {
                                vm_bail!(e);
                            }
                            stack.push(Value::Tensor(t));
                        }
                    }
                    Value::GpuTensor(h) => {
                        let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                            mex(
                                "AccelerationProviderUnavailable",
                                "No acceleration provider registered",
                            )
                        })?;
                        let host = provider
                            .download(&h)
                            .await
                            .map_err(|e| format!("gather for range-end assign: {e}"))?;
                        let mut t = runmat_builtins::Tensor::new(host.data, host.shape)
                            .map_err(|e| format!("range-end assign: {e}"))?;
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
                        let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        let mut rp_iter = 0usize;
                        for d in 0..dims {
                            if let Some(pos) = range_dims.iter().position(|&rd| rd == d) {
                                let (raw_st, raw_sp) = range_params[rp_iter];
                                let dim_len = *t.shape.get(d).unwrap_or(&1);
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
                                let step_i = if sp >= 0.0 {
                                    sp as i64
                                } else {
                                    -(sp.abs() as i64)
                                };
                                selectors.push(Sel::Range {
                                    start: st as i64,
                                    step: step_i,
                                    end_off: range_end_exprs[pos].clone(),
                                });
                                continue;
                            }
                            let is_colon = (colon_mask & (1u32 << d)) != 0;
                            let is_end = (end_mask & (1u32 << d)) != 0;
                            if is_colon {
                                selectors.push(Sel::Colon);
                                continue;
                            }
                            if is_end {
                                selectors.push(Sel::Scalar(*t.shape.get(d).unwrap_or(&1)));
                                continue;
                            }
                            let v = numeric
                                .get(num_iter)
                                .ok_or(mex("MissingNumericIndex", "missing numeric index"))?;
                            num_iter += 1;
                            if let Some(idx) = index_scalar_from_value(v).await? {
                                if idx < 1 {
                                    vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                                }
                                selectors.push(Sel::Scalar(idx as usize));
                            } else {
                                match v {
                                    Value::Tensor(idx_t) => {
                                        let dim_len = *t.shape.get(d).unwrap_or(&1);
                                        let len = idx_t.shape.iter().product::<usize>();
                                        if len == dim_len {
                                            let mut vi = Vec::new();
                                            for (i, &val) in idx_t.data.iter().enumerate() {
                                                if val != 0.0 {
                                                    vi.push(i + 1);
                                                }
                                            }
                                            selectors.push(Sel::Indices(vi));
                                        } else {
                                            let mut vi = Vec::with_capacity(len);
                                            for &val in &idx_t.data {
                                                let idx = val as isize;
                                                if idx < 1 {
                                                    vm_bail!(mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds"
                                                    ));
                                                }
                                                vi.push(idx as usize);
                                            }
                                            selectors.push(Sel::Indices(vi));
                                        }
                                    }
                                    _ => {
                                        vm_bail!(mex(
                                            "UnsupportedIndexType",
                                            "Unsupported index type"
                                        ))
                                    }
                                }
                            }
                        }
                        // Build index lists and scatter rhs with broadcasting
                        // debug removed
                        let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                        for (d, sel) in selectors.iter().enumerate().take(dims) {
                            let dim_len = if dims == 1 {
                                total_len_from_shape(&t.shape)
                            } else {
                                *t.shape.get(d).unwrap_or(&1)
                            };
                            let idxs = match sel {
                                Sel::Colon => (1..=dim_len).collect::<Vec<usize>>(),
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
                                        dim_len,
                                        end_off,
                                        &mut vars,
                                        &context.functions,
                                    )
                                    .await?;
                                    let stp = *step;
                                    if stp == 0 {
                                        vm_bail!(mex("IndexStepZero", "Index step cannot be zero"));
                                    }
                                    if stp > 0 {
                                        while cur <= end_i {
                                            if cur < 1 || cur > dim_len as i64 {
                                                break;
                                            }
                                            v.push(cur as usize);
                                            cur += stp;
                                        }
                                    } else {
                                        while cur >= end_i {
                                            if cur < 1 || cur > dim_len as i64 {
                                                break;
                                            }
                                            v.push(cur as usize);
                                            cur += stp;
                                        }
                                    }
                                    v
                                }
                            };
                            if idxs.iter().any(|&i| i == 0 || i > dim_len) {
                                vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            per_dim_indices.push(idxs);
                        }
                        let mut strides: Vec<usize> = vec![0; dims];
                        let mut acc = 1usize;
                        for (d, stride) in strides.iter_mut().enumerate().take(dims) {
                            *stride = acc;
                            acc *= *t.shape.get(d).unwrap_or(&1);
                        }
                        let selection_empty = per_dim_indices.iter().any(|v| v.is_empty());
                        if selection_empty {
                            let view = runmat_accelerate_api::HostTensorView {
                                data: &t.data,
                                shape: &t.shape,
                            };
                            let new_h = provider
                                .upload(&view)
                                .map_err(|e| format!("reupload after range-end assign: {e}"))?;
                            stack.push(Value::GpuTensor(new_h));
                        } else {
                            // Build broadcasting view for RHS with per-dimension shape
                            enum RhsView {
                                Scalar(f64),
                                Tensor {
                                    data: Vec<f64>,
                                    shape: Vec<usize>,
                                    strides: Vec<usize>,
                                },
                            }
                            let rhs_view = match rhs {
                                Value::Num(n) => RhsView::Scalar(n),
                                Value::Tensor(rt) => {
                                    if rt.data.is_empty() {
                                        vm_bail!("shape mismatch for slice assign".to_string());
                                    }
                                    // Normalize RHS shape to dims by padding with ones or validating extra dims are ones
                                    let mut rshape = if dims == 1 {
                                        vec![rt.data.len()]
                                    } else {
                                        rt.shape.clone()
                                    };
                                    if rshape.len() < dims {
                                        rshape.resize(dims, 1);
                                    }
                                    if rshape.len() > dims {
                                        if rshape.iter().skip(dims).any(|&s| s != 1) {
                                            vm_bail!("shape mismatch for slice assign".to_string());
                                        }
                                        rshape.truncate(dims);
                                    }
                                    // Validate broadcasting compatibility
                                    for d in 0..dims {
                                        let out_len = per_dim_indices[d].len();
                                        let rhs_len = rshape[d];
                                        if !(rhs_len == 1 || rhs_len == out_len) {
                                            vm_bail!("shape mismatch for slice assign".to_string());
                                        }
                                    }
                                    // Build column-major strides for RHS
                                    let mut rstrides = vec![0usize; dims];
                                    let mut racc = 1usize;
                                    for d in 0..dims {
                                        rstrides[d] = racc;
                                        racc *= rshape[d];
                                    }
                                    if racc != rt.data.len() {
                                        vm_bail!("shape mismatch for slice assign".to_string());
                                    }
                                    RhsView::Tensor {
                                        data: rt.data,
                                        shape: rshape,
                                        strides: rstrides,
                                    }
                                }
                                _ => vm_bail!("rhs must be numeric or tensor".to_string()),
                            };
                            // Precompute mapping from absolute index to position-in-selection per dimension to ensure column-major consistent mapping
                            use std::collections::HashMap;
                            let mut pos_maps: Vec<HashMap<usize, usize>> = Vec::with_capacity(dims);
                            for dim_idxs in per_dim_indices.iter().take(dims) {
                                let mut m: HashMap<usize, usize> = HashMap::new();
                                for (p, &idx) in dim_idxs.iter().enumerate() {
                                    m.insert(idx, p);
                                }
                                pos_maps.push(m);
                            }
                            // Iterate selection cartesian and scatter
                            let mut err_opt: Option<String> = None;
                            // Local cartesian iterator
                            fn cartesian2<F: FnMut(&[usize])>(lists: &[Vec<usize>], mut f: F) {
                                let dims = lists.len();
                                let mut idx = vec![0usize; dims];
                                loop {
                                    let cur: Vec<usize> =
                                        (0..dims).map(|d| lists[d][idx[d]]).collect();
                                    f(&cur);
                                    let mut d = 0usize;
                                    while d < dims {
                                        idx[d] += 1;
                                        if idx[d] < lists[d].len() {
                                            break;
                                        }
                                        idx[d] = 0;
                                        d += 1;
                                    }
                                    if d == dims {
                                        break;
                                    }
                                }
                            }
                            cartesian2(&per_dim_indices, |multi| {
                                if err_opt.is_some() {
                                    return;
                                }
                                let mut lin = 0usize;
                                for d in 0..dims {
                                    let i0 = multi[d] - 1;
                                    lin += i0 * strides[d];
                                }
                                match &rhs_view {
                                    RhsView::Scalar(val) => t.data[lin] = *val,
                                    RhsView::Tensor {
                                        data,
                                        shape,
                                        strides: rstrides,
                                    } => {
                                        let mut rlin = 0usize;
                                        for d in 0..dims {
                                            let rhs_len = shape[d];
                                            let pos_in_dim = if rhs_len == 1 {
                                                0
                                            } else {
                                                *pos_maps[d].get(&multi[d]).unwrap_or(&0)
                                            };
                                            rlin += pos_in_dim * rstrides[d];
                                        }
                                        if rlin >= data.len() {
                                            err_opt =
                                                Some("shape mismatch for slice assign".to_string());
                                            return;
                                        }
                                        t.data[lin] = data[rlin];
                                    }
                                }
                            });
                            if let Some(e) = err_opt {
                                vm_bail!(e);
                            }
                            let view = runmat_accelerate_api::HostTensorView {
                                data: &t.data,
                                shape: &t.shape,
                            };
                            let new_h = provider
                                .upload(&view)
                                .map_err(|e| format!("reupload after range-end assign: {e}"))?;
                            stack.push(Value::GpuTensor(new_h));
                        }
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
                let cell = runmat_runtime::make_cell_with_shape(elems, vec![rows, cols])
                    .map_err(|e| format!("Cell creation error: {e}"))?;
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
                    Value::Cell(ca) => match indices.len() {
                        1 => {
                            let i = indices[0];
                            if i == 0 || i > ca.data.len() {
                                return Err(mex(
                                    "CellIndexOutOfBounds",
                                    "Cell index out of bounds",
                                ));
                            }
                            stack.push((*ca.data[i - 1]).clone());
                        }
                        2 => {
                            let r = indices[0];
                            let c = indices[1];
                            if r == 0 || r > ca.rows || c == 0 || c > ca.cols {
                                return Err(mex(
                                    "CellSubscriptOutOfBounds",
                                    "Cell subscript out of bounds",
                                ));
                            }
                            stack.push((*ca.data[(r - 1) * ca.cols + (c - 1)]).clone());
                        }
                        _ => {
                            return Err(mex(
                                "UnsupportedCellIndexCount",
                                "Unsupported number of cell indices",
                            ))
                        }
                    },
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
                        // Expand in column-major order up to out_count elements
                        let mut values: Vec<Value> = Vec::new();
                        if indices.is_empty() {
                            // Expand all elements in column-major order
                            values.extend(ca.data.iter().map(|p| (*(*p)).clone()));
                        } else {
                            match indices.len() {
                                1 => {
                                    let i = indices[0];
                                    if i == 0 || i > ca.data.len() {
                                        return Err(mex(
                                            "CellIndexOutOfBounds",
                                            "Cell index out of bounds",
                                        ));
                                    }
                                    values.push((*ca.data[i - 1]).clone());
                                }
                                2 => {
                                    let r = indices[0];
                                    let c = indices[1];
                                    if r == 0 || r > ca.rows || c == 0 || c > ca.cols {
                                        return Err(mex(
                                            "CellSubscriptOutOfBounds",
                                            "Cell subscript out of bounds",
                                        ));
                                    }
                                    values.push((*ca.data[(r - 1) * ca.cols + (c - 1)]).clone());
                                }
                                _ => {
                                    return Err(mex(
                                        "UnsupportedCellIndexCount",
                                        "Unsupported number of cell indices",
                                    ))
                                }
                            }
                        }
                        // Pad or truncate to out_count
                        if values.len() >= out_count {
                            for v in values.iter().take(out_count) {
                                stack.push(v.clone());
                            }
                        } else {
                            for v in &values {
                                stack.push(v.clone());
                            }
                            for _ in values.len()..out_count {
                                stack.push(Value::Num(0.0));
                            }
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
                let value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                if out_count == 0 {
                    pc += 1;
                    continue;
                }
                match value {
                    Value::OutputList(values) => {
                        for i in 0..out_count {
                            if let Some(v) = values.get(i) {
                                stack.push(v.clone());
                            } else {
                                stack.push(Value::Num(0.0));
                            }
                        }
                    }
                    other => {
                        stack.push(other);
                        for _ in 1..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                }
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
                    Value::Tensor(mut t) => {
                        // Helper to coerce RHS to scalar f64, supporting 1x1 tensors and gpu tensors
                        async fn rhs_to_scalar(rhs: &Value) -> Result<f64, RuntimeError> {
                            match rhs {
                                Value::Num(x) => Ok(*x),
                                Value::Tensor(t2) => {
                                    if t2.data.len() == 1 {
                                        Ok(t2.data[0])
                                    } else {
                                        Err(mex("ScalarRequired", "RHS must be scalar"))
                                    }
                                }
                                Value::GpuTensor(h2) => {
                                    let total = h2.shape.iter().copied().product::<usize>();
                                    if total != 1 {
                                        return Err(mex("ScalarRequired", "RHS must be scalar"));
                                    }
                                    if let Some(p) = runmat_accelerate_api::provider() {
                                        let host = p
                                            .download(h2)
                                            .await
                                            .map_err(|e| format!("gather rhs: {e}"))?;
                                        Ok(host.data[0])
                                    } else {
                                        Err(mex(
                                            "AccelerationProviderUnavailable",
                                            "No acceleration provider registered",
                                        ))
                                    }
                                }
                                _ => rhs
                                    .try_into()
                                    .map_err(|_| mex("NumericRequired", "RHS must be numeric")),
                            }
                        }
                        // 1D linear or 2D scalar assignment only for now
                        if indices.len() == 1 {
                            let total = t.rows * t.cols;
                            let idx = indices[0];
                            if idx == 0 || idx > total {
                                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            let val: f64 = rhs_to_scalar(&rhs).await?;
                            t.data[idx - 1] = val;
                            stack.push(Value::Tensor(t));
                        } else if indices.len() == 2 {
                            let i = indices[0];
                            let mut j = indices[1];
                            let rows = t.rows;
                            let cols = t.cols;
                            // Clamp column index within [1..cols] to accommodate end-offset semantics
                            if j == 0 {
                                j = 1;
                            }
                            if j > cols {
                                j = cols;
                            }
                            if i == 0 || i > rows {
                                if std::env::var("RUNMAT_DEBUG_INDEX").as_deref() == Ok("1") {
                                    debug!(
                                        i,
                                        j_clamped = j,
                                        rows,
                                        cols,
                                        shape = ?t.shape,
                                        "[vm] StoreIndex Tensor OOB"
                                    );
                                }
                                return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
                            }
                            let val: f64 = rhs_to_scalar(&rhs).await?;
                            let idx = (i - 1) + (j - 1) * rows;
                            t.data[idx] = val;
                            stack.push(Value::Tensor(t));
                        } else {
                            return Err(mex(
                                "UnsupportedAssignmentRank",
                                "Only 1D/2D scalar assignment supported",
                            ));
                        }
                    }
                    Value::ComplexTensor(mut t) => {
                        if indices.len() == 1 {
                            let total = t.rows * t.cols;
                            let idx = indices[0];
                            if idx == 0 || idx > total {
                                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            let val = value_to_complex_scalar(&rhs).await?;
                            t.data[idx - 1] = val;
                            stack.push(Value::ComplexTensor(t));
                        } else if indices.len() == 2 {
                            let i = indices[0];
                            let mut j = indices[1];
                            let rows = t.rows;
                            let cols = t.cols;
                            if j == 0 {
                                j = 1;
                            }
                            if j > cols {
                                j = cols;
                            }
                            if i == 0 || i > rows {
                                return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
                            }
                            let val = value_to_complex_scalar(&rhs).await?;
                            let idx = (i - 1) + (j - 1) * rows;
                            t.data[idx] = val;
                            stack.push(Value::ComplexTensor(t));
                        } else {
                            return Err(mex(
                                "UnsupportedAssignmentRank",
                                "Only 1D/2D scalar assignment supported",
                            ));
                        }
                    }
                    Value::GpuTensor(h) => {
                        // Stage F1: gather–mutate–reupload for simple 1D/2D scalar assignments
                        let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                            mex(
                                "AccelerationProviderUnavailable",
                                "No acceleration provider registered",
                            )
                        })?;
                        let host = provider
                            .download(&h)
                            .await
                            .map_err(|e| format!("gather for assignment: {e}"))?;
                        let mut t = runmat_builtins::Tensor::new(host.data, host.shape)
                            .map_err(|e| format!("assignment: {e}"))?;
                        // Reuse same scalar coercion
                        async fn rhs_to_scalar(
                            rhs: &Value,
                            provider: &dyn runmat_accelerate_api::AccelProvider,
                        ) -> Result<f64, RuntimeError> {
                            match rhs {
                                Value::Num(x) => Ok(*x),
                                Value::Tensor(t2) => {
                                    if t2.data.len() == 1 {
                                        Ok(t2.data[0])
                                    } else {
                                        Err(mex("ScalarRequired", "RHS must be scalar"))
                                    }
                                }
                                Value::GpuTensor(h2) => {
                                    let total = h2.shape.iter().copied().product::<usize>();
                                    if total != 1 {
                                        return Err(mex("ScalarRequired", "RHS must be scalar"));
                                    }
                                    let host2 = provider
                                        .download(h2)
                                        .await
                                        .map_err(|e| format!("gather rhs: {e}"))?;
                                    Ok(host2.data[0])
                                }
                                _ => rhs
                                    .try_into()
                                    .map_err(|_| mex("NumericRequired", "RHS must be numeric")),
                            }
                        }
                        if indices.len() == 1 {
                            let total = t.rows() * t.cols();
                            let idx = indices[0];
                            if idx == 0 || idx > total {
                                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            let val: f64 = rhs_to_scalar(&rhs, provider).await?;
                            t.data[idx - 1] = val;
                        } else if indices.len() == 2 {
                            let i = indices[0];
                            let mut j = indices[1];
                            let rows = t.rows();
                            let cols = t.cols();
                            // Clamp column index within [1..cols] to accommodate end-offset semantics
                            if j == 0 {
                                j = 1;
                            }
                            if j > cols {
                                j = cols;
                            }
                            if i == 0 || i > rows {
                                if std::env::var("RUNMAT_DEBUG_INDEX").as_deref() == Ok("1") {
                                    debug!(
                                        i,
                                        j_clamped = j,
                                        rows,
                                        cols,
                                        shape = ?t.shape,
                                        "[vm] StoreIndex GpuTensor OOB"
                                    );
                                }
                                return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
                            }
                            let val: f64 = rhs_to_scalar(&rhs, provider).await?;
                            let idx = (i - 1) + (j - 1) * rows;
                            t.data[idx] = val;
                        } else if indices.is_empty() {
                            // Trivial colon slice cases from parser may encode as zero indices; handle full-row/col scalar broadcast
                            let val: f64 = rhs_to_scalar(&rhs, provider).await?;
                            for k in 0..t.data.len() {
                                t.data[k] = val;
                            }
                        } else {
                            return Err(mex(
                                "UnsupportedAssignmentRank",
                                "Only 1D/2D scalar assignment supported",
                            ));
                        }
                        let view = runmat_accelerate_api::HostTensorView {
                            data: &t.data,
                            shape: &t.shape,
                        };
                        let new_h = provider
                            .upload(&view)
                            .map_err(|e| format!("reupload after assignment: {e}"))?;
                        stack.push(Value::GpuTensor(new_h));
                    }
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
                    Value::Cell(mut ca) => match indices.len() {
                        1 => {
                            let i = indices[0];
                            if i == 0 || i > ca.data.len() {
                                return Err(mex(
                                    "CellIndexOutOfBounds",
                                    "Cell index out of bounds",
                                ));
                            }
                            if let Some(oldv) = ca.data.get(i - 1) {
                                runmat_gc::gc_record_write(oldv, &rhs);
                            }
                            *ca.data[i - 1] = rhs;
                            stack.push(Value::Cell(ca));
                        }
                        2 => {
                            let i = indices[0];
                            let j = indices[1];
                            if i == 0 || i > ca.rows || j == 0 || j > ca.cols {
                                return Err(mex(
                                    "CellSubscriptOutOfBounds",
                                    "Cell subscript out of bounds",
                                ));
                            }
                            let lin = (i - 1) * ca.cols + (j - 1);
                            if let Some(oldv) = ca.data.get(lin) {
                                runmat_gc::gc_record_write(oldv, &rhs);
                            }
                            *ca.data[lin] = rhs;
                            stack.push(Value::Cell(ca));
                        }
                        _ => {
                            return Err(mex(
                                "UnsupportedCellIndexCount",
                                "Unsupported number of cell indices",
                            ))
                        }
                    },
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
                match base {
                    Value::Object(obj) => {
                        if let Some((p, _owner)) =
                            runmat_builtins::lookup_property(&obj.class_name, &field)
                        {
                            if p.is_static {
                                vm_bail!(format!(
                                    "Property '{}' is static; use classref('{}').{}",
                                    field, obj.class_name, field
                                ));
                            }
                            if p.get_access == runmat_builtins::Access::Private {
                                vm_bail!(format!("Property '{}' is private", field))
                            }
                            if p.is_dependent {
                                // Call get.<field>(obj)
                                let getter = format!("get.{field}");
                                match call_builtin_vm!(&getter, &[Value::Object(obj.clone())],) {
                                    Ok(v) => {
                                        stack.push(v);
                                        continue;
                                    }
                                    Err(_e) => {}
                                }
                            }
                        }
                        if let Some(v) = obj.properties.get(&field) {
                            stack.push(v.clone());
                        } else if let Some((p2, _)) =
                            runmat_builtins::lookup_property(&obj.class_name, &field)
                        {
                            if p2.is_dependent {
                                let backing = format!("{field}_backing");
                                if let Some(vb) = obj.properties.get(&backing) {
                                    stack.push(vb.clone());
                                    continue;
                                }
                            }
                        } else if let Some(cls) = runmat_builtins::get_class(&obj.class_name) {
                            if cls.methods.contains_key("subsref") {
                                match call_builtin_vm!(
                                    "call_method",
                                    &[
                                        Value::Object(obj),
                                        Value::String("subsref".to_string()),
                                        Value::String(".".to_string()),
                                        Value::String(field),
                                    ],
                                ) {
                                    Ok(v) => stack.push(v),
                                    Err(e) => vm_bail!(e.to_string()),
                                }
                            } else {
                                vm_bail!(format!(
                                    "Undefined property '{}' for class {}",
                                    field, obj.class_name
                                ));
                            }
                        } else {
                            vm_bail!(format!("Unknown class {}", obj.class_name));
                        }
                    }
                    Value::HandleObject(handle) => {
                        match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::HandleObject(handle),
                                Value::String("subsref".to_string()),
                                Value::String(".".to_string()),
                                Value::String(field),
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e.to_string()),
                        }
                    }
                    Value::ClassRef(cls) => {
                        if let Some((p, owner)) = runmat_builtins::lookup_property(&cls, &field) {
                            if !p.is_static {
                                vm_bail!(format!("Property '{}' is not static", field));
                            }
                            if p.get_access == runmat_builtins::Access::Private {
                                vm_bail!(format!("Property '{}' is private", field))
                            }
                            if let Some(v) =
                                runmat_builtins::get_static_property_value(&owner, &field)
                            {
                                stack.push(v);
                            } else if let Some(v) = &p.default_value {
                                stack.push(v.clone());
                            } else {
                                stack.push(Value::Num(0.0));
                            }
                        } else if let Some((m, _owner)) =
                            runmat_builtins::lookup_method(&cls, &field)
                        {
                            if !m.is_static {
                                vm_bail!(format!("Method '{}' is not static", field));
                            }
                            stack.push(Value::Closure(runmat_builtins::Closure {
                                function_name: m.function_name,
                                captures: vec![],
                            }));
                        } else {
                            let qualified = format!("{cls}.{field}");
                            if runmat_builtins::builtin_functions()
                                .iter()
                                .any(|b| b.name == qualified)
                            {
                                stack.push(Value::Closure(runmat_builtins::Closure {
                                    function_name: qualified,
                                    captures: vec![],
                                }));
                            } else {
                                vm_bail!(format!("Unknown property '{}' on class {}", field, cls));
                            }
                        }
                    }
                    Value::Struct(st) => {
                        if let Some(v) = st.fields.get(&field) {
                            stack.push(v.clone());
                        } else if allow_init {
                            stack.push(Value::Struct(runmat_builtins::StructValue::new()));
                        } else {
                            vm_bail!(format!("Undefined field '{}'", field));
                        }
                    }
                    Value::Cell(ca) => {
                        // Extract field from each struct element; build a cell with same shape
                        let mut out: Vec<Value> = Vec::with_capacity(ca.data.len());
                        for v in &ca.data {
                            match &**v {
                                Value::Struct(st) => {
                                    if let Some(fv) = st.fields.get(&field) {
                                        out.push(fv.clone());
                                    } else {
                                        out.push(Value::Num(0.0));
                                    }
                                }
                                other => {
                                    out.push(other.clone());
                                }
                            }
                        }
                        let new_cell = runmat_builtins::CellArray::new(out, ca.rows, ca.cols)
                            .map_err(|e| format!("cell field gather: {e}"))?;
                        stack.push(Value::Cell(new_cell));
                    }
                    Value::MException(mex) => {
                        let value = match field.as_str() {
                            "identifier" => Value::String(mex.identifier.clone()),
                            "message" => Value::String(mex.message.clone()),
                            "stack" => {
                                let values: Vec<Value> =
                                    mex.stack.iter().map(|s| Value::String(s.clone())).collect();
                                let rows = values.len();
                                let cell = runmat_builtins::CellArray::new(values, rows, 1)
                                    .map_err(|e| format!("MException.stack: {e}"))?;
                                Value::Cell(cell)
                            }
                            other => {
                                vm_bail!(format!("Reference to non-existent field '{}'.", other))
                            }
                        };
                        stack.push(value);
                    }
                    _ => vm_bail!("LoadMember on non-object".to_string()),
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
                match base {
                    Value::Object(obj) => {
                        if let Some((p, _owner)) =
                            runmat_builtins::lookup_property(&obj.class_name, &name)
                        {
                            if p.is_static {
                                vm_bail!(format!(
                                    "Property '{}' is static; use classref('{}').{}",
                                    name, obj.class_name, name
                                ));
                            }
                            if p.get_access == runmat_builtins::Access::Private {
                                vm_bail!(format!("Property '{}' is private", name))
                            }
                        }
                        if let Some(v) = obj.properties.get(&name) {
                            stack.push(v.clone());
                        } else if let Some(cls) = runmat_builtins::get_class(&obj.class_name) {
                            if cls.methods.contains_key("subsref") {
                                match call_builtin_vm!(
                                    "call_method",
                                    &[
                                        Value::Object(obj),
                                        Value::String("subsref".to_string()),
                                        Value::String(".".to_string()),
                                        Value::String(name),
                                    ],
                                ) {
                                    Ok(v) => stack.push(v),
                                    Err(e) => vm_bail!(e.to_string()),
                                }
                            } else {
                                vm_bail!(format!(
                                    "Undefined property '{}' for class {}",
                                    name, obj.class_name
                                ));
                            }
                        } else {
                            vm_bail!(format!("Unknown class {}", obj.class_name));
                        }
                    }
                    Value::HandleObject(handle) => {
                        match call_builtin_vm!(
                            "call_method",
                            &[
                                Value::HandleObject(handle),
                                Value::String("subsref".to_string()),
                                Value::String(".".to_string()),
                                Value::String(name),
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e.to_string()),
                        }
                    }
                    Value::ClassRef(cls) => {
                        if let Some((p, owner)) = runmat_builtins::lookup_property(&cls, &name) {
                            if !p.is_static {
                                vm_bail!(format!("Property '{}' is not static", name));
                            }
                            if p.get_access == runmat_builtins::Access::Private {
                                vm_bail!(format!("Property '{}' is private", name))
                            }
                            if let Some(v) =
                                runmat_builtins::get_static_property_value(&owner, &name)
                            {
                                stack.push(v);
                            } else if let Some(v) = &p.default_value {
                                stack.push(v.clone());
                            } else {
                                stack.push(Value::Num(0.0));
                            }
                        } else if let Some((m, _owner)) =
                            runmat_builtins::lookup_method(&cls, &name)
                        {
                            if !m.is_static {
                                vm_bail!(format!("Method '{}' is not static", name));
                            }
                            stack.push(Value::Closure(runmat_builtins::Closure {
                                function_name: m.function_name,
                                captures: vec![],
                            }));
                        } else {
                            let qualified = format!("{cls}.{name}");
                            if runmat_builtins::builtin_functions()
                                .iter()
                                .any(|b| b.name == qualified)
                            {
                                stack.push(Value::Closure(runmat_builtins::Closure {
                                    function_name: qualified,
                                    captures: vec![],
                                }));
                            } else {
                                vm_bail!(format!("Unknown property '{}' on class {}", name, cls));
                            }
                        }
                    }
                    Value::Struct(st) => {
                        if let Some(v) = st.fields.get(&name) {
                            stack.push(v.clone());
                        } else if allow_init {
                            stack.push(Value::Struct(runmat_builtins::StructValue::new()));
                        } else {
                            vm_bail!(format!("Undefined field '{}'", name));
                        }
                    }
                    Value::MException(mex) => {
                        let value = match name.as_str() {
                            "identifier" => Value::String(mex.identifier.clone()),
                            "message" => Value::String(mex.message.clone()),
                            "stack" => {
                                let values: Vec<Value> =
                                    mex.stack.iter().map(|s| Value::String(s.clone())).collect();
                                let rows = values.len();
                                let cell = runmat_builtins::CellArray::new(values, rows, 1)
                                    .map_err(|e| format!("MException.stack: {e}"))?;
                                Value::Cell(cell)
                            }
                            other => {
                                vm_bail!(format!("Reference to non-existent field '{}'.", other))
                            }
                        };
                        stack.push(value);
                    }
                    _ => vm_bail!("LoadMemberDynamic on non-struct/object".to_string()),
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
                // TODO(GC): write barrier hook for object/struct field write
                match base {
                    Value::Object(mut obj) => {
                        if let Some((p, _owner)) =
                            runmat_builtins::lookup_property(&obj.class_name, &field)
                        {
                            if p.is_static {
                                vm_bail!(format!(
                                    "Property '{}' is static; use classref('{}').{}",
                                    field, obj.class_name, field
                                ));
                            }
                            if p.set_access == runmat_builtins::Access::Private {
                                vm_bail!(format!("Property '{}' is private", field))
                            }
                            if p.is_dependent {
                                // Call set.<field>(obj, rhs)
                                let setter = format!("set.{field}");
                                match call_builtin_vm!(
                                    &setter,
                                    &[Value::Object(obj.clone()), rhs.clone()],
                                ) {
                                    Ok(v) => {
                                        stack.push(v);
                                        continue;
                                    }
                                    Err(_e) => {}
                                }
                            }
                            if let Some(oldv) = obj.properties.get(&field) {
                                runmat_gc::gc_record_write(oldv, &rhs);
                            }
                            obj.properties.insert(field, rhs);
                            stack.push(Value::Object(obj));
                        } else if let Some(cls) = runmat_builtins::get_class(&obj.class_name) {
                            if cls.methods.contains_key("subsasgn") {
                                match call_builtin_vm!(
                                    "call_method",
                                    &[
                                        Value::Object(obj),
                                        Value::String("subsasgn".to_string()),
                                        Value::String(".".to_string()),
                                        Value::String(field),
                                        rhs,
                                    ],
                                ) {
                                    Ok(v) => stack.push(v),
                                    Err(e) => vm_bail!(e),
                                }
                            } else {
                                vm_bail!(format!(
                                    "Undefined property '{}' for class {}",
                                    field, obj.class_name
                                ));
                            }
                        } else {
                            vm_bail!(format!("Unknown class {}", obj.class_name));
                        }
                    }
                    Value::ClassRef(cls) => {
                        if let Some((p, owner)) = runmat_builtins::lookup_property(&cls, &field) {
                            if !p.is_static {
                                vm_bail!(format!("Property '{}' is not static", field));
                            }
                            if p.set_access == runmat_builtins::Access::Private {
                                vm_bail!(format!("Property '{}' is private", field))
                            }
                            runmat_builtins::set_static_property_value_in_owner(
                                &owner, &field, rhs,
                            )?;
                            stack.push(Value::ClassRef(cls));
                        } else {
                            vm_bail!(format!("Unknown property '{}' on class {}", field, cls));
                        }
                    }
                    Value::HandleObject(handle) => match call_builtin_vm!(
                        "call_method",
                        &[
                            Value::HandleObject(handle),
                            Value::String("subsasgn".to_string()),
                            Value::String(".".to_string()),
                            Value::String(field),
                            rhs,
                        ],
                    ) {
                        Ok(v) => stack.push(v),
                        Err(e) => vm_bail!(e.to_string()),
                    },
                    Value::Struct(mut st) => {
                        if let Some(oldv) = st.fields.get(&field) {
                            runmat_gc::gc_record_write(oldv, &rhs);
                        }
                        st.fields.insert(field, rhs);
                        stack.push(Value::Struct(st));
                    }
                    Value::Cell(mut ca) => {
                        // Assign field across each element; support scalar rhs or cell rhs of same shape
                        let is_cell_rhs = matches!(rhs, Value::Cell(_));
                        let rhs_cell = if let Value::Cell(rc) = &rhs {
                            Some(rc)
                        } else {
                            None
                        };
                        if is_cell_rhs {
                            if let Some(rc) = rhs_cell {
                                if rc.rows != ca.rows || rc.cols != ca.cols {
                                    vm_bail!(
                                        "Field assignment: cell rhs shape mismatch".to_string()
                                    );
                                }
                            }
                        }
                        for i in 0..ca.data.len() {
                            let rv = if let Some(rc) = rhs_cell {
                                (*rc.data[i]).clone()
                            } else {
                                rhs.clone()
                            };
                            match &mut *ca.data[i] {
                                Value::Struct(st) => {
                                    if let Some(oldv) = st.fields.get(&field) {
                                        runmat_gc::gc_record_write(oldv, &rv);
                                    }
                                    st.fields.insert(field.clone(), rv);
                                }
                                other => {
                                    // If not struct, convert to struct with this single field
                                    let mut st = runmat_builtins::StructValue::new();
                                    st.fields.insert(field.clone(), rv);
                                    *other = Value::Struct(st);
                                }
                            }
                        }
                        stack.push(Value::Cell(ca));
                    }
                    Value::Num(0.0) if allow_init => {
                        let mut st = runmat_builtins::StructValue::new();
                        st.fields.insert(field, rhs);
                        stack.push(Value::Struct(st));
                    }
                    _ => vm_bail!("StoreMember on non-object".to_string()),
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
                // TODO(GC): write barrier hook for dynamic field write
                match base {
                    Value::Object(mut obj) => {
                        if let Some((p, _owner)) =
                            runmat_builtins::lookup_property(&obj.class_name, &name)
                        {
                            if p.is_static {
                                vm_bail!(format!(
                                    "Property '{}' is static; use classref('{}').{}",
                                    name, obj.class_name, name
                                ));
                            }
                            if p.set_access == runmat_builtins::Access::Private {
                                vm_bail!(format!("Property '{}' is private", name))
                            }
                        }
                        if let Some(oldv) = obj.properties.get(&name) {
                            runmat_gc::gc_record_write(oldv, &rhs);
                        }
                        obj.properties.insert(name, rhs);
                        stack.push(Value::Object(obj));
                    }
                    Value::HandleObject(handle) => match call_builtin_vm!(
                        "call_method",
                        &[
                            Value::HandleObject(handle),
                            Value::String("subsasgn".to_string()),
                            Value::String(".".to_string()),
                            Value::String(name),
                            rhs,
                        ],
                    ) {
                        Ok(v) => stack.push(v),
                        Err(e) => vm_bail!(e.to_string()),
                    },
                    Value::Struct(mut st) => {
                        if let Some(oldv) = st.fields.get(&name) {
                            runmat_gc::gc_record_write(oldv, &rhs);
                        }
                        st.fields.insert(name, rhs);
                        stack.push(Value::Struct(st));
                    }
                    Value::Cell(mut ca) => {
                        let is_cell_rhs = matches!(rhs, Value::Cell(_));
                        let rhs_cell = if let Value::Cell(rc) = &rhs {
                            Some(rc)
                        } else {
                            None
                        };
                        if is_cell_rhs {
                            if let Some(rc) = rhs_cell {
                                if rc.rows != ca.rows || rc.cols != ca.cols {
                                    vm_bail!(
                                        "Field assignment: cell rhs shape mismatch".to_string()
                                    );
                                }
                            }
                        }
                        for i in 0..ca.data.len() {
                            let rv = if let Some(rc) = rhs_cell {
                                (*rc.data[i]).clone()
                            } else {
                                rhs.clone()
                            };
                            match &mut *ca.data[i] {
                                Value::Struct(st) => {
                                    if let Some(oldv) = st.fields.get(&name) {
                                        runmat_gc::gc_record_write(oldv, &rv);
                                    }
                                    st.fields.insert(name.clone(), rv);
                                }
                                other => {
                                    let mut st = runmat_builtins::StructValue::new();
                                    st.fields.insert(name.clone(), rv);
                                    *other = Value::Struct(st);
                                }
                            }
                        }
                        stack.push(Value::Cell(ca));
                    }
                    Value::Num(0.0) if allow_init => {
                        let mut st = runmat_builtins::StructValue::new();
                        st.fields.insert(name, rhs);
                        stack.push(Value::Struct(st));
                    }
                    _ => vm_bail!("StoreMemberDynamic on non-struct/object".to_string()),
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
                match call_closures::load_static_property(&class_name, &prop) {
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
                // Build a minimal ClassDef and register it in runtime builtins registry
                let mut prop_map = std::collections::HashMap::new();
                for (p, is_static, get_access, set_access) in properties {
                    let gacc = if get_access.eq_ignore_ascii_case("private") {
                        runmat_builtins::Access::Private
                    } else {
                        runmat_builtins::Access::Public
                    };
                    let sacc = if set_access.eq_ignore_ascii_case("private") {
                        runmat_builtins::Access::Private
                    } else {
                        runmat_builtins::Access::Public
                    };
                    let (is_dep, clean_name) = if let Some(stripped) = p.strip_prefix("@dep:") {
                        (true, stripped.to_string())
                    } else {
                        (false, p.clone())
                    };
                    prop_map.insert(
                        clean_name.clone(),
                        runmat_builtins::PropertyDef {
                            name: clean_name,
                            is_static,
                            is_dependent: is_dep,
                            get_access: gacc,
                            set_access: sacc,
                            default_value: None,
                        },
                    );
                }
                let mut method_map = std::collections::HashMap::new();
                for (mname, fname, is_static, access) in methods {
                    let access = if access.eq_ignore_ascii_case("private") {
                        runmat_builtins::Access::Private
                    } else {
                        runmat_builtins::Access::Public
                    };
                    method_map.insert(
                        mname.clone(),
                        runmat_builtins::MethodDef {
                            name: mname,
                            is_static,
                            access,
                            function_name: fname,
                        },
                    );
                }
                let def = runmat_builtins::ClassDef {
                    name: name.clone(),
                    parent: super_class.clone(),
                    properties: prop_map,
                    methods: method_map,
                };
                runmat_builtins::register_class(def);
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
    // Persist any variables declared persistent in this bytecode under the given function name
    let func_name = name.to_string();
    for instr in &bytecode.instructions {
        match instr {
            crate::instr::Instr::DeclarePersistent(indices) => {
                for &i in indices {
                    if i < vars.len() {
                        let key = (func_name.clone(), i);
                        PERSISTENTS.with(|p| {
                            p.borrow_mut().insert(key, vars[i].clone());
                        });
                    }
                }
            }
            crate::instr::Instr::DeclarePersistentNamed(indices, names) => {
                for (pos, &i) in indices.iter().enumerate() {
                    if i < vars.len() {
                        let key = (func_name.clone(), i);
                        let name_key = (
                            func_name.clone(),
                            names
                                .get(pos)
                                .cloned()
                                .unwrap_or_else(|| format!("var_{i}")),
                        );
                        let val = vars[i].clone();
                        PERSISTENTS.with(|p| {
                            p.borrow_mut().insert(key, val.clone());
                        });
                        PERSISTENTS_BY_NAME.with(|p| {
                            p.borrow_mut().insert(name_key, val);
                        });
                    }
                }
            }
            _ => {}
        }
    }
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
