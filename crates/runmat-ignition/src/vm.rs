use crate::functions::{Bytecode, ExecutionContext, UserFunction};
use crate::gc_roots::InterpretContext;
use crate::instr::{EmitLabel, Instr};
#[cfg(feature = "native-accel")]
use runmat_accelerate::fusion_exec::{
    execute_centered_gram, execute_elementwise, execute_explained_variance,
    execute_image_normalize, execute_matmul_epilogue, execute_power_step_normalize,
    execute_reduction, FusionExecutionRequest,
};
#[cfg(feature = "native-accel")]
use runmat_accelerate::{
    activate_fusion_plan, deactivate_fusion_plan, fusion_residency, prepare_fusion_plan,
    set_current_pc, FusionPlan,
};
#[cfg(feature = "native-accel")]
use runmat_accelerate::{
    active_group_plan_clone, value_is_all_keyword, FusionKind, ReductionAxes, ShapeInfo,
    ValueOrigin, VarKind,
};
use runmat_builtins::{Type, Value};
use runmat_runtime::{
    builtins::common::tensor,
    builtins::stats::random::stochastic_evolution::stochastic_evolution_host,
    build_runtime_error, call_builtin, gather_if_needed, RuntimeControlFlow,
    workspace::{self as runtime_workspace, WorkspaceResolver},
};
use runmat_thread_local::runmat_thread_local;
#[cfg(not(target_arch = "wasm32"))]
use runmat_time::Instant;
#[cfg(target_arch = "wasm32")]
type Instant = ();
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt;
#[cfg(feature = "native-accel")]
use std::sync::Arc;
use std::sync::Once;
#[cfg(feature = "native-accel")]
use std::sync::OnceLock;
use tracing::{debug, info_span};

runmat_thread_local! {
    static CURRENT_PC: Cell<usize> = const { Cell::new(0) };
}

#[inline]
fn set_vm_pc(pc: usize) {
    CURRENT_PC.with(|cell| cell.set(pc));
}

#[inline]
fn current_pc() -> usize {
    CURRENT_PC.with(|cell| cell.get())
}

#[cfg(feature = "native-accel")]
struct FusionPlanGuard;

#[cfg(feature = "native-accel")]
impl Drop for FusionPlanGuard {
    fn drop(&mut self) {
        deactivate_fusion_plan();
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct InterpreterTiming {
    enabled: bool,
    host_span_start: Option<(Instant, usize)>,
    host_span_last_pc: Option<usize>,
    host_span_instrs: u64,
    seq: u64,
}

#[cfg(not(target_arch = "wasm32"))]
impl InterpreterTiming {
    fn new() -> Self {
        let enabled = std::env::var("RUNMAT_INTERPRETER_TIMING")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
            .unwrap_or(false);
        Self {
            enabled,
            host_span_start: None,
            host_span_last_pc: None,
            host_span_instrs: 0,
            seq: 0,
        }
    }

    fn note_host_instr(&mut self, pc: usize) {
        if !self.enabled {
            return;
        }
        if self.host_span_start.is_none() {
            self.host_span_start = Some((Instant::now(), pc));
            self.host_span_instrs = 0;
        }
        self.host_span_instrs += 1;
        self.host_span_last_pc = Some(pc);
    }

    fn flush_host_span(&mut self, reason: &str, detail: Option<&str>) {
        if !self.enabled {
            return;
        }
        let Some((start, start_pc)) = self.host_span_start.take() else {
            return;
        };
        let duration = start.elapsed();
        let end_pc = self.host_span_last_pc.unwrap_or(start_pc);
        let instrs = self.host_span_instrs.max(1);
        if let Some(extra) = detail {
            log::debug!(
                "interpreter_host_span seq={} reason={} detail={} pc_span=[{}..{}] instrs={} duration_ns={}",
                self.seq,
                reason,
                extra,
                start_pc,
                end_pc,
                instrs,
                duration.as_nanos()
            );
        } else {
            log::debug!(
                "interpreter_host_span seq={} reason={} pc_span=[{}..{}] instrs={} duration_ns={}",
                self.seq,
                reason,
                start_pc,
                end_pc,
                instrs,
                duration.as_nanos()
            );
        }
        self.seq += 1;
        self.host_span_last_pc = None;
        self.host_span_instrs = 0;
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for InterpreterTiming {
    fn drop(&mut self) {
        self.flush_host_span("drop", None);
    }
}

#[cfg(target_arch = "wasm32")]
struct InterpreterTiming;

#[cfg(target_arch = "wasm32")]
impl InterpreterTiming {
    fn new() -> Self {
        Self
    }

    fn note_host_instr(&mut self, _pc: usize) {}

    fn flush_host_span(&mut self, _reason: &str, _detail: Option<&str>) {}
}

#[cfg(target_arch = "wasm32")]
impl Drop for InterpreterTiming {
    fn drop(&mut self) {}
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
fn accel_promote_binary(op: AutoBinaryOp, a: &Value, b: &Value) -> VmResult<(Value, Value)> {
    use runmat_accelerate::{promote_binary, BinaryOp};
    let mapped = match op {
        AutoBinaryOp::Elementwise => BinaryOp::Elementwise,
        AutoBinaryOp::MatMul => BinaryOp::MatMul,
    };
    Ok(promote_binary(mapped, a, b).map_err(|e| e.to_string())?)
}

#[cfg(not(feature = "native-accel"))]
fn accel_promote_binary(_op: AutoBinaryOp, a: &Value, b: &Value) -> VmResult<(Value, Value)> {
    Ok((a.clone(), b.clone()))
}

#[cfg(feature = "native-accel")]
fn accel_promote_unary(op: AutoUnaryOp, value: &Value) -> VmResult<Value> {
    use runmat_accelerate::{promote_unary, UnaryOp};
    let mapped = match op {
        AutoUnaryOp::Transpose => UnaryOp::Transpose,
    };
    Ok(promote_unary(mapped, value).map_err(|e| e.to_string())?)
}

#[cfg(not(feature = "native-accel"))]
fn accel_promote_unary(_op: AutoUnaryOp, value: &Value) -> VmResult<Value> {
    Ok(value.clone())
}

#[cfg(feature = "native-accel")]
fn accel_prepare_args(name: &str, args: &[Value]) -> VmResult<Vec<Value>> {
    Ok(runmat_accelerate::prepare_builtin_args(name, args).map_err(|e| e.to_string())?)
}

#[cfg(not(feature = "native-accel"))]
fn accel_prepare_args(_name: &str, args: &[Value]) -> VmResult<Vec<Value>> {
    Ok(args.to_vec())
}

fn call_builtin_auto(name: &str, args: &[Value]) -> VmResult<Value> {
    let prepared = accel_prepare_args(name, args)?;
    Ok(runmat_runtime::call_builtin(name, &prepared)?)
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

// Namespace used for error identifiers (e.g., "MATLAB:..." or "RunMat:...")
const ERROR_NAMESPACE: &str = "MATLAB";

type VmResult<T> = Result<T, RuntimeControlFlow>;

#[inline]
fn mex(id: &str, msg: &str) -> RuntimeControlFlow {
    // Normalize identifier to always use the configured namespace prefix.
    // If caller passes "Namespace:suffix", strip the namespace and re-prefix with ERROR_NAMESPACE.
    let suffix = match id.find(':') {
        Some(pos) => &id[pos + 1..],
        None => id,
    };
    let ident = format!("{ERROR_NAMESPACE}:{suffix}");
    let pc = current_pc();
    let message = format!("{msg} (pc={pc})");
    build_runtime_error(message)
        .with_identifier(ident)
        .build()
        .into()
}

#[derive(Clone)]
enum SliceSelector {
    Colon,
    Scalar(usize),
    Indices(Vec<usize>),
}

#[derive(Debug, Clone)]
struct SlicePlan {
    indices: Vec<u32>,
    output_shape: Vec<usize>,
    selection_lengths: Vec<usize>,
    dims: usize,
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
    if shape.is_empty() {
        1
    } else {
        shape.iter().copied().product()
    }
}

fn indices_from_value_linear(value: &Value, total_len: usize) -> VmResult<Vec<usize>> {
    match value {
        Value::Num(n) => {
            let idx = *n as isize;
            if idx < 1 || (idx as usize) > total_len {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            Ok(vec![idx as usize])
        }
        Value::Int(int_val) => {
            let idx = int_val.to_i64();
            if idx < 1 || (idx as usize) > total_len {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            Ok(vec![idx as usize])
        }
        Value::Tensor(idx_t) => {
            let len = idx_t.shape.iter().product::<usize>();
            if len == total_len {
                let mut indices = Vec::new();
                for (i, &val) in idx_t.data.iter().enumerate() {
                    if val != 0.0 {
                        indices.push(i + 1);
                    }
                }
                Ok(indices)
            } else {
                let mut indices = Vec::with_capacity(len);
                for &val in &idx_t.data {
                    let idx = val as isize;
                    if idx < 1 || (idx as usize) > total_len {
                        return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                    }
                    indices.push(idx as usize);
                }
                Ok(indices)
            }
        }
        Value::LogicalArray(la) => {
            if la.data.len() != total_len {
                return Err(mex(
                    "IndexShape",
                    "Logical mask length mismatch for linear indexing",
                ));
            }
            let mut indices = Vec::new();
            for (i, &b) in la.data.iter().enumerate() {
                if b != 0 {
                    indices.push(i + 1);
                }
            }
            Ok(indices)
        }
        _ => Err(mex(
            "UnsupportedIndexType",
            "Unsupported index type for linear indexing",
        )),
    }
}

fn selector_from_value_dim(value: &Value, dim_len: usize) -> VmResult<SliceSelector> {
    match value {
        Value::Num(n) => {
            let idx = *n as isize;
            if idx < 1 || (idx as usize) > dim_len {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            Ok(SliceSelector::Scalar(idx as usize))
        }
        Value::Int(int_val) => {
            let idx = int_val.to_i64();
            if idx < 1 || (idx as usize) > dim_len {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            Ok(SliceSelector::Scalar(idx as usize))
        }
        Value::Tensor(idx_t) => {
            let len = idx_t.shape.iter().product::<usize>();
            if len == dim_len {
                let mut indices = Vec::new();
                for (i, &val) in idx_t.data.iter().enumerate() {
                    if val != 0.0 {
                        indices.push(i + 1);
                    }
                }
                Ok(SliceSelector::Indices(indices))
            } else {
                let mut indices = Vec::with_capacity(len);
                for &val in &idx_t.data {
                    let idx = val as isize;
                    if idx < 1 || (idx as usize) > dim_len {
                        return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                    }
                    indices.push(idx as usize);
                }
                Ok(SliceSelector::Indices(indices))
            }
        }
        Value::LogicalArray(la) => {
            if la.data.len() != dim_len {
                return Err(mex(
                    "IndexShape",
                    "Logical mask length mismatch for dimension",
                ));
            }
            let mut indices = Vec::new();
            for (i, &b) in la.data.iter().enumerate() {
                if b != 0 {
                    indices.push(i + 1);
                }
            }
            Ok(SliceSelector::Indices(indices))
        }
        _ => Err(mex(
            "UnsupportedIndexType",
            "Unsupported index type for slicing",
        )),
    }
}

fn build_slice_selectors(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
) -> VmResult<Vec<SliceSelector>> {
    let mut selectors = Vec::with_capacity(dims);
    if dims == 1 {
        let total_len = total_len_from_shape(base_shape);
        if (colon_mask & 1u32) != 0 {
            selectors.push(SliceSelector::Indices((1..=total_len).collect()));
            return Ok(selectors);
        }
        if (end_mask & 1u32) != 0 {
            selectors.push(SliceSelector::Scalar(total_len.max(1)));
            return Ok(selectors);
        }
        let value = numeric.first().ok_or_else(|| {
            mex(
                "MissingNumericIndex",
                "missing numeric index for linear slice",
            )
        })?;
        let idxs = indices_from_value_linear(value, total_len)?;
        selectors.push(SliceSelector::Indices(idxs));
        return Ok(selectors);
    }

    let mut numeric_iter = 0usize;
    for d in 0..dims {
        let is_colon = (colon_mask & (1u32 << d)) != 0;
        if is_colon {
            selectors.push(SliceSelector::Colon);
            continue;
        }
        let dim_len = base_shape.get(d).copied().unwrap_or(1);
        let is_end = (end_mask & (1u32 << d)) != 0;
        if is_end {
            selectors.push(SliceSelector::Scalar(dim_len));
            continue;
        }
        let value = numeric
            .get(numeric_iter)
            .ok_or_else(|| mex("MissingNumericIndex", "missing numeric index for slice"))?;
        numeric_iter += 1;
        selectors.push(selector_from_value_dim(value, dim_len)?);
    }
    Ok(selectors)
}

fn build_slice_plan(
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
) -> VmResult<SlicePlan> {
    let total_len = total_len_from_shape(base_shape);
    if dims == 1 {
        let list = selectors
            .first()
            .cloned()
            .unwrap_or(SliceSelector::Indices(Vec::new()));
        let indices = match list {
            SliceSelector::Colon => (1..=total_len).collect::<Vec<usize>>(),
            SliceSelector::Scalar(i) => vec![i],
            SliceSelector::Indices(v) => v,
        };
        if indices.iter().any(|&i| i == 0 || i > total_len) {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        let zero_based: Vec<u32> = indices.iter().map(|&i| (i - 1) as u32).collect();
        let count = zero_based.len();
        let shape = if count <= 1 {
            vec![1, 1]
        } else {
            vec![count, 1]
        };
        return Ok(SlicePlan {
            indices: zero_based,
            output_shape: shape,
            selection_lengths: vec![count],
            dims,
        });
    }

    let mut selection_lengths = Vec::with_capacity(dims);
    let mut per_dim_lists: Vec<Vec<usize>> = Vec::with_capacity(dims);
    for (d, sel) in selectors.iter().enumerate().take(dims) {
        let dim_len = base_shape.get(d).copied().unwrap_or(1);
        let idxs = match sel {
            SliceSelector::Colon => (1..=dim_len).collect::<Vec<usize>>(),
            SliceSelector::Scalar(i) => vec![*i],
            SliceSelector::Indices(v) => v.clone(),
        };
        if idxs.iter().any(|&i| i == 0 || i > dim_len) {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        selection_lengths.push(idxs.len());
        per_dim_lists.push(idxs);
    }

    if selection_lengths.contains(&0) {
        let mut out_shape = selection_lengths.clone();
        if dims == 2 {
            if selection_lengths[0] > 1 && selection_lengths[1] == 1 {
                out_shape = vec![selection_lengths[0], 1];
            } else if selection_lengths[0] == 1 && selection_lengths[1] > 1 {
                out_shape = vec![1, selection_lengths[1]];
            }
        }
        return Ok(SlicePlan {
            indices: Vec::new(),
            output_shape: out_shape,
            selection_lengths,
            dims,
        });
    }

    let mut base_norm = base_shape.to_vec();
    if base_norm.len() < dims {
        base_norm.resize(dims, 1);
    }

    let mut strides = vec![1usize; dims];
    for d in 1..dims {
        strides[d] = strides[d - 1] * base_norm[d - 1].max(1);
    }

    let mut indices = Vec::new();
    cartesian_product(&per_dim_lists, |multi| {
        let mut lin = 0usize;
        for d in 0..dims {
            let idx = multi[d] - 1;
            lin += idx * strides[d];
        }
        indices.push(lin as u32);
    });

    let mut out_shape = selection_lengths.clone();
    if dims == 2 {
        if selection_lengths[0] > 1 && selection_lengths[1] == 1 {
            out_shape = vec![selection_lengths[0], 1];
        } else if selection_lengths[0] == 1 && selection_lengths[1] > 1 {
            out_shape = vec![1, selection_lengths[1]];
        }
    }
    let total_out: usize = selection_lengths.iter().product();
    if total_out == 1 {
        out_shape = vec![1, 1];
    }

    Ok(SlicePlan {
        indices,
        output_shape: out_shape,
        selection_lengths,
        dims,
    })
}

fn gather_string_slice(
    sa: &runmat_builtins::StringArray,
    plan: &SlicePlan,
) -> VmResult<Value> {
    if plan.indices.is_empty() {
        let empty = runmat_builtins::StringArray::new(Vec::new(), plan.output_shape.clone())
            .map_err(|e| format!("Slice error: {e}"))?;
        return Ok(Value::StringArray(empty));
    }
    if plan.indices.len() == 1 {
        let lin = plan.indices[0] as usize;
        let value = sa
            .data
            .get(lin)
            .cloned()
            .ok_or_else(|| "Slice error: string index out of bounds".to_string())?;
        return Ok(Value::String(value));
    }
    let mut out = Vec::with_capacity(plan.indices.len());
    for &lin in &plan.indices {
        let idx = lin as usize;
        let value = sa
            .data
            .get(idx)
            .cloned()
            .ok_or_else(|| "Slice error: string index out of bounds".to_string())?;
        out.push(value);
    }
    let out_sa = runmat_builtins::StringArray::new(out, plan.output_shape.clone())
        .map_err(|e| format!("Slice error: {e}"))?;
    Ok(Value::StringArray(out_sa))
}

enum StringAssignView {
    Scalar(String),
    Array {
        data: Vec<String>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
}

fn build_string_rhs_view(
    rhs: &Value,
    selection_lengths: &[usize],
) -> VmResult<StringAssignView> {
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
                    return Err("shape mismatch for slice assign".to_string().into());
                }
                shape.truncate(dims);
            }
            for (rhs_len, sel_len) in shape.iter().zip(selection_lengths.iter()) {
                if !(*rhs_len == 1 || *rhs_len == *sel_len) {
                    return Err("shape mismatch for slice assign".to_string().into());
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
                    return Err("shape mismatch for slice assign".to_string().into());
                }
                shape.truncate(dims);
            }
            for (rhs_len, sel_len) in shape.iter().zip(selection_lengths.iter()) {
                if !(*rhs_len == 1 || *rhs_len == *sel_len) {
                    return Err("shape mismatch for slice assign".to_string().into());
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
        _ => Err("rhs must be string or string array".to_string().into()),
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

fn apply_end_offsets_to_numeric(
    numeric: &[Value],
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    end_offsets: &[(usize, i64)],
    base_shape: &[usize],
) -> Vec<Value> {
    let mut adjusted = numeric.to_vec();
    for (position, offset) in end_offsets {
        if let Some(value) = adjusted.get_mut(*position) {
            let mut seen_numeric = 0usize;
            let mut dim_for_pos = 0usize;
            for d in 0..dims {
                let is_colon = (colon_mask & (1u32 << d)) != 0;
                let is_end = (end_mask & (1u32 << d)) != 0;
                if is_colon || is_end {
                    continue;
                }
                if seen_numeric == *position {
                    dim_for_pos = d;
                    break;
                }
                seen_numeric += 1;
            }
            let dim_len = base_shape.get(dim_for_pos).copied().unwrap_or(1);
            let idx_val = (dim_len as isize) - (*offset as isize);
            *value = Value::Num(idx_val as f64);
        }
    }
    adjusted
}

fn materialize_rhs_linear(rhs: &Value, count: usize) -> VmResult<Vec<f64>> {
    let host_rhs = runmat_runtime::gather_if_needed(rhs)?;

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
                Err("shape mismatch for slice assign".to_string().into())
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
                Err("shape mismatch for slice assign".to_string().into())
            }
        }
        other => Err(format!("slice assign: unsupported RHS type {:?}", other).into()),
    }
}

fn materialize_rhs_nd(rhs: &Value, selection_lengths: &[usize]) -> VmResult<Vec<f64>> {
    let rhs_host = runmat_runtime::gather_if_needed(rhs)?;

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
                    return Err("shape mismatch for slice assign".to_string().into());
                }
                shape.truncate(selection_lengths.len());
            }
            for (dim_len, &sel_len) in shape.iter().zip(selection_lengths.iter()) {
                if *dim_len != 1 && *dim_len != sel_len {
                    return Err("shape mismatch for slice assign".to_string().into());
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
                return Err("shape mismatch for slice assign".to_string().into());
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
                return Err("shape mismatch for slice assign".to_string().into());
            }
            let mut shape = la.shape.clone();
            if shape.len() < selection_lengths.len() {
                shape.resize(selection_lengths.len(), 1);
            } else {
                shape.truncate(selection_lengths.len());
            }
            for (dim_len, &sel_len) in shape.iter().zip(selection_lengths.iter()) {
                if *dim_len != 1 && *dim_len != sel_len {
                    return Err("shape mismatch for slice assign".to_string().into());
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
                return Err("shape mismatch for slice assign".to_string().into());
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
        other => return Err(format!("slice assign: unsupported RHS type {:?}", other).into()),
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

struct WorkspaceState {
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    data_ptr: *const Value,
    len: usize,
}

type WorkspaceSnapshot = (HashMap<String, usize>, HashSet<String>);

runmat_thread_local! {
    static WORKSPACE_STATE: RefCell<Option<WorkspaceState>> = const { RefCell::new(None) };
    static PENDING_WORKSPACE: RefCell<Option<WorkspaceSnapshot>> = const { RefCell::new(None) };
    static LAST_WORKSPACE_STATE: RefCell<Option<WorkspaceSnapshot>> = const { RefCell::new(None) };
}

struct WorkspaceStateGuard;

impl Drop for WorkspaceStateGuard {
    fn drop(&mut self) {
        WORKSPACE_STATE.with(|state| {
            let mut state_mut = state.borrow_mut();
            if let Some(ws) = state_mut.take() {
                LAST_WORKSPACE_STATE.with(|slot| {
                    *slot.borrow_mut() = Some((ws.names, ws.assigned));
                });
            }
        });
    }
}

fn set_workspace_state(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    vars: &[Value],
) -> WorkspaceStateGuard {
    WORKSPACE_STATE.with(|state| {
        *state.borrow_mut() = Some(WorkspaceState {
            names,
            assigned,
            data_ptr: vars.as_ptr(),
            len: vars.len(),
        });
    });
    WorkspaceStateGuard
}

fn refresh_workspace_state(vars: &[Value]) {
    WORKSPACE_STATE.with(|state| {
        if let Some(ws) = state.borrow_mut().as_mut() {
            ws.data_ptr = vars.as_ptr();
            ws.len = vars.len();
        }
    });
}

fn workspace_lookup(name: &str) -> Option<Value> {
    WORKSPACE_STATE.with(|state| {
        let state_ref = state.borrow();
        let ws = state_ref.as_ref()?;
        let idx = ws.names.get(name)?;
        if !ws.assigned.contains(name) {
            return None;
        }
        if *idx >= ws.len {
            return None;
        }
        unsafe {
            let ptr = ws.data_ptr.add(*idx);
            Some((*ptr).clone())
        }
    })
}

fn workspace_snapshot() -> Vec<(String, Value)> {
    WORKSPACE_STATE.with(|state| {
        if let Some(ws) = state.borrow().as_ref() {
            let mut entries: Vec<(String, Value)> = ws
                .names
                .iter()
                .filter_map(|(name, idx)| {
                    if *idx >= ws.len {
                        return None;
                    }
                    if !ws.assigned.contains(name) {
                        return None;
                    }
                    unsafe {
                        let ptr = ws.data_ptr.add(*idx);
                        Some((name.clone(), (*ptr).clone()))
                    }
                })
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            entries
        } else {
            Vec::new()
        }
    })
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

fn set_workspace_variable(name: &str, value: Value, vars: &mut Vec<Value>) -> VmResult<()> {
    let mut result = Ok(());
    WORKSPACE_STATE.with(|state| {
        let mut state_mut = state.borrow_mut();
        match state_mut.as_mut() {
            Some(ws) => {
                let idx = if let Some(idx) = ws.names.get(name).copied() {
                    idx
                } else {
                    let idx = vars.len();
                    ws.names.insert(name.to_string(), idx);
                    idx
                };
                if idx >= vars.len() {
                    vars.resize(idx + 1, Value::Num(0.0));
                }
                vars[idx] = value;
                ws.data_ptr = vars.as_ptr();
                ws.len = vars.len();
                ws.assigned.insert(name.to_string());
            }
            None => {
                result = Err("load: workspace state unavailable".to_string().into());
            }
        }
    });
    result
}

fn assign_loaded_variables(
    vars: &mut Vec<Value>,
    entries: &[(String, Value)],
) -> VmResult<()> {
    for (name, value) in entries {
        set_workspace_variable(name, value.clone(), vars)?;
    }
    refresh_workspace_state(vars);
    Ok(())
}

fn ensure_workspace_resolver_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        runtime_workspace::register_workspace_resolver(WorkspaceResolver {
            lookup: workspace_lookup,
            snapshot: workspace_snapshot,
            globals: workspace_global_names,
        });
    });
}

pub struct PendingWorkspaceGuard;

impl Drop for PendingWorkspaceGuard {
    fn drop(&mut self) {
        PENDING_WORKSPACE.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}

pub fn push_pending_workspace(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
) -> PendingWorkspaceGuard {
    PENDING_WORKSPACE.with(|slot| {
        *slot.borrow_mut() = Some((names, assigned));
    });
    PendingWorkspaceGuard
}

pub fn take_updated_workspace_state() -> Option<(HashMap<String, usize>, HashSet<String>)> {
    LAST_WORKSPACE_STATE.with(|slot| slot.borrow_mut().take())
}

runmat_thread_local! {
    // (nargin, nargout) for current call
    static CALL_COUNTS: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

#[derive(Debug)]
pub enum InterpreterOutcome {
    Completed(Vec<Value>),
    Pending(Box<PendingExecution>),
}

pub struct PendingExecution {
    pub state: InterpreterState,
    pub interaction: runmat_runtime::interaction::PendingInteraction,
}

impl fmt::Debug for PendingExecution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingExecution")
            .field("prompt", &self.interaction.prompt)
            .finish()
    }
}

#[derive(Debug)]
pub struct InterpreterState {
    bytecode: Bytecode,
    stack: Vec<Value>,
    vars: Vec<Value>,
    pc: usize,
    context: ExecutionContext,
    try_stack: Vec<(usize, Option<usize>)>,
    last_exception: Option<runmat_builtins::MException>,
    imports: Vec<(Vec<String>, bool)>,
    global_aliases: HashMap<usize, String>,
    persistent_aliases: HashMap<usize, String>,
    current_function_name: String,
    call_counts: Vec<(usize, usize)>,
    #[cfg(feature = "native-accel")]
    fusion_plan: Option<Arc<FusionPlan>>,
}

impl InterpreterState {
    fn new(
        bytecode: Bytecode,
        initial_vars: &mut [Value],
        current_function_name: Option<&str>,
    ) -> Self {
        let mut vars = initial_vars.to_vec();
        if vars.len() < bytecode.var_count {
            vars.resize(bytecode.var_count, Value::Num(0.0));
        }
        let call_counts = CALL_COUNTS.with(|cc| cc.borrow().clone());
        Self {
            stack: Vec::new(),
            context: ExecutionContext {
                call_stack: Vec::new(),
                locals: Vec::new(),
                instruction_pointer: 0,
                functions: bytecode.functions.clone(),
            },
            try_stack: Vec::new(),
            last_exception: None,
            imports: Vec::new(),
            global_aliases: HashMap::new(),
            persistent_aliases: HashMap::new(),
            vars,
            pc: 0,
            call_counts,
            current_function_name: current_function_name
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<main>".to_string()),
            #[cfg(feature = "native-accel")]
            fusion_plan: prepare_fusion_plan(
                bytecode.accel_graph.as_ref(),
                &bytecode.fusion_groups,
            ),
            bytecode,
        }
    }
}

fn sync_initial_vars(initial: &mut [Value], vars: &[Value]) {
    for (i, var) in vars.iter().enumerate() {
        if i < initial.len() {
            initial[i] = var.clone();
        }
    }
}

fn is_suspend_flow(flow: &runmat_runtime::RuntimeControlFlow) -> bool {
    matches!(flow, runmat_runtime::RuntimeControlFlow::Suspend(_))
}

fn resolve_emit_label_text(
    label: &EmitLabel,
    var_names: &HashMap<usize, String>,
) -> Option<String> {
    match label {
        EmitLabel::Ans => Some("ans".to_string()),
        EmitLabel::Var(idx) => var_names
            .get(idx)
            .cloned()
            .or_else(|| Some(format!("var{idx}"))),
    }
}

macro_rules! handle_rel_binary { ($op:tt, $name:literal, $stack:ident) => {{
    let b = $stack.pop().ok_or(mex("StackUnderflow","stack underflow"))?; let a = $stack.pop().ok_or(mex("StackUnderflow","stack underflow"))?;
    match (&a, &b) {
        (Value::Object(obj), _) => { let args = vec![Value::Object(obj.clone()), Value::String($name.to_string()), b.clone()]; match call_builtin("call_method", &args) { Ok(v) => $stack.push(v), Err(_) => { let aa: f64 = (&a).try_into()?; let bb: f64 = (&b).try_into()?; $stack.push(Value::Num(if aa $op bb {1.0}else{0.0})) } } }
        (_, Value::Object(obj)) => { let rev = match $name { "lt" => "gt", "le" => "ge", "gt" => "lt", "ge" => "le", other => other };
            let args = vec![Value::Object(obj.clone()), Value::String(rev.to_string()), a.clone()]; match call_builtin("call_method", &args) { Ok(v) => $stack.push(v), Err(_) => { let aa: f64 = (&a).try_into()?; let bb: f64 = (&b).try_into()?; $stack.push(Value::Num(if aa $op bb {1.0}else{0.0})) } } }
        _ => { let bb: f64 = (&b).try_into()?; let aa: f64 = (&a).try_into()?; $stack.push(Value::Num(if aa $op bb {1.0}else{0.0})) }
    }
}}; }
pub fn interpret_with_vars(
    bytecode: &Bytecode,
    initial_vars: &mut [Value],
    current_function_name: Option<&str>,
) -> VmResult<InterpreterOutcome> {
    let state = InterpreterState::new(bytecode.clone(), initial_vars, current_function_name);
    run_interpreter(state, initial_vars)
}

pub fn resume_with_state(
    state: InterpreterState,
    initial_vars: &mut [Value],
) -> VmResult<InterpreterOutcome> {
    run_interpreter(state, initial_vars)
}

fn run_interpreter(
    state: InterpreterState,
    initial_vars: &mut [Value],
) -> VmResult<InterpreterOutcome> {
    let run_span = info_span!(
        "interpreter.run",
        function = state.current_function_name.as_str()
    );
    let _run_guard = run_span.enter();
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
        fusion_plan,
        bytecode,
    } = state;
    CALL_COUNTS.with(|cc| {
        *cc.borrow_mut() = call_counts.clone();
    });
    macro_rules! suspend_pending {
        ($restore:expr, $pending:expr) => {{
            $restore;
            let pending = $pending;
            sync_initial_vars(initial_vars, &vars);
            return Ok(InterpreterOutcome::Pending(Box::new(PendingExecution {
                state: InterpreterState {
                    bytecode,
                    stack,
                    vars,
                    pc,
                    context,
                    try_stack,
                    last_exception,
                    imports,
                    global_aliases,
                    persistent_aliases,
                    current_function_name,
                    call_counts,
                    #[cfg(feature = "native-accel")]
                    fusion_plan,
                },
                interaction: pending,
            })));
        }};
    }
    let pending_state = PENDING_WORKSPACE.with(|slot| slot.borrow_mut().take());
    let _workspace_guard = pending_state.map(|(names, assigned)| {
        let filtered_assigned: HashSet<String> = assigned
            .into_iter()
            .filter(|name| names.contains_key(name))
            .collect();
        set_workspace_state(names, filtered_assigned, &vars)
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
            let flow: RuntimeControlFlow = $err.into();
            match flow {
                RuntimeControlFlow::Suspend(pending) => {
                    suspend_pending!({}, pending);
                }
                RuntimeControlFlow::Error(err) => {
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
                        return Err(RuntimeControlFlow::Error(err));
                    }
                }
            }
        }};
    }
    while pc < bytecode.instructions.len() {
        set_vm_pc(pc);
        #[cfg(feature = "native-accel")]
        set_current_pc(pc);
        if runmat_runtime::interrupt::is_cancelled() {
            return Err(mex(
                "MATLAB:runmat:ExecutionCancelled",
                "Execution cancelled by user",
            ));
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
                let _fusion_span = info_span!(
                    "fusion.execute",
                    span_start = plan.group.span.start,
                    span_end = plan.group.span.end,
                    kind = ?plan.group.kind
                )
                .entered();
                match try_execute_fusion_group(&plan, graph, &mut stack, &mut vars, &context) {
                    Ok(result) => {
                        stack.push(result);
                        pc = plan.group.span.end + 1;
                        continue;
                    }
                    Err(err) => {
                        log::debug!("fusion fallback at pc {}: {}", pc, err);
                    }
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
                if let Some(value) = stack.last() {
                    let label_text = resolve_emit_label_text(&label, &bytecode.var_names);
                    runmat_runtime::console::record_value_output(label_text.as_deref(), value);
                }
            }
            Instr::EmitVar { var_index, label } => {
                if let Some(value) = vars.get(var_index) {
                    let label_text = resolve_emit_label_text(&label, &bytecode.var_names);
                    runmat_runtime::console::record_value_output(label_text.as_deref(), value);
                }
            }
            Instr::AndAnd(target) => {
                let lhs: f64 = (&stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?)
                    .try_into()?;
                if lhs == 0.0 {
                    pc = target;
                    continue;
                }
            }
            Instr::OrOr(target) => {
                let lhs: f64 = (&stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?)
                    .try_into()?;
                if lhs != 0.0 {
                    pc = target;
                    continue;
                }
            }
            Instr::Swap => {
                let a = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let b = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                stack.push(a);
                stack.push(b);
            }
            Instr::CallFeval(argc) => {
                // Pop explicit args
                let mut args = Vec::with_capacity(argc);
                for _ in 0..argc {
                    args.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                args.reverse();
                // Pop function value
                let func_val = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match func_val {
                    Value::Closure(c) => {
                        // User-defined function via closure: prepend captures then dispatch like CallFunction
                        let name = c.function_name;
                        let mut call_args = c.captures.clone();
                        call_args.extend(args);
                        // Try runtime builtin target for closures (e.g., call_method)
                        if let Ok(result) = runmat_runtime::call_builtin(&name, &call_args) {
                            stack.push(result);
                            pc += 1;
                            continue;
                        }
                        let func: UserFunction = match context
                            .functions
                            .get(&name)
                            .or_else(|| bytecode.functions.get(&name))
                        {
                            Some(f) => f.clone(),
                            None => vm_bail!(mex(
                                "UndefinedFunction",
                                &format!("Undefined function: {name}")
                            )),
                        };
                        let arg_count = call_args.len();
                        if !func.has_varargin {
                            if arg_count < func.params.len() {
                                vm_bail!(mex(
                                    "NotEnoughInputs",
                                    &format!(
                                        "Function '{name}' expects {} inputs, got {arg_count}",
                                        func.params.len()
                                    )
                                ));
                            }
                            if arg_count > func.params.len() {
                                vm_bail!(mex(
                                    "TooManyInputs",
                                    &format!(
                                        "Function '{name}' expects {} inputs, got {arg_count}",
                                        func.params.len()
                                    )
                                ));
                            }
                        }
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
                        if func.has_varargin {
                            let fixed = func.params.len().saturating_sub(1);
                            for i in 0..fixed {
                                if i < call_args.len() && i < func_vars.len() {
                                    func_vars[i] = call_args[i].clone();
                                }
                            }
                            let mut rest: Vec<Value> = if call_args.len() > fixed {
                                call_args[fixed..].to_vec()
                            } else {
                                Vec::new()
                            };
                            let cell = runmat_builtins::CellArray::new(
                                std::mem::take(&mut rest),
                                1,
                                if call_args.len() > fixed {
                                    call_args.len() - fixed
                                } else {
                                    0
                                },
                            )
                            .map_err(|e| format!("varargin: {e}"))?;
                            if fixed < func_vars.len() {
                                func_vars[fixed] = Value::Cell(cell);
                            }
                        } else {
                            for (i, _param_id) in func.params.iter().enumerate() {
                                if i < call_args.len() && i < func_vars.len() {
                                    func_vars[i] = call_args[i].clone();
                                }
                            }
                        }
                        // Copy referenced globals into local frame
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
                        // Initialize varargout if needed
                        if func.has_varargout {
                            if let Some(varargout_oid) = func.outputs.last() {
                                if let Some(local_id) = var_map.get(varargout_oid) {
                                    if local_id.0 < func_vars.len() {
                                        let empty = runmat_builtins::CellArray::new(vec![], 1, 0)
                                            .map_err(|e| format!("varargout init: {e}"))?;
                                        func_vars[local_id.0] = Value::Cell(empty);
                                    }
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
                        let func_bytecode =
                            crate::compile_with_functions(&func_program, &bytecode.functions)?;
                        // Merge nested functions into current execution context for future closure calls
                        for (k, v) in func_bytecode.functions.iter() {
                            context.functions.insert(k.clone(), v.clone());
                        }
                        let func_result_vars = match interpret_function(&func_bytecode, func_vars) {
                            Ok(v) => v,
                            Err(e) => vm_bail!(e),
                        };
                        if let Some(output_var_id) = func.outputs.first() {
                            let local_output_index =
                                var_map.get(output_var_id).map(|id| id.0).unwrap_or(0);
                            if local_output_index < func_result_vars.len() {
                                stack.push(func_result_vars[local_output_index].clone());
                            } else {
                                stack.push(Value::Num(0.0));
                            }
                        } else {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    other => {
                        // Forward to runtime feval for string/char handles and builtins
                        let mut argv = Vec::with_capacity(1 + args.len());
                        argv.push(other);
                        argv.extend(args);
                        match runmat_runtime::call_builtin("feval", &argv) {
                            Ok(result) => stack.push(result),
                            Err(err) => vm_bail!(err),
                        }
                    }
                }
            }
            Instr::CallFevalExpandMulti(_specs) => {
                vm_bail!("feval expand not supported in this execution mode".to_string());
            }
            Instr::LoadConst(c) => {
                stack.push(Value::Num(c));
                if debug_stack {
                    debug!(const_value = c, stack_len = stack.len(), "[vm] load const");
                }
            }
            Instr::LoadComplex(re, im) => {
                stack.push(Value::Complex(re, im));
                if debug_stack {
                    eprintln!(
                        "  -> LoadComplex pushed ({}, {}), new_len={}",
                        re,
                        im,
                        stack.len()
                    );
                }
            }
            Instr::LoadBool(b) => stack.push(Value::Bool(b)),
            Instr::LoadString(s) => stack.push(Value::String(s)),
            Instr::LoadCharRow(s) => {
                let ca = runmat_builtins::CharArray::new(s.chars().collect(), 1, s.chars().count())
                    .map_err(|e| mex("CharError", &e))?;
                stack.push(Value::CharArray(ca));
            }
            Instr::LoadVar(i) => {
                let v = vars[i].clone();
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
                stack.push(v)
            }
            Instr::StoreVar(i) => {
                let val = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                if let Ok(filter) = std::env::var("RUNMAT_DEBUG_STORE_VAR") {
                    let log_this = if filter.trim().eq_ignore_ascii_case("*") {
                        true
                    } else if let Ok(target) = filter.trim().parse::<usize>() {
                        target == i
                    } else {
                        false
                    };
                    if log_this {
                        debug!(pc, var = i, ?val, "[vm] StoreVar value");
                    }
                }
                if std::env::var("RUNMAT_DEBUG_INDEX").as_deref() == Ok("1") {
                    match &val {
                        Value::GpuTensor(h) => {
                            debug!(pc, var = i, shape = ?h.shape, "[vm] StoreVar GPU tensor");
                        }
                        Value::Tensor(t) => {
                            debug!(pc, var = i, shape = ?t.shape, "[vm] StoreVar tensor");
                        }
                        _ => {}
                    }
                }
                if i < vars.len() {
                    #[cfg(feature = "native-accel")]
                    clear_residency(&vars[i]);
                }
                if i >= vars.len() {
                    vars.resize(i + 1, Value::Num(0.0));
                    refresh_workspace_state(&vars);
                }
                vars[i] = val;
                // If this var is declared global, update the global table entry
                // We optimistically write-through whenever StoreVar happens and a global exists for this name
                let key = format!("var_{i}");
                GLOBALS.with(|g| {
                    let mut m = g.borrow_mut();
                    if m.contains_key(&key) {
                        m.insert(key, vars[i].clone());
                    }
                });
                if let Some(name) = global_aliases.get(&i) {
                    GLOBALS.with(|g| {
                        g.borrow_mut().insert(name.clone(), vars[i].clone());
                    });
                }
            }
            Instr::LoadLocal(offset) => {
                if let Some(current_frame) = context.call_stack.last() {
                    let local_index = current_frame.locals_start + offset;
                    if local_index >= context.locals.len() {
                        vm_bail!("Local variable index out of bounds".to_string());
                    }
                    stack.push(context.locals[local_index].clone());
                } else if offset < vars.len() {
                    stack.push(vars[offset].clone());
                } else {
                    stack.push(Value::Num(0.0));
                }
            }
            Instr::StoreLocal(offset) => {
                let val = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                if let Some(current_frame) = context.call_stack.last() {
                    let local_index = current_frame.locals_start + offset;
                    while context.locals.len() <= local_index {
                        context.locals.push(Value::Num(0.0));
                    }
                    #[cfg(feature = "native-accel")]
                    if local_index < context.locals.len() {
                        clear_residency(&context.locals[local_index]);
                    }
                    context.locals[local_index] = val;
                } else {
                    if offset >= vars.len() {
                        vars.resize(offset + 1, Value::Num(0.0));
                        refresh_workspace_state(&vars);
                    }
                    #[cfg(feature = "native-accel")]
                    if offset < vars.len() {
                        clear_residency(&vars[offset]);
                    }
                    vars[offset] = val;
                    // write-through to persistents if this local is a declared persistent for current function
                    let func_name = context
                        .call_stack
                        .last()
                        .map(|f| f.function_name.clone())
                        .unwrap_or_else(|| "<main>".to_string());
                    let key = (func_name, offset);
                    PERSISTENTS.with(|p| {
                        let mut m = p.borrow_mut();
                        if m.contains_key(&key) {
                            m.insert(key, vars[offset].clone());
                        }
                    });
                }
            }
            Instr::EnterScope(local_count) => {
                for _ in 0..local_count {
                    context.locals.push(Value::Num(0.0));
                }
            }
            Instr::ExitScope(local_count) => {
                for _ in 0..local_count {
                    if let Some(val) = context.locals.pop() {
                        #[cfg(feature = "native-accel")]
                        clear_residency(&val);
                    }
                }
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
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = call_builtin("plus", &[a.clone(), b.clone()])?;
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
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = call_builtin("plus", &[a.clone(), b.clone()])?;
                                stack.push(v)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                        let v = call_builtin("plus", &[a_acc, b_acc])?;
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
                        let args = vec![Value::Object(obj.clone()), b.clone()];
                        match call_builtin("minus", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = call_builtin("minus", &[a.clone(), b.clone()])?;
                                stack.push(v)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![Value::Object(obj.clone()), a.clone()];
                        match call_builtin("uminus", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = call_builtin("minus", &[a.clone(), b.clone()])?;
                                stack.push(v)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                        let v = call_builtin("minus", &[a_acc, b_acc])?;
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
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = runmat_runtime::matrix::value_matmul(&a, &b)?;
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
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = runmat_runtime::matrix::value_matmul(&a, &b)?;
                                stack.push(v)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) = accel_promote_binary(AutoBinaryOp::MatMul, &a, &b)?;
                        let v = runmat_runtime::matrix::value_matmul(&a_acc, &b_acc)?;
                        stack.push(v)
                    }
                }
            }
            Instr::Div => {
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
                            Value::String("mrdivide".to_string()),
                            b.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                                let v = runmat_runtime::call_builtin("rdivide", &[a_acc, b_acc])?;
                                stack.push(v)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("mrdivide".to_string()),
                            a.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                                let v = runmat_runtime::call_builtin("rdivide", &[a_acc, b_acc])?;
                                stack.push(v)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                        let v = runmat_runtime::call_builtin("rdivide", &[a_acc, b_acc])?;
                        stack.push(v)
                    }
                }
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
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let v = runmat_runtime::power(&a, &b)?;
                                stack.push(v)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
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
                        match call_builtin("uminus", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let result = runmat_runtime::call_builtin(
                                    "times",
                                    &[value.clone(), runmat_builtins::Value::Num(-1.0)],
                                )?;
                                stack.push(result)
                            }
                        }
                    }
                    _ => {
                        let result = runmat_runtime::call_builtin(
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
                        match call_builtin("uplus", &args) {
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
                let promoted = accel_promote_unary(AutoUnaryOp::Transpose, &value)?;
                let args = [promoted];
                let result = runmat_runtime::call_builtin("transpose", &args)?;
                stack.push(result);
            }
            Instr::ConjugateTranspose => {
                let value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let promoted = accel_promote_unary(AutoUnaryOp::Transpose, &value)?;
                let args = [promoted];
                let result = runmat_runtime::call_builtin("ctranspose", &args)?;
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
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                                stack.push(runmat_runtime::call_builtin("times", &[a_acc, b_acc])?)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("times".to_string()),
                            a.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                                stack.push(runmat_runtime::call_builtin("times", &[a_acc, b_acc])?)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                        stack.push(runmat_runtime::call_builtin("times", &[a_acc, b_acc])?)
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
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                                stack
                                    .push(runmat_runtime::call_builtin("rdivide", &[a_acc, b_acc])?)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("rdivide".to_string()),
                            a.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                                stack
                                    .push(runmat_runtime::call_builtin("rdivide", &[a_acc, b_acc])?)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                        stack.push(runmat_runtime::call_builtin("rdivide", &[a_acc, b_acc])?)
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
                        match call_builtin("power", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (a_acc, b_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                                stack.push(runmat_runtime::call_builtin("power", &[a_acc, b_acc])?)
                            }
                        }
                    }
                    _ => {
                        let (a_acc, b_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b)?;
                        stack.push(runmat_runtime::call_builtin("power", &[a_acc, b_acc])?)
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
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (b_acc, a_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &b, &a)?;
                                stack
                                    .push(runmat_runtime::call_builtin("rdivide", &[b_acc, a_acc])?)
                            }
                        }
                    }
                    (_, Value::Object(obj)) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("ldivide".to_string()),
                            a.clone(),
                        ];
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                let (b_acc, a_acc) =
                                    accel_promote_binary(AutoBinaryOp::Elementwise, &b, &a)?;
                                stack
                                    .push(runmat_runtime::call_builtin("rdivide", &[b_acc, a_acc])?)
                            }
                        }
                    }
                    _ => {
                        let (b_acc, a_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &b, &a)?;
                        stack.push(runmat_runtime::call_builtin("rdivide", &[b_acc, a_acc])?)
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
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: le(a,b) = ~gt(a,b)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("gt".to_string()),
                                    b.clone(),
                                ];
                                match call_builtin("call_method", &args2) {
                                    Ok(v) => {
                                        let truth: f64 = (&v).try_into()?;
                                        stack.push(Value::Num(if truth == 0.0 {
                                            1.0
                                        } else {
                                            0.0
                                        }));
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
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: ge(b,a) = ~lt(b,a) hence le(a,b) = ge(b,a)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("lt".to_string()),
                                    a.clone(),
                                ];
                                match call_builtin("call_method", &args2) {
                                    Ok(v) => {
                                        let truth: f64 = (&v).try_into()?;
                                        stack.push(Value::Num(if truth == 0.0 {
                                            1.0
                                        } else {
                                            0.0
                                        }));
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
                        let bb: f64 = (&b).try_into()?;
                        let aa: f64 = (&a).try_into()?;
                        stack.push(Value::Num(if aa <= bb { 1.0 } else { 0.0 }));
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
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: ge(a,b) = ~lt(a,b)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("lt".to_string()),
                                    b.clone(),
                                ];
                                match call_builtin("call_method", &args2) {
                                    Ok(v) => {
                                        let truth: f64 = (&v).try_into()?;
                                        stack.push(Value::Num(if truth == 0.0 {
                                            1.0
                                        } else {
                                            0.0
                                        }));
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
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: le(b,a) = ~gt(b,a); hence ge(a,b) = le(b,a)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("gt".to_string()),
                                    a.clone(),
                                ];
                                match call_builtin("call_method", &args2) {
                                    Ok(v) => {
                                        let truth: f64 = (&v).try_into()?;
                                        stack.push(Value::Num(if truth == 0.0 {
                                            1.0
                                        } else {
                                            0.0
                                        }));
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
                        let bb: f64 = (&b).try_into()?;
                        let aa: f64 = (&a).try_into()?;
                        stack.push(Value::Num(if aa >= bb { 1.0 } else { 0.0 }));
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
                match (&a, &b) {
                    (Value::Object(obj), _) => {
                        let args = vec![
                            Value::Object(obj.clone()),
                            Value::String("eq".to_string()),
                            b.clone(),
                        ];
                        match call_builtin("call_method", &args) {
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
                        match call_builtin("call_method", &args) {
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
                        let v = runmat_runtime::call_builtin("eq", &[a.clone(), b.clone()])?;
                        stack.push(v);
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
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: ne(a,b) = ~eq(a,b)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("eq".to_string()),
                                    b.clone(),
                                ];
                                match call_builtin("call_method", &args2) {
                                    Ok(v) => {
                                        let truth: f64 = (&v).try_into()?;
                                        stack.push(Value::Num(if truth == 0.0 {
                                            1.0
                                        } else {
                                            0.0
                                        }));
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
                        match call_builtin("call_method", &args) {
                            Ok(v) => stack.push(v),
                            Err(_) => {
                                // Fallback: ne(b,a) = ~eq(b,a)
                                let args2 = vec![
                                    Value::Object(obj.clone()),
                                    Value::String("eq".to_string()),
                                    a.clone(),
                                ];
                                match call_builtin("call_method", &args2) {
                                    Ok(v) => {
                                        let truth: f64 = (&v).try_into()?;
                                        stack.push(Value::Num(if truth == 0.0 {
                                            1.0
                                        } else {
                                            0.0
                                        }));
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
                        let v = runmat_runtime::call_builtin("ne", &[a.clone(), b.clone()])?;
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
                let cond: f64 = (&stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?)
                    .try_into()?;
                if cond == 0.0 {
                    pc = target;
                    continue;
                }
            }
            Instr::Jump(target) => {
                pc = target;
                continue;
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
                )?;
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
                if name == "nargin" {
                    if arg_count != 0 {
                        vm_bail!(mex("TooManyInputs", "nargin takes no arguments"));
                    }
                    let (nin, _) =
                        CALL_COUNTS.with(|cc| cc.borrow().last().cloned().unwrap_or((0, 0)));
                    stack.push(Value::Num(nin as f64));
                    pc += 1;
                    continue;
                }
                if name == "nargout" {
                    if arg_count != 0 {
                        vm_bail!(mex("TooManyInputs", "nargout takes no arguments"));
                    }
                    let (_, nout) =
                        CALL_COUNTS.with(|cc| cc.borrow().last().cloned().unwrap_or((0, 0)));
                    stack.push(Value::Num(nout as f64));
                    pc += 1;
                    continue;
                }
                let mut args = Vec::new();

                for _ in 0..arg_count {
                    args.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                args.reverse();

                let prepared_primary = accel_prepare_args(&name, &args)?;
                match runmat_runtime::call_builtin(&name, &prepared_primary) {
                    Ok(result) => stack.push(result),
                    Err(e) => {
                        if is_suspend_flow(&e) {
                            for arg in args.iter().rev() {
                                stack.push(arg.clone());
                            }
                            match e {
                                runmat_runtime::RuntimeControlFlow::Suspend(pending) => {
                                    suspend_pending!({}, pending);
                                }
                                runmat_runtime::RuntimeControlFlow::Error(_) => {}
                            }
                        }
                        let runmat_runtime::RuntimeControlFlow::Error(e) = e else {
                            unreachable!("suspend handled above");
                        };
                        // Specific-import matches: import pkg.foo; name == foo
                        let mut specific_matches: Vec<(String, Vec<Value>, Value)> = Vec::new();
                        for (path, wildcard) in &imports {
                            if *wildcard {
                                continue;
                            }
                            if path.last().map(|s| s.as_str()) == Some(name.as_str()) {
                                let qual = path.join(".");
                                let qual_args = accel_prepare_args(&qual, &prepared_primary)?;
                                match runmat_runtime::call_builtin(&qual, &qual_args) {
                                    Ok(value) => specific_matches.push((qual, qual_args, value)),
                                    Err(err) => {
                                        if is_suspend_flow(&err) {
                                            for arg in args.iter().rev() {
                                                stack.push(arg.clone());
                                            }
                                            match err {
                                                runmat_runtime::RuntimeControlFlow::Suspend(pending) => {
                                                    suspend_pending!({}, pending);
                                                }
                                                runmat_runtime::RuntimeControlFlow::Error(_) => {}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if specific_matches.len() > 1 {
                            let msg = specific_matches
                                .iter()
                                .map(|(q, _, _)| q.clone())
                                .collect::<Vec<_>>()
                                .join(", ");
                            vm_bail!(format!("ambiguous builtin '{}' via imports: {}", name, msg)
                                .to_string());
                        }
                        if let Some((_, _, value)) = specific_matches.pop() {
                            stack.push(value);
                        } else {
                            // Wildcard-import matches: import pkg.*; try pkg.name
                            let mut wildcard_matches: Vec<(String, Vec<Value>, Value)> = Vec::new();
                            for (path, wildcard) in &imports {
                                if !*wildcard {
                                    continue;
                                }
                                if path.is_empty() {
                                    continue;
                                }
                                let mut qual = String::new();
                                for (i, part) in path.iter().enumerate() {
                                    if i > 0 {
                                        qual.push('.');
                                    }
                                    qual.push_str(part);
                                }
                                qual.push('.');
                                qual.push_str(&name);
                                let qual_args = accel_prepare_args(&qual, &prepared_primary)?;
                                match runmat_runtime::call_builtin(&qual, &qual_args) {
                                    Ok(value) => wildcard_matches.push((qual, qual_args, value)),
                                    Err(err) => {
                                        if is_suspend_flow(&err) {
                                            for arg in args.iter().rev() {
                                                stack.push(arg.clone());
                                            }
                                            match err {
                                                runmat_runtime::RuntimeControlFlow::Suspend(pending) => {
                                                    suspend_pending!({}, pending);
                                                }
                                                runmat_runtime::RuntimeControlFlow::Error(_) => {}
                                            }
                                        }
                                    }
                                }
                            }
                            if wildcard_matches.len() > 1 {
                                let msg = wildcard_matches
                                    .iter()
                                    .map(|(q, _, _)| q.clone())
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                vm_bail!(format!(
                                    "ambiguous builtin '{}' via wildcard imports: {}",
                                    name, msg
                                )
                                .to_string());
                            }
                            if let Some((_, _, value)) = wildcard_matches.pop() {
                                stack.push(value);
                            } else {
                                // Special-case: rethrow() without explicit e uses last caught
                                if name == "rethrow" && args.is_empty() {
                                    if let Some(le) = &last_exception {
                                        vm_bail!(format!("{}: {}", le.identifier, le.message)
                                            .to_string());
                                    }
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
                                    return Err(e.to_string().into());
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
                    (Value::Cell(ca), 1) => {
                        match &indices[0] {
                            Value::Num(n) => {
                                let i = *n as usize;
                                if i == 0 || i > ca.data.len() {
                                    return Err(mex(
                                        "CellIndexOutOfBounds",
                                        "Cell index out of bounds",
                                    ));
                                }
                                vec![(*ca.data[i - 1]).clone()]
                            }
                            Value::Int(i) => {
                                let iu = i.to_i64() as usize;
                                if iu == 0 || iu > ca.data.len() {
                                    return Err(mex(
                                        "CellIndexOutOfBounds",
                                        "Cell index out of bounds",
                                    ));
                                }
                                vec![(*ca.data[iu - 1]).clone()]
                            }
                            Value::Tensor(t) => {
                                // Treat as list of 1-based indices; expand each
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
                            _ => return Err(mex("CellIndexType", "Unsupported cell index type")),
                        }
                    }
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
                    (other, _) => {
                        // Route to subsref(obj,'{}',{indices...}) if object
                        match other {
                            Value::Object(obj) => {
                                let cell = runmat_builtins::CellArray::new(
                                    indices.clone(),
                                    1,
                                    indices.len(),
                                )
                                .map_err(|e| format!("subsref build error: {e}"))?;
                                let v = match runmat_runtime::call_builtin(
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
                match call_builtin_auto(&name, &args) {
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
                        _ => return Err(mex("CellIndexType", "Unsupported cell index type")),
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
                        let idx_vals: Vec<Value> = indices
                            .iter()
                            .map(|v| Value::Num((v).try_into().unwrap_or(0.0)))
                            .collect();
                        let cell = runmat_runtime::call_builtin("__make_cell", &idx_vals)?;
                        let v = match runmat_runtime::call_builtin(
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
                match call_builtin_auto(&name, &args) {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::CallBuiltinExpandMulti(name, specs) => {
                // Build final args by walking specs left-to-right and popping from stack accordingly.
                let mut args: Vec<Value> = Vec::with_capacity(specs.len());
                // We'll reconstruct by first collecting a temporary vector and then reversing (since stack is LIFO)
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
                        #[cfg(feature = "native-accel")]
                        clear_residency(&base);
                        let expanded = if spec.expand_all {
                            match base {
                                Value::Cell(ca) => {
                                    ca.data.iter().map(|p| (*(*p)).clone()).collect()
                                }
                                Value::Object(obj) => {
                                    // subsref(obj,'{}', {}) with empty indices; expect a cell or value
                                    let empty = runmat_builtins::CellArray::new(vec![], 1, 0)
                                        .map_err(|e| format!("subsref build error: {e}"))?;
                                    let v = match runmat_runtime::call_builtin(
                                        "call_method",
                                        &[
                                            Value::Object(obj),
                                            Value::String("subsref".to_string()),
                                            Value::String("{}".to_string()),
                                            Value::Cell(empty),
                                        ],
                                    ) {
                                        Ok(v) => v,
                                        Err(e) => vm_bail!(e),
                                    };
                                    match v {
                                        Value::Cell(ca) => {
                                            ca.data.iter().map(|p| (*(*p)).clone()).collect()
                                        }
                                        other => vec![other],
                                    }
                                }
                                _ => return Err(mex(
                                    "ExpandError",
                                    "CallBuiltinExpandMulti requires cell or object for expand_all",
                                )),
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
                                    let idx_vals: Vec<Value> = indices
                                        .iter()
                                        .map(|v| Value::Num((v).try_into().unwrap_or(0.0)))
                                        .collect();
                                    let cell =
                                        runmat_runtime::call_builtin("__make_cell", &idx_vals)?;
                                    let v = match runmat_runtime::call_builtin(
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
                                _ => return Err(mex(
                                    "ExpandError",
                                    "CallBuiltinExpandMulti requires cell or object cell access",
                                )),
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
                args.extend(temp.into_iter());
                match call_builtin_auto(&name, &args) {
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
                                    let v = match runmat_runtime::call_builtin("call_method", &[
                                        Value::Object(obj),
                                        Value::String("subsref".to_string()),
                                        Value::String("{}".to_string()),
                                        Value::Cell(empty),
                                    ]) { Ok(v) => v, Err(e) => vm_bail!(e) };
                                    match v { Value::Cell(ca) => ca.data.iter().map(|p| (*(*p)).clone()).collect::<Vec<Value>>(), other => vec![other] }
                                }
                                _ => return Err("CallFunctionExpandMulti requires cell or object for expand_all".to_string().into()),
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
                                    let v = match runmat_runtime::call_builtin(
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
                let func_bytecode =
                    crate::compile_with_functions(&func_program, &bytecode.functions)?;
                // Make nested closures visible to outer frames
                for (k, v) in func_bytecode.functions.iter() {
                    context.functions.insert(k.clone(), v.clone());
                }
                let func_result_vars = match interpret_function(&func_bytecode, func_vars) {
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
                    let mut args = Vec::new();
                    for _ in 0..arg_count {
                        args.push(
                            stack
                                .pop()
                                .ok_or(mex("StackUnderflow", "stack underflow"))?,
                        );
                    }
                    args.reverse();
                    let prepared_primary = accel_prepare_args(&name, &args)?;
                    if let Ok(result) = runmat_runtime::call_builtin(&name, &prepared_primary) {
                        stack.push(result);
                        pc += 1;
                        continue;
                    }
                    // Put args back if not a builtin: we'll handle as user function below
                    for v in prepared_primary.into_iter().rev() {
                        stack.push(v);
                    }
                }
                let func: UserFunction = match bytecode.functions.get(&name) {
                    Some(f) => f.clone(),
                    None => vm_bail!(mex(
                        "UndefinedFunction",
                        &format!("Undefined function: {name}")
                    )),
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
                if !func.has_varargin {
                    if arg_count < func.params.len() {
                        vm_bail!(mex(
                            "NotEnoughInputs",
                            &format!(
                                "Function '{name}' expects {} inputs, got {arg_count}",
                                func.params.len()
                            )
                        ));
                    }
                    if arg_count > func.params.len() {
                        vm_bail!(mex(
                            "TooManyInputs",
                            &format!(
                                "Function '{name}' expects {} inputs, got {arg_count}",
                                func.params.len()
                            )
                        ));
                    }
                } else {
                    let min_args = func.params.len().saturating_sub(1);
                    if arg_count < min_args {
                        vm_bail!(mex(
                            "NotEnoughInputs",
                            &format!("Function '{name}' expects at least {min_args} inputs, got {arg_count}")
                        ));
                    }
                }
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
                if func.has_varargin {
                    // All fixed parameters except the last (varargin placeholder) are positional; pack the rest into a cell
                    let fixed = func.params.len().saturating_sub(1);
                    for i in 0..fixed {
                        if i < args.len() && i < func_vars.len() {
                            func_vars[i] = args[i].clone();
                        }
                    }
                    let mut rest: Vec<Value> = if args.len() > fixed {
                        args[fixed..].to_vec()
                    } else {
                        Vec::new()
                    };
                    // Create row cell for varargin
                    let cell = runmat_builtins::CellArray::new(
                        std::mem::take(&mut rest),
                        1,
                        if args.len() > fixed {
                            args.len() - fixed
                        } else {
                            0
                        },
                    )
                    .map_err(|e| format!("varargin: {e}"))?;
                    if fixed < func_vars.len() {
                        func_vars[fixed] = Value::Cell(cell);
                    }
                } else {
                    for (i, _param_id) in func.params.iter().enumerate() {
                        if i < args.len() && i < func_vars.len() {
                            func_vars[i] = args[i].clone();
                        }
                    }
                }
                // Copy referenced globals into local frame
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
                // Initialize varargout cell if needed
                if func.has_varargout {
                    if let Some(varargout_oid) = func.outputs.last() {
                        if let Some(local_id) = var_map.get(varargout_oid) {
                            if local_id.0 < func_vars.len() {
                                let empty = runmat_builtins::CellArray::new(vec![], 1, 0)
                                    .map_err(|e| format!("varargout init: {e}"))?;
                                func_vars[local_id.0] = Value::Cell(empty);
                            }
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
                let func_bytecode =
                    crate::compile_with_functions(&func_program, &bytecode.functions)?;
                let func_result_vars = match interpret_function_with_counts(
                    &func_bytecode,
                    func_vars,
                    &name,
                    1,
                    arg_count,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        if let Some((catch_pc, catch_var)) = try_stack.pop() {
                            if let Some(var_idx) = catch_var {
                                if var_idx >= vars.len() {
                                    vars.resize(var_idx + 1, Value::Num(0.0));
                                    refresh_workspace_state(&vars);
                                }
                                let runtime_error = build_runtime_error(e.clone()).build();
                                let mex = parse_exception(&runtime_error);
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
                if func.has_varargout {
                    // Single-output call: return first varargout element if any, else 0
                    // For true multi-assign we already have CallFunctionMulti path
                    let first = func
                        .outputs
                        .first()
                        .and_then(|oid| var_map.get(oid))
                        .map(|lid| lid.0)
                        .unwrap_or(0);
                    if let Some(Value::Cell(ca)) = func_result_vars.get(first) {
                        if !ca.data.is_empty() {
                            stack.push((*ca.data[0]).clone());
                        } else {
                            stack.push(Value::Num(0.0));
                        }
                    } else if let Some(v) = func_result_vars.get(first) {
                        stack.push(v.clone());
                    } else {
                        stack.push(Value::Num(0.0));
                    }
                } else if let Some(output_var_id) = func.outputs.first() {
                    let local_output_index = var_map.get(output_var_id).map(|id| id.0).unwrap_or(0);
                    if local_output_index < func_result_vars.len() {
                        stack.push(func_result_vars[local_output_index].clone());
                    } else {
                        stack.push(Value::Num(0.0));
                    }
                } else {
                    vm_bail!(mex(
                        "TooManyOutputs",
                        &format!("Function '{name}' does not return outputs")
                    ));
                }
            }
            Instr::CallFunctionExpandAt(name, before_count, num_indices, after_count) => {
                // Assemble argument list with expansion at position
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
                        _ => return Err(mex("CellIndexType", "Unsupported cell index type")),
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
                        let idx_vals: Vec<Value> = indices
                            .iter()
                            .map(|v| Value::Num((v).try_into().unwrap_or(0.0)))
                            .collect();
                        let cell = runmat_runtime::call_builtin("__make_cell", &idx_vals)?;
                        let v = match runmat_runtime::call_builtin(
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
                match call_builtin(&name, &args) {
                    Ok(v) => stack.push(v),
                    Err(e) => vm_bail!(e),
                }
            }
            Instr::CallFunctionMulti(name, arg_count, out_count) => {
                let func: UserFunction = match bytecode.functions.get(&name) {
                    Some(f) => f.clone(),
                    None => vm_bail!(format!("undefined function: {name}")),
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
                if !func.has_varargin {
                    if arg_count < func.params.len() {
                        vm_bail!(mex(
                            "NotEnoughInputs",
                            &format!(
                                "Function '{name}' expects {} inputs, got {arg_count}",
                                func.params.len()
                            )
                        ));
                    }
                    if arg_count > func.params.len() {
                        vm_bail!(mex(
                            "TooManyInputs",
                            &format!(
                                "Function '{name}' expects {} inputs, got {arg_count}",
                                func.params.len()
                            )
                        ));
                    }
                } else if arg_count + 1 < func.params.len() {
                    vm_bail!(mex(
                        "NotEnoughInputs",
                        &format!(
                            "Function '{name}' expects at least {} inputs, got {arg_count}",
                            func.params.len() - 1
                        )
                    ));
                }
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
                if func.has_varargin {
                    let fixed = func.params.len().saturating_sub(1);
                    for i in 0..fixed {
                        if i < args.len() && i < func_vars.len() {
                            func_vars[i] = args[i].clone();
                        }
                    }
                    let mut rest: Vec<Value> = if args.len() > fixed {
                        args[fixed..].to_vec()
                    } else {
                        Vec::new()
                    };
                    let cell = runmat_builtins::CellArray::new(
                        std::mem::take(&mut rest),
                        1,
                        if args.len() > fixed {
                            args.len() - fixed
                        } else {
                            0
                        },
                    )
                    .map_err(|e| format!("varargin: {e}"))?;
                    if fixed < func_vars.len() {
                        func_vars[fixed] = Value::Cell(cell);
                    }
                } else {
                    for (i, _param_id) in func.params.iter().enumerate() {
                        if i < args.len() && i < func_vars.len() {
                            func_vars[i] = args[i].clone();
                        }
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
                // Initialize varargout cell if needed
                if func.has_varargout {
                    if let Some(varargout_oid) = func.outputs.last() {
                        if let Some(local_id) = var_map.get(varargout_oid) {
                            if local_id.0 < func_vars.len() {
                                let empty = runmat_builtins::CellArray::new(vec![], 1, 0)
                                    .map_err(|e| format!("varargout init: {e}"))?;
                                func_vars[local_id.0] = Value::Cell(empty);
                            }
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
                let func_bytecode =
                    crate::compile_with_functions(&func_program, &bytecode.functions)?;
                let func_result_vars = match interpret_function_with_counts(
                    &func_bytecode,
                    func_vars,
                    &name,
                    out_count,
                    arg_count,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        if let Some((catch_pc, catch_var)) = try_stack.pop() {
                            if let Some(var_idx) = catch_var {
                                if var_idx >= vars.len() {
                                    vars.resize(var_idx + 1, Value::Num(0.0));
                                    refresh_workspace_state(&vars);
                                }
                                let runtime_error = build_runtime_error(e.clone()).build();
                                let mex = parse_exception(&runtime_error);
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
                if func.has_varargout {
                    // Push named outputs first (excluding varargout itself), then fill from varargout cell, then pad with 0.0
                    let total_named = func.outputs.len().saturating_sub(1);
                    let mut pushed = 0usize;
                    // Push named outputs in order
                    for i in 0..total_named.min(out_count) {
                        if let Some(oid) = func.outputs.get(i) {
                            if let Some(local_id) = var_map.get(oid) {
                                let idx = local_id.0;
                                let v = func_result_vars
                                    .get(idx)
                                    .cloned()
                                    .unwrap_or(Value::Num(0.0));
                                stack.push(v);
                                pushed += 1;
                            }
                        }
                    }
                    if pushed < out_count {
                        // Now consume from varargout cell (last output)
                        if let Some(varargout_oid) = func.outputs.last() {
                            if let Some(local_id) = var_map.get(varargout_oid) {
                                if let Some(Value::Cell(ca)) = func_result_vars.get(local_id.0) {
                                    let available = ca.data.len();
                                    let need = out_count - pushed;
                                    if need > available {
                                        vm_bail!(mex("VarargoutMismatch", &format!("Function '{name}' returned {available} varargout values, {need} requested")));
                                    }
                                    for vi in 0..need {
                                        stack.push((*ca.data[vi]).clone());
                                    }
                                }
                            }
                        }
                    }
                    // No padding
                } else {
                    // Push out_count values; error if requesting more than defined
                    let defined = func.outputs.len();
                    if out_count > defined {
                        vm_bail!(mex(
                            "TooManyOutputs",
                            &format!("Function '{name}' defines {defined} outputs, {out_count} requested")
                        ));
                    }
                    for i in 0..out_count {
                        let v = func
                            .outputs
                            .get(i)
                            .and_then(|oid| var_map.get(oid))
                            .map(|lid| lid.0)
                            .and_then(|idx| func_result_vars.get(idx))
                            .cloned()
                            .unwrap_or(Value::Num(0.0));
                        stack.push(v);
                    }
                }
            }
            Instr::CallBuiltinMulti(name, arg_count, out_count) => {
                // Default behavior: try to call builtin; if success, use first output; pad rest with 0.0
                let mut args = Vec::new();
                for _ in 0..arg_count {
                    args.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                args.reverse();
                if name == "gather" {
                    let eval = match runmat_runtime::builtins::acceleration::gpu::gather::evaluate(
                        &args,
                    ) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    let len = eval.len();
                    if out_count == 0 {
                        continue;
                    }
                    if len == 1 {
                        if out_count > 1 {
                            vm_bail!(mex("TooManyOutputs", "gather: too many output arguments"));
                        }
                        stack.push(eval.into_first());
                        continue;
                    }
                    if out_count != len {
                        vm_bail!(mex(
                            "TooManyOutputs",
                            "gather: number of outputs must match number of inputs"
                        ));
                    }
                    for value in eval.into_outputs() {
                        stack.push(value);
                    }
                    continue;
                }
                if name == "meshgrid" {
                    let eval = match runmat_runtime::builtins::array::creation::meshgrid::evaluate(
                        &args,
                    ) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    let available = eval.output_count();
                    if out_count > available {
                        let msg = if available == 2 {
                            "meshgrid with two inputs supports at most two outputs"
                        } else {
                            "meshgrid supports at most three outputs"
                        };
                        vm_bail!(mex("TooManyOutputs", msg));
                    }
                    let first = match eval.first() {
                        Ok(value) => value,
                        Err(err) => vm_bail!(err),
                    };
                    stack.push(first);
                    if out_count >= 2 {
                        let second = match eval.second() {
                            Ok(value) => value,
                            Err(err) => vm_bail!(err),
                        };
                        stack.push(second);
                    }
                    if out_count >= 3 {
                        let third = match eval.third() {
                            Ok(value) => value,
                            Err(err) => vm_bail!(err),
                        };
                        stack.push(third);
                    }
                    continue;
                }
                if name == "load" {
                    let eval = match runmat_runtime::builtins::io::mat::load::evaluate(&args) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        if let Err(err) = assign_loaded_variables(&mut vars, eval.variables()) {
                            vm_bail!(err);
                        }
                        continue;
                    }
                    if out_count > 1 {
                        vm_bail!(mex(
                            "TooManyOutputs",
                            "load supports at most one output argument"
                        ));
                    }
                    stack.push(eval.first_output());
                    for _ in 1..out_count {
                        stack.push(Value::Num(0.0));
                    }
                    continue;
                }
                if name == "fopen" {
                    let eval = match runmat_runtime::builtins::io::filetext::fopen::evaluate(&args)
                    {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    let outputs = eval.outputs();
                    for i in 0..out_count {
                        if let Some(value) = outputs.get(i) {
                            stack.push(value.clone());
                        } else {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "fgets" {
                    if args.is_empty() {
                        vm_bail!(mex(
                            "RuntimeError",
                            "fgets requires at least one input argument"
                        ));
                    }
                    let eval = match runmat_runtime::builtins::io::filetext::fgets::evaluate(
                        &args[0],
                        &args[1..],
                    ) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    let outputs = eval.outputs();
                    for i in 0..out_count {
                        if let Some(value) = outputs.get(i) {
                            stack.push(value.clone());
                        } else {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "fclose" {
                    let eval = match runmat_runtime::builtins::io::filetext::fclose::evaluate(&args)
                    {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    let outputs = eval.outputs();
                    for i in 0..out_count {
                        if let Some(value) = outputs.get(i) {
                            stack.push(value.clone());
                        } else {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "mkdir" {
                    let eval = match runmat_runtime::builtins::io::repl_fs::mkdir::evaluate(&args) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    let outputs = eval.outputs();
                    for i in 0..out_count {
                        if let Some(value) = outputs.get(i) {
                            stack.push(value.clone());
                        } else {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "setenv" {
                    let eval = match runmat_runtime::builtins::io::repl_fs::setenv::evaluate(&args)
                    {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    let outputs = eval.outputs();
                    for i in 0..out_count {
                        if let Some(value) = outputs.get(i) {
                            stack.push(value.clone());
                        } else {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "savepath" {
                    let eval =
                        match runmat_runtime::builtins::io::repl_fs::savepath::evaluate(&args) {
                            Ok(eval) => eval,
                            Err(err) => vm_bail!(err),
                        };
                    if out_count == 0 {
                        continue;
                    }
                    let outputs = eval.outputs();
                    for i in 0..out_count {
                        if let Some(value) = outputs.get(i) {
                            stack.push(value.clone());
                        } else {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "copyfile" {
                    let eval =
                        match runmat_runtime::builtins::io::repl_fs::copyfile::evaluate(&args) {
                            Ok(eval) => eval,
                            Err(err) => vm_bail!(err),
                        };
                    if out_count == 0 {
                        continue;
                    }
                    let outputs = eval.outputs();
                    for i in 0..out_count {
                        if let Some(value) = outputs.get(i) {
                            stack.push(value.clone());
                        } else {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "movefile" {
                    let eval =
                        match runmat_runtime::builtins::io::repl_fs::movefile::evaluate(&args) {
                            Ok(eval) => eval,
                            Err(err) => vm_bail!(err),
                        };
                    if out_count == 0 {
                        continue;
                    }
                    let outputs = eval.outputs();
                    for i in 0..out_count {
                        if let Some(value) = outputs.get(i) {
                            stack.push(value.clone());
                        } else {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "rmdir" {
                    let eval = match runmat_runtime::builtins::io::repl_fs::rmdir::evaluate(&args) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    let outputs = eval.outputs();
                    for i in 0..out_count {
                        if let Some(value) = outputs.get(i) {
                            stack.push(value.clone());
                        } else {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "orderfields" && !args.is_empty() {
                    let eval = match runmat_runtime::builtins::structs::core::orderfields::evaluate(
                        args[0].clone(),
                        &args[1..],
                    ) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    let (ordered, permutation) = eval.into_values();
                    stack.push(ordered);
                    if out_count >= 2 {
                        stack.push(permutation);
                    }
                    if out_count > 2 {
                        for _ in 2..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "chol" {
                    if args.is_empty() {
                        vm_bail!(mex("NotEnoughInputs", "chol requires an input matrix"));
                    }
                    let eval = match runmat_runtime::builtins::math::linalg::factor::chol::evaluate(
                        args[0].clone(),
                        &args[1..],
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            if is_suspend_flow(&err) {
                                for arg in args.iter().rev() {
                                    stack.push(arg.clone());
                                }
                                if let runmat_runtime::RuntimeControlFlow::Suspend(pending) = err {
                                    suspend_pending!({}, pending);
                                }
                            }
                            let runmat_runtime::RuntimeControlFlow::Error(err) = err else {
                                unreachable!("suspend handled above");
                            };
                            vm_bail!(err)
                        }
                    };
                    match out_count {
                        0 => continue,
                        1 => {
                            if !eval.is_positive_definite() {
                                vm_bail!(
                                    runmat_runtime::build_runtime_error(
                                        "Matrix must be positive definite.",
                                    )
                                    .with_builtin("chol")
                                    .build()
                                );
                            }
                            stack.push(eval.factor());
                            continue;
                        }
                        2 => {
                            stack.push(eval.factor());
                            stack.push(eval.flag());
                            continue;
                        }
                        _ => vm_bail!(mex(
                            "TooManyOutputs",
                            "chol currently supports at most two outputs"
                        )),
                    }
                }
                if name == "lu" {
                    if args.is_empty() {
                        vm_bail!(mex("NotEnoughInputs", "lu requires an input matrix"));
                    }
                    let eval = match runmat_runtime::builtins::math::linalg::factor::lu::evaluate(
                        args[0].clone(),
                        &args[1..],
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            if is_suspend_flow(&err) {
                                for arg in args.iter().rev() {
                                    stack.push(arg.clone());
                                }
                                if let runmat_runtime::RuntimeControlFlow::Suspend(pending) = err {
                                    suspend_pending!({}, pending);
                                }
                            }
                            let runmat_runtime::RuntimeControlFlow::Error(err) = err else {
                                unreachable!("suspend handled above");
                            };
                            vm_bail!(err)
                        }
                    };
                    match out_count {
                        0 => continue,
                        1 => {
                            stack.push(eval.combined());
                            continue;
                        }
                        2 => {
                            stack.push(eval.lower());
                            stack.push(eval.upper());
                            continue;
                        }
                        3 => {
                            stack.push(eval.lower());
                            stack.push(eval.upper());
                            stack.push(eval.permutation());
                            continue;
                        }
                        _ => vm_bail!(mex(
                            "TooManyOutputs",
                            "lu currently supports at most three outputs"
                        )),
                    }
                }
                if name == "linsolve" {
                    if args.len() < 2 {
                        vm_bail!(mex(
                            "NotEnoughInputs",
                            "linsolve requires coefficient and right-hand side inputs"
                        ));
                    }
                    let eval =
                        match runmat_runtime::builtins::math::linalg::solve::linsolve::evaluate_args(
                            args[0].clone(),
                            args[1].clone(),
                            &args[2..],
                        ) {
                            Ok(v) => v,
                            Err(err) => vm_bail!(err),
                        };
                    match out_count {
                        0 => continue,
                        1 => {
                            stack.push(eval.solution());
                            continue;
                        }
                        2 => {
                            stack.push(eval.solution());
                            stack.push(eval.reciprocal_condition());
                            continue;
                        }
                        _ => vm_bail!(mex(
                            "TooManyOutputs",
                            "linsolve currently supports at most two outputs"
                        )),
                    }
                }
                if name == "qr" {
                    if args.is_empty() {
                        vm_bail!(mex("NotEnoughInputs", "qr requires an input matrix"));
                    }
                    let eval = match runmat_runtime::builtins::math::linalg::factor::qr::evaluate(
                        args[0].clone(),
                        &args[1..],
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            if is_suspend_flow(&err) {
                                for arg in args.iter().rev() {
                                    stack.push(arg.clone());
                                }
                                if let runmat_runtime::RuntimeControlFlow::Suspend(pending) = err {
                                    suspend_pending!({}, pending);
                                }
                            }
                            let runmat_runtime::RuntimeControlFlow::Error(err) = err else {
                                unreachable!("suspend handled above");
                            };
                            vm_bail!(err)
                        }
                    };
                    match out_count {
                        0 => {
                            pc += 1;
                            continue;
                        }
                        1 => {
                            stack.push(eval.r());
                            pc += 1;
                            continue;
                        }
                        2 => {
                            stack.push(eval.q());
                            stack.push(eval.r());
                            pc += 1;
                            continue;
                        }
                        3 => {
                            stack.push(eval.q());
                            stack.push(eval.r());
                            stack.push(eval.permutation());
                            pc += 1;
                            continue;
                        }
                        _ => vm_bail!(mex(
                            "TooManyOutputs",
                            "qr currently supports at most three outputs"
                        )),
                    }
                }
                if name == "svd" {
                    if args.is_empty() {
                        vm_bail!(mex("NotEnoughInputs", "svd requires an input matrix"));
                    }
                    let eval = match runmat_runtime::builtins::math::linalg::factor::svd::evaluate(
                        args[0].clone(),
                        &args[1..],
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            if is_suspend_flow(&err) {
                                for arg in args.iter().rev() {
                                    stack.push(arg.clone());
                                }
                                if let runmat_runtime::RuntimeControlFlow::Suspend(pending) = err {
                                    suspend_pending!({}, pending);
                                }
                            }
                            let runmat_runtime::RuntimeControlFlow::Error(err) = err else {
                                unreachable!("suspend handled above");
                            };
                            vm_bail!(err)
                        }
                    };
                    match out_count {
                        0 => continue,
                        1 => {
                            stack.push(eval.singular_values());
                            continue;
                        }
                        2 => {
                            stack.push(eval.u());
                            stack.push(eval.sigma());
                            continue;
                        }
                        3 => {
                            stack.push(eval.u());
                            stack.push(eval.sigma());
                            stack.push(eval.v());
                            continue;
                        }
                        _ => vm_bail!(mex(
                            "TooManyOutputs",
                            "svd currently supports at most three outputs"
                        )),
                    }
                }
                if name == "eig" {
                    if args.is_empty() {
                        vm_bail!(mex("NotEnoughInputs", "eig requires an input matrix"));
                    }
                    let require_left = out_count >= 3;
                    let eval = match runmat_runtime::builtins::math::linalg::factor::eig::evaluate(
                        args[0].clone(),
                        &args[1..],
                        require_left,
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            if is_suspend_flow(&err) {
                                for arg in args.iter().rev() {
                                    stack.push(arg.clone());
                                }
                                if let runmat_runtime::RuntimeControlFlow::Suspend(pending) = err {
                                    suspend_pending!({}, pending);
                                }
                            }
                            let runmat_runtime::RuntimeControlFlow::Error(err) = err else {
                                unreachable!("suspend handled above");
                            };
                            vm_bail!(err)
                        }
                    };
                    match out_count {
                        0 => continue,
                        1 => {
                            stack.push(eval.eigenvalues());
                            continue;
                        }
                        2 => {
                            stack.push(eval.right());
                            stack.push(eval.diagonal());
                            continue;
                        }
                        3 => {
                            stack.push(eval.right());
                            stack.push(eval.diagonal());
                            let left = match eval.left() {
                                Ok(value) => value,
                                Err(err) => {
                                    if is_suspend_flow(&err) {
                                        if let runmat_runtime::RuntimeControlFlow::Suspend(pending) =
                                            err
                                        {
                                            suspend_pending!({}, pending);
                                        }
                                    }
                                    let runmat_runtime::RuntimeControlFlow::Error(err) = err else {
                                        unreachable!("suspend handled above");
                                    };
                                    vm_bail!(err)
                                }
                            };
                            stack.push(left);
                            continue;
                        }
                        _ => vm_bail!(mex(
                            "TooManyOutputs",
                            "eig currently supports at most three outputs"
                        )),
                    }
                }
                // Special-case for 'find' to support [i,j,v] = find(A)
                if name == "find" && !args.is_empty() {
                    let eval = match runmat_runtime::builtins::array::indexing::find::evaluate(
                        args[0].clone(),
                        &args[1..],
                    ) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    if out_count <= 1 {
                        let linear = match eval.linear_value() {
                            Ok(v) => v,
                            Err(err) => vm_bail!(err),
                        };
                        stack.push(linear);
                        for _ in 1..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    } else {
                        let rows = match eval.row_value() {
                            Ok(v) => v,
                            Err(err) => vm_bail!(err),
                        };
                        stack.push(rows);
                        let cols = match eval.column_value() {
                            Ok(v) => v,
                            Err(err) => vm_bail!(err),
                        };
                        stack.push(cols);
                        if out_count >= 3 {
                            let vals = match eval.values_value() {
                                Ok(v) => v,
                                Err(err) => vm_bail!(err),
                            };
                            stack.push(vals);
                        }
                        if out_count > 3 {
                            for _ in 3..out_count {
                                stack.push(Value::Num(0.0));
                            }
                        }
                    }
                    continue;
                }
                if name == "regexp" && args.len() >= 2 {
                    let eval = match runmat_runtime::builtins::strings::regex::regexp::evaluate(
                        args[0].clone(),
                        args[1].clone(),
                        &args[2..],
                    ) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    let mut values = match eval.outputs_for_multi() {
                        Ok(values) => values,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    for _ in 0..out_count {
                        if !values.is_empty() {
                            stack.push(values.remove(0));
                        } else {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "deconv" {
                    if args.len() < 2 {
                        vm_bail!(mex("MATLAB:minrhs", "Not enough input arguments."));
                    }
                    let eval = match runmat_runtime::builtins::math::signal::deconv::evaluate(
                        args[0].clone(),
                        args[1].clone(),
                    ) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    stack.push(eval.quotient());
                    if out_count >= 2 {
                        stack.push(eval.remainder());
                    }
                    if out_count > 2 {
                        for _ in 2..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "polyder" {
                    if args.is_empty() {
                        vm_bail!(mex("MATLAB:minrhs", "Not enough input arguments."));
                    }
                    if out_count <= 1 {
                        let result = match args.len() {
                            1 => runmat_runtime::builtins::math::poly::polyder::derivative_single(
                                args[0].clone(),
                            ),
                            2 => runmat_runtime::builtins::math::poly::polyder::derivative_product(
                                args[0].clone(),
                                args[1].clone(),
                            ),
                            _ => vm_bail!("polyder: too many input arguments.".to_string()),
                        };
                        match result {
                            Ok(value) => {
                                if out_count == 0 {
                                    continue;
                                }
                                stack.push(value);
                            }
                            Err(err) => vm_bail!(err),
                        }
                        if out_count > 1 {
                            for _ in 1..out_count {
                                stack.push(Value::Num(0.0));
                            }
                        }
                        continue;
                    }
                    if args.len() != 2 {
                        vm_bail!(mex(
                            "MATLAB:minrhs",
                            "Not enough input arguments for quotient form."
                        ));
                    }
                    let eval =
                        match runmat_runtime::builtins::math::poly::polyder::evaluate_quotient(
                            args[0].clone(),
                            args[1].clone(),
                        ) {
                            Ok(eval) => eval,
                            Err(err) => vm_bail!(err),
                        };
                    stack.push(eval.numerator());
                    stack.push(eval.denominator());
                    if out_count > 2 {
                        for _ in 2..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "polyval" {
                    if args.len() < 2 {
                        vm_bail!(mex("MATLAB:minrhs", "Not enough input arguments."));
                    }
                    let eval = match runmat_runtime::builtins::math::poly::polyval::evaluate(
                        args[0].clone(),
                        args[1].clone(),
                        &args[2..],
                        out_count >= 2,
                    ) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    stack.push(eval.value());
                    if out_count >= 2 {
                        let delta = match eval.delta() {
                            Ok(v) => v,
                            Err(err) => vm_bail!(err),
                        };
                        stack.push(delta);
                    }
                    if out_count > 2 {
                        for _ in 2..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "polyfit" {
                    if args.len() < 3 {
                        vm_bail!(mex("MATLAB:minrhs", "Not enough input arguments."));
                    }
                    let eval = match runmat_runtime::builtins::math::poly::polyfit::evaluate(
                        args[0].clone(),
                        args[1].clone(),
                        args[2].clone(),
                        &args[3..],
                    ) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    stack.push(eval.coefficients());
                    if out_count >= 2 {
                        stack.push(eval.stats());
                    }
                    if out_count >= 3 {
                        stack.push(eval.mu());
                    }
                    if out_count > 3 {
                        for _ in 3..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "filter" {
                    if args.len() < 3 {
                        vm_bail!(mex("MATLAB:minrhs", "Not enough input arguments."));
                    }
                    let eval = match runmat_runtime::builtins::math::signal::filter::evaluate(
                        args[0].clone(),
                        args[1].clone(),
                        args[2].clone(),
                        &args[3..],
                    ) {
                        Ok(eval) => eval,
                        Err(err) => {
                            if is_suspend_flow(&err) {
                                for arg in args.iter().rev() {
                                    stack.push(arg.clone());
                                }
                                match err {
                                    runmat_runtime::RuntimeControlFlow::Suspend(pending) => {
                                        suspend_pending!({}, pending);
                                    }
                                    runmat_runtime::RuntimeControlFlow::Error(_) => {}
                                }
                            }
                            let runmat_runtime::RuntimeControlFlow::Error(e) = err else {
                                unreachable!("suspend handled above");
                            };
                            vm_bail!(e)
                        }
                    };
                    if out_count == 0 {
                        continue;
                    }
                    if out_count == 1 {
                        stack.push(eval.into_value());
                    } else {
                        let (output, final_state) = eval.into_pair();
                        stack.push(output);
                        stack.push(final_state);
                        if out_count > 2 {
                            for _ in 2..out_count {
                                stack.push(Value::Num(0.0));
                            }
                        }
                    }
                    continue;
                }
                if name == "sort" && !args.is_empty() {
                    let eval = match runmat_runtime::builtins::array::sorting_sets::sort::evaluate(
                        args[0].clone(),
                        &args[1..],
                    ) {
                        Ok(eval) => eval,
                        Err(err) => {
                            if is_suspend_flow(&err) {
                                for arg in args.iter().rev() {
                                    stack.push(arg.clone());
                                }
                                match err {
                                    runmat_runtime::RuntimeControlFlow::Suspend(pending) => {
                                        suspend_pending!({}, pending);
                                    }
                                    runmat_runtime::RuntimeControlFlow::Error(_) => {}
                                }
                            }
                            let runmat_runtime::RuntimeControlFlow::Error(e) = err else {
                                unreachable!("suspend handled above");
                            };
                            vm_bail!(e)
                        }
                    };
                    if out_count == 0 {
                        continue;
                    }
                    let (sorted, indices) = eval.into_values();
                    stack.push(sorted);
                    if out_count >= 2 {
                        stack.push(indices);
                    }
                    if out_count > 2 {
                        for _ in 2..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "cummin" && !args.is_empty() {
                    let eval = match runmat_runtime::builtins::math::reduction::evaluate_cummin(
                        args[0].clone(),
                        &args[1..],
                    ) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    let (values, indices) = eval.into_pair();
                    stack.push(values);
                    if out_count >= 2 {
                        stack.push(indices);
                    }
                    if out_count > 2 {
                        for _ in 2..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "min" && !args.is_empty() {
                    let eval = match runmat_runtime::builtins::math::reduction::evaluate_min(
                        args[0].clone(),
                        &args[1..],
                    ) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    let (values, indices) = eval.into_pair();
                    stack.push(values);
                    if out_count >= 2 {
                        stack.push(indices);
                    }
                    if out_count > 2 {
                        for _ in 2..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "sortrows" && !args.is_empty() {
                    let eval =
                        match runmat_runtime::builtins::array::sorting_sets::sortrows::evaluate(
                            args[0].clone(),
                            &args[1..],
                        ) {
                            Ok(eval) => eval,
                            Err(err) => {
                                if is_suspend_flow(&err) {
                                    for arg in args.iter().rev() {
                                        stack.push(arg.clone());
                                    }
                                    match err {
                                        runmat_runtime::RuntimeControlFlow::Suspend(pending) => {
                                            suspend_pending!({}, pending);
                                        }
                                        runmat_runtime::RuntimeControlFlow::Error(_) => {}
                                    }
                                }
                                let runmat_runtime::RuntimeControlFlow::Error(e) = err else {
                                    unreachable!("suspend handled above");
                                };
                                vm_bail!(e)
                            }
                        };
                    if out_count == 0 {
                        continue;
                    }
                    let (sorted, indices) = eval.into_values();
                    stack.push(sorted);
                    if out_count >= 2 {
                        stack.push(indices);
                    }
                    if out_count > 2 {
                        for _ in 2..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "ismember" && args.len() >= 2 {
                    let eval =
                        match runmat_runtime::builtins::array::sorting_sets::ismember::evaluate(
                            args[0].clone(),
                            args[1].clone(),
                            &args[2..],
                        ) {
                            Ok(eval) => eval,
                            Err(err) => {
                                if is_suspend_flow(&err) {
                                    for arg in args.iter().rev() {
                                        stack.push(arg.clone());
                                    }
                                    match err {
                                        runmat_runtime::RuntimeControlFlow::Suspend(pending) => {
                                            suspend_pending!({}, pending);
                                        }
                                        runmat_runtime::RuntimeControlFlow::Error(_) => {}
                                    }
                                }
                                let runmat_runtime::RuntimeControlFlow::Error(e) = err else {
                                    unreachable!("suspend handled above");
                                };
                                vm_bail!(e)
                            }
                        };
                    if out_count == 0 {
                        continue;
                    }
                    if out_count == 1 {
                        stack.push(eval.into_mask_value());
                        continue;
                    }
                    let (mask, loc) = eval.into_pair();
                    stack.push(mask);
                    stack.push(loc);
                    if out_count > 2 {
                        for _ in 2..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "intersect" && args.len() >= 2 {
                    let eval =
                        match runmat_runtime::builtins::array::sorting_sets::intersect::evaluate(
                            args[0].clone(),
                            args[1].clone(),
                            &args[2..],
                        ) {
                            Ok(eval) => eval,
                            Err(err) => {
                                if is_suspend_flow(&err) {
                                    for arg in args.iter().rev() {
                                        stack.push(arg.clone());
                                    }
                                    match err {
                                        runmat_runtime::RuntimeControlFlow::Suspend(pending) => {
                                            suspend_pending!({}, pending);
                                        }
                                        runmat_runtime::RuntimeControlFlow::Error(_) => {}
                                    }
                                }
                                let runmat_runtime::RuntimeControlFlow::Error(e) = err else {
                                    unreachable!("suspend handled above");
                                };
                                vm_bail!(e)
                            }
                        };
                    if out_count == 0 {
                        continue;
                    }
                    if out_count == 1 {
                        stack.push(eval.into_values_value());
                        continue;
                    }
                    if out_count == 2 {
                        let (values, ia) = eval.into_pair();
                        stack.push(values);
                        stack.push(ia);
                        continue;
                    }
                    let (values, ia, ib) = eval.into_triple();
                    stack.push(values);
                    stack.push(ia);
                    stack.push(ib);
                    if out_count > 3 {
                        for _ in 3..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "union" && args.len() >= 2 {
                    let eval = match runmat_runtime::builtins::array::sorting_sets::union::evaluate(
                        args[0].clone(),
                        args[1].clone(),
                        &args[2..],
                    ) {
                        Ok(eval) => eval,
                        Err(err) => {
                            if is_suspend_flow(&err) {
                                for arg in args.iter().rev() {
                                    stack.push(arg.clone());
                                }
                                match err {
                                    runmat_runtime::RuntimeControlFlow::Suspend(pending) => {
                                        suspend_pending!({}, pending);
                                    }
                                    runmat_runtime::RuntimeControlFlow::Error(_) => {}
                                }
                            }
                            let runmat_runtime::RuntimeControlFlow::Error(e) = err else {
                                unreachable!("suspend handled above");
                            };
                            vm_bail!(e)
                        }
                    };
                    if out_count == 0 {
                        continue;
                    }
                    if out_count == 1 {
                        stack.push(eval.into_values_value());
                        continue;
                    }
                    if out_count == 2 {
                        let (values, ia) = eval.into_pair();
                        stack.push(values);
                        stack.push(ia);
                        continue;
                    }
                    let (values, ia, ib) = eval.into_triple();
                    stack.push(values);
                    stack.push(ia);
                    stack.push(ib);
                    if out_count > 3 {
                        for _ in 3..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                #[cfg(feature = "plot-core")]
                {
                    if name == "hist" && !args.is_empty() {
                        let eval = match runmat_runtime::builtins::plotting::ops::hist::evaluate(
                            args[0].clone(),
                            &args[1..],
                        ) {
                            Ok(eval) => eval,
                            Err(err) => vm_bail!(err.to_string()),
                        };
                        if let Err(err) = eval.render_plot() {
                            vm_bail!(err.to_string());
                        }
                        if out_count == 0 {
                            continue;
                        }
                        if out_count == 1 {
                            stack.push(eval.counts_value());
                            continue;
                        }
                        stack.push(eval.counts_value());
                        stack.push(eval.centers_value());
                        if out_count > 2 {
                            for _ in 2..out_count {
                                stack.push(Value::Num(0.0));
                            }
                        }
                        continue;
                    }
                }
                #[cfg(not(feature = "plot-core"))]
                {
                    if name == "hist" {
                        vm_bail!("hist requires plot-core feature".to_string());
                    }
                }
                if name == "histcounts" && !args.is_empty() {
                    let eval = match runmat_runtime::builtins::stats::hist::histcounts::evaluate(
                        args[0].clone(),
                        &args[1..],
                    ) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err.to_string()),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    if out_count == 1 {
                        stack.push(eval.into_counts_value());
                        continue;
                    }
                    let (counts, edges) = eval.into_pair();
                    stack.push(counts);
                    stack.push(edges);
                    if out_count > 2 {
                        for _ in 2..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "histcounts2" && args.len() >= 2 {
                    let eval = match runmat_runtime::builtins::stats::hist::histcounts2::evaluate(
                        args[0].clone(),
                        args[1].clone(),
                        &args[2..],
                    ) {
                        Ok(eval) => eval,
                        Err(err) => vm_bail!(err.to_string()),
                    };
                    if out_count == 0 {
                        continue;
                    }
                    if out_count == 1 {
                        stack.push(eval.into_counts_value());
                        continue;
                    }
                    if out_count == 2 {
                        let (counts, xedges) = eval.into_pair();
                        stack.push(counts);
                        stack.push(xedges);
                        continue;
                    }
                    let (counts, xedges, yedges) = eval.into_triple();
                    stack.push(counts);
                    stack.push(xedges);
                    stack.push(yedges);
                    if out_count > 3 {
                        for _ in 3..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                if name == "unique" && !args.is_empty() {
                    let eval = match runmat_runtime::builtins::array::sorting_sets::unique::evaluate(
                        args[0].clone(),
                        &args[1..],
                    ) {
                        Ok(eval) => eval,
                        Err(err) => {
                            if is_suspend_flow(&err) {
                                for arg in args.iter().rev() {
                                    stack.push(arg.clone());
                                }
                                match err {
                                    runmat_runtime::RuntimeControlFlow::Suspend(pending) => {
                                        suspend_pending!({}, pending);
                                    }
                                    runmat_runtime::RuntimeControlFlow::Error(_) => {}
                                }
                            }
                            let runmat_runtime::RuntimeControlFlow::Error(e) = err else {
                                unreachable!("suspend handled above");
                            };
                            vm_bail!(e)
                        }
                    };
                    if out_count == 0 {
                        continue;
                    }
                    if out_count == 1 {
                        stack.push(eval.into_values_value());
                        continue;
                    }
                    if out_count == 2 {
                        let (values, ia) = eval.into_pair();
                        stack.push(values);
                        stack.push(ia);
                        continue;
                    }
                    let (values, ia, ic) = eval.into_triple();
                    stack.push(values);
                    stack.push(ia);
                    stack.push(ic);
                    if out_count > 3 {
                        for _ in 3..out_count {
                            stack.push(Value::Num(0.0));
                        }
                    }
                    continue;
                }
                match call_builtin(&name, &args) {
                    Ok(v) => match v {
                        Value::Tensor(t) => {
                            let mut pushed = 0usize;
                            for &val in t.data.iter() {
                                if pushed >= out_count {
                                    break;
                                }
                                stack.push(Value::Num(val));
                                pushed += 1;
                            }
                            for _ in pushed..out_count {
                                stack.push(Value::Num(0.0));
                            }
                        }
                        Value::Cell(ca) => {
                            let mut pushed = 0usize;
                            for v in &ca.data {
                                if pushed >= out_count {
                                    break;
                                }
                                stack.push((**v).clone());
                                pushed += 1;
                            }
                            for _ in pushed..out_count {
                                stack.push(Value::Num(0.0));
                            }
                        }
                        other => {
                            stack.push(other);
                            for _ in 1..out_count {
                                stack.push(Value::Num(0.0));
                            }
                        }
                    },
                    Err(e) => {
                        // Try wildcard imports resolution similar to CallBuiltin
                        let mut resolved = None;
                        for (path, wildcard) in &imports {
                            if !*wildcard {
                                continue;
                            }
                            let mut qual = String::new();
                            for (i, part) in path.iter().enumerate() {
                                if i > 0 {
                                    qual.push('.');
                                }
                                qual.push_str(part);
                            }
                            qual.push('.');
                            qual.push_str(&name);
                            if let Ok(v) = call_builtin(&qual, &args) {
                                resolved = Some(v);
                                break;
                            }
                        }
                        if let Some(v) = resolved {
                            match v {
                                Value::Tensor(t) => {
                                    let mut pushed = 0usize;
                                    for &val in t.data.iter() {
                                        if pushed >= out_count {
                                            break;
                                        }
                                        stack.push(Value::Num(val));
                                        pushed += 1;
                                    }
                                    for _ in pushed..out_count {
                                        stack.push(Value::Num(0.0));
                                    }
                                }
                                Value::Cell(ca) => {
                                    let mut pushed = 0usize;
                                    for v in &ca.data {
                                        if pushed >= out_count {
                                            break;
                                        }
                                        stack.push((**v).clone());
                                        pushed += 1;
                                    }
                                    for _ in pushed..out_count {
                                        stack.push(Value::Num(0.0));
                                    }
                                }
                                other => {
                                    stack.push(other);
                                    for _ in 1..out_count {
                                        stack.push(Value::Num(0.0));
                                    }
                                }
                            }
                        } else {
                            vm_bail!(e.to_string());
                        }
                    }
                }
            }
            Instr::EnterTry(catch_pc, catch_var) => {
                try_stack.push((catch_pc, catch_var));
            }
            Instr::PopTry => {
                try_stack.pop();
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
                let result = runmat_runtime::create_matrix_from_values(&rows_data)?;
                stack.push(result);
            }
            Instr::CreateRange(has_step) => {
                if has_step {
                    let end: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    let step: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    let start: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    let range_result = runmat_runtime::create_range(start, Some(step), end)?;
                    stack.push(range_result);
                } else {
                    let end: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    let start: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    let range_result = runmat_runtime::create_range(start, None, end)?;
                    stack.push(range_result);
                }
            }
            Instr::Index(num_indices) => {
                let mut indices = Vec::new();
                let count = num_indices;
                for _ in 0..count {
                    let index_val: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    indices.push(index_val);
                }
                indices.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                #[cfg(feature = "native-accel")]
                clear_residency(&base);
                match base {
                    Value::Object(obj) => {
                        let cell = runmat_builtins::CellArray::new(
                            indices.iter().map(|n| Value::Num(*n)).collect(),
                            1,
                            indices.len(),
                        )
                        .map_err(|e| format!("subsref build error: {e}"))?;
                        match runmat_runtime::call_builtin(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("()".to_string()),
                                Value::Cell(cell),
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e.to_string()),
                        }
                    }
                    other => {
                        let result = match runmat_runtime::perform_indexing(&other, &indices) {
                            Ok(v) => v,
                            Err(e) => vm_bail!(e.to_string()),
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
                        let cell =
                            runmat_builtins::CellArray::new(numeric.to_vec(), 1, numeric.len())
                                .map_err(|e| format!("subsref build error: {e}"))?;
                        match runmat_runtime::call_builtin(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("()".to_string()),
                                Value::Cell(cell),
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e.to_string()),
                        }
                    }
                    Value::Tensor(t) => {
                        let rank = t.shape.len();
                        // Build per-dimension selectors
                        #[derive(Clone)]
                        enum Sel {
                            Colon,
                            Scalar(usize),
                            Indices(Vec<usize>),
                        }
                        let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        if dims == 1 {
                            let total = t.data.len();
                            let mut idxs: Vec<usize> = Vec::new();
                            let is_colon = (colon_mask & 1u32) != 0;
                            let is_end = (end_mask & 1u32) != 0;
                            if is_colon {
                                idxs = (1..=total).collect();
                            } else if is_end {
                                idxs = vec![total];
                            } else if let Some(v) = numeric.first() {
                                match v {
                                    Value::Num(n) => {
                                        let i = *n as isize;
                                        if i < 1 {
                                            vm_bail!(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds"
                                            ));
                                        }
                                        idxs = vec![i as usize];
                                    }
                                    Value::Tensor(idx_t) => {
                                        let len = idx_t.shape.iter().product::<usize>();
                                        if len == total {
                                            for (i, &val) in idx_t.data.iter().enumerate() {
                                                if val != 0.0 {
                                                    idxs.push(i + 1);
                                                }
                                            }
                                        } else {
                                            for &val in &idx_t.data {
                                                let i = val as isize;
                                                if i < 1 {
                                                    vm_bail!(mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds"
                                                    ));
                                                }
                                                idxs.push(i as usize);
                                            }
                                        }
                                    }
                                    _ => vm_bail!(mex(
                                        "UnsupportedIndexType",
                                        "Unsupported index type"
                                    )),
                                }
                            } else {
                                vm_bail!(mex("MissingNumericIndex", "missing numeric index"));
                            }
                            if idxs.iter().any(|&i| i == 0 || i > total) {
                                vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            if idxs.len() == 1 {
                                stack.push(Value::Num(t.data[idxs[0] - 1]));
                            } else {
                                let mut out = Vec::with_capacity(idxs.len());
                                for &i in &idxs {
                                    out.push(t.data[i - 1]);
                                }
                                let tens = runmat_builtins::Tensor::new(out, vec![idxs.len(), 1])
                                    .map_err(|e| format!("Slice error: {e}"))?;
                                stack.push(Value::Tensor(tens));
                            }
                        } else {
                            for d in 0..dims {
                                let is_colon = (colon_mask & (1u32 << d)) != 0;
                                let is_end = (end_mask & (1u32 << d)) != 0;
                                if is_colon {
                                    selectors.push(Sel::Colon);
                                } else if is_end {
                                    // Plain 'end' -> scalar size of this dim
                                    let dim_len = *t.shape.get(d).unwrap_or(&1);
                                    selectors.push(Sel::Scalar(dim_len));
                                } else {
                                    let v = numeric.get(num_iter).ok_or(mex(
                                        "MissingNumericIndex",
                                        "missing numeric index",
                                    ))?;
                                    num_iter += 1;
                                    match v {
                                        Value::Num(n) => {
                                            let idx = *n as isize;
                                            if idx < 1 {
                                                return Err(mex(
                                                    "IndexOutOfBounds",
                                                    "Index out of bounds",
                                                ));
                                            }
                                            selectors.push(Sel::Scalar(idx as usize));
                                        }
                                        Value::Tensor(idx_t) => {
                                            // Logical mask if length matches dimension
                                            let dim_len = *t.shape.get(d).unwrap_or(&1);
                                            let len = idx_t.shape.iter().product::<usize>();
                                            if len == dim_len {
                                                let mut indices = Vec::new();
                                                for (i, &val) in idx_t.data.iter().enumerate() {
                                                    if val != 0.0 {
                                                        indices.push(i + 1);
                                                    }
                                                }
                                                selectors.push(Sel::Indices(indices));
                                            } else {
                                                // Treat as explicit indices (1-based)
                                                let mut indices = Vec::with_capacity(len);
                                                for &val in &idx_t.data {
                                                    let idx = val as isize;
                                                    if idx < 1 {
                                                        return Err(mex(
                                                            "IndexOutOfBounds",
                                                            "Index out of bounds",
                                                        ));
                                                    }
                                                    indices.push(idx as usize);
                                                }
                                                selectors.push(Sel::Indices(indices));
                                            }
                                        }
                                        Value::LogicalArray(la) => {
                                            let dim_len = *t.shape.get(d).unwrap_or(&1);
                                            if la.data.len() == dim_len {
                                                let mut indices = Vec::new();
                                                for (i, &b) in la.data.iter().enumerate() {
                                                    if b != 0 {
                                                        indices.push(i + 1);
                                                    }
                                                }
                                                selectors.push(Sel::Indices(indices));
                                            } else {
                                                return Err(mex(
                                                    "IndexShape",
                                                    "Logical mask shape mismatch",
                                                ));
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
                            // 2-D fast paths
                            if dims == 2 {
                                let rows = if rank >= 1 { t.shape[0] } else { 1 };
                                let cols = if rank >= 2 { t.shape[1] } else { 1 };
                                match (&selectors[0], &selectors[1]) {
                                    // Full column
                                    (Sel::Colon, Sel::Scalar(j)) => {
                                        let j0 = *j - 1;
                                        if j0 >= cols {
                                            return Err(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds",
                                            ));
                                        }
                                        let start = j0 * rows;
                                        let out = t.data[start..start + rows].to_vec();
                                        if out.len() == 1 {
                                            stack.push(Value::Num(out[0]));
                                        } else {
                                            let tens =
                                                runmat_builtins::Tensor::new(out, vec![rows, 1])
                                                    .map_err(|e| format!("Slice error: {e}"))?;
                                            stack.push(Value::Tensor(tens));
                                        }
                                        bench_end("IndexSlice2D.fast_col", __b);
                                        pc += 1;
                                        continue;
                                    }
                                    // Full row
                                    (Sel::Scalar(i), Sel::Colon) => {
                                        let i0 = *i - 1;
                                        if i0 >= rows {
                                            return Err(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds",
                                            ));
                                        }
                                        let mut out: Vec<f64> = Vec::with_capacity(cols);
                                        for c in 0..cols {
                                            out.push(t.data[i0 + c * rows]);
                                        }
                                        if out.len() == 1 {
                                            stack.push(Value::Num(out[0]));
                                        } else {
                                            let tens =
                                                runmat_builtins::Tensor::new(out, vec![1, cols])
                                                    .map_err(|e| format!("Slice error: {e}"))?;
                                            stack.push(Value::Tensor(tens));
                                        }
                                        bench_end("IndexSlice2D.fast_row", __b);
                                        pc += 1;
                                        continue;
                                    }
                                    // Full columns subset: A(:, J)
                                    (Sel::Colon, Sel::Indices(js)) => {
                                        // Gather selected full columns into a [rows, |J|] tensor
                                        if js.is_empty() {
                                            let tens = runmat_builtins::Tensor::new(
                                                Vec::new(),
                                                vec![rows, 0],
                                            )
                                            .map_err(|e| format!("Slice error: {e}"))?;
                                            stack.push(Value::Tensor(tens));
                                        } else {
                                            let mut out: Vec<f64> =
                                                Vec::with_capacity(rows * js.len());
                                            for &j in js {
                                                let j0 = j - 1;
                                                if j0 >= cols {
                                                    return Err(mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds",
                                                    ));
                                                }
                                                let start = j0 * rows;
                                                out.extend_from_slice(&t.data[start..start + rows]);
                                            }
                                            let tens = runmat_builtins::Tensor::new(
                                                out,
                                                vec![rows, js.len()],
                                            )
                                            .map_err(|e| format!("Slice error: {e}"))?;
                                            stack.push(Value::Tensor(tens));
                                        }
                                        bench_end("IndexSlice2D.fast_cols", __b);
                                        pc += 1;
                                        continue;
                                    }
                                    // Selected rows full: A(I, :)
                                    (Sel::Indices(is), Sel::Colon) => {
                                        // Gather selected rows across all columns into [|I|, cols]
                                        if is.is_empty() {
                                            let tens = runmat_builtins::Tensor::new(
                                                Vec::new(),
                                                vec![0, cols],
                                            )
                                            .map_err(|e| format!("Slice error: {e}"))?;
                                            stack.push(Value::Tensor(tens));
                                        } else {
                                            let mut out: Vec<f64> =
                                                Vec::with_capacity(is.len() * cols);
                                            for c in 0..cols {
                                                for &i in is {
                                                    let i0 = i - 1;
                                                    if i0 >= rows {
                                                        return Err(mex(
                                                            "IndexOutOfBounds",
                                                            "Index out of bounds",
                                                        ));
                                                    }
                                                    out.push(t.data[i0 + c * rows]);
                                                }
                                            }
                                            let tens = runmat_builtins::Tensor::new(
                                                out,
                                                vec![is.len(), cols],
                                            )
                                            .map_err(|e| format!("Slice error: {e}"))?;
                                            stack.push(Value::Tensor(tens));
                                        }
                                        bench_end("IndexSlice2D.fast_rows_multi", __b);
                                        pc += 1;
                                        continue;
                                    }
                                    _ => {}
                                }
                            }
                            {
                                // Compute output shape and gather
                                let mut out_dims: Vec<usize> = Vec::new();
                                let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                                for (d, sel) in selectors.iter().enumerate().take(dims) {
                                    let dim_len = *t.shape.get(d).unwrap_or(&1);
                                    let idxs = match sel {
                                        Sel::Colon => (1..=dim_len).collect::<Vec<usize>>(),
                                        Sel::Scalar(i) => vec![*i],
                                        Sel::Indices(v) => v.clone(),
                                    };
                                    if idxs.iter().any(|&i| i == 0 || i > dim_len) {
                                        return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                                    }
                                    if idxs.len() > 1 {
                                        out_dims.push(idxs.len());
                                    } else {
                                        out_dims.push(1);
                                    }
                                    per_dim_indices.push(idxs);
                                }
                                let mut out_dims: Vec<usize> =
                                    per_dim_indices.iter().map(|v| v.len()).collect();
                                // 2D mixed selectors shape correction to match MATLAB:
                                // (I, scalar) => column vector [len(I), 1]; (scalar, J) => row vector [1, len(J)]
                                if dims == 2 {
                                    match (
                                        &per_dim_indices[0].as_slice(),
                                        &per_dim_indices[1].as_slice(),
                                    ) {
                                        // I (len>1), scalar
                                        (i_list, j_list)
                                            if i_list.len() > 1 && j_list.len() == 1 =>
                                        {
                                            out_dims = vec![i_list.len(), 1];
                                        }
                                        // scalar, J (len>1)
                                        (i_list, j_list)
                                            if i_list.len() == 1 && j_list.len() > 1 =>
                                        {
                                            out_dims = vec![1, j_list.len()];
                                        }
                                        _ => {}
                                    }
                                }
                                // Strides for column-major order (first dimension fastest)
                                let mut strides: Vec<usize> = vec![0; dims];
                                let full_shape: Vec<usize> = if rank < dims {
                                    let mut s = t.shape.clone();
                                    s.resize(dims, 1);
                                    s
                                } else {
                                    t.shape.clone()
                                };
                                let mut acc = 1usize;
                                for (d, stride) in strides.iter_mut().enumerate().take(dims) {
                                    *stride = acc;
                                    acc *= full_shape[d];
                                }
                                // Cartesian product gather
                                let total_out: usize = out_dims.iter().product();
                                let mut out_data: Vec<f64> = Vec::with_capacity(total_out);
                                if out_dims.contains(&0)
                                    || per_dim_indices.iter().any(|v| v.is_empty())
                                {
                                    // Empty selection on some dimension -> empty tensor
                                    let out_tensor =
                                        runmat_builtins::Tensor::new(out_data, out_dims)
                                            .map_err(|e| format!("Slice error: {e}"))?;
                                    stack.push(Value::Tensor(out_tensor));
                                } else {
                                    fn cartesian<F: FnMut(&[usize])>(
                                        lists: &[Vec<usize>],
                                        mut f: F,
                                    ) {
                                        let dims = lists.len();
                                        let mut idx = vec![0usize; dims];
                                        loop {
                                            let current: Vec<usize> =
                                                (0..dims).map(|d| lists[d][idx[d]]).collect();
                                            f(&current);
                                            // Increment first dimension fastest (column-major order)
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
                                    cartesian(&per_dim_indices, |multi| {
                                        let mut lin = 0usize;
                                        for d in 0..dims {
                                            let i0 = multi[d] - 1;
                                            lin += i0 * strides[d];
                                        }
                                        out_data.push(t.data[lin]);
                                    });
                                    if out_data.len() == 1 {
                                        stack.push(Value::Num(out_data[0]));
                                    } else {
                                        let out_tensor =
                                            runmat_builtins::Tensor::new(out_data, out_dims)
                                                .map_err(|e| format!("Slice error: {e}"))?;
                                        stack.push(Value::Tensor(out_tensor));
                                    }
                                }
                            }
                        }
                    }
                    Value::GpuTensor(handle) => {
                        let provider = runmat_accelerate_api::provider()
                            .ok_or_else(|| "No acceleration provider registered".to_string())?;
                        let base_shape = handle.shape.clone();
                        let selectors = build_slice_selectors(
                            dims,
                            colon_mask,
                            end_mask,
                            &numeric,
                            &base_shape,
                        )
                        .map_err(|e| format!("slice: {e}"))?;
                        let plan = build_slice_plan(&selectors, dims, &base_shape)
                            .map_err(|e| map_slice_plan_error("slice", e))?;
                        if plan.indices.is_empty() {
                            let zeros = provider
                                .zeros(&plan.output_shape)
                                .map_err(|e| format!("slice: {e}"))?;
                            stack.push(Value::GpuTensor(zeros));
                        } else {
                            let result = provider
                                .gather_linear(&handle, &plan.indices, &plan.output_shape)
                                .map_err(|e| format!("slice: {e}"))?;
                            stack.push(Value::GpuTensor(result));
                        }
                    }
                    Value::StringArray(sa) => {
                        let rank = sa.shape.len();
                        #[derive(Clone)]
                        enum Sel {
                            Colon,
                            Scalar(usize),
                            Indices(Vec<usize>),
                        }
                        let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        if dims == 1 {
                            let total = sa.data.len();
                            let mut idxs: Vec<usize> = Vec::new();
                            let is_colon = (colon_mask & 1u32) != 0;
                            let is_end = (end_mask & 1u32) != 0;
                            if is_colon {
                                idxs = (1..=total).collect();
                            } else if is_end {
                                idxs = vec![total];
                            } else if let Some(v) = numeric.first() {
                                match v {
                                    Value::Num(n) => {
                                        let i = *n as isize;
                                        if i < 1 {
                                            vm_bail!(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds"
                                            ));
                                        }
                                        idxs = vec![i as usize];
                                    }
                                    Value::Tensor(idx_t) => {
                                        let len = idx_t.shape.iter().product::<usize>();
                                        if len == total {
                                            for (i, &val) in idx_t.data.iter().enumerate() {
                                                if val != 0.0 {
                                                    idxs.push(i + 1);
                                                }
                                            }
                                        } else {
                                            for &val in &idx_t.data {
                                                let i = val as isize;
                                                if i < 1 {
                                                    vm_bail!(mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds"
                                                    ));
                                                }
                                                idxs.push(i as usize);
                                            }
                                        }
                                    }
                                    _ => vm_bail!(mex(
                                        "UnsupportedIndexType",
                                        "Unsupported index type"
                                    )),
                                }
                            } else {
                                vm_bail!(mex("MissingNumericIndex", "missing numeric index"));
                            }
                            if idxs.iter().any(|&i| i == 0 || i > total) {
                                vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            if idxs.len() == 1 {
                                // MATLAB semantics: string array indexing returns a String (double-quoted)
                                stack.push(Value::String(sa.data[idxs[0] - 1].clone()));
                            } else {
                                let mut out: Vec<String> = Vec::with_capacity(idxs.len());
                                for &i in &idxs {
                                    out.push(sa.data[i - 1].clone());
                                }
                                let out_sa =
                                    runmat_builtins::StringArray::new(out, vec![idxs.len(), 1])
                                        .map_err(|e| format!("Slice error: {e}"))?;
                                stack.push(Value::StringArray(out_sa));
                            }
                        } else {
                            for d in 0..dims {
                                let is_colon = (colon_mask & (1u32 << d)) != 0;
                                let is_end = (end_mask & (1u32 << d)) != 0;
                                if is_colon {
                                    selectors.push(Sel::Colon);
                                } else if is_end {
                                    let dim_len = *sa.shape.get(d).unwrap_or(&1);
                                    selectors.push(Sel::Scalar(dim_len));
                                } else {
                                    let v = numeric.get(num_iter).ok_or(mex(
                                        "MissingNumericIndex",
                                        "missing numeric index",
                                    ))?;
                                    num_iter += 1;
                                    match v {
                                        Value::Num(n) => {
                                            let idx = *n as isize;
                                            if idx < 1 {
                                                return Err(mex(
                                                    "IndexOutOfBounds",
                                                    "Index out of bounds",
                                                ));
                                            }
                                            selectors.push(Sel::Scalar(idx as usize));
                                        }
                                        Value::Tensor(idx_t) => {
                                            let dim_len = *sa.shape.get(d).unwrap_or(&1);
                                            let len = idx_t.shape.iter().product::<usize>();
                                            let is_binary_mask = len == dim_len
                                                && idx_t.data.iter().all(|&x| x == 0.0 || x == 1.0);
                                            if is_binary_mask {
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
                            let mut out_dims: Vec<usize> = Vec::new();
                            let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                            for (d, sel) in selectors.iter().enumerate().take(dims) {
                                let dim_len = *sa.shape.get(d).unwrap_or(&1);
                                let idxs = match sel {
                                    Sel::Colon => (1..=dim_len).collect::<Vec<usize>>(),
                                    Sel::Scalar(i) => vec![*i],
                                    Sel::Indices(v) => v.clone(),
                                };
                                if idxs.iter().any(|&i| i == 0 || i > dim_len) {
                                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                                }
                                if idxs.len() > 1 {
                                    out_dims.push(idxs.len());
                                } else {
                                    out_dims.push(1);
                                }
                                per_dim_indices.push(idxs);
                            }
                            if dims == 2 {
                                match (
                                    &per_dim_indices[0].as_slice(),
                                    &per_dim_indices[1].as_slice(),
                                ) {
                                    (i_list, j_list) if i_list.len() > 1 && j_list.len() == 1 => {
                                        out_dims = vec![i_list.len(), 1];
                                    }
                                    (i_list, j_list) if i_list.len() == 1 && j_list.len() > 1 => {
                                        out_dims = vec![1, j_list.len()];
                                    }
                                    _ => {}
                                }
                            }
                            let mut strides: Vec<usize> = vec![0; dims];
                            let full_shape: Vec<usize> = if rank < dims {
                                let mut s = sa.shape.clone();
                                s.resize(dims, 1);
                                s
                            } else {
                                sa.shape.clone()
                            };
                            let mut acc = 1usize;
                            for (d, stride) in strides.iter_mut().enumerate().take(dims) {
                                *stride = acc;
                                acc *= full_shape[d];
                            }
                            let total_out: usize = out_dims.iter().product();
                            if total_out == 0 {
                                stack.push(Value::StringArray(
                                    runmat_builtins::StringArray::new(Vec::new(), out_dims)
                                        .map_err(|e| format!("Slice error: {e}"))?,
                                ));
                            } else {
                                fn cartesian<F: FnMut(&[usize])>(lists: &[Vec<usize>], mut f: F) {
                                    let dims = lists.len();
                                    let mut idx = vec![0usize; dims];
                                    loop {
                                        let current: Vec<usize> =
                                            (0..dims).map(|d| lists[d][idx[d]]).collect();
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
                                let mut out_data: Vec<String> = Vec::with_capacity(total_out);
                                cartesian(&per_dim_indices, |multi| {
                                    let mut lin = 0usize;
                                    for d in 0..dims {
                                        let i0 = multi[d] - 1;
                                        lin += i0 * strides[d];
                                    }
                                    out_data.push(sa.data[lin].clone());
                                });
                                if out_data.len() == 1 {
                                    stack.push(Value::String(out_data[0].clone()));
                                } else {
                                    let out_sa =
                                        runmat_builtins::StringArray::new(out_data, out_dims)
                                            .map_err(|e| format!("Slice error: {e}"))?;
                                    stack.push(Value::StringArray(out_sa));
                                }
                            }
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
                            let v = match runmat_runtime::perform_indexing(&other, &[idx_val]) {
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
            Instr::IndexRangeEnd {
                dims,
                numeric_count,
                colon_mask,
                end_mask,
                range_dims,
                range_has_step,
                end_offsets,
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
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                #[cfg(feature = "native-accel")]
                clear_residency(&base);
                match base {
                    Value::Tensor(t) => {
                        let rank = t.shape.len();
                        #[derive(Clone)]
                        enum Sel {
                            Colon,
                            Scalar(usize),
                            Indices(Vec<usize>),
                            Range { start: i64, step: i64, end_off: i64 },
                        }
                        let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        let mut rp_iter = 0usize;
                        for d in 0..dims {
                            let is_colon = (colon_mask & (1u32 << d)) != 0;
                            let is_end = (end_mask & (1u32 << d)) != 0;
                            if is_colon {
                                selectors.push(Sel::Colon);
                            } else if is_end {
                                selectors.push(Sel::Scalar(*t.shape.get(d).unwrap_or(&1)));
                            } else if let Some(pos) = range_dims.iter().position(|&rd| rd == d) {
                                let (st, sp) = range_params[rp_iter];
                                rp_iter += 1;
                                let off = end_offsets[pos];
                                selectors.push(Sel::Range {
                                    start: st as i64,
                                    step: if sp >= 0.0 {
                                        sp as i64
                                    } else {
                                        -(sp.abs() as i64)
                                    },
                                    end_off: off,
                                });
                            } else {
                                let v = numeric
                                    .get(num_iter)
                                    .ok_or(mex("MissingNumericIndex", "missing numeric index"))?;
                                num_iter += 1;
                                match v {
                                    Value::Num(n) => {
                                        let idx = *n as isize;
                                        if idx < 1 {
                                            vm_bail!(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds"
                                            ));
                                        }
                                        selectors.push(Sel::Scalar(idx as usize));
                                    }
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
                        // Materialize per-dim indices, resolving ranges with end_off
                        let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                        let full_shape: Vec<usize> = if rank < dims {
                            let mut s = t.shape.clone();
                            s.resize(dims, 1);
                            s
                        } else {
                            t.shape.clone()
                        };
                        for (d, sel) in selectors.iter().enumerate().take(dims) {
                            let dim_len = full_shape[d] as i64;
                            let idxs: Vec<usize> = match sel {
                                Sel::Colon => (1..=full_shape[d]).collect(),
                                Sel::Scalar(i) => vec![*i],
                                Sel::Indices(v) => v.clone(),
                                Sel::Range {
                                    start,
                                    step,
                                    end_off,
                                } => {
                                    let mut v = Vec::new();
                                    let mut cur = *start;
                                    let stp = *step;
                                    let end_i = dim_len - *end_off;
                                    if stp == 0 {
                                        vm_bail!(mex("IndexStepZero", "Index step cannot be zero"));
                                    }
                                    if stp > 0 {
                                        while cur <= end_i {
                                            if cur < 1 || cur > dim_len {
                                                break;
                                            }
                                            v.push(cur as usize);
                                            cur += stp;
                                        }
                                    } else {
                                        while cur >= end_i {
                                            if cur < 1 || cur > dim_len {
                                                break;
                                            }
                                            v.push(cur as usize);
                                            cur += stp;
                                        }
                                    }
                                    v
                                }
                            };
                            if idxs.iter().any(|&i| i == 0 || i > full_shape[d]) {
                                vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            per_dim_indices.push(idxs);
                        }
                        // Strides and gather
                        let mut strides: Vec<usize> = vec![0; dims];
                        let mut acc = 1usize;
                        for (d, stride) in strides.iter_mut().enumerate().take(dims) {
                            *stride = acc;
                            acc *= full_shape[d];
                        }
                        let total_out: usize = per_dim_indices.iter().map(|v| v.len()).product();
                        if total_out == 0 {
                            stack.push(Value::Tensor(
                                runmat_builtins::Tensor::new(Vec::new(), vec![0, 0])
                                    .map_err(|e| format!("Slice error: {e}"))?,
                            ));
                            continue;
                        }
                        let mut out_data: Vec<f64> = Vec::with_capacity(total_out);
                        fn cartesian<F: FnMut(&[usize])>(lists: &[Vec<usize>], mut f: F) {
                            let dims = lists.len();
                            let mut idx = vec![0usize; dims];
                            loop {
                                let current: Vec<usize> =
                                    (0..dims).map(|d| lists[d][idx[d]]).collect();
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
                        cartesian(&per_dim_indices, |multi| {
                            let mut lin = 0usize;
                            for d in 0..dims {
                                let i0 = multi[d] - 1;
                                lin += i0 * strides[d];
                            }
                            out_data.push(t.data[lin]);
                        });
                        if out_data.len() == 1 {
                            stack.push(Value::Num(out_data[0]));
                        } else {
                            let shape: Vec<usize> =
                                per_dim_indices.iter().map(|v| v.len().max(1)).collect();
                            let tens = runmat_builtins::Tensor::new(out_data, shape)
                                .map_err(|e| format!("Slice error: {e}"))?;
                            stack.push(Value::Tensor(tens));
                        }
                    }
                    Value::StringArray(sa) => {
                        let selectors =
                            build_slice_selectors(dims, colon_mask, end_mask, &numeric, &sa.shape)
                                .map_err(|e| format!("slice: {e}"))?;
                        let plan = build_slice_plan(&selectors, dims, &sa.shape)
                            .map_err(|e| map_slice_plan_error("slice", e))?;
                        let result =
                            gather_string_slice(&sa, &plan).map_err(|e| format!("slice: {e}"))?;
                        stack.push(result);
                    }
                    _ => vm_bail!(mex("SliceNonTensor", "Slicing only supported on tensors")),
                }
            }

            Instr::IndexSliceEx(dims, numeric_count, colon_mask, end_mask, end_offsets) => {
                // Like IndexSlice, but apply end arithmetic to specified numeric indices
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
                let mut numeric_values = numeric.clone();
                if let Value::GpuTensor(handle) = &base {
                    let adjusted = apply_end_offsets_to_numeric(
                        &numeric_values,
                        dims,
                        colon_mask,
                        end_mask,
                        &end_offsets,
                        &handle.shape,
                    );
                    if let Some(provider) = runmat_accelerate_api::provider() {
                        if let Ok(selectors) = build_slice_selectors(
                            dims,
                            colon_mask,
                            end_mask,
                            &adjusted,
                            &handle.shape,
                        ) {
                            if let Ok(plan) = build_slice_plan(&selectors, dims, &handle.shape) {
                                if plan.indices.is_empty() {
                                    let zeros = provider
                                        .zeros(&plan.output_shape)
                                        .map_err(|e| format!("slice: {e}"))?;
                                    stack.push(Value::GpuTensor(zeros));
                                    pc += 1;
                                    continue;
                                } else {
                                    let result = provider
                                        .gather_linear(handle, &plan.indices, &plan.output_shape)
                                        .map_err(|e| format!("slice: {e}"))?;
                                    stack.push(Value::GpuTensor(result));
                                    pc += 1;
                                    continue;
                                }
                            }
                        }
                        let host = provider
                            .download(handle)
                            .map_err(|e| format!("slice: {e}"))?;
                        let tensor = runmat_builtins::Tensor::new(host.data, host.shape)
                            .map_err(|e| format!("slice: {e}"))?;
                        base = Value::Tensor(tensor);
                        numeric_values = adjusted;
                    } else {
                        return Err("No acceleration provider registered".to_string().into());
                    }
                }
                match base {
                    Value::Tensor(t) => {
                        let adjusted = apply_end_offsets_to_numeric(
                            &numeric_values,
                            dims,
                            colon_mask,
                            end_mask,
                            &end_offsets,
                            &t.shape,
                        );
                        // Build selectors identical to IndexSlice path
                        let mut tmp_stack = Vec::new();
                        tmp_stack.push(Value::Tensor(t));
                        for v in adjusted {
                            tmp_stack.push(v);
                        }
                        // Swap stacks for reuse: assign and then fallthrough to IndexSlice body via small duplication
                        let mut numeric_vals: Vec<Value> = Vec::new();
                        let count = numeric_count;
                        let mut idx_iter = tmp_stack.into_iter();
                        let base = idx_iter
                            .next()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?;
                        for _ in 0..count {
                            match idx_iter.next() {
                                Some(v) => numeric_vals.push(v),
                                None => return Err(mex("StackUnderflow", "stack underflow")),
                            }
                        }
                        match base {
                            Value::Tensor(t2) => {
                                // Inline small subset of IndexSlice gather for t2
                                let rank = t2.shape.len();
                                #[derive(Clone)]
                                enum Sel {
                                    Colon,
                                    Scalar(usize),
                                    Indices(Vec<usize>),
                                }
                                let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                                let mut num_iter = 0usize;
                                if dims == 1 {
                                    let total = t2.data.len();
                                    let mut idxs: Vec<usize> = Vec::new();
                                    let is_colon = (colon_mask & 1u32) != 0;
                                    let is_end = (end_mask & 1u32) != 0;
                                    if is_colon {
                                        idxs = (1..=total).collect();
                                    } else if is_end {
                                        idxs = vec![total];
                                    } else if let Some(v) = numeric_vals.first() {
                                        match v {
                                            Value::Num(n) => {
                                                let i = *n as isize;
                                                if i < 1 {
                                                    vm_bail!(mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds"
                                                    ));
                                                }
                                                idxs = vec![i as usize];
                                            }
                                            Value::Tensor(idx_t) => {
                                                let len = idx_t.shape.iter().product::<usize>();
                                                if len == total {
                                                    for (i, &val) in idx_t.data.iter().enumerate() {
                                                        if val != 0.0 {
                                                            idxs.push(i + 1);
                                                        }
                                                    }
                                                } else {
                                                    for &val in &idx_t.data {
                                                        let i = val as isize;
                                                        if i < 1 {
                                                            vm_bail!(mex(
                                                                "IndexOutOfBounds",
                                                                "Index out of bounds"
                                                            ));
                                                        }
                                                        idxs.push(i as usize);
                                                    }
                                                }
                                            }
                                            _ => vm_bail!(mex(
                                                "UnsupportedIndexType",
                                                "Unsupported index type"
                                            )),
                                        }
                                    } else {
                                        vm_bail!(mex(
                                            "MissingNumericIndex",
                                            "missing numeric index"
                                        ));
                                    }
                                    if idxs.iter().any(|&i| i == 0 || i > total) {
                                        vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                                    }
                                    if idxs.len() == 1 {
                                        stack.push(Value::Num(t2.data[idxs[0] - 1]));
                                    } else {
                                        let mut out = Vec::with_capacity(idxs.len());
                                        for &i in &idxs {
                                            out.push(t2.data[i - 1]);
                                        }
                                        let tens =
                                            runmat_builtins::Tensor::new(out, vec![idxs.len(), 1])
                                                .map_err(|e| format!("Slice error: {e}"))?;
                                        stack.push(Value::Tensor(tens));
                                    }
                                } else {
                                    for d in 0..dims {
                                        let is_colon = (colon_mask & (1u32 << d)) != 0;
                                        let is_end = (end_mask & (1u32 << d)) != 0;
                                        if is_colon {
                                            selectors.push(Sel::Colon);
                                        } else if is_end {
                                            let dim_len = *t2.shape.get(d).unwrap_or(&1);
                                            selectors.push(Sel::Scalar(dim_len));
                                        } else {
                                            let v = numeric_vals.get(num_iter).ok_or(mex(
                                                "MissingNumericIndex",
                                                "missing numeric index",
                                            ))?;
                                            num_iter += 1;
                                            match v {
                                                Value::Num(n) => {
                                                    let idx = *n as isize;
                                                    if idx < 1 {
                                                        return Err(mex(
                                                            "IndexOutOfBounds",
                                                            "Index out of bounds",
                                                        ));
                                                    }
                                                    selectors.push(Sel::Scalar(idx as usize));
                                                }
                                                Value::Tensor(idx_t) => {
                                                    let dim_len = *t2.shape.get(d).unwrap_or(&1);
                                                    let len = idx_t.shape.iter().product::<usize>();
                                                    if len == dim_len {
                                                        let mut indices = Vec::new();
                                                        for (i, &val) in
                                                            idx_t.data.iter().enumerate()
                                                        {
                                                            if val != 0.0 {
                                                                indices.push(i + 1);
                                                            }
                                                        }
                                                        selectors.push(Sel::Indices(indices));
                                                    } else {
                                                        let mut indices = Vec::with_capacity(len);
                                                        for &val in &idx_t.data {
                                                            let idx = val as isize;
                                                            if idx < 1 {
                                                                return Err(mex(
                                                                    "IndexOutOfBounds",
                                                                    "Index out of bounds",
                                                                ));
                                                            }
                                                            indices.push(idx as usize);
                                                        }
                                                        selectors.push(Sel::Indices(indices));
                                                    }
                                                }
                                                Value::LogicalArray(la) => {
                                                    let dim_len = *t2.shape.get(d).unwrap_or(&1);
                                                    if la.data.len() == dim_len {
                                                        let mut indices = Vec::new();
                                                        for (i, &b) in la.data.iter().enumerate() {
                                                            if b != 0 {
                                                                indices.push(i + 1);
                                                            }
                                                        }
                                                        selectors.push(Sel::Indices(indices));
                                                    } else {
                                                        return Err(mex(
                                                            "IndexShape",
                                                            "Logical mask shape mismatch",
                                                        ));
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
                                    let mut out_dims: Vec<usize> = Vec::new();
                                    let mut per_dim_indices: Vec<Vec<usize>> =
                                        Vec::with_capacity(dims);
                                    for (d, sel) in selectors.iter().enumerate().take(dims) {
                                        let dim_len = *t2.shape.get(d).unwrap_or(&1);
                                        let idxs = match sel {
                                            Sel::Colon => (1..=dim_len).collect::<Vec<usize>>(),
                                            Sel::Scalar(i) => vec![*i],
                                            Sel::Indices(v) => v.clone(),
                                        };
                                        if idxs.iter().any(|&i| i == 0 || i > dim_len) {
                                            return Err(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds",
                                            ));
                                        }
                                        if idxs.len() > 1 {
                                            out_dims.push(idxs.len());
                                        } else {
                                            out_dims.push(1);
                                        }
                                        per_dim_indices.push(idxs);
                                    }
                                    if dims == 2 {
                                        match (
                                            &per_dim_indices[0].as_slice(),
                                            &per_dim_indices[1].as_slice(),
                                        ) {
                                            (i_list, j_list)
                                                if i_list.len() > 1 && j_list.len() == 1 =>
                                            {
                                                out_dims = vec![i_list.len(), 1];
                                            }
                                            (i_list, j_list)
                                                if i_list.len() == 1 && j_list.len() > 1 =>
                                            {
                                                out_dims = vec![1, j_list.len()];
                                            }
                                            _ => {}
                                        }
                                    }
                                    let mut strides: Vec<usize> = vec![0; dims];
                                    let full_shape: Vec<usize> = if rank < dims {
                                        let mut s = t2.shape.clone();
                                        s.resize(dims, 1);
                                        s
                                    } else {
                                        t2.shape.clone()
                                    };
                                    let mut acc = 1usize;
                                    for d in 0..dims {
                                        strides[d] = acc;
                                        acc *= full_shape[d];
                                    }
                                    let total_out: usize = out_dims.iter().product();
                                    let mut out_data: Vec<f64> = Vec::with_capacity(total_out);
                                    if out_dims.contains(&0) {
                                        let out_tensor =
                                            runmat_builtins::Tensor::new(out_data, out_dims)
                                                .map_err(|e| format!("Slice error: {e}"))?;
                                        stack.push(Value::Tensor(out_tensor));
                                    } else {
                                        fn cartesian<F: FnMut(&[usize])>(
                                            lists: &[Vec<usize>],
                                            mut f: F,
                                        ) {
                                            let dims = lists.len();
                                            let mut idx = vec![0usize; dims];
                                            loop {
                                                let current: Vec<usize> =
                                                    (0..dims).map(|d| lists[d][idx[d]]).collect();
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
                                        cartesian(&per_dim_indices, |multi| {
                                            let mut lin = 0usize;
                                            for d in 0..dims {
                                                let i0 = multi[d] - 1;
                                                lin += i0 * strides[d];
                                            }
                                            out_data.push(t2.data[lin]);
                                        });
                                        if out_data.len() == 1 {
                                            stack.push(Value::Num(out_data[0]));
                                        } else {
                                            let out_tensor =
                                                runmat_builtins::Tensor::new(out_data, out_dims)
                                                    .map_err(|e| format!("Slice error: {e}"))?;
                                            stack.push(Value::Tensor(out_tensor));
                                        }
                                    }
                                }
                            }
                            other => {
                                stack.push(other);
                            }
                        }
                    }
                    other => {
                        vm_bail!(mex(
                            "SliceNonTensor",
                            &format!("Slicing only supported on tensors: got {other:?}")
                        ));
                    }
                }
            }
            Instr::Index1DRangeEnd { has_step, offset } => {
                // Legacy 1-D path for end arithmetic
                let step_val: f64 = if has_step {
                    let v: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    v
                } else {
                    1.0
                };
                let start_val: f64 = (&stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?)
                    .try_into()?;
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Tensor(t) => {
                        let total = t.data.len();
                        let end_idx = (total as i64) - offset; // inclusive
                        let mut out: Vec<f64> = Vec::new();
                        let mut cur = start_val as i64;
                        let step_i = if step_val >= 0.0 {
                            step_val as i64
                        } else {
                            -(step_val.abs() as i64)
                        };
                        if step_i == 0 {
                            return Err(mex("IndexStepZero", "Index step cannot be zero"));
                        }
                        if step_i > 0 {
                            while cur as i64 <= end_idx {
                                let idx0 = cur as usize;
                                if idx0 == 0 || idx0 > total {
                                    break;
                                }
                                out.push(t.data[idx0 - 1]);
                                cur += step_i;
                            }
                        } else {
                            while (cur as i64) >= end_idx {
                                let idx0 = cur as usize;
                                if idx0 == 0 || idx0 > total {
                                    break;
                                }
                                out.push(t.data[idx0 - 1]);
                                cur += step_i;
                            }
                        }
                        if out.len() == 1 {
                            stack.push(Value::Num(out[0]));
                        } else {
                            let tens =
                                runmat_builtins::Tensor::new(out.clone(), vec![out.len(), 1])
                                    .map_err(|e| format!("Range slice error: {e}"))?;
                            stack.push(Value::Tensor(tens));
                        }
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
                        match runmat_runtime::call_builtin(
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
                                match runmat_runtime::call_builtin(
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
                    Value::Tensor(mut t) => {
                        // F4: write barrier hook (placeholder) – in a full GC integration, call into GC pre/post here
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
                                match v {
                                    Value::Num(n) => {
                                        let i = *n as isize;
                                        if i < 1 || (i as usize) > total {
                                            vm_bail!(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds"
                                            ));
                                        }
                                        lin_indices.push(i as usize);
                                    }
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
                                    match v {
                                        Value::Num(n) => {
                                            let idx = *n as isize;
                                            if idx < 1 {
                                                vm_bail!(mex(
                                                    "IndexOutOfBounds",
                                                    "Index out of bounds"
                                                ));
                                            }
                                            selectors.push(Sel::Scalar(idx as usize));
                                        }
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
                            ) {
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
                                        materialize_rhs_linear(&rhs, count)
                                    } else {
                                        materialize_rhs_nd(&rhs, &plan.selection_lengths)
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
                        let h = handle;
                        // Attempt provider fast-paths for contiguous 2D row/col writes with GPU RHS
                        if dims == 2 {
                            let rows = h.shape.first().copied().unwrap_or(1);
                            let cols = h.shape.get(1).copied().unwrap_or(1);
                            // Build minimal selectors using handle shape for 'end'
                            #[derive(Clone)]
                            enum Sel {
                                Colon,
                                Scalar(usize),
                            }
                            #[allow(unused_assignments)]
                            let mut num_iter_fast = 0usize;
                            let sel0;
                            let sel1;
                            // d=0
                            let is_colon0 = (colon_mask & (1u32 << 0)) != 0;
                            let is_end0 = (end_mask & (1u32 << 0)) != 0;
                            if is_colon0 {
                                sel0 = Sel::Colon;
                            } else if is_end0 {
                                sel0 = Sel::Scalar(rows);
                            } else {
                                let v = numeric
                                    .get(num_iter_fast)
                                    .ok_or(mex("MissingNumericIndex", "missing numeric index"))?;
                                num_iter_fast += 1;
                                let n: f64 = v.try_into()?;
                                if n < 1.0 {
                                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                                }
                                sel0 = Sel::Scalar(n as usize);
                            }
                            // d=1
                            let is_colon1 = (colon_mask & (1u32 << 1)) != 0;
                            let is_end1 = (end_mask & (1u32 << 1)) != 0;
                            if is_colon1 {
                                sel1 = Sel::Colon;
                            } else if is_end1 {
                                sel1 = Sel::Scalar(cols);
                            } else {
                                let v = numeric
                                    .get(num_iter_fast)
                                    .ok_or(mex("MissingNumericIndex", "missing numeric index"))?;
                                let n: f64 = v.try_into()?;
                                if n < 1.0 {
                                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                                }
                                sel1 = Sel::Scalar(n as usize);
                            }
                            // silence unused-assignment lint in builds with two scalar indices
                            let _ = num_iter_fast;
                            // Column write A(:, j) = rhs (gpu)
                            if let (Sel::Colon, Sel::Scalar(j)) = (&sel0, &sel1) {
                                let j0 = *j - 1;
                                if j0 < cols {
                                    if let Value::GpuTensor(vh) = &rhs {
                                        let v_rows = match vh.shape.len() {
                                            1 | 2 => vh.shape[0],
                                            _ => 0,
                                        };
                                        if v_rows == rows {
                                            if let Some(p) = runmat_accelerate_api::provider() {
                                                match p.scatter_column(&h, j0, vh) {
                                                    Ok(new_h) => {
                                                        stack.push(Value::GpuTensor(new_h));
                                                        bench_end("StoreSlice2D.fast_col", __b);
                                                        pc += 1;
                                                        continue;
                                                    }
                                                    Err(_) => { /* fall through to gather path */ }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // Row write A(i, :) = rhs (gpu)
                            if let (Sel::Scalar(i), Sel::Colon) = (&sel0, &sel1) {
                                let i0 = *i - 1;
                                if i0 < rows {
                                    if let Value::GpuTensor(vh) = &rhs {
                                        let v_cols = match vh.shape.len() {
                                            1 => vh.shape[0],
                                            2 => vh.shape[1],
                                            _ => 0,
                                        };
                                        if v_cols == cols {
                                            if let Some(p) = runmat_accelerate_api::provider() {
                                                match p.scatter_row(&h, i0, vh) {
                                                    Ok(new_h) => {
                                                        stack.push(Value::GpuTensor(new_h));
                                                        bench_end("StoreSlice2D.fast_row", __b);
                                                        pc += 1;
                                                        continue;
                                                    }
                                                    Err(_) => { /* fall through */ }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // Gather–mutate–reupload fallback for slice assignment on GPU bases
                        let provider = runmat_accelerate_api::provider()
                            .ok_or_else(|| "No acceleration provider registered".to_string())?;
                        let host = provider
                            .download(&h)
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
                                match v {
                                    Value::Num(n) => {
                                        let i = *n as isize;
                                        if i < 1 || (i as usize) > total {
                                            vm_bail!(mex(
                                                "IndexOutOfBounds",
                                                "Index out of bounds"
                                            ));
                                        }
                                        lin_indices.push(i as usize);
                                    }
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
                                    match v {
                                        Value::Num(n) => {
                                            let idx = *n as isize;
                                            if idx < 1 {
                                                vm_bail!(mex(
                                                    "IndexOutOfBounds",
                                                    "Index out of bounds"
                                                ));
                                            }
                                            selectors.push(Sel::Scalar(idx as usize));
                                        }
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
                    Value::StringArray(mut sa) => {
                        let selectors =
                            build_slice_selectors(dims, colon_mask, end_mask, &numeric, &sa.shape)
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
                    _ => vm_bail!(
                        "Slicing assignment only supported on tensors or string arrays".to_string()
                    ),
                }
                bench_end("StoreSlice", __b);
            }
            Instr::StoreSliceEx(dims, numeric_count, colon_mask, end_mask, end_offsets) => {
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
                let mut base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                if let Value::GpuTensor(handle) = &base {
                    let adjusted = apply_end_offsets_to_numeric(
                        &numeric,
                        dims,
                        colon_mask,
                        end_mask,
                        &end_offsets,
                        &handle.shape,
                    );
                    if let Some(provider) = runmat_accelerate_api::provider() {
                        if let Ok(selectors) = build_slice_selectors(
                            dims,
                            colon_mask,
                            end_mask,
                            &adjusted,
                            &handle.shape,
                        ) {
                            if let Ok(plan) = build_slice_plan(&selectors, dims, &handle.shape) {
                                let values = if plan.dims == 1 {
                                    let count =
                                        plan.selection_lengths.first().copied().unwrap_or(0);
                                    materialize_rhs_linear(&rhs, count)
                                } else {
                                    materialize_rhs_nd(&rhs, &plan.selection_lengths)
                                }
                                .map_err(|e| format!("slice assign: {e}"))?;
                                if values.len() == plan.indices.len() {
                                    let value_shape = vec![values.len().max(1), 1];
                                    let upload_result = if values.is_empty() {
                                        provider.zeros(&[0, 1])
                                    } else {
                                        provider.upload(&runmat_accelerate_api::HostTensorView {
                                            data: &values,
                                            shape: &value_shape,
                                        })
                                    };
                                    if let Ok(values_handle) = upload_result {
                                        if provider
                                            .scatter_linear(handle, &plan.indices, &values_handle)
                                            .is_ok()
                                        {
                                            stack.push(Value::GpuTensor(handle.clone()));
                                            pc += 1;
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                        let host = provider
                            .download(handle)
                            .map_err(|e| format!("slice assign: {e}"))?;
                        let tensor = runmat_builtins::Tensor::new(host.data, host.shape)
                            .map_err(|e| format!("slice assign: {e}"))?;
                        base = Value::Tensor(tensor);
                    } else {
                        return Err("No acceleration provider registered".to_string().into());
                    }
                }
                match base {
                    Value::Tensor(t) => {
                        // Adjust numeric indices for end offsets, mapping numeric position to actual dimension
                        let mut adjusted = numeric.clone();
                        for (pos, off) in end_offsets {
                            if let Some(v) = adjusted.get_mut(pos) {
                                // Map numeric index position to dimension index by skipping colon and plain end dims
                                let mut seen_numeric = 0usize;
                                let mut dim_for_pos = 0usize;
                                for d in 0..dims {
                                    let is_colon = (colon_mask & (1u32 << d)) != 0;
                                    let is_end = (end_mask & (1u32 << d)) != 0;
                                    if is_colon || is_end {
                                        continue;
                                    }
                                    if seen_numeric == pos {
                                        dim_for_pos = d;
                                        break;
                                    }
                                    seen_numeric += 1;
                                }
                                let dim_len = *t.shape.get(dim_for_pos).unwrap_or(&1);
                                let idx_val = (dim_len as isize) - (off as isize);
                                *v = Value::Num(idx_val as f64);
                            }
                        }
                        // Reuse StoreSlice by pushing base back along with adjusted numerics and rhs
                        stack.push(Value::Tensor(t));
                        for v in adjusted {
                            stack.push(v);
                        }
                        stack.push(rhs);
                        // Fallthrough emulation: replicate logic of StoreSlice with broadcasting
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
                            Value::Tensor(mut t) => {
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
                                        match v {
                                            Value::Num(n) => {
                                                let idx = *n as isize;
                                                if idx < 1 {
                                                    vm_bail!(mex(
                                                        "IndexOutOfBounds",
                                                        "Index out of bounds"
                                                    ));
                                                }
                                                selectors.push(Sel::Scalar(idx as usize));
                                            }
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
                                            _ => vm_bail!(mex(
                                                "UnsupportedIndexType",
                                                "Unsupported index type"
                                            )),
                                        }
                                    }
                                }
                                // Compute per-dim indices and strides
                                let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                                for (d, sel) in selectors.iter().enumerate().take(dims) {
                                    let dim_len = *t.shape.get(d).unwrap_or(&1);
                                    let idxs = match sel {
                                        Sel::Colon => (1..=dim_len).collect::<Vec<usize>>(),
                                        Sel::Scalar(i) => vec![*i],
                                        Sel::Indices(v) => v.clone(),
                                    };
                                    per_dim_indices.push(idxs);
                                }
                                let mut strides: Vec<usize> = vec![0; dims];
                                let mut acc = 1usize;
                                for (d, stride) in strides.iter_mut().enumerate().take(dims) {
                                    *stride = acc;
                                    acc *= *t.shape.get(d).unwrap_or(&1);
                                }
                                // Build RHS view with broadcasting like StoreSlice
                                enum RhsView {
                                    Scalar(f64),
                                    Tensor {
                                        data: Vec<f64>,
                                        shape: Vec<usize>,
                                        strides: Vec<usize>,
                                    },
                                }
                                let rhs_view =
                                    match rhs {
                                        Value::Num(n) => RhsView::Scalar(n),
                                        Value::Tensor(rt) => {
                                            let mut rshape = rt.shape.clone();
                                            if rshape.len() < dims {
                                                rshape.resize(dims, 1);
                                            }
                                            if rshape.len() > dims {
                                                if rshape.iter().skip(dims).any(|&s| s != 1) {
                                                    vm_bail!("shape mismatch for slice assign"
                                                        .to_string());
                                                }
                                                rshape.truncate(dims);
                                            }
                                            for d in 0..dims {
                                                let out_len = per_dim_indices[d].len();
                                                let rhs_len = rshape[d];
                                                if !(rhs_len == 1 || rhs_len == out_len) {
                                                    vm_bail!("shape mismatch for slice assign"
                                                        .to_string());
                                                }
                                            }
                                            let mut rstrides = vec![0usize; dims];
                                            let mut racc = 1usize;
                                            for d in 0..dims {
                                                rstrides[d] = racc;
                                                racc *= rshape[d];
                                            }
                                            RhsView::Tensor {
                                                data: rt.data,
                                                shape: rshape,
                                                strides: rstrides,
                                            }
                                        }
                                        _ => vm_bail!("rhs must be numeric or tensor".to_string()),
                                    };
                                // Map absolute indices to selection positions per dimension
                                use std::collections::HashMap;
                                let mut pos_maps: Vec<HashMap<usize, usize>> =
                                    Vec::with_capacity(dims);
                                for (_d, dim_idxs) in per_dim_indices.iter().enumerate().take(dims)
                                {
                                    let mut m = HashMap::new();
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
                                cartesian2(&per_dim_indices, |multi| {
                                    let mut lin = 0usize;
                                    for d in 0..dims {
                                        let i0 = multi[d] - 1;
                                        lin += i0 * strides[d];
                                    }
                                    match &rhs_view {
                                        RhsView::Scalar(v) => {
                                            t.data[lin] = *v;
                                        }
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
                                            t.data[lin] = data[rlin];
                                        }
                                    }
                                });
                                stack.push(Value::Tensor(t));
                            }
                            Value::StringArray(mut sa) => {
                                let selectors = build_slice_selectors(
                                    dims, colon_mask, end_mask, &numeric, &sa.shape,
                                )
                                .map_err(|e| format!("slice assign: {e}"))?;
                                let plan = build_slice_plan(&selectors, dims, &sa.shape)
                                    .map_err(|e| map_slice_plan_error("slice assign", e))?;
                                if plan.indices.is_empty() {
                                    stack.push(Value::StringArray(sa));
                                    pc += 1;
                                    continue;
                                }
                                let rhs_view = build_string_rhs_view(&rhs, &plan.selection_lengths)
                                    .map_err(|e| format!("slice assign: {e}"))?;
                                scatter_string_with_plan(&mut sa, &plan, &rhs_view)
                                    .map_err(|e| format!("slice assign: {e}"))?;
                                stack.push(Value::StringArray(sa));
                                pc += 1;
                                continue;
                            }
                            other => vm_bail!(format!("StoreSliceEx unsupported base: {other:?}")),
                        }
                    }
                    other => vm_bail!(format!(
                        "StoreSliceEx only supports tensors currently, got {other:?}"
                    )),
                }
            }
            Instr::StoreRangeEnd {
                dims,
                numeric_count,
                colon_mask,
                end_mask,
                range_dims,
                range_has_step,
                end_offsets,
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
                    Value::Object(_) | Value::Tensor(_) | Value::GpuTensor(_)
                );
                if !base_assignable
                    && matches!(
                        rhs,
                        Value::Object(_) | Value::Tensor(_) | Value::GpuTensor(_)
                    )
                {
                    std::mem::swap(&mut base, &mut rhs);
                }
                match base {
                    Value::Tensor(mut t) => {
                        #[derive(Clone)]
                        enum Sel {
                            Colon,
                            Scalar(usize),
                            Indices(Vec<usize>),
                            Range { start: i64, step: i64, end_off: i64 },
                        }
                        let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        let mut rp_iter = 0usize;
                        for d in 0..dims {
                            if let Some(pos) = range_dims.iter().position(|&rd| rd == d) {
                                let (st, sp) = range_params[rp_iter];
                                rp_iter += 1;
                                let step_i = if sp >= 0.0 {
                                    sp as i64
                                } else {
                                    -(sp.abs() as i64)
                                };
                                selectors.push(Sel::Range {
                                    start: st as i64,
                                    step: step_i,
                                    end_off: end_offsets[pos],
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
                            match v {
                                Value::Num(n) => {
                                    let idx = *n as isize;
                                    if idx < 1 {
                                        vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                                    }
                                    selectors.push(Sel::Scalar(idx as usize));
                                }
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
                                    vm_bail!(mex("UnsupportedIndexType", "Unsupported index type"))
                                }
                            }
                        }
                        // Build index lists and scatter rhs with broadcasting
                        // debug removed
                        let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                        for (d, sel) in selectors.iter().enumerate().take(dims) {
                            let dim_len = *t.shape.get(d).unwrap_or(&1);
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
                                    let end_i = (dim_len as i64) - *end_off;
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
                                    let mut rshape = rt.shape.clone();
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
                        let provider = runmat_accelerate_api::provider()
                            .ok_or_else(|| "No acceleration provider registered".to_string())?;
                        let host = provider
                            .download(&h)
                            .map_err(|e| format!("gather for range-end assign: {e}"))?;
                        let mut t = runmat_builtins::Tensor::new(host.data, host.shape)
                            .map_err(|e| format!("range-end assign: {e}"))?;
                        #[derive(Clone)]
                        enum Sel {
                            Colon,
                            Scalar(usize),
                            Indices(Vec<usize>),
                            Range { start: i64, step: i64, end_off: i64 },
                        }
                        let mut selectors: Vec<Sel> = Vec::with_capacity(dims);
                        let mut num_iter = 0usize;
                        let mut rp_iter = 0usize;
                        for d in 0..dims {
                            if let Some(pos) = range_dims.iter().position(|&rd| rd == d) {
                                let (st, sp) = range_params[rp_iter];
                                rp_iter += 1;
                                let step_i = if sp >= 0.0 {
                                    sp as i64
                                } else {
                                    -(sp.abs() as i64)
                                };
                                selectors.push(Sel::Range {
                                    start: st as i64,
                                    step: step_i,
                                    end_off: end_offsets[pos],
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
                            match v {
                                Value::Num(n) => {
                                    let idx = *n as isize;
                                    if idx < 1 {
                                        vm_bail!(mex("IndexOutOfBounds", "Index out of bounds"));
                                    }
                                    selectors.push(Sel::Scalar(idx as usize));
                                }
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
                                    vm_bail!(mex("UnsupportedIndexType", "Unsupported index type"))
                                }
                            }
                        }
                        // Build index lists and scatter rhs with broadcasting
                        // debug removed
                        let mut per_dim_indices: Vec<Vec<usize>> = Vec::with_capacity(dims);
                        for (d, sel) in selectors.iter().enumerate().take(dims) {
                            let dim_len = *t.shape.get(d).unwrap_or(&1);
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
                                    let end_i = (dim_len as i64) - *end_off;
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
                                    let mut rshape = rt.shape.clone();
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
                                let (st, sp) = range_params[rp_iter];
                                rp_iter += 1;
                                let off = end_offsets[pos];
                                let cell = runmat_builtins::CellArray::new(
                                    vec![
                                        Value::Num(st),
                                        Value::Num(sp),
                                        Value::String("end".to_string()),
                                        Value::Num(off as f64),
                                    ],
                                    1,
                                    4,
                                )
                                .map_err(|e| format!("obj range: {e}"))?;
                                idx_values.push(Value::Cell(cell));
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
                                        ).into())
                                    }
                                }
                            }
                        }
                        let cell = runmat_builtins::CellArray::new(idx_values, 1, dims)
                            .map_err(|e| format!("subsasgn build error: {e}"))?;
                        match runmat_runtime::call_builtin(
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
                    _ => vm_bail!("StoreRangeEnd only supports tensors currently".to_string()),
                }
            }
            Instr::StoreSlice1DRangeEnd { has_step, offset } => {
                // RHS, then start[, step], then base
                let rhs = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let step_val: f64 = if has_step {
                    let v: f64 = (&stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?)
                        .try_into()?;
                    v
                } else {
                    1.0
                };
                let start_val: f64 = (&stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?)
                    .try_into()?;
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                #[cfg(feature = "native-accel")]
                clear_residency(&base);
                match base {
                    Value::Tensor(mut t) => {
                        let total = t.data.len();
                        let end_idx = (total as i64) - offset;
                        let mut cur = start_val as i64;
                        let step_i = if step_val >= 0.0 {
                            step_val as i64
                        } else {
                            -(step_val.abs() as i64)
                        };
                        if step_i == 0 {
                            return Err(mex("IndexStepZero", "Index step cannot be zero"));
                        }
                        // Broadcast rhs if scalar
                        let rhs_vals: Vec<f64> = match rhs {
                            Value::Num(n) => vec![n],
                            Value::Tensor(rt) => rt.data.clone(),
                            _ => vec![0.0],
                        };
                        let mut rpos = 0usize;
                        if step_i > 0 {
                            while cur as i64 <= end_idx {
                                let idx0 = cur as usize;
                                if idx0 == 0 || idx0 > total {
                                    break;
                                }
                                let v = rhs_vals
                                    .get(rpos)
                                    .cloned()
                                    .unwrap_or(*rhs_vals.last().unwrap_or(&0.0));
                                t.data[idx0 - 1] = v;
                                rpos += 1;
                                cur += step_i;
                            }
                        } else {
                            while (cur as i64) >= end_idx {
                                let idx0 = cur as usize;
                                if idx0 == 0 || idx0 > total {
                                    break;
                                }
                                let v = rhs_vals
                                    .get(rpos)
                                    .cloned()
                                    .unwrap_or(*rhs_vals.last().unwrap_or(&0.0));
                                t.data[idx0 - 1] = v;
                                rpos += 1;
                                cur += step_i;
                            }
                        }
                        stack.push(Value::Tensor(t));
                    }
                    _ => vm_bail!("Store range with end only supported on tensors".to_string()),
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
                let ca = runmat_builtins::CellArray::new(elems, rows, cols)
                    .map_err(|e| format!("Cell creation error: {e}"))?;
                stack.push(Value::Cell(ca));
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
                        let cell = runmat_runtime::call_builtin(
                            "__make_cell",
                            &indices
                                .iter()
                                .map(|n| Value::Num(*n as f64))
                                .collect::<Vec<_>>(),
                        )?;
                        match runmat_runtime::call_builtin(
                            "call_method",
                            &[
                                Value::Object(obj),
                                Value::String("subsref".to_string()),
                                Value::String("{}".to_string()),
                                cell,
                            ],
                        ) {
                            Ok(v) => stack.push(v),
                            Err(e) => vm_bail!(e.to_string()),
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
                        _ => return Err("Unsupported number of cell indices".to_string().into()),
                    },
                    _ => return Err("Cell indexing on non-cell".to_string().into()),
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
                                _ => return Err("Unsupported number of cell indices".to_string().into()),
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
                        let cell = runmat_runtime::call_builtin(
                            "__make_cell",
                            &indices
                                .iter()
                                .map(|n| Value::Num(*n as f64))
                                .collect::<Vec<_>>(),
                        )?;
                        let v = match runmat_runtime::call_builtin(
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
                    _ => return Err("Cell expansion on non-cell".to_string().into()),
                }
            }
            Instr::Pop => {
                stack.pop();
            }
            Instr::ReturnValue => {
                let return_value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                stack.push(return_value);
                interpreter_timing.flush_host_span("return_value", None);
                break;
            }
            Instr::Return => {
                interpreter_timing.flush_host_span("return", None);
                break;
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
                    matches!(v, Value::Object(_) | Value::Tensor(_) | Value::GpuTensor(_))
                };
                let base_idx_opt = (0..stack.len()).rev().find(|&j| assignable(&stack[j]));
                let base_pos = if let Some(j) = base_idx_opt {
                    j
                } else {
                    return Err("Index assignment only for tensors".to_string().into());
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
                            match (&stack[idx_pos]).try_into() as Result<f64, _> {
                                Ok(v) => indices.push(v as usize),
                                Err(_) => {
                                    contiguous_ok = false;
                                    indices.clear();
                                    break;
                                }
                            }
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
                        if let Ok(v) = (&stack[idx]).try_into() as Result<f64, _> {
                            numeric_above.push((idx, v as usize));
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
                    return Err("Index assignment only for tensors".to_string().into());
                }
                // TODO(GC): write barrier hook if base is in older generation and rhs/indices reference younger objects
                match base {
                    Value::Object(obj) => {
                        // subsasgn(obj, '()', {indices...}, rhs)
                        let cell = runmat_runtime::call_builtin(
                            "__make_cell",
                            &indices
                                .iter()
                                .map(|n| Value::Num(*n as f64))
                                .collect::<Vec<_>>(),
                        )?;
                        match runmat_runtime::call_builtin(
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
                    Value::Tensor(mut t) => {
                        // Helper to coerce RHS to scalar f64, supporting 1x1 tensors and gpu tensors
                        let rhs_to_scalar = |rhs: &Value| -> VmResult<f64> {
                            match rhs {
                                Value::Num(x) => Ok(*x),
                                Value::Tensor(t2) => {
                                    if t2.data.len() == 1 {
                                        Ok(t2.data[0])
                                    } else {
                                        Err("RHS must be scalar".to_string().into())
                                    }
                                }
                                Value::GpuTensor(h2) => {
                                    let total = h2.shape.iter().copied().product::<usize>();
                                    if total != 1 {
                                        return Err("RHS must be scalar".to_string().into());
                                    }
                                    if let Some(p) = runmat_accelerate_api::provider() {
                                        let host = p
                                            .download(h2)
                                            .map_err(|e| format!("gather rhs: {e}"))?;
                                        Ok(host.data[0])
                                    } else {
                                        Err("No acceleration provider registered".to_string().into())
                                    }
                                }
                                _ => rhs
                                    .try_into()
                                    .map_err(|_| "RHS must be numeric".to_string().into()),
                            }
                        };
                        // 1D linear or 2D scalar assignment only for now
                        if indices.len() == 1 {
                            let total = t.rows() * t.cols();
                            let idx = indices[0];
                            if idx == 0 || idx > total {
                                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            let val: f64 = rhs_to_scalar(&rhs)?;
                            t.data[idx - 1] = val;
                            stack.push(Value::Tensor(t));
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
                                        "[vm] StoreIndex Tensor OOB"
                                    );
                                }
                                return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
                            }
                            let val: f64 = rhs_to_scalar(&rhs)?;
                            let idx = (i - 1) + (j - 1) * rows;
                            t.data[idx] = val;
                            stack.push(Value::Tensor(t));
                        } else {
                            return Err("Only 1D/2D scalar assignment supported".to_string().into());
                        }
                    }
                    Value::GpuTensor(h) => {
                        // Stage F1: gather–mutate–reupload for simple 1D/2D scalar assignments
                        let provider = runmat_accelerate_api::provider()
                            .ok_or_else(|| "No acceleration provider registered".to_string())?;
                        let host = provider
                            .download(&h)
                            .map_err(|e| format!("gather for assignment: {e}"))?;
                        let mut t = runmat_builtins::Tensor::new(host.data, host.shape)
                            .map_err(|e| format!("assignment: {e}"))?;
                        // Reuse same scalar coercion
                        let rhs_to_scalar = |rhs: &Value| -> VmResult<f64> {
                            match rhs {
                                Value::Num(x) => Ok(*x),
                                Value::Tensor(t2) => {
                                    if t2.data.len() == 1 {
                                        Ok(t2.data[0])
                                    } else {
                                        Err("RHS must be scalar".to_string().into())
                                    }
                                }
                                Value::GpuTensor(h2) => {
                                    let total = h2.shape.iter().copied().product::<usize>();
                                    if total != 1 {
                                        return Err("RHS must be scalar".to_string().into());
                                    }
                                    let host2 = provider
                                        .download(h2)
                                        .map_err(|e| format!("gather rhs: {e}"))?;
                                    Ok(host2.data[0])
                                }
                                _ => rhs
                                    .try_into()
                                    .map_err(|_| "RHS must be numeric".to_string().into()),
                            }
                        };
                        if indices.len() == 1 {
                            let total = t.rows() * t.cols();
                            let idx = indices[0];
                            if idx == 0 || idx > total {
                                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                            }
                            let val: f64 = rhs_to_scalar(&rhs)?;
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
                            let val: f64 = rhs_to_scalar(&rhs)?;
                            let idx = (i - 1) + (j - 1) * rows;
                            t.data[idx] = val;
                        } else if indices.is_empty() {
                            // Trivial colon slice cases from parser may encode as zero indices; handle full-row/col scalar broadcast
                            let val: f64 = rhs_to_scalar(&rhs)?;
                            for k in 0..t.data.len() {
                                t.data[k] = val;
                            }
                        } else {
                            return Err("Only 1D/2D scalar assignment supported".to_string().into());
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
                        return Err("Index assignment only for tensors".to_string().into());
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
                        match runmat_runtime::call_builtin(
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
                        _ => return Err("Unsupported number of cell indices".to_string().into()),
                    },
                    _ => return Err("Cell assignment on non-cell".to_string().into()),
                }
            }
            Instr::LoadMember(field) => {
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
                                match runmat_runtime::call_builtin(
                                    &getter,
                                    &[Value::Object(obj.clone())],
                                ) {
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
                                match runmat_runtime::call_builtin(
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
                    Value::Struct(st) => {
                        if let Some(v) = st.fields.get(&field) {
                            stack.push(v.clone());
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
                    _ => vm_bail!("LoadMember on non-object".to_string()),
                }
            }
            Instr::LoadMemberDynamic => {
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
                                match runmat_runtime::call_builtin(
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
                    Value::Struct(st) => {
                        if let Some(v) = st.fields.get(&name) {
                            stack.push(v.clone());
                        } else {
                            vm_bail!(format!("Undefined field '{}'", name));
                        }
                    }
                    _ => vm_bail!("LoadMemberDynamic on non-struct/object".to_string()),
                }
            }
            Instr::StoreMember(field) => {
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
                                match runmat_runtime::call_builtin(
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
                                match runmat_runtime::call_builtin(
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
                    _ => vm_bail!("StoreMember on non-object".to_string()),
                }
            }
            Instr::StoreMemberDynamic => {
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
                    _ => vm_bail!("StoreMemberDynamic on non-struct/object".to_string()),
                }
            }
            Instr::CallMethod(name, arg_count) => {
                // base, then args are on stack in order: [..., base, a1, a2, ...]
                let mut args = Vec::with_capacity(arg_count);
                for _ in 0..arg_count {
                    args.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                args.reverse();
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Object(obj) => {
                        // Compose qualified and try runtime builtin dispatch, passing receiver first
                        if let Some((m, _owner)) =
                            runmat_builtins::lookup_method(&obj.class_name, &name)
                        {
                            if m.is_static {
                                vm_bail!(format!(
                                    "Method '{}' is static; use classref({}).{}",
                                    name, obj.class_name, name
                                ));
                            }
                            if m.access == runmat_builtins::Access::Private {
                                vm_bail!(format!("Method '{}' is private", name))
                            }
                            let mut full_args = Vec::with_capacity(1 + args.len());
                            full_args.push(Value::Object(obj));
                            full_args.extend(args.into_iter());
                            let v = runmat_runtime::call_builtin(&m.function_name, &full_args)?;
                            stack.push(v);
                            continue;
                        }
                        let qualified = format!("{}.{}", obj.class_name, name);
                        let mut full_args = Vec::with_capacity(1 + args.len());
                        full_args.push(Value::Object(obj));
                        full_args.extend(args.into_iter());
                        if let Ok(v) = runmat_runtime::call_builtin(&qualified, &full_args) {
                            stack.push(v);
                        } else {
                            match runmat_runtime::call_builtin(&name, &full_args) {
                                Ok(v) => {
                                    stack.push(v);
                                }
                                Err(e) => {
                                    vm_bail!(e);
                                }
                            }
                        }
                    }
                    _ => vm_bail!("CallMethod on non-object".to_string()),
                }
            }
            Instr::LoadMethod(name) => {
                // Base object on stack; return a closure that calls the method with receiver as first captured arg
                let base = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                match base {
                    Value::Object(obj) => {
                        let func_qual = format!("{}.{}", obj.class_name, name);
                        stack.push(Value::Closure(runmat_builtins::Closure {
                            function_name: func_qual,
                            captures: vec![Value::Object(obj)],
                        }));
                    }
                    Value::ClassRef(cls) => {
                        // Bound static method handle (no receiver capture), resolve via inheritance
                        if let Some((m, _owner)) = runmat_builtins::lookup_method(&cls, &name) {
                            if !m.is_static {
                                vm_bail!(format!("Method '{}' is not static", name));
                            }
                            stack.push(Value::Closure(runmat_builtins::Closure {
                                function_name: m.function_name,
                                captures: vec![],
                            }));
                        } else {
                            vm_bail!(format!("Unknown static method '{}' on class {}", name, cls));
                        }
                    }
                    _ => vm_bail!("LoadMethod requires object or classref".to_string()),
                }
            }
            Instr::CreateClosure(func_name, capture_count) => {
                let mut captures = Vec::with_capacity(capture_count);
                for _ in 0..capture_count {
                    captures.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                captures.reverse();
                stack.push(Value::Closure(runmat_builtins::Closure {
                    function_name: func_name,
                    captures,
                }));
            }
            Instr::LoadStaticProperty(class_name, prop) => {
                // Enforce access and static-ness via registry (with inheritance)
                if let Some((p, owner)) = runmat_builtins::lookup_property(&class_name, &prop) {
                    if !p.is_static {
                        vm_bail!(format!("Property '{}' is not static", prop));
                    }
                    if p.get_access == runmat_builtins::Access::Private {
                        vm_bail!(format!("Property '{}' is private", prop))
                    }
                    if let Some(v) = runmat_builtins::get_static_property_value(&owner, &prop) {
                        stack.push(v);
                    } else if let Some(v) = &p.default_value {
                        stack.push(v.clone());
                    } else {
                        stack.push(Value::Num(0.0));
                    }
                } else {
                    vm_bail!(format!(
                        "Unknown property '{}' on class {}",
                        prop, class_name
                    ));
                }
            }
            Instr::CallStaticMethod(class_name, method, arg_count) => {
                let mut args = Vec::with_capacity(arg_count);
                for _ in 0..arg_count {
                    args.push(
                        stack
                            .pop()
                            .ok_or(mex("StackUnderflow", "stack underflow"))?,
                    );
                }
                args.reverse();
                if let Some((m, _owner)) = runmat_builtins::lookup_method(&class_name, &method) {
                    if !m.is_static {
                        vm_bail!(format!("Method '{}' is not static", method));
                    }
                    if m.access == runmat_builtins::Access::Private {
                        vm_bail!(format!("Method '{}' is private", method))
                    }
                    let v = match runmat_runtime::call_builtin(&m.function_name, &args) {
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

fn stochastic_evolution_dispatch(
    state: Value,
    drift: Value,
    scale: Value,
    steps: Value,
) -> VmResult<Value> {
    let steps_u32 = parse_steps_value(&steps)?;
    if steps_u32 == 0 {
        return Ok(state);
    }

    #[cfg(feature = "native-accel")]
    {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let (state_handle, state_owned) = ensure_gpu_tensor_for_stochastic(provider, &state)?;
            let drift_scalar = scalar_from_value_scalar(&drift, "stochastic_evolution drift")?;
            let scale_scalar = scalar_from_value_scalar(&scale, "stochastic_evolution scale")?;
            let output = provider
                .stochastic_evolution(&state_handle, drift_scalar, scale_scalar, steps_u32)
                .map_err(|e| format!("stochastic_evolution: {e}"))?;
            if let Some(temp) = state_owned {
                let _ = provider.free(&temp);
            }
            fusion_residency::mark(&output);
            return Ok(Value::GpuTensor(output));
        }
    }

    let gathered_state =
        gather_if_needed(&state).map_err(|e| format!("stochastic_evolution: {e}"))?;
    let mut tensor_value = match gathered_state {
        Value::Tensor(t) => t,
        other => tensor::value_into_tensor_for("stochastic_evolution", other)?,
    };
    let drift_scalar = scalar_from_value_scalar(&drift, "stochastic_evolution drift")?;
    let scale_scalar = scalar_from_value_scalar(&scale, "stochastic_evolution scale")?;
    stochastic_evolution_host(&mut tensor_value, drift_scalar, scale_scalar, steps_u32).map_err(
        |flow| match flow {
            runmat_runtime::RuntimeControlFlow::Error(err) => err.message().to_string(),
            runmat_runtime::RuntimeControlFlow::Suspend(_) => {
                "stochastic_evolution: unexpected suspension".to_string()
            }
        },
    )?;
    Ok(Value::Tensor(tensor_value))
}

fn scalar_from_value_scalar(value: &Value, label: &str) -> VmResult<f64> {
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
            let gathered = gather_if_needed(value).map_err(|e| format!("{label}: {e}"))?;
            scalar_from_value_scalar(&gathered, label)
        }
        other => Err(format!("{label}: expected numeric scalar, got {:?}", other).into()),
    }
}

fn parse_steps_value(value: &Value) -> VmResult<u32> {
    let raw = scalar_from_value_scalar(value, "stochastic_evolution steps")?;
    if !raw.is_finite() || raw < 0.0 {
        return Err("stochastic_evolution: steps must be a non-negative scalar".to_string().into());
    }
    Ok(raw.round() as u32)
}

#[cfg(feature = "native-accel")]
fn ensure_gpu_tensor_for_stochastic(
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
            let gathered =
                gather_if_needed(value).map_err(|e| format!("stochastic_evolution: {e}"))?;
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
    provider.upload(&view).map_err(|e| e.to_string().into())
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
    }
}
#[cfg(feature = "native-accel")]
#[inline]
fn summarize_value(i: usize, v: &Value) -> String {
    match v {
        Value::GpuTensor(h) => format!("in#{i}:GpuTensor shape={:?}", h.shape),
        Value::Tensor(t) => format!("in#{i}:Tensor shape={:?}", t.shape),
        Value::String(s) => format!("in#{i}:String({})", s),
        _ => format!("in#{i}:{}", value_kind(v)),
    }
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
fn try_execute_fusion_group(
    plan: &runmat_accelerate::FusionGroupPlan,
    graph: &runmat_accelerate::AccelGraph,
    stack: &mut Vec<Value>,
    vars: &mut [Value],
    context: &ExecutionContext,
) -> VmResult<Value> {
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
        let stack_needed_preview = plan.stack_pattern.len();
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

    let pattern_len = plan.stack_pattern.len();
    if stack.len() < pattern_len {
        if fusion_debug_enabled() {
            log::debug!(
                "fusion stack underflow: plan={} needed={} available={} pattern={:?}",
                plan.index,
                pattern_len,
                stack.len(),
                plan.stack_pattern
            );
        }
        return Err("fusion: stack underflow gathering inputs".to_string().into());
    }
    let available = pattern_len;
    let slice_start = stack.len() - available;
    let stack_guard = StackSliceGuard::new(stack, slice_start);
    let slice = stack_guard.slice().to_vec();
    let mut consumed: Vec<Option<Value>> = vec![None; pattern_len];
    let skip = 0;

    for (offset, input_idx) in plan.stack_pattern.iter().enumerate() {
        if offset < skip {
            continue;
        }
        let slice_idx = offset - skip;
        let Some(val) = slice.get(slice_idx).cloned() else {
            continue;
        };
        consumed[offset] = Some(val.clone());
        if inputs[*input_idx].is_none() {
            // For reductions, only populate from stack if the value is a numeric tensor.
            // This avoids accidentally binding non-tensor metadata (e.g., dim strings) into the fused kernel inputs.
            let allow_stack_value = if plan.group.kind.is_reduction() {
                matches!(val, Value::GpuTensor(_) | Value::Tensor(_))
            } else {
                true
            };
            if allow_stack_value {
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
        .map(|opt| opt.ok_or_else(|| "fusion: missing input value".to_string()))
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
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => Err(err.to_string().into()),
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
            for v in consumed.iter().filter_map(|v| v.as_ref()) {
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
                    // If this input came from the stack, it will have an entry in stack_pattern; use consumed value
                    if let Some(stack_offset) = plan
                        .stack_pattern
                        .iter()
                        .position(|&idx| idx == input_index)
                    {
                        if let Some(val) = consumed.get(stack_offset).and_then(|v| v.as_ref()) {
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
                for v in consumed.iter().filter_map(|v| v.as_ref()) {
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
                    for value in consumed.iter().filter_map(|v| v.as_ref()) {
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
                    return Err("fusion: reduction all extent unknown".to_string().into());
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
            for v in consumed.iter().filter_map(|v| v.as_ref()) {
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
            return Err("fusion: reduction shape unresolved".to_string().into());
        }

        // Optional escape hatch: disable fused reductions to force provider path
        if std::env::var("RUNMAT_DISABLE_FUSED_REDUCTION")
            .ok()
            .as_deref()
            == Some("1")
        {
            return Err("fusion: fused reductions disabled".to_string().into());
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
            Err(err) => Err(err.to_string().into()),
        }
    } else if plan.group.kind == FusionKind::CenteredGram {
        match execute_centered_gram(request) {
            Ok(result) => {
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => Err(err.to_string().into()),
        }
    } else if plan.group.kind == FusionKind::PowerStepNormalize {
        match execute_power_step_normalize(request) {
            Ok(result) => {
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => Err(err.to_string().into()),
        }
    } else if plan.group.kind == FusionKind::ExplainedVariance {
        log::debug!("explained variance plan inputs {:?}", plan.inputs);
        match execute_explained_variance(request) {
            Ok(result) => {
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => {
                log::debug!("explained variance fusion fallback: {}", err);
                Err(err.to_string().into())
            }
        }
    } else if plan.group.kind == FusionKind::MatmulEpilogue {
        match execute_matmul_epilogue(request) {
            Ok(result) => {
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => Err(err.to_string().into()),
        }
    } else if plan.group.kind == FusionKind::ImageNormalize {
        match execute_image_normalize(request) {
            Ok(result) => {
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => Err(err.to_string().into()),
        }
    } else {
        // Unknown fusion kind; restore stack and report
        Err("fusion: unsupported fusion kind".to_string().into())
    }
}

#[cfg(feature = "native-accel")]
fn clear_residency(value: &Value) {
    if let Value::GpuTensor(handle) = value {
        fusion_residency::clear(handle);
    }
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
            format!("{ERROR_NAMESPACE}:error")
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
            format!("{ERROR_NAMESPACE}:error")
        } else {
            id.trim().to_string()
        };
        runmat_builtins::MException::new(ident, message)
    } else {
        runmat_builtins::MException::new(format!("{ERROR_NAMESPACE}:error"), message.to_string())
    }
}

fn map_slice_plan_error(context: &str, flow: RuntimeControlFlow) -> RuntimeControlFlow {
    match flow {
        RuntimeControlFlow::Error(err) => {
            let is_oob = err
                .identifier()
                .map(|id| id.contains("IndexOutOfBounds"))
                .unwrap_or_else(|| err.message().contains("IndexOutOfBounds"));
            if is_oob {
                RuntimeControlFlow::Error(err)
            } else {
                build_runtime_error(format!("{context}: {}", err.message()))
                    .build()
                    .into()
            }
        }
        RuntimeControlFlow::Suspend(pending) => RuntimeControlFlow::Suspend(pending),
    }
}

fn flow_to_string(flow: RuntimeControlFlow) -> String {
    match flow {
        RuntimeControlFlow::Error(err) => err.message().to_string(),
        RuntimeControlFlow::Suspend(_) => "interaction pending is unsupported here".to_string(),
    }
}

/// Interpret bytecode with default variable initialization
pub fn interpret(bytecode: &Bytecode) -> Result<Vec<Value>, String> {
    let mut vars = vec![Value::Num(0.0); bytecode.var_count];
    match interpret_with_vars(bytecode, &mut vars, Some("<main>")) {
        Ok(InterpreterOutcome::Completed(values)) => Ok(values),
        Ok(InterpreterOutcome::Pending(_)) => {
            Err("interaction pending is unsupported in interpret".to_string())
        }
        Err(e) => Err(flow_to_string(e)),
    }
}

pub fn interpret_function(bytecode: &Bytecode, vars: Vec<Value>) -> Result<Vec<Value>, String> {
    // Delegate to the counted variant with anonymous name and zero counts
    interpret_function_with_counts(bytecode, vars, "<anonymous>", 0, 0)
}

fn interpret_function_with_counts(
    bytecode: &Bytecode,
    mut vars: Vec<Value>,
    name: &str,
    out_count: usize,
    in_count: usize,
) -> Result<Vec<Value>, String> {
    // Push (nargin, nargout), run, then pop
    let res = CALL_COUNTS.with(|cc| {
        cc.borrow_mut().push((in_count, out_count));
        let r = interpret_with_vars(bytecode, &mut vars, Some(name));
        cc.borrow_mut().pop();
        r
    });
    let res = match res {
        Ok(InterpreterOutcome::Completed(values)) => Ok(values),
        Ok(InterpreterOutcome::Pending(_)) => {
            Err("interaction pending is unsupported in interpret_function".to_string())
        }
        Err(e) => Err(flow_to_string(e)),
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
