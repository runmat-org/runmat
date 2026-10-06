use futures::future::{Either, LocalBoxFuture, Ready};
mod brace;
mod cell_read;
mod cell_write;
mod linear_write;
mod paren;
mod slice_read;
mod slice_write;
use brace::*;
use paren::apply_symbolic_function_reference;
pub use paren::paren_index_value;
#[cfg(test)]
use paren::symbolic_scalar_from_value;

use runmat_runtime::builtins::common::tensor::{tensor_value_f64, tensor_values_f64_cow};
use runmat_runtime::call::arguments::expand_brace_values;
use runmat_runtime::indexing as runtime_indexing;
use runmat_runtime::indexing::plan::{
    build_expr_index_plan, build_expr_sparse_assignment_plan, build_index_plan,
    build_sparse_assignment_plan, ExprPlanSpec, IndexPlan,
};
use runmat_runtime::indexing::read_linear as idx_read_linear;
use runmat_runtime::indexing::read_slice as idx_read_slice;
use runmat_runtime::indexing::selectors::{
    build_cell_scalar_selectors, build_slice_selectors, index_scalar_from_value, SliceSelector,
};
use runmat_runtime::indexing::write_linear as idx_write_linear;
use runmat_runtime::indexing::write_slice as idx_write_slice;
use runmat_runtime::indexing::EndExpr;
use runmat_runtime::object::dispatch::{
    call_object_index_descriptor_method, call_object_index_descriptor_method_with_outputs,
    class_defines_member_subsasgn, class_defines_member_subsref,
};
use runmat_runtime::object::indexing::{
    ObjectIndexDescriptor, ObjectIndexOp, ObjectIndexSelector, ObjectParenExprSelectorSpec,
};
use runmat_runtime::{build_runtime_error, RuntimeError};
use runmat_value::{CellArray, IntValue, IntegerStorage, SymbolicExpr, Tensor, Value};

fn map_slice_plan_error(context: &str, err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{context}: {}", err.message()));
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    builder.build()
}

fn map_slice_shape_error(context: &str, err: impl std::fmt::Display) -> RuntimeError {
    crate::interpreter::errors::mex("ShapeMismatch", &format!("{context}: {err}"))
}

/// Materializes an integer scalar for indexed assignment without routing it
/// through the lossy floating-point tensor mirror.
fn integer_scalar_tensor(value: IntValue) -> Result<Tensor, RuntimeError> {
    let storage = match value {
        IntValue::I8(value) => IntegerStorage::I8(vec![value]),
        IntValue::I16(value) => IntegerStorage::I16(vec![value]),
        IntValue::I32(value) => IntegerStorage::I32(vec![value]),
        IntValue::I64(value) => IntegerStorage::I64(vec![value]),
        IntValue::U8(value) => IntegerStorage::U8(vec![value]),
        IntValue::U16(value) => IntegerStorage::U16(vec![value]),
        IntValue::U32(value) => IntegerStorage::U32(vec![value]),
        IntValue::U64(value) => IntegerStorage::U64(vec![value]),
    };
    Tensor::new_integer(storage, vec![1, 1])
        .map_err(|e| map_slice_shape_error("scalar index assign", e))
}

/// Applies a slice assignment to an integer scalar after materializing its
/// exact storage. This keeps StoreSlice and StoreSliceExpr on the same path as
/// StoreIndex instead of constructing an f64 compatibility tensor first.
async fn assign_integer_scalar_with_plan(
    value: IntValue,
    plan: &IndexPlan,
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let tensor = integer_scalar_tensor(value)?;
    if delete {
        idx_write_slice::delete_tensor_with_plan(tensor, plan, rhs)
    } else {
        idx_write_slice::assign_tensor_with_plan(tensor, plan, rhs).await
    }
}

fn logical_value_from_tensor(t: runmat_value::Tensor) -> Result<Value, RuntimeError> {
    let logical_data: Vec<u8> = tensor_values_f64_cow(&t)
        .iter()
        .map(|&v| if v != 0.0 { 1 } else { 0 })
        .collect();
    if logical_data.len() <= 1 {
        Ok(Value::Bool(logical_data.first().copied().unwrap_or(0) != 0))
    } else {
        let logical = runmat_value::LogicalArray::new(logical_data, t.shape.clone())
            .map_err(|e| map_slice_shape_error("slice assign", e))?;
        Ok(Value::LogicalArray(logical))
    }
}

fn map_cell_scalar_selector_error(err: RuntimeError) -> RuntimeError {
    match err.identifier() {
        Some("RunMat:ScalarIndexRequired") => {
            crate::interpreter::errors::mex("CellIndexType", "Unsupported cell index type")
        }
        Some("RunMat:IndexOutOfBounds") => {
            crate::interpreter::errors::mex("CellIndexOutOfBounds", "Cell index out of bounds")
        }
        _ => err,
    }
}

async fn read_scalar_like_slice(
    base: &Value,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    if let Value::LogicalArray(la) = base {
        let data: Vec<f64> = la
            .data
            .iter()
            .map(|&b| if b != 0 { 1.0 } else { 0.0 })
            .collect();
        let tensor = runmat_value::Tensor::new(data, la.shape.clone())
            .map_err(|e| map_slice_shape_error("slice", e))?;
        let sliced = if dims == 1 {
            idx_read_slice::read_tensor_slice_1d(&tensor, colon_mask, end_mask, numeric).await?
        } else {
            idx_read_slice::read_tensor_slice_nd(&tensor, dims, colon_mask, end_mask, numeric)
                .await?
        };
        return match sliced {
            Value::Tensor(t) => logical_value_from_tensor(t),
            Value::Num(n) => Ok(Value::Bool(n != 0.0)),
            other => Ok(other),
        };
    }

    if dims != 1 {
        return Err(crate::interpreter::errors::mex(
            "SliceNonTensor",
            "Slicing only supported on tensors",
        ));
    }
    if (colon_mask & 1u32) != 0 {
        return Err(crate::interpreter::errors::mex(
            "SliceNonTensor",
            "Slicing only supported on tensors",
        ));
    }
    let selectors = build_slice_selectors(1, colon_mask, end_mask, numeric, &[1usize]).await?;
    let linear_indices: Vec<f64> = match selectors.first() {
        Some(SliceSelector::Scalar(index)) => vec![*index as f64],
        Some(SliceSelector::Indices(indices)) => {
            indices.iter().map(|&index| index as f64).collect()
        }
        Some(SliceSelector::LinearIndices { values, .. }) => {
            values.iter().map(|&index| index as f64).collect()
        }
        Some(SliceSelector::Colon) | None => {
            return Err(crate::interpreter::errors::mex(
                "SliceNonTensor",
                "Slicing only supported on tensors",
            ));
        }
    };
    if matches!(base, Value::Symbolic(_)) {
        if linear_indices.len() == 1 && linear_indices[0] == 1.0 {
            return Ok(base.clone());
        }
        return Err(crate::interpreter::errors::mex(
            "SliceNonTensor",
            "Slicing only supported on tensors",
        ));
    }
    runmat_runtime::perform_indexing(base, &linear_indices)
        .await
        .map_err(|_| {
            crate::interpreter::errors::mex("SliceNonTensor", "Slicing only supported on tensors")
        })
}

fn missing_member_index_overload_error(base: &Value, op: ObjectIndexOp) -> Option<RuntimeError> {
    let class_name = match base {
        Value::Object(obj) => obj.class_name.as_str(),
        Value::HandleObject(handle) => handle.class_name.as_str(),
        _ => return None,
    };
    let class = runmat_runtime::class_registry::get_class(class_name)?;
    let supported = match op {
        ObjectIndexOp::Subsref => class_defines_member_subsref(&class),
        ObjectIndexOp::Subsasgn => class_defines_member_subsasgn(&class),
    };
    if supported {
        return None;
    }
    match op {
        ObjectIndexOp::Subsref => Some(crate::interpreter::errors::mex(
            "MissingSubsref",
            "class does not define subsref for indexing operation",
        )),
        ObjectIndexOp::Subsasgn => Some(crate::interpreter::errors::mex(
            "MissingSubsasgn",
            "class does not define subsasgn for indexed assignment",
        )),
    }
}

async fn linear_index_values(values: &[Value]) -> Result<Vec<usize>, RuntimeError> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let mut index_value = value.clone();
        if matches!(index_value, Value::GpuTensor(_)) {
            index_value = runmat_runtime::dispatcher::gather_if_needed_async(&index_value).await?;
        }
        let index_val = index_scalar_from_value(&index_value)
            .await?
            .ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "UnsupportedIndexType",
                    &format!("Unsupported index type: expected numeric scalar, got {value:?}"),
                )
            })?;
        let index = index_val.positive_usize().ok_or_else(|| {
            crate::interpreter::errors::mex("IndexOutOfBounds", "Index out of bounds")
        })?;
        out.push(index);
    }
    Ok(out)
}

async fn range_selector_scalar_to_f64(value: &Value) -> Result<f64, RuntimeError> {
    let mut scalar = value.clone();
    if matches!(scalar, Value::GpuTensor(_)) {
        scalar = runmat_runtime::dispatcher::gather_if_needed_async(&scalar).await?;
    }
    match scalar {
        Value::Num(n) => Ok(n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(t) if runmat_runtime::builtins::common::tensor::is_scalar_tensor(&t) => {
            Ok(tensor_value_f64(&t, 0))
        }
        _ => Err(crate::interpreter::errors::mex(
            "UnsupportedIndexType",
            "Range selector operands must be numeric scalars",
        )),
    }
}

fn validate_expr_range_step_metadata(
    range_dims: &[usize],
    range_has_step: &[bool],
) -> Result<(), RuntimeError> {
    if range_dims.len() != range_has_step.len() {
        return Err(crate::interpreter::errors::mex(
            "InvalidRangeSelectorPlan",
            "inconsistent range step metadata",
        ));
    }
    Ok(())
}

fn assign_scalar_struct_index(
    _base: runmat_value::StructValue,
    indices: &[usize],
    rhs: Value,
) -> Result<Value, RuntimeError> {
    match indices {
        [1] | [1, 1] => Ok(rhs),
        _ => Err(crate::interpreter::errors::mex(
            "IndexOutOfBounds",
            "Struct subscript out of bounds",
        )),
    }
}

fn gather_cell_with_plan(
    ca: &CellArray,
    plan: &runmat_runtime::indexing::plan::IndexPlan,
) -> Result<Value, RuntimeError> {
    let indices: Vec<usize> = plan.indices.iter().map(|idx| (*idx as usize) + 1).collect();
    runmat_runtime::object::cell::gather_cell_paren_linear_indices(ca, &indices, &plan.output_shape)
}

fn gather_object_array_with_plan(
    array: &runmat_value::ObjectArray,
    plan: &runmat_runtime::indexing::plan::IndexPlan,
) -> Result<Value, RuntimeError> {
    if plan.indices.len() == 1 {
        return array
            .get_linear(plan.indices[0] as usize)
            .cloned()
            .ok_or_else(|| {
                crate::interpreter::errors::mex("IndexOutOfBounds", "Index out of bounds")
            });
    }
    let indices = plan
        .indices
        .iter()
        .map(|index| *index as usize)
        .collect::<Vec<_>>();
    array
        .select_linear(&indices, plan.output_shape.clone())
        .map(Value::ObjectArray)
        .map_err(|error| crate::interpreter::errors::mex("IndexOutOfBounds", &error))
}

fn pop_index_values(stack: &mut Vec<Value>, count: usize) -> Result<Vec<Value>, RuntimeError> {
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        let value = stack.pop().ok_or(crate::interpreter::errors::mex(
            "StackUnderflow",
            "stack underflow",
        ))?;
        values.push(value);
    }
    values.reverse();
    Ok(values)
}

fn pop_index_base(stack: &mut Vec<Value>) -> Result<Value, RuntimeError> {
    stack.pop().ok_or(crate::interpreter::errors::mex(
        "StackUnderflow",
        "stack underflow",
    ))
}

#[derive(Clone, Copy)]
struct IndexContext<'a> {
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    range_dims: &'a [usize],
    base_shape: &'a [usize],
}

impl<'a> IndexContext<'a> {
    fn new(
        dims: usize,
        colon_mask: u32,
        end_mask: u32,
        range_dims: &'a [usize],
        base_shape: &'a [usize],
    ) -> Self {
        Self {
            dims,
            colon_mask,
            end_mask,
            range_dims,
            base_shape,
        }
    }

    fn dim_len_for_numeric_position(&self, numeric_position: usize) -> usize {
        let mut seen_numeric = 0usize;
        let mut dim_for_pos = 0usize;
        for d in 0..self.dims {
            let is_colon = selector_mask_has_dim(self.colon_mask, d);
            let is_end = selector_mask_has_dim(self.end_mask, d);
            let is_range = self.range_dims.contains(&d);
            if is_colon || is_end || is_range {
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

fn selector_mask_has_dim(mask: u32, dim: usize) -> bool {
    dim < u32::BITS as usize && (mask & (1u32 << dim)) != 0
}

struct SparseFullRangeExprSpec<'a> {
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    range_dims: &'a [usize],
    range_params: &'a [(f64, f64)],
    range_start_exprs: &'a [Option<EndExpr>],
    range_step_exprs: &'a [Option<EndExpr>],
    range_end_exprs: &'a [EndExpr],
}

fn sparse_full_range_exprs_as_colon_mask(spec: SparseFullRangeExprSpec<'_>) -> Option<u32> {
    if spec.range_dims.is_empty()
        || spec.range_dims.len() != spec.range_params.len()
        || spec.range_dims.len() != spec.range_start_exprs.len()
        || spec.range_dims.len() != spec.range_step_exprs.len()
        || spec.range_dims.len() != spec.range_end_exprs.len()
    {
        return None;
    }
    let mut mask = spec.colon_mask;
    for (pos, &dim) in spec.range_dims.iter().enumerate() {
        if dim >= spec.dims || dim >= u32::BITS as usize {
            return None;
        }
        if selector_mask_has_dim(mask, dim) || selector_mask_has_dim(spec.end_mask, dim) {
            return None;
        }
        let (start, step) = spec.range_params[pos];
        if start != 1.0 || step != 1.0 {
            return None;
        }
        if spec.range_start_exprs[pos].is_some() || spec.range_step_exprs[pos].is_some() {
            return None;
        }
        if !matches!(spec.range_end_exprs[pos], EndExpr::End) {
            return None;
        }
        mask |= 1u32 << dim;
    }
    Some(mask)
}

fn validate_index_context_plan(ctx: IndexContext<'_>) -> Result<(), RuntimeError> {
    if ctx.dims > u32::BITS as usize {
        return Err(crate::interpreter::errors::mex(
            "InvalidRangeSelectorPlan",
            "selector dimension metadata exceeds mask width",
        ));
    }
    let mut seen = vec![false; ctx.dims];
    for &dim in ctx.range_dims {
        if dim >= ctx.dims {
            return Err(crate::interpreter::errors::mex(
                "InvalidRangeSelectorDim",
                "range selector dimension is out of bounds",
            ));
        }
        if std::mem::replace(&mut seen[dim], true) {
            return Err(crate::interpreter::errors::mex(
                "InvalidRangeSelectorPlan",
                "range selector dimension appears more than once",
            ));
        }
        if selector_mask_has_dim(ctx.colon_mask, dim) || selector_mask_has_dim(ctx.end_mask, dim) {
            return Err(crate::interpreter::errors::mex(
                "InvalidRangeSelectorPlan",
                "range selector conflicts with colon/end selector masks",
            ));
        }
    }
    Ok(())
}

async fn apply_end_offsets_to_numeric(
    numeric: &[Value],
    ctx: IndexContext<'_>,
    end_offsets: &[(usize, EndExpr)],
    vars: &mut [Value],
) -> Result<Vec<Value>, RuntimeError> {
    validate_index_context_plan(ctx)?;
    let mut seen = vec![false; numeric.len()];
    for (position, _) in end_offsets {
        if *position >= numeric.len() {
            return Err(crate::interpreter::errors::mex(
                "InvalidEndSelectorPlan",
                "end-selector position is out of bounds",
            ));
        }
        if std::mem::replace(&mut seen[*position], true) {
            return Err(crate::interpreter::errors::mex(
                "InvalidEndSelectorPlan",
                "end-selector position appears more than once",
            ));
        }
    }

    let mut adjusted = numeric.to_vec();
    for (position, end_expr) in end_offsets {
        if let Some(value) = adjusted.get_mut(*position) {
            let dim_len = ctx.dim_len_for_numeric_position(*position);
            let idx_val = resolve_end_expr_value(dim_len, end_expr, vars).await?;
            *value = Value::Num(idx_val);
        }
    }
    Ok(adjusted)
}

fn exact_index_from_f64(value: f64) -> Option<i64> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return None;
    }
    if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
        return None;
    }
    Some(rounded as i64)
}

async fn resolve_end_expr_value(
    dim_len: usize,
    end_expr: &EndExpr,
    vars: &[Value],
) -> Result<f64, RuntimeError> {
    runtime_indexing::resolve_end_expr_value(dim_len, end_expr, |local| vars.get(local).cloned())
        .await
}

async fn resolve_end_expr_index(
    dim_len: usize,
    end_expr: &EndExpr,
    vars: &[Value],
) -> Result<i64, RuntimeError> {
    let value = resolve_end_expr_value(dim_len, end_expr, vars).await?;
    exact_index_from_f64(value).ok_or_else(|| {
        crate::interpreter::errors::mex(
            "UnsupportedIndexType",
            "Index values must be positive integers or logical values",
        )
    })
}

async fn resolve_range_end_value(
    dim_len: usize,
    end_expr: &EndExpr,
    vars: &[Value],
) -> Result<f64, RuntimeError> {
    resolve_end_expr_value(dim_len, end_expr, vars).await
}

async fn build_expr_slice_plan(
    spec: ExprPlanSpec<'_>,
    vars: &[Value],
) -> Result<runmat_runtime::indexing::plan::IndexPlan, RuntimeError> {
    build_expr_index_plan(spec, |dim_len, expr| {
        let expr = expr.clone();
        async move { resolve_range_end_value(dim_len, &expr, vars).await }
    })
    .await
}

type IndexDispatch<'a> =
    Either<Ready<Result<bool, RuntimeError>>, LocalBoxFuture<'a, Result<bool, RuntimeError>>>;

pub fn dispatch_indexing<'a>(
    instr: &'a crate::bytecode::Instr,
    stack: &'a mut Vec<Value>,
    vars: &'a mut [Value],
    function_registry: &'a crate::bytecode::FunctionRegistry,
    pc: usize,
) -> IndexDispatch<'a> {
    // Each operation owns its async state. An indexing call must not retain
    // every unrelated assignment/read handler while polling a nested function.
    match instr {
        crate::bytecode::Instr::Index(..) => {
            Either::Right(Box::pin(paren::dispatch(instr, stack, function_registry)))
        }
        crate::bytecode::Instr::IndexCell { .. }
        | crate::bytecode::Instr::IndexCellExpand { .. }
        | crate::bytecode::Instr::IndexCellList { .. } => {
            Either::Right(Box::pin(cell_read::dispatch(instr, stack, vars)))
        }
        crate::bytecode::Instr::StoreIndexCell { .. }
        | crate::bytecode::Instr::StoreIndexCellDelete { .. } => {
            Either::Right(Box::pin(cell_write::dispatch(instr, stack, vars)))
        }
        crate::bytecode::Instr::StoreIndex(..) | crate::bytecode::Instr::StoreIndexDelete(..) => {
            Either::Right(Box::pin(linear_write::dispatch(instr, stack, pc)))
        }
        crate::bytecode::Instr::IndexSlice(..) | crate::bytecode::Instr::IndexSliceExpr { .. } => {
            Either::Right(Box::pin(slice_read::dispatch(
                instr,
                stack,
                vars,
                function_registry,
            )))
        }
        crate::bytecode::Instr::StoreSlice(..)
        | crate::bytecode::Instr::StoreSliceDelete(..)
        | crate::bytecode::Instr::StoreSliceExpr { .. }
        | crate::bytecode::Instr::StoreSliceExprDelete { .. } => {
            Either::Right(Box::pin(slice_write::dispatch(instr, stack, vars)))
        }
        _ => Either::Left(futures::future::ready(Ok(false))),
    }
}

#[cfg(test)]
mod tests;
