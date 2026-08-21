use runmat_mir::{MirIndexComponent, MirIndexing, MirOperand};
use runmat_native_codegen::{NativeIndexBound, NativeIndexExpressionKind, NativeRangeExpression};
use runmat_runtime::indexing::plan::{build_index_plan, IndexPlan};
use runmat_runtime::indexing::read_slice;
use runmat_runtime::indexing::selectors::build_slice_selectors;
use runmat_runtime::indexing::write_slice;
use runmat_runtime::object::dispatch::call_object_index_descriptor_method;
use runmat_runtime::object::indexing::ObjectIndexDescriptor;
use runmat_runtime::RuntimeError;
use runmat_types::{IndexKind, IndexResultContext};
use runmat_value::{CharArray, LogicalArray, ObjectArray, SymbolicArray, Value};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::operand::materialize_operand;
use super::state::HostState;

pub(super) fn read(
    state: &mut HostState,
    base: &MirOperand,
    indexing: &MirIndexing,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    let base = materialize_operand(state, base)?;
    read_value(state, base, indexing, requested_outputs)
}

pub(super) fn read_value(
    state: &mut HostState,
    base: Value,
    indexing: &MirIndexing,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    let selectors = materialize_selectors(state, &base, indexing)?;
    match indexing.kind {
        IndexKind::Paren => read_paren(state, base, selectors, requested_outputs),
        IndexKind::Brace => read_brace(base, selectors, indexing, requested_outputs),
    }
}

pub(super) fn assign(
    state: &mut HostState,
    base: Value,
    indexing: &MirIndexing,
    rhs: Value,
    delete: bool,
) -> NativeExecutorResult<Value> {
    let selectors = materialize_selectors(state, &base, indexing)?;
    if indexing.kind == IndexKind::Brace {
        return assign_brace(base, selectors, rhs, delete);
    }
    if matches!(base, Value::Object(_) | Value::HandleObject(_)) {
        let descriptor = ObjectIndexDescriptor::subsasgn_paren_from_slice(
            base,
            selectors.dims,
            selectors.colon_mask,
            selectors.end_mask,
            &selectors.numeric,
            rhs,
        )?;
        return super::sync::complete(
            &state.runtime,
            call_object_index_descriptor_method(descriptor),
            "object indexed assignment",
        );
    }
    let plan = super::sync::complete(
        &state.runtime,
        plan_for_assignment(&base, &selectors, !delete),
        "indexed assignment planning",
    )?;
    let updated = match base {
        Value::Tensor(value) => {
            if delete {
                write_slice::delete_tensor_with_plan(value, &plan, &rhs)
                    .map_err(NativeExecutorError::from)
            } else {
                super::sync::complete(
                    &state.runtime,
                    write_slice::assign_tensor_with_plan(value, &plan, &rhs),
                    "tensor indexed assignment",
                )
            }
        }
        Value::ComplexTensor(value) => {
            if delete {
                write_slice::delete_complex_with_plan(value, &plan, &rhs)
                    .map_err(NativeExecutorError::from)
            } else {
                super::sync::complete(
                    &state.runtime,
                    write_slice::assign_complex_with_plan(value, &plan, &rhs),
                    "complex indexed assignment",
                )
            }
        }
        Value::SparseTensor(value) => {
            if delete {
                write_slice::delete_sparse_with_plan(value, &plan, &rhs)
                    .map_err(NativeExecutorError::from)
            } else {
                super::sync::complete(
                    &state.runtime,
                    write_slice::assign_sparse_with_plan(value, &plan, &rhs),
                    "sparse indexed assignment",
                )
            }
        }
        Value::GpuTensor(value) => {
            if delete {
                super::sync::complete(
                    &state.runtime,
                    write_slice::delete_gpu_slice_with_plan(&value, &plan, &rhs),
                    "GPU indexed deletion",
                )
            } else {
                super::sync::complete(
                    &state.runtime,
                    write_slice::assign_gpu_slice_with_plan(&value, &plan, &rhs),
                    "GPU indexed assignment",
                )
            }
        }
        Value::Cell(value) => {
            let indices = plan
                .indices
                .iter()
                .map(|index| *index as usize + 1)
                .collect::<Vec<_>>();
            runmat_runtime::object::cell::assign_cell_paren_linear_indices_with_policy(
                value, &indices, &rhs, delete,
            )
            .map_err(NativeExecutorError::from)
        }
        Value::StringArray(mut value) if !delete => {
            if !plan.indices.is_empty() {
                let rhs = write_slice::build_string_rhs_view(&rhs, &plan.selection_lengths)
                    .map_err(NativeExecutorError::from)?;
                write_slice::scatter_string_with_plan(&mut value, &plan, &rhs)
                    .map_err(NativeExecutorError::from)?;
            }
            Ok(Value::StringArray(value))
        }
        Value::LogicalArray(value) => assign_logical(state, value, &plan, &rhs, delete),
        _ => Err(NativeExecutorError::from(semantic_error(
            "SliceNonTensor",
            "Indexed assignment is unsupported for this value",
        ))),
    }?;
    Ok(updated)
}

struct MaterializedSelectors {
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: Vec<Value>,
    positional: Vec<Value>,
}

fn materialize_selectors(
    state: &mut HostState,
    base: &Value,
    indexing: &MirIndexing,
) -> NativeExecutorResult<MaterializedSelectors> {
    if indexing.components.len() > u32::BITS as usize {
        return Err(NativeExecutorError::Host(
            "verified Native IR exceeds the shared 32-dimension selector mask".into(),
        ));
    }
    let shape = value_shape(base);
    let dims = indexing.components.len();
    let mut colon_mask = 0_u32;
    let mut end_mask = 0_u32;
    let mut numeric = Vec::new();
    let mut positional = Vec::with_capacity(dims);
    for (dimension, component) in indexing.components.iter().enumerate() {
        match component {
            MirIndexComponent::Colon => {
                colon_mask |= 1_u32 << dimension;
                positional.push(Value::String(":".into()));
            }
            MirIndexComponent::End { dim, offset } => {
                let resolved_dimension = dim.unwrap_or(dimension);
                let extent = selector_dimension_length(&shape, dims, resolved_dimension)?;
                let resolved = extent.checked_add_signed(*offset).ok_or_else(|| {
                    NativeExecutorError::from(runmat_runtime::runtime_error::semantic_error(
                        "IndexOutOfBounds",
                        "Index out of bounds",
                    ))
                })?;
                let value = Value::Num(resolved as f64);
                if *offset == 0 {
                    end_mask |= 1_u32 << dimension;
                } else {
                    numeric.push(value.clone());
                }
                positional.push(value);
            }
            MirIndexComponent::Expr(operand) => {
                let value =
                    materialize_selector_expression(state, operand, &shape, dims, dimension)?;
                numeric.push(value.clone());
                positional.push(value);
            }
        }
    }
    Ok(MaterializedSelectors {
        dims,
        colon_mask,
        end_mask,
        numeric,
        positional,
    })
}

fn materialize_selector_expression(
    state: &mut HostState,
    operand: &MirOperand,
    shape: &[usize],
    dims: usize,
    dimension: usize,
) -> NativeExecutorResult<Value> {
    let MirOperand::Local(local) = operand else {
        return materialize_operand(state, operand);
    };
    let local = u32::try_from(local.0)
        .map(runmat_native_codegen::NativeLocalId)
        .map_err(|_| NativeExecutorError::Host("selector local exceeds native schema".into()))?;
    let Some(expression) = state.function.index_expression(local).cloned() else {
        return materialize_operand(state, operand);
    };
    let dimension_length = selector_dimension_length(shape, dims, dimension)?;
    match expression.kind {
        NativeIndexExpressionKind::Scalar(expression) => {
            resolve_end_expression(state, dimension_length, &expression).map(Value::Num)
        }
        NativeIndexExpressionKind::Range(range) => {
            materialize_range_expression(state, dimension_length, &range)
        }
    }
}

fn materialize_range_expression(
    state: &mut HostState,
    dimension_length: usize,
    range: &NativeRangeExpression,
) -> NativeExecutorResult<Value> {
    let start = materialize_index_bound(state, dimension_length, &range.start)?;
    let step = range
        .step
        .as_ref()
        .map(|step| materialize_index_bound(state, dimension_length, step))
        .transpose()?;
    let end = resolve_end_expression(state, dimension_length, &range.end)?;
    let mut arguments = vec![Value::Num(start)];
    if let Some(step) = step {
        arguments.push(Value::Num(step));
    }
    arguments.push(Value::Num(end));
    let mut values = super::call::builtin(state, "colon", arguments, 1)?;
    if values.len() != 1 {
        return Err(NativeExecutorError::Host(
            "context-dependent range did not produce one selector".into(),
        ));
    }
    Ok(values.remove(0))
}

fn materialize_index_bound(
    state: &mut HostState,
    dimension_length: usize,
    bound: &NativeIndexBound,
) -> NativeExecutorResult<f64> {
    match bound {
        NativeIndexBound::Expression(expression) => {
            resolve_end_expression(state, dimension_length, expression)
        }
        NativeIndexBound::Operand(MirOperand::Constant(runmat_mir::MirConstant::Number(value))) => {
            value.parse().map_err(|error| {
                NativeExecutorError::Host(format!("invalid range bound {value:?}: {error}"))
            })
        }
        NativeIndexBound::Operand(MirOperand::Local(local)) => resolve_end_expression(
            state,
            dimension_length,
            &runmat_runtime::indexing::EndExpr::Var(local.0),
        ),
        NativeIndexBound::Operand(operand) => {
            let value = materialize_operand(state, operand)?;
            runmat_runtime::indexing::value_to_f64(&value).map_err(|_| {
                NativeExecutorError::from(semantic_error(
                    "UnsupportedIndexType",
                    "range bound must be numeric",
                ))
            })
        }
    }
}

fn resolve_end_expression(
    state: &HostState,
    dimension_length: usize,
    expression: &runmat_runtime::indexing::EndExpr,
) -> NativeExecutorResult<f64> {
    super::sync::complete(
        &state.runtime,
        runmat_runtime::indexing::resolve_end_expr_value(dimension_length, expression, |local| {
            state
                .locals
                .get(local)
                .copied()
                .filter(|value| !value.is_null())
                .and_then(|value| state.arena.get(value).ok().cloned())
        }),
        "end expression resolution",
    )
}

fn selector_dimension_length(
    shape: &[usize],
    dims: usize,
    dimension: usize,
) -> NativeExecutorResult<usize> {
    if dims == 1 {
        shape
            .iter()
            .try_fold(1_usize, |total, extent| total.checked_mul(*extent))
            .ok_or_else(|| NativeExecutorError::Host("index shape exceeds platform limits".into()))
    } else {
        Ok(*shape.get(dimension).unwrap_or(&1))
    }
}

fn read_paren(
    state: &HostState,
    base: Value,
    selectors: MaterializedSelectors,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    if is_function_value(&base) {
        if selectors.colon_mask != 0 || selectors.end_mask != 0 {
            return Err(NativeExecutorError::from(semantic_error(
                "UnsupportedFunctionHandleSelector",
                "Function handle call does not support colon or end selector syntax",
            )));
        }
        let value = super::sync::complete(
            &state.runtime,
            runmat_runtime::call_feval_async_with_outputs(
                base,
                &selectors.numeric,
                requested_outputs,
            ),
            "function-handle indexed call",
        )?;
        return normalize_outputs(value, requested_outputs);
    }
    if matches!(base, Value::Object(_) | Value::HandleObject(_)) {
        let descriptor = ObjectIndexDescriptor::subsref_paren_from_slice(
            base,
            selectors.dims,
            selectors.colon_mask,
            selectors.end_mask,
            &selectors.numeric,
        )?;
        let value = super::sync::complete(
            &state.runtime,
            call_object_index_descriptor_method(descriptor),
            "object indexed read",
        )?;
        return normalize_outputs(value, requested_outputs);
    }
    let plan = super::sync::complete(
        &state.runtime,
        plan_for(&base, &selectors),
        "indexed read planning",
    )?;
    let value = read_with_plan(base, &plan)?;
    normalize_outputs(value, requested_outputs)
}

fn read_brace(
    base: Value,
    selectors: MaterializedSelectors,
    indexing: &MirIndexing,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    let Value::Cell(cell) = base else {
        return Err(NativeExecutorError::from(semantic_error(
            "CellIndexType",
            "Brace indexing requires a cell array",
        )));
    };
    let values = runmat_runtime::object::cell::expand_cell_indices(&cell, &selectors.positional)?;
    if indexing.result_context == IndexResultContext::ReadCommaList || indexing.cell_expand_all {
        if values.len() != requested_outputs {
            return Err(NativeExecutorError::from(semantic_error(
                "OutputArityMismatch",
                format!(
                    "cell comma-list produced {} values for {requested_outputs} outputs",
                    values.len()
                ),
            )));
        }
        return Ok(values);
    }
    match values.as_slice() {
        [value] if requested_outputs == 1 => Ok(vec![value.clone()]),
        _ => Err(NativeExecutorError::from(semantic_error(
            "CellIndexArity",
            "Cell brace indexing must select exactly one value in scalar context",
        ))),
    }
}

async fn plan_for(
    base: &Value,
    selectors: &MaterializedSelectors,
) -> NativeExecutorResult<IndexPlan> {
    let shape = value_shape(base);
    let built = build_slice_selectors(
        selectors.dims,
        selectors.colon_mask,
        selectors.end_mask,
        &selectors.numeric,
        &shape,
    )
    .await?;
    build_index_plan(&built, selectors.dims, &shape).map_err(NativeExecutorError::from)
}

async fn plan_for_assignment(
    base: &Value,
    selectors: &MaterializedSelectors,
    allow_sparse_growth: bool,
) -> NativeExecutorResult<IndexPlan> {
    let shape = value_shape(base);
    if matches!(base, Value::SparseTensor(_)) && allow_sparse_growth {
        let built = runmat_runtime::indexing::selectors::build_sparse_assignment_selectors(
            selectors.dims,
            selectors.colon_mask,
            selectors.end_mask,
            &selectors.numeric,
            &shape,
        )
        .await?;
        return runmat_runtime::indexing::plan::build_sparse_assignment_plan(
            &built,
            selectors.dims,
            &shape,
        )
        .map_err(NativeExecutorError::from);
    }
    plan_for(base, selectors).await
}

fn assign_brace(
    base: Value,
    selectors: MaterializedSelectors,
    rhs: Value,
    delete: bool,
) -> NativeExecutorResult<Value> {
    if delete {
        return Err(NativeExecutorError::from(semantic_error(
            "UnsupportedCellBraceDeletion",
            "Cell brace assignment does not support deletion",
        )));
    }
    let Value::Cell(cell) = base else {
        return Err(NativeExecutorError::from(semantic_error(
            "CellIndexType",
            "Brace assignment requires a cell array",
        )));
    };
    let positions = runmat_runtime::object::cell::resolve_cell_assignment_positions(
        &cell,
        &selectors.positional,
    )?;
    let values = match rhs {
        Value::OutputList(values) => values,
        value if positions.len() == 1 => vec![value],
        value => vec![value; positions.len()],
    };
    runmat_runtime::object::cell::assign_cell_value_multi(cell, &positions, &values, |_, _| {})
        .map_err(NativeExecutorError::from)
}

fn assign_logical(
    state: &HostState,
    value: LogicalArray,
    plan: &IndexPlan,
    rhs: &Value,
    delete: bool,
) -> NativeExecutorResult<Value> {
    let tensor = runmat_value::Tensor::new(
        value
            .data
            .iter()
            .map(|value| if *value != 0 { 1.0 } else { 0.0 })
            .collect(),
        value.shape,
    )
    .map_err(|error| NativeExecutorError::from(shape_error(error)))?;
    let updated = if delete {
        write_slice::delete_tensor_with_plan(tensor, plan, rhs)
            .map_err(NativeExecutorError::from)?
    } else {
        super::sync::complete(
            &state.runtime,
            write_slice::assign_tensor_with_plan(tensor, plan, rhs),
            "logical indexed assignment",
        )?
    };
    match updated {
        Value::Num(value) => Ok(Value::Bool(value != 0.0)),
        Value::Tensor(value) => {
            let data = value
                .materialize_f64()
                .into_iter()
                .map(|value| u8::from(value != 0.0))
                .collect();
            LogicalArray::new(data, value.shape)
                .map(Value::LogicalArray)
                .map_err(|error| NativeExecutorError::from(shape_error(error)))
        }
        value => Ok(value),
    }
}

fn read_with_plan(base: Value, plan: &IndexPlan) -> NativeExecutorResult<Value> {
    match base {
        Value::Tensor(value) => {
            read_slice::read_tensor_slice_from_plan(&value, plan).map_err(NativeExecutorError::from)
        }
        Value::ComplexTensor(value) => read_slice::read_complex_slice_from_plan(&value, plan)
            .map_err(NativeExecutorError::from),
        Value::SparseTensor(value) => {
            read_slice::read_sparse_slice_from_plan(&value, plan).map_err(NativeExecutorError::from)
        }
        Value::GpuTensor(value) => {
            read_slice::read_gpu_slice_from_plan(&value, plan).map_err(NativeExecutorError::from)
        }
        Value::StringArray(value) => {
            read_slice::gather_string_slice(&value, plan).map_err(NativeExecutorError::from)
        }
        Value::LogicalArray(value) => gather_logical(&value, plan),
        Value::CharArray(value) => gather_char(&value, plan),
        Value::Cell(value) => {
            let indices = plan
                .indices
                .iter()
                .map(|index| *index as usize + 1)
                .collect::<Vec<_>>();
            runmat_runtime::object::cell::gather_cell_paren_linear_indices(
                &value,
                &indices,
                &plan.output_shape,
            )
            .map_err(NativeExecutorError::from)
        }
        Value::ObjectArray(value) => gather_object(&value, plan),
        Value::SymbolicArray(value) => gather_symbolic(&value, plan),
        value => read_scalar(value, plan),
    }
}

fn gather_logical(value: &LogicalArray, plan: &IndexPlan) -> NativeExecutorResult<Value> {
    let values = gather(&value.data, plan, "logical")?;
    if let [value] = values.as_slice() {
        return Ok(Value::Bool(*value != 0));
    }
    LogicalArray::new(values, plan.output_shape.clone())
        .map(Value::LogicalArray)
        .map_err(|error| NativeExecutorError::from(shape_error(error)))
}

fn gather_char(value: &CharArray, plan: &IndexPlan) -> NativeExecutorResult<Value> {
    let values = gather(&value.to_column_major(), plan, "character")?;
    CharArray::from_column_major(values, plan.output_shape.clone())
        .map(Value::CharArray)
        .map_err(|error| NativeExecutorError::from(shape_error(error)))
}

fn gather_object(value: &ObjectArray, plan: &IndexPlan) -> NativeExecutorResult<Value> {
    if let [index] = plan.indices.as_slice() {
        return value.get_linear(*index as usize).cloned().ok_or_else(|| {
            NativeExecutorError::from(semantic_error("IndexOutOfBounds", "Index out of bounds"))
        });
    }
    let indices = plan
        .indices
        .iter()
        .map(|index| *index as usize)
        .collect::<Vec<_>>();
    value
        .select_linear(&indices, plan.output_shape.clone())
        .map(Value::ObjectArray)
        .map_err(|error| NativeExecutorError::from(shape_error(error)))
}

fn gather_symbolic(value: &SymbolicArray, plan: &IndexPlan) -> NativeExecutorResult<Value> {
    let values = gather(&value.data, plan, "symbolic")?;
    if let [value] = values.as_slice() {
        return Ok(Value::Symbolic(value.clone()));
    }
    SymbolicArray::new(values, plan.output_shape.clone())
        .map(Value::SymbolicArray)
        .map_err(|error| NativeExecutorError::from(shape_error(error)))
}

fn read_scalar(value: Value, plan: &IndexPlan) -> NativeExecutorResult<Value> {
    if plan.indices.as_slice() == [0] {
        Ok(value)
    } else {
        Err(NativeExecutorError::from(semantic_error(
            "IndexOutOfBounds",
            "Index out of bounds",
        )))
    }
}

fn gather<T: Clone>(data: &[T], plan: &IndexPlan, kind: &str) -> NativeExecutorResult<Vec<T>> {
    plan.indices
        .iter()
        .map(|index| {
            data.get(*index as usize).cloned().ok_or_else(|| {
                NativeExecutorError::from(semantic_error(
                    "IndexOutOfBounds",
                    format!("{kind} index is out of bounds"),
                ))
            })
        })
        .collect()
}

fn normalize_outputs(value: Value, requested_outputs: usize) -> NativeExecutorResult<Vec<Value>> {
    match (requested_outputs, value) {
        (0, _) => Ok(Vec::new()),
        (1, Value::OutputList(mut values)) if values.len() == 1 => Ok(vec![values.remove(0)]),
        (1, value) => Ok(vec![value]),
        (expected, Value::OutputList(values)) if values.len() == expected => Ok(values),
        (expected, value) => Err(NativeExecutorError::from(semantic_error(
            "OutputArityMismatch",
            format!("indexing produced one value for {expected} outputs: {value:?}"),
        ))),
    }
}

fn value_shape(value: &Value) -> Vec<usize> {
    match value {
        Value::Tensor(value) => value.shape.clone(),
        Value::ComplexTensor(value) => value.shape.clone(),
        Value::SparseTensor(value) => value.shape(),
        Value::GpuTensor(value) => value.shape.clone(),
        Value::StringArray(value) => value.shape.clone(),
        Value::LogicalArray(value) => value.shape.clone(),
        Value::CharArray(value) => value.shape().to_vec(),
        Value::Cell(value) => value.shape.clone(),
        Value::ObjectArray(value) => value.shape().to_vec(),
        Value::SymbolicArray(value) => value.shape.clone(),
        _ => vec![1, 1],
    }
}

fn is_function_value(value: &Value) -> bool {
    matches!(
        value,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    )
}

fn semantic_error(identifier: &str, message: impl Into<String>) -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(identifier, message)
}

fn shape_error(error: impl std::fmt::Display) -> RuntimeError {
    semantic_error("ShapeMismatch", error.to_string())
}
