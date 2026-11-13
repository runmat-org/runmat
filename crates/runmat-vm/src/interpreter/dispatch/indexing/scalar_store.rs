use super::{
    assign_object_scalar_indices, idx_write_linear, idx_write_slice, RuntimeError, SliceSelector,
    Value,
};
use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::indexing::plan::{build_indexed_store_plan_in_context, IndexedStorePlanMode};
use runmat_runtime::indexing::value as index_value;
use runmat_value::IntValue;

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    _vars: &mut InterpreterVars,
    _function_registry: &crate::bytecode::FunctionRegistry,
    pc: usize,
    runtime: &runmat_runtime::context::RuntimeContext,
    _sequence_register: &mut super::super::SequenceState,
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::StoreIndex(num_indices)
        | crate::bytecode::Instr::StoreIndexDelete(num_indices) => {
            let delete = matches!(instr, crate::bytecode::Instr::StoreIndexDelete(_));
            if num_indices
                .checked_add(2)
                .is_none_or(|width| width > stack.len())
            {
                return Err(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ));
            }
            let mut indices = runmat_runtime::context::host_allocation::initialize_host_array(
                Some(runtime),
                *num_indices,
                |_| Ok(0usize),
            )?;
            let (numeric_owner, _operand_roots) =
                super::extract_selector_operands(stack, runtime, *num_indices, true)?;
            let rhs = stack
                .pop()
                .expect("validated scalar assignment window contains RHS");
            for (ordinal, value) in numeric_owner.iter().rev().enumerate() {
                let idx_val = runmat_runtime::indexing::selectors::index_scalar_from_value_in_context(Some(runtime), value).await?.ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "ScalarIndexRequired",
                        "StoreIndex requires scalar indices; use StoreSlice for vector, range, or logical indices",
                    )
                })?;
                if idx_val.is_below_one() {
                    return Err(crate::interpreter::errors::mex(
                        "IndexOutOfBounds",
                        "Index out of bounds",
                    ));
                }
                let index = idx_val.positive_usize().ok_or_else(|| {
                    crate::interpreter::errors::mex("IndexOutOfBounds", "Index out of bounds")
                })?;
                indices
                    .try_as_mut_slice_unique()
                    .expect("fresh exact scalar index storage")[*num_indices - ordinal - 1] = index;
            }
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            match base {
                Value::Foreign(reference) => {
                    if delete {
                        return Err(crate::interpreter::errors::mex(
                            "ForeignIndexDelete",
                            "Indexed deletion is not supported for foreign objects",
                        ));
                    }
                    let arguments =
                        runmat_runtime::execution::RootedValues::try_initialize_in_context(
                            indices.len(),
                            runtime,
                            |ordinal| {
                                Ok::<_, RuntimeError>(Value::Int(IntValue::U64(
                                    indices[ordinal] as u64,
                                )))
                            },
                        )?;
                    runmat_runtime::foreign::assign_foreign_resource_index_retained_in_context(
                        runtime,
                        reference,
                        arguments.into(),
                        rhs,
                    )
                    .await?
                    .with_owned(|value| stack.try_push(value))?;
                }
                Value::Object(obj) => {
                    assign_object_scalar_indices(runtime, Value::Object(obj), &indices, rhs, delete)
                        .await?
                        .with_owned(|value| stack.try_push(value))?
                }
                Value::HandleObject(handle) => assign_object_scalar_indices(
                    runtime,
                    Value::HandleObject(handle),
                    &indices,
                    rhs,
                    delete,
                )
                .await?
                .with_owned(|value| stack.try_push(value))?,
                Value::ObjectArray(array) => assign_object_scalar_indices(
                    runtime,
                    Value::ObjectArray(array),
                    &indices,
                    rhs,
                    delete,
                )
                .await?
                .with_owned(|value| stack.try_push(value))?,
                Value::FunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
                | Value::BoundFunctionHandle(_)
                | Value::Closure(_)
                | Value::ScopedFunctionHandle(_) => {
                    return Err(crate::interpreter::errors::mex(
                        "UnsupportedFunctionHandleSelector",
                        "Function handle call does not support assignment selector syntax",
                    ));
                }
                base @ (Value::Tensor(_)
                | Value::ComplexTensor(_)
                | Value::Num(_)
                | Value::Int(_)
                | Value::GpuTensor(_)) => {
                    if indices.is_empty() || indices.len() > 2 {
                        return Err(crate::interpreter::errors::mex(
                            "UnsupportedAssignmentRank",
                            "Only 1D/2D scalar assignment supported",
                        ));
                    }
                    if delete && indices.len() != 1 {
                        return Err(crate::interpreter::errors::mex(
                            "UnsupportedDeletion",
                            "Indexed deletion is only supported for linear vector indices",
                        ));
                    }
                    let shape = index_value::shape_borrowed(&base).ok_or_else(|| {
                        crate::interpreter::errors::mex(
                            "IndexAssignmentUnsupportedBase",
                            "Numeric indexed assignment requires a numeric base",
                        )
                    })?;
                    let numeric = runmat_runtime::context::host_allocation::initialize_host_array(
                        Some(runtime),
                        indices.len(),
                        |ordinal| {
                            u64::try_from(indices[ordinal])
                                .map(|index| Value::Int(IntValue::U64(index)))
                                .map_err(|_| {
                                    crate::interpreter::errors::mex(
                                        "IndexOutOfBounds",
                                        "Index out of bounds",
                                    )
                                })
                        },
                    )?;
                    let mode = if delete {
                        IndexedStorePlanMode::Bounded
                    } else {
                        IndexedStorePlanMode::Grow
                    };
                    let plan = build_indexed_store_plan_in_context(
                        runtime,
                        indices.len(),
                        0,
                        0,
                        &numeric,
                        shape.as_slice(),
                        mode,
                    )
                    .await?;
                    let updated = if delete {
                        match base {
                            Value::Tensor(t) => {
                                idx_write_slice::delete_tensor_with_plan(t, &plan, &rhs)?
                            }
                            Value::ComplexTensor(t) => {
                                idx_write_slice::delete_complex_with_plan(t, &plan, &rhs)?
                            }
                            value @ (Value::Num(_) | Value::Int(_)) => {
                                let tensor = index_value::numeric_scalar_tensor_in_context(
                                    Some(runtime),
                                    value,
                                )?;
                                idx_write_slice::delete_tensor_with_plan(tensor, &plan, &rhs)?
                            }
                            Value::GpuTensor(handle) => {
                                idx_write_slice::delete_gpu_slice_with_plan_in_context(
                                    Some(runtime),
                                    &handle,
                                    &plan,
                                    &rhs,
                                )
                                .await?
                            }
                            _ => {
                                return Err(crate::interpreter::errors::mex(
                                    "IndexAssignmentUnsupportedBase",
                                    "Numeric indexed assignment requires a numeric base",
                                ));
                            }
                        }
                    } else {
                        index_value::assign_with_plan_in_context(Some(runtime), base, &plan, rhs)
                            .await?
                    };
                    stack.try_push(updated)?;
                }
                Value::Bool(b) => {
                    let scalar = index_value::numeric_scalar_tensor_in_context(
                        Some(runtime),
                        Value::Num(if b { 1. } else { 0. }),
                    )?;
                    stack.try_push(
                        idx_write_linear::assign_tensor_scalar_in_context(
                            Some(runtime),
                            scalar,
                            &indices,
                            &rhs,
                            delete,
                        )
                        .await?,
                    )?;
                }
                Value::StringArray(sa) => {
                    if delete {
                        return Err(crate::interpreter::errors::mex(
                            "UnsupportedSliceDeletion",
                            "Slice deletion currently supports cell arrays only",
                        ));
                    }
                    let selectors =
                        runmat_runtime::context::host_allocation::initialize_host_array(
                            Some(runtime),
                            indices.len(),
                            |ordinal| Ok(SliceSelector::Scalar(indices[ordinal])),
                        )?;
                    let plan = runmat_runtime::indexing::plan::build_index_plan_in_context(
                        runtime,
                        &selectors,
                        indices.len(),
                        &sa.shape,
                    )?;
                    if !plan.indices.is_empty() {
                        stack.try_push(idx_write_slice::assign_string_with_plan_in_context(
                            Some(runtime),
                            sa,
                            &plan,
                            &rhs,
                        )?)?;
                    } else {
                        stack.try_push(Value::StringArray(sa))?;
                    }
                }
                Value::Cell(ca) => stack.try_push(
                    runmat_runtime::object::cell::assign_cell_paren_with_policy_in_context(
                        Some(runtime),
                        ca,
                        &indices,
                        &rhs,
                        delete,
                    )?,
                )?,
                Value::Struct(structure) => {
                    if delete {
                        stack.try_push(
                            runmat_runtime::indexing::structure::delete_scalar_indices_in_context(
                                Some(runtime),
                                Value::Struct(structure),
                                &indices,
                                &rhs,
                            )?,
                        )?;
                        return Ok(true);
                    }
                    let selectors =
                        runmat_runtime::context::host_allocation::initialize_host_array(
                            Some(runtime),
                            indices.len(),
                            |ordinal| Ok(SliceSelector::Scalar(indices[ordinal])),
                        )?;
                    let plan = runmat_runtime::indexing::plan::build_assignment_plan_in_context(
                        runtime,
                        &selectors,
                        indices.len(),
                        &[1, 1],
                    )?;
                    stack.try_push(
                        runmat_runtime::indexing::structure::assign_with_plan_in_context(
                            Some(runtime),
                            Value::Struct(structure),
                            &plan,
                            rhs,
                        )?,
                    )?;
                }
                Value::StructArray(array) => {
                    if delete {
                        stack.try_push(
                            runmat_runtime::indexing::structure::delete_scalar_indices_in_context(
                                Some(runtime),
                                Value::StructArray(array),
                                &indices,
                                &rhs,
                            )?,
                        )?;
                        return Ok(true);
                    }
                    let selectors =
                        runmat_runtime::context::host_allocation::initialize_host_array(
                            Some(runtime),
                            indices.len(),
                            |ordinal| Ok(SliceSelector::Scalar(indices[ordinal])),
                        )?;
                    let plan = runmat_runtime::indexing::plan::build_assignment_plan_in_context(
                        runtime,
                        &selectors,
                        indices.len(),
                        array.shape(),
                    )?;
                    stack.try_push(
                        runmat_runtime::indexing::structure::assign_with_plan_in_context(
                            Some(runtime),
                            Value::StructArray(array),
                            &plan,
                            rhs,
                        )?,
                    )?;
                }
                Value::SparseTensor(sparse) => stack.try_push(
                    idx_write_linear::assign_sparse_scalar_in_context(
                        Some(runtime),
                        sparse,
                        &indices,
                        &rhs,
                        delete,
                    )
                    .await?,
                )?,
                _ => {
                    if std::env::var("RUNMAT_DEBUG_INDEX").as_deref() == Ok("1") {
                        let kind = |v: &Value| match v {
                            Value::Object(_) => "Object",
                            Value::Struct(_) => "Struct",
                            Value::Tensor(_) => "Tensor",
                            Value::GpuTensor(_) => "GpuTensor",
                            Value::Num(_) => "Num",
                            Value::Int(_) => "Int",
                            _ => "Other",
                        };
                        log::debug!(
                            "[vm] StoreIndex default branch pc={} base_kind={} rhs_kind={} indices={:?}",
                            pc,
                            kind(&base),
                            kind(&rhs),
                            indices
                        );
                    }
                    return Err(crate::interpreter::errors::mex(
                        "IndexAssignmentUnsupportedBase",
                        "Index assignment only for tensors",
                    ));
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
